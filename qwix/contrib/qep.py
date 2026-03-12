# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Integration of exact stagewise QEP into Qwix.

QEP (Quantization Error Propagation) extends GPTQ by accounting for
quantization noise in the input activations seen by later layers.

The exact reference algorithm is stagewise: for each stage, it collects paired
float vs progressively quantized inputs across the full calibration set,
quantizes that stage, updates the quantized model state, and then continues to
the next stage.

This implementation currently supports Flax linen models only.
"""

import dataclasses
from collections.abc import Callable, Collection, Iterable
from typing import Any

import flax
from flax import linen as nn
import jax
import jax.numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import model as qwix_model
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.providers import ptq
from qwix.contrib import calibration
from qwix.contrib import gptq_core
from qwix.contrib import qep_core
from qwix.contrib.gptq import GptqRule


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepRule(GptqRule):
  """Use this rule to enable exact stagewise QEP.

  Attributes:
    correction_factor: Weight correction factor. 0.0 = no correction,
      1.0 = full correction. Default 0.5 per QEP paper recommendations.
    damping_factor: Dampening factor for QEP weight correction Hessian
      inversion. Default 1.0.
    apply_correction: Whether to apply QEP weight correction before GPTQ.
      Set this to ``False`` for stages that should run GPTQ without the QEP
      correction term.
  """

  correction_factor: float = 0.5
  damping_factor: float = 1.0
  apply_correction: bool = True


@dataclasses.dataclass(frozen=True)
class QepStage:
  """Metadata about one exact QEP stage."""

  index: int
  param_paths: tuple[tuple[str, ...], ...]
  module_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepResult:
  """Result of an exact stagewise QEP run."""

  model: nn.Module
  params: Any
  quant_stats: Any
  stages: tuple[QepStage, ...]


@dataclasses.dataclass(frozen=True)
class _MatchedOp:
  """One supported op matched during discovery."""

  op_key: tuple[Any, ...]
  path: tuple[str, ...]
  module_path: tuple[str, ...]
  discovery_index: int
  lhs_id: int
  rule: QepRule


@dataclasses.dataclass(frozen=True)
class _StageSpec:
  """Internal stage specification."""

  index: int
  members: tuple[_MatchedOp, ...]


def _default_batch_adapter(batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Treats one calibration batch as a single positional model argument."""
  return (batch,), {}


def _normalize_calibration_data(
    calibration_data: Iterable[Any] | Callable[[], Iterable[Any]]
) -> Callable[[], Iterable[Any]]:
  """Normalizes calibration data into a factory that yields fresh iterables.

  Exact QEP replays the full calibration set once per inferred stage, so the
  input must be reiterable. This helper accepts either:

  - a zero-arg callable that already returns a fresh iterable, or
  - a reiterable collection such as a list or tuple.

  A one-shot iterator is rejected because later stages would otherwise see an
  empty dataset.
  """
  if callable(calibration_data):
    return calibration_data
  iterator = iter(calibration_data)
  if iterator is calibration_data:
    raise ValueError(
        'calibration_data must be reiterable, or a zero-arg callable that '
        'returns a fresh iterable.'
    )
  return lambda: iter(calibration_data)


def _mutable_tree(tree: Any) -> Any:
  """Returns a mutable mapping view of a params tree when needed."""
  if isinstance(tree, flax.core.FrozenDict):
    return flax.core.unfreeze(tree)
  return tree


def _flatten_tree(tree: Any) -> dict[tuple[str, ...], Any]:
  """Flattens a nested params tree into tuple paths for incremental updates."""
  return flax.traverse_util.flatten_dict(_mutable_tree(tree))


def _unflatten_tree(flat_tree: dict[tuple[str, ...], Any]) -> Any:
  """Reconstructs a nested params tree from tuple-path leaves."""
  return flax.traverse_util.unflatten_dict(flat_tree)


def _merge_trees(base: Any, updates: Any) -> Any:
  """Overlays one params subtree onto another by flattened path."""
  base_flat = _flatten_tree(base)
  base_flat.update(_flatten_tree(updates))
  return _unflatten_tree(base_flat)


def _dequantize_params_tree(tree: Any) -> Any:
  """Converts PTQ ``WithAux`` leaves back to float arrays.

  During exact QEP, already-quantized stages are fed back into later calibration
  passes as dequantized float weights so that subsequent layers observe the
  correct propagated quantization error in their inputs.
  """
  return jax.tree.map(
      lambda leaf: (
          qarray.dequantize(leaf.array) if isinstance(leaf, ptq.WithAux) else leaf
      ),
      _mutable_tree(tree),
      is_leaf=lambda leaf: isinstance(leaf, ptq.WithAux),
  )


def _stats_path(path: tuple[str, ...]) -> tuple[str, ...]:
  """Maps a weight path such as ``Dense_0/kernel`` to its ``_qep`` stats leaf."""
  return (*path[:-1], path[-1] + '_qep')


def _accumulate_flat_stats(
    flat_stats: dict[tuple[str, ...], Any],
    path: tuple[str, ...],
    stats: dict[str, jax.Array],
) -> None:
  """Accumulates per-weight calibration stats with ``SimpleMovingAverage``.

  ``compute_qep_stats`` returns raw batch-level statistics. QEP needs their
  dataset-wide averages before quantizing a stage, so this helper maintains the
  running aggregate in a flat dict keyed by the eventual ``quant_stats`` path.
  """
  aggregator = averaging.SimpleMovingAverage()
  stat_path = _stats_path(path)
  quant_stat = flat_stats.get(stat_path)
  if quant_stat is None:
    quant_stat = aggregator.init(stats)
  quant_stat = aggregator.update(quant_stat, stats)
  flat_stats[stat_path] = quant_stat


def _extract_batch(
    batch_adapter: Callable[[Any], tuple[tuple[Any, ...], dict[str, Any]]],
    batch: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
  """Applies the user batch adapter and normalizes the result types."""
  args, kwargs = batch_adapter(batch)
  return tuple(args), dict(kwargs)


class _CaptureProvider(calibration.CalibrationProvider):
  """Calibration provider that records matched ops and captured activations.

  Unlike GPTQ/AWQ calibration providers, this provider does not write into the
  model's ``quant_stats`` collection. Instead, it exposes a Python-side API for
  two separate tasks used by exact QEP:

  - discovery: enumerate supported matched ops in forward-pass order
  - capture: record the reshaped LHS activation for a selected set of ops

  The provider still reuses ``CalibrationProvider`` for all rule matching,
  supported-op validation, weight lookup, and LHS normalization.
  """

  def __init__(self, rules: Collection[qconfig.QuantizationRule]):
    super().__init__(rules)
    self._discovered_ops: list[_MatchedOp] = []
    self._capture_keys: set[tuple[Any, ...]] | None = None
    self._captures: dict[tuple[Any, ...], jax.Array] = {}

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    """Restricts capture to ops matched by ``QepRule``."""
    return QepRule

  def get_stats_suffix(self) -> str:
    """Returns the suffix associated with QEP calibration stats."""
    return '_qep'

  def start_discovery(self) -> None:
    """Resets provider state before the initial discovery forward pass."""
    self._discovered_ops.clear()
    self._capture_keys = None
    self._captures = {}

  def start_capture(self, op_keys: Collection[tuple[Any, ...]]) -> None:
    """Enables activation capture for the selected ops on the next forward."""
    self._capture_keys = set(op_keys)
    self._captures = {}

  def finish_capture(self) -> dict[tuple[Any, ...], jax.Array]:
    """Returns captured activations from the last forward and clears them."""
    captures = self._captures
    self._captures = {}
    return captures

  @property
  def discovered_ops(self) -> tuple[_MatchedOp, ...]:
    """Matched supported ops seen during discovery, in forward order."""
    return tuple(self._discovered_ops)

  def _collect_stats(
      self,
      lhs: jax.Array,
      weight_name: str,
      *,
      module_path: tuple[str, ...],
      op_name: str,
      op_id: str | None,
      lhs_id: int,
  ) -> None:
    """Records one supported matched op or one requested activation capture.

    ``CalibrationProvider`` calls this only for supported weight-bearing
    dot/einsum ops. During discovery, this stores enough metadata to later:

    - group sibling ops that share the same input activation,
    - identify the weight path to quantize, and
    - re-identify the same op on later forwards.

    During capture, the provider instead records the normalized LHS activation
    for op keys selected by ``start_capture``.
    """
    path = (*module_path, weight_name)
    op_key = (op_name, module_path, op_id)

    if self._capture_keys is None:
      rule, _ = self._get_current_rule_and_op_id(op_name)
      assert isinstance(rule, QepRule)
      self._discovered_ops.append(
          _MatchedOp(
              op_key=op_key,
              path=path,
              module_path=module_path,
              discovery_index=len(self._discovered_ops),
              lhs_id=lhs_id,
              rule=rule,
          )
      )
    elif op_key in self._capture_keys:
      self._captures[op_key] = lhs


def _build_stages(discovered_ops: tuple[_MatchedOp, ...]) -> tuple[_StageSpec, ...]:
  """Groups discovered ops into inferred QEP stages.

  In the current linen-only design, stages are inferred purely from shared
  runtime inputs: consecutive matched ops with the same original ``lhs`` object
  id are treated as one stage. This preserves the intended semantics for common
  branched patterns such as sibling linear projections fed by the same tensor.
  """
  if not discovered_ops:
    raise ValueError(
        'No supported QEP ops were discovered. Ensure the rules match '
        'supported dot_general or einsum weight ops.'
    )

  stages = []
  current_members: list[_MatchedOp] = []
  current_lhs_id = None
  for op in discovered_ops:
    if current_members and op.lhs_id != current_lhs_id:
      stages.append(
          _StageSpec(index=len(stages), members=tuple(current_members))
      )
      current_members = []
    current_members.append(op)
    current_lhs_id = op.lhs_id

  if current_members:
    stages.append(_StageSpec(index=len(stages), members=tuple(current_members)))

  seen_paths: set[tuple[str, ...]] = set()
  for stage in stages:
    stage_paths = {op.path for op in stage.members}
    if seen_paths & stage_paths:
      raise ValueError(
          'QEP does not support quantizing the same param path across multiple '
          'stages.'
      )
    seen_paths.update(stage_paths)
  return tuple(stages)


def _stage_to_public(stage: _StageSpec) -> QepStage:
  """Converts the internal stage representation into public debug metadata."""
  unique_paths = tuple(dict.fromkeys(op.path for op in stage.members))
  unique_module_paths = tuple(
      dict.fromkeys('/'.join(op.module_path) for op in stage.members)
  )
  return QepStage(
      index=stage.index,
      param_paths=unique_paths,
      module_paths=unique_module_paths,
  )


def _apply_with_params(
    model: nn.Module,
    variables: Any,
    params: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Any:
  """Runs a linen model with a replacement params tree."""
  apply_variables = {**variables, 'params': params}
  return model.apply(apply_variables, *args, **kwargs)


def _prepare_ptq_model(
    model: nn.Module,
    rules: Collection[qconfig.QuantizationRule],
    methods: Collection[str],
    sample_args: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
    abstract_quantized: Any,
) -> tuple[nn.Module, Any]:
  """Builds the PTQ inference model and its abstract quantized params tree."""
  ptq_model = qwix_model.quantize_model(
      model, ptq.PtqProvider(rules), methods=methods
  )
  if abstract_quantized is None:
    abstract_quantized = jax.eval_shape(
        ptq_model.init,
        jax.random.key(0),
        *sample_args,
        **sample_kwargs,
    )['params']
  return ptq_model, abstract_quantized


def _quantize_stage(
    *,
    float_params: Any,
    abstract_quantized: Any,
    quant_stats: Any,
    stage_rules: dict[tuple[str, ...], QepRule],
    gptq_block_size: int,
    gptq_damping_factor: float,
) -> Any:
  """Quantizes one inferred QEP stage from its finalized calibration stats.

  The stage is always quantized from the original float weights. Previously
  quantized stages only affect the collected activations, not the source weights
  being updated here.
  """
  def _quantize_weight(prepared: calibration.PreparedWeight) -> Any:
    """Applies optional QEP correction followed by GPTQ for one weight leaf."""
    rule = stage_rules[prepared.path]
    hessian = prepared.calibration_stats['hessian']
    assert (
        hessian.shape[0] == prepared.weight.shape[1]
        and hessian.shape[1] == prepared.weight.shape[1]
    )

    weight = prepared.weight
    if rule.apply_correction:
      hessian_delta = prepared.calibration_stats.get('hessian_delta')
      if hessian_delta is None:
        raise ValueError(
            f'hessian_delta not found in QEP stats for {prepared.path}.'
        )
      weight = qep_core.weight_correct(
          weight,
          hessian,
          hessian_delta,
          perccorr=rule.correction_factor,
          percdamp=rule.damping_factor,
      )

    weight = gptq_core.quantize_weight(
        weight,
        hessian,
        prepared.how,
        blocksize=gptq_block_size,
        percdamp=gptq_damping_factor,
    )[0]
    weight = prepared.restore_shape(weight)
    return prepared.abs_w.replace(array=weight)

  return calibration.quantize_params_with_calibration(
      float_params,
      abstract_quantized,
      quant_stats,
      '_qep',
      _quantize_weight,
      selected_paths=tuple(stage_rules),
      ptq_fallback=False,
  )


def quantize(
    model: nn.Module,
    calibration_data: Iterable[Any] | Callable[[], Iterable[Any]],
    rules: Collection[QepRule],
    *,
    variables: Any = None,
    batch_adapter: Callable[
        [Any], tuple[tuple[Any, ...], dict[str, Any]]
    ] | None = None,
    methods: Collection[str] = ('__call__',),
    abstract_quantized: Any = None,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
) -> QepResult:
  """Runs exact stagewise QEP for a linen model.

  Args:
    model: The linen model to quantize.
    calibration_data: Reiterable calibration batches, or a zero-arg callable
      that returns a fresh iterable each time.
    rules: The QEP rules to apply.
    variables: Linen variables dict. Required.
    batch_adapter: Converts one calibration batch into ``(args, kwargs)``.
      Defaults to treating each batch as a single positional argument.
    methods: Methods to intercept via ``qwix.quantize_model``.
    abstract_quantized: Optional PTQ abstract params tree.
    allow_extra_params: See ``ptq.quantize_params``.
    gptq_block_size: GPTQ block size used for each QEP stage.
    gptq_damping_factor: GPTQ damping factor used for each QEP stage.

  Returns:
    A ``QepResult`` containing the PTQ inference model clone, final quantized
    params, accumulated QEP stats, and inferred stages.

  The algorithm proceeds as follows:

  1. Discover supported matched ops in forward order on one float pass.
  2. Infer stages by grouping consecutive matched ops that share the same input
     activation object.
  3. For each stage, replay the calibration set twice per batch:
     one float forward and one forward using dequantized weights from already
     quantized earlier stages.
  4. Accumulate ``_qep`` stats for the current stage only.
  5. Quantize that stage from the original float weights, then dequantize the
     newly quantized leaves into the running params tree for later stages.
  6. PTQ-quantize any remaining rule-matched weights that never participated in
     a supported QEP stage.
  """
  if not isinstance(model, nn.Module):
    raise ValueError('qep.quantize currently supports linen models only.')
  if variables is None:
    raise ValueError('variables is required for linen QEP quantization.')

  batch_adapter = batch_adapter or _default_batch_adapter
  batch_iter_fn = _normalize_calibration_data(calibration_data)
  first_iterator = iter(batch_iter_fn())
  try:
    first_batch = next(first_iterator)
  except StopIteration as exc:
    raise ValueError('calibration_data must contain at least one batch.') from exc

  sample_args, sample_kwargs = _extract_batch(batch_adapter, first_batch)
  float_params = _mutable_tree(variables['params'])

  float_provider = _CaptureProvider(rules)
  quant_provider = _CaptureProvider(rules)
  float_model = qwix_model.quantize_model(model, float_provider, methods=methods)
  quant_model = qwix_model.quantize_model(model, quant_provider, methods=methods)

  float_provider.start_discovery()
  _apply_with_params(float_model, variables, float_params, sample_args, sample_kwargs)
  stages = _build_stages(float_provider.discovered_ops)

  ptq_model, abstract_quantized = _prepare_ptq_model(
      model,
      rules,
      methods,
      sample_args,
      sample_kwargs,
      abstract_quantized,
  )

  current_dequantized_params = float_params
  final_quantized_params: Any = {}
  flat_quant_stats: dict[tuple[str, ...], Any] = {}
  staged_paths: set[tuple[str, ...]] = set()

  for stage in stages:
    stage_rule_by_path = {op.path: op.rule for op in stage.members}
    stage_op_keys = tuple(op.op_key for op in stage.members)
    stage_flat_stats: dict[tuple[str, ...], Any] = {}

    for batch in batch_iter_fn():
      args, kwargs = _extract_batch(batch_adapter, batch)

      float_provider.start_capture(stage_op_keys)
      _apply_with_params(float_model, variables, float_params, args, kwargs)
      float_captures = float_provider.finish_capture()

      quant_provider.start_capture(stage_op_keys)
      _apply_with_params(
          quant_model, variables, current_dequantized_params, args, kwargs
      )
      quant_captures = quant_provider.finish_capture()

      for op in stage.members:
        float_lhs = float_captures.get(op.op_key)
        quant_lhs = quant_captures.get(op.op_key)
        if float_lhs is None or quant_lhs is None:
          raise ValueError(
              f'Missing captured QEP activations for {"/".join(op.module_path)}.'
          )
        _accumulate_flat_stats(
            stage_flat_stats,
            op.path,
            qep_core.compute_qep_stats(quant_lhs, float_lhs),
        )

    stage_quant_stats = _unflatten_tree(stage_flat_stats)
    stage_quantized_params = _quantize_stage(
        float_params=float_params,
        abstract_quantized=abstract_quantized,
        quant_stats=stage_quant_stats,
        stage_rules=stage_rule_by_path,
        gptq_block_size=gptq_block_size,
        gptq_damping_factor=gptq_damping_factor,
    )
    final_quantized_params = _merge_trees(
        final_quantized_params, stage_quantized_params
    )
    current_dequantized_params = _merge_trees(
        current_dequantized_params, _dequantize_params_tree(stage_quantized_params)
    )
    flat_quant_stats.update(stage_flat_stats)
    staged_paths.update(stage_rule_by_path)

  flat_float_params = _flatten_tree(float_params)
  remaining_flat_params = {
      path: value
      for path, value in flat_float_params.items()
      if path not in staged_paths
  }
  if remaining_flat_params:
    remaining_quantized = ptq.quantize_params(
        _unflatten_tree(remaining_flat_params),
        abstract_quantized,
        allow_extra_params=allow_extra_params,
    )
    final_quantized_params = _merge_trees(
        final_quantized_params, remaining_quantized
    )

  return QepResult(
      model=ptq_model,
      params=final_quantized_params,
      quant_stats=_unflatten_tree(flat_quant_stats),
      stages=tuple(_stage_to_public(stage) for stage in stages),
  )


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    qep_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    gptq_block_size: int = 128,
    gptq_damping_factor: float = 0.01,
    correction_factor: float = 0.5,
    damping_factor: float = 1.0,
    apply_correction: bool = True,
) -> Any:
  """Quantizes params from precomputed QEP stats.

  This helper consumes existing ``_qep`` stats and applies QEP correction plus
  GPTQ, but it does not run the exact stagewise reference algorithm by itself.

  Args:
    params: Floating-point params to quantize.
    abstract_quantized_params: PTQ abstract params tree containing ``WithAux``
      leaves that describe how each supported weight should be quantized.
    qep_quant_stats: Pure ``_qep`` stats tree, typically produced by
      :func:`quantize`.
    allow_extra_params: Passed through to ``ptq.quantize_params`` for PTQ
      fallback leaves.
    gptq_block_size: GPTQ block size for each quantized weight.
    gptq_damping_factor: GPTQ Hessian damping factor.
    correction_factor: QEP correction factor used when
      ``apply_correction=True``.
    damping_factor: QEP Hessian damping factor used when
      ``apply_correction=True``.
    apply_correction: If ``True``, apply the QEP weight correction before GPTQ.

  Returns:
    A params tree consumable by ``PtqProvider``.
  """

  def _quantize(prepared: calibration.PreparedWeight) -> Any:
    hessian = prepared.calibration_stats['hessian']
    assert (
        hessian.shape[0] == prepared.weight.shape[1]
        and hessian.shape[1] == prepared.weight.shape[1]
    )

    weight = prepared.weight
    if apply_correction:
      hessian_delta = prepared.calibration_stats.get('hessian_delta')
      if hessian_delta is None:
        raise ValueError(
            f'hessian_delta not found in QEP stats for {prepared.path}.'
        )
      weight = qep_core.weight_correct(
          weight,
          hessian,
          hessian_delta,
          perccorr=correction_factor,
          percdamp=damping_factor,
      )

    weight = gptq_core.quantize_weight(
        weight,
        hessian,
        prepared.how,
        blocksize=gptq_block_size,
        percdamp=gptq_damping_factor,
    )[0]
    weight = prepared.restore_shape(weight)
    return prepared.abs_w.replace(array=weight)

  return calibration.quantize_params_with_calibration(
      params,
      abstract_quantized_params,
      qep_quant_stats,
      '_qep',
      _quantize,
      allow_extra_params=allow_extra_params,
  )
