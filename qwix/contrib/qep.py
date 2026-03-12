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
  return (batch,), {}


def _normalize_calibration_data(
    calibration_data: Iterable[Any] | Callable[[], Iterable[Any]]
) -> Callable[[], Iterable[Any]]:
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
  if isinstance(tree, flax.core.FrozenDict):
    return flax.core.unfreeze(tree)
  return tree


def _flatten_tree(tree: Any) -> dict[tuple[str, ...], Any]:
  return flax.traverse_util.flatten_dict(_mutable_tree(tree))


def _unflatten_tree(flat_tree: dict[tuple[str, ...], Any]) -> Any:
  return flax.traverse_util.unflatten_dict(flat_tree)


def _merge_trees(base: Any, updates: Any) -> Any:
  base_flat = _flatten_tree(base)
  base_flat.update(_flatten_tree(updates))
  return _unflatten_tree(base_flat)


def _dequantize_params_tree(tree: Any) -> Any:
  return jax.tree.map(
      lambda leaf: (
          qarray.dequantize(leaf.array) if isinstance(leaf, ptq.WithAux) else leaf
      ),
      _mutable_tree(tree),
      is_leaf=lambda leaf: isinstance(leaf, ptq.WithAux),
  )


def _stats_path(path: tuple[str, ...]) -> tuple[str, ...]:
  return (*path[:-1], path[-1] + '_qep')


def _accumulate_flat_stats(
    flat_stats: dict[tuple[str, ...], Any],
    path: tuple[str, ...],
    stats: dict[str, jax.Array],
) -> None:
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
  args, kwargs = batch_adapter(batch)
  return tuple(args), dict(kwargs)


class _CaptureProvider(calibration.CalibrationProvider):
  """CalibrationProvider variant that records matched op metadata and inputs."""

  def __init__(self, rules: Collection[qconfig.QuantizationRule]):
    super().__init__(rules)
    self._discovered_ops: list[_MatchedOp] = []
    self._capture_keys: set[tuple[Any, ...]] | None = None
    self._captures: dict[tuple[Any, ...], jax.Array] = {}

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return QepRule

  def get_stats_suffix(self) -> str:
    return '_qep'

  def start_discovery(self) -> None:
    self._discovered_ops.clear()
    self._capture_keys = None
    self._captures = {}

  def start_capture(self, op_keys: Collection[tuple[Any, ...]]) -> None:
    self._capture_keys = set(op_keys)
    self._captures = {}

  def finish_capture(self) -> dict[tuple[Any, ...], jax.Array]:
    captures = self._captures
    self._captures = {}
    return captures

  @property
  def discovered_ops(self) -> tuple[_MatchedOp, ...]:
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
  def _quantize_weight(prepared: calibration.PreparedWeight) -> Any:
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
