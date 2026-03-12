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
the next stage. This module implements that exact workflow for both linen and
NNX through :func:`quantize`.

The lower-level :func:`quantize_params` helper remains available for consuming
precomputed ``_qep`` statistics, but it does not orchestrate the exact
stagewise algorithm by itself.
"""

import dataclasses
from collections.abc import Callable, Collection, Iterable
from typing import Any

import flax
from flax import linen as nn
from flax import nnx
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
    stage_order: Optional explicit ordering for stagewise QEP. If any matched
      rule sets this field, all matched rules must set it.
    stage_group: Optional explicit grouping within a ``stage_order``.
      Matched ops with the same ``stage_order`` and ``stage_group`` are
      calibrated and quantized together.
    apply_correction: Whether to apply QEP weight correction before GPTQ.
      Set this to ``False`` for down-proj-style stages that should run GPTQ
      without correction.
  """

  correction_factor: float = 0.5
  damping_factor: float = 1.0
  stage_order: int | None = None
  stage_group: str | None = None
  apply_correction: bool = True


@dataclasses.dataclass(frozen=True)
class QepStage:
  """Metadata about one exact QEP stage."""

  index: int
  order: int
  group: str | None
  param_paths: tuple[tuple[str, ...], ...]
  module_paths: tuple[str, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepResult:
  """Result of an exact stagewise QEP run."""

  model: Any
  params: Any
  quant_stats: Any
  stages: tuple[QepStage, ...]


@dataclasses.dataclass(frozen=True)
class _MatchedOp:
  """One supported op matched during discovery."""

  op_key: tuple[Any, ...]
  path: tuple[str, ...]
  module_path: tuple[str, ...]
  module_path_str: str
  discovery_index: int
  lhs_id: int
  rule: QepRule


@dataclasses.dataclass(frozen=True)
class _StageSpec:
  """Internal stage specification."""

  index: int
  order: int
  group: str | None
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


def _tree_to_dict(tree: Any) -> Any:
  if isinstance(tree, flax.core.FrozenDict):
    return flax.core.unfreeze(tree)
  return tree


def _flatten_tree(
    tree: Any,
    prefix: tuple[str, ...] = (),
) -> dict[tuple[str, ...], Any]:
  tree = _tree_to_dict(tree)
  flat_tree = {}
  if isinstance(tree, dict):
    for key, value in tree.items():
      flat_tree.update(_flatten_tree(value, (*prefix, key)))
  elif prefix:
    flat_tree[prefix] = tree
  else:
    raise ValueError('Cannot flatten a non-dict tree without a prefix.')
  return flat_tree


def _unflatten_tree(flat_tree: dict[tuple[str, ...], Any]) -> Any:
  root = {}
  for path, value in flat_tree.items():
    cursor = root
    for key in path[:-1]:
      cursor = cursor.setdefault(key, {})
    cursor[path[-1]] = value
  return root


def _merge_trees(base: Any, updates: Any) -> Any:
  base = _tree_to_dict(base)
  updates = _tree_to_dict(updates)
  if isinstance(base, dict) and isinstance(updates, dict):
    merged = dict(base)
    for key, value in updates.items():
      if key in merged:
        merged[key] = _merge_trees(merged[key], value)
      else:
        merged[key] = value
    return merged
  return updates


def _dict_to_qarray(tree: dict[str, Any]) -> qarray.QArray | None:
  if not isinstance(tree, dict) or 'qvalue' not in tree or 'scale' not in tree:
    return None
  allowed_keys = {'qvalue', 'scale', 'zero_point', 'qtype'}
  if not set(tree).issubset(allowed_keys):
    return None
  return qarray.QArray(
      qvalue=tree['qvalue'],
      scale=tree['scale'],
      zero_point=tree.get('zero_point'),
      qtype=tree.get('qtype'),
  )


def _dequantize_params_tree(tree: Any) -> Any:
  tree = _tree_to_dict(tree)
  if isinstance(tree, ptq.WithAux):
    return qarray.dequantize(tree.array)
  if isinstance(tree, qarray.QArray):
    return qarray.dequantize(tree)
  if isinstance(tree, dict):
    if 'array' in tree:
      array = _tree_to_dict(tree['array'])
      qarr = _dict_to_qarray(array)
      if qarr is not None:
        return qarray.dequantize(qarr)
    return {key: _dequantize_params_tree(value) for key, value in tree.items()}
  return tree


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


class _CaptureProvider(qconfig.QuantizationProvider):
  """Eager provider that records matched op metadata and activations."""

  def __init__(self, rules: Collection[qconfig.QuantizationRule]):
    super().__init__(rules)
    self._discovered_ops: list[_MatchedOp] = []
    self._capture_keys: set[tuple[Any, ...]] | None = None
    self._captures: dict[tuple[Any, ...], jax.Array] = {}

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

  def _record_supported_op(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      rule: qconfig.QuantizationRule | None,
      op_name: str,
      op_id: str | None,
  ) -> None:
    if not isinstance(rule, QepRule):
      return

    lhs_id = id(lhs)
    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      return

    weight_name = flax_util.find_param(rhs)
    if weight_name is None:
      return

    lhs = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs = lhs.reshape(lhs.shape[0], -1)

    module_path = tuple(map(str, flax_util.get_current_module_path()))
    path = (*module_path, weight_name)
    op_key = (op_name, module_path, op_id)

    if self._capture_keys is None:
      self._discovered_ops.append(
          _MatchedOp(
              op_key=op_key,
              path=path,
              module_path=module_path,
              module_path_str='/'.join(module_path),
              discovery_index=len(self._discovered_ops),
              lhs_id=lhs_id,
              rule=rule,
          )
      )
    elif op_key in self._capture_keys:
      self._captures[op_key] = lhs

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      *args,
      rule: qconfig.QuantizationRule | None = None,
      op_id: str | None = None,
      **kwargs,
  ) -> jax.Array:
    res = jax.lax.dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)
    if rule is None or op_id is None:
      rule, op_id = self._get_current_rule_and_op_id('dot_general')
    self._record_supported_op(lhs, rhs, dimension_numbers, rule, 'dot_general', op_id)
    return res

  def einsum(self, einsum_str, *operands, **kwargs):
    rule, op_id = self._get_current_rule_and_op_id('einsum')
    if not isinstance(rule, QepRule):
      return jnp.einsum(einsum_str, *operands, **kwargs)
    if not isinstance(einsum_str, str) or len(operands) != 2:
      return jnp.einsum(einsum_str, *operands, **kwargs)

    def capture_dot_general(lhs, rhs, dimension_numbers, *args, **kwargs):
      return self.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          *args,
          rule=rule,
          op_id=op_id,
          **kwargs,
      )

    with jax.disable_jit():
      return jnp.einsum(
          einsum_str,
          *operands,
          _dot_general=capture_dot_general,
          **kwargs,
      )

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'jax.numpy.einsum': self.einsum,
    }


class _LinenAdapter:
  """State adapter for linen models."""

  def __init__(
      self,
      model: nn.Module,
      variables: Any,
      sample_args: tuple[Any, ...],
      sample_kwargs: dict[str, Any],
      methods: Collection[str],
  ):
    self._model = model
    self._variables = variables
    self._sample_args = sample_args
    self._sample_kwargs = sample_kwargs
    self._methods = methods
    self._float_params = _tree_to_dict(variables['params'])

  @property
  def float_params(self) -> Any:
    return self._float_params

  def make_capture_model(self, provider: qconfig.QuantizationProvider) -> nn.Module:
    return qwix_model.quantize_model(self._model, provider, methods=self._methods)

  def run_capture(
      self,
      model: nn.Module,
      params: Any,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ) -> Any:
    variables = {**self._variables, 'params': params}
    return model.apply(variables, *args, **kwargs)

  def prepare_ptq_model(
      self,
      rules: Collection[qconfig.QuantizationRule],
      abstract_quantized: Any,
  ) -> tuple[nn.Module, Any]:
    ptq_model = qwix_model.quantize_model(
        self._model, ptq.PtqProvider(rules), methods=self._methods
    )
    if abstract_quantized is None:
      abstract_quantized = jax.eval_shape(
          ptq_model.init,
          jax.random.key(0),
          *self._sample_args,
          **self._sample_kwargs,
      )['params']
    return ptq_model, abstract_quantized

  def finalize_model(self, ptq_model: nn.Module, params: Any) -> nn.Module:
    del params
    return ptq_model


class _NnxAdapter:
  """State adapter for NNX models."""

  def __init__(
      self,
      model: nnx.Module,
      sample_args: tuple[Any, ...],
      sample_kwargs: dict[str, Any],
      methods: Collection[str],
  ):
    self._model = model
    self._sample_args = sample_args
    self._sample_kwargs = sample_kwargs
    self._methods = methods
    self._float_params = nnx.to_pure_dict(nnx.state(model, nnx.Param))

  @property
  def float_params(self) -> Any:
    return self._float_params

  def make_capture_model(self, provider: qconfig.QuantizationProvider) -> nnx.Module:
    return qwix_model.quantize_model(
        self._model,
        provider,
        *self._sample_args,
        methods=self._methods,
        skip_nnx_init=True,
        **self._sample_kwargs,
    )

  def run_capture(
      self,
      model: nnx.Module,
      params: Any,
      args: tuple[Any, ...],
      kwargs: dict[str, Any],
  ) -> Any:
    nnx.update(model, params)
    return model(*args, **kwargs)

  def prepare_ptq_model(
      self,
      rules: Collection[qconfig.QuantizationRule],
      abstract_quantized: Any,
  ) -> tuple[nnx.Module, Any]:
    ptq_provider = ptq.PtqProvider(rules)
    ptq_model = qwix_model.quantize_model(
        self._model,
        ptq_provider,
        *self._sample_args,
        methods=self._methods,
        **self._sample_kwargs,
    )
    if abstract_quantized is None:
      abstract_quantized = nnx.eval_shape(
          lambda: qwix_model.quantize_model(
              self._model,
              ptq_provider,
              *self._sample_args,
              methods=self._methods,
              **self._sample_kwargs,
          )
      )
    return ptq_model, abstract_quantized

  def finalize_model(self, ptq_model: nnx.Module, params: Any) -> nnx.Module:
    nnx.update(ptq_model, params)
    return ptq_model


def _build_adapter(
    model: Any,
    *,
    variables: Any,
    sample_args: tuple[Any, ...],
    sample_kwargs: dict[str, Any],
    methods: Collection[str],
) -> _LinenAdapter | _NnxAdapter:
  if isinstance(model, nn.Module):
    if variables is None:
      raise ValueError('variables is required for linen QEP quantization.')
    return _LinenAdapter(model, variables, sample_args, sample_kwargs, methods)
  if isinstance(model, nnx.Module):
    if variables is not None:
      raise ValueError('variables must not be set for NNX QEP quantization.')
    return _NnxAdapter(model, sample_args, sample_kwargs, methods)
  raise ValueError(f'Unsupported model type: {type(model)}')


def _build_stages(discovered_ops: tuple[_MatchedOp, ...]) -> tuple[_StageSpec, ...]:
  if not discovered_ops:
    raise ValueError(
        'No supported QEP ops were discovered. Ensure the rules match '
        'supported dot_general or einsum weight ops.'
    )

  explicit = any(op.rule.stage_order is not None for op in discovered_ops)
  if explicit and any(op.rule.stage_order is None for op in discovered_ops):
    raise ValueError(
        'If any matched QepRule sets stage_order, all matched QepRule values '
        'must set stage_order.'
    )

  if explicit:
    grouped: dict[tuple[int, str], list[_MatchedOp]] = {}
    discovery_groups: list[tuple[int, str]] = []
    for op in discovered_ops:
      group = op.rule.stage_group or f'lhs:{op.lhs_id}'
      key = (op.rule.stage_order, group)
      grouped.setdefault(key, []).append(op)
      discovery_groups.append(key)

    seen_group_order: dict[tuple[int, str], int] = {}
    last_key = None
    for index, key in enumerate(discovery_groups):
      if key != last_key:
        if key in seen_group_order:
          raise ValueError(
              'Explicit QEP stage groups must be contiguous in discovery order.'
          )
        seen_group_order[key] = index
        last_key = key

    stage_items = sorted(
        grouped.items(),
        key=lambda item: (item[0][0], item[1][0].discovery_index),
    )
    stages = [
        _StageSpec(
            index=index,
            order=stage_order,
            group=group_name if not group_name.startswith('lhs:') else None,
            members=tuple(members),
        )
        for index, ((stage_order, group_name), members) in enumerate(stage_items)
    ]
  else:
    stages = []
    current_members: list[_MatchedOp] = []
    current_lhs_id = None
    for op in discovered_ops:
      if current_members and op.lhs_id != current_lhs_id:
        stages.append(
            _StageSpec(
                index=len(stages),
                order=len(stages),
                group=None,
                members=tuple(current_members),
            )
        )
        current_members = []
      current_members.append(op)
      current_lhs_id = op.lhs_id
    if current_members:
      stages.append(
          _StageSpec(
              index=len(stages),
              order=len(stages),
              group=None,
              members=tuple(current_members),
          )
      )

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
      dict.fromkeys(op.module_path_str for op in stage.members)
  )
  return QepStage(
      index=stage.index,
      order=stage.order,
      group=stage.group,
      param_paths=unique_paths,
      module_paths=unique_module_paths,
  )


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
    model: Any,
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
  """Runs exact stagewise QEP and returns final quantized params and model.

  Args:
    model: The linen or NNX model to quantize.
    calibration_data: Reiterable calibration batches, or a zero-arg callable
      that returns a fresh iterable each time.
    rules: The QEP rules to apply.
    variables: Linen variables dict. Required for linen, forbidden for NNX.
    batch_adapter: Converts one calibration batch into ``(args, kwargs)``.
      Defaults to treating each batch as a single positional argument.
    methods: Methods to intercept via ``qwix.quantize_model``.
    abstract_quantized: Optional PTQ abstract template. For linen, this should
      be the abstract params tree. For NNX, this should be the abstract PTQ
      model.
    allow_extra_params: See ``ptq.quantize_params``.
    gptq_block_size: GPTQ block size used for each QEP stage.
    gptq_damping_factor: GPTQ damping factor used for each QEP stage.

  Returns:
    A ``QepResult`` containing the PTQ inference model clone, final quantized
    params, accumulated QEP stats, and discovered stages.
  """
  batch_adapter = batch_adapter or _default_batch_adapter
  batch_iter_fn = _normalize_calibration_data(calibration_data)
  first_iterator = iter(batch_iter_fn())
  try:
    first_batch = next(first_iterator)
  except StopIteration as exc:
    raise ValueError('calibration_data must contain at least one batch.') from exc

  sample_args, sample_kwargs = _extract_batch(batch_adapter, first_batch)
  adapter = _build_adapter(
      model,
      variables=variables,
      sample_args=sample_args,
      sample_kwargs=sample_kwargs,
      methods=methods,
  )

  float_provider = _CaptureProvider(rules)
  quant_provider = _CaptureProvider(rules)
  float_model = adapter.make_capture_model(float_provider)
  quant_model = adapter.make_capture_model(quant_provider)

  float_provider.start_discovery()
  adapter.run_capture(float_model, adapter.float_params, sample_args, sample_kwargs)
  stages = _build_stages(float_provider.discovered_ops)

  ptq_model, abstract_quantized = adapter.prepare_ptq_model(
      rules, abstract_quantized
  )

  current_dequantized_params = adapter.float_params
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
      adapter.run_capture(float_model, adapter.float_params, args, kwargs)
      float_captures = float_provider.finish_capture()

      quant_provider.start_capture(stage_op_keys)
      adapter.run_capture(quant_model, current_dequantized_params, args, kwargs)
      quant_captures = quant_provider.finish_capture()

      for op in stage.members:
        float_lhs = float_captures.get(op.op_key)
        quant_lhs = quant_captures.get(op.op_key)
        if float_lhs is None or quant_lhs is None:
          raise ValueError(
              f'Missing captured QEP activations for {op.module_path_str}.'
          )
        _accumulate_flat_stats(
            stage_flat_stats,
            op.path,
            qep_core.compute_qep_stats(quant_lhs, float_lhs),
        )

    stage_quant_stats = _unflatten_tree(stage_flat_stats)
    stage_quantized_params = _quantize_stage(
        float_params=adapter.float_params,
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

  flat_float_params = _flatten_tree(adapter.float_params)
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

  result_model = adapter.finalize_model(ptq_model, final_quantized_params)
  return QepResult(
      model=result_model,
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
    params: See ``ptq.quantize_params``.
    abstract_quantized_params: See ``ptq.quantize_params``.
    qep_quant_stats: Pure ``_qep`` quant_stats dict.
    allow_extra_params: See ``ptq.quantize_params``.
    gptq_block_size: The block size of GPTQ.
    gptq_damping_factor: The damping factor of GPTQ.
    correction_factor: QEP correction factor used when
      ``apply_correction=True``.
    damping_factor: QEP damping factor used when ``apply_correction=True``.
    apply_correction: Whether to apply the QEP weight correction before GPTQ.

  Returns:
    The quantized params consumable by ``PtqProvider``.
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
