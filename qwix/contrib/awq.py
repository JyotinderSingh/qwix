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

"""Integration of AWQ into Qwix.

AWQ (Activation-aware Weight Quantization) identifies salient weight channels
based on activation magnitudes and applies per-channel scaling to improve
quantization accuracy.

Usage:
1. Calibration: Use AwqCalibrationProvider to collect activation statistics.
2. Quantization: Use quantize_params to quantize weights with AWQ.
3. Inference: Use AwqInferenceProvider (not PtqProvider) to run the model.

The AWQ scales are stored separately from the quantized weights and applied
per-channel during inference for maximum accuracy.

Please check the test for an example usage.
"""

import dataclasses
import functools
from typing import Any, Callable, Sequence

import flax
import jax
from jax import numpy as jnp
from qwix._src import averaging
from qwix._src import flax_util
from qwix._src import qconfig
from qwix._src.core import qarray
from qwix._src.core import dot_general as dot_general_module
from qwix._src.providers import ptq
from qwix.contrib import awq_core


@dataclasses.dataclass(frozen=True, kw_only=True)
class AwqRule(qconfig.QuantizationRule):
  """Use this rule to enable AWQ.

  Attributes:
    n_grid: Number of grid points for AWQ scale search. Default is 20.
  """

  n_grid: int = 20


@flax.struct.dataclass
class WithAwqScale:
  """A quantized array with AWQ per-channel scales.

  This wrapper stores the quantized weights along with the per-channel AWQ
  scales. During inference, the AwqInferenceProvider dequantizes the weights
  and divides by the AWQ scales to get the final weights.

  Attributes:
    array: The quantized QArray.
    awq_scale: Per-channel AWQ scales with shape (in_features,). This is a 1D
      array that will be broadcast along the contracting axis during inference.
    contracting_axis: The axis of the weight that is contracted in dot_general.
    how: How the array was quantized.
  """

  array: qarray.QArray
  awq_scale: jax.Array  # Shape: (in_features,)
  contracting_axis: int = flax.struct.field(pytree_node=False)
  how: qarray.HowToQuantize = flax.struct.field(pytree_node=False)

  # This allows us to appear like nnx.Variable.
  value = property(flax_util.unbox)
  shape = property(lambda self: flax_util.unbox(self.array).shape)
  ndim = property(lambda self: flax_util.unbox(self.array).ndim)
  __getitem__ = lambda self, key: jax.tree.map(lambda x: x[key], self.value)

  def reshape(self, *shape):
    if len(shape) == 1:
      try:
        shape = tuple(shape[0])
      except TypeError:
        pass
    if tuple(self.shape) != tuple(shape):
      raise ValueError(
          'AWQ weights should already have the target shape. Got'
          f' {self.shape=} but {shape=} is requested.'
      )
    return self


class AwqCalibrationProvider(qconfig.QuantizationProvider):
  """Calibration for AWQ.

  This provider is used to collect awq_quant_stats (per-channel activation
  scales), which will be used by the quantize_params function below. It does
  not perform the actual quantization of the model parameters, nor use any
  quantized ops.
  """

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array,
      dimension_numbers: jax.lax.DotDimensionNumbers,
      *args,
      **kwargs,
  ) -> jax.Array:
    res = jax.lax.dot_general(lhs, rhs, dimension_numbers, *args, **kwargs)
    rule, _ = self._get_current_rule_and_op_id('dot_general')
    if not isinstance(rule, AwqRule):
      return res

    (lhs_ca, rhs_ca), (lhs_ba, rhs_ba) = dimension_numbers
    if lhs_ba or rhs_ba or len(lhs_ca) != 1 or len(rhs_ca) != 1:
      raise NotImplementedError(f'Unsupported: {dimension_numbers}')

    weight_name = flax_util.find_param(rhs)
    assert weight_name is not None

    # Reorder lhs to (ca, rest) and compute per-channel activation scale.
    lhs = jnp.moveaxis(lhs, lhs_ca[0], 0)
    lhs = lhs.reshape(lhs.shape[0], -1)
    # Compute mean absolute activation for each channel (along sample axis).
    act_scale = awq_core.compute_act_scale(lhs.T, axis=0)

    # Collect the activation scale.
    act_stats = {'act_scale': act_scale}
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = flax_util.get_or_create_variable(
        'quant_stats', weight_name + '_awq', lambda: aggregator.init(act_stats)
    )
    if flax_util.should_update_quant_stats():
      quant_stat.value = aggregator.update(quant_stat.value, act_stats)

    return res

  def get_intercept_map(self) -> dict[str, Callable[..., Any]]:
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general
    }


def quantize_params(
    params: Any,
    abstract_quantized_params: Any,
    awq_quant_stats: Any,
    *,
    allow_extra_params: bool = False,
    n_grid: int = 20,
) -> Any:
  """Quantizes the params with AWQ.

  Args:
    params: The floating-point param tree to quantize.
    abstract_quantized_params: The param tree generated by the PTQ model,
      containing WithAux wrappers with HowToQuantize information.
    awq_quant_stats: The quant_stats dict from AwqCalibrationProvider. For
      params with no awq_quant_stats, they will be quantized with the default
      PTQ algorithm.
    allow_extra_params: If True, allow extra parameters not in
      abstract_quantized_params.
    n_grid: Number of grid points for AWQ scale search.

  Returns:
    The quantized params consumable by AwqInferenceProvider. For AWQ-quantized
    weights, returns WithAwqScale wrappers containing the QArray and per-channel
    AWQ scales. For non-AWQ weights, returns WithAux wrappers (same as PTQ).
  """
  quantized_params = {}
  not_quantized_params = {}
  for path, w in flax.traverse_util.flatten_dict(params).items():
    abs_w = ptq.get_value_from_path(abstract_quantized_params, path)
    awq_stats_path = (*path[:-1], path[-1] + '_awq')
    awq_stats = ptq.get_value_from_path(awq_quant_stats, awq_stats_path)

    if not isinstance(abs_w, ptq.WithAux) or awq_stats is None:
      # Not quantized by AWQ.
      not_quantized_params[path] = w
      continue

    # Get the contracting axis by assuming that all non-contracting axes
    # are in channelwise_axes.
    contracting_axis = set(range(w.ndim)) - set(abs_w.how.channelwise_axes)
    assert len(contracting_axis) == 1
    contracting_axis = list(contracting_axis)[0]

    # Normalize the weight to (ra, ca) format.
    w, restore_shape = awq_core.normalize_weight(w, contracting_axis)
    how = dataclasses.replace(abs_w.how, channelwise_axes=[0])
    if contracting_axis in how.tiled_axes:
      how = dataclasses.replace(
          how, tiled_axes={1: how.tiled_axes[contracting_axis]}
      )

    # Get the activation scale, which should be (ca,).
    calibration = averaging.SimpleMovingAverage().get_calibration(awq_stats)
    act_scale = calibration['act_scale']
    assert act_scale.shape[0] == w.shape[1]

    # Quantize the weight with AWQ.
    w_q, scales = awq_core.quantize_weight(w, act_scale, how, n_grid=n_grid)

    # Restore original shape for QArray.
    w_q = restore_shape(w_q)

    # Store AWQ scales as 1D array (in_features,) for simplicity.
    # scales is (1, in_features), squeeze to (in_features,).
    awq_scale_1d = scales.squeeze(0)

    # Store AWQ scales separately for per-channel compensation during inference.
    quantized_params[path] = WithAwqScale(
        array=w_q,
        awq_scale=awq_scale_1d,
        contracting_axis=contracting_axis,
        how=abs_w.how,
    )

  # Quantize the non-AWQ params with PTQ.
  not_quantized_params = flax.traverse_util.unflatten_dict(not_quantized_params)
  ptq_quantized_params = ptq.quantize_params(
      not_quantized_params,
      abstract_quantized_params,
      allow_extra_params=allow_extra_params,
  )
  ptq_quantized_params = flax.traverse_util.flatten_dict(ptq_quantized_params)
  quantized_params.update(ptq_quantized_params)

  return flax.traverse_util.unflatten_dict(quantized_params)


class AwqInferenceProvider(qconfig.QuantizationProvider):
  """Inference provider for AWQ.

  This provider handles both WithAwqScale (AWQ-quantized weights) and
  WithAux (PTQ-quantized weights). For AWQ weights, it applies per-channel
  scale compensation after dequantization.
  """

  def __init__(
      self,
      rules: Sequence[qconfig.QuantizationRule],
      *,
      _dot_general_fn=dot_general_module.dot_general,
  ):
    """Initializes the AWQ inference provider."""
    super().__init__(rules)
    self._dot_general_fn = _dot_general_fn

  def dot_general(
      self,
      lhs: jax.Array,
      rhs: jax.Array | WithAwqScale | ptq.WithAux[qarray.QArray],
      dimension_numbers: jax.lax.DotDimensionNumbers,
      precision: jax.lax.PrecisionLike = None,
      preferred_element_type: jax.typing.DTypeLike | None = None,
      *,
      out_sharding: jax.sharding.NamedSharding | None = None,
  ) -> jax.Array:
    rule, _ = self._get_current_rule_and_op_id('dot_general')
    if rule is None or rule.weight_qtype is None:
      return jax.lax.dot_general(
          lhs,
          rhs,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          out_sharding=out_sharding,
      )

    # Handle AWQ-quantized weights with per-channel scale compensation.
    if isinstance(rhs, WithAwqScale):
      # Dequantize and apply per-channel AWQ scale compensation.
      rhs_dq = qarray.dequantize(rhs.array)

      # Reshape AWQ scales to broadcast along the contracting axis.
      # awq_scale is 1D (in_features,), need to add dims for broadcasting.
      scale_shape = [1] * rhs_dq.ndim
      scale_shape[rhs.contracting_axis] = rhs.awq_scale.shape[0]
      awq_scale_broadcast = rhs.awq_scale.reshape(scale_shape)

      rhs_compensated = rhs_dq / awq_scale_broadcast
      return jax.lax.dot_general(
          lhs,
          rhs_compensated,
          dimension_numbers,
          precision=precision,
          preferred_element_type=preferred_element_type,
          out_sharding=out_sharding,
      )

    # Handle PTQ-quantized weights (WithAux).
    if isinstance(rhs, ptq.WithAux):
      rhs = rhs.array

    # Use quantized dot_general for QArray.
    return self._dot_general_fn(
        lhs, rhs, dimension_numbers, out_sharding=out_sharding
    )

  def nn_param(self, module, name: str, *args, **kwargs):
    """Intercepts nn.Module.param to handle quantized params."""
    from flax import linen as nn

    existing_param = module.get_variable('params', name)
    if isinstance(existing_param, (WithAwqScale, ptq.WithAux)):
      return nn.unbox(existing_param)
    return module.param(name, *args, **kwargs)

  def promote_dtype(self, *args, **kwargs):
    """Intercepts flax.{linen,nnx.nn}.dtypes.promote_dtype."""
    if len(args) == 1 and isinstance(args[0], Sequence):
      args = args[0]  # nnx version
    # Skip WithAwqScale and WithAux.
    array_args = [x if isinstance(x, jax.Array) else None for x in args]
    array_args = flax.linen.dtypes.promote_dtype(*array_args, **kwargs)
    return [x if x is not None else y for x, y in zip(array_args, args)]

  def get_intercept_map(self):
    """Used for interception."""
    return super().get_intercept_map() | {
        'jax.lax.dot_general': self.dot_general,
        'flax.linen.Module.param': self.nn_param,
        'flax.linen.dtypes.promote_dtype': self.promote_dtype,
        'flax.nnx.nn.dtypes.promote_dtype': self.promote_dtype,
    }
