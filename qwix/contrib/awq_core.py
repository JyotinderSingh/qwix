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
"""AWQ (Activation-aware Weight Quantization) algorithm.

This is a JAX implementation of the AWQ algorithm from:
https://arxiv.org/abs/2306.00978

AWQ identifies salient weights based on activation magnitudes and applies
equivalent transformations to improve quantization accuracy. The key insight
is that not all weights are equally important - weights connected to channels
with high activation magnitudes are more critical to preserve accurately.

AWQ scales are always per-channel (shape: 1, in_features). When combined with
groupwise quantization, the per-channel scales help protect salient channels
within each group, while the quantization uses per-group scales.
"""

from collections.abc import Callable

import jax
import jax.numpy as jnp
from qwix._src.core import qarray


def compute_act_scale(X: jax.Array, axis: int = 0) -> jax.Array:
  """Computes per-channel activation magnitudes.

  This function calculates the mean absolute value of activations for each
  channel, which is used to identify salient weight channels in AWQ.

  Args:
    X: Input activations with shape (n_samples, in_features) when axis=0, or
      (in_features, n_samples) when axis=1.
    axis: The sample axis to reduce over. Default is 0 (samples in first dim).

  Returns:
    Per-channel mean absolute activation with shape (in_features,).
  """
  return jnp.mean(jnp.abs(X), axis=axis)


def search_optimal_scales(
    W: jax.Array,
    act_scale: jax.Array,
    how: qarray.HowToQuantize,
    n_grid: int = 20,
    min_scale: float = 1e-4,
) -> tuple[jax.Array, jax.Array]:
  """Searches for optimal per-channel scaling factors using grid search.

  The AWQ algorithm searches for an optimal exponent 'ratio' such that
  scales = act_scale^ratio provides the best quantization accuracy. Larger
  ratios give more protection to salient channels (those with high activations).

  The key insight of AWQ is to minimize OUTPUT error (activation-weighted),
  not raw weight error. Channels with higher activation magnitudes contribute
  more to the output, so their weight errors should be weighted more heavily.

  AWQ scales are always per-channel, even when using groupwise quantization.
  This allows protecting salient channels within each quantization group.

  Args:
    W: Weight matrix with shape (out_features, in_features), where in_features
      is the contraction dimension.
    act_scale: Per-channel activation scale with shape (in_features,).
    how: How to quantize the weights.
    n_grid: Number of grid points to search. Default is 20.
    min_scale: Minimum scale value to prevent division by zero.

  Returns:
    A tuple of (optimal_scales, best_ratio):
      - optimal_scales: Per-channel scaling factors with shape (1, in_features).
      - best_ratio: The optimal ratio that was found.
  """
  ratios = jnp.linspace(0.0, 1.0, n_grid)

  # Normalize act_scale to prevent numerical issues.
  act_scale_normalized = act_scale / (act_scale.max() + 1e-8)
  act_scale_normalized = jnp.clip(act_scale_normalized, min=min_scale)

  def compute_loss_for_ratio(ratio: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Compute per-channel scales from activation magnitudes.
    scales = jnp.power(act_scale_normalized, ratio)
    scales = jnp.clip(scales, min=min_scale)
    # Normalize scales to prevent extreme values.
    scales = scales / jnp.sqrt(jnp.maximum(scales.max() * scales.min(), 1e-8))
    scales = scales.reshape(1, -1)

    # Apply per-channel scaling and quantize (possibly with groupwise quant).
    W_scaled = W * scales
    W_q = qarray.quantize(W_scaled, how)
    W_dq = qarray.dequantize(W_q)
    # Restore original scale with per-channel division.
    W_restored = W_dq / scales

    # Compute ACTIVATION-WEIGHTED loss (key AWQ insight).
    # This approximates output error: ||W @ X - W_q @ X||^2
    weight_error = W - W_restored
    weighted_error = weight_error * act_scale.reshape(1, -1)
    loss = jnp.mean(weighted_error ** 2)
    return loss, scales

  # Vectorize over all ratios for efficient computation.
  losses, all_scales = jax.vmap(compute_loss_for_ratio)(ratios)

  # Find the ratio with minimum loss.
  best_idx = jnp.argmin(losses)
  best_scales = all_scales[best_idx]
  best_ratio = ratios[best_idx]

  return best_scales, best_ratio


def quantize_weight(
    W: jax.Array,
    act_scale: jax.Array,
    how: qarray.HowToQuantize,
    n_grid: int = 20,
) -> tuple[qarray.QArray, jax.Array]:
  """Quantizes a weight matrix using AWQ.

  This function finds optimal per-channel scaling factors based on activation
  magnitudes and applies them before quantization to preserve salient channels.

  The returned scales are per-channel (1, in_features). When using groupwise
  quantization, the per-channel scales help protect important channels within
  each group while the quantization itself uses per-group scales.

  Args:
    W: Weight matrix with shape (out_features, in_features), where in_features
      is the contraction dimension.
    act_scale: Per-channel activation scale from compute_act_scale with shape
      (in_features,).
    how: How to quantize the weights.
    n_grid: Number of grid points for scale search.

  Returns:
    A tuple of (W_q, scales):
      - W_q: The quantized weight matrix as a QArray. The weights have been
        scaled by per-channel factors before quantization.
      - scales: The per-channel scaling factors that were applied, shape
        (1, in_features). Dequantizing and dividing by scales gives the best
        approximation of the original weights.
  """
  optimal_scales, _ = search_optimal_scales(W, act_scale, how, n_grid)

  # Scale up salient channels before quantization.
  W_scaled = W * optimal_scales

  # Quantize the scaled weights (may use groupwise quantization).
  W_q = qarray.quantize(W_scaled, how)

  return W_q, optimal_scales


def normalize_weight(
    x: jax.Array, contraction_axis: int
) -> tuple[jax.Array, Callable[..., qarray.MaybeQArray]]:
  """Normalizes the weight into (ra, ca) format for AWQ.

  This function reshapes a weight tensor of arbitrary rank into a 2D matrix
  where the contraction axis becomes the last dimension. This is needed because
  AWQ operates on (out_features, in_features) matrices.

  Args:
    x: Weight tensor of arbitrary shape.
    contraction_axis: The axis that will be contracted in the matrix multiply.

  Returns:
    A tuple of (normalized_weight, restore_shape):
      - normalized_weight: The weight reshaped to (ra, ca) format.
      - restore_shape: A function to restore the original shape.
  """
  # Move the contraction axis to the last dimension.
  x = jnp.moveaxis(x, contraction_axis, -1)
  before_shape = x.shape
  # Reshape the weight to (ra, ca).
  x = x.reshape(-1, x.shape[-1])

  def restore_shape(x: qarray.MaybeQArray) -> qarray.MaybeQArray:
    x = x.reshape(before_shape)
    return jax.tree.map(lambda x: jnp.moveaxis(x, -1, contraction_axis), x)

  return x, restore_shape
