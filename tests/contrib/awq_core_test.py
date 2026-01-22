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
"""Tests for AWQ algorithm."""

import functools
import logging

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib import awq_core


def rel_rmse(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.mean((x - y) ** 2)) / jnp.sqrt(jnp.mean(y**2))


class AwqCoreTest(parameterized.TestCase):

  def test_compute_act_scale_basic(self):
    """Test basic activation scale computation."""
    x = jnp.array([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    act_scale = awq_core.compute_act_scale(x, axis=0)
    expected = jnp.array([1.0, 2.0, 3.0])  # mean(abs) along axis 0
    self.assertTrue(jnp.allclose(act_scale, expected))

  def test_compute_act_scale_axis1(self):
    """Test activation scale computation with axis=1."""
    x = jnp.array([[1.0, 2.0], [-3.0, 4.0], [5.0, -6.0]])
    act_scale = awq_core.compute_act_scale(x, axis=1)
    expected = jnp.array([1.5, 3.5, 5.5])
    self.assertTrue(jnp.allclose(act_scale, expected))

  def test_compute_act_scale_jit_compatible(self):
    """Test that compute_act_scale is JIT compatible."""
    x = jax.random.normal(jax.random.key(0), (100, 32))
    act_scale_eager = awq_core.compute_act_scale(x)
    act_scale_jit = jax.jit(awq_core.compute_act_scale)(x)
    self.assertTrue(jnp.allclose(act_scale_eager, act_scale_jit))

  @parameterized.named_parameters(
      dict(testcase_name='g128', groupsize=128),
      dict(testcase_name='g256', groupsize=256),
      dict(testcase_name='no_group', groupsize=None),
  )
  def test_search_optimal_scales(self, groupsize):
    """Test that grid search finds reasonable scales."""
    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (256, 512), jnp.float32
    )
    x = jax.random.normal(jax.random.key(1), (512, 1024))
    act_scale = awq_core.compute_act_scale(x.T, axis=0)

    tiled_axes = {1: groupsize} if groupsize else {}
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[0],
        tiled_axes=tiled_axes,
    )

    scales, ratio = awq_core.search_optimal_scales(w, act_scale, how)

    # Verify scale shape.
    self.assertEqual(scales.shape, (1, 512))
    # Verify ratio is in valid range.
    self.assertGreaterEqual(float(ratio), 0.0)
    self.assertLessEqual(float(ratio), 1.0)
    # Verify scales are positive.
    self.assertTrue(jnp.all(scales > 0))

  @parameterized.named_parameters(
      dict(testcase_name='int8_g128', qtype=jnp.int8, groupsize=128),
      dict(testcase_name='int4_g128', qtype=jnp.int4, groupsize=128),
      dict(testcase_name='int4_g256', qtype=jnp.int4, groupsize=256),
  )
  def test_quantize_weight(self, qtype, groupsize):
    """Test AWQ quantization improves matmul error over RTN."""
    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (256, 512), jnp.float32
    )
    x = jax.random.t(jax.random.key(1), 5, (512, 1024), jnp.float32)
    act_scale = awq_core.compute_act_scale(x.T, axis=0)

    how = qarray.HowToQuantize(
        qtype=qtype,
        channelwise_axes=[0],
        tiled_axes={1: groupsize},
    )

    # RTN (Round-to-Nearest) baseline.
    w_rtn = qarray.quantize(w, how)

    # AWQ quantization.
    w_awq, scales = jax.jit(
        functools.partial(awq_core.quantize_weight, how=how)
    )(w, act_scale)

    # Verify shapes match.
    self.assertEqual(
        jax.tree.map(lambda x: (x.shape, x.dtype), w_awq),
        jax.tree.map(lambda x: (x.shape, x.dtype), w_rtn),
    )

    # Dequantize.
    w_rtn_dq = qarray.dequantize(w_rtn)
    # For AWQ, we need to compensate for the scales using per-channel division.
    # This is the proper AWQ compensation - divide dequantized weights by scales.
    w_awq_dq = qarray.dequantize(w_awq) / scales

    # RTN typically has lower dequant error (because AWQ scales up before quant).
    mse_rtn_dq = rel_rmse(w_rtn_dq, w)
    mse_awq_dq = rel_rmse(w_awq_dq, w)
    logging.info('dequant loss rtn: %s vs. awq: %s', mse_rtn_dq, mse_awq_dq)
    # Note: AWQ trades weight accuracy for matmul accuracy, so dequant error
    # comparison is not strictly ordered.

    # But AWQ should have lower matmul error - this is the key metric.
    mse_rtn_matmul = rel_rmse(w_rtn_dq @ x, w @ x)
    mse_awq_matmul = rel_rmse(w_awq_dq @ x, w @ x)
    logging.info('matmul loss rtn: %s vs. awq: %s', mse_rtn_matmul, mse_awq_matmul)
    self.assertGreater(mse_rtn_matmul, mse_awq_matmul)

  def test_quantize_weight_jit_compatible(self):
    """Test that quantize_weight is JIT compatible."""
    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (64, 128), jnp.float32
    )
    act_scale = jax.random.uniform(jax.random.key(1), (128,), minval=0.1)
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[0],
        tiled_axes={1: 64},
    )

    # Should run without errors under JIT.
    w_awq, scales = jax.jit(
        functools.partial(awq_core.quantize_weight, how=how)
    )(w, act_scale)

    self.assertEqual(w_awq.shape, w.shape)
    self.assertEqual(scales.shape, (1, 128))

  def test_normalize_weight(self):
    """Test weight normalization for different axes."""
    w = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    w2, restore_shape = awq_core.normalize_weight(w, 1)
    self.assertEqual(w2.shape, (8, 3))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 3, 4))
    self.assertTrue(jnp.all(w == w3))

  def test_normalize_weight_different_axes(self):
    """Test normalization with contraction on last axis."""
    w = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    w2, restore_shape = awq_core.normalize_weight(w, 2)
    self.assertEqual(w2.shape, (6, 4))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 3, 4))
    self.assertTrue(jnp.all(w == w3))


if __name__ == '__main__':
  absltest.main()
