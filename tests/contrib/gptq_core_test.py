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
"""Tests for GPTQ algorithm."""

import functools
import logging

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from qwix._src.core import qarray
from qwix.contrib import gptq_core


def rel_rmse(x: jax.Array, y: jax.Array) -> jax.Array:
  return jnp.sqrt(jnp.mean((x - y) ** 2)) / jnp.sqrt(jnp.mean(y**2))


class GptqCoreTest(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="g128b128",
          groupsize=128,
          blocksize=128,
      ),
      dict(
          testcase_name="g256b128",
          groupsize=256,
          blocksize=128,
      ),
      dict(
          testcase_name="g128b256",
          groupsize=128,
          blocksize=256,
      ),
  )
  def test_quantize_weight(self, groupsize, blocksize):
    w = jax.nn.initializers.lecun_normal()(
        jax.random.key(0), (256, 512), jnp.float32
    )
    x = jax.random.t(jax.random.key(1), 5, (512, 1024), jnp.float32)
    how = qarray.HowToQuantize(
        qtype=jnp.int8,
        channelwise_axes=[0],
        tiled_axes={1: groupsize},
    )
    w_rtn = qarray.quantize(w, how)
    h = gptq_core.compute_hessian(x)
    w_gptq, losses = jax.jit(
        functools.partial(
            gptq_core.quantize_weight, how=how, blocksize=blocksize
        )
    )(w, h)
    self.assertEqual(
        jax.tree.map(lambda x: (x.shape, x.dtype), w_gptq),
        jax.tree.map(lambda x: (x.shape, x.dtype), w_rtn),
    )
    self.assertEqual(losses.shape, w_rtn.shape)

    # dequant loss.
    w_rtn = qarray.dequantize(w_rtn)
    w_gptq = qarray.dequantize(w_gptq)
    mse_rtn = rel_rmse(w_rtn, w)
    mse_gptq = rel_rmse(w_gptq, w)
    logging.info("dequant loss rtn: %s vs. gptq: %s", mse_rtn, mse_gptq)
    self.assertLess(mse_rtn, mse_gptq)

    # matmul loss.
    mse_rtn = rel_rmse(w_rtn @ x, w @ x)
    mse_gptq = rel_rmse(w_gptq @ x, w @ x)
    logging.info("matmul loss rtn: %s vs. gptq: %s", mse_rtn, mse_gptq)
    self.assertGreater(mse_rtn, mse_gptq)

  def test_normalize_weight(self):
    w = jnp.arange(2 * 3 * 4).reshape(2, 3, 4)
    w2, restore_shape = gptq_core.normalize_weight(w, 1)
    self.assertEqual(w2.shape, (8, 3))
    w3 = restore_shape(w2)
    self.assertEqual(w3.shape, (2, 3, 4))
    self.assertTrue(jnp.all(w == w3))

  def test_qarray_identity_reshape_with_non_divisible_scale(self):
    """Test that QArray identity reshape works when scale doesn't evenly divide qvalue."""
    # Create a QArray where scale dimension doesn't evenly divide qvalue dimension
    # This simulates what happens after GPTQ quantization with subchannel quantization
    # when the number of columns isn't evenly divisible by groupsize
    qvalue = jnp.ones((256, 3525), dtype=jnp.int8)
    # 14 scale groups for 3525 columns: 3525 % 14 = 251 != 0
    scale = jnp.ones((256, 14), dtype=jnp.float32)

    q = qarray.QArray(qvalue, scale)

    # Identity reshape should work even when scale doesn't evenly divide
    # This tests the fix for the ValueError:
    # "Cannot reshape (256, 3525) into (256, 3525) for (256, 14)."
    q_reshaped = q.reshape((256, 3525))
    self.assertEqual(q_reshaped.shape, (256, 3525))
    # For identity reshape, the result should be the same object or equivalent
    self.assertTrue(jnp.array_equal(q_reshaped.qvalue, q.qvalue))
    self.assertTrue(jnp.array_equal(q_reshaped.scale, q.scale))


if __name__ == "__main__":
  absltest.main()
