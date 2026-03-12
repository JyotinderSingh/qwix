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

from absl.testing import absltest
from flax import nnx
import jax
import jax.numpy as jnp
from qwix._src import averaging
from qwix._src import model as qwix_model
from qwix._src.providers import ptq
from qwix.contrib import qep
from qwix.contrib import qep_core


def _mae(a, b):
  return jnp.mean(jnp.abs(a - b))


def _merge_dict_trees(base, updates):
  if isinstance(base, dict) and isinstance(updates, dict):
    merged = dict(base)
    for key, value in updates.items():
      if key in merged:
        merged[key] = _merge_dict_trees(merged[key], value)
      else:
        merged[key] = value
    return merged
  return updates


class DenseNnxModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.Dense_0 = nnx.Linear(32, 128, rngs=rngs)
    self.Dense_1 = nnx.Linear(128, 64, rngs=rngs)

  def __call__(self, x, return_hidden=False):
    x = self.Dense_0(x)
    x = jax.nn.gelu(x)
    hidden = x
    x = self.Dense_1(x)
    if return_hidden:
      return hidden, x
    return x


class BranchNnxModel(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.DenseA = nnx.Linear(12, 16, rngs=rngs)
    self.DenseB = nnx.Linear(12, 16, rngs=rngs)
    self.DenseC = nnx.Linear(16, 8, rngs=rngs)

  def __call__(self, x):
    a = self.DenseA(x)
    b = self.DenseB(x)
    return self.DenseC(jax.nn.relu(a + b))


class QepNnxTest(absltest.TestCase):

  def test_exact_qep_beats_ptq_and_result_model_is_callable(self):
    model = DenseNnxModel(nnx.Rngs(0))
    x = jax.random.normal(jax.random.key(0), (8, 32))
    fp_y = model(x)
    rules = [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]

    result = qep.quantize(model, [x], rules)
    qep_y = result.model(x)

    ptq_model = qwix_model.quantize_model(model, ptq.PtqProvider(rules), x)
    ptq_y = ptq_model(x)

    self.assertLess(_mae(fp_y, qep_y), _mae(fp_y, ptq_y))

  def test_exact_stagewise_matches_manual_two_stage_reference(self):
    model = DenseNnxModel(nnx.Rngs(1))
    x = jax.random.normal(jax.random.key(1), (8, 32))

    result = qep.quantize(
        model, [x], [qep.QepRule(module_path='Dense_.*', weight_qtype=jnp.int8)]
    )
    exact_y = result.model(x)

    stage0_result = qep.quantize(
        model, [x], [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
    )
    hidden_q, _ = stage0_result.model(x, return_hidden=True)
    hidden_fp, _ = model(x, return_hidden=True)

    stage1_abs = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            model,
            ptq.PtqProvider(
                [qep.QepRule(module_path='Dense_1', weight_qtype=jnp.int8)]
            ),
            x,
        )
    )
    orig_params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    stats = qep_core.compute_qep_stats(hidden_q.T, hidden_fp.T)
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = aggregator.update(aggregator.init(stats), stats)
    stage1_params = qep.quantize_params(
        orig_params,
        stage1_abs,
        {'Dense_1': {'kernel_qep': quant_stat}},
    )
    merged_params = _merge_dict_trees(
        stage1_params, {'Dense_0': stage0_result.params['Dense_0']}
    )

    ptq_model = qwix_model.quantize_model(
        model,
        ptq.PtqProvider([qep.QepRule(module_path='Dense_.*', weight_qtype=jnp.int8)]),
        x,
    )
    nnx.update(ptq_model, merged_params)
    ref_y = ptq_model(x)

    self.assertLess(_mae(exact_y, ref_y), 1e-6)

  def test_infers_shared_input_branch_stage(self):
    model = BranchNnxModel(nnx.Rngs(2))
    x = jax.random.normal(jax.random.key(2), (8, 12))
    result = qep.quantize(
        model, [x], [qep.QepRule(module_path='.*', weight_qtype=jnp.int8)]
    )

    self.assertLen(result.stages, 2)
    self.assertEqual(set(result.stages[0].module_paths), {'DenseA', 'DenseB'})
    self.assertEqual(result.stages[1].module_paths, ('DenseC',))

  def test_quantize_params_returns_pure_dict_for_nnx(self):
    model = DenseNnxModel(nnx.Rngs(3))
    x = jax.random.normal(jax.random.key(3), (4, 32))
    abs_quantized = nnx.eval_shape(
        lambda: qwix_model.quantize_model(
            model,
            ptq.PtqProvider(
                [qep.QepRule(module_path='Dense_0', weight_qtype=jnp.int8)]
            ),
            x,
        )
    )
    orig_params = nnx.to_pure_dict(nnx.state(model, nnx.Param))
    fake_hessian = jnp.eye(32)
    aggregator = averaging.SimpleMovingAverage()
    quant_stat = aggregator.update(
        aggregator.init({'hessian': fake_hessian}),
        {'hessian': fake_hessian},
    )
    quantized_params = qep.quantize_params(
        orig_params,
        abs_quantized,
        {'Dense_0': {'kernel_qep': quant_stat}},
        apply_correction=False,
    )

    self.assertIsInstance(quantized_params, dict)
    nnx.update(abs_quantized, quantized_params)
    y = abs_quantized(x)
    self.assertEqual(y.shape, (4, 64))

  def test_sharded_linear_smoke(self):
    if jax.device_count() < 4:
      self.skipTest('test requires at least 4 devices')
    mesh = jax.make_mesh(
        (2, 2),
        ('contraction', 'remaining'),
        axis_types=(
            jax.sharding.AxisType.Auto,
            jax.sharding.AxisType.Auto,
        ),
    )
    x = jnp.ones((8, 12))

    with jax.set_mesh(mesh):
      model = nnx.Linear(
          in_features=12,
          out_features=6,
          rngs=nnx.Rngs(0),
          kernel_init=nnx.with_partitioning(
              nnx.initializers.lecun_normal(), ('contraction', 'remaining')
          ),
      )
      result = qep.quantize(
          model, [x], [qep.QepRule(module_path='.*', weight_qtype=jnp.int8)]
      )

    y = result.model(x)
    self.assertEqual(y.shape, (8, 6))
    self.assertEqual(
        result.model.kernel.array.qvalue.sharding_names,
        ('contraction', 'remaining'),
    )


if __name__ == '__main__':
  absltest.main()
