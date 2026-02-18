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

"""Integration of QEP (Quantization Error Propagation) into Qwix.

QEP extends GPTQ by accounting for quantization noise in input activations
from previous layers. It requires a two-pass calibration (float + quantized)
to collect hessian_delta statistics, then applies weight correction before
standard GPTQ quantization.

During inference, QEP uses the same PtqProvider as GPTQ/PTQ. The only
difference is the calibration and weight quantization steps.

Please check the test for an example usage.
"""

import dataclasses
from typing import Any

import jax
from qwix._src import qconfig
from qwix.contrib import calibration
from qwix.contrib import gptq_core
from qwix.contrib import qep_core
from qwix.contrib.gptq import GptqRule


@dataclasses.dataclass(frozen=True, kw_only=True)
class QepRule(GptqRule):
  """Use this rule to enable QEP (input-compensated GPTQ).

  QEP extends GPTQ by accounting for quantization noise in input activations
  from previous layers.

  Attributes:
    correction_factor: Weight correction factor. 0.0 = no correction,
      1.0 = full correction. Default 0.5 per QEP paper recommendations.
    damping_factor: Dampening factor for QEP weight correction Hessian
      inversion. Default 1.0 (much higher than GPTQ's default of 0.01
      because this inversion needs more regularization).
  """

  correction_factor: float = 0.5
  damping_factor: float = 1.0


class QepCalibrationProvider(calibration.TwoPassCalibrationProvider):
  """Calibration provider for QEP (input-compensated GPTQ).

  QEP extends GPTQ by accounting for quantization noise in input activations.
  See ``TwoPassCalibrationProvider`` for the two-pass protocol documentation.

  Use calibrate_batch() to run both passes for a single batch::

    qep_provider = QepCalibrationProvider(rules)
    cal_model = qwix_model.quantize_model(model, qep_provider)

    for batch in calibration_data:
      new_vars = qep_provider.calibrate_batch(
          cal_model, float_variables, quant_variables, batch
      )
      variables.update(new_vars)
  """

  def get_rule_type(self) -> type[qconfig.QuantizationRule]:
    return QepRule

  def get_stats_suffix(self) -> str:
    return '_qep'

  def compute_stats(
      self, quantized_lhs: jax.Array, float_lhs: jax.Array
  ) -> dict[str, Any]:
    return qep_core.compute_qep_stats(quantized_lhs, float_lhs)


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
) -> Any:
  """Quantizes the params with QEP (weight correction + GPTQ).

  Args:
    params: See ptq.quantize_params.
    abstract_quantized_params: See ptq.quantize_params.
    qep_quant_stats: The quant_stats dict from QepCalibrationProvider.
      For params with no qep_quant_stats, they will be quantized with
      the default PTQ algorithm.
    allow_extra_params: See ptq.quantize_params.
    gptq_block_size: The block size of GPTQ.
    gptq_damping_factor: The damping factor of GPTQ.
    correction_factor: QEP weight correction factor. 0.0 = no correction,
      1.0 = full correction. Default 0.5 per QEP paper.
    damping_factor: QEP damping factor for Hessian inversion.
      Default 1.0.

  Returns:
    The quantized params consumable by PtqProvider.
  """
  def _quantize(prepared: calibration.PreparedWeight) -> Any:
    hessian = prepared.calibration_stats['hessian']
    assert (
        hessian.shape[0] == prepared.weight.shape[1]
        and hessian.shape[1] == prepared.weight.shape[1]
    )

    # QEP weight correction (applied BEFORE GPTQ quantization).
    hessian_delta = prepared.calibration_stats.get('hessian_delta')
    if hessian_delta is None:
      raise ValueError(
          f'hessian_delta not found in QEP stats for {prepared.path}. '
          f'Ensure QepCalibrationProvider was used for calibration.'
      )
    w = qep_core.weight_correct(
        prepared.weight,
        hessian,
        hessian_delta,
        perccorr=correction_factor,
        percdamp=damping_factor,
    )

    # Quantize the weight with GPTQ.
    w = gptq_core.quantize_weight(
        w,
        hessian,
        prepared.how,
        blocksize=gptq_block_size,
        percdamp=gptq_damping_factor,
    )[0]
    w = prepared.restore_shape(w)
    return prepared.abs_w.replace(array=w)

  return calibration.quantize_params_with_calibration(
      params,
      abstract_quantized_params,
      qep_quant_stats,
      '_qep',
      _quantize,
      allow_extra_params=allow_extra_params,
  )
