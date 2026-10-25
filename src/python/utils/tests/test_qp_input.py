"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================

Unit tests for the conversion of the pre-block flat quasiparticle keys into the nested
``qp_eqn`` block (utils.qp_input), the only place where the flat spellings are still accepted.
"""

import copy

import pytest

from coqui.utils.qp_input import (
  nest_flat_qp_keys,
  MBPT_FLAT_QP_KEYS,
  BAND_INTERPOLATION_FLAT_QP_KEYS,
)


def test_nest_flat_qp_keys():
  """Flat quasiparticle keys are moved into the qp_eqn block, per call-site key map.

  1. mbpt map: ac_alg, Nfit, eta, off_diag_mode become qp_eqn.{ac_alg, ac_nfit, eta,
     off_diag_mode}; unrelated keys pass through.
  2. band_interpolation map: qp_eta, qp_Nfit, qp_tol and ac_alg are moved, while flat eta and
     Nfit -- the spectral-function continuation keys of spectral_interpolation, which callers
     share dictionaries with -- are left in place. The same flat name means different things at
     the two call sites, which is why the map is an argument.
  3. qp_type is split into (qp_approx, qp_eqn.solver), case-insensitively; "lqp" selects the
     linearized family and sets no solver.
  4. A parameter given both flat and nested is ambiguous and raises, as does qp_type next to
     qp_approx or (for a qp_eqn solver) next to qp_eqn.solver.
  5. The caller's dictionary, including its nested qp_eqn block, is not modified, and input that
     is already nested comes back equal.
  """
  # 1.
  out = nest_flat_qp_keys(params={"beta": 100.0, "ac_alg": "pade", "Nfit": 26, "eta": 1e-6,
                                  "off_diag_mode": "qp_energy"},
                          flat_keys=MBPT_FLAT_QP_KEYS, caller="run_qpgw")
  assert out == {"beta": 100.0, "qp_eqn": {"ac_alg": "pade", "ac_nfit": 26, "eta": 1e-6,
                                           "off_diag_mode": "qp_energy"}}

  # 2.
  out = nest_flat_qp_keys(params={"ac_alg": "pade", "qp_eta": 1e-6, "qp_Nfit": 18,
                                  "qp_tol": 1e-9, "eta": 0.01, "Nfit": 30},
                          flat_keys=BAND_INTERPOLATION_FLAT_QP_KEYS, caller="band_interpolation")
  assert out == {"eta": 0.01, "Nfit": 30,
                 "qp_eqn": {"ac_alg": "pade", "eta": 1e-6, "ac_nfit": 18, "tol": 1e-9}}

  # 3.
  out = nest_flat_qp_keys(params={"qp_type": "SC_Newton", "qp_eqn": {"eta": 1e-6}},
                          flat_keys=MBPT_FLAT_QP_KEYS, caller="run_evgw")
  assert out == {"qp_approx": "qp_eqn", "qp_eqn": {"eta": 1e-6, "solver": "sc_newton"}}
  out = nest_flat_qp_keys(params={"qp_type": "lqp"},
                          flat_keys=BAND_INTERPOLATION_FLAT_QP_KEYS, caller="band_interpolation")
  assert out == {"qp_approx": "lqp"}

  # 4.
  for params in ({"eta": 1e-6, "qp_eqn": {"eta": 1e-4}},
                 {"qp_type": "sc", "qp_approx": "qp_eqn"},
                 {"qp_type": "sc", "qp_eqn": {"solver": "sc_newton"}}):
    with pytest.raises(ValueError):
      nest_flat_qp_keys(params=params, flat_keys=MBPT_FLAT_QP_KEYS, caller="run_evgw")

  # 5.
  params = {"Nfit": 26, "qp_eqn": {"solver": "sc"}}
  before = copy.deepcopy(params)
  nest_flat_qp_keys(params=params, flat_keys=MBPT_FLAT_QP_KEYS, caller="run_evgw")
  assert params == before
  nested = {"qp_approx": "qp_eqn", "qp_eqn": {"solver": "sc", "ac_nfit": 26}, "lqp": {"n_fit": 4}}
  assert nest_flat_qp_keys(params=nested, flat_keys=MBPT_FLAT_QP_KEYS,
                           caller="run_evgw") == nested
