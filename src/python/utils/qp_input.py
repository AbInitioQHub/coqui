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

Backward compatibility of the quasiparticle input, in one place.

The C++ drivers read the quasiparticle controls only in their nested form -- ``qp_approx``
plus the ``qp_eqn`` and ``lqp`` blocks. Input written before the blocks existed spelled the
``qp_eqn`` controls as flat keys of the solver's own dictionary and selected the scheme with
``qp_type``. The Python entry points that used to accept those spellings pass their parameters
through :func:`nest_flat_qp_keys` before handing them to C++; TOML input has no such layer and
must use the nested form.

The flat names differ per call site (``eta`` in the MBPT and downfolding drivers, ``qp_eta`` in
``band_interpolation``, where a flat ``eta`` belongs to the spectral-function continuation), so
each site passes the key map it used to accept. The ``lqp`` controls never had a flat form.
"""

from coqui._lib.utils_module import app_log

# flat key -> qp_eqn key, for run_evgw, run_qpgw and downfold_1e (qp_selfenergy = True)
MBPT_FLAT_QP_KEYS = {"ac_alg": "ac_alg", "Nfit": "ac_nfit", "eta": "eta",
                     "off_diag_mode": "off_diag_mode"}

# flat key -> qp_eqn key, for band_interpolation
BAND_INTERPOLATION_FLAT_QP_KEYS = {"ac_alg": "ac_alg", "qp_Nfit": "ac_nfit", "qp_eta": "eta",
                                   "qp_tol": "tol"}


def nest_flat_qp_keys(params, flat_keys, caller):
  """Return a copy of ``params`` with the flat quasiparticle keys moved into the qp_eqn block.

  Parameters
  ----------
  params : dict
      Parameters of one solver or post-processing call. Not modified.
  flat_keys : dict
      Map from the flat spellings this call site accepted to their ``qp_eqn`` keys
      (:data:`MBPT_FLAT_QP_KEYS` or :data:`BAND_INTERPOLATION_FLAT_QP_KEYS`).
  caller : str
      Name of the public function, used in the messages.

  Returns
  -------
  dict
      ``params`` with every key of ``flat_keys`` moved to ``qp_eqn.<key>``, and ``qp_type``
      replaced by ``qp_approx = "lqp"`` (for ``"lqp"``) or by ``qp_approx = "qp_eqn"`` plus
      ``qp_eqn.solver`` (for any other value). Input without flat keys comes back equal.

  Raises
  ------
  ValueError
      If a parameter is given both flat and nested, or ``qp_type`` is given together with
      ``qp_approx`` or with ``qp_eqn.solver``: the old code silently preferred the nested key,
      which hides a typo in either.
  """
  out = dict(params)
  qp_eqn = dict(out.get("qp_eqn", {}))
  moved = []

  for flat, nested in flat_keys.items():
    if flat not in out:
      continue
    if nested in qp_eqn:
      raise ValueError(f"{caller}: '{flat}' and 'qp_eqn.{nested}' set the same parameter; "
                       f"keep only 'qp_eqn.{nested}'.")
    qp_eqn[nested] = out.pop(flat)
    moved.append(f"{flat} -> qp_eqn.{nested}")

  if "qp_type" in out:
    legacy = str(out.pop("qp_type")).lower()
    if "qp_approx" in out:
      raise ValueError(f"{caller}: 'qp_type' and 'qp_approx' both select the quasiparticle "
                       f"approximation; keep only 'qp_approx'.")
    if legacy == "lqp":
      out["qp_approx"] = "lqp"
      moved.append("qp_type = lqp -> qp_approx = lqp")
    else:
      if "solver" in qp_eqn:
        raise ValueError(f"{caller}: 'qp_type' and 'qp_eqn.solver' both select the "
                         f"quasiparticle-equation solver; keep only 'qp_eqn.solver'.")
      out["qp_approx"] = "qp_eqn"
      qp_eqn["solver"] = legacy
      moved.append(f"qp_type = {legacy} -> qp_approx = qp_eqn, qp_eqn.solver = {legacy}")

  if qp_eqn or "qp_eqn" in out:
    out["qp_eqn"] = qp_eqn
  if moved:
    app_log(1, f" [WARNING] {caller}: deprecated flat quasiparticle keys, read as\n"
               + "\n".join(f"           {m}" for m in moved)
               + "\n           Use the nested form.")
  return out
