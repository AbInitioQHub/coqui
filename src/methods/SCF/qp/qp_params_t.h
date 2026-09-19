/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef COQUI_QP_PARAMS_T_H
#define COQUI_QP_PARAMS_T_H

#include <string>

namespace methods::lqp {

/**
 * Controls of the low-frequency fit of Sigma(iw) and of the pole-weight extraction, consumed
 * by the linearized quasiparticle kernel (linearized_qp.hpp, which includes this header rather
 * than the reverse: the inputs must not depend on the kernel). Defaults are the ones validated
 * in implementation_notes/quasiparticle_approx/quasiparticle_Z_LQSGW_note.tex.
 */
struct fit_params_t {
  // Number of lowest positive fermionic Matsubara frequencies in the fit window. With
  // symmetric_window the window is n = +/-1, +/-3, ..., +/-(2 n_fit - 1), i.e. 2 n_fit nodes.
  int    n_fit            = 6;
  // Highest power of (i w) in the polynomial; -1 means "#nodes - 1", the exactly-determined
  // (interpolating) fit, which is the recommended setting.
  int    fit_order        = -1;
  // Use +/-w pairs. This makes the Hermiticity of the coefficients exact by construction and
  // decouples the even/odd powers; turning it off is for diagnostics only.
  bool   symmetric_window = true;
  // Residual gate of an exactly-determined fit: the polynomial must reproduce the data to this
  // relative accuracy, otherwise the design matrix has lost conditioning (too many nodes) and
  // the coefficients are unreliable.
  double fit_resid_tol    = 1e-8;
};

} // methods::lqp

namespace methods {

/**
 * Controls of the quasiparticle-equation schemes: the diagonal Sigma_aa(iw) is continued to
 * the real axis and the scalar quasiparticle equation is solved there. The block configures
 * the whole scheme, not only the continuation -- the ac_* keys are the continuation that
 * feeds the equation. Input block [<block>.qp_eqn]; solvers in qp_solvers.hpp (qp_eqn_*).
 */
struct qp_eqn_params_t {
  // Which scalar quasiparticle equation is solved on the real axis:
  // - "sc" / "sc_bisection": bisection on w = eps + Re Sigma(w) ("sc" is an alias)
  // - "sc_newton":           secant iteration on the same equation
  // - "spectral":            peak of the spectral function on a scan of w
  // - "linearized":          first-order Taylor expansion of Sigma(w) around eps_KS
  //                          (a different approximation, not a cheaper solver)
  std::string solver        = "sc";
  // Analytic continuation algorithm passed to analyt_cont::AC_t.
  std::string ac_alg        = "pade";
  // Number of Matsubara points used by the continuation. Unrelated to lqp.n_fit, which counts
  // the Matsubara nodes of the polynomial fit in the linearized scheme.
  int         ac_nfit       = 18;
  // Broadening [Ha] at which the continued Sigma(w + i*eta) is evaluated.
  double      eta           = 1e-4;
  // Convergence tolerance of the root finder ("sc*", "spectral").
  double      tol           = 1e-8;
  // Off-diagonal mode of T. Kotani et al., Phys. Rev. B 76, 165106 (2007):
  // "fermi": evaluate off-diagonal elements of the self-energy at the Fermi level;
  // "qp_energy": at the quasiparticle energy (the average of the two diagonal elements).
  std::string off_diag_mode = "fermi";
};

/**
 * Inputs of every quasiparticle calculation, self-consistent or one-shot.
 *
 * Two independent axes:
 * - qp_approx: how the dynamic Sigma(iw) is reduced to a static quasiparticle Hamiltonian;
 * - qp_scf_mode: what is updated in a self-consistent loop. 
 * The scheme-specific controls live in the member named after the scheme (qp_eqn, lqp),
 * which mirrors the input blocks [<block>.qp_eqn] and [<block>.lqp].
 */
struct qp_params_t {
  // How Sigma(iw) becomes a static quasiparticle Hamiltonian:
  // - "qp_eqn": analytic continuation + the scalar quasiparticle equation on the real axis
  //             (see qp_eqn)
  // - "lqp":    matrix linearization Sigma(iw) ~ A + iw B on the Matsubara axis (see lqp)
  std::string qp_approx   = "qp_eqn";

  // What the self-consistent loop updates;
  // - "evscf": QP energies only, QP wavefunctions stay at the mean-field ones
  // - "qpscf": both QP energies and QP wavefunctions
  std::string qp_scf_mode = "qpscf";

  // whether to update dynamically screened interaction W in evscf.
  bool        keep_scr_coulomb_fixed = false;

  double      mu_tolerance  = 1e-9;
  std::string mu_update_alg = "bisection";

  // Scheme-specific controls, mirroring the input blocks of the same name.
  qp_eqn_params_t   qp_eqn;
  lqp::fit_params_t lqp;
};

} // methods

#endif //COQUI_QP_PARAMS_T_H
