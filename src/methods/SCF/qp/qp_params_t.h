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


#ifndef COQUI_QP_CONTEXT_H
#define COQUI_QP_CONTEXT_H

#include <string>

namespace methods {

struct qp_params_t {
  std::string qp_type = "sc";
  std::string ac_alg = "pade";
  int Nfit = 18;
  double eta = 0.0001;
  double tol = 1e-8;

  // SCF mode selector:
  // - evscf: update only QP energies and keep QP wavefunctions fixed to mean-field ones.
  // - qpscf: update both QP energies and QP wavefunctions.
  // - lqsscf: update QP energies and QP wavefunctions using a matrix expansion of Sigma(iw) around w = 0 on the Matsubara axis. 
  std::string qp_scf_mode = "qpscf";

  // whether to update dynamically screened interaction W in evscf.
  bool keep_scr_coulomb_fixed = false;

  // off-diagonal mode defined in T. Kotani et. al., Phys. Rev. B 76, 165106 (2007)
  // "fermi": evaluate off-diagonal elements of self-energy at the Fermi level;
  // "qp_energy": evaluate off-diagonal elements of self-energy at the quasiparticle energy
  // (defined as the average of the two diagonal elements)
  std::string off_diag_mode = "fermi";

  double mu_tolerance = 1e-9;
  std::string mu_update_alg = "bisection";

  // Linearized QP (LQSGW) kernel controls; see methods::lqp::fit_params_t.
  // Sigma(iw) is fitted to a polynomial in (iw) on the 2*lqp_n_fit lowest Matsubara nodes
  // (+-1, +-3, ..., +-(2 n_fit - 1)).
  int         lqp_n_fit           = 6;
  // lqp_fit_order = -1 means fitting order 2 * n_fit - 1.
  int         lqp_fit_order       = -1;
  // Residual tolerance for the polynomial fit of Sigma(iw). 
  // Residual larger than this value (e.g. fitting becomes ill-conditioned) will abort the calculation.
  // Only enforced when the fit is exactly determined (lqp_fit_order = -1), where the polynomial
  // interpolates the data and the residual should be at machine precision; a reduced fit_order is a
  // least-squares fit whose residual is nonzero by construction and is not gated.
  double      lqp_fit_resid_tol   = 1e-8;
};

} // methods

#endif //COQUI_QP_CONTEXT_H
