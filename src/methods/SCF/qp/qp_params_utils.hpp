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


#ifndef COQUI_QP_PARAMS_UTILS_HPP
#define COQUI_QP_PARAMS_UTILS_HPP

#include <string>

#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "methods/SCF/qp/qp_context_t.h"

namespace methods {

/**
 * Reads the quasiparticle input of one block on top of the call site's own defaults:
 *
 *   qp_approx = "qp_eqn" | "lqp"
 *     [<block>.qp_eqn]  solver, ac_alg, ac_nfit, eta, tol, off_diag_mode
 *     [<block>.lqp]     n_fit, fit_order, fit_resid_tol
 *
 * Defaults belong to the call site, not to this function. They differ between drivers and some
 * are only known at run time (eta = pi/beta, tol = conv_thr in evgw, ac_nfit = 30 in the
 * downfolding drivers), so the caller sets them on `defaults` first and this function overrides
 * exactly the members whose key is present in `pt`. 
 *
 * @param pt       - [INPUT] the block's subtree (e.g. [evgw], [qpgw], [...band_interpolation])
 * @param defaults - [INPUT/OUTPUT] enters with the call site's defaults, leaves with the input
 *                   applied on top: updated in place. 
 * @note Aborts on an unknown qp_approx or qp_eqn.off_diag_mode.
 */
inline void read_qp_block(ptree const& pt, qp_context_t& defaults) {
  using io::get_value_with_default;
  defaults.qp_approx = get_value_with_default<std::string>(pt, "qp_approx", defaults.qp_approx);
  io::tolower(defaults.qp_approx);
  utils::check(defaults.qp_approx == "qp_eqn" or defaults.qp_approx == "lqp",
               "read_qp_block: unknown qp_approx {}. Valid options are \"qp_eqn\" (analytic "
               "continuation + scalar QP equation) and \"lqp\" (matrix linearization).",
               defaults.qp_approx);
  // Quasiparticle equation on the real axis
  auto& qp_eqn = defaults.qp_eqn;
  qp_eqn.solver        = get_value_with_default<std::string>(pt, "qp_eqn.solver", qp_eqn.solver);
  qp_eqn.ac_alg        = get_value_with_default<std::string>(pt, "qp_eqn.ac_alg", qp_eqn.ac_alg);
  qp_eqn.ac_nfit       = get_value_with_default<int>(pt, "qp_eqn.ac_nfit", qp_eqn.ac_nfit);
  qp_eqn.eta           = get_value_with_default<double>(pt, "qp_eqn.eta", qp_eqn.eta);
  qp_eqn.tol           = get_value_with_default<double>(pt, "qp_eqn.tol", qp_eqn.tol);
  qp_eqn.off_diag_mode = get_value_with_default<std::string>(pt, "qp_eqn.off_diag_mode",
                                                             qp_eqn.off_diag_mode);
  // Matrix linearization on the Matsubara axis
  auto& lqp = defaults.lqp;
  lqp.n_fit         = get_value_with_default<int>(pt, "lqp.n_fit", lqp.n_fit);
  lqp.fit_order     = get_value_with_default<int>(pt, "lqp.fit_order", lqp.fit_order);
  lqp.fit_resid_tol = get_value_with_default<double>(pt, "lqp.fit_resid_tol", lqp.fit_resid_tol);

  io::tolower(qp_eqn.solver);
  io::tolower(qp_eqn.ac_alg);
  io::tolower(qp_eqn.off_diag_mode);
  utils::check(qp_eqn.off_diag_mode == "fermi" or qp_eqn.off_diag_mode == "qp_energy",
               "read_qp_block: unknown qp_eqn.off_diag_mode {}. Valid options are \"fermi\" "
               "and \"qp_energy\".", qp_eqn.off_diag_mode);
}

} // methods

#endif //COQUI_QP_PARAMS_UTILS_HPP
