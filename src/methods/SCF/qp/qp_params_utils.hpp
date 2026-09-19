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

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/check.hpp"
#include "methods/SCF/qp/qp_params_t.h"

namespace methods {

/**
 * Reads the quasiparticle input of one solver or post-processing block:
 *
 *   qp_approx = "qp_eqn" | "lqp"
 *     [<block>.qp_eqn]  solver, ac_alg, ac_nfit, eta, tol, off_diag_mode
 *     [<block>.lqp]     n_fit, fit_order, fit_resid_tol
 *
 * @param pt - [INPUT] the block's subtree (e.g. [gw], [qpgw], [...band_interpolation])
 * @param p  - [INPUT/OUTPUT] enters carrying the call site's defaults, which any key present
 *             in pt overrides. Fields the input never sets (qp_scf_mode) are left untouched.
 *
 * Deprecated spelling: a single `qp_type` used to select both the family and, within the
 * qp_eqn family, the solver. It is still read, mapped onto (qp_approx, qp_eqn.solver), and warned
 * about once. Flat keys that predate the blocks (`ac_alg`, `Nfit`, `eta`, ...) are *not*
 * handled here: their names and defaults differ per call site, so each site reads its own
 * and passes them in through `p`.
 */
inline void read_qp_block(ptree const& pt, qp_params_t& p) {
  auto legacy = io::get_value_with_default<std::string>(pt, "qp_type", "");
  if (not legacy.empty()) {
    io::tolower(legacy);
    app_log(1, " [WARNING] qp_type is deprecated. Use qp_approx = \"qp_eqn\" or \"lqp\", and,\n"
               "           for the quasiparticle-equation schemes, the qp_eqn.solver key. Reading\n"
               "           qp_type = {} as {}.", legacy,
            (legacy == "lqp")? "qp_approx = lqp" : "qp_approx = qp_eqn, qp_eqn.solver = " + legacy);
    if (legacy == "lqp") {
      p.qp_approx = "lqp";
    } else {
      p.qp_approx = "qp_eqn";
      p.qp_eqn.solver = legacy;
    }
  }

  p.qp_approx        = io::get_value_with_default<std::string>(pt, "qp_approx", p.qp_approx);
  utils::check(p.qp_approx == "qp_eqn" or p.qp_approx == "lqp",
               "read_qp_block: unknown qp_approx {}. Valid options are \"qp_eqn\" (analytic "
               "continuation + scalar QP equation) and \"lqp\" (matrix linearization).",
               p.qp_approx);
  // Quasiparticle equation on the real axis
  p.qp_eqn.solver  = io::get_value_with_default<std::string>(pt, "qp_eqn.solver", p.qp_eqn.solver);
  p.qp_eqn.ac_alg  = io::get_value_with_default<std::string>(pt, "qp_eqn.ac_alg", p.qp_eqn.ac_alg);
  p.qp_eqn.ac_nfit = io::get_value_with_default<int>(pt, "qp_eqn.ac_nfit", p.qp_eqn.ac_nfit);
  p.qp_eqn.eta     = io::get_value_with_default<double>(pt, "qp_eqn.eta", p.qp_eqn.eta);
  p.qp_eqn.tol     = io::get_value_with_default<double>(pt, "qp_eqn.tol", p.qp_eqn.tol);
  p.qp_eqn.off_diag_mode = io::get_value_with_default<std::string>(pt, "qp_eqn.off_diag_mode",
                                                                   p.qp_eqn.off_diag_mode);
  // Matrix linearization on the Matsubara axis
  p.lqp.n_fit         = io::get_value_with_default<int>(pt, "lqp.n_fit", p.lqp.n_fit);
  p.lqp.fit_order     = io::get_value_with_default<int>(pt, "lqp.fit_order", p.lqp.fit_order);
  p.lqp.fit_resid_tol = io::get_value_with_default<double>(pt, "lqp.fit_resid_tol",
                                                           p.lqp.fit_resid_tol);

  io::tolower(p.qp_approx);
  io::tolower(p.qp_eqn.solver);
  io::tolower(p.qp_eqn.ac_alg);
  io::tolower(p.qp_eqn.off_diag_mode);
}

} // methods

#endif //COQUI_QP_PARAMS_UTILS_HPP
