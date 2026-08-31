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


#ifndef COQUI_ITER_SCF_UTILS_HPP
#define COQUI_ITER_SCF_UTILS_HPP

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "numerics/iter_scf/iter_scf_t.hpp"

namespace iter_scf {
  /**
   * Create a self-consistent iterative solver with arguments in property tree.
   * Requires:
   *  - alg: algorithm. {choices: "damping", "DIIS"}
   * Damping options:
   *  - mixing: "0.7" Mixing of the current iteration.
   * DIIS options:
   *  - mixing: "0.7" For initial damping
   *  - max_subsp_size: "5" Maximal dimension of the extrapolation subspace
   *  - diis_start: "3" When to start applying DIIS extrapolation. 
   */
  inline decltype(auto) make_iter_scf(ptree const& base_pt, double default_mixing=0.7, bool is_evscf=false) {
    for (auto const& it : base_pt) {
      if (it.first == "iter_alg") {
        ptree pt = it.second;
        auto v = pt.get_value_optional<std::string>();
        if (v.has_value() and *v != "") {
          utils::check(false, "iter_alg has to be a property tree.");
        } else {
          auto alg = io::get_value<std::string>(pt,"alg","iter_alg - missing alg type: damping");
          io::tolower(alg);
          auto mixing = io::get_value_with_default<double>(pt,"mixing",0.7);

          if (alg == "damping") {

            return iter_scf_t(damp_t(mixing));

          } else if (alg == "diis") {

            auto max_subsp_size = io::get_value_with_default<size_t>(pt,"max_subsp_size",5);
            auto warmup_iter = io::get_value_with_default<size_t>(pt,"diis_warmup",3);
            auto residual_type = io::get_value_with_default<std::string>(
              pt, "residual_type", is_evscf ? "vector_diff" : "commutator");
            io::tolower(residual_type);
            auto diis_start = io::get_value_with_default<size_t>(pt,"diis_start",-1);
            if (diis_start != -1) {
              app_log(1, "[WARNING] 'diis_start' is deprecated. Please use 'diis_warmup' instead \n"
                         "for setting up the number of damping iteration before the DIIS execution.\n");
              app_log(1, "╔═════════════════════════════════════════════════╗");
              app_log(1, "║ [ WARNING ]                                     ║");
              app_log(1, "║ \"diis_start\" is deprecated.                     ║");
              app_log(1, "║ Use \"diis_warmup\" instead to set the number of  ║");
              app_log(1, "║ damping iteration before DIIS execution.        ║");
              app_log(1, "╚═════════════════════════════════════════════════╝\n");
              warmup_iter = diis_start;
            }

            return iter_scf_t(diis_t(mixing, max_subsp_size, warmup_iter, residual_type));

          } else {
            utils::check(false, "Unrecognized algorithm type for iterative solver. ");
            return iter_scf_t("damping");
          }
        }
      }
    }
    return iter_scf_t(diis_t(default_mixing, 5, 3, is_evscf ? "vector_diff" : "commutator"));
  }
} // iter_scf

#endif //COQUI_ITER_SCF_UTILS_HPP
