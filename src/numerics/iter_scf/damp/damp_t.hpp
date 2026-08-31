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


#ifndef COQUI_DAMP_T_HPP
#define COQUI_DAMP_T_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

namespace iter_scf {
  /**
   * Damping for an arbitrary target
   */
  struct damp_t {
  public:

    static constexpr iter_alg_e iter_alg = damping;
    static constexpr iter_alg_e get_iter_alg() { return iter_alg; }

    damp_t() = default;
    damp_t(double mixing_): mixing(mixing_) {}

    damp_t(const damp_t& other) = default;
    damp_t(damp_t&& other) = default;
    damp_t& operator=(const damp_t& other) = default;
    damp_t& operator=(damp_t&& other) = default;

    ~damp_t(){}

    template<nda::MemoryArray Array_H_t>
    double solve(Array_H_t &&H, std::string dataset, h5::group &grp, long iter) {
      utils::check(grp.has_subgroup("iter" + std::to_string(iter-1)), "damp: h5 group /scf/iter{} does not exist.", iter-1);
      auto iter_grp = grp.open_group("iter" + std::to_string(iter-1));

      auto H_previous = nda::make_regular(H);
      nda::h5_read(iter_grp, dataset, H_previous);

      H = mixing*H + (1.0-mixing)*H_previous;

      H_previous -= H;
      auto max_iter = max_element(H_previous.data(), H_previous.data()+H_previous.size(),
                              [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      return std::abs((*max_iter));
    }

    // This version is compatible with DIIS
    template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_5D_t>
    std::array<double, 2> solve(Array_4D_t &&F, std::string dataset_F, Array_5D_t &&Sigma, std::string dataset_Sigma,
                 h5::group &grp, long iter)  {
        return std::array<double, 2>{
            solve(F, dataset_F, grp, iter),
            solve(Sigma, dataset_Sigma, grp, iter)
        };
    }

    void metadata_log() const {
      app_log(2, "\nIterative algorithm for SCF");
      app_log(2, "-----------------------------");
      app_log(2, "  * algorithm: simple mixing");
      app_log(2, "  * mixing parameter = {}\n", mixing);
    }

    // Dummy function to comply with the interface
    template<class... Args>
    void initialize([[maybe_unused]] Args&&... args) {}

  public:
    double mixing = 0.7;
    bool initialized = true;

  };
} // iter_scf

#endif //COQUI_DAMP_T_HPP
