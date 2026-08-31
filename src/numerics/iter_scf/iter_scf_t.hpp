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


#ifndef COQUI_ITER_SCF_T_HPP
#define COQUI_ITER_SCF_T_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "numerics/iter_scf/damp/damp_t.hpp"
#include "numerics/iter_scf/diis/diis_t.hpp"

#include "numerics/iter_scf/iter_scf_type_e.hpp"

namespace iter_scf {
  /**
   * Interface for iterative self-consistent algorithms, e.g. damping, DIIS, etc
   */
  class iter_scf_t {
  public:
    iter_scf_t(std::string alg) {
      if (string_to_alg_enum(alg) == iter_alg_e::damping) {
        _alg_var = damp_t();
      } else if (string_to_alg_enum(alg) == iter_alg_e::DIIS) {
        _alg_var = diis_t();
      } else {
        APP_ABORT(" iter_scf_t: Invalid type of iterative algorithm");
      }
    }

    explicit iter_scf_t(const damp_t &damp): _alg_var(damp) {}
    explicit iter_scf_t(damp_t&& damp): _alg_var(std::move(damp)) {}
    iter_scf_t& operator=(const damp_t& damp) { _alg_var = damp; return *this; }
    iter_scf_t& operator=(damp_t&& damp) { _alg_var = std::move(damp); return *this; }

    explicit iter_scf_t(const diis_t &diis): _alg_var(diis) {}
    explicit iter_scf_t(diis_t&& diis): _alg_var(std::move(diis)) {}
    iter_scf_t& operator=(const diis_t& diis) { _alg_var = diis; return *this; }
    iter_scf_t& operator=(diis_t&& diis) { _alg_var = std::move(diis); return *this; }

    template<class... Args>
    auto solve(Args&&... args) {
      return std::visit( [&](auto&& v) { return v.solve(std::forward<Args>(args)...); }, _alg_var);
    }

    // Needed to prepare complicated algorithms, such as DIIS
    template<class... Args>
    void initialize(Args&&... args) {
      return std::visit( [&](auto&& v) { return v.initialize(std::forward<Args>(args)...); }, _alg_var);
    }

    bool is_initialized() const {
      return std::visit( [&](auto&& v) { return v.initialized; }, _alg_var);
    }

    void metadata_log() const {
      std::visit( [&](auto&& v) { v.metadata_log(); }, _alg_var);
    }

    // return type of iterative solver
    iter_alg_e iter_alg() const
    { return std::visit( [&](auto&& v) { return v.get_iter_alg(); }, _alg_var); }

    ~iter_scf_t() {};
  private:
    std::variant<damp_t,diis_t> _alg_var;

  };

} // iter_scf

#endif //COQUI_ITER_SCF_T_HPP
