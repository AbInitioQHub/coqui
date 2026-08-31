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


#ifndef COQUI_AC_T_HPP
#define COQUI_AC_T_HPP

#include "nda/nda.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/ac/ac_type_e.hpp"
#include "numerics/ac/pade/pade_driver.hpp"
#include "ac_context.h"

namespace analyt_cont {
  // TODO Clean up evaluate()
  /**
   * Interface for analytical continuation of Green's functions and self-energies
   * Usage:
   */
   class AC_t {

   public:
     AC_t(std::string ac_type = "pade") {
       if (string_to_ac_enum(ac_type) == ac_type_e::pade) {
         _ac_var = pade_driver();
       } else {
         APP_ABORT(" AC_t: Invalid type of analytical continuation. \n");
       }
     }

     explicit AC_t(const pade_driver& pade): _ac_var(pade) {}
     explicit AC_t(pade_driver&& pade): _ac_var(std::move(pade)) {}
     AC_t& operator=(const pade_driver& pade) { _ac_var = pade; return *this; }
     AC_t& operator=(pade_driver&& pade) { _ac_var = std::move(pade); return *this; }

     ~AC_t() {};

     static nda::array<ComplexType, 1> w_grid(double w_min, double w_max, long Nw, double eta) {
       int i = 0;
       nda::array<ComplexType, 1> w_grid(Nw);
       std::transform(w_grid.begin(), w_grid.end(), w_grid.begin(),
                      [&](const ComplexType & ) {return w_min + (i++)*(w_max - w_min)/(Nw - 1) + eta*1i;} );
       return w_grid;
     }

    template<class... Args>
    void init(Args&&... args) {
      std::visit( [&](auto&& v) { v.init(std::forward<Args>(args)...); }, _ac_var);
    }

    template<nda::MemoryArrayOfRank<1> mesh_w_t, nda::MemoryArray Array_w_t>
    void evaluate(mesh_w_t &&w_mesh, Array_w_t &&A_w) {
      std::visit( [&](auto&& v) { v.evaluate(w_mesh, A_w); }, _ac_var);
    }

    template<nda::MemoryArray Array_w_t>
    void evaluate(ComplexType w, Array_w_t &&A) {
      std::visit( [&](auto&& v) { v.evaluate(w, A); }, _ac_var);
    }

    template<nda::MemoryArrayOfRank<1> mesh_w_t, nda::MemoryArrayOfRank<1> Array_w_1D_t>
    void evaluate(mesh_w_t &&w_mesh, Array_w_1D_t &&A_w, long idx) {
      std::visit( [&](auto&& v) { v.evaluate(w_mesh, A_w, idx); }, _ac_var);
    }

    ComplexType evaluate(ComplexType w, long idx) {
      return std::visit( [&](auto&& v) { return v.evaluate(w, idx); }, _ac_var);
    }

    template<class... Args>
    void iw_to_w(Args&&... args) {
      std::visit( [&](auto&& v) { v.iw_to_w(std::forward<Args>(args)...); }, _ac_var);
    }

   private:
   std::variant<pade_driver> _ac_var;
   };
} // analy_cont

#endif //COQUI_AC_T_HPP
