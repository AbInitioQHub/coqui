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


#ifndef COQUI_PADE_DRIVER_HPP
#define COQUI_PADE_DRIVER_HPP

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "numerics/ac/pade/pade_t.hpp"

namespace analyt_cont {
  struct pade_driver {
  public:
    explicit pade_driver(pade_impl_e impl = pade_impl_e::original): _pade_kernel(impl) {}

    pade_driver(const pade_driver& other) = default;
    pade_driver(pade_driver&& other) = default;

    pade_driver& operator=(const pade_driver& other) = default;
    pade_driver& operator=(pade_driver&& other) = default;

    ~pade_driver(){}

    // FIXME We should not assume that iw_mesh has negative frequencies in the first half and positive 
    //       frequencies in the second half. This excludes arbitrary input Matsubara data. 
    //       Extracting positive frequencies should be done before calling pade_driver::init.
    template< nda::ArrayOfRank<1> mesh_iw_t, nda::MemoryArray Array_iw_t>
    void init(mesh_iw_t &&iw_mesh, Array_iw_t &&A_iw, int Nfit=-1, bool is_iw_pos_only=false) {
      using Aiw_value_type = typename std::decay_t<Array_iw_t>::value_type;
      static_assert(nda::is_complex_v<Aiw_value_type>, "pade_driver::init: A_iw is not complex");

      long dim1 = std::accumulate(A_iw.shape().begin()+1, A_iw.shape().end(), 1, std::multiplies<>{});
      long niw = A_iw.shape(0);
      if (!is_iw_pos_only) utils::check(niw % 2 == 0, "pade_driver::init: niw ({}) % 2 != 0", niw);

      long niw_pos = (is_iw_pos_only)? niw : niw/2;
      if (Nfit == -1) Nfit = niw_pos;
      if (Nfit > niw_pos) {
        app_log(4, "[WARNING]: Pade: Nfit ({}) is larger than the number of positive imaginary frequency points ({}). "
                   "Setting Nfit to {}.\n", Nfit, niw_pos, niw_pos);
        Nfit = niw_pos;
      }

      // determine the imaginary frequencies for fitting
      auto Aiw_2D = nda::reshape(A_iw, std::array<long, 2>{niw, dim1});
      nda::array<ComplexType, 1> iw_fit(Nfit);
      nda::array<ComplexType, 2> A_fit(Nfit, dim1);
      long shift = (is_iw_pos_only)? 0 : niw/2;
      long num_lchunk = niw_pos % Nfit;
      if (num_lchunk != 0) {
        long step = niw_pos/Nfit + 1;
        auto low_iw_rng = nda::range(0, step*num_lchunk, step) + shift;
        auto high_iw_rng = nda::range(step*num_lchunk, niw_pos, step-1) + shift;
        utils::check(low_iw_rng.size()+high_iw_rng.size() == Nfit, "pade_driver::init: iw_rng.size() != Nfit");

        iw_fit(nda::range(0,low_iw_rng.size())) = iw_mesh(low_iw_rng);
        iw_fit(nda::range(low_iw_rng.size(),Nfit)) = iw_mesh(high_iw_rng);
        A_fit(nda::range(0,low_iw_rng.size()), nda::range::all) = Aiw_2D(low_iw_rng, nda::range::all);
        A_fit(nda::range(low_iw_rng.size(),Nfit), nda::range::all) = Aiw_2D(high_iw_rng, nda::range::all);
      } else {
        long step = niw_pos/Nfit;
        auto fit_iw_rng = nda::range(0, niw_pos, step) + shift;
        utils::check(fit_iw_rng.size() == Nfit, "pade_driver::init: iw_rng.size() != Nfit");
        iw_fit(nda::range::all) = iw_mesh(fit_iw_rng);
        A_fit(nda::ellipsis{}) = Aiw_2D(fit_iw_rng, nda::range::all);
      }

      app_log(2, "Solving {}-point Pade interpolation using {} implementation.\n",
              Nfit, (_pade_kernel.implementation() == pade_impl_e::updated)? "updated" : "original");
      _pade_kernel.init(iw_fit, A_fit);
    }

    template<nda::MemoryArrayOfRank<1> mesh_w_t, nda::MemoryArray Array_w_t>
    void evaluate(mesh_w_t &&w_mesh, Array_w_t &&A_w) {
      using Aw_value_type = typename std::decay_t<Array_w_t>::value_type;
      static_assert(nda::is_complex_v<Aw_value_type>, "pade_driver::evaluate: A_w is not complex");
      utils::check(w_mesh.shape(0) == A_w.shape(0), "pade_driver::evaluate: nw is not consistent");

      long dim1 = std::accumulate(A_w.shape().begin()+1, A_w.shape().end(), 1, std::multiplies<>{});
      auto Aw_2D   = nda::reshape(A_w, std::array<long, 2>{A_w.shape(0), dim1});
      Aw_2D = _pade_kernel.evaluate(w_mesh);
    }

    template<nda::MemoryArray Array_w_t>
    void evaluate(ComplexType w, Array_w_t &&A) {
      using Aw_value_type = typename std::decay_t<Array_w_t>::value_type;
      static_assert(nda::is_complex_v<Aw_value_type>, "pade_driver::evaluate: A_w is not complex");

      long dim1 = std::accumulate(A.shape().begin(), A.shape().end(), 1, std::multiplies<>{});
      auto Aw_2D   = nda::reshape(A, std::array<long, 2>{1, dim1});
      nda::array<ComplexType, 1> w_mesh(1);
      w_mesh(0) = w;

      Aw_2D = _pade_kernel.evaluate(w_mesh);
    }

    template<nda::MemoryArrayOfRank<1> mesh_w_t, nda::MemoryArrayOfRank<1> Array_w_1D_t>
    void evaluate(mesh_w_t &&w_mesh, Array_w_1D_t &&A_w, long d1) {
      using Aw_value_type = typename std::decay_t<Array_w_1D_t>::value_type;
      static_assert(nda::is_complex_v<Aw_value_type>, "pade_driver::evaluate: A_w is not complex");

      A_w = _pade_kernel.evaluate(w_mesh, d1);
    }

    ComplexType evaluate(ComplexType w, long d1) {
      return _pade_kernel.evaluate(w, d1);
    }


    template<nda::MemoryArray Array_iw_t, nda::MemoryArrayOfRank<1> mesh_iw_t,
        nda::MemoryArray Array_w_t, nda::MemoryArrayOfRank<1> mesh_w_t>
    void iw_to_w(Array_iw_t &&A_iw, mesh_iw_t &&iw_mesh, Array_w_t &&A_w, mesh_w_t &&w_mesh,
                 bool is_iw_pos_only = false, int Nfit = -1) {
      static_assert(nda::get_rank<Array_iw_t> == nda::get_rank<Array_w_t>,
                    "pade_driver::iw_to_w: Inconsistent ranks of A_iw and A_w.");

      init(iw_mesh, A_iw, Nfit, is_iw_pos_only);
      evaluate(w_mesh, A_w);
      _pade_kernel.reset();
    }

    const pade_t<100>& get_kernel() const {return _pade_kernel; }

  private:
    pade_t<100> _pade_kernel;

  };
} // analyt_cont



#endif //COQUI_PADE_DRIVER_HPP
