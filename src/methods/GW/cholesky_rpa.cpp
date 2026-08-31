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


#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "utilities/proc_grid_partition.hpp"
#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/GW/gw_t.h"
#include "methods/GW/cholesky_rpa.icc"


namespace methods {
  namespace solvers {

    double gw_t::rpa_energy(const nda::MemoryArrayOfRank<5> auto &G_tskij, Cholesky_ERI auto &chol) {
      using namespace math::shm;
      using Array_3D_t = nda::array_view<ComplexType, 3>;
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-rpa
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┬─┐┌─┐┌─┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───├┬┘├─┘├─┤\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ ┴└─┴  ┴ ┴\n");
      _ft->metadata_log();
      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(),
                   "gw_t::rpa_energy(Cholesky_ERI): Symmetry not yet implemented.");
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "chol-rpa:: we assume nt_f == nt_b at least for now \n"
                   "(will lift the restriction at some point...)");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "chol-rpa: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL", "ALLOC", "COMM",
                     "EVALUATE_P0", "EVALUATE_RPA",
                     "IMAG_FT", "ERI_READER"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");

      auto mpi = chol.mpi();

      _Timer.start("ALLOC");
      size_t nt_half = (_ft->nt_f()%2==0)? _ft->nt_f()/2 : _ft->nt_f()/2 + 1;
      size_t nw_half = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;

      sArray_t<Array_3D_t> sP0_tPQ(*mpi, {nt_half, chol.Np(), chol.Np()});
      sArray_t<Array_3D_t> sP0_wPQ(*mpi, {nw_half, chol.Np(), chol.Np()});
      _Timer.stop("ALLOC");

      ComplexType e_rpa = 0;

      // TODO these should be input parameters
      int Np_batch = sP0_tPQ.local().shape(1);
      for (size_t iq = 0; iq < chol.MF()->nkpts(); ++iq) {
        _Timer.start("EVALUATE_P0");
        evaluate_P0(iq, G_tskij, sP0_tPQ, chol, Np_batch, (iq==0)? true : false);
        _Timer.stop("EVALUATE_P0");

        _Timer.start("IMAG_FT");
        sP0_wPQ.win().fence();
        if (sP0_wPQ.node_comm()->root())
          _ft->tau_to_w_PHsym(sP0_tPQ.local(), sP0_wPQ.local());
        sP0_wPQ.win().fence();
        _Timer.stop("IMAG_FT");

        _Timer.start("EVALUATE_RPA");
        e_rpa += chol_rpa_energy_impl(sP0_wPQ);
        _Timer.stop("EVALUATE_RPA");
      }
      e_rpa /= (2.0 * chol.MF()->nkpts());
      if (e_rpa.imag()/e_rpa.real() >= 1e-10) {
        app_log(2, "Warning: e_rpa.imag()/e_rpa.real() = {}, e_rpa.imag() = {}, e_rpa.real() = {}",
                e_rpa.imag()/e_rpa.real(), e_rpa.imag(), e_rpa.real());
      }
      _Timer.stop("TOTAL");
      print_rpa_gw_timers();
      chol.print_timers();

      return e_rpa.real();
    }

    // instantiations
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;
    using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;
    template double gw_t::rpa_energy(const Arr &, chol_reader_t &); 
    template double gw_t::rpa_energy(const Arrv &, chol_reader_t &); 
    template double gw_t::rpa_energy(const Arrv2 &, chol_reader_t &); 

  } // solvers
} // methods
