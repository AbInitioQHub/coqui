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
#include "numerics/nda_functions.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/HF/hf_t.h"
#include "methods/HF/cholesky_hf.icc"

namespace methods {
  namespace solvers {
    template<nda::MemoryArray AF_t>
    void hf_t::evaluate(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij, 
                        Cholesky_ERI auto &&chol, const nda::MemoryArrayOfRank<4> auto &S_skij,
                        bool hartree, bool exchange) {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-hf
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┬ ┬┌─┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───├─┤├┤ \n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ ┴ ┴└  \n");
      app_log(1, "  Hartree, Exchange = {}, {}\n"
                 "  nbnd  = {}\n"
                 "  Auxiliary basis  = {}\n"
                 "  nkpts = {}\n"
                 "  nkptz_ibz = {}\n"
                 "  divergent treatment at q->0 = {}\n",
              hartree, exchange,
              chol.MF()->nbnd(), chol.Np(), chol.MF()->nkpts(), chol.MF()->nkpts_ibz(),
              _div_treatment);
      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(), "hf_t::cholesky_hf::evaluate: Symmetry not yet implemented.");

      for( auto& v: {"TOTAL", "ALLOC",
                     "COULOMB", "EXCHANGE", "CHOL", "CHOL J", "CHOL K", "J BLAS", "K BLAS"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
      sF_skij.set_zero();

      _Timer.start("COULOMB");
      if (hartree) add_J(sF_skij, Dm_skij, chol);
      _Timer.stop("COULOMB");

      _Timer.start("EXCHANGE");
      if (exchange) add_K(sF_skij, Dm_skij, chol, S_skij);
      _Timer.stop("EXCHANGE");

      _Timer.stop("TOTAL");

      print_chol_hf_timers();
      chol.print_timers();
    }

    // instantiate templates
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    using Arrv4D2 = nda::array_view<ComplexType, 4, nda::C_layout>;
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arr4D const&, chol_reader_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arrv4D2 const&, chol_reader_t&, Arrv4D2 const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arr4D const&, chol_reader_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arrv4D const&, chol_reader_t&, Arrv4D const&, bool, bool);

  } // solvers
} // methods
