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
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/AppAbort.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "numerics/sparse/csr_blas.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/HF/thc_solver_comm.hpp"
#include "methods/HF/hf_t.h"
#include "methods/HF/thc_hf.icc"

namespace methods {
  namespace solvers {
    template<nda::MemoryArray AF_t>
    void hf_t::evaluate(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij, 
                        THC_ERI auto &&thc, const nda::MemoryArrayOfRank<4> auto &S_skij,
                        bool hartree, bool exchange) {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20thc-hf
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐ ┬ ┬┌─┐\n"
                 "║  ║ ║║═╬╗║ ║║   │ ├─┤│───├─┤├┤ \n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ ┴ ┴└─┘ ┴ ┴└  \n");
      app_log(1, "  Hartree, Exchange             = {}, {}\n"
                 "  Number of spins               = {}\n"
                 "  Number of polarizations       = {}\n"
                 "  Number of bands               = {}\n"
                 "  Number of THC auxiliary basis = {}\n"
                 "  K-points                      = {} total, {} in the IBZ\n"
                 "  Divergent treatment at q->0   = {}\n",
                 hartree, exchange, thc.MF()->nspin(), thc.MF()->npol(),
                 thc.MF()->nbnd(), thc.Np(), thc.MF()->nkpts(), thc.MF()->nkpts_ibz(),
                 _div_treatment);

      for( auto& v: {"TOTAL", "ALLOC",
                     "PRIM_TO_AUX", "AUX_TO_PRIM",
                     "COULOMB", "EXCHANGE"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
      if (thc.thc_X_type() == "q_dep") {
        APP_ABORT("hf_t: HF with q-dependent Inpts is not implemented!");
      } else if (thc.thc_X_type() == "q_indep") {
        if(thc.MF()->nqpts_ibz() < thc.MF()->nqpts())
          thc_hf_Xqindep_wsymm(Dm_skij, sF_skij, thc, S_skij, hartree, exchange);
        else
          thc_hf_Xqindep(Dm_skij, sF_skij, thc, S_skij, hartree, exchange);
      } else {
        APP_ABORT("hf_t::evaluate: Invalid thc_X_type.\n");
      }
      _Timer.stop("TOTAL");
      print_thc_hf_timers();
      thc.print_timers();
    }

    // instantiate templates
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>;
    using Arrv4D2 = nda::array_view<ComplexType, 4, nda::C_layout>;
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arr4D const&, thc_reader_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arr4D> &,Arrv4D2 const&, thc_reader_t&, Arrv4D2 const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arr4D const&, thc_reader_t&, Arr4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arr4D const&, thc_reader_t&, Arrv4D const&, bool, bool);
    template void hf_t::evaluate(sArray_t<Arrv4D> &,Arrv4D const&, thc_reader_t&, Arrv4D const&, bool, bool);

  } // solvers
} // methods
