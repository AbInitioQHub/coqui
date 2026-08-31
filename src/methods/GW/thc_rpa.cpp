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
#include "numerics/distributed_array/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/HF/thc_solver_comm.hpp"

#include "methods/ERI/thc_reader_t.hpp"
#include "methods/GW/gw_t.h"
#include "methods/GW/thc_rpa.icc"

namespace methods {
  namespace solvers {
    double gw_t::rpa_energy(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc) {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20thc-rpa
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐ ┬─┐┌─┐┌─┐\n"
                 "║  ║ ║║═╬╗║ ║║   │ ├─┤│───├┬┘├─┘├─┤\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ ┴ ┴└─┘ ┴└─┴  ┴ ┴\n");
      _ft->metadata_log();
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "chol-rpa: We assume nt_f == nt_b at least for now. \n"
                   "         And we assume tau sampling for fermions and bosons are the same. \n"
                   "(will lift the restriction at some point...)");
      {
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "thc-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL",
                     "PI_PRIM_TO_AUX",
                     "EVALUATE_PI_K", "PI_ALLOC_K", "PI_HADPROD_K",
                     "EVALUATE_PI_R", "PI_ALLOC_R", "PI_FT_R", "PI_HADPROD_R",
                     "EVALUATE_RPA", "RPA_ALLOC",
                     "IMAG_FT_TtoW", "FT_REDISTRIBUTE"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");
      double e_rpa = thc_rpa_Xqindep(G_tskij, thc);
      _Timer.stop("TOTAL");

      print_thc_rpa_timers();
      thc.print_timers();

      return e_rpa;
    }

    // instantiations
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;
    using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;
    template double gw_t::rpa_energy(const Arr &, thc_reader_t &);
    template double gw_t::rpa_energy(const Arrv &, thc_reader_t &);
    template double gw_t::rpa_energy(const Arrv2 &, thc_reader_t &);

  }
}

