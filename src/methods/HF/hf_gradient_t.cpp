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


#include "hf_gradient_t.h"

namespace methods {

  namespace solvers {

    namespace mpi3 = boost::mpi3;

    hf_gradient_t::hf_gradient_t(std::shared_ptr<mf::MF> MF):
      _mf(MF),
      _Timer(),
      _natoms(_mf->number_of_atoms()),
      _nbnd(_mf->nbnd()),
      _nbnd_aux(_mf->nbnd_aux()),
      _nspin(_mf->nspin()),
      _nkpts(_mf->nkpts()),
      _npol(_mf->npol()),
      _k_weight(_mf->k_weight())
    {
    }

    void hf_gradient_t::print_chol_hf_grad_timers()
    {
      app_log(2, "\n  CHOL-HF-GRAD timers");
      app_log(2, "  --------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    Coulomb:               {0:.3f} sec", _Timer.elapsed("COULOMB"));
      app_log(2, "    Exchange:              {0:.3f} sec", _Timer.elapsed("EXCHANGE"));
    }

 } // namespace solvers

} // namespace method
