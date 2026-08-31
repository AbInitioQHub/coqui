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


#include "methods/ERI/chol_reader_t.hpp"
#include "methods/GW/g0_div_utils.hpp"
#include "scr_coulomb_t.h"

namespace methods {
namespace solvers {

  void scr_coulomb_t::update_w(MBState &mb_state, Cholesky_ERI auto &chol, long h5_iter) {
    (void) mb_state; (void) chol; (void) h5_iter;
    app_log(2, "\nscr_coulomb_t::update_w: the new screen interaction interface is not implemented for Choleskky-ERI. "
               "CoQui will do nothing here, and instead the screened interactions will be computed using the gw solver.\n");
  }


  // template instantiations
  template void scr_coulomb_t::update_w(MBState&, chol_reader_t&, long);

}  // solvers
}  // methods
