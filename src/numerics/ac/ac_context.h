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


#ifndef COQUI_AC_CONTEXT_H
#define COQUI_AC_CONTEXT_H

#include "numerics/imag_axes_ft/iaft_enum_e.hpp"

namespace analyt_cont {

struct ac_context_t {
  std::string ac_alg = "pade";
  imag_axes_ft::stats_e stats = imag_axes_ft::fermion;
  int Nfit = -1;
  double eta = 0.0001;
  // params for real w mesh
  double w_min = -10.0;
  double w_max = 10.0;
  long Nw = 5000;
};

} // analyt_cont

#endif //COQUI_AC_CONTEXT_H
