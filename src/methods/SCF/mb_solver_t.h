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


#ifndef COQUI_MB_SOLVER_T_H
#define COQUI_MB_SOLVER_T_H

#include "methods/HF/hf_t.h"
#include "methods/embedding/projector_boson_t.h"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/GW/gw_t.h"
#include "methods/GF2/gf2_t.h"

namespace methods::solvers {

template<typename corr_solver_t = gw_t>
struct mb_solver_t {
  hf_t *hf;
  corr_solver_t *corr = nullptr;
  scr_coulomb_t *scr_eri = nullptr;

  mb_solver_t(hf_t *hf_) : hf(hf_) {}
  mb_solver_t(hf_t *hf_, corr_solver_t *corr_) : 
                           hf(hf_), corr(corr_) {}
  mb_solver_t(hf_t *hf_, corr_solver_t *corr_, scr_coulomb_t *scr_eri_) : 
                           hf(hf_), corr(corr_), scr_eri(scr_eri_) {}

  ~mb_solver_t() = default;
};

} // methods::solvers

#endif //COQUI_MB_SOLVER_T_H
