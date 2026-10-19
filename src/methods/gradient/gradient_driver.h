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


#ifndef METHODS_GRADIENT_GRADIENT_DRIVER_H
#define METHODS_GRADIENT_GRADIENT_DRIVER_H

#include <string>

#include "methods/mb_state/mb_state.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace methods
{

template<typename dyson_type, typename eri_t, typename corr_solver_t>
void evaluate_gradient(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri_t,
                       const imag_axes_ft::IAFT &FT, solvers::mb_solver_t<corr_solver_t> mb_solver,
                       const std::string &solver_type, const std::string &input_grp, int input_iter);

} // namespace methods

#endif
