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


//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef SYMMETRY_TOOLS_CUDA_KERNELS_HPP
#define SYMMETRY_TOOLS_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"

namespace kernels::device
{

template<typename V1> 
void transform_k2g(bool trev, nda::stack_array<double, 3, 3> const& Rinv, 
                   nda::stack_array<double, 3> const& Gs, 
                   nda::stack_array<long, 3> const& mesh,
                   V1 &&k2g);


} // namespace kernels::device

#endif
