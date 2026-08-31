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

#ifndef ARGMAX_KERNELS_HPP
#define ARGMAX_KERNELS_HPP

#include <complex>
#include <tuple>
#include "numerics/device_kernels/cuda/cuda_aux.hpp"

namespace kernels::device
{

template<typename T>
std::tuple<long,T> argmax(T const*, long N);
template<typename T>
std::tuple<long,T> argmin(T const*, long N);

} // namespace kernels::device

#endif
