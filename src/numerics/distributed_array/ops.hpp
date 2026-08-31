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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_OPS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_OPS_HPP

/*
 * Utilities for use of SLATE with math::nda::distributed_matrix
 */

#include <functional>

#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/distributed_array/detail/ops_aux.hpp"

namespace math::nda
{

/***************************************************************************/
/*                              blas/lapack tags                           */
/***************************************************************************/

//DistributedMatrix
template<DistributedArray MA>
auto normal(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto transpose(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto dagger(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto N(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto T(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<DistributedArray MA>
auto H(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

// put custom (non slate) distributed operations here, e.g. custom matrix multiplication

} // math::nda

#endif

