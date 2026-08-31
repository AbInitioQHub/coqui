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


#ifndef NUMERICS_SPARSE_DETAILS_OPS_AUX_HPP
#define NUMERICS_SPARSE_DETAILS_OPS_AUX_HPP

/*
 * Auxiliary functions 
 */ 

#include <utility>
#include <type_traits>
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "numerics/detail/ops_aux.hpp"
#include "numerics/sparse/detail/concepts.hpp"

namespace math::detail
{

using math::sparse::CSRMatrix;

// CSRMatrix
template<CSRMatrix MA>
struct op_tag<normal_tag<MA>> : std::integral_constant<char, 'N'>  {};
template<CSRMatrix MA>
struct op_tag<transpose_tag<MA>> : std::integral_constant<char, 'T'>  {};
template<CSRMatrix MA>
struct op_tag<conjugate_transpose_tag<MA>> : std::integral_constant<char, 'C'>  {};

// CSRMatrix 
template<CSRMatrix MA>
MA&& arg(normal_tag<MA> const& nt)
{ return nt.arg1; }
template<CSRMatrix MA>
MA&& arg(transpose_tag<MA> const& t)
{ return t.arg1; }
template<CSRMatrix MA>
MA&& arg(conjugate_transpose_tag<MA> const& ht)
{ return ht.arg1; }
template<CSRMatrix MA>
MA&& arg(normal_tag<MA> & nt)
{ return nt.arg1; }
template<CSRMatrix MA>
MA&& arg(transpose_tag<MA> & t)
{ return t.arg1; }
template<CSRMatrix MA>
MA&& arg(conjugate_transpose_tag<MA> & ht)
{ return ht.arg1; }

} // math::sparse::detail

#endif
