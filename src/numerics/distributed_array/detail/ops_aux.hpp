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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_OPS_AUX_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_OPS_AUX_HPP

/*
 * Auxiliary functions 
 */ 

#include <utility>
#include <type_traits>
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "numerics/detail/ops_aux.hpp"
#include "numerics/distributed_array/detail/concepts.hpp"

namespace math::detail
{

using math::nda::DistributedArray;

//DistributedArray
template<DistributedArray MA>
struct op_tag<detail::normal_tag<MA>> : std::integral_constant<char, 'N'>  {};
template<DistributedArray MA>
struct op_tag<detail::transpose_tag<MA>> : std::integral_constant<char, 'T'>  {};
template<DistributedArray MA>
struct op_tag<detail::conjugate_transpose_tag<MA>> : std::integral_constant<char, 'C'>  {};

// DistributedArray
template<DistributedArray MA>
MA&& arg(detail::normal_tag<MA> const& nt)
{ return nt.arg1; }
template<DistributedArray MA>
MA&& arg(detail::transpose_tag<MA> const& t)
{ return t.arg1; }
template<DistributedArray MA>
MA&& arg(detail::conjugate_transpose_tag<MA> const& ht)
{ return ht.arg1; }
template<DistributedArray MA>
MA&& arg(detail::normal_tag<MA> & nt)
{ return nt.arg1; }
template<DistributedArray MA>
MA&& arg(detail::transpose_tag<MA> & t)
{ return t.arg1; }
template<DistributedArray MA>
MA&& arg(detail::conjugate_transpose_tag<MA> & ht)
{ return ht.arg1; }

template<DistributedArray MA> 
auto local_with_tags(normal_tag<MA> const& nt)  { return nt.arg1.local(); }
template<DistributedArray MA> 
auto local_with_tags(transpose_tag<MA> const& t)  { return ::nda::transpose(t.arg1.local()); }
template<DistributedArray MA> 
//auto local_with_tags(conjugate_transpose_tag<MA> const& ht)  { return ::nda::transpose(ht.arg1.local()); }
auto local_with_tags(conjugate_transpose_tag<MA> const& ht)  { return ::nda::dagger(ht.arg1.local()); }
template<DistributedArray MA> 
auto local_with_tags(normal_tag<MA>& nt)  { return nt.arg1.local(); }
template<DistributedArray MA> 
auto local_with_tags(transpose_tag<MA>& t)  { return ::nda::transpose(t.arg1.local()); }
template<DistributedArray MA> 
//auto local_with_tags(conjugate_transpose_tag<MA>& ht)  { return ::nda::transpose(ht.arg1.local()); }
auto local_with_tags(conjugate_transpose_tag<MA>& ht)  { return ::nda::dagger(ht.arg1.local()); }

template<bool trans, bool conjg, DistributedArray MA>
decltype(auto) add_tags(MA&& A) { 
  if constexpr (trans) 
    return transpose_tag<MA>(std::forward<MA>(A));
  else if constexpr (conjg) 
    return conjugate_transpose_tag<MA>(std::forward<MA>(A));
  else 
    return A;
}

} // math::detail

#endif
