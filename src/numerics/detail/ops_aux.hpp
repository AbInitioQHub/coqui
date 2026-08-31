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


#ifndef NUMERICS_DETAILS_OPS_AUX_HPP
#define NUMERICS_DETAILS_OPS_AUX_HPP

/*
 * Auxiliary functions 
 */ 

#include <utility>
#include <type_traits>
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace math::detail
{

/***************************************************************************/
/*  				Utils	  				   */
// MAM: not sure if this is the best way!
/***************************************************************************/

template<typename MA>
struct normal_tag
{
  MA arg1;
  using Array_t			   = typename std::decay_t<MA>::Array_t;
  using value_type                 = typename std::decay_t<MA>::value_type;
  static constexpr bool is_stride_order_Fortran() noexcept
    { return Array_t::layout_t::is_stride_order_Fortran(); }
  static constexpr bool is_stride_order_C() noexcept
    { return Array_t::layout_t::is_stride_order_C(); }
  normal_tag(MA m) : arg1(m) {}
  normal_tag()    		   = delete;
  normal_tag(normal_tag const&)    = delete;
  normal_tag(normal_tag&&)         = default;
  static const char tag            = 'N';
};

template<typename MA>
struct transpose_tag
{
  MA arg1;
  using Array_t			   = typename std::decay_t<MA>::Array_t;
  using value_type                 = typename std::decay_t<MA>::value_type;
  static constexpr bool is_stride_order_Fortran() noexcept
    { return Array_t::layout_t::is_stride_order_Fortran(); }
  static constexpr bool is_stride_order_C() noexcept
    { return Array_t::layout_t::is_stride_order_C(); }
  transpose_tag(MA m) : arg1(m) {}
  transpose_tag()                     = delete;
  transpose_tag(transpose_tag const&)    = delete;
  transpose_tag(transpose_tag&&)         = default;
  static const char tag            = 'T';
};

template<typename MA>
struct conjugate_transpose_tag
{
  MA arg1;
  using Array_t			   = typename std::decay_t<MA>::Array_t;
  using value_type                 = typename std::decay_t<MA>::value_type;
  static constexpr bool is_stride_order_Fortran() noexcept
    { return Array_t::layout_t::is_stride_order_Fortran(); }
  static constexpr bool is_stride_order_C() noexcept
    { return Array_t::layout_t::is_stride_order_C(); }
  conjugate_transpose_tag(MA m) : arg1(m) {}
  conjugate_transpose_tag()                     = delete;
  conjugate_transpose_tag(conjugate_transpose_tag const&)    = delete;
  conjugate_transpose_tag(conjugate_transpose_tag&&)         = default;
  static const char tag            = 'C';
};

// extract tag
template<class MA>  
struct op_tag : std::integral_constant<char, 'N'>  {}; 

// extract matrix
template<class MA>
MA arg(MA&& ma) { return std::forward<MA>(ma); } 

// apply tags and return local array
template<class MA>
auto local_with_tags(MA&& A) { return std::forward<MA>(A).local(); }

template<typename M>
inline constexpr bool is_tagged_matrix = false;  
template<typename A>
inline constexpr bool is_tagged_matrix<normal_tag<A>> = true;
template<typename A>
inline constexpr bool is_tagged_matrix<transpose_tag<A>> = true;
template<typename A>
inline constexpr bool is_tagged_matrix<conjugate_transpose_tag<A>> = true;

template<typename M>
inline constexpr bool is_conjugate_transpose = false;  
template<typename A>
inline constexpr bool is_conjugate_transpose<conjugate_transpose_tag<A>> = true;

template<typename M>
inline constexpr bool is_transpose = false;  
template<typename A>
inline constexpr bool is_transpose<transpose_tag<A>> = true;

} // math::detail

#endif
