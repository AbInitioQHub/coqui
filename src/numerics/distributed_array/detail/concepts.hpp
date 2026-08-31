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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_DETAIL_CONCEPTS_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_DETAIL_CONCEPTS_HPP

#include <concepts>
#include <type_traits>

namespace math::nda
{

template <typename A>
using get_value_t = typename std::decay_t<A>::value_type;

/*
 * Some concepts
 */
template <typename A>
concept DistributedArray = requires(A const& a) {
  { ::nda::MemoryArray<typename std::decay_t<A>::Array_t> };
  { std::decay_t<A>::rank > 0 };
  { std::is_scalar<get_value_t<A>>::value };
  { a.communicator() };
  { a.grid() };
  { a.local() };
  { a.local_shape() } -> ::nda::StdArrayOfLong;
  { a.global_shape() } -> ::nda::StdArrayOfLong;
  { a.block_size() } -> ::nda::StdArrayOfLong;
  { a.origin() } -> ::nda::StdArrayOfLong;
};

template <typename A>
concept DistributedArrayView = DistributedArray<A> and requires(A const& a) {
  { ::nda::MemoryArray<typename std::decay_t<A>::Array_view_t> };
  { std::decay_t<A>::is_view == true };
};

template <typename A>
constexpr int get_rank = std::tuple_size<std::decay_t<decltype(std::declval<A const>().global_shape())>>::value;

template <typename A, int R>
concept DistributedArrayOfRank = DistributedArray<A> and ( ::math::nda::get_rank<A> == R );

template <typename A>
concept DistributedMatrix = DistributedArrayOfRank<A,2>; 

// Slate Matrix specific
template <typename A>
concept SlateMatrix = DistributedArrayOfRank<A,2> and requires(A const& a) {
  { a.mb() }; 
  { a.nb() }; 
};

template <typename A>
concept SlateMatrixView = SlateMatrix<A> and requires(A const& a) {
  { std::decay_t<A>::is_view == true };
};

}

#endif
