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


#ifndef UTILITIES_CONCEPTS_HPP
#define UTILITIES_CONCEPTS_HPP

#include <concepts>

namespace utils
{

/*
 * Some concepts
 */
template <typename A, typename T=double, typename I=long>
concept Communicator = requires(A& a, T* ptr, I ) {
  { a.rank() };
  { a.size() };
  { a.split() };
/*
  { a.split_shared() };
  { a.template broadcast_n<T,I>() };
  { a.template send_n<T,I>() };
  { a.template receive_n<T,I>() };
  { a.template gather_n<T,I,T>() };
*/
/// and many more...
};

}

#endif
