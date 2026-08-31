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


#ifndef METHODS_TESTS_TEST_COMMON_HPP
#define METHODS_TESTS_TEST_COMMON_HPP

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "numerics/shared_array/nda.hpp"

namespace methods {
  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;

  namespace mpi3 = boost::mpi3;

  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  template<int N>
  using shape_t = std::array<long,N>;
} // methods

#endif // METHODS_TESTS_TEST_COMMON_HPP
