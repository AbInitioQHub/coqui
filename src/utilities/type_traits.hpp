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


#ifndef UTILS_TYPE_TRAITS_HPP
#define UTILS_TYPE_TRAITS_HPP

#include <complex>

namespace utils
{

  template <typename T>
  struct remove_complex {typedef T type;};
  template <typename T>
  struct remove_complex<std::complex<T> > {typedef T type;};
    
  template<typename T>
  using remove_complex_t = typename remove_complex<T>::type;

  template <typename T>
  struct add_complex {typedef std::complex<T> type;};
  template <typename T>
  struct add_complex<std::complex<T> > {typedef std::complex<T> type;};
    
  template<typename T>
  using add_complex_t = typename add_complex<T>::type;

} 

#endif
