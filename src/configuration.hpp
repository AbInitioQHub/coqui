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


#ifndef COQUI_TOP_CONFIGURATION_HPP
#define COQUI_TOP_CONFIGURATION_HPP

#include<complex>
#include "config.h"

#include "nda/nda.hpp"

template<typename T>
T* raw_pointer_cast(T* p) { return p; }

using RealType = double;
using SPRealType = float;
using ComplexType = std::complex<RealType>;
using SPComplexType = std::complex<RealType>;

enum MEMORY_SPACE { HOST_MEMORY, DEVICE_MEMORY, UNIFIED_MEMORY, DEFAULT_MEMORY };

#if defined(ENABLE_UNIFIED_MEMORY)
// compile time check for ENABLE_DEVICE done in cmake!
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = UNIFIED_MEMORY;
#elif defined(ENABLE_DEVICE)
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = DEVICE_MEMORY;
#else
static constexpr  MEMORY_SPACE DEFAULT_MEMORY_SPACE = HOST_MEMORY;
#endif

inline static constexpr nda::mem::AddressSpace to_nda_address_space(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return nda::mem::Host; 
  else if(m == DEVICE_MEMORY)
    return nda::mem::Device;  
  else if(m == UNIFIED_MEMORY)
    return nda::mem::Unified;  
  else if(m == DEFAULT_MEMORY)
#if defined(ENABLE_UNIFIED_MEMORY)
    return nda::mem::Unified;
#elif defined(ENABLE_DEVICE)
    return nda::mem::Device;  
#else
    return nda::mem::Host; 
#endif
  return nda::mem::None; 
}

inline auto memory_space_to_string(MEMORY_SPACE m)
{
  if(m == HOST_MEMORY)
    return std::string("host");
  else if(m == DEVICE_MEMORY)
    return std::string("device");
  else if(m == UNIFIED_MEMORY)
    return std::string("unified");
  else
    return std::string("unknown"); 
  return std::string("host");
}

namespace memory 
{

template<nda::Array a_t>
constexpr MEMORY_SPACE get_memory_space()
{
  static_assert(nda::mem::on_host<a_t> or nda::mem::on_device<a_t> or nda::mem::on_unified<a_t>, "Unknown memory space");
  if constexpr (nda::mem::on_host<a_t>)
    return HOST_MEMORY;
  else if constexpr (nda::mem::on_device<a_t>)
    return DEVICE_MEMORY;
  else if constexpr (nda::mem::on_unified<a_t>)
    return UNIFIED_MEMORY;
  return HOST_MEMORY; 
}

template<typename T, int N, typename Layout = nda::C_layout>
using host_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using host_array_view = nda::array_view<T,N,Layout>;

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::cuarray<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::cuarray_view<T,N,Layout>;
#else
template<typename T, int N, typename Layout = nda::C_layout>
using device_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using device_array_view = nda::array_view<T,N,Layout>;
#endif

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::basic_array<T, N, Layout, 'A', nda::heap<nda::mem::Unified>>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::basic_array_view<T, N, Layout, 'A', nda::default_accessor, nda::borrowed<nda::mem::Unified>>; 
#else
template<typename T, int N, typename Layout = nda::C_layout>
using unified_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using unified_array_view = nda::array_view<T,N,Layout>;
#endif

#if defined(ENABLE_DEVICE)
template<typename T, int N, typename Layout = nda::C_layout>
using default_array = nda::cuarray<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using default_array_view = nda::cuarray_view<T,N,Layout>;
#else
template<typename T, int N, typename Layout = nda::C_layout>
using default_array = nda::array<T,N,Layout>;
template<typename T, int N, typename Layout = nda::C_stride_layout>
using default_array_view = nda::array_view<T,N,Layout>;
#endif

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_layout>
using array = std::conditional_t<MEM==HOST_MEMORY, host_array<T,N,Layout>,
              std::conditional_t<MEM==DEVICE_MEMORY, device_array<T,N,Layout>,
              std::conditional_t<MEM==UNIFIED_MEMORY, unified_array<T,N,Layout>,
						        default_array<T,N,Layout>>>>;

template<MEMORY_SPACE MEM, typename T, int N, typename Layout = nda::C_stride_layout>
using array_view = std::conditional_t<MEM==HOST_MEMORY, host_array_view<T,N,Layout>,
                   std::conditional_t<MEM==DEVICE_MEMORY, device_array_view<T,N,Layout>,
                   std::conditional_t<MEM==UNIFIED_MEMORY, unified_array_view<T,N,Layout>,
                                                           default_array_view<T,N,Layout>>>>;

template<MEMORY_SPACE MEM>
auto to_memory_space(auto &&A)
{
  if constexpr (MEM==HOST_MEMORY) {
    return nda::to_host(A);
  } else if constexpr (MEM==DEVICE_MEMORY) {
    return nda::to_device(A);
  } else if constexpr (MEM==UNIFIED_MEMORY) {
    return nda::to_unified(A);
  } else {
    return to_memory_space<DEFAULT_MEMORY_SPACE>(A); 
  }
}

}

#endif
