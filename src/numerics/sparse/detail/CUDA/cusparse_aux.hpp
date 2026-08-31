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


#ifndef CUSPARSE_AUX_HPP
#define CUSPARSE_AUX_HPP

#include <type_traits>
#include <complex>

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

// Should only be here if ENABLE_CUDA
#include <cuda_runtime.h>
#include "cusparse.h"

#include "nda/nda.hpp"
#include "nda/macros.hpp"

#include "numerics/sparse/detail/concepts.hpp"
#include "numerics/sparse/detail/ops_aux.hpp"

namespace math::sparse::device
{

#define CUSPARSE_CHECK(X, ...)                                                                                                                       \
  {                                                                                                                                                  \
    auto err = X(__VA_ARGS__);                                                                                                                       \
    utils::check( err == CUSPARSE_STATUS_SUCCESS, std::string(AS_STRING(X)) + " failed with error code: " + std::to_string(err) + ", error message: " + std::string(cusparseGetErrorString(err)) + "\n");  \
  }

template <typename T>
auto cusparse_datatype = std::enable_if_t<sizeof(T *) == 0>{};
template <> inline auto cusparse_datatype<float> = CUDA_R_32F;
template <> inline auto cusparse_datatype<double> = CUDA_R_64F;
template <> inline auto cusparse_datatype<std::complex<float>> = CUDA_C_32F;
template <> inline auto cusparse_datatype<std::complex<double>> = CUDA_C_64F;

template <typename T> auto cusparse_indextype = std::enable_if_t<sizeof(T *) == 0>{};
template <> inline auto cusparse_indextype<int> = CUSPARSE_INDEX_32I;
template <> inline auto cusparse_indextype<long> = CUSPARSE_INDEX_64I; 

inline cusparseOrder_t get_cusparseOrder(const char* matdescra)
{
  if(matdescra[3]=='C') return CUSPARSE_ORDER_ROW;
  else if(matdescra[3]=='F') return CUSPARSE_ORDER_COL;
  else utils::check(false,"Invalid matdescra:{}",std::string(matdescra));
}

template <typename T>
uint32_t find_alignment(T *p) {
  if (uintptr_t(p) % uint32_t(256) == 0)
     return uint32_t(256);
  else if (uintptr_t(p) % uint32_t(128) == 0)
    return uint32_t(128);
  else if (uintptr_t(p) % uint32_t(64) == 0)
    return uint32_t(64);
  else if (uintptr_t(p) % uint32_t(32) == 0)
    return uint32_t(32);
  else if (uintptr_t(p) % uint32_t(16) == 0)
    return uint32_t(16);
  else if (uintptr_t(p) % uint32_t(8) == 0)
    return uint32_t(8);
  else if (uintptr_t(p) % uint32_t(4) == 0)
    return uint32_t(4);
  else if (uintptr_t(p) % uint32_t(2) == 0)
    return uint32_t(2);
  else
    return sizeof(T);
}

template<typename A_t>
requires(CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) 
constexpr auto get_operation() {
  constexpr char op_a = math::detail::op_tag<A_t>::value;
  using value_t = std::decay_t<typename std::decay_t<A_t>::value_type>;
  if constexpr (op_a == 'n' or op_a == 'N') 
    return CUSPARSE_OPERATION_NON_TRANSPOSE; 
  else if constexpr (op_a == 't' or op_a == 'T') 
    return CUSPARSE_OPERATION_TRANSPOSE;
  else if constexpr (op_a == 'c' or op_a == 'C' or op_a == 'h' or op_a == 'H') { 
    if constexpr (::nda::is_complex_v<value_t>) { 
      return CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; 
    } else {
      return CUSPARSE_OPERATION_TRANSPOSE;
    }
  }
}

template<::nda::MemoryArrayOfRank<1> X>
auto cuDn(X& x) {
  static_assert(::nda::mem::have_device_compatible_addr_space<X>, "Memory space mismatch");
  utils::check(x.indexmap().min_stride() == 1, "Stride mismatch");
  if constexpr (std::is_const_v<std::remove_pointer_t<decltype(x.data())>>) {
    cusparseConstDnVecDescr_t cuX;
    CUSPARSE_CHECK( cusparseCreateConstDnVec, &cuX, x.extent(0), x.data(), cusparse_datatype<::nda::get_value_t<X>>)
    return cuX; 
  } else {
    cusparseDnVecDescr_t cuX;
    CUSPARSE_CHECK( cusparseCreateDnVec, &cuX, x.extent(0), x.data(), cusparse_datatype<::nda::get_value_t<X>>)
    return cuX;
  }
}

template<::nda::MemoryArrayOfRank<2> X>
auto cuDn(X& x) {
  static_assert(::nda::mem::have_device_compatible_addr_space<X>, "Memory space mismatch");
  utils::check(x.indexmap().min_stride() == 1, "Stride mismatch");
  if constexpr (std::is_const_v<std::remove_pointer_t<decltype(x.data())>>) { 
    cusparseConstDnMatDescr_t cuX;
    if constexpr (std::decay_t<X>::is_stride_order_C()) {
      CUSPARSE_CHECK( cusparseCreateConstDnMat, &cuX, x.extent(0), x.extent(1), x.strides()[0], x.data(), cusparse_datatype<::nda::get_value_t<X>>,CUSPARSE_ORDER_ROW); 
    } else {
      CUSPARSE_CHECK( cusparseCreateConstDnMat, &cuX, x.extent(0), x.extent(1), x.strides()[1], x.data(), cusparse_datatype<::nda::get_value_t<X>>,CUSPARSE_ORDER_COL); 
    }
    return cuX; 
  } else {
    cusparseDnMatDescr_t cuX;
    if constexpr (std::decay_t<X>::is_stride_order_C()) {
      CUSPARSE_CHECK( cusparseCreateDnMat, &cuX, x.extent(0), x.extent(1), x.strides()[0], x.data(), cusparse_datatype<::nda::get_value_t<X>>,CUSPARSE_ORDER_ROW);
    } else {
      CUSPARSE_CHECK( cusparseCreateDnMat, &cuX, x.extent(0), x.extent(1), x.strides()[1], x.data(), cusparse_datatype<::nda::get_value_t<X>>,CUSPARSE_ORDER_COL);
    }
    return cuX;
  }
}


template<typename csr>
requires( CSRMatrix<csr> )
auto cuCSR(csr& spA, ::nda::MemoryArrayOfRank<1> auto& ofs) {
  constexpr MEMORY_SPACE MEM = csr::mem_type; 
  using value_type = std::decay_t<typename csr::value_type>;
  using index_type = std::decay_t<typename csr::index_type>;
  using int_type   = std::decay_t<typename csr::int_type>;
  static_assert( (MEM == DEVICE_MEMORY) or (MEM == UNIFIED_MEMORY), "Memory space mismatch.");
  static_assert(::nda::mem::have_device_compatible_addr_space<decltype(ofs)>, "Memory space mismatch");
  static_assert( std::is_same_v<index_type,int_type>, "Incompatible types: cuSparse requires index_type==int_type.");
  utils::check(spA.compact(), "device::csrmv: Sparse matrix must be in compact form.");
  if constexpr (std::is_const_v<std::remove_pointer_t<decltype(spA.values().data())>>) { 
    cusparseConstSpMatDescr_t cuA;
    auto [m, n] = spA.shape();
    if(ofs.extent(0) < m+1) ofs.resize(m+1);
    ofs(::nda::range(m)) = spA.row_begin()(::nda::range(m));
    ofs(::nda::range(m,m+1)) = spA.row_end()(::nda::range(m-1,m));
    CUSPARSE_CHECK( cusparseCreateConstCsr, &cuA, m, n, spA.nnz(),
                    ofs.data(), spA.columns().data(), spA.values().data(),
                    cusparse_indextype<int_type>, cusparse_indextype<index_type>,
                    CUSPARSE_INDEX_BASE_ZERO, cusparse_datatype<value_type> )
    return cuA;
  } else {
    cusparseSpMatDescr_t cuA;
    auto [m, n] = spA.shape();
    if(ofs.extent(0) < m+1) ofs.resize(m+1);
    ofs(::nda::range(m)) = spA.row_begin()(::nda::range(m));
    ofs(::nda::range(m,m+1)) = spA.row_end()(::nda::range(m-1,m));
    CUSPARSE_CHECK( cusparseCreateCsr, &cuA, m, n, spA.nnz(),
                    ofs.data(), spA.columns().data(), spA.values().data(),
                    cusparse_indextype<int_type>, cusparse_indextype<index_type>,
                    CUSPARSE_INDEX_BASE_ZERO, cusparse_datatype<value_type> )
    return cuA;
  }
}

} // namespace math::sparse::device 

#endif
