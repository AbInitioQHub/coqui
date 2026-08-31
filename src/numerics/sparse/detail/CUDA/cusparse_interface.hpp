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


#ifndef SPARSE_CUDA_GPU_HPP
#define SPARSE_CUDA_GPU_HPP

#include <type_traits>
#include <cassert>
#include <vector>
#include <complex>
#include <string>

#include "configuration.hpp"
#include "arch/arch.h"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"

#include "numerics/sparse/detail/CUDA/cusparse_aux.hpp"

namespace math::sparse::device
{

// defined in cusparse_interface.cpp
cusparseHandle_t &get_cusparse_handle_ptr();

// MAM: csr_matrix stores row pointers in host, so right now pntrb/pntre are host pointers.
//      They are copied to devide on the fly here. Write alternative routine that takes
//      the (compact) arrays in device
template<typename A_t, ::nda::MemoryVector X, ::nda:: MemoryVector Y>
requires((CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) and
         ::nda::have_same_value_type_v<X, Y>) 
void csrmv(::nda::get_value_t<X> alpha, A_t const& a, X const &x, ::nda::get_value_t<X> beta, Y &&y)
{
  using math::detail::arg;

  auto spA = arg(a);
  auto [m, n] = arg(a).shape();
 
  using csr = std::decay_t<decltype(spA)>;
  constexpr MEMORY_SPACE MEM = csr::mem_type; 
  using value_type = std::decay_t<typename csr::value_type>;
  using int_type   = std::decay_t<typename csr::int_type>;
  static_assert( std::is_same_v<value_type,::nda::get_value_t<X>>, "value_type mismatch.");
  static_assert( std::is_same_v<value_type,::nda::get_value_t<Y>>, "value_type mismatch.");

  auto handle = get_cusparse_handle_ptr(); 
  auto op_A = get_operation<A_t>();
  auto cuX = cuDn(x);  
  auto cuY = cuDn(y);  

  memory::array<MEM,int_type,1> ofs(m+1,int_type(0));
  auto cuA = cuCSR(spA,ofs);
   
  // allocate an external buffer if needed
  size_t bufferSize = 0;
  CUSPARSE_CHECK( cusparseSpMV_bufferSize, handle, op_A, 
                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
                  CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize) 
  memory::array<MEM,char,1> buffer(bufferSize,char(0));

  // execute preprocess (optional)
//  CUSPARSE_CHECK( cusparseSpMV_preprocess, handle, op_A, 
//                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
//                  CUSPARSE_SPMV_ALG_DEFAULT, (void*) buffer.data() )

  // execute SpMV
  CUSPARSE_CHECK( cusparseSpMV, handle, op_A, 
                  &alpha, cuA, cuX, &beta, cuY, cusparse_datatype<value_type>,
                  CUSPARSE_SPMV_ALG_DEFAULT, (void*) buffer.data() )

  CUSPARSE_CHECK( cusparseDestroySpMat, cuA )
  CUSPARSE_CHECK( cusparseDestroyDnVec, cuX )
  CUSPARSE_CHECK( cusparseDestroyDnVec, cuY )
  arch::synchronize_if_set();
}

template<typename T, typename A_t, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C> 
requires((CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) and
         (::nda::MemoryMatrix<B>) and   
         ::nda::have_same_value_type_v<B,C>)
void csrmm(T alpha, A_t const& a, B const &b, T beta, C &&c) 
{ 
  using math::detail::arg;

  auto spA = arg(a);
  auto [m, n] = arg(a).shape();
  
  using csr = std::decay_t<decltype(spA)>;
  constexpr MEMORY_SPACE MEM = csr::mem_type; 
  using value_type = std::decay_t<typename csr::value_type>;
  using int_type   = std::decay_t<typename csr::int_type>;

  if constexpr (std::is_same_v<::nda::get_value_t<B>,std::complex<T>>) {
    static_assert(std::decay_t<B>::is_stride_order_C() and std::decay_t<C>::is_stride_order_C(),
        "Mixed real/complex csrmm only with row-major matrices.");
    utils::check(b.indexmap().min_stride() == 1, "Stride mismatch");
    utils::check(c.indexmap().min_stride() == 1, "Stride mismatch");
    memory::array_view< memory::get_memory_space<B>(), T, 2, typename B::layout_policy_t> b_(std::array<long,2>{b.extent(0),2*b.extent(1)},reinterpret_cast<const T*>(b.data()));
    memory::array_view< memory::get_memory_space<C>(), T, 2, typename C::layout_policy_t> c_(std::array<long,2>{c.extent(0),2*c.extent(1)},reinterpret_cast<const T*>(c.data()));
    csrmm(alpha,a,b_,beta,c_);
    return;
  }

  static_assert( std::is_same_v<value_type,T>, "value_type mismatch.");
  static_assert( std::is_same_v<value_type,::nda::get_value_t<B>>, "value_type mismatch.");
  static_assert( std::is_same_v<value_type,::nda::get_value_t<C>>, "value_type mismatch.");
  
  auto handle = get_cusparse_handle_ptr();
  auto op_A = get_operation<A_t>();
  auto op_B = CUSPARSE_OPERATION_NON_TRANSPOSE; 
  auto cuB = cuDn(b);
  auto cuC = cuDn(c);
  
  memory::array<MEM,int_type,1> ofs(m+1,int_type(0));
  auto cuA = cuCSR(spA,ofs);

  // allocate an external buffer if needed
  size_t bufferSize = 0;
  CUSPARSE_CHECK( cusparseSpMM_bufferSize, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, &bufferSize)
  memory::array<MEM,char,1> buffer(bufferSize,char{0});
  
  // execute preprocess (optional)
  CUSPARSE_CHECK( cusparseSpMM_preprocess, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, (void*) buffer.data() )
  
  // execute SpMM
  CUSPARSE_CHECK( cusparseSpMM, handle, op_A, op_B, 
                  &alpha, cuA, cuB, &beta, cuC, cusparse_datatype<value_type>,
                  CUSPARSE_SPMM_CSR_ALG2, (void*) buffer.data() )
  
  CUSPARSE_CHECK( cusparseDestroySpMat, cuA )
  CUSPARSE_CHECK( cusparseDestroyDnMat, cuB )
  CUSPARSE_CHECK( cusparseDestroyDnMat, cuC )
  arch::synchronize_if_set();
}

} // namespace math::sparse::device


#endif
