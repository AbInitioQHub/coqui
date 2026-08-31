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


#ifndef SPARSE_CSR_BLAS_HPP
#define SPARSE_CSR_BLAS_HPP

#include "numerics/sparse/detail/concepts.hpp"
#include "numerics/sparse/detail/ops_aux.hpp"
#include <cassert>
#include "numerics/sparse/detail/CPU/sparse.hpp"
#if defined(ENABLE_CUDA)
//#include "numerics/sparse/detail/CUDA/sparse_cuda_gpu.hpp"
//#elif defined(ENABLE_HIP)
//#include "numerics/sparse/detail/HIP/sparse_hip_gpu.hpp"
#endif

#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"

namespace math::sparse
{

/***************************************************************************/
/*                              blas/lapack tags                           */
/***************************************************************************/

//CSRMatrix
template<CSRMatrix MA>
auto normal(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<CSRMatrix MA>
auto transpose(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<CSRMatrix MA>
auto dagger(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

template<CSRMatrix MA>
auto N(MA&& arg)
{ return math::detail::normal_tag<MA>(std::forward<MA>(arg)); }

template<CSRMatrix MA>
auto T(MA&& arg)
{ return math::detail::transpose_tag<MA>(std::forward<MA>(arg)); }

template<CSRMatrix MA>
auto H(MA&& arg)
{ return math::detail::conjugate_transpose_tag<MA>(std::forward<MA>(arg)); }

/***************************************************************************/
/*                              blas                                       */
/***************************************************************************/

template<typename A_t, ::nda::MemoryVector X, ::nda:: MemoryVector Y>
requires((CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) and 
         ::nda::have_same_value_type_v<X, Y> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<X>>)
void csrmv(::nda::get_value_t<X> alpha, A_t const& a, X const &x, ::nda::get_value_t<X> beta, Y &&y) {  

  using math::detail::arg;

  char op_a = math::detail::op_tag<A_t>::value;

  using A = std::decay_t<decltype(arg(a))>;
  static_assert(::nda::mem::have_compatible_addr_space<A, X, Y>);
  static_assert( std::is_same_v<typename A::index_type,int> or std::is_same_v<typename A::index_type,long>, "Invalid type");
  static_assert( std::is_same_v<typename A::int_type,int> or std::is_same_v<typename A::int_type,long>, "Invalid type");

  auto [m, n] = arg(a).shape();

  if(op_a == 'N') {
    utils::check(m == y.extent(0), "Shape mismatch");
    utils::check(n == x.extent(0), "Shape mismatch");
  } else {
    utils::check(m == x.extent(0), "Shape mismatch");
    utils::check(n == y.extent(0), "Shape mismatch");
  }

  // Must be lapack compatible
  utils::check(x.indexmap().min_stride() == 1, "Stride mismatch");
  utils::check(y.indexmap().min_stride() == 1, "Stride mismatch");

  if constexpr (::nda::mem::have_device_compatible_addr_space<A,X,Y>) {
#if defined(ENABLE_DEVICE)
    device::csrmv(alpha,a,x,beta,y);
#else
    utils::check(false," csr_blas on device without gpu support! Compile for GPU. ");
#endif
  } else {
    cpu::csrmv(op_a, m, n, alpha, "GxxCxx", arg(a).values().data(), arg(a).columns().data(), 
               arg(a).row_begin().data(), arg(a).row_end().data(), x.data(), beta, y.data());
  }  
}

template<typename T, typename A_t, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires((CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) and
         (::nda::MemoryMatrix<B> or ::nda::blas::is_conj_array_expr<B>) and
         ::nda::have_same_value_type_v<B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(T alpha, B const& b, A_t const &a, T beta, C &&c) {
  static_assert( std::is_same_v<T,::nda::get_value_t<B>>, "Type mismatch.");
  static_assert((std::decay_t<B>::is_stride_order_C() and std::decay_t<C>::is_stride_order_C()) or
                (std::decay_t<B>::is_stride_order_Fortran() and std::decay_t<C>::is_stride_order_Fortran()));

  char op_a = math::detail::op_tag<A_t>::value;
  utils::check(op_a == 'N' or op_a == 'T', "Error: No hermitian_tag allowed in csrmm(Dense,Sparse,Dense).");

  auto bt = ::nda::transpose(b);
  auto ct = ::nda::transpose(c);
 
  if(op_a == 'N') 
   csrmm(alpha,transpose(a),bt,beta,ct);
  else
   csrmm(alpha,a,bt,beta,ct);

}

template<typename T, typename A_t, ::nda::MemoryMatrix B, ::nda::MemoryMatrix C>
requires((CSRMatrix<A_t> or math::detail::is_tagged_matrix<A_t>) and
         (::nda::MemoryMatrix<B> or ::nda::blas::is_conj_array_expr<B>) and 
         ::nda::have_same_value_type_v<B, C> and
         ::nda::is_blas_lapack_v<::nda::get_value_t<C>>)
void csrmm(T alpha, A_t const& a, B const &b, T beta, C &&c) {          
  using math::detail::arg;
  using A = std::decay_t<decltype(arg(a))>;
  static_assert( std::is_same_v<T,::nda::get_value_t<B>>, "Type mismatch.");
  static_assert( std::is_same_v<typename A::index_type,int> or std::is_same_v<typename A::index_type,long>, "Invalid type");
  static_assert( std::is_same_v<typename A::int_type,int> or std::is_same_v<typename A::int_type,long>, "Invalid type");
  static_assert(::nda::mem::have_compatible_addr_space<A, B, C>);
  static_assert((std::decay_t<B>::is_stride_order_C() and std::decay_t<C>::is_stride_order_C()) or
                (std::decay_t<B>::is_stride_order_Fortran() and std::decay_t<C>::is_stride_order_Fortran()));
    
  char op_a = math::detail::op_tag<A_t>::value;
  if(op_a == 'N') {
    utils::check(arg(a).shape(0) == c.extent(0), "Shape mismatch");
    utils::check(arg(a).shape(1) == b.extent(0), "Shape mismatch");
  } else {
    utils::check(arg(a).shape(0) == b.extent(0), "Shape mismatch");
    utils::check(arg(a).shape(1) == c.extent(0), "Shape mismatch");
  }
  utils::check(b.shape(1) == c.extent(1), "Shape mismatch");
    
  // Must be lapack compatible
  utils::check(b.indexmap().min_stride() == 1, "Stride mismatch");
  utils::check(c.indexmap().min_stride() == 1, "Stride mismatch");
    
  auto [m, k] = arg(a).shape();
  auto n = c.extent(1);
    
  if constexpr (::nda::mem::have_device_compatible_addr_space<A,B,C>) {
#if defined(ENABLE_DEVICE)
    device::csrmm(alpha,a,b,beta,c);
#else 
    utils::check(false," csr_blas on device without gpu support! Compile for GPU. ");
#endif
  } else {
    if constexpr (std::decay_t<B>::is_stride_order_C()) {
      cpu::csrmm(op_a, m, n, k, alpha, "GxxCxx", arg(a).values().data(), arg(a).columns().data(), 
                 arg(a).row_begin().data(), arg(a).row_end().data(), b.data(), b.strides()[0], 0, 
                 beta, c.data(), c.strides()[0], 0, 1);
    } else if (std::decay_t<B>::is_stride_order_Fortran()) {
      cpu::csrmm(op_a, m, n, k, alpha, "GxxFxx", arg(a).values().data(), arg(a).columns().data(), 
                 arg(a).row_begin().data(), arg(a).row_end().data(), b.data(), b.strides()[1], 0, 
                 beta, c.data(), c.strides()[1], 0, 1);
    }
  }
}

}

#endif

