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


#ifndef NUMERICS_NDA_FUNCTIONS_HPP
#define NUMERICS_NDA_FUNCTIONS_HPP

/*
 * Collection of functions of nda arrays.
 * Routines here don't yet exist in nda and/or itertools
 * Move well defined routines to the original libraries eventually
 */ 

#include <algorithm>
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "numerics/device_kernels/kernels.h"

namespace nda
{

/*
 * Returns a tuple with the position (e.g. std::array with n-dimensional indexes) and value
 * of the maximum element in the array. For complex values, it considers only the real part 
 * of the number and return the real part of the number. 
 */
template<Array Arr>
auto argmin(Arr const& A)
{
  using T = std::decay_t<typename Arr::value_type>;
  using Tp = utils::remove_complex_t<T>;
  constexpr int rank = get_rank<Arr>;
  using ret_t = std::tuple<std::array<long, rank>, Tp>;
  if constexpr(nda::mem::on_host<Arr>) {
    using c_iter = array_iterator<rank, typename Arr::value_type const, 
					         typename Arr::value_type*>; 
    auto itb = c_iter{A.indexmap().lengths(), A.indexmap().strides(), A.data(), false}; 
    auto ite = c_iter{A.indexmap().lengths(), A.indexmap().strides(), A.data(), true}; 
    auto it = std::min_element(itb,ite, 
	[] (auto const& a, auto const& b) {return std::real(a) < std::real(b);});
    return std::tuple<decltype(it.indices()),Tp>{it.indices(),std::real(*it)};
  } else {
#if defined(ENABLE_DEVICE)
    if(A.is_contiguous()) {
      auto [p_d,v_d] = kernels::device::argmin(A.data(),A.size());
      std::array<long, rank> indx;
      if constexpr (Arr::is_stride_order_C()) {
	for(int i=rank-1; i>=0; --i) {
	  indx[i] = p_d%A.shape()[i];
	  p_d /= A.shape()[i];
	}
      } else if constexpr (Arr::is_stride_order_Fortran()) {
	for(int i=0; i<rank; ++i) {
	  indx[i] = p_d%A.shape()[i];
	  p_d /= A.shape()[i];
	}
      } else
        APP_ABORT("Error: Missing device function argmin with generic array layout.");
      return std::make_tuple(indx,std::real(v_d)); 
    } else {
      APP_ABORT("Error: Missing device function argmin.");
      return ret_t{}; 
    }
#else
    APP_ABORT("Error: Missing device function argmin.");
    return ret_t{}; 
#endif
  }
}

/*
 * Returns a tuple with the position (e.g. std::array with n-dimensional indexes) and value
 * of the minimum element in the array. For complex values, it considers only the real part 
 * of the number and return the real part of the number. 
 */
template<Array Arr>
auto argmax(Arr & A)
{   
  using T = std::decay_t<typename Arr::value_type>;
  using Tp = utils::remove_complex_t<T>;
  constexpr int rank = get_rank<Arr>;
  using ret_t = std::tuple<std::array<long, rank>, Tp>; 
  if constexpr(nda::mem::on_host<Arr>) {
    using c_iter = array_iterator<get_rank<Arr>, typename Arr::value_type const, 
					        typename Arr::value_type*>; 
    auto itb = c_iter{A.indexmap().lengths(), A.indexmap().strides(), A.data(), false}; 
    auto ite = c_iter{A.indexmap().lengths(), A.indexmap().strides(), A.data(), true}; 
    auto it = std::max_element(itb,ite,
	[] (auto const& a, auto const& b) {return std::real(a) < std::real(b);});
    return std::tuple<decltype(it.indices()),Tp>{it.indices(),std::real(*it)};
  } else {
#if defined(ENABLE_DEVICE)
    if(A.is_contiguous()) {
      auto [p_d,v_d] = kernels::device::argmax(A.data(),A.size());
      std::array<long, rank> indx;
      if constexpr (Arr::is_stride_order_C()) {
        for(int i=rank-1; i>=0; --i) {
          indx[i] = p_d%A.shape()[i];
          p_d /= A.shape()[i];
        }
      } else if constexpr (Arr::is_stride_order_Fortran()) {
        for(int i=0; i<rank; ++i) {
          indx[i] = p_d%A.shape()[i];
          p_d /= A.shape()[i];
        }
      } else
        APP_ABORT("Error: Missing device function argmin with generic array layout.");
      return std::make_tuple(indx,std::real(v_d));
    } else {
      APP_ABORT("Error: Missing device function argmax.");
      return ret_t{};
    } 
#else
    APP_ABORT("Error: Missing device function argmax.");
    return ret_t{};
#endif
  } 
}

// MAM: direction of copy can be reversed, add template parameter to control direction of copy
// e.g. expand = true:B(m(i))=A(i), false:B(i)=A(m(i))
template<MemoryArrayOfRank<1> V1, MemoryArrayOfRank<1> V3, MemoryArrayOfRank<1> V4, typename T>
void copy_select(bool expand, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  if(expand) {
    utils::check( m.shape() == A.shape(), "Shape mismatch");
    utils::check( B.shape()[0] >= A.shape()[0], "Shape mismatch");
  } else {
    utils::check( m.shape() == B.shape(), "Shape mismatch");
    utils::check( A.shape()[0] >= B.shape()[0], "Shape mismatch");
  }
  static_assert(nda::mem::have_compatible_addr_space<V1,V3,V4>, "Address space mismatch.");
  if constexpr(nda::mem::have_device_compatible_addr_space<V1,V3,V4>) {
#if defined(ENABLE_DEVICE)
    kernels::device::copy_select(expand,m,alpha,A,scl,B);
#else
    APP_ABORT("Error: Missing device function copy_select.");
#endif
  } else {
    if(expand)  
      for( auto [i,n] : itertools::enumerate(m) )
        B(n) = scl*B(n) + alpha*A(i);
    else
      for( auto [i,n] : itertools::enumerate(m) )
        B(i) = scl*B(i) + alpha*A(n);
  }
}

template<MemoryArrayOfRank<1> V1, MemoryArrayOfRank<1> V2, MemoryArrayOfRank<1> V3, MemoryArrayOfRank<1> V4, typename T>
void copy_select(bool expand, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{
  utils::check( s.shape() == m.shape(), "Shape mismatch");
  if(expand) {
    utils::check( s.shape() == A.shape(), "Shape mismatch");
    utils::check( B.shape()[0] >= A.shape()[0], "Shape mismatch");
  } else {
    utils::check( s.shape() == B.shape(), "Shape mismatch");
    utils::check( A.shape()[0] >= B.shape()[0], "Shape mismatch");
  }
  static_assert(nda::mem::have_compatible_addr_space<V1,V2,V3,V4>, "Address space mismatch.");
  if constexpr(nda::mem::have_device_compatible_addr_space<V1,V2,V3,V4>) {  
#if defined(ENABLE_DEVICE)
    kernels::device::copy_select(expand,m,s,alpha,A,scl,B);
#else
    APP_ABORT("Error: Missing device function copy_select.");
#endif
  } else {
    if(expand)
      for( auto [i,n] : itertools::enumerate(m) )
        B(n) = scl*B(n) + alpha * s(i) * A(i);
    else
      for( auto [i,n] : itertools::enumerate(m) )
        B(i) = scl*B(i) + alpha * s(i) * A(n);
  }
}

//update with have_host/device_address_space_v!!!
template<MemoryArrayOfRank<1> V1, MemoryArrayOfRank<2> V3, MemoryArrayOfRank<2> V4, typename T>
void copy_select(bool expand, int indx, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  utils::check( indx >= 0 and indx <= 1, "Index mismatch");
  if(expand) {
    utils::check( m.shape()[0] == A.shape()[indx], "Shape mismatch");
    utils::check( B.shape()[1-indx] == A.shape()[1-indx], "Shape mismatch");
    utils::check( B.shape()[indx] >= A.shape()[indx], "Shape mismatch");
  } else {
    utils::check( m.shape()[0] == B.shape()[indx], "Shape mismatch");
    utils::check( A.shape()[1-indx] == B.shape()[1-indx], "Shape mismatch");
    utils::check( A.shape()[indx] >= B.shape()[indx], "Shape mismatch");
  }
  static_assert(nda::mem::have_compatible_addr_space<V1,V3,V4>, "Address space mismatch.");
  if constexpr(nda::mem::have_device_compatible_addr_space<V1,V3,V4>) {
#if defined(ENABLE_DEVICE)
    kernels::device::copy_select(expand,indx,m,alpha,A,scl,B);   
#else
    APP_ABORT("Error: Missing device function copy_select.");
#endif
  } else {
    if(expand) {
      if(indx==0) {
        for( auto [i,n] : itertools::enumerate(m) )
          for( auto r : itertools::range(B.shape()[1]) )
            B(n,r) = scl*B(n,r) + alpha * A(i,r);
      } else if(indx == 1) {
        for( auto r : itertools::range(B.shape()[0]) )
          for( auto [i,n] : itertools::enumerate(m) )
            B(r,n) = scl*B(r,n) + alpha * A(r,i);
      } 
    } else {
      if(indx==0) {
        for( auto [i,n] : itertools::enumerate(m) )
          for( auto r : itertools::range(B.shape()[1]) )
            B(i,r) = scl*B(i,r) + alpha * A(n,r);
      } else if(indx == 1) {
        for( auto r : itertools::range(B.shape()[0]) )
          for( auto [i,n] : itertools::enumerate(m) )
            B(r,i) = scl*B(r,i) + alpha * A(r,n);
      }
    }
  }
}

template<MemoryArrayOfRank<1> V1, MemoryArrayOfRank<1> V2, MemoryArrayOfRank<2> V3, MemoryArrayOfRank<2> V4, typename T>
void copy_select(bool expand, int indx, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{
  utils::check( indx >= 0 and indx <= 1, "Index mismatch");
  utils::check( s.shape() == m.shape(), "Shape mismatch");
  if(expand) {
    utils::check( s.shape()[0] == A.shape()[indx], "Shape mismatch");
    utils::check( B.shape()[1-indx] == A.shape()[1-indx], "Shape mismatch");
    utils::check( B.shape()[indx] >= A.shape()[indx], "Shape mismatch");
  } else {
    utils::check( s.shape()[0] == B.shape()[indx], "Shape mismatch");
    utils::check( A.shape()[1-indx] == B.shape()[1-indx], "Shape mismatch");
    utils::check( A.shape()[indx] >= B.shape()[indx], "Shape mismatch");
  }
  static_assert(nda::mem::have_compatible_addr_space<V1,V2,V3,V4>, "Address space mismatch.");
  if constexpr(nda::mem::have_device_compatible_addr_space<V1,V2,V3,V4>) {
#if defined(ENABLE_DEVICE)
    kernels::device::copy_select(expand,indx,m,s,alpha,A,scl,B);
#else
    APP_ABORT("Error: Missing device function copy_select.");
#endif
  } else {
    if(expand) {
      if(indx==0) {
        for( auto [i,n] : itertools::enumerate(m) )
          for( auto r : itertools::range(B.shape()[1]) )
            B(n,r) = scl*B(n,r) + alpha * s(i) * A(i,r);
      } else if(indx == 1) {
       for( auto r : itertools::range(B.shape()[0]) )
          for( auto [i,n] : itertools::enumerate(m) )
            B(r,n) = scl*B(r,n) + alpha * s(i) * A(r,i);
      }
    } else {
      if(indx==0) {
        for( auto [i,n] : itertools::enumerate(m) )
          for( auto r : itertools::range(B.shape()[1]) )
            B(i,r) = scl*B(i,r) + alpha * s(i) * A(n,r);
      } else if(indx == 1) {
       for( auto r : itertools::range(B.shape()[0]) )
          for( auto [i,n] : itertools::enumerate(m) )
            B(r,i) = scl*B(r,i) + alpha * s(i) * A(r,n);
      }
    }
  }
}

template<MemoryArray Arr>
void zero_imag(Arr && A) 
{
  using value_type = get_value_t<Arr>;
  if constexpr (is_complex_v<value_type>) { 
    if constexpr(nda::mem::on_host<Arr>) {
      for( auto& v : A ) v = value_type(v.real(),0.0);
    } else {
#if defined(ENABLE_DEVICE)
      kernels::device::zero_imag(A);
#else
      APP_ABORT("Error: Found device array without ENABLE_DEVICE."); 
#endif
    }
  }
}

namespace blas
{

template <typename A, MemoryVector B, MemoryVector C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and                        
             have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
void gemv(A const &a, B const &b, C &&c) 
{
  if constexpr (is_conj_array_expr<A>) {
    auto mat = std::get<0>(a.a); 
    using T = typename decltype(mat)::value_type;
    gemv(T(1.0),a,b,T(0.0),c);
  } else {
    using T = typename A::value_type;
    gemv(T(1.0),a,b,T(0.0),c);
  }
}

template <typename A, typename B, MemoryMatrix C>
    requires((MemoryMatrix<A> or is_conj_array_expr<A>) and                        
             (MemoryMatrix<B> or is_conj_array_expr<B>) and                        
             have_same_value_type_v<A, B, C> and is_blas_lapack_v<get_value_t<A>>)
void gemm(A const &a, B const &b, C &&c) 
{
  if constexpr (is_conj_array_expr<A>) {
    auto mat = std::get<0>(a.a); 
    using T = typename decltype(mat)::value_type;
    gemm(T(1.0),a,b,T(0.0),c);
  } else {
    using T = typename A::value_type;
    gemm(T(1.0),a,b,T(0.0),c);
  }
}

}

namespace tensor::cutensor
{

#if defined(ENABLE_DEVICE)
template<nda::MemoryArray Arr>
requires( nda::mem::have_device_compatible_addr_space<Arr> )
auto to_cutensor(Arr&& A) {
  using T = nda::get_value_t<Arr>;
  constexpr int R = nda::get_rank<Arr>;
  return nda::tensor::cutensor::cutensor_desc<T,R>(A);
}
#endif

}

}

#endif
