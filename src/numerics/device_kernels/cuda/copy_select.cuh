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


//////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef COPY_SELECT_CUDA_KERNELS_HPP
#define COPY_SELECT_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename V1, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, V1 const& m, T alpha, V3 const& A, T scl, V4&& B);

template<typename V1, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, int dim, V1 const& m, T alpha, V3 const& A, T scl, V4&& B);

template<typename V1, typename V2, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B);

template<typename V1, typename V2, typename V3, typename V4, typename T>
void copy_select_impl(bool expand, int dim, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B);

}

/*
 * Given a pair of arrays, copies selected values using supplied mapping.
 * expand=true: 
 *   B(m(n)) = scl * B(m(n)) + alpha * A(n)
 * expand=false:
 *   B(n) = scl * B(n) + alpha * A(m(n))
 *
 * Assumes:
 *   - map 'm' is consistent with the dimension of arrays being copied, 
 *      --> 0 <= m(i) < A.shape() for i in {0,B.shape()} 
 */
template<nda::MemoryArrayOfRank<1> V1, nda::MemoryArrayOfRank<1> V3, nda::MemoryArrayOfRank<1> V4, typename T>
void copy_select(bool expand, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  auto m_b = to_basic_layout(m()); 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  detail::copy_select_impl(expand,m_b,alpha,A_b,scl,B_b);  
}


/*
 * Assumes row major order.
 * Given a pair of 2D arrays, copies selected rows/cols using supplied mapping.
 * expand=true: 
 *   dim=0: B(m(n),i) = scl * B(m(n),i) + alpha * A(n,i)
 *   dim=1: B(i,m(n)) = scl * B(i,m(n)) + alpha * A(i,n)
 * expand=false: 
 *   dim=0: B(n,i) = scl * B(n,i) + alpha * A(m(n),i)
 *   dim=1: B(i,n) = scl * B(i,n) + alpha * A(i,m(n))
 *
 * Assumes:
 *   - map 'm' is consistent with the dimension of arrays being copied, 
 *      --> 0 <= m(i) < A.shape(dim) for i in {0,B.shape(dim)} 
 *   - B.shape(1-dim) = A.shape(1-dim) 
 */
template<nda::MemoryArrayOfRank<1> V1, nda::MemoryArrayOfRank<2> V3, nda::MemoryArrayOfRank<2> V4, typename T>
void copy_select(bool expand, int dim, V1 const& m, T alpha, V3 const& A, T scl, V4&& B)
{
  auto m_b = to_basic_layout(m()); 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  detail::copy_select_impl(expand,dim,m_b,alpha,A_b,scl,B_b);  
}

/*
 * Given a pair of arrays, copies selected values using supplied mapping.
 * expand=true: 
 *   B(m(n)) = scl * B(m(n)) + alpha * s(n) * A(n)
 * expand=true: 
 *   B(n) = scl * B(n) + alpha * s(n) * A(m(n))
 *
 * Assumes:
 *   - map 'm' is consistent with the dimension of arrays being copied, 
 *      --> 0 <= m(i) < A.shape() for i in {0,B.shape()} 
 */
template<nda::MemoryArrayOfRank<1> V1, nda::MemoryArrayOfRank<1> V2, nda::MemoryArrayOfRank<1> V3, nda::MemoryArrayOfRank<1> V4, typename T>
void copy_select(bool expand, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{
  auto m_b = to_basic_layout(m()); 
  auto s_b = to_basic_layout(s()); 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  detail::copy_select_impl(expand,m_b,s_b,alpha,A_b,scl,B_b);  
}

/*
 * Assumes row major order.
 * Given a pair of 2D arrays, copies selected rows/cols using supplied mapping.
 * expand=true: 
 *   dim=0: B(m(n),i) = scl * B(m(n),i) + alpha * s(n) * A(n,i)
 *   dim=1: B(i,m(n)) = scl * B(i,m(n)) + alpha * s(n) * A(i,n)
 * expand=true: 
 *   dim=0: B(n,i) = scl B(n,i) + alpha * s(n) * A(m(n),i)
 *   dim=1: B(i,n) = scl * B(i,n) + alpha * s(n) * A(i,m(n))
 *
 * Assumes:
 *   - map 'm' is consistent with the dimension of arrays being copied, 
 *   	--> 0 <= m(i) < A.shape(dim) for i in {0,B.shape(dim)} 
 *   - B.shape(1-dim) = A.shape(1-dim) 
 */ 
template<nda::MemoryArrayOfRank<1> V1, nda::MemoryArrayOfRank<1> V2, nda::MemoryArrayOfRank<2> V3, nda::MemoryArrayOfRank<2> V4, typename T>
void copy_select(bool expand, int dim, V1 const& m, V2 const& s, T alpha, V3 const& A, T scl, V4&& B)
{
  auto m_b = to_basic_layout(m()); 
  auto s_b = to_basic_layout(s()); 
  auto A_b = to_basic_layout(A()); 
  auto B_b = to_basic_layout(B()); 
  detail::copy_select_impl(expand,dim,m_b,s_b,alpha,A_b,scl,B_b);  
}

} // namespace kernels::device

#endif
