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


#ifndef NUMERICS_SHARED_ARRAY_NDA_BLAS_HPP
#define NUMERICS_SHARED_ARRAY_NDA_BLAS_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "mpi3/shared_window.hpp"
#include "nda/nda.hpp"
#include "nda/concepts.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/detail/concepts.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "utilities/check.hpp"

namespace math::shm::blas
{

namespace detail
{

template<typename A_t, typename B_t, ::nda::MemoryMatrix C_t>
requires((::nda::MemoryMatrix<A_t> or ::nda::blas::is_conj_array_expr<A_t>) and                  //
         (::nda::MemoryMatrix<B_t> or ::nda::blas::is_conj_array_expr<B_t>) and                  //
         ::nda::have_same_value_type_v<A_t,B_t,C_t>)
void gemm(mpi3::shared_communicator &node_comm, 
          typename std::decay_t<C_t>::value_type a, A_t const& A, B_t const& B, 
          typename std::decay_t<C_t>::value_type c, C_t && C)
{
  using ::nda::range;
  using ::nda::transpose;
  using ::nda::blas::has_C_layout;
  using ::nda::blas::has_F_layout;
  decltype(range::all) all;
  if constexpr (has_C_layout<C_t>) {
    gemm(node_comm,a,transpose(B),transpose(A),c,transpose(std::forward<C_t>(C)));
    return;
  }

  static constexpr bool conj_A = ::nda::blas::is_conj_array_expr<A_t>;
  static constexpr bool conj_B = ::nda::blas::is_conj_array_expr<B_t>;

  long np = node_comm.size();
  long rk = node_comm.rank();
  long nx = utils::find_proc_grid_min_diff(np,C.extent(0),C.extent(1));
  long ny = np/nx;
  long ix = rk/ny;
  long iy = rk%ny;

  auto [x0,x1] = itertools::chunk_range(0, C.extent(0), nx, ix);
  auto [y0,y1] = itertools::chunk_range(0, C.extent(1), ny, iy);
  auto C_ = C(range(x0,x1),range(y0,y1));

  auto to_mat = []<typename Z>(Z const &z) -> auto & {
    if constexpr (::nda::blas::is_conj_array_expr<Z>)
      return std::get<0>(z.a);
    else
      return z;
  };
  auto &Am = to_mat(A);
  auto &Bm = to_mat(B);

  using Am_t = decltype(Am);
  using Bm_t = decltype(Bm);

  auto call = [&](auto v1, auto const& M1, auto const& M2, auto v2, auto && M3) {
    if constexpr ( conj_A and conj_B )
      ::nda::blas::gemm(v1,::nda::conj(M1),::nda::conj(M2),v2,M3);
    else if constexpr (conj_A) 
      ::nda::blas::gemm(v1,::nda::conj(M1),M2,v2,M3);
    else if constexpr (conj_B) 
      ::nda::blas::gemm(v1,M1,::nda::conj(M2),v2,M3);
    else 
      ::nda::blas::gemm(v1,M1,M2,v2,M3);
  }; 

  node_comm.barrier();
  if constexpr (has_F_layout<Am_t> and has_F_layout<Bm_t>) {
  
    auto A_ = Am(range(x0,x1),all);
    auto B_ = Bm(all,range(y0,y1));
    call(a,A_,B_,c,C_);

  } else if constexpr (has_F_layout<Am_t>) {

    auto A_ = Am(range(x0,x1),all);
    auto B_ = Bm(range(y0,y1),all);
    call(a,A_,transpose(B_),c,C_);

  } else if constexpr (has_F_layout<Bm_t>) {

    auto A_ = Am(all,range(x0,x1));
    auto B_ = Bm(all,range(y0,y1));
    call(a,transpose(A_),B_,c,C_);

  } else {

    auto A_ = Am(all,range(x0,x1));
    auto B_ = Bm(range(y0,y1),all);
    call(a,transpose(A_),transpose(B_),c,C_);

  }
  node_comm.barrier();
   
}

}

// add decorations later... (e.g. H(), T())
template<SharedArray A_t, SharedArray B_t, SharedArray C_t>
void gemm(typename std::decay_t<A_t>::value_type a, A_t const& A, B_t const& B, 
          typename std::decay_t<C_t>::value_type c, C_t && C)
{
  utils::check( *A.node_comm() == *B.node_comm(), "Node comm mismatch");
  utils::check( *A.node_comm() == *C.node_comm(), "Node comm mismatch");
  detail::gemm(*A.node_comm(),a,A.local(),B.local(),c,C.local());
}

template<SharedArray A_t, SharedArray B_t, SharedArray C_t>
void gemm(A_t const& A, B_t const& B, C_t && C) 
{
  using A_v = typename std::decay_t<A_t>::value_type; 
  using C_v = typename std::decay_t<C_t>::value_type; 
  gemm(A_v(1.0),A.local(),B.local(),C_v(0.0),C.local());
}

}

#endif
