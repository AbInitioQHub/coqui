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


#ifndef UTILITIES_DETAILS_KPOINT_IMPL_HPP
#define UTILITIES_DETAILS_KPOINT_IMPL_HPP

#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#include <cuda/std/complex>
#else
#include <complex>
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#endif

namespace utils::detail
{

/*
 * Kernels for for_each for routines in utilities/kpoint_utils.hpp 
 */

template<typename V1>
struct rspace_phase_factor_mesh 
{

  long n0;
#if defined(__CUDACC__)
  cuda::std::array<long,3> mesh;
  cuda::std::array<double, 3> G;
  cuda::std::array<double, 9> lattv_v;
#else
  static_assert(nda::is_view_v<V1>, "Dispatch requires views.");
  nda::stack_array<long,3> mesh;
  nda::stack_array<double, 3> G;
  nda::stack_array<double, 3, 3> lattv;
#endif
  V1 f;

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long nn)
  {
#if defined(__CUDACC__)
    using cuda::std::complex;
    cuda::std::mdspan<double const,cuda::std::extents<int, 3, 3>> lattv(lattv_v.data(),3,3);
#else
    using std::complex;
    using std::exp;
#endif  
    double kn = double( (nn+n0) % mesh[2] ) / double(mesh[2]);
    long n_ = (nn+n0)/mesh[2];
    double jn = double( n_%mesh[1] ) / double(mesh[1]);
    double in = double( n_/mesh[1] ) / double(mesh[0]);
    double Gr = G[0] * (in*lattv(0,0) + jn*lattv(1,0) + kn*lattv(2,0)) +
                G[1] * (in*lattv(0,1) + jn*lattv(1,1) + kn*lattv(2,1)) +
                G[2] * (in*lattv(0,2) + jn*lattv(1,2) + kn*lattv(2,2));
    f(nn) = exp( complex{0.0, Gr} );
  };
};

template<typename V1, typename V2>
struct rspace_phase_factor_crystal
{

#if defined(__CUDACC__)
  cuda::std::array<long,3> mesh;
  cuda::std::array<double, 3> G;
#else
  static_assert(nda::is_view_v<V1>, "Dispatch requires views.");
  static_assert(nda::is_view_v<V2>, "Dispatch requires views.");
  nda::stack_array<long,3> mesh;
  nda::stack_array<double, 3> G;
#endif
  V1 gv;
  V2 f;

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long p)
  {
#if defined(__CUDACC__)
    using cuda::std::complex;
#else
    using std::complex;
    using std::exp;
#endif
    long n = gv(p);
    // n = ( i*nj + j ) * nk + k
    double kn = double(n%mesh[2])/double(mesh[2]);
    long n_ = n/mesh[2];
    double jn = double(n_%mesh[1])/double(mesh[1]);
    double in = double(n_/mesh[1])/double(mesh[0]);
    f(p) = exp( complex{0.0, 6.283185307179586*(G[0]*in + G[1]*jn + G[2]*kn)} );
  };
};

// This routine assumes that the node communicator in f_Rk lives within comm.
/**
 * Fourier transform kernel from k to R space with provided R vectors
 * @param Rpts_idx - [INPUT] R vectors in the unit of lattice vectors. Dimension: (nRpts, 3).
 * @param kpts - [INPUT] Target k-points in the unit of reciprocal lattice vectors. Dimension: (nkpts, 3).
 * @param lattv - [INPUT] Lattice vectors
 * @param f_kR - [OUTPUT] Fourier kernel. Dimension: (nRpts, nkpts)
 */
/*
inline void k_to_R_coefficients(nda::ArrayOfRank<2> auto const& Rpts_idx,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                nda::ArrayOfRank<2> auto &&f_Rk) {
  using value_type = typename std::decay_t<decltype(f_Rk)>::value_type;
  constexpr double tpi = 2.0 * 3.14159265358979;
  long nR = Rpts_idx.shape(0);
  long nk = kpts.shape(0);
  utils::check(f_Rk.shape() == std::array<long,2>{nR,nk},
               "k_to_r_coefficients: f_Rk shape mismatches. ({},{}), ({},{})",
               f_Rk.shape(0), f_Rk.shape(1), nR, nk);
  for (int i=0; i<nR; ++i) {
    long a = Rpts_idx(i,0);
    long b = Rpts_idx(i,1);
    long c = Rpts_idx(i,2);
    for (long q=0; q<nk; ++q) {
      //      double kR = kpts_crys(q,0)*a +  kpts_crys(q,1)*b + kpts_crys(q,2)*c;
      //      f_Rk(i, q) = std::exp(value_type(0.0, -tpi*kR)) / scl;
      double kR = kpts(q,0) * (a*lattv(0,0) + b*lattv(1,0) + c*lattv(2,0)) +
                  kpts(q,1) * (a*lattv(0,1) + b*lattv(1,1) + c*lattv(2,1)) +
                  kpts(q,2) * (a*lattv(0,2) + b*lattv(1,2) + c*lattv(2,2));
      f_Rk(i, q) = std::exp(value_type(0.0, -kR)) / nk;
    }
  }
}

// This routine assumes that the node communicator in f_Rk lives within comm.
// remove lattv when merged with new_symmetry branch!!!
template<class Communicator>
inline void k_to_R_coefficients(Communicator & comm, 
                                nda::range r_range,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                nda::ArrayOfRank<1> auto const& fft_dim,
                                math::shm::SharedArray auto &&f_Rk) 
{
  using value_type = typename std::decay_t<decltype(f_Rk)>::value_type;
  long np = r_range.size();
  long nk = kpts.shape(0);
  long nnr = fft_dim(0)*fft_dim(1)*fft_dim(2);
  utils::check(r_range.first() >= 0 and r_range.last() <= nnr, 
      "k_to_r_coefficients: Range mismatch: ", r_range.first(),r_range.last());
  utils::check(nk <= nnr, "k_to_r_coefficients: fft_sim.size() ({}) < nkpts ({})",
               nnr, nk);
  utils::check(f_Rk.shape()[0] >= np and f_Rk.shape()[1] >= nk, 
      "k_to_r_coefficients: f_Rk shape mismatches. ({},{}), ({},{})", 
      f_Rk.shape()[0], f_Rk.shape()[1], np, nk);

  f_Rk.set_zero();
  auto f_loc = f_Rk.local()(nda::range(np),nda::range(nk));
  auto[r0, r1] = itertools::chunk_range(r_range.first(), r_range.last(), comm.size(), comm.rank());
  nda::range r_range_loc(r0,r1);
  // calculate f_Rk in local r_range
  if(r_range_loc.size() > 0) {
    long nx = fft_dim(0);
    long ny = fft_dim(1);
    long nz = fft_dim(2);
    nda::array<long, 2> Rpts_idx(r_range_loc.size(), 3);
    for ( auto [i, p]: itertools::enumerate(r_range_loc) ) {
      long a = p / (ny * nz);
      long b = (p / nz) % ny;
      long c = p % nz;
      if (a > nx / 2) a -= nx;
      if (b > ny / 2) b -= ny;
      if (c > nz / 2) c -= nz;
      Rpts_idx(i, 0) = a;
      Rpts_idx(i, 1) = b;
      Rpts_idx(i, 2) = c;
    }
    k_to_R_coefficients(Rpts_idx, kpts, lattv,
                        f_loc(nda::range(r0-r_range.first(), r1-r_range.first()), nda::range::all));
  }
  f_Rk.node_comm()->barrier();
  // reduce over comm 
  if (f_Rk.node_comm()->root()) {
    f_Rk.internode_comm()->all_reduce_in_place_n(f_loc.data(), np*f_Rk.shape()[1], std::plus<>{});
  }
  f_Rk.node_sync();
  comm.barrier();
}

template<class Communicator>
inline void k_to_R_coefficients(Communicator & comm,
                                nda::ArrayOfRank<2> auto const& Rpts_idx,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                math::shm::SharedArray auto &&f_Rk)
{
  using value_type = typename std::decay_t<decltype(f_Rk)>::value_type;
  long nR = Rpts_idx.shape(0);
  nda::range R_range(nR);
  long nk = kpts.shape(0);
  utils::check(f_Rk.shape()[0] >= nR and f_Rk.shape()[1] >= nk,
               "k_to_r_coefficients: f_Rk shape mismatches. ({},{}), ({},{})",
               f_Rk.shape()[0], f_Rk.shape()[1], nR, nk);

  f_Rk.set_zero();
  auto f_loc = f_Rk.local()(nda::range(nR),nda::range(nk));
  auto[r0, r1] = itertools::chunk_range(R_range.first(), R_range.last(), comm.size(), comm.rank());
  nda::range R_range_loc(r0, r1);
  // calculate f_Rk in local r_range
  if(R_range_loc.size() > 0)
    k_to_R_coefficients(Rpts_idx(R_range_loc, nda::range::all), kpts, lattv,
                        f_loc(R_range_loc, nda::range::all));
  f_Rk.node_comm()->barrier();
  // reduce over comm
  if (f_Rk.node_comm()->root()) {
    f_Rk.internode_comm()->all_reduce_in_place_n(f_loc.data(), nR*f_Rk.shape()[1], std::plus<>{});
  }
  f_Rk.node_sync();
  comm.barrier();
}
*/


// This routine assumes that the node communicator in f_kR lives within comm.
/**
 * Generalized Fourier transform kernel from R to k space with provided R vectors and the corresponding weights
 * @param Rpts_idx - [INPUT] R vectors in the unit of lattice vectors. Dimension: (nRpts, 3).
 * @param Rpts_weights - [INPUT] Weights for each R pointt. Dimension: (nRpts).
 * @param kpts - [INPUT] Target k-points in the unit of reciprocal lattice vectors. Dimension: (nkpts, 3).
 * @param lattv - [INPUT] Lattice vectors
 * @param f_kR - [OUTPUT] Fourier kernel. Dimension: (nkpts, nRpts)
 */
/*
inline void R_to_k_coefficients(nda::ArrayOfRank<2> auto const& Rpts_idx,
                                nda::ArrayOfRank<1> auto const& Rpts_weights,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                nda::ArrayOfRank<2> auto &&f_kR) {
  using value_type = typename std::decay_t<decltype(f_kR)>::value_type;
  constexpr double tpi = 2.0 * 3.14159265358979;
  long nR = Rpts_idx.shape(0);
  long nk = kpts.shape(0);
  utils::check(f_kR.shape() == std::array<long,2>{nk,nR},
               "R_to_k_coefficients: f_kR shape mismatches. ({},{}), ({},{})",
               f_kR.shape(0), f_kR.shape(1), nk, nR);

  for (long q = 0; q < nk; ++q) {
    for (long R=0; R<nR; ++R) {
      long a = Rpts_idx(R, 0);
      long b = Rpts_idx(R, 1);
      long c = Rpts_idx(R, 2);
      double kR = kpts(q,0) * (a*lattv(0,0) + b*lattv(1,0) + c*lattv(2,0)) +
                  kpts(q,1) * (a*lattv(0,1) + b*lattv(1,1) + c*lattv(2,1)) +
                  kpts(q,2) * (a*lattv(0,2) + b*lattv(1,2) + c*lattv(2,2));
      f_kR(q, R) = std::exp(value_type(0.0, kR)) / Rpts_weights(R);
    }
  }
}

// This routine assumes that the node communicator in f_kR lives within comm.
// remove lattv when merged with new_symmetry branch!!!
template<class Communicator>
inline void R_to_k_coefficients(Communicator & comm,
                                nda::range r_range,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                nda::ArrayOfRank<1> auto const& fft_dim,
                                math::shm::SharedArray auto &&f_kR)
{ 
  using value_type = typename std::decay_t<decltype(f_kR)>::value_type;
  long nR = r_range.size();
  long nk = kpts.shape(0);
  long nnr = fft_dim(0)*fft_dim(1)*fft_dim(2);
  utils::check(r_range.first() >= 0 and r_range.last() <= nnr, 
      "R_to_k_coefficients: Range mismatch: ", r_range.first(),r_range.last());
  utils::check(f_kR.shape()[0] >= nk and f_kR.shape()[1] >= nR,
      "R_to_k_coefficients: f_kR shape mismatches. ({},{}), ({},{})",
      f_kR.shape()[0], f_kR.shape()[1], nk, nR);
  
  f_kR.set_zero();
  auto f_loc = f_kR.local()(nda::range(nk),nda::range(nR));
  auto[r0, r1] = itertools::chunk_range(r_range.first(), r_range.last(), comm.size(), comm.rank());
  nda::range r_range_loc(r0,r1);
  // calculate f_kR in local r_range
  if(r_range_loc.size() > 0) {
    nda::array<long ,2> Rpts_idx(r_range_loc.size(), 3);
    nda::array<long, 1> Rpt_weights(r_range_loc.size());
    Rpt_weights() = 1;
    long nx = fft_dim(0);
    long ny = fft_dim(1);
    long nz = fft_dim(2);
    for ( auto [i, p]: itertools::enumerate(r_range_loc) ) {
      long a = p / (ny * nz);
      long b = (p / nz) % ny;
      long c = p % nz;
      if (a > nx / 2) a -= nx;
      if (b > ny / 2) b -= ny;
      if (c > nz / 2) c -= nz;
      Rpts_idx(i, 0) = a;
      Rpts_idx(i, 1) = b;
      Rpts_idx(i, 2) = c;
    }

    R_to_k_coefficients(Rpts_idx, Rpt_weights, kpts, lattv,
                        f_loc(nda::range::all, nda::range(r0-r_range.first(), r1-r_range.first())));
  }
  f_kR.node_comm()->barrier();
  // reduce over comm 
  if (f_kR.node_comm()->root()) {
    f_kR.internode_comm()->all_reduce_in_place_n(f_loc.data(), nk*f_kR.shape()[1], std::plus<>{});
  }
  f_kR.node_sync();
  comm.barrier();
}

template<class Communicator>
inline void R_to_k_coefficients(Communicator & comm,
                                nda::ArrayOfRank<2> auto const& Rpt_idx,
                                nda::ArrayOfRank<1> auto const& Rpt_weights,
                                nda::ArrayOfRank<2> auto const& kpts,
                                nda::ArrayOfRank<2> auto const& lattv,
                                math::shm::SharedArray auto &&f_kR)
{
  using value_type = typename std::decay_t<decltype(f_kR)>::value_type;
  long nR = Rpt_idx.shape(0);
  nda::range R_range(nR);
  long nk = kpts.shape(0);
  utils::check(f_kR.shape()[0] >= nk and f_kR.shape()[1] >= nR,
               "R_to_k_coefficients: f_kR shape mismatches. ({},{}), ({},{})",
               f_kR.shape()[0], f_kR.shape()[1], nk, nR);

  f_kR.set_zero();
  auto f_loc = f_kR.local()(nda::range(nk),nda::range(nR));
  auto[r0, r1] = itertools::chunk_range(R_range.first(), R_range.last(), comm.size(), comm.rank());
  nda::range R_range_loc(r0,r1);
  // calculate f_kR in local r_range
  if(R_range_loc.size() > 0)
    R_to_k_coefficients(Rpt_idx(R_range_loc, nda::range::all), Rpt_weights(R_range_loc), kpts,lattv,
                        f_loc(nda::range::all,nda::range(r0-R_range.first(),r1-R_range.first())));
  f_kR.node_comm()->barrier();
  // reduce over comm
  if (f_kR.node_comm()->root()) {
    f_kR.internode_comm()->all_reduce_in_place_n(f_loc.data(), nk*f_kR.shape()[1], std::plus<>{});
  }
  f_kR.node_sync();
  comm.barrier();
}
*/

} // utils

#endif
