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

#ifndef POTENTIALS_POTENTIALS_IMPL_HPP
#define POTENTIALS_POTENTIALS_IMPL_HPP

#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "nda/nda.hpp"

#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#endif

namespace pots::detail
{

template<typename Arr>
struct eval_mesh_2d_impl {
  long g0;
  double cutoff;
  int screen_type;
  double screen_length;
  double tpitz;
  Arr V;
#if defined(__CUDACC__)
  cuda::std::array<long,3> mesh;
  cuda::std::array<double, 9> recv_;
  cuda::std::array<double, 3> dk;
#else
  static_assert(nda::is_view_v<Arr>, "Dispatch requires views.");
  nda::stack_array<long,3> mesh;
  nda::stack_array<double, 3, 3> recv;
  nda::stack_array<double, 3> dk;
#endif

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long ig)
  {
#if defined(__CUDACC__)
    using T = cuda::std::decay_t<decltype(V(0))>;
    cuda::std::mdspan<double const,cuda::std::extents<int, 3, 3>> recv(recv_.data(),3,3);
#else
    using T = std::decay_t<decltype(V(0))>;
#endif

    long g = g0+ig;
    long NX = mesh[0], NY = mesh[1], NZ = mesh[2];
    long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
    long iz = g%NZ; if(iz>NZ2) iz -= NZ;
    long n_ = g/NZ;
    long iy = n_%NY; if(iy>NY2) iy -= NY;
    long ix = n_/NY; if(ix>NX2) ix -= NX;
    double dx(ix), dy(iy), dz(iz);
    double gx = dx*recv(0,0) + dy*recv(1,0) + dz*recv(2,0) + dk[0];
    double gy = dx*recv(0,1) + dy*recv(1,1) + dz*recv(2,1) + dk[1];
    double gz = dx*recv(0,2) + dy*recv(1,2) + dz*recv(2,2) + dk[2];
    double gm = sqrt(gx*gx+gy*gy+gz*gz);
    if( gm <= cutoff or abs(gz) > 1e-8) {
      V(g) = T(0);
    } else if(screen_type == 0) {
      V(g) =  T(tpitz/gm);
    } else if(screen_type == 1) {
      V(g) =  T( tpitz / gm * tanh(gm*screen_length) );
    }
  };
};

template<typename Arr>
struct eval_mesh_3d_impl {
  long g0;
  double cutoff;
  int screen_type;
  double screen_length;
  Arr V;
#if defined(__CUDACC__)
  cuda::std::array<long,3> mesh;
  cuda::std::array<double, 9> recv_;
  cuda::std::array<double, 3> dk;
#else
  static_assert(nda::is_view_v<Arr>, "Dispatch requires views.");
  nda::stack_array<long,3> mesh;
  nda::stack_array<double, 3, 3> recv;
  nda::stack_array<double, 3> dk;
#endif

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long ig)
  {
#if defined(__CUDACC__)
    using T = cuda::std::decay_t<decltype(V(0))>;
    cuda::std::mdspan<double const,cuda::std::extents<int, 3, 3>> recv(recv_.data(),3,3);
#else
    using T = std::decay_t<decltype(V(0))>;
#endif

    long g = g0+ig;
    long NX = mesh[0], NY = mesh[1], NZ = mesh[2];
    long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
    long iz = g%NZ; if(iz>NZ2) iz -= NZ;
    long n_ = g/NZ;
    long iy = n_%NY; if(iy>NY2) iy -= NY;
    long ix = n_/NY; if(ix>NX2) ix -= NX;
    double dx(ix), dy(iy), dz(iz);
    double gx = dx*recv(0,0) + dy*recv(1,0) + dz*recv(2,0) + dk[0];
    double gy = dx*recv(0,1) + dy*recv(1,1) + dz*recv(2,1) + dk[1];
    double gz = dx*recv(0,2) + dy*recv(1,2) + dz*recv(2,2) + dk[2];
    double g2 = gx*gx+gy*gy+gz*gz;
    if( g2 <= cutoff) {
      V(g) = T(0);
    } else if(screen_type == 0) {
      V(g) =  T(4.0*3.14159265358979323846/(g2+screen_length)); 
    } else if(screen_type == 1) {
      V(g) = T(4.0*3.14159265358979323846 * ( 1.0 - erfc(-0.25*g2/(screen_length*screen_length)) ) /g2); 
    } else if(screen_type == 2) {
      V(g) =  T(4.0*3.14159265358979323846 * erfc(-0.25*g2/(screen_length*screen_length)) /g2);
    }
  };
};

template<typename Arr1, typename Arr2>
struct eval_2d_impl {
  long g0;
  double cutoff;
  int screen_type;
  double screen_length;
  double tpitz;
  Arr1 V;
  Arr2 gv;
#if defined(__CUDACC__)
  cuda::std::array<double, 3> dk;
#else
  static_assert(nda::is_view_v<Arr1>, "Dispatch requires views.");
  static_assert(nda::is_view_v<Arr2>, "Dispatch requires views.");
  nda::stack_array<double, 3> dk;
#endif

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long ig)
  {
#if defined(__CUDACC__)
    using T = cuda::std::decay_t<decltype(V(0))>;
#else
    using T = std::decay_t<decltype(V(0))>;
#endif
    long g = g0+ig;
    double gx = gv(g, 0) + dk[0]; 
    double gy = gv(g, 1) + dk[1]; 
    double gz = gv(g, 2) + dk[2];
    double gm = sqrt(gx*gx + gy*gy + gz*gz);
    if( gm <= cutoff or abs(gz) > 1e-8) {
      V(g) = T(0); 
    } else if(screen_type == 0) {
      V(g) =  T(tpitz/gm);
    } else if(screen_type == 1) {
      V(g) =  T( tpitz / gm * tanh(gm*screen_length) );
    }
  };
};

template<typename Arr1, typename Arr2>
struct eval_3d_impl {
  long g0;
  double cutoff;
  int screen_type;
  double screen_length;
  Arr1 V;
  Arr2 gv;
#if defined(__CUDACC__)
  cuda::std::array<double, 3> dk;
#else
  static_assert(nda::is_view_v<Arr1>, "Dispatch requires views.");
  static_assert(nda::is_view_v<Arr2>, "Dispatch requires views.");
  nda::stack_array<double, 3> dk;
#endif

#if defined(__CUDACC__)
  __device__
#endif
  void operator()(long ig)
  {
#if defined(__CUDACC__)
    using T = cuda::std::decay_t<decltype(V(0))>;
#else
    using T = std::decay_t<decltype(V(0))>;
#endif
    long g = g0+ig;
    double gx = gv(g, 0) + dk[0];
    double gy = gv(g, 1) + dk[1];
    double gz = gv(g, 2) + dk[2];
    double g2 = gx*gx + gy*gy + gz*gz;
    if( g2 <= cutoff) {
      V(g) = T(0);
    } else if(screen_type == 0) {
      V(g) =  T(4.0*3.14159265358979323846/(g2+screen_length)); 
    } else if(screen_type == 1) {
      V(g) = T(4.0*3.14159265358979323846 * ( 1.0 - erfc(-0.25*g2/(screen_length*screen_length)) ) /g2); 
    } else if(screen_type == 2) {
      V(g) =  T(4.0*3.14159265358979323846 * erfc(-0.25*g2/(screen_length*screen_length)) /g2);
    }
  };
};

}

#endif
