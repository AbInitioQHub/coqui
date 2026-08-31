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


#ifndef SYMMETRY_UTILS_UTILITIES_HXX
#define SYMMETRY_UTILS_UTILITIES_HXX

#include <complex>
#include <algorithm>
#include <array>

#include "configuration.hpp"
#include "utilities/type_traits.hpp"
#if defined(__CUDACC__)
#include <cuda/std/mdspan>
#else
#include "nda/nda.hpp"
#endif

namespace utils::detail
{

template<typename V1>
struct transform_k2g {

  double sg; 
#if defined(__CUDACC__)
  cuda::std::array<long,3> mesh;
  cuda::std::array<double, 3> Gs; 
  cuda::std::array<double, 9> Rinv_v;
#else
  static_assert(nda::is_view_v<V1>, "Dispatch requires views.");
  nda::stack_array<long,3> mesh;
  nda::stack_array<double, 3> Gs; 
  nda::stack_array<double, 3, 3> Rinv;
#endif
  V1 k2g;
  int* err;

#if defined(__CUDACC__)
  __device__ 
#endif
  void operator()(int i)
  {

#ifndef __CUDACC__
    using std::abs;
#endif
    int err_l=0;
    long NX = mesh[0], NY = mesh[1], NZ = mesh[2];
    long nnr = NX*NY*NZ;
    long NX2 = NX/2, NY2 = NY/2, NZ2 = NZ/2;
#if defined(__CUDACC__)
    cuda::std::mdspan<double const,cuda::std::extents<int, 3, 3>> Rinv(Rinv_v.data(),3,3);
#endif

    long n = k2g(i);
    long n2 = n%NZ; if( n2 > NZ2 ) n2 -= NZ;
    long n_ = n/NZ;
    long n1 = n_%NY; if( n1 > NY2 ) n1 -= NY;
    long n0 = n_/NY; if( n0 > NX2 ) n0 -= NX;

    if(abs(n0) > NX2 or 
       abs(n1) > NY2 or
       abs(n2) > NZ2) 
      err_l = 1;

    // G*Rinv - Gs 
    double ni_d = double(n0)*Rinv(0,0) + double(n1)*Rinv(1,0) + double(n2)*Rinv(2,0) - Gs[0];
    double nj_d = double(n0)*Rinv(0,1) + double(n1)*Rinv(1,1) + double(n2)*Rinv(2,1) - Gs[1];
    double nk_d = double(n0)*Rinv(0,2) + double(n1)*Rinv(1,2) + double(n2)*Rinv(2,2) - Gs[2];

    // trev
    ni_d *= sg;
    nj_d *= sg;
    nk_d *= sg;

    long ni_i = long(round(ni_d));
    long nj_i = long(round(nj_d));
    long nk_i = long(round(nk_d));

    if(abs( ni_d - double(ni_i) ) > 1e-6 or
       abs( nj_d - double(nj_i) ) > 1e-6 or
       abs( nk_d - double(nk_i) ) > 1e-6)
      err_l = 2;

    while(ni_i<0) ni_i += NX;
    while(nj_i<0) nj_i += NY;
    while(nk_i<0) nk_i += NZ;
    while(ni_i>=NX) ni_i -= NX;
    while(nj_i>=NY) nj_i -= NY;
    while(nk_i>=NZ) nk_i -= NZ;

    k2g(i) = (ni_i*NY + nj_i)*NZ + nk_i;

    if(k2g(i) < 0 or k2g(i) > nnr or err_l > 0) {
      k2g(i) = -1;
#if defined(__CUDACC__)
      // unlikely to happen, so atomic operation is fine
      atomicAdd(err,1); 
#else
      *err = 1;
#endif
    }
  };
};


}

#endif
