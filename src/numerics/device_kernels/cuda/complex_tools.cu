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


#include "stdio.h"
#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include "cub/device/device_for.cuh"
#include "thrust/for_each.h"
#include "thrust/iterator/counting_iterator.h"

namespace kernels::device::detail
{

template<typename Arr>
void zero_imag_impl(Arr & A)
{
  long N = A.size();
  constexpr int rank = nda::get_rank<Arr>;
  auto A_d = to_cuda_std_mdspan(A);
  if constexpr (rank==1) {
    auto f = [=] __device__(long i) {
      A_d(i) = cuda::std::complex<double>{A_d(i).real(),0.0};  
    };
    cub::DeviceFor::Bulk(N,f);
  } else if constexpr (rank==2) {
    long nc = A.extent(1);
    auto f = [=] __device__(long i) {
      long b = i/nc; 
      long a = i - b*nc;
      A_d(a,b) = cuda::std::complex<double>{A_d(a,b).real(),0.0};
    };
    cub::DeviceFor::Bulk(N,f);
  } else if constexpr (rank==3) {
    long n1 = A.extent(1);
    long n2 = A.extent(2);
    auto f = [=] __device__(long i) {
      long i_ = i/n2;
      long a = i_/n1;
      long b = i_ - a*n1; 
      long c = i - i_*n2;
      A_d(a,b,c) = cuda::std::complex<double>{A_d(a,b,c).real(),0.0};
    };
    cub::DeviceFor::Bulk(N,f);
  }
  
}

using memory::device_array_view;
using memory::unified_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define __inst__(V,T,R) template void zero_imag_impl(V<complex<T>,R,basic_layout_t<R>> &);

__inst__(device_array_view,double,1)
__inst__(device_array_view,double,2)
__inst__(device_array_view,double,3)

__inst__(unified_array_view,double,1)
__inst__(unified_array_view,double,2)
__inst__(unified_array_view,double,3)

} // kernels::device::detail

