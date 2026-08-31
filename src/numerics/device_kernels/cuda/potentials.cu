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

#include <complex>
#include <algorithm>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "nda/nda.hpp"
#include "potentials/potentials_impl.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_vector.h>

namespace kernels::device::detail
{

template<typename V1>
void eval_mesh_2d_impl(double tpitz, double cutoff, int screen_type, double screen_length,
     nda::range rng, V1 & V, nda::stack_array<long,3> const& mesh,
     nda::stack_array<double,3,3> const& recv, nda::stack_array<double,3> const& k)
{
  long N = rng.size();
  if(N==0) return;
  auto V_d = to_cuda_std_mdspan(V);
  auto F = pots::detail::eval_mesh_2d_impl<decltype(V_d)>{rng.first(),cutoff,screen_type,
    screen_length, tpitz, V_d,to_cuda_std_array<3>(mesh),
    to_cuda_std_array<9>(recv), to_cuda_std_array<3>(k)};
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}

template<typename V1>
void eval_mesh_3d_impl(double cutoff, int screen_type, double screen_length,
     nda::range rng, V1 & V, nda::stack_array<long,3> const& mesh,
     nda::stack_array<double,3,3> const& recv, nda::stack_array<double,3> const& k)
{
  long N = rng.size();
  if(N==0) return;
  auto V_d = to_cuda_std_mdspan(V);
  auto F = pots::detail::eval_mesh_3d_impl<decltype(V_d)>{rng.first(),cutoff,screen_type,
    screen_length, V_d,to_cuda_std_array<3>(mesh),
    to_cuda_std_array<9>(recv), to_cuda_std_array<3>(k)};
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}

template<typename V1, typename V2>
void eval_2d_impl(double tpitz, double cutoff, int screen_type, double screen_length,
     nda::range rng, V1 & V, V2 const& gv, nda::stack_array<double,3> const& k)
{
  long N = rng.size();
  if(N==0) return;
  auto V_d = to_cuda_std_mdspan(V);
  auto gv_d = to_cuda_std_mdspan(gv);
  auto F = pots::detail::eval_2d_impl<decltype(V_d),decltype(gv_d)>{rng.first(),cutoff,screen_type,
    screen_length, tpitz, V_d, gv_d, to_cuda_std_array<3>(k)}; 
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}

template<typename V1, typename V2>
void eval_3d_impl(double cutoff, int screen_type, double screen_length,
     nda::range rng, V1 & V, V2 const& gv, nda::stack_array<double,3> const& k) 
{
  long N = rng.size();
  if(N==0) return;
  auto V_d = to_cuda_std_mdspan(V);
  auto gv_d = to_cuda_std_mdspan(gv);
  auto F = pots::detail::eval_3d_impl<decltype(V_d),decltype(gv_d)>{rng.first(),cutoff,screen_type,
    screen_length, V_d, gv_d, to_cuda_std_array<3>(k)}; 
  cub::DeviceFor::Bulk(N,F);
  arch::synchronize_if_set();
}

using memory::device_array_view;
using memory::unified_array_view;
using std::complex;

template<int Rank>
using basic_layout_t = typename nda::basic_layout<0, nda::C_stride_order<Rank>, nda::layout_prop_e::none>;

#define _impl_(T,Arr)  \
template void eval_mesh_2d_impl(double,double,int,double,nda::range,Arr<T,1,basic_layout_t<1>>&, \
  nda::stack_array<long,3>const&,nda::stack_array<double,3,3>const&, \
  nda::stack_array<double,3>const&); \
template void eval_mesh_2d_impl(double,double,int,double,nda::range,Arr<complex<T>,1,basic_layout_t<1>>&, \
  nda::stack_array<long,3>const&,nda::stack_array<double,3,3>const&, \
  nda::stack_array<double,3>const&); \
template void eval_mesh_3d_impl(double,int,double,nda::range,Arr<T,1,basic_layout_t<1>>&, \
  nda::stack_array<long,3>const&,nda::stack_array<double,3,3>const&, \
  nda::stack_array<double,3>const&); \
template void eval_mesh_3d_impl(double,int,double,nda::range,Arr<complex<T>,1,basic_layout_t<1>>&, \
  nda::stack_array<long,3>const&,nda::stack_array<double,3,3>const&, \
  nda::stack_array<double,3>const&); \
template void eval_2d_impl(double,double,int,double,nda::range rng, \
  Arr<T,1,basic_layout_t<1>>&,Arr<const double,2,basic_layout_t<2>> const&,nda::stack_array<double,3> const&); \
template void eval_2d_impl(double,double,int,double,nda::range rng, \
  Arr<complex<T>,1,basic_layout_t<1>>&,Arr<const double,2,basic_layout_t<2>> const&,nda::stack_array<double,3>const&); \
template void eval_3d_impl(double,int,double,nda::range rng, \
  Arr<T,1,basic_layout_t<1>>&,Arr<const double,2,basic_layout_t<2>> const&,nda::stack_array<double,3>const&); \
template void eval_3d_impl(double,int,double,nda::range rng, \
  Arr<complex<T>,1,basic_layout_t<1>>&,Arr<const double,2,basic_layout_t<2>> const&,nda::stack_array<double,3>const&); 

_impl_(double,device_array_view)
_impl_(double,unified_array_view)

} // namespace kernels::device::detail

