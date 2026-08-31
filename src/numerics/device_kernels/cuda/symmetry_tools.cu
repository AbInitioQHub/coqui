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

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "cuda_runtime.h"
#include "utilities/type_traits.hpp"
#include "numerics/device_kernels/cuda/cuda_settings.h"
#include "numerics/device_kernels/cuda/cuda_aux.hpp"
#include "arch/CUDA/cuda_init.h"
#include "arch/arch.h"
#include "nda/nda.hpp"
#include <cuda/std/mdspan>
#include <cub/device/device_for.cuh>

#include "utilities/details/symmetry_utils.hpp"

namespace kernels::device
{

template<typename V1> 
void transform_k2g(bool trev, nda::stack_array<double, 3, 3> const& Rinv, 
                   nda::stack_array<double, 3> const& Gs, 
                   nda::stack_array<long, 3> const& mesh,
                   V1 &&k2g)
{ 
  int* err_d;
  int err = 0;
  cuda::cuda_check(cudaMalloc((void**)&err_d, sizeof(int)), "cudaMalloc");
  cuda::synchronize();
  auto k2g_d = to_cuda_std_mdspan(k2g);
  auto kernel = utils::detail::transform_k2g<decltype(k2g_d)>{(trev?-1.0:1.0),
            to_cuda_std_array<3>(mesh),to_cuda_std_array<3>(Gs),
            to_cuda_std_array<9>(Rinv),k2g_d,err_d};
  cub::DeviceFor::Bulk(k2g.extent(0),kernel);
  cuda::synchronize();
  cuda::cuda_check( cudaMemcpy(&err, err_d, sizeof(int), cudaMemcpyDefault), "CudaMemcpy" );
  cuda::cuda_check(cudaFree(err_d), "cudaFree");
  if( err > 0 ) 
    APP_ABORT(" Error in device::transform_k2g"); 
}

using nda::stack_array;
using nda::array_view;
using memory::device_array;
using memory::unified_array;
using memory::device_array_view;
using memory::unified_array_view;
using nda::C_layout;
using nda::C_stride_layout;

template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  device_array_view<long, 1> &); 
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  device_array_view<long, 1, C_layout> &); 
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  unified_array_view<long, 1> &); 
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  unified_array_view<long, 1, C_layout> &); 
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  device_array_view<long, 1> &&);
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  device_array_view<long, 1, C_layout> &&);
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  unified_array_view<long, 1> &&);
template void transform_k2g(bool,nda::stack_array<double, 3, 3> const&,
  nda::stack_array<double, 3> const&, nda::stack_array<long, 3> const&,
  unified_array_view<long, 1, C_layout> &&);
                                  



} // namespace kernels::device

