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

#ifndef POTENTIALS_CUDA_KERNELS_HPP
#define POTENTIALS_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"
#include "numerics/device_kernels/cuda/nda_aux.hpp"

namespace kernels::device
{

namespace detail
{

template<typename V1>
void eval_mesh_2d_impl(double tpitz, double cutoff, int screen_type, double screen_length, 
     nda::range rng, V1 & V, nda::stack_array<long,3> const& mesh,
     nda::stack_array<double,3,3> const& recv, nda::stack_array<double,3> const& k);

template<typename V1>
void eval_mesh_3d_impl(double cutoff, int screen_type, double screen_length, 
     nda::range rng, V1 & V, nda::stack_array<long,3> const& mesh,
     nda::stack_array<double,3,3> const& recv, nda::stack_array<double,3> const& k);

template<typename V1, typename V2>
void eval_2d_impl(double tpitz, double cutoff, int screen_type, double screen_length,    
     nda::range rng, V1 & V, V2 const& gv, nda::stack_array<double,3> const& k);

template<typename V1, typename V2>
void eval_3d_impl(double cutoff, int screen_type, double screen_length,
     nda::range rng, V1 & V, V2 const& gv, nda::stack_array<double,3> const& k);

}

/*
 * Evaluates vG[g,Q] on the fft grid
 */
void eval_mesh_2d(double cutoff, int screen_type, double screen_length,
                nda::range g_rng,
                nda::MemoryArrayOfRank<1> auto&& V,
                nda::stack_array<long,3> const& mesh,
                nda::stack_array<double,3,3> const& lattv,
                nda::stack_array<double,3,3> const& recv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
{
  auto V_b = to_basic_layout(V()); 
  auto k = kp-kq;
  const double tpitz = 2.0*3.14159265358979323846*lattv(2,2);
  detail::eval_mesh_2d_impl(tpitz,cutoff,screen_type,screen_length,g_rng,V_b,
       mesh,recv,nda::stack_array<double,3>(k));
}

void eval_mesh_3d(double cutoff, int screen_type, double screen_length,
                nda::range g_rng,
                nda::MemoryArrayOfRank<1> auto&& V,
                nda::stack_array<long,3> const& mesh,
                nda::stack_array<double,3,3> const& recv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
{
  auto V_b = to_basic_layout(V());
  auto k = kp-kq;
  detail::eval_mesh_3d_impl(cutoff,screen_type,screen_length,g_rng,V_b,
       mesh,recv,nda::stack_array<double,3>(k));
}

/*
 * Evaluates vG[g,Q] on a list of g-vectors
 */
void eval_2d(double cutoff, int screen_type, double screen_length,
                nda::range g_rng,
                nda::MemoryArrayOfRank<1> auto&& V,
                nda::MemoryArrayOfRank<2> auto const& gv,
                nda::stack_array<double,3,3> const& lattv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
{
  auto V_b = to_basic_layout(V());
  auto gv_b = to_basic_layout(gv());
  auto k = kp-kq;
  const double tpitz = 2.0*3.14159265358979323846*lattv(2,2);
  detail::eval_2d_impl(tpitz,cutoff,screen_type,screen_length,g_rng,V_b,gv_b,
       nda::stack_array<double,3>(k));
}

void eval_3d(double cutoff, int screen_type, double screen_length,
                nda::range g_rng,
                nda::MemoryArrayOfRank<1> auto&& V,
                nda::MemoryArrayOfRank<2> auto const& gv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
{
  auto V_b = to_basic_layout(V());
  auto gv_b = to_basic_layout(gv());
  auto k = kp-kq;
  detail::eval_3d_impl(cutoff,screen_type,screen_length,g_rng,V_b,gv_b,
       nda::stack_array<double,3>(k));
}

} // namespace kernels::device

#endif
