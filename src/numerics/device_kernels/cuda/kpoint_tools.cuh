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

#ifndef KPOINT_TOOLS_CUDA_KERNELS_HPP
#define KPOINT_TOOLS_CUDA_KERNELS_HPP

#include <complex>
#include "nda/nda.hpp"
#include "nda/tensor.hpp"

namespace kernels::device
{

namespace detail
{

template<typename V1>
void rspace_phase_factor(nda::stack_array<double,3,3> const& lattv,
                         nda::stack_array<double,3> const& G,
                         nda::stack_array<long,3> const& mesh,
                         nda::range rng, V1& f);

template<typename V1, typename V2>
void rspace_phase_factor(nda::stack_array<double,3> const& G,
                         nda::stack_array<long,3> const& mesh,
                         V1 const& rp, V2& f);

}  // detail



void rspace_phase_factor(nda::ArrayOfRank<2> auto const& lattv,
                         nda::ArrayOfRank<1> auto const& G,
                         nda::ArrayOfRank<1> auto const& mesh,
                         nda::range rng,
                         nda::MemoryArrayOfRank<1> auto&& f)
{
  auto f_b = to_basic_layout(f());
  detail::rspace_phase_factor(nda::stack_array<double,3,3>{lattv},
                              nda::stack_array<double,3>{G},
                              nda::stack_array<long,3>{mesh},rng,f_b);
}

void rspace_phase_factor(nda::ArrayOfRank<1> auto const& G,
                         nda::ArrayOfRank<1> auto const& mesh,
                         nda::MemoryArrayOfRank<1> auto const& rp,
                         nda::MemoryArrayOfRank<1> auto&& f)
{
  auto f_b = to_basic_layout(f());
  auto rp_b = to_basic_layout(rp());
  detail::rspace_phase_factor(nda::stack_array<double,3>{G},
                              nda::stack_array<long,3>{mesh},rp_b,f_b);
}

template<nda::MemoryArrayOfRank<3> Arr>
void rspace_phase_factor(nda::ArrayOfRank<2> auto const& lattv,
                         nda::ArrayOfRank<1> auto const& G,
                         Arr&& f)
{
  constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(f())>>();
  nda::stack_array<long,3> mesh = {f.extent(0),f.extent(1),f.extent(2)};
  nda::range rng(f.size());
  if (f.is_stride_order_C() and f.indexmap().is_contiguous()) {
    auto f1d = nda::reshape(f,std::array<long,1>{f.size()});
    rspace_phase_factor(lattv,G,mesh,rng,f1d);
  } else {
    // make contiguous array in device, compute and copy
    using T = nda::get_value_t<Arr>;
    memory::array<MEM, T, 3> f_c(f.shape());
    auto f1d = nda::reshape(f_c,std::array<long,1>{f_c.size()});
    rspace_phase_factor(lattv,G,mesh,rng,f1d);
    nda::tensor::add(T(1.0),f_c,"ijk",T(0.0),f,"ijk"); 
  }
}


} // namespace kernels::device

#endif
