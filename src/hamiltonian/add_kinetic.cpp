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




#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "grids/g_grids.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "hamiltonian/potentials.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"

namespace hamilt
{

/*
 * Adds the kinetic contribution to hpsi:   hpsi(s,k,a,G) += 0.5*|G+k|^2 * psi(s,k,b,G) 
 * Assumes psi and hpsi are in the 'w' grid.
 */
void add_kinetic(int npol, nda::ArrayOfRank<2> auto const& kpts,
                 grids::truncated_g_grid const& wfc_g,
                 math::nda::DistributedArrayOfRank<4> auto const& psi,
                 math::nda::DistributedArrayOfRank<4> auto & hpsi)
{
  using value_t = typename std::decay_t<decltype(psi)>::value_type;
  static_assert( std::is_same_v<typename std::decay_t<decltype(hpsi)>::value_type,value_t>, "Type mismatch.");
  decltype(nda::range::all) all;
  auto k_range_loc = psi.local_range(1);  
  auto g_range = psi.local_range(3);
  long p0 = g_range.first();
  long ng = wfc_g.size();
  utils::check(g_range.first() >= 0 and g_range.last() <= npol*wfc_g.size(),"Shape mismatch.");
  // MAM: Not allowing a range in the pw axis, otherwise I need to know the origin
  utils::check(psi.global_shape()[3] == npol*wfc_g.size(),"Shape mismatch.");
  utils::check(psi.global_shape()[1] == kpts.extent(0),"Shape mismatch.");
  utils::check(kpts.extent(1) == 3,"Shape mismatch.");
  utils::check(psi.global_shape() == hpsi.global_shape() ,"Shape mismatch.");
  utils::check(psi.local_shape() == hpsi.local_shape() ,"Shape mismatch.");
  utils::check(psi.origin() == hpsi.origin() ,"Shape mismatch.");

  auto kpts_loc = kpts(k_range_loc,all);

  // utility
  auto range_overlap = [](nda::range rng1, nda::range rng2) {
          if(rng1.last() <= rng2.first() or rng1.first() >= rng2.last()) return nda::range(0);
          return nda::range( std::max(rng1.first(),rng2.first()),
                             std::min(rng1.last(),rng2.last()) );
        };

  auto ploc = psi.local();
  auto hloc = hpsi.local();  
  if constexpr(nda::mem::on_host<decltype(ploc),decltype(hloc)>) {
    nda::array<ComplexType,1> g2( std::min(ng,g_range.size()) );
    for( auto ip : nda::range(npol) ) { 
      auto p_rng = range_overlap(nda::range{ip*ng,(ip+1)*ng},g_range);    
      if(p_rng.size() == 0) continue;
      auto g2p = g2(nda::range(p_rng.size()));
      for( auto [ik,k] : itertools::enumerate(psi.local_range(1)) ) {
        utils::g2kin(kpts_loc(ik,all),wfc_g.g_vectors()(p_rng+(-ip*ng),all),g2p);
        for( auto [is,s] : itertools::enumerate(psi.local_range(0)) )
          for( auto [ia,a] : itertools::enumerate(psi.local_range(2)) )
            hloc(is,ik,ia,p_rng+(-p0)) += ploc(is,ik,ia,p_rng+(-p0))*g2p(); 
      } 
    }  
  }  
  else 
  {
    for( auto ip : nda::range(npol) ) { 
      auto p_rng = range_overlap(nda::range{ip*ng,(ip+1)*ng},g_range);    
      if(p_rng.size() == 0) continue;
      memory::unified_array<ComplexType,2> g2(k_range_loc.size(),p_rng.size());
      utils::g2kin(kpts_loc,wfc_g.g_vectors()(p_rng+(-ip*ng),all),g2);
      // hloc(is,ik,ia,G) += g2(ik,G) * psi(is,ik,ia,G)
      nda::tensor::contract(value_t{1.0},ploc(all,all,all,p_rng+(-p0)),"skaG",g2,"kG",
                            value_t{1.0},hloc(all,all,all,p_rng+(-p0)),"skaG");
    }
  }
}

using memory::darray_t;
using memory::host_array;
using memory::host_array_view;
using boost::mpi3::communicator;

// add_kinetic 'w' grid
template void add_kinetic(int,host_array_view<const double,2> const&, 
        grids::truncated_g_grid const&,
        darray_t<host_array<ComplexType,4>,communicator> const&, 
        darray_t<host_array<ComplexType,4>,communicator>&);
#if defined(ENABLE_DEVICE)
using memory::device_array;
using memory::unified_array;
using memory::device_array_view;
using memory::unified_array_view;

template void add_kinetic(int,host_array_view<const double,2> const&,
        grids::truncated_g_grid const&,
        darray_t<device_array<ComplexType,4>,communicator> const&,
        darray_t<device_array<ComplexType,4>,communicator>&);
template void add_kinetic(int,host_array_view<const double,2> const&,
        grids::truncated_g_grid const&,
        darray_t<unified_array<ComplexType,4>,communicator> const&,
        darray_t<unified_array<ComplexType,4>,communicator>&);
#endif


}

