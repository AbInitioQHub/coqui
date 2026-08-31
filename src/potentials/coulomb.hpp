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


#ifndef COQUI_POTENTIALS_COULOMB_H
#define COQUI_POTENTIALS_COULOMB_H

#include <cmath>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "nda/nda.hpp"
#include "grids/g_grids.hpp"
#include "numerics/device_kernels/kernels.h"
#include "potentials/potentials_impl.hpp"

namespace pots
{

/*
 * Calculates the fourier transform of a (possibly screened) coulomb potential, v[G].
 * Parameters read from ptree:
 * - ndim = 3, Number of dimensions. choices: {2, 3}.
 * - cutoff = 1e-8, Cutoff for small-G limit. For |G|< cutoff, v[g] = 0.0;
 * - screen_type = "none". Screening type. Choices: {"none", "yukawa", "erfc", "erf", "tanh"}
 * - screen_length = 1.0. Screening length. Used depends on screen_type.
 *
 * In 3d:
 *   - "none":    v(G, kp, kq) = 4*pi / |G + kp - kq|^2
 *   - "yukawa":  v(G, kp, kq) = 4*pi / ( |G + kp - kq|^2 + screen_length )
 *   - "erfc":    v(G, kp, kq) = ( 4*pi / |G + kp - kq|^2 ) * ( 1 - exp( -|G + kp - kq|^2 / 4.0 / screen_length^2 ) )
 *   - "erf":     v(G, kp, kq) = 4*pi / |G + kp - kq|^2 * exp( -|G + kp - kq|^2 / 4.0 / screen_length^2 ) 
 *
 * In 2d:  (limited to lattices where a3 = (0,0,Lz))
 *   - "none":    v(G, kp, kq) = 2*pi*Lz / |G + kp - kq|
 *   - "tanh":    v(G, kp, kq) = 2*pi*Lz / |G + kp - kq| * tanh( |G + kp - kq| * screen_length ) 
 */
class coulomb_t
{
  
  public:

  coulomb_t(bool print_metadata = true):
    ndim(3), cutoff(1e-8), screen_type("none"), screen_length(0.0) 
  {
    if (print_metadata) print_meta();
  }

  coulomb_t(ptree const& pt, bool print_metadata = true):
    ndim( io::get_value_with_default<int>(pt,"ndim",3) ),
    cutoff( io::get_value_with_default<double>(pt,"cutoff",1e-8) ),
    screen_type( io::get_value_with_default<std::string>(pt,"screen_type","none") ),
    screen_length( io::get_value_with_default<double>(pt,"screen_length",1.0) )
  {
    utils::check( ndim==2 or ndim==3, "Error in coulomb_t: Invalid ndim:{}", ndim);
    utils::check( cutoff>=0.0, "Error in coulomb_t: Invalid cutoff:{}", cutoff);
    if(ndim==3) {
      utils::check( screen_type=="none" or screen_type=="yukawa" or screen_type=="erf" or screen_type=="erfc",
                    "Error in coulomb_t: Invalid screen_type:{}",screen_type); 
      if(screen_type == "none" or screen_type == "yukawa") {
        screen_type_id = 0;
      } else if(screen_type == "erfc") {
        screen_type_id = 1;
      } else if(screen_type == "erf") {
        screen_type_id = 2;
      }
    } else {
      utils::check( screen_type=="none" or screen_type=="tanh",
                    "Error in coulomb_t: Invalid screen_type:{}",screen_type); 
      if(screen_type == "none") {
        screen_type_id = 0;
      } else if(screen_type == "tanh") {
        screen_type_id = 1;
      }
    }
    if(screen_type == "none") screen_length = 0.0;
    if(print_metadata) print_meta();
  }
    
  void print_meta() const
  {
    app_log(2,"  Electron-electron interaction kernel");
    app_log(2,"  ------------------------------------");
    app_log(2,"  type          = coulomb");
    app_log(2,"  ndim          = {}",ndim);
    app_log(2,"  cutoff        = {}", cutoff);
    app_log(2,"  screen_type   = {}",screen_type);
    if(screen_type != "none")  app_log(2,"  screen_length = {}",screen_length);
    app_log(2,"");
  }

  void evaluate(nda::MemoryArrayOfRank<1> auto&& V,
                nda::stack_array<double,3,3> const& lattv,
                nda::MemoryArrayOfRank<2> auto const& gv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
  {
    constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(V())>>();
    utils::check( V.shape()[0] >= gv.shape()[0], "coulomb_t::evaluate - Dimension mismatch.");
    utils::check( gv.shape()[1] == 3,"coulomb_t::evaluate - size mismatch: {},{}", gv.shape()[1],3);
    utils::check( kp.shape()[0] == 3,"coulomb_t::evaluate - size mismatch: {},{}", kp.shape()[0],3);
    utils::check( kq.shape()[0] == 3,"coulomb_t::evaluate - size mismatch: {},{}", kq.shape()[0],3);
    
    if(ndim==2) {
      utils::check( std::abs(lattv(2,0))+std::abs(lattv(2,1)) < 1e-8,
                 "coulomb_t::evaluate - ndim=2 requires a3=(0,0,Lz). Found: ({},{},{})",
                 lattv(2,0),lattv(2,1),lattv(2,2) );
      if constexpr (MEM==HOST_MEMORY) {
        const double tpitz = 2.0*3.14159265358979323846*lattv(2,2);
        auto dk = kp-kq;
        auto F = pots::detail::eval_2d_impl<decltype(V()),decltype(gv())>(0l,cutoff,screen_type_id,screen_length,tpitz, 
            V(), gv(), nda::stack_array<double,3>{dk});
        std::ranges::for_each(nda::range(V.extent(0)),F);
      } else {
#if defined(ENABLE_CUDA)
        kernels::device::eval_2d(cutoff,screen_type_id,screen_length,nda::range(V.extent(0)),V,gv,lattv,kp,kq);
#else
        static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
      }

    } else if(ndim==3) {
       if constexpr (MEM==HOST_MEMORY) {
        auto dk = kp-kq;
        auto F = pots::detail::eval_3d_impl<decltype(V()),decltype(gv())>(0l,cutoff,screen_type_id,screen_length,
            V(), gv(), nda::stack_array<double,3>{dk});
        std::ranges::for_each(nda::range(V.extent(0)),F);
      } else {
#if defined(ENABLE_CUDA)
        kernels::device::eval_3d(cutoff,screen_type_id,screen_length,nda::range(V.extent(0)),V,gv,kp,kq);
#else
        static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
      }
    } 
  }

  void evaluate_in_mesh(nda::range g_rng, nda::MemoryArrayOfRank<1> auto&& V,
              nda::stack_array<long,3> const& mesh,
              nda::stack_array<double,3,3> const& lattv,
              nda::stack_array<double,3,3> const& recv,
              nda::ArrayOfRank<1> auto const& kp,
              nda::ArrayOfRank<1> auto const& kq)
  {
    constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(V())>>();
    utils::check( g_rng.first() >= 0 and g_rng.first() <= g_rng.last() and g_rng.last() <= V.extent(0),
                  "coulomb_t::evaluate - range mismatch"); 
    utils::check( mesh.extent(0) == 3,"coulomb_t::evaluate - size mismatch: {},{}", mesh.extent(0),3);
    utils::check( lattv.extent(0) == 3 and lattv.extent(1) == 3,
                  "coulomb_t::evaluate - shape mismatch: ({},{}),({},{})", lattv.extent(0),lattv.extent(1),3,3); 
    utils::check( recv.extent(0) == 3 and recv.extent(1) == 3,
                  "coulomb_t::evaluate - shape mismatch: ({},{}),({},{})", recv.extent(0),recv.extent(1),3,3); 
    utils::check( kp.extent(0) == 3,"coulomb_t::evaluate - size mismatch: {},{}", kp.extent(0),3);
    utils::check( kq.extent(0) == 3,"coulomb_t::evaluate - size mismatch: {},{}", kp.extent(0),3);
    
    if(ndim==2) {
      utils::check( std::abs(lattv(2,0))+std::abs(lattv(2,1)) < 1e-8,
                 "coulomb_t::evaluate - ndim=2 requires a3=(0,0,Lz). Found: ({},{},{})",lattv(2,0),lattv(2,1),lattv(2,2) ); 
      if constexpr (MEM==HOST_MEMORY) {
        const double tpitz = 2.0*3.14159265358979323846*lattv(2,2);
        auto dk = kp-kq;
        auto F = pots::detail::eval_mesh_2d_impl<decltype(V())>(0l,cutoff,screen_type_id,screen_length, tpitz, 
            V(), mesh, recv, nda::stack_array<double,3>{dk}); 
        std::ranges::for_each(g_rng,F);
      } else {
#if defined(ENABLE_CUDA)
        kernels::device::eval_mesh_2d(cutoff,screen_type_id,screen_length,g_rng,V,mesh,lattv,recv,kp,kq);
#else
        static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
      }
    } else if(ndim==3) {
      if constexpr (MEM==HOST_MEMORY) {
        auto dk = kp-kq;
        auto F = pots::detail::eval_mesh_3d_impl<decltype(V())>(0l,cutoff,screen_type_id,screen_length, 
            V(), mesh, recv, nda::stack_array<double,3>{dk});
        std::ranges::for_each(g_rng,F);
      } else {
#if defined(ENABLE_CUDA)
        kernels::device::eval_mesh_3d(cutoff,screen_type_id,screen_length,g_rng,V,mesh,recv,kp,kq);
#else
        static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
      }
    }
  }

  private:

  // # of dimensions, only 2 or 3 allowed. Defines the shape of the potential 
  int ndim = 3;

  // truncate small G 
  double cutoff = 1e-8; 

  // type of screening
  std::string screen_type = "none";

  // integer code to screen_type for easier dispatching
  int screen_type_id = 0;

  // screening length parameter
  double screen_length = 1.0;

}; 

}

#endif
