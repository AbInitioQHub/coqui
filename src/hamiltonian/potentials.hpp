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


#ifndef UTILITIES_POTENTIALS_HPP
#define UTILITIES_POTENTIALS_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "grids/g_grids.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "nda/nda.hpp"

namespace hamilt 
{

/**
 * Calculate Fourier transform of a given potential type.
 * For coulomb type: V(G, k_{p}, k_{q}) = 4*pi / |G + k_{p} - k_{q}|^{2}
 * @param V - [OUTPUT] potential
 * @param g - [INPUT] list of gvectors 
 * @param kp - [INPUT] k_{p}
 * @param kq - [INPUT] k_{q}
 * @param type - [INPUT] potential type
 */
template<nda::MemoryArrayOfRank<1> Arr, 
         class Vec = std::array<double,1>
        >
void potential_g(Arr&& V, 
		nda::MemoryArrayOfRank<2> auto const& gv,
		nda::ArrayOfRank<1> auto const& kp,
		nda::ArrayOfRank<1> auto const& kq,
		std::string type = "coulomb",
		[[maybe_unused]] Vec const& params = {})
{
  int ngm = gv.shape()[0];
  const double fpi = 4.0*3.14159265358979323846;
  using T = typename std::decay_t<Arr>::value_type;
  utils::check( V.shape()[0] >= ngm, "Dimension mismatch.");
  utils::check( gv.shape()[1] == 3,"potential_g - size mismatch: {},{}", gv.shape()[1],3);    
  utils::check( kp.shape()[0] == 3,"potential_g - size mismatch: {},{}", kp.shape()[0],3);    
  utils::check( kq.shape()[0] == 3,"potential_g - size mismatch: {},{}", kq.shape()[0],3);    

  if(type == "coulomb") {
    for( int n=0; n<ngm; ++n ) { 
      double gx = gv(n, 0) + kp(0) - kq(0);
      double gy = gv(n, 1) + kp(1) - kq(1);
      double gz = gv(n, 2) + kp(2) - kq(2);
      double g2 = gx*gx + gy*gy + gz*gz;
      if( g2 > 1e-8 ) 
        V(n) =  T(fpi/g2);
      else 
        V(n) = T(0.0);   
    }
  } else {
    APP_ABORT("Error: Unknown type in coulomb::evaluate.");
  }
} 

template<nda::MemoryArrayOfRank<1> Arr,
         class Vec = std::array<double,1>
        >
void potential_full_g(Arr&& V,
		nda::ArrayOfRank<1> auto const& mesh,
		nda::ArrayOfRank<2> auto const& recv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq,
                std::string type = "coulomb",
                [[maybe_unused]] Vec const& params = {})
{ 
  int ngm = mesh(0)*mesh(1)*mesh(2); 
  const double fpi = 4.0*3.14159265358979323846;
  using T = typename std::decay_t<Arr>::value_type;
  utils::check( ngm>0, "Dimension mismatch.");
  utils::check( V.shape()[0] >= ngm, "Dimension mismatch.");
  utils::check( kp.shape()[0] == 3,"potential_g - size mismatch: {},{}", kp.shape()[0],3);
  utils::check( kq.shape()[0] == 3,"potential_g - size mismatch: {},{}", kq.shape()[0],3);
  
  if(type == "coulomb") {
    long ni = (mesh(0)-1)/2;
    long nj = (mesh(1)-1)/2;
    long nk = (mesh(2)-1)/2;
    for(long i = -ni; i < ni; ++i ) 
    for(long j = -nj; j < nj; ++j )
    for(long k = -nk; k < nk; ++k ) {
      long ii = (i < 0 ? i+mesh(0) : i);
      long ij = (j < 0 ? j+mesh(1) : j);
      long ik = (k < 0 ? k+mesh(2) : k);
      long N = (ii*mesh(1) + ij)*mesh(2) + ik;
      double gx = double(i)*recv(0,0)+double(j)*recv(1,0)+double(k)*recv(2,0) + kp(0) - kq(0);
      double gy = double(i)*recv(0,1)+double(j)*recv(1,1)+double(k)*recv(2,1) + kp(1) - kq(1);
      double gz = double(i)*recv(0,2)+double(j)*recv(1,2)+double(k)*recv(2,2) + kp(2) - kq(2);
      double g2 = gx*gx + gy*gy + gz*gz;
      if( g2 > 1e-8 ) 
        V(N) =  T(fpi/g2);
      else 
        V(N) = T(0.0);
    }
  } else {
    APP_ABORT("Error: Unknown type in coulomb::evaluate.");
  }
}

} // utils 

#endif

