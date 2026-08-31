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


#ifndef GRIDS_GRID_UTILS_HPP
#define GRIDS_GRID_UTILS_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "nda/nda.hpp"
#include "itertools/itertools.hpp"
#include "utilities/concepts.hpp"
#include "utilities/symmetry.hpp"
#include "utilities/mpi_context.h"

namespace grids
{

/* counts the number of G vectors with kinetic energy below ecut */
long get_ngm(double ecut,
            nda::ArrayOfRank<1> auto const& fft_mesh,
            nda::ArrayOfRank<2> auto const& recv) {
  long cnt=0;
  long ni = fft_mesh(0)/2;
  long nj = fft_mesh(1)/2;
  long nk = fft_mesh(2)/2;
  for(long i = (ni-fft_mesh(0)+1); i <= ni; ++i )
  for(long j = (nj-fft_mesh(1)+1); j <= nj; ++j )
  for(long k = (nk-fft_mesh(2)+1); k <= nk; ++k ) {
    double gx = double(i)*recv(0,0)+double(j)*recv(1,0)+double(k)*recv(2,0);
    double gy = double(i)*recv(0,1)+double(j)*recv(1,1)+double(k)*recv(2,1);
    double gz = double(i)*recv(0,2)+double(j)*recv(1,2)+double(k)*recv(2,2);
    if( gx*gx+gy*gy+gz*gz <= 2.0*ecut ) cnt++;
  }
  return cnt;
}

namespace detail
{

// These routines can be drastically improved!!!
template<bool full>
auto check_boundary(nda::array<int,1> &n, double ecut, nda::ArrayOfRank<2> auto const& recv)
{
  using itertools::enumerate;
  using itertools::range;
  decltype(nda::range::all) all;
  int nmax = std::max(n(1),n(2)); 
  nda::array<double, 2> r(2*nmax,3);
  nda::array<double, 2> g(2*nmax,3);
  nda::array<bool,1> which(3,false);
  {
    double i0 = double(n(0)/2 - n(0) + 1); 
    double i1 = double(n(0)/2); 
    int p0 = n(2);
    bool found = false;
    for(int j=n(1)/2-n(1)+1; j<=n(1)/2; ++j) {
      for( auto [p,k] : enumerate(range(n(2)/2-n(2)+1,n(2)/2+1)) ) { 
        r(p,0) = i0;
        r(p0+p,0) = i1;
        r(p,1) = r(p0+p,1) = double(j);
        r(p,2) = r(p0+p,2) = double(k);
      }
      nda::blas::gemm(double(1.0),r(range(2*n(2)),all),recv,double(0.0),g(range(2*n(2)),all));
      auto it = g.data();
      for( int p = 0; p<2*p0; ++p, it+=3 ) 
        if( ecut >=  (*it)*(*it) + (*(it+1))*(*(it+1)) + (*(it+2))*(*(it+2)) ) {
          found=true;
          break; 
        } 
      if(found) break;
    }
    if constexpr (full) {
      which(0) = found;
    } else {
      if(found) return 1;
    } 
  }
  { 
    double j0 = double(n(1)/2 - n(1) + 1);
    double j1 = double(n(1)/2); 
    int p0 = n(2);
    bool found = false;
    for(int i=n(0)/2-n(0)+1; i<=n(0)/2; ++i) {
      for( auto [p,k] : enumerate(range(n(2)/2-n(2)+1,n(2)/2+1)) ) {
        r(p,0) = r(p0+p,0) = double(i);
        r(p,1) = j0;
        r(p0+p,1) = j1;
        r(p,2) = r(p0+p,2) = double(k);
      }
      nda::blas::gemm(double(1.0),r(range(2*n(2)),all),recv,double(0.0),g(range(2*n(2)),all));
      auto it = g.data();
      for( int p = 0; p<2*p0; ++p, it+=3 ) 
        if( ecut >=  (*it)*(*it) + (*(it+1))*(*(it+1)) + (*(it+2))*(*(it+2)) ) {
          found=true;
          break;
        }
      if(found) break;
    }
    if constexpr (full) {
      which(1) = found;
    } else {
      if(found) return 2;
    } 
  }
  { 
    double k0 = double(n(2)/2 - n(2) + 1);
    double k1 = double(n(2)/2); 
    int p0 = n(1);
    bool found = false;
    for(int i=n(0)/2-n(0)+1; i<=n(0)/2; ++i) {
      for( auto [p,j] : enumerate(range(n(1)/2-n(1)+1,n(1)/2+1)) ) {
        r(p,0) = r(p0+p,0) = double(i);
        r(p,1) = r(p0+p,1) = double(j);
        r(p,2) = k0;
        r(p0+p,2) = k1;
      }
      nda::blas::gemm(double(1.0),r(range(2*n(1)),all),recv,double(0.0),g(range(2*n(1)),all));
      auto it = g.data();
      for( int p = 0; p<2*p0; ++p, it+=3 ) 
        if( ecut >=  (*it)*(*it) + (*(it+1))*(*(it+1)) + (*(it+2))*(*(it+2)) ) {
          found=true;
          break;
        }
      if(found) break;
    }
    if constexpr (full) {
      which(2) = found;
    } else {
      if(found) return 3;
    }
  }
  if constexpr (full)
    return which;
  else
    return 0; 
}


void adjust_in(nda::array<int,1> &mesh, double ecut, nda::ArrayOfRank<2> auto const& recv)
{
  mesh -= 2;
  while( mesh(0) > 0 and mesh(1) > 0 and  mesh(2) > 0 ) {
    auto which = detail::check_boundary<true>(mesh,ecut,recv); 
    if( which(0) and which(1) and which(2) ) return;
    for( int p=0; p<3; ++p ) 
      if(not which(p)) mesh(p)-=2;
  }
  APP_ABORT("Error: Problems finding fft grid for ecut:{}",ecut); 
}

// moves it out until it is outside, if already outside, does not touch it.
// for best smallest mesh, need to call adjust_in after in case any mesh point was originally too large
void adjust_out(nda::array<int,1> &mesh, double ecut, nda::ArrayOfRank<2> auto const& recv)
{ 
  auto which = detail::check_boundary<true>(mesh,ecut,recv);
  while( which(0) or which(1) or which(2) ) { 
    for( int p=0; p<3; ++p ) 
      if(which(p)) mesh(p)+=2;
    which = detail::check_boundary<true>(mesh,ecut,recv);
  }
}

auto fft_mesh_initial_guess(double ecut, nda::ArrayOfRank<2> auto const& recv)
{
  nda::array<int,1> mesh(3);
  nda::array<double,1> r(3), g(3);
  for(int i=0; i<3; ++i)
  {
    r() = 0;
    r(i) = 1.0;
    while( true ) {
      nda::blas::gemv(double(1.0),nda::transpose(recv),r,double(0.0),g);
      if( ecut <  g(0)*g(0)+g(1)*g(1)+g(2)*g(2) ) break; 
       r(i) += 1.0;
    }
    mesh(i) = int(std::round(r(i)));
  }  
  return mesh;
}

}

/* 
 * Given an ecut and recv array, determines the smallest fft mesh that contains all
 * g vectors with energy smaller or equal too ecut.
 * The grid is adjusted to make it consistent with a list of symmetries.
 */
auto find_fft_mesh(utils::Communicator auto&& comm, 
                   double ecut, 
                   nda::ArrayOfRank<2> auto const& recv,
                   std::vector<utils::symm_op> const& symm_list) 
{
  utils::check(recv.extent(0) == 3 and recv.extent(1) == 3, "Shape mismatch.");
  utils::check(ecut > 0.0, "ecut must be finite.");
  // include factor of 2 here
  ecut *= 2.0;
  nda::array<int,1> mesh = detail::fft_mesh_initial_guess(ecut,recv);
  // make grid odd
  if(mesh(0)%2==0) mesh(0)++;
  if(mesh(1)%2==0) mesh(1)++;
  if(mesh(2)%2==0) mesh(2)++;
  // serial for now
  if( comm.root() ) {
    int b = detail::check_boundary<false>(mesh,ecut,recv);
    if( b > 0 ) {
      // need to move outwards
      detail::adjust_out(mesh,ecut,recv);
      // adjust_in 
      detail::adjust_in(mesh,ecut,recv);
    } else {
      // need to move inwards     
      detail::adjust_in(mesh,ecut,recv);
    }
    mesh = utils::generate_consistent_fft_mesh(mesh,symm_list,1e-4,"find_fft_mesh");
    mesh = utils::generate_consistent_fft_mesh(mesh,symm_list,1e-4,"find_fft_mesh",true);
  }
  comm.broadcast_n(mesh.data(),3,0);
  return mesh;  
}

}

#endif
