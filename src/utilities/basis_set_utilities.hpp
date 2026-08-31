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


#ifndef __UTILITIES_BASIS_SET_UTILITIES_HPP__
#define __UTILITIES_BASIS_SET_UTILITIES_HPP__

#include <math.h>
#include <cmath>
#include "boost/math/special_functions/bessel.hpp"

#include "configuration.hpp"
#include "utilities/integration.hpp"
#include "nda/nda.hpp"

namespace utils
{

// following upflib implementation
template<typename T>
void spherical_bessel_function(int l, T const q, long N, T const* ri, T * jl)  
{
  utils::check(l>=0 and l<=5, "spherical_bessel_function: Invalid l:{}",l);
  if(std::abs(q) < 1e-14) {
    if(l==0) {
      for(long i=0; i<N; ++i) 
        jl[i] = T(1);
    } else {
      for(long i=0; i<N; ++i) 
        jl[i] = T(0);
    }
    return;
  }

  long i0(N);
  T semifact = T(1);
  for( int i = 2*l+1; i>=1; i-=2 ) 
    semifact = T(i)*semifact;
  for(long i=0; i<N; ++i) {

    if( std::abs(q*ri[i]) <= 0.05 ) {

      T x = q*ri[i], xl;
      double x2 = double(x*x);
      if( l == 0 ) { 
        xl = T(1.0);
      } else {
        xl = std::pow(x,l);
      }
      jl[i] = xl/semifact * 
                T( 1.0 - x2/1.0/2.0/double(2*l+3) *
                ( 1.0 - x2/2.0/2.0/double(2*l+5) * 
                ( 1.0 - x2/3.0/2.0/double(2*l+7) * 
                ( 1.0 - x2/4.0/2.0/double(2*l+9) ) ) ) );

    } else {
      i0 = i;
      break;
    }

  }

  if(i0 >= N ) return;

  if(l==0) {
    for(int i = i0; i<N; ++i) { 
      double x = q*ri[i];
      jl[i] = T(std::sin(x)/x);
    }
  } else if(l==1) {
    for(int i = i0; i<N; ++i) {
      double x = q*ri[i];
      jl[i] = T( (std::sin(x)/x - std::cos(x))/x );
    }
  } else if(l==2) {
    for(int i = i0; i<N; ++i) {
      double x = q*ri[i];
      jl[i] = T( ((3.0/x-x)*std::sin(x) - 3.0*std::cos(x))/(x*x) );
    }
  } else if(l==3) {
    for(int i = i0; i<N; ++i) {
      double x = q*ri[i];
      jl[i] = T( (std::sin(x)*(15.0/x -6.0*x) + std::cos(x)*(x*x-15.0))/(x*x*x) );
    }
  } else if(l==4) {
    for(int i = i0; i<N; ++i) {
      double x = q*ri[i];
      jl[i] = T( (std::sin(x)*(105.0 - 45.0*x*x + std::pow(x,4)) + 
                  std::cos(x)*(10.0*x*x*x-105.0*x))/(std::pow(x,5)) );
    }
  } else if(l==5) {
    for(int i = i0; i<N; ++i) {
      double x = q*ri[i];
      double si = std::sin(x);
      double ci = std::cos(x);
      jl[i] = T( (-ci -
                  (945.0*ci) / std::pow(x,4) + 
                  (105.0*ci) / std::pow(x,2) + 
                  (945.0*si) / std::pow(x,5) - 
                  (420.0*si) / std::pow(x,3) + 
                  (15.0*si / x) / x ) ); 
    }
  }
}

/*
 * Computes the spherical bessel transform of a function for all values of l up to lmax. 
 * F[l] = int_0_infty g(r) * sph_bessel(l,q*r) * r * r 
 */
template<typename T, typename range_t, typename grid_t, typename func_t>
void sph_bessel_transform_boost(range_t l_rng, T q, grid_t && r, func_t && g, 
                          ::nda::MemoryArrayOfRank<1> auto && F)
{
  utils::check(F.size() >= l_rng.size(), 
               "sph_bessel_transform: Size mismatch - F.size():{}, l.size():{}",
               F.size(),l_rng.size()); 

  for( auto [i,l] : itertools::enumerate(l_rng) ) {
    unsigned int li(l);
// not available in Apple's libc++, also in libstdc++, so use boost's for now 
//    using std::sph_bessel;
    using boost::math::sph_bessel;
    auto fun = [&](T ri) {
      double qr = q*ri;
      return g(ri) * sph_bessel(li,qr) * ri * ri;
    };
    F(i) = utils::simpson_rule_f<T>(r,fun);
  }
}

template<typename T, typename range_t, typename grid_t, typename func_t>
void sph_bessel_transform(range_t l_rng, T q, grid_t && r, func_t && g,
                          ::nda::MemoryArrayOfRank<1> auto && F)
{
  utils::check(F.size() >= l_rng.size(),
               "sph_bessel_transform: Size mismatch - F.size():{}, l.size():{}",
               F.size(),l_rng.size());

  for( auto [i,l] : itertools::enumerate(l_rng) ) {
    unsigned int li(l);
    auto fun = [&](T ri) {
      double res = 0.0, r_ = double(ri);
      spherical_bessel_function(li,q,1,std::addressof(r_),std::addressof(res));
      return g(ri) * res * ri * ri;
    };
    F(i) = utils::simpson_rule_f<T>(r,fun);
  }
}

template<typename T, typename range_t>
void sph_bessel_transform(range_t l_rng, T q, ::nda::MemoryArrayOfRank<1> auto const& r,
    ::nda::MemoryArrayOfRank<1> auto const& dr, ::nda::MemoryArrayOfRank<1> auto const& fr,
    ::nda::MemoryArrayOfRank<1> auto && w, ::nda::MemoryArrayOfRank<1> auto && F)
{
  utils::check(F.size() >= l_rng.size(),
               "sph_bessel_transform: Size mismatch - F.size():{}, l.size():{}",
               F.size(),l_rng.size());
  utils::check(r.size() == dr.size(), "sph_bessel_transform: Size mismatch"); 
  utils::check(r.size() == fr.size(), "sph_bessel_transform: Size mismatch"); 
  utils::check(r.size() == w.size(), "sph_bessel_transform: Size mismatch"); 

  for( auto [i,l] : itertools::enumerate(l_rng) ) {
    unsigned int li(l);
    spherical_bessel_function(li,q,r.size(),r.data(),w.data());
    w() *=  (fr() * r() * r());
    F(i) = utils::simpson_rule_array<T>(dr,w);
  }
}


template<typename T, typename range_t, typename grid_t, typename func_t>
auto sph_bessel_transform(range_t l_rng, T q, grid_t && r, func_t && g)
{ 
  ::nda::array<T,1> F(l_rng.size());
  sph_bessel_transform<T>(l_rng,q,r,g,F);
  return F;
}

}

#endif
