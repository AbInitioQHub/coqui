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


#ifndef UTILITIES_HARMONICS_H
#define UTILITIES_HARMONICS_H

#include <array>
#include <cmath>
#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace utils
{

/* 
 * Class to compute spherical and solid harmonics.
 * Hard-coded implementation for small L.
 * Use sphericart for large L if needed.
 * Not optimized, used as a backup when sphericart is not available.
 */
template<typename T>
class harmonics
{
  public:

  harmonics() = default;
  ~harmonics() = default;

  harmonics(harmonics const&) = default; 
  harmonics(harmonics &&) = default; 

  harmonics& operator=(harmonics const&) = default; 
  harmonics& operator=(harmonics &&) = default; 


  /* Computes spherical harmonics for angular momentum L */ 
  void spherical_harmonics_l(int L, T const* r, long r_size, T* Ylm, long Ylm_size);

  void spherical_harmonics_l(int L, 
                           nda::ArrayOfRank<1> auto const& r, 
                           nda::ArrayOfRank<1> auto && Ylm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(Ylm)>>, "Type mismatch");
//    utils::check(r.size()>=3 and Ylm.size() >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    utils::check(r.size()>=3 and Ylm.size() >= (2*L+1), "Size mismatch.");
    spherical_harmonics_l(L,r.data(),3l,Ylm.data(),(2l*L+1l));
  }

  void spherical_harmonics_l(int L,
                           nda::ArrayOfRank<2> auto const& r,
                           nda::ArrayOfRank<2> auto && Ylm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(Ylm)>>, "Type mismatch");
    auto all = ::nda::range::all; 
    utils::check(r.extent(0) == Ylm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and Ylm.extent(1) >= (2*L+1), "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      spherical_harmonics_l(L,r(i,all),Ylm(i,all));
  }  

  /* Computes spherical harmonics for all angular momentum up to and including Lmax */  
  void spherical_harmonics(int Lmax, T const* r, long r_size, T* Ylm, long Ylm_size)
  {
    utils::check(r_size>=3 and Ylm_size >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    if(Lmax < 0) return;
    spherical_harmonics_l(0,r,3l,Ylm,1l);
    Ylm++;
    for(int l=1; l<Lmax+1; ++l) {
      spherical_harmonics_l(l,r,3l,Ylm,(2*l+1l));
      Ylm += (2*l+1l);
    }
  }

  void spherical_harmonics(int Lmax,
                           nda::ArrayOfRank<1> auto const& r,
                           nda::ArrayOfRank<1> auto && Ylm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(Ylm)>>, "Type mismatch");
    utils::check(r.size()>=3 and Ylm.size() >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    spherical_harmonics(Lmax,r.data(),3l,Ylm.data(),(Lmax+1)*(Lmax+1));
  }

  void spherical_harmonics(int Lmax,
                           nda::ArrayOfRank<2> auto const& r,
                           nda::ArrayOfRank<2> auto && Ylm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(Ylm)>>, "Type mismatch");
    auto all = ::nda::range::all;
    utils::check(r.extent(0) == Ylm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and Ylm.extent(1) >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      spherical_harmonics(Lmax,r(i,all),Ylm(i,all));
  }  


  /* Computes solid harmonics for angular momentum L */ 
  void solid_harmonics_l(int L, 
                         T const* r, long r_size, 
                         T* rlYlm, long rlYlm_size);

  void solid_harmonics_l(int L, 
                         nda::ArrayOfRank<1> auto const& r, 
                         nda::ArrayOfRank<1> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    utils::check(r.size()>=3 and rlYlm.size() >= 2*L+1, "Size mismatch.");
    solid_harmonics_l(L,r.data(),3l,rlYlm.data(),2l*L+1l);
  }

  void solid_harmonics_l(int L,
                         nda::ArrayOfRank<2> auto const& r,
                         nda::ArrayOfRank<2> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    auto all = ::nda::range::all; 
    utils::check(r.extent(0) == rlYlm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and rlYlm.extent(1) >= 2*L+1, "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      solid_harmonics_l(L,r(i,all),rlYlm(i,all));
  }

  /* Computes solid harmonics for all angular momentum up to and including Lmax */
  void solid_harmonics(int Lmax, T const* r, long r_size, T* rlYlm, long rlYlm_size)
  {
    utils::check(r_size>=3 and rlYlm_size >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    if(Lmax < 0) return;
    solid_harmonics_l(0,r,3l,rlYlm,1l);
    rlYlm++;
    for(int l=1; l<Lmax+1; ++l) {
      solid_harmonics_l(l,r,3l,rlYlm,(2*l+1l));
      rlYlm += (2*l+1l);
    }
  }

  void solid_harmonics(int Lmax,
                         nda::ArrayOfRank<1> auto const& r,
                         nda::ArrayOfRank<1> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    utils::check(r.size()>=3 and rlYlm.size() >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    solid_harmonics(Lmax,r.data(),3l,rlYlm.data(),(Lmax+1l)*(Lmax+1l));
  }

  void solid_harmonics(int Lmax,
                         nda::ArrayOfRank<2> auto const& r,
                         nda::ArrayOfRank<2> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    auto all = ::nda::range::all;
    utils::check(r.extent(0) == rlYlm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and rlYlm.extent(1) >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      solid_harmonics(Lmax,r(i,all),rlYlm(i,all));
  }

  /* Computes unnormalized solid harmonics for angular momentum L */ 
  void unnormalized_solid_harmonics_l(int L, 
                                      T const* r, long r_size, T* rlYlm, long rlYlm_size);

  void unnormalized_solid_harmonics_l(int L, 
                                      nda::ArrayOfRank<1> auto const& r, 
                                      nda::ArrayOfRank<1> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    utils::check(r.size()>=3 and rlYlm.size() >= 2*L+1, "Size mismatch.");
    unnormalized_solid_harmonics_l(L,r.data(),3l,rlYlm.data(),2l*L+1l);
  }

  void unnormalized_solid_harmonics_l(int L,
                         nda::ArrayOfRank<2> auto const& r,
                         nda::ArrayOfRank<2> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    auto all = ::nda::range::all; 
    utils::check(r.extent(0) == rlYlm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and rlYlm.extent(1) >= 2*L+1, "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      unnormalized_solid_harmonics_l(L,r(i,all),rlYlm(i,all));
  }

  /* Computes unnormalized solid harmonics for all angular momentum up to and including Lmax */
  void unnormalized_solid_harmonics(int Lmax, T const* r, long r_size, T* rlYlm, long rlYlm_size)
  {
    utils::check(r_size>=3 and rlYlm_size >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    if(Lmax < 0) return;
    unnormalized_solid_harmonics_l(0,r,3l,rlYlm,1l);
    rlYlm++;
    for(int l=1; l<Lmax+1; ++l) {
      unnormalized_solid_harmonics_l(l,r,3l,rlYlm,(2*l+1l));
      rlYlm += (2*l+1l);
    }
  }

  void unnormalized_solid_harmonics(int Lmax,
                         nda::ArrayOfRank<1> auto const& r,
                         nda::ArrayOfRank<1> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    utils::check(r.size()>=3 and rlYlm.size() >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    unnormalized_solid_harmonics(Lmax,r.data(),3l,rlYlm.data(),(Lmax+1l)*(Lmax+1l));
  }

  void unnormalized_solid_harmonics(int Lmax,
                         nda::ArrayOfRank<2> auto const& r,
                         nda::ArrayOfRank<2> auto && rlYlm)
  {
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(r)>>, "Type mismatch");
    static_assert(std::is_same_v<T,::nda::get_value_t<decltype(rlYlm)>>, "Type mismatch");
    auto all = ::nda::range::all;
    utils::check(r.extent(0) == rlYlm.extent(0), "Size mismatch.");
    utils::check(r.extent(1)>=3 and rlYlm.extent(1) >= (Lmax+1)*(Lmax+1), "Size mismatch.");
    for(int i=0; i<r.extent(0); ++i)
      unnormalized_solid_harmonics(Lmax,r(i,all),rlYlm(i,all));
  }

  private:

  // some constants
  T N1_2 = T(0.5*std::sqrt(0.318309886183791));       // sqrt(1/4*pi) 
  T N3_2 = T(0.5*std::sqrt(3.0*0.318309886183791));   // sqrt(3/4*pi)
  T N5_4 = T(0.25*std::sqrt(5.0*0.318309886183791));  // sqrt(5/16*pi) 
  T N7_4 = T(0.25*std::sqrt(7.0*0.318309886183791));  // sqrt(7/16*pi)
  T N9_16 = T(std::sqrt(0.318309886183791)*3.0/16.0); // sqrt(9/32*pi)

  T N15_2 = T(0.5*std::sqrt(15.0*0.318309886183791));        // sqrt(15/4*pi)
  T N35_2_4 = T(std::sqrt(35.0/2.0*0.318309886183791)/4.0);  // sqrt(35/32*pi)
  T N105_2 = T(std::sqrt(105.0*0.318309886183791)/2.0);      // sqrt(105/4*pi) 
  T N21_2_4 = T(std::sqrt(21.0/2.0*0.318309886183791)/4.0);  // sqrt(21/32*pi)
  T N35_4 = T(std::sqrt(35.0*0.318309886183791)/4.0);        // sqrt(35/16*pi)
  T N5_2_4 = T(std::sqrt(5.0/2.0*0.318309886183791)/4.0);    // sqrt(5/32*pi)

};

}

#endif
