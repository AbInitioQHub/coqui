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



#include <array>
#include "utilities/check.hpp"
#include "utilities/harmonics.h"
#include "nda/nda.hpp"

namespace utils
{

/* Following convention from wannier90 projection */
template<typename T>
void harmonics<T>::spherical_harmonics_l(int L, T const* r, long r_size, T* Ylm, long Ylm_size) 
{
  utils::check(L>=0 and L<4,"Spherical harmonics not implemented yet for Lmax>3, L:{}",L);
  utils::check(r_size >= 3 and Ylm_size >= (2*L+1l), "Size mismatch.");

  if (L==0) {
    Ylm[0] = N1_2;
    return;
  }

  // MAM: consider just calling solid_harmonics_l with a normalized r 
  T dr = std::sqrt(r[0]*r[0] + r[1]*r[1] + r[2]*r[2]);
  T cost =  ( dr<T(1e-9) ? T(0.0) : r[2]/dr ), sint = std::sqrt(T(1.0)-cost*cost);
  T phi = std::atan2(r[1], r[0]);
  T cosp = std::cos(phi), sinp = std::sin(phi); 
  T cos2p = T(2.0)*cosp*cosp-T(1.0), sin2p = T(2.0)*sinp*cosp; 
  switch (L)
  {
    case 1:
      Ylm[0] = N3_2*sint*sinp;
      Ylm[1] = N3_2*cost;
      Ylm[2] = N3_2*sint*cosp;
      break;
    case 2:
      Ylm[0] = T(0.5)*N15_2*sint*sint*sin2p;
      Ylm[1] = N15_2*sint*cost*sinp;
      Ylm[2] = N5_4*(T(3.0)*cost*cost-T(1.0));
      Ylm[3] = N15_2*sint*cost*cosp;
      Ylm[4] = T(0.5)*N15_2*sint*sint*cos2p;
      break;
    case 3:
    {
      T sint_2 = sint*sint, cost_2 = cost*cost;
      T sint_3 = sint_2*sint, cost_3 = cost_2*cost;
      Ylm[0] = N35_2_4*sint_3*(T(3.0)*cosp*cosp-sinp*sinp)*sinp;
      Ylm[1] = T(0.5)*N105_2*sint_2*cost*sin2p;
      Ylm[2] = N21_2_4*(T(5.0)*cost_2-T(1.0))*sint*sinp;
      Ylm[3] = N7_4*(T(5.0)*cost_3-T(3.0)*cost);
      Ylm[4] = N21_2_4*(T(5.0)*cost_2-T(1.0))*sint*cosp;
      Ylm[5] = T(0.5)*N105_2*sint_2*cost*cos2p;
      Ylm[6] = N35_2_4*sint_3*(cosp*cosp-T(3.0)*sinp*sinp)*cosp;
      break;
    }
  }

}

template<typename T>
void harmonics<T>::solid_harmonics_l(int L, T const* r, long r_size, T* rlYlm, long rlYlm_size) 
{
  utils::check(L>=0 and L<5,"Solid harmonics not implemented yet for L>4, L:{}",L);
  utils::check(r_size >= 3 and rlYlm_size >= (2*L+1l), "Size mismatch.");

  if(L==0) {
    rlYlm[0] = N1_2; 
    return;
  }

  T x = r[0], y = r[1], z = r[2];
  switch (L)
  {
    case 1:
      rlYlm[0] = N3_2*x; 
      rlYlm[1] = N3_2*y; 
      rlYlm[2] = N3_2*z; 
      break;
    case 2:
      rlYlm[0] = N15_2*x*y;
      rlYlm[1] = N15_2*y*z;
      rlYlm[2] = N5_4*(T(2.0)*z*z-x*x-y*y);
      rlYlm[3] = N15_2*x*z;
      rlYlm[4] = T(0.5)*N15_2*(x*x-y*y);
      break;
    case 3:
    {
      T x2=x*x, y2=y*y, z2=z*z;
      rlYlm[0] = N35_2_4*y*(T(3.0)*x2-y2);
      rlYlm[1] = N105_2*x*y*z;
      rlYlm[2] = N21_2_4*y*(T(4.0)*z2-x2-y2);
      rlYlm[3] = N7_4*z*(T(2.0)*z2 - T(3.0)*x2 - T(3.0)*y2);
      rlYlm[4] = N21_2_4*x*(T(4.0)*z2-x2-y2);
      rlYlm[5] = T(0.5)*N105_2*z*(x2-y2);
      rlYlm[6] = N35_2_4*x*(x2-T(3.0)*y2);
      break;
    }
    case 4:
    {
      T x2=x*x, y2=y*y, z2=z*z, r2=x2+y2+z2;
      rlYlm[0] = T(3.0)*N35_4*x*y*(x2-y2); 
      rlYlm[1] = T(3.0)*N35_2_4*y*z*(T(3.0)*x2-y2);
      rlYlm[2] = T(3.0)*N5_4*x*y*(T(7.0)*z2-r2);
      rlYlm[3] = T(3.0)*N5_2_4*y*z*(T(7.0)*z2-T(3.0)*r2);
      rlYlm[4] = N9_16*(T(35.0)*z2*z2 - T(30.0)*z2*r2 + T(3.0)*r2*r2);
      rlYlm[5] = T(3.0)*N5_2_4*x*z*(T(7.0)*z2-T(3.0)*r2);
      rlYlm[6] = T(1.5)*N5_4*(x2-y2)*(T(7.0)*z2-r2);
      rlYlm[7] = T(3.0)*N35_2_4*x*z*(x2-T(3.0)*y2);
      rlYlm[8] = T(0.75)*N35_4*(x2*(x2-T(3.0)*y2) - y2*(T(3.0)*x2-y2));
      break;
    }
  }
}

template<typename T>
void harmonics<T>::unnormalized_solid_harmonics_l(int L, T const* r, long r_size, T* rlYlm, long rlYlm_size) 
{
  utils::check(L>=0 and L<6,"Unnormalized solid harmonics not implemented yet for L>5, L:{}",L);
  utils::check(r_size >= 3 and rlYlm_size >= (2*L+1l), "Size mismatch.");

  if(L==0) {
    rlYlm[0] = 1.0; 
    return;
  }

  T x = r[0], y = r[1], z = r[2];
  switch (L)
  {
    case 1:
      rlYlm[0] = x; 
      rlYlm[1] = y; 
      rlYlm[2] = z; 
      break;
    case 2:
      rlYlm[0] = x*y;
      rlYlm[1] = y*z;
      rlYlm[2] = (T(2.0)*z*z-x*x-y*y);
      rlYlm[3] = x*z;
      rlYlm[4] = x*x-y*y;
      break;
    case 3:
    {
      T x2=x*x, y2=y*y, z2=z*z;
      rlYlm[0] = y*(T(3.0)*x2-y2);
      rlYlm[1] = x*y*z;
      rlYlm[2] = y*(T(4.0)*z2-x2-y2);
      rlYlm[3] = z*(T(2.0)*z2 - T(3.0)*x2 - T(3.0)*y2);
      rlYlm[4] = x*(T(4.0)*z2-x2-y2);
      rlYlm[5] = z*(x2-y2);
      rlYlm[6] = x*(x2-T(3.0)*y2);
      break;
    }
    case 4:
    {
      T x2=x*x, y2=y*y, z2=z*z, r2=x2+y2+z2;
      rlYlm[0] = x*y*(x2-y2); 
      rlYlm[1] = y*z*(T(3.0)*x2-y2);
      rlYlm[2] = x*y*(T(7.0)*z2-r2);
      rlYlm[3] = y*z*(T(7.0)*z2-T(3.0)*r2);
      rlYlm[4] = (T(35.0)*z2*z2 - T(30.0)*z2*r2 + T(3.0)*r2*r2);
      rlYlm[5] = x*z*(T(7.0)*z2-T(3.0)*r2);
      rlYlm[6] = (x2-y2)*(T(7.0)*z2-r2);
      rlYlm[7] = x*z*(x2-T(3.0)*y2);
      rlYlm[8] = (x2*(x2-T(3.0)*y2) - y2*(T(3.0)*x2-y2));
      break;
    }
    case 5:
    {
      T x2=x*x, y2=y*y, z2=z*z, r2=x2+y2+z2;
      T x4=x2*x2, y4=y2*y2, z4=z2*z2, r4=r2*r2;
      rlYlm[0] = y*(T(5.0)*x4 - T(10.0)*x2*y2 + y4);      
      rlYlm[1] = x*y*z*(x2 - y2);      
      rlYlm[2] = y*(T(3.0)*x2 - y2)*(x2 + y2 - T(8.0)*z2);      
      rlYlm[3] = x*y*z*(x2 + y2 - T(2.0)*z2);      
      rlYlm[4] = y*(x4 + y4 - T(12.0)*y2*z2 + T(8.0)*z4 + T(2.0)*x2*(y2 - T(6.0)*z2));
      rlYlm[5] = z*(T(63.0)*z4 - T(70.0)*z2*r2 + T(15.0)*r4);       
      rlYlm[6] = x*(x4 + y4 - T(12.0)*y2*z2 + T(8.0)*z4 + T(2.0)*x2*(y2 - T(6.0)*z2)); 
      rlYlm[7] = z*(x2 - y2)*(x2 + y2 - T(2.0)*z2);      
      rlYlm[8] = (x2*x - T(3.0)*x*y2)*(x2 + y2 - T(8.0)*z2);      
      rlYlm[9] = z*(x4 - T(6.0)*x2*y2 + y4);      
      rlYlm[10] = x*(x4 - T(10.0)*x2*y2 + T(5.0)*y4);      
      break;
    }
  }
}

template void harmonics<double>::spherical_harmonics_l(int,double const*,long,double*,long);
template void harmonics<double>::solid_harmonics_l(int,double const*,long,double*,long);
template void harmonics<double>::unnormalized_solid_harmonics_l(int,double const*,long,double*,long);

template void harmonics<float>::spherical_harmonics_l(int,float const*,long,float*,long);
template void harmonics<float>::solid_harmonics_l(int,float const*,long,float*,long);
template void harmonics<float>::unnormalized_solid_harmonics_l(int,float const*,long,float*,long);

}

