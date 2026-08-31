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


#ifndef NUMERICS_FFT_FFTW_H
#define NUMERICS_FFT_FFTW_H

#include<map>
#include <fftw3.h>
#include "configuration.hpp"
#include "numerics/fft/fft_define.hpp"

namespace math::fft::impl::host 
{

// batched FFT interface
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             ComplexType *in, const int *inembed,
                             int istride, int idist,
                             ComplexType *out, const int *onembed,
                             int ostride, int odist,
			     const unsigned flags);
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             RealType *in, const int *inembed,
                             int istride, int idist,
                             ComplexType *out, const int *onembed,
                             int ostride, int odist,
			     const unsigned flags);
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             ComplexType *in, const int *inembed,
                             int istride, int idist,
                             RealType *out, const int *onembed,
                             int ostride, int odist,
			     const unsigned flags);
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             RealType *in, const int *inembed,
                             int istride, int idist,
                             RealType *out, const int *onembed,
                             int ostride, int odist,
			     const unsigned flags);

template<int rank, typename T1, typename T2>
fftplan_t create_plan_many(const int *n, int howmany,
                           T1 *in, const int *inembed,
                           int istride, int idist,
                           T2 *out, const int *onembed,
                           int ostride, int odist,
                           const unsigned flags)
{
  return create_plan_many_impl_(rank,n,howmany,in,inembed,istride,idist,
			               out,onembed,ostride,odist,flags);
}
template<int rank, typename T1, typename T2>
fftplan_t create_plan_many(const long int *n, int howmany,
                           T1 *in, const long int *inembed,
                           int istride, int idist,
                           T2 *out, const long int *onembed,
                           int ostride, int odist,
                           const unsigned flags)
{
  std::array<int,rank> n_,inembed_,onembed_;
  int* inem(nullptr), *onem(nullptr);
  std::copy_n(n,rank,n_.data());
  if(inembed != nullptr) {
    std::copy_n(inembed,rank,inembed_.data());
    inem=inembed_.data();
  }
  if(onembed != nullptr) {
    std::copy_n(onembed,rank,onembed_.data());
    onem=onembed_.data();
  }
  return create_plan_many_impl_(rank,n_.data(),howmany,in,inem,istride,idist,
                                       out,onem,ostride,odist,flags);
}

template<int rank, typename T1, typename T2>
fftplan_t create_plan(const int *n, T1 *in, const int *inembed,  
                      T2 *out, const int *onembed, const unsigned flags)
{
  // MAM: howmany==0 means single FFT interface
  return create_plan_many_impl_(rank,n,0,in,inembed,1,1,out,onembed,1,1,flags); 
}

template<int rank, typename T1, typename T2>
fftplan_t create_plan(const long int *n, T1 *in, const long int *inembed,
                      T2 *out, const long int *onembed, const unsigned flags)
{
  std::array<int,rank> n_,inembed_,onembed_;
  int* inem(nullptr), *onem(nullptr);
  std::copy_n(n,rank,n_.data());
  if(inembed != nullptr) {
    std::copy_n(inembed,rank,inembed_.data());
    inem=inembed_.data();
  }
  if(onembed != nullptr) {
    std::copy_n(onembed,rank,onembed_.data());
    onem=onembed_.data();
  }
  // MAM: howmany==0 means single FFT interface
  return create_plan_many_impl_(rank,n_.data(),0,in,inem,1,0,
			       out,onem,1,0,flags);
}

void destroy_plan(fftplan_t& p);

void fwdfft(fftplan_t const& p, ComplexType *in, ComplexType *out);
void fwdfft(fftplan_t const& p, RealType *in, ComplexType *out);
void fwdfft(fftplan_t const& p, ComplexType *in, RealType *out);
void fwdfft(fftplan_t const& p, RealType *in, RealType *out);

void invfft(fftplan_t const& p, ComplexType *in, ComplexType *out);
void invfft(fftplan_t const& p, RealType *in, ComplexType *out);
void invfft(fftplan_t const& p, ComplexType *in, RealType *out);
void invfft(fftplan_t const& p, RealType *in, RealType *out);

} // math::fft::impl::host

#endif
