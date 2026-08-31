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


#include <map>
#include "utilities/check.hpp"
#include "numerics/fft/fft_define.hpp"
#include "numerics/fft/fftw.h"

namespace math::fft::impl::host
{

// Note: right now this assumes alignment to 16 Bytes in order to use advanced execute interface, should I check?
//int fftw_alignment_of(double *p);

fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             ComplexType *in, const int *inembed,
                             int istride, int idist,
                             ComplexType *out, const int *onembed,
                             int ostride, int odist,
                             const unsigned flags) 
{
  fftw_plan* fwd = new fftw_plan{};
  *fwd = fftw_plan_many_dft(rank,n,std::max(howmany,1),
                        reinterpret_cast<fftw_complex*>(in),inembed,istride,idist,
                        reinterpret_cast<fftw_complex*>(out),onembed,ostride,odist,
                        FFTW_FORWARD,flags); 
  fftw_plan* inv = new fftw_plan{};
  *inv = fftw_plan_many_dft(rank,n,std::max(howmany,1),
                        reinterpret_cast<fftw_complex*>(in),inembed,istride,idist,
                        reinterpret_cast<fftw_complex*>(out),onembed,ostride,odist,
                        FFTW_BACKWARD,flags);
  return fftplan_t {FFT_BACKEND_FFTW,howmany,rank,fwd,inv};
}
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             RealType *in, const int *inembed,
                             int istride, int idist,
                             ComplexType *out, const int *onembed,
                             int ostride, int odist,
                             const unsigned flags)
{
  fftw_plan* fwd = new fftw_plan{};
  *fwd = fftw_plan_many_dft_r2c(rank,n,std::max(howmany,1),
                        in,inembed,istride,idist,
                        reinterpret_cast<fftw_complex*>(out),onembed,ostride,odist,
                        flags);
  return fftplan_t {FFT_BACKEND_FFTW,howmany,rank,fwd,nullptr};
}
fftplan_t create_plan_many_impl_(int rank, const int *n, int howmany,
                             ComplexType *in, const int *inembed,
                             int istride, int idist,
                             RealType *out, const int *onembed,
                             int ostride, int odist,
                             const unsigned flags)
{
  fftw_plan* fwd = new fftw_plan{};
  *fwd = fftw_plan_many_dft_c2r(rank,n,std::max(howmany,1),
                        reinterpret_cast<fftw_complex*>(in),inembed,istride,idist,
                        out,onembed,ostride,odist,
                        flags);
  return fftplan_t {FFT_BACKEND_FFTW,howmany,rank,fwd,nullptr};
}
fftplan_t create_plan_many_impl_(int rank, [[maybe_unused]] const int *n, int howmany,
                             [[maybe_unused]] RealType *in, [[maybe_unused]] const int *inembed,
                             [[maybe_unused]] int istride, [[maybe_unused]] int idist,
                             [[maybe_unused]] RealType *out, [[maybe_unused]] const int *onembed,
                             [[maybe_unused]] int ostride, [[maybe_unused]] int odist,
                             [[maybe_unused]] const unsigned flags)
{
  utils::check(false,"r2r transforms not yet working.");
  return fftplan_t {
		FFT_BACKEND_FFTW,
		howmany,
		rank,
		nullptr,
		nullptr
   		   };  
}

void destroy_plan(fftplan_t& p) 
{
  // if uninitialized, don;t check for backend
  if(p.fwd==nullptr and p.inv==nullptr) return; 
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  if(fwd != nullptr) {
    fftw_destroy_plan(*fwd);
    delete fwd;
    fwd = nullptr;
  }
  fftw_plan* inv = reinterpret_cast<fftw_plan*>(p.inv);
  if(inv != nullptr) {
    fftw_destroy_plan(*inv);
    delete inv;
    inv = nullptr;
  }
}

// check for alignment???
void fwdfft(fftplan_t const & p, ComplexType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft(*fwd,
		   reinterpret_cast<fftw_complex*>(in),
		   reinterpret_cast<fftw_complex*>(out));
}
void fwdfft(fftplan_t const& p, RealType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft_r2c(*fwd,in,
		   reinterpret_cast<fftw_complex*>(out));
}
void fwdfft(fftplan_t const& p, ComplexType *in, RealType *out)
{
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft_c2r(*fwd,
		   reinterpret_cast<fftw_complex*>(in),out);
}
void fwdfft(fftplan_t const& p, [[maybe_unused]] RealType *in, [[maybe_unused]] RealType *out)
{
  utils::check(false,"r2r transforms not yet working.");
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
//  fftw_execute_dft_r2r(*fwd,in,out);
}

void invfft(fftplan_t const& p, ComplexType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* inv = reinterpret_cast<fftw_plan*>(p.inv);
  utils::check(inv != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft(*inv,
                   reinterpret_cast<fftw_complex*>(in),
                   reinterpret_cast<fftw_complex*>(out));
}
void invfft(fftplan_t const& p, RealType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft_r2c(*fwd,in,
                   reinterpret_cast<fftw_complex*>(out));
}
void invfft(fftplan_t const& p, ComplexType *in, RealType *out)
{ 
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
  fftw_plan* fwd = reinterpret_cast<fftw_plan*>(p.fwd);
  utils::check(fwd != nullptr,"Uninitiated fftw plan.");
  fftw_execute_dft_c2r(*fwd,
                   reinterpret_cast<fftw_complex*>(in),out);
}
void invfft(fftplan_t const& p, [[maybe_unused]] RealType *in, [[maybe_unused]] RealType *out)
{ 
  utils::check(false,"r2r transforms not yet working.");
  utils::check(p.bend == FFT_BACKEND_FFTW,"Incorrect FFT backend.");
//  fftw_execute_dft_r2r(p.inv,in,out);
}

} // math::fft::impl::host
