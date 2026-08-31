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
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "numerics/fft/fft_define.hpp"
#include "numerics/fft/cufft.h"

namespace math::fft::impl::dev
{

/*
typedef enum cufftResult_t {
    CUFFT_SUCCESS        = 0,  //  The cuFFT operation was successful
    CUFFT_INVALID_PLAN   = 1,  //  cuFFT was passed an invalid plan handle
    CUFFT_ALLOC_FAILED   = 2,  //  cuFFT failed to allocate GPU or CPU memory
    CUFFT_INVALID_TYPE   = 3,  //  No longer used
    CUFFT_INVALID_VALUE  = 4,  //  User specified an invalid pointer or parameter
    CUFFT_INTERNAL_ERROR = 5,  //  Driver or internal cuFFT library error
    CUFFT_EXEC_FAILED    = 6,  //  Failed to execute an FFT on the GPU
    CUFFT_SETUP_FAILED   = 7,  //  The cuFFT library failed to initialize
    CUFFT_INVALID_SIZE   = 8,  //  User specified an invalid transform size
    CUFFT_UNALIGNED_DATA = 9,  //  No longer used
    CUFFT_INCOMPLETE_PARAMETER_LIST = 10, //  Missing parameters in call
    CUFFT_INVALID_DEVICE = 11, //  Execution of a plan was on different GPU than plan creation
    CUFFT_PARSE_ERROR    = 12, //  Internal plan database error
    CUFFT_NO_WORKSPACE   = 13  //  No workspace has been provided prior to plan execution
    CUFFT_NOT_IMPLEMENTED = 14, // Function does not implement functionality for parameters given.
    CUFFT_LICENSE_ERROR  = 15, // Used in previous versions.
    CUFFT_NOT_SUPPORTED  = 16  // Operation is not supported for parameters given.
} cufftResult;
*/
void cufft_check(cufftResult_t result, std::string message)
{
  if(result == CUFFT_SUCCESS) return;
  std::string err;
  switch(result)
  {
    case CUFFT_INVALID_PLAN:
      err = "CUFFT_INVALID_PLAN: cuFFT was passed an invalid plan handle";	
      break;
    case CUFFT_ALLOC_FAILED:
      err = "CUFFT_ALLOC_FAILED: cuFFT failed to allocate GPU or CPU memory";
      break;
    case CUFFT_INVALID_TYPE:
      err = "CUFFT_INVALID_TYPE: No longer used";
      break;
    case CUFFT_INVALID_VALUE:
      err = "CUFFT_INVALID_VALUE: User specified an invalid pointer or parameter";
      break;
    case CUFFT_INTERNAL_ERROR:
      err = "CUFFT_INTERNAL_ERROR:  Driver or internal cuFFT library error";
      break;
    case CUFFT_EXEC_FAILED:
      err = "CUFFT_EXEC_FAILED: Failed to execute an FFT on the GPU";
      break;
    case CUFFT_SETUP_FAILED:
      err = "CUFFT_SETUP_FAILED: The cuFFT library failed to initialize";
      break;
    case CUFFT_INVALID_SIZE:
      err = "CUFFT_INVALID_SIZE: User specified an invalid transform size";
      break;
    case CUFFT_UNALIGNED_DATA:
      err = "CUFFT_UNALIGNED_DATA: No longer used";
      break;
    case CUFFT_INCOMPLETE_PARAMETER_LIST:
      err = "CUFFT_INCOMPLETE_PARAMETER_LIST: Missing parameters in call";
      break;
    case CUFFT_INVALID_DEVICE:
      err = "CUFFT_INVALID_DEVICE: Execution of a plan was on different GPU than plan creation";
      break;
    case CUFFT_PARSE_ERROR:
      err = "CUFFT_PARSE_ERROR: Internal plan database error";
      break;
    case CUFFT_NO_WORKSPACE:
      err = "CUFFT_NO_WORKSPACE: No workspace has been provided prior to plan execution";
      break;
    case CUFFT_NOT_IMPLEMENTED:
      err = "CUFFT_NOT_IMPLEMENTED: Function does not implement functionality for parameters given";
      break;
    case CUFFT_LICENSE_ERROR:
      err = "CUFFT_LICENSE_ERROR: Used in previous versions.";
      break;
    case CUFFT_NOT_SUPPORTED:
      err = "CUFFT_NOT_SUPPORTED: Operation is not supported for parameters given.";
      break;
    default:
      err = "CUFFT_UNKNOWN_ERROR";
  }
  app_error(message);
  err = std::string(" Error code returned by cufft: ") + err;
  APP_ABORT(err);
}

// MAM: Not using custom memory allocation for now, revisit if needed!

fftplan_t create_plan_many_impl_(int rank, int *n, int howmany,
                             [[maybe_unused]] ComplexType *in, [[maybe_unused]] int *inembed,
                             int istride, int idist,
                             [[maybe_unused]] ComplexType *out, int *onembed,
                             int ostride, int odist)
{
  cufftHandle* p = new cufftHandle{};
  cufft_check(cufftPlanMany(p,rank,n,inembed,istride,idist,onembed,ostride,odist,
                CUFFT_Z2Z,std::max(howmany,1)), "cufftPlanMany");
  return fftplan_t {FFT_BACKEND_CUFFT,howmany,rank,p,nullptr};
}
fftplan_t create_plan_many_impl_(int rank, int *n, int howmany,
                             [[maybe_unused]] RealType *in, int *inembed,
                             int istride, int idist,
                             [[maybe_unused]] ComplexType *out, int *onembed,
                             int ostride, int odist)
{
  cufftHandle* p = new cufftHandle{};
  cufft_check(cufftPlanMany(p,rank,n,inembed,istride,idist,onembed,ostride,odist,
                CUFFT_D2Z,std::max(howmany,1)), "cufftPlanMany");
  return fftplan_t {FFT_BACKEND_CUFFT,howmany,rank,p,nullptr};
}
fftplan_t create_plan_many_impl_(int rank, int *n, int howmany,
                             [[maybe_unused]] ComplexType *in, int *inembed,
                             int istride, int idist,
                             [[maybe_unused]] RealType *out, int *onembed,
                             int ostride, int odist)
{
  cufftHandle* p = new cufftHandle{};
  cufft_check(cufftPlanMany(p,rank,n,inembed,istride,idist,onembed,ostride,odist,
                CUFFT_Z2D,std::max(howmany,1)), "cufftPlanMany");
  return fftplan_t {FFT_BACKEND_CUFFT,howmany,rank,p,nullptr};
}
fftplan_t create_plan_many_impl_(int rank, [[maybe_unused]] int *n, int howmany,
                             [[maybe_unused]] RealType *in,  [[maybe_unused]] int *inembed,
                             [[maybe_unused]] int istride,   [[maybe_unused]] int idist,
                             [[maybe_unused]] RealType *out, [[maybe_unused]] int *onembed,
                             [[maybe_unused]] int ostride,   [[maybe_unused]] int odist)
{
  utils::check(false,"r2r transforms not yet working.");
  return fftplan_t {
		FFT_BACKEND_CUFFT,
		howmany,
		rank,
		nullptr,
		nullptr
   		   };  
}

void destroy_plan(fftplan_t& p) 
{
  if(p.fwd==nullptr) return;
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  if(plan != nullptr) {
    cufft_check(cufftDestroy(*plan),"cufftDestroy");
    delete plan;
    plan = nullptr;
  }
}

// check for alignment???
void fwdfft(fftplan_t& p, ComplexType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecZ2Z(*plan,
		   reinterpret_cast<cufftDoubleComplex*>(in),
		   reinterpret_cast<cufftDoubleComplex*>(out),
		   CUFFT_FORWARD), "cufftExecZ2Z");
  cudaDeviceSynchronize();
}
void fwdfft(fftplan_t& p, RealType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecD2Z(*plan,
                   reinterpret_cast<cufftDoubleReal*>(in),
                   reinterpret_cast<cufftDoubleComplex*>(out)), "cufftExecD2Z");
  cudaDeviceSynchronize();
}
void fwdfft(fftplan_t& p, ComplexType *in, RealType *out)
{
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecZ2D(*plan,
                   reinterpret_cast<cufftDoubleComplex*>(in),
                   reinterpret_cast<cufftDoubleReal*>(out)), "cufftExecZ2D");
  cudaDeviceSynchronize();
}
void fwdfft(fftplan_t& p, [[maybe_unused]]  RealType *in, [[maybe_unused]]  RealType *out)
{
  utils::check(false,"r2r transforms not yet working.");
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
}

void invfft(fftplan_t& p, ComplexType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecZ2Z(*plan,
                   reinterpret_cast<cufftDoubleComplex*>(in),
                   reinterpret_cast<cufftDoubleComplex*>(out),
		   CUFFT_INVERSE), "cufftExecZ2Z");
  cudaDeviceSynchronize();
}
void invfft(fftplan_t& p, RealType *in, ComplexType *out)
{
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecD2Z(*plan,
                   reinterpret_cast<cufftDoubleReal*>(in),
                   reinterpret_cast<cufftDoubleComplex*>(out)), "cufftExecD2Z");
  cudaDeviceSynchronize();
}
void invfft(fftplan_t& p, ComplexType *in, RealType *out)
{ 
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
  cufftHandle* plan = reinterpret_cast<cufftHandle*>(p.fwd);
  utils::check(plan != nullptr,"Uninitiated cufft plan.");
  cufft_check(cufftExecZ2D(*plan,
                   reinterpret_cast<cufftDoubleComplex*>(in),
                   reinterpret_cast<cufftDoubleReal*>(out)), "cufftExecZ2D");
  cudaDeviceSynchronize();
}
void invfft(fftplan_t& p, [[maybe_unused]] RealType *in, [[maybe_unused]] RealType *out)
{ 
  utils::check(false,"r2r transforms not yet working.");
  utils::check(p.bend == FFT_BACKEND_CUFFT,"Incorrect FFT backend.");
}

} // math::fft::impl::dev
