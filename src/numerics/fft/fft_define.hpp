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


#ifndef NUMERICS_FFT_FFT_DEFINE_HPP
#define NUMERICS_FFT_FFT_DEFINE_HPP

#include "configuration.hpp"

// define generic fftplan_t
namespace math::fft
{

enum FFT_BACKEND {FFT_BACKEND_UNDEFINED,FFT_BACKEND_FFTW,FFT_BACKEND_CUFFT,FFT_BACKEND_ROCM};

#if defined(ENABLE_FFTW)
#include <fftw3.h>
enum FFT_FLAGS {
   FFT_ESTIMATE=FFTW_ESTIMATE,
   FFT_MEASURE=FFTW_MEASURE,
   FFT_PATIENT=FFTW_PATIENT,
   FFT_DESTROY_INPUT=FFTW_DESTROY_INPUT,
   FFT_PRESERVE_INPUT=FFTW_PRESERVE_INPUT,
   FFT_UNALIGNED=FFTW_UNALIGNED,
   FFT_DEFAULT = 99999
};
#else
enum FFT_FLAGS {
   FFT_ESTIMATE,
   FFT_MEASURE,
   FFT_PATIENT,
   FFT_DESTROY_INPUT,
   FFT_PRESERVE_INPUT,
   FFT_UNALIGNED,
   FFT_DEFAULT = 99999
};
#endif

struct fftplan_t 
{
  // can check for initialized state for more graceful error messages
  // bool init;   
  FFT_BACKEND bend = FFT_BACKEND_UNDEFINED;
  int howmany = 0;
  int rank = 0;
  // store alignment?
  // store types for checking?
  void* fwd = nullptr;
  void* inv = nullptr;
};


} // namespace math::fft

#endif
