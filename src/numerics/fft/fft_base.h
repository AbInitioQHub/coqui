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


#ifndef NUMERICS_FFT_FFT_BASE_H
#define NUMERICS_FFT_FFT_BASE_H

#include "configuration.hpp"
#include "numerics/fft/fft_define.hpp"

// CPU dispatch
#if defined(ENABLE_FFTW)
#include "numerics/fft/fftw.h"
namespace math::fft::impl::host{
  static const bool __normalize__ = true;
}
#else
#error FFTW currently required.
#endif  

// device dispatch
#if defined(ENABLE_DEVICE)

#if defined(ENABLE_CUDA)
#include "numerics/fft/cufft.h"
namespace math::fft::impl::dev{
  static const bool __normalize__ = true;
}
#elif defined(ENABLE_ROCM)
#error ROCM fft backend currently unavailable 
#endif  

#else   // no device backend, set dev to host

namespace math::fft::impl{
  // forces identical interfaces, if this is a problem then move #if defined(ENABLE_CUDA) to nda.hpp
  namespace dev = host;
}

#endif

#endif
