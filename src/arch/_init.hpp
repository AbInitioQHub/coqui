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


////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

#ifndef ARCH_INIT_HPP
#define ARCH_INIT_HPP

#include <cstdlib>
#include "config.h"

#include "IO/app_loggers.h"

#if defined(ENABLE_CUDA)
#include "CUDA/cuda_init.hpp"
#elif defined(ENABLE_ROCM)
//#include "ROCm/rocm_arch.h"
#endif

namespace arch 
{

void init(bool active_log, int output_level=2, int debug_level=2)
{
  // setup loggers, can always be changed later
  setup_loggers(active_log, output_level, debug_level);

#if defined(ENABLE_CUDA)
  cuda::init();
#endif

 // setup shared memory, memory buffers, etc, etc
}

}

#endif
