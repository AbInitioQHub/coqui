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
#include <cstdlib>
#include "config.h"

#if defined(ENABLE_CUDA)

#include "CUDA/cuda_init.h"
#include "CUDA/cuda_sync.h"

namespace arch
{
  bool get_device_synchronization() {return cuda::get_device_synchronization();};
  void set_device_synchronization(bool s) { cuda::set_device_synchronization(s); };
  void synchronize_if_set() { cuda::synchronize_if_set(); };
  void synchronize() { cuda::synchronize(); };
}

#else


namespace arch
{
  bool get_device_synchronization() {return true;};
  void set_device_synchronization(bool) {};
  void synchronize_if_set() {};
  void synchronize() {};
}

#endif

#include "IO/app_loggers.h"

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
