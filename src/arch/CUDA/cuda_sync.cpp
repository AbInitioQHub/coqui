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



#include "cuda_init.h"
#include "cuda_sync.h"

namespace cuda 
{
  static bool __sync__ = true;
  void synchronize() 
  {
    cuda_check(cudaGetLastError());
    cuda_check(cudaDeviceSynchronize());    
  }
  bool get_device_synchronization() {return __sync__;}
  void set_device_synchronization(bool s) 
  {
    synchronize();
    __sync__ = s;
  }
  void synchronize_if_set()
  {
    if(__sync__) synchronize(); 
  }
}

