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

#include "IO/app_loggers.h"
#include "cuda_runtime.h" 

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

namespace cuda
{

void cuda_check(cudaError_t sucess, std::string message)
{
  if (sucess != cudaSuccess) {
   app_error(" Cuda runtime error: {}",std::to_string(sucess));
   if(message != "")
     app_error(" message: {}",message);
   app_error(" cudaGetErrorName: {}",std::string(cudaGetErrorName(sucess)));
   app_error(" cudaGetErrorString: {}",std::string(cudaGetErrorString(sucess)));
   APP_ABORT(" Cuda runtime error"); 
  }
}

void init() 
{
  auto world = boost::mpi3::environment::get_world_instance();
  auto node = world.split_shared(world.rank());

  int num_devices = 0;
  cudaGetDeviceCount(&num_devices);
  app_log(1, " Running in node with {} GPUs. ", num_devices);
  cudaDeviceProp dev;
  cuda_check(cudaGetDeviceProperties(&dev, 0), "cudaGetDeviceProperties");
  app_log(1, " CUDA compute capability: {}.{} \n ", dev.major, dev.minor);
  app_log(1, " Device Name: {} ", dev.name);
  if (dev.major <= 6 and world.root())
  {
    app_warning(" Warning CUDA major compute capability < 6.0");
  }
  if (num_devices > node.size() and world.root())
  {
    app_warning("WARNING: Unused devices !!!!!!!!!!!!!! ");
    app_warning("         # tasks: {} ", node.size());
    app_warning("         # number of devices: {} ", num_devices);
  }

  cuda_check(cudaSetDevice(node.rank()%num_devices), "cudaSetDevice()");
  int devn = 0;
  cuda_check(cudaGetDevice(&devn), "cudaGetDevice()");
  app_debug(3,"MPI world rank: {}, node rank{}, cuda device number: {}",
	    world.rank(),node.rank(),devn);
}

}

