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



#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#if defined(ENABLE_NCCL)
#include "mpi3/nccl/communicator.hpp"
#endif

namespace mpi3 = boost::mpi3;
namespace utils {

mpi_context_t<mpi3::communicator> make_mpi_context() {
  using mpi3::communicator;
  using mpi3::shared_communicator;
  utils::check(boost::mpi3::initialized(), "Uninitialized MPI3.");
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  auto internode_comm = world.split(node_comm.rank(), world.rank());
#if defined(ENABLE_NCCL)
  mpi3::nccl::communicator dev_comm(world);
  return mpi_context_t<communicator,shared_communicator,mpi3::nccl::communicator>(std::move(world),std::move(node_comm),std::move(internode_comm),std::move(dev_comm));
#else
  return mpi_context_t<communicator,shared_communicator>(std::move(world),std::move(node_comm),std::move(internode_comm));
#endif
};

mpi_context_t<mpi3::communicator> make_mpi_context(mpi3::communicator& comm) {
  using mpi3::communicator;
  using mpi3::shared_communicator;
  utils::check(boost::mpi3::initialized(), "Uninitialized MPI3.");
  mpi3::communicator comm_copy(comm); 
  auto node_comm = comm.split_shared();
  auto internode_comm = comm.split(node_comm.rank(), comm.rank());
#if defined(ENABLE_NCCL)
  mpi3::nccl::communicator dev_comm(comm);
  return mpi_context_t<communicator,shared_communicator,mpi3::nccl::communicator>(std::move(comm_copy),std::move(node_comm),std::move(internode_comm),std::move(dev_comm));
#else
  return mpi_context_t<communicator,shared_communicator>(std::move(comm_copy),std::move(node_comm),std::move(internode_comm));
#endif
};

}
