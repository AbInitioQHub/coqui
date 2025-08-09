
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
