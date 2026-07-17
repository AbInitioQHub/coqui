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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/proc_grid_partition.hpp"

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"

namespace bdft_tests
{

namespace mpi3 = boost::mpi3;
using namespace math::shm;
using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

TEST_CASE("distributed_shared_nda", "[math]") {
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  // Setup internode communicator
  int node_size = node_comm.size();
  int color = world.rank()%node_size;
  int key   = world.rank()/node_size;
  auto internode_comm = world.split(color, key);

  using Array_view_base_t = nda::array_view<ComplexType, 3>;

  int n_nodes = internode_comm.size();
  int node_rank = internode_comm.rank();
  shape_t<3> grid = {n_nodes, 1, 1};
  shape_t<3> gshape = {39, 2, 2};

  auto array = make_distributed_shared_array<Array_view_base_t>(world, internode_comm, node_comm,
                                                                grid, gshape);
  app_log(2, "Global shape = ({}, {}, {})", array.global_shape()[0], array.global_shape()[1], array.global_shape()[2]);
  world.barrier();
  std::cout << "At node " << node_rank << ", local shape = (" << array.local_shape()[0] <<
  ", " << array.local_shape()[1] << ", " << array.local_shape()[2] << ")" << std::endl;
  std::cout << "At node " << node_rank << ", local origin = (" << array.origin()[0] <<
  ", " << array.origin()[1] << ", " << array.origin()[2] << ")" << std::endl;

  int rank = array.node_comm()->rank();
  int group_size = array.node_comm()->size();
  auto array_loc = array.local();
  nda::matrix<ComplexType> eye(2, 2);
  eye() = 2.0;
  int t_offset = array.origin()[0];
  for (int it = rank; it < array.local_shape()[0]; it += group_size) {
    int t = it + t_offset;
    nda::matrix_view<ComplexType> array_t = array_loc(t, nda::range::all, nda::range::all);
    array_t += 2.0;
  }
  array.node_sync();
}

TEST_CASE("shared_nda_chunked_all_reduce", "[math][shared][all_reduce]") {
  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  int node_size = node_comm.size();
  int color = world.rank() % node_size;
  int key = world.rank() / node_size;
  auto internode_comm = world.split(color, key);

  auto check_type = [&]<typename T>() {
    using Array_view_t = nda::array_view<T, 1>;
    auto array = make_shared_array<Array_view_t>(world, internode_comm, node_comm, shape_t<1>{17});
    if (node_comm.root()) {
      T value = T(internode_comm.rank() + 1);
      array.local() = value;
    }
    array.node_sync();

    // Force the 17 elements through a 3,3,3,3,3,2 chunk sequence.
    array.all_reduce(3 * sizeof(T));

    double expected = 0.5 * internode_comm.size() * (internode_comm.size() + 1);
    auto local = array.local();
    for (long i = 0; i < local.size(); ++i)
      REQUIRE(std::abs(local(i) - T(expected)) < 1.0e-12);
  };

  check_type.template operator()<double>();
  check_type.template operator()<ComplexType>();
}

} // bdft_tests
