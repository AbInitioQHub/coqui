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

#include <complex>

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "nda/nda.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/test_common.hpp"
#include "utilities/Timer.hpp"

namespace bdft_tests
{

namespace mpi3 = boost::mpi3;
using namespace math::shm;
using namespace math::nda;
template <int Rank> using shape_t = std::array<long, Rank>;

template<typename comm_t, typename Arr_t>
void reduce(comm_t &comm, Arr_t && A) {
  size_t delta = 2e9; 
  size_t _size = size_t(A.size());
  for (size_t shift=0; shift<_size; shift+=delta) {
    ComplexType *start = A.data(); 
    size_t count = (shift+delta < _size) ? delta : _size-shift;
    comm.all_reduce_in_place_n(start, count, std::plus<>{});
  }
}

template<typename comm_t, typename Arr_t>
void reduce(comm_t &comm, Arr_t && A, Arr_t && B) {
  size_t delta = 2e9; 
  size_t _size = size_t(A.size());
  for (size_t shift=0; shift<_size; shift+=delta) {
    ComplexType *start = A.data();
    ComplexType *Bstart = B.data();
    size_t count = (shift+delta < _size) ? delta : _size-shift;
    comm.all_reduce_n(start, count, Bstart, std::plus<>{});
  }
}

TEST_CASE("time_collectives", "[sandbox][mpi]")
{

  auto world = mpi3::environment::get_world_instance();
  auto node_comm = world.split_shared();
  // Setup internode communicator
  int node_size = node_comm.size();
  auto internode_comm = world.split(world.rank()%node_size,world.rank()/node_size);

  using Array_view_t = nda::array_view<ComplexType, 3>;
  using Array_t = nda::array<ComplexType, 3>;

  utils::check(world.size()%8 == 0, "ntasks must be divisible by 8");

  utils::TimerManager Timer;
  long N1_ = 1024, N2_=64, N3_=32;
  long p1 = world.size() / 8, p2 = 4, p3 = 2; 
  utils::check(world.size() == p1*p2*p3, "Error with proc partition.");
  //auto pgrid = shape_t<3>{p1,p2,p3};

  //for(int s=1; s<=8; s*=2)
  for(int s=1; s<=2; s*=2)
  {

    long N1 = N1_*s;
    long N2 = N2_*s;
    long N3 = N3_*s;
    app_log(0,"N1:{} N2:{} N3:{} MEM:{} MB",N1,N2,N3,16.0*N1*N2*N3/1024.0/1024.0);

    if(s < 4)
    {
      Array_t A(N1,N2,N3);
      Array_t B(N1,N2,N3);
      A() = ComplexType(0.0);
      B() = ComplexType(0.0);

      world.barrier();
      Timer.start("red");
      reduce(world,A);
      Timer.stop("red");
      world.barrier();
      Timer.start("red2");
      //reduce(world,A,B);
      reduce(world,A);
      Timer.stop("red2");

      app_log(0,"global reduce_in_place: {}  reduce:{}",Timer.elapsed("red"),Timer.elapsed("red2"));
    }
    Timer.reset_all();
    world.barrier();

    if(node_comm.root()) {
      Array_t A(N1,N2,N3);
      Array_t B(N1,N2,N3);
      A() = ComplexType(0.0);
      B() = ComplexType(0.0);
    
      internode_comm.barrier();
      Timer.start("red");
      reduce(internode_comm,A);
      Timer.stop("red");
      internode_comm.barrier();
      Timer.start("red2");
      reduce(internode_comm,A,B);
      Timer.stop("red2");
    
      app_log(0,"global internode reduce_in_place: {}  reduce:{}",Timer.elapsed("red"),Timer.elapsed("red2"));
    }
    Timer.reset_all();
    world.barrier();

    {
      long N = N1*N2*N3;
      auto [a,b] = itertools::chunk_range(0,N,node_comm.size(),node_comm.rank()); 
      N = b-a;
      nda::array<ComplexType, 1> A(N);
      nda::array<ComplexType, 1> B(N);
      A() = ComplexType(0.0);
      B() = ComplexType(0.0);
 
      world.barrier();
      Timer.start("red");
      reduce(internode_comm,A);
      world.barrier();
      Timer.stop("red");
      Timer.start("red2");
      //reduce(internode_comm,A,B);
      reduce(internode_comm,A);
      world.barrier();
      Timer.stop("red2");

      app_log(0,"split reduce_in_place: {}  reduce:{}",Timer.elapsed("red"),Timer.elapsed("red2"));
    }
    Timer.reset_all();
    world.barrier();
  
    {
      auto shape = shape_t<3>{N1,N2,N3};
      //auto dA = make_distributed_array<Array_t>(world, pgrid, shape); 
      //dA.local() = ComplexType(0.0);
      auto sA = make_shared_array<Array_view_t>(world, internode_comm, node_comm, shape);
  
      world.barrier();
      Timer.start("red");
      sA.all_reduce(); 
      world.barrier();
      Timer.stop("red");
      Timer.start("red3");
      sA.all_reduce(); 
      world.barrier();
      Timer.stop("red3");
      if(node_comm.root()) {
        auto A = sA.local();
        Timer.start("red2");
        reduce(internode_comm,A);
        Timer.stop("red2");
      }
      app_log(0,"shm reduce: {} reduce 2: {}, reduce 3: {}",
                Timer.elapsed("red"),Timer.elapsed("red2"),Timer.elapsed("red3"));
    }
    Timer.reset_all();

  } // N1,N2,N3

}

}
