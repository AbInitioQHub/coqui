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


#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include "cxxopts.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/check.hpp"
#include "utilities/Timer.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"

#include "h5/h5.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"

namespace mpi3 = boost::mpi3;
using namespace math::nda;

int main(int argc, char* argv[])
{
  using local_Array_t = nda::array<ComplexType,2>;
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  setup_loggers(world.root(), 10, 10); 
  std::vector<std::string> inputs;

  cxxopts::Options options(argv[0], "Slate Test");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
    ("nR", "number of processors along rows in pgrid", cxxopts::value<long>()->default_value("0"))
    ("nC", "number of processors along cols in pgrid", cxxopts::value<long>()->default_value("0"))
  ;
  auto args = options.parse(argc, argv);

  auto nR = args["nR"].as<long>();
  auto nC = args["nC"].as<long>();

  utils::check(nR*nC == world.size(), "Error: nR*nC != world.size()");

  long M = 1800;
  long bz = std::min({256l,M/nR,M/nC});

  auto A =  make_distributed_array<local_Array_t>(world, {nR,nC}, {M,M}, {bz,bz}, true);
  auto B =  make_distributed_array<local_Array_t>(world, {nR,nC}, {M,M}, {bz,bz}, true);
  auto C =  make_distributed_array<local_Array_t>(world, {nR,nC}, {M,M}, {bz,bz}, true);

  {
    local_Array_t At(M,M);
    h5::file h5f("slate_mat.h5",'r');
    h5::group grp(h5f);
    nda::h5_read(grp,"matrix",At);
    A.local() = At(A.local_range(0),A.local_range(1));
  }

  B.local() = A.local();

  inverse(A);

  multiply(A,B,C);

  auto Cloc = C.local();
  double s=0.0;
  double smax=0.0;
  for( auto [i,in] : itertools::enumerate(C.local_range(0)) ) 
    for( auto [j,jn] : itertools::enumerate(C.local_range(1)) ) {
      s += std::abs( Cloc(i,j)-(in==jn?1.0:0.0) );
      smax = std::max(smax,std::abs( Cloc(i,j)-(in==jn?1.0:0.0) ));
    }
  world.all_reduce_in_place_n(&s,1,std::plus<>{});
  world.all_reduce_in_place_n(&smax,1,boost::mpi3::max<>{});
  app_log(0," nR:{}, nC:{}, av:{}, max:{}",nR,nC,s,smax);

  return 0;
}
