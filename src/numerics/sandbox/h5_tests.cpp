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
#include <filesystem>
#include "stdio.h"
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

#define __TIME__H5__
utils::TimerManager H5_Timer;
#include "numerics/distributed_array/h5.hpp"

namespace mpi3 = boost::mpi3;
using namespace math::nda;

int main(int argc, char* argv[])
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  setup_loggers(world.root(), 10, 10); 
  std::vector<std::string> inputs;

  cxxopts::Options options(argv[0], "HDF5 Performance Tests");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
    ("pgrid", "processor grid", cxxopts::value<std::vector<long>>())
    ("dims", "matrix dimensions", cxxopts::value<std::vector<long>>())
  ;
  auto args = options.parse(argc, argv);
  utils::check(args.count("pgrid") == 1 and args.count("dims") == 1, 
               "Error: pgrid and dims must be provided (... and only once each)");
  auto pg = args["pgrid"].as<std::vector<long>>();
  auto dims = args["dims"].as<std::vector<long>>();
  long rank = pg.size();;

  utils::check(rank > 0 and rank < 5 and rank == dims.size(), 
               "Error: pgrid must have same dimensions as dims");

  long np = std::accumulate(pg.begin(),pg.end(),1,std::multiplies<>{});
  utils::check(np == world.size(), "Error: np != world.size()");
  std::string fname = "__h5_test__.h5";
  if(world.root() and std::filesystem::exists(fname.c_str())) remove(fname.c_str());

  auto run = [&]<int R>(auto&& pg_, auto&& dims_) {
    std::array<long,R> p,d;
    std::copy_n(pg_.begin(),R,p.begin());
    std::copy_n(dims_.begin(),R,d.begin());
    using local_Array_t = nda::array<ComplexType,R>;
    long N = std::accumulate(d.begin(),d.end(),1,std::multiplies<>{});
    auto A =  make_distributed_array<local_Array_t>(world, p, d); 
    H5_Timer.clear_timers();
    if(world.root()) {
      std::cout<<"pgrid:       " <<p <<"\n";
      std::cout<<"matrix dims: " <<d <<"\n";
      h5::file h5f(fname,'w');
      h5::group g(h5f);
      math::nda::h5_write(g, "h5_test_distr", A);
      // baseline
      H5_Timer.start("LOC");
      auto Aloc = nda::array<ComplexType,1>::zeros({N});
      ::nda::h5_write(g,"h5_test_serial",Aloc,false);
      H5_Timer.stop("LOC");
      app_log(0," Data Size:           {} GB",double(N)*1.49011611938477e-08);
      app_log(0," Serial IO rate:      {}",double(N)*1.49011611938477e-08/double(H5_Timer.elapsed("LOC")));
      app_log(0," Distributed IO rate: {}",double(N)*1.49011611938477e-08/double(H5_Timer.elapsed("IO")));
      if(world.size()>1) app_log(0," Comm overhead:      {}",double(H5_Timer.elapsed("COMM"))/double(H5_Timer.elapsed("TOTAL")));
      H5_Timer.print_all();
    } else {
      h5::file h5f;
      h5::group g(h5f);
      math::nda::h5_write(g, "h5_test_distr", A);
    }
  };

  if(rank==1) run.template operator()<1>(pg,dims);
  if(rank==2) run.template operator()<2>(pg,dims);
  if(rank==3) run.template operator()<3>(pg,dims);
  if(rank==4) run.template operator()<4>(pg,dims);

  if(world.root() and std::filesystem::exists(fname.c_str())) remove(fname.c_str());
  return 0;
}
