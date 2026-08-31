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
#include <complex>

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "cxxopts.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "nda/nda.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "grids/g_grids.hpp"
#include "methods/ERI/cholesky.h"

namespace mpi3 = boost::mpi3;

int main(int argc, char* argv[])
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  
  int output_level, debug_level;
  cxxopts::Options options(argv[0], "Utility to test/time cholesky factorizations.");
  options
    .positional_help("[optional args]")
    .show_positional_help();
  options.add_options()
      ("h,help", "print help message")
      ("verbosity", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("2"))
      ("debug", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("0"))
      ("mftype", "type of MF: qe", cxxopts::value<std::string>()->default_value("qe"))
      ("outdir","QE outdir",cxxopts::value<std::string>()->default_value("./"))
      ("prefix","QE prefix",cxxopts::value<std::string>()->default_value("pwscf"))
      ("cutoff","Cholesky cutoff",cxxopts::value<double>()->default_value("1e-4"))
      ("bmin","Minimum block size",cxxopts::value<double>()->default_value("1"))
      ("bmax","Maximum block size",cxxopts::value<double>()->default_value("64"))
  ;
  auto args = options.parse(argc, argv);
  // record program options
  if (args.count("help"))
  {
    if(world.root())
      std::cout << options.help({"", "Group"}) << std::endl;
    mpi3::environment::finalize();
    exit(0);
  }
  output_level = args["verbosity"].as<int>();
  debug_level = args["debug"].as<int>();
  // setup loggers 
  setup_loggers(world.root(), output_level, debug_level); 
  if (args["mftype"].as<std::string>() == "qe")
  {
    mf::qe::qe_readonly qe(world, args["outdir"].as<std::string>(),args["prefix"].as<std::string>());
    mf::MF mf(qe);

    int bmin = std::max(1,args["bmin"].as<int>());
    int bmax = std::min(int(mf.nbnd()),args["bmax"].as<int>());
    for(int i=bmin; i<=bmax; i*=2)
    {
      app_log(2,"Block Size: {}",i);
      methods::cholesky chol(std::addressof(mf),world,args["cutoff"].as<double>(),mf.ecutrho(),i);
      auto L = chol.evaluate<HOST_MEMORY>(0);
      chol.print_timers();
    } 
  } else {
    app_error("Error: Unknown mftype");
  }

  return 0;
}

