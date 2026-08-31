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




#define CATCH_CONFIG_RUNNER
#include "catch2/catch.hpp"
//#include "catch2/catch_test_macros.hpp"

#include<iostream>
#include <cstdlib>
#include <memory>

#include "utilities/mpi_context.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "arch/arch.h" 

std::string qe_prefix, qe_outdir;
std::string bdft_prefix, bdft_outdir;
std::string pyscf_prefix, pyscf_outdir;

namespace utils::detail {
  // gets allocated in utilities/test_common.hpp when requested, cleanup below before exiting main
  std::shared_ptr<utils::mpi_context_t<boost::mpi3::communicator>> __unit_test_mpi_context__ = nullptr;
}

int main(int argc, char* argv[])
{
  boost::mpi3::environment env(argc, argv);
  auto world = boost::mpi3::environment::get_world_instance();

  int output_level=2, debug_level=2;
  if(const char* env_p = std::getenv("OUTPUT_LEVEL")) {
    output_level = std::atoi(env_p);     
    if(output_level < 0) output_level=2;
    if(output_level > 5) output_level=2;
  }
  if(const char* env_p = std::getenv("DEBUG_LEVEL")) {
    debug_level = std::atoi(env_p);     
    if(debug_level < 0) debug_level=2;
    if(debug_level > 5) debug_level=2;
  }
  arch::init(world.root(),output_level,debug_level);

  // from Catch2 docs
  Catch::Session session;

  // Build a new parser on top of Catch2's
  using namespace Catch::clara;

  auto cli
    = session.cli()           
    | Opt( qe_prefix, "qe_prefix" ) 
        ["--qe_prefix"]    
        ("QE prefix.")        
    | Opt( qe_outdir, "qe_outdir" ) 
        ["--qe_outdir"]    
        ("QE outdir.")        
    | Opt( pyscf_prefix, "pyscf_prefix" ) 
        ["--pyscf_prefix"]    
        ("PYSCF prefix.")        
    | Opt( pyscf_outdir, "pyscf_outdir" ) 
        ["--pyscf_outdir"]    
        ("PYSCF outdir.")        
    | Opt( bdft_prefix, "bdft_prefix" ) 
        ["--bdft_prefix"]    
        ("BDFT prefix.")        
    | Opt( bdft_outdir, "bdft_outdir" ) 
        ["--bdft_outdir"]    
        ("BDFT outdir.");        
  session.cli(cli);

  // Let Catch2 (using Clara) parse the command line
  int returnCode = session.applyCommandLine( argc, argv );
  if( returnCode != 0 ) // Indicates a command line error
      return returnCode;

  auto ret = session.run( argc, argv );

  // cleanup mpi context 
  utils::detail::__unit_test_mpi_context__.reset();

  return ret;
}
