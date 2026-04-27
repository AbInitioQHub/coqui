/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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
#include <stdexcept>
#include <stack>
#include "cxxopts.hpp"
#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/ptree/InputParser.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "IO/app_loggers.h"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "orbitals/orbital_generator.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/pproc/pproc_drivers.hpp"
#include "methods/pproc/hamiltonians.h"
#include "methods/MBPT_drivers.h"
#include "methods/pproc/wavefunction_utils.h"
#include "wannier/wan90.h"

namespace mpi3 = boost::mpi3;

template<MEMORY_SPACE MEM>
void run(mpi3::communicator & comm, InputParser &parser);

/** @file main.cpp
 */
int main(int argc, char** argv)
{
  mpi3::environment env(argc, argv);
  auto world = mpi3::environment::get_world_instance();
  bool root(world.root());

  // compute backend
  std::string compute = "default"; 

  // parse command line inputs
  std::vector<std::string> inputs;
  int output_level, debug_level; 
  //if (world.root()) // everyone parse input file (e.g. hdf can be read in parallel)
  { // parse command line inputs
    cxxopts::Options options(argv[0], "FlatIron Quantum Many-body Acceleration Framework");
    options
      .positional_help("[optional args]")
      .show_positional_help();
    options.add_options()
      ("h,help", "print help message")
      ("compute", "cpu, gpu, unified or default", cxxopts::value<std::string>()->default_value("default"))
      ("verbosity", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("2"))
      ("debug", "0, 1, 2, ...: higher means more", cxxopts::value<int>()->default_value("0"))
      ("stacktrace", "print stack trace if available: default:true", cxxopts::value<bool>()->default_value("true"))
      ("filenames", "input filenames", cxxopts::value<std::vector<std::string>>())
    ;
    options.parse_positional({"filenames"});
    auto args = options.parse(argc, argv);
    // record program options
    if (args.count("help"))
    {
      if(root)
        std::cout << options.help({"", "Group"}) << std::endl;
      mpi3::environment::finalize();
      exit(0);
    }

    // compute 
    compute = args["compute"].as<std::string>();
    io::tolower(compute);

    // output level 
    output_level = args["verbosity"].as<int>();
    // check program options
    if (output_level < 0) 
    {
      std::cerr << "verbosity < 0: " << output_level << std::endl;
      mpi3::environment::finalize();
      exit(1);
    }
 
    // debug level
    debug_level = args["debug"].as<int>();
    if (debug_level < 0) 
    {
      std::cerr << "debug < 0: " << debug_level << std::endl;
      mpi3::environment::finalize();
      exit(1);
    }

    // stack trace control
    set_stacktrace(args["stacktrace"].as<bool>());

    // input files are positional arguments
    int nfile = args.count("filenames");
    if (nfile < 1)
    {
      if(root)
        std::cout << "no input file given; exiting ..." << std::endl;
      mpi3::environment::finalize();
      exit(1);
    } else {
      inputs = args["filenames"].as<std::vector<std::string>>();
    }
  }

  // setup output loggers
  setup_loggers(world.root(), output_level, debug_level);

  std::string welcome(
      std::string("\n ---------------------------------\n") +
                  "     ____ ___   ___  _   _ ___ \n" +
                  "    / ___/ _ \\ / _ \\| | | |_ _|\n" +
                  "   | |  | | | | | | | | | || | \n" +
                  "   | |__| |_| | |_| | |_| || | \n" +
                  "    \\____\\___/ \\__\\_\\\\___/|___|\n" +
                  "  --------------------------------\n" +
                  " |  Correlated Quantum Interface  |\n" +
                  "  --------------------------------");
  app_log(1, welcome);

  // !!!! assume a single input for now
  std::string myinput = inputs[0];
  InputParser parser;
  try {
    parser.read(myinput);
  } catch (std::exception const& e) {
    app_error("Error parsing input file. Check format.");
    mpi3::environment::finalize();
    exit(1);	
  }

  // dispatch based on compute 
  if(compute == "default") {
    run<DEFAULT_MEMORY_SPACE>(world,parser);
  } else if(compute == "cpu") { 
    run<HOST_MEMORY>(world,parser);
#if defined(ENABLE_DEVICE)
  } else if(compute == "gpu") { 
    run<DEVICE_MEMORY>(world,parser);
#endif
#if defined(ENABLE_UNIFIED_MEMORY)
  } else if(compute == "unified") { 
    run<UNIFIED_MEMORY>(world,parser);
#endif
  } else {
    std::cerr << " Invalid command line argument - compute: " <<compute <<std::endl;
    mpi3::environment::finalize();
    exit(1);
  }

  return 0;
}


// process input file...
template<MEMORY_SPACE MEM>
void run(mpi3::communicator &comm, InputParser &parser)
{
  using methods::thc_reader_t;
  using methods::chol_reader_t;
  // container of MF and eri objects. Enables reuse in multiple input blocks! 
  std::map<std::string, std::shared_ptr<mf::MF>> mf_list;
  std::map<std::string, std::tuple<std::string,std::unique_ptr<thc_reader_t>>> thc_list;
  std::map<std::string, std::tuple<std::string,std::unique_ptr<chol_reader_t>>> chol_list;

  auto mpi_context = std::make_shared<utils::mpi_context_t<>>(utils::make_mpi_context(comm));

  for (auto const& it : parser.get_root())
  { // go through all executable blocks 
    std::string cname = it.first;
    if (cname == "mean_field") {

      ptree pt = it.second;
      if (auto v = pt.get_value_optional<std::string>())
        utils::check(*v == "", "mean_field reference not allowed at top level.");
      for (auto const& mf_it : pt) {
        std::string mf_type = mf_it.first;
        ptree mf_pt = mf_it.second;
        utils::check(!mf_pt.empty(), "Every entry of \'mean_field\' should be a node.");
        auto name = mf::add_mf(mpi_context, mf_pt, mf_type, mf_list, true);
      }

    } else if (cname == "interaction") {

      ptree pt = it.second;
      if (auto v = pt.get_value_optional<std::string>())
        utils::check(*v == "", "interaction reference not allowed at top level.");
      for (auto const& int_it : pt) {
        std::string int_type = int_it.first;
        ptree int_pt = int_it.second;
        utils::check(!int_pt.empty(), "Every entry of \'interaction\' should be a node.");
        if (int_type == "thc") {
          auto name = methods::add_thc(mpi_context, int_pt, mf_list, thc_list);
        } else if (int_type == "cholesky") {
          auto name = methods::add_cholesky(mpi_context, int_pt, mf_list, chol_list);
        } else
          APP_ABORT("Error: Invalid interaction type: {}",int_type);
      }

    } else if (cname == "isdf") {

      ptree pt = it.second;
      auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
      methods::make_isdf(mf_list[mf_name], pt);

    } else if (cname == "orbitals") {

      ptree pt = it.second; 
      auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
      orbitals::orbital_factory(*mf_list[mf_name],pt);

    } else if (cname == "mp2") {

      ptree pt = it.second; 
      app_error("calculation type: {} not implemented yet \n",cname.c_str());

    } else if (cname == "hf" or cname == "qphf" or cname == "rpa" or cname == "gw" or cname == "qpgw" or cname == "gw_dca"
               or cname == "evgw" or cname == "gf2") {

      // all based on mbpt, lump together
      ptree pt = it.second;
      auto [eri_name, eri_type] = methods::get_eri_block(mpi_context, pt, mf_list,
                                                         thc_list, chol_list, "interaction");
      utils::check(eri_name != "" and eri_type != "", 
                   "Error: Failed to find interaction block needed by {}",cname);
      auto [hf_eri_name, hf_eri_type] = methods::get_eri_block(mpi_context, pt, mf_list,
                                                               thc_list, chol_list, "interaction_hf");
      auto [hartree_eri_name, hartree_eri_type] = methods::get_eri_block(mpi_context, pt, mf_list,
                                                               thc_list, chol_list, "interaction_hartree");
      auto [exchange_eri_name, exchange_eri_type] = methods::get_eri_block(mpi_context, pt, mf_list,
                                                               thc_list, chol_list, "interaction_exchange");

      auto mf_name = (eri_type=="thc")? std::get<0>(thc_list[eri_name]) : std::get<0>(chol_list[eri_name]);

      if (hf_eri_type=="" and hartree_eri_type=="" and exchange_eri_type=="") {
        // consistent eri for all
        if (eri_type == "thc") {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(chol_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        }

      } else if (hf_eri_type!="") {

        // separate eri for hf and post-hf
        if (hf_eri_type=="thc" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(thc_list[hf_eri_name]), *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else if (hf_eri_type=="thc" and eri_type=="cholesky") {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(thc_list[hf_eri_name]), *std::get<1>(chol_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else if (hf_eri_type=="cholesky" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(chol_list[hf_eri_name]), *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else {
          auto mb_eri = methods::mb_eri_t(*std::get<1>(chol_list[hf_eri_name]), *std::get<1>(chol_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        }

      } else if (hartree_eri_type!="" and exchange_eri_type!="") {

        // separate eri for J, K and post-hf
        if (hartree_eri_type=="thc" and exchange_eri_type=="thc" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(
              *std::get<1>(thc_list[hartree_eri_name]),
              *std::get<1>(thc_list[exchange_eri_name]),
              *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else if (hartree_eri_type=="cholesky" and exchange_eri_type=="thc" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(
              *std::get<1>(chol_list[hartree_eri_name]),
              *std::get<1>(thc_list[exchange_eri_name]),
              *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else if (hartree_eri_type=="thc" and exchange_eri_type=="cholesky" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(
              *std::get<1>(thc_list[hartree_eri_name]),
              *std::get<1>(chol_list[exchange_eri_name]),
              *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else if (hartree_eri_type=="cholesky" and exchange_eri_type=="cholesky" and eri_type=="thc") {
          auto mb_eri = methods::mb_eri_t(
              *std::get<1>(chol_list[hartree_eri_name]),
              *std::get<1>(chol_list[exchange_eri_name]),
              *std::get<1>(thc_list[eri_name]));
          methods::mbpt(cname, mb_eri, pt);
        } else {
          APP_ABORT("main::run: Unrecognized interaction setup for mbpt. "
                    "hf_eri_type = {}, hartree_eri_type = {}, exchange_eri_type = {}, eri_type = {}",
                    hf_eri_type, hartree_eri_type, exchange_eri_type, eri_type);
        }
      } else {
        APP_ABORT("main::run: Unrecognized interaction setup for mbpt. "
                  "hf_eri_type = {}, hartree_eri_type = {}, exchange_eri_type = {}, eri_type = {}",
                  hf_eri_type, hartree_eri_type, exchange_eri_type, eri_type);
      }

    } else if (cname == "downfold_1e") {

      ptree pt = it.second;
      auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
      methods::downfolding_1e(mf_list[mf_name], pt);

    } else if (cname == "downfold_2e") {

      ptree pt = it.second;
      if (auto name = methods::get_thc(mpi_context, pt, mf_list, thc_list)) {
        auto mf_name = std::get<0>(thc_list[*name]);
        methods::downfolding_2e(*std::get<1>(thc_list[*name]), pt);
      } else {
        APP_ABORT(cname+": Could not find interaction block.");
      }

    } else if (cname == "hf_downfold") {

      ptree pt = it.second;
      if (auto name = methods::get_thc(mpi_context, pt, mf_list, thc_list)) {
        methods::hf_downfold(*std::get<1>(thc_list[*name]), pt);
      } else {
        APP_ABORT(cname+": Could not find integral block.");
      }

    } else if (cname == "gw_downfold") {

      ptree pt = it.second;
      if (auto name = methods::get_thc(mpi_context, pt, mf_list, thc_list)) {
        methods::gw_downfold(*std::get<1>(thc_list[*name]), pt);
      } else {
        APP_ABORT(cname+": Could not find integral block.");
      }

    } else if (cname == "dmft_embed") {

      ptree pt = it.second;
      auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
      methods::dmft_embed_with_projector_from_h5(mf_list[mf_name], pt);

    } else if (cname == "ac" or cname == "unfold_bz"
               or cname == "band_interpolation" or cname == "spectral_interpolation" or cname == "local_dos"
               or cname == "dump_vxc" or cname == "dump_hartree") { // more pproc options would be added.

      ptree pt = it.second;
      auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
      methods::post_processing(cname, mf_list[mf_name], pt);

    } else if (cname == "unfold_wfc") {

      ptree pt = it.second;
      auto mf_sym_name = mf::get_mf(mpi_context, pt, mf_list, "mean_field_sym");
      auto mf_nosym_name = mf::get_mf(mpi_context, pt, mf_list, "mean_field_nosym");
      methods::unfold_wfc(*mf_list[mf_sym_name], *mf_list[mf_nosym_name]);

    } else if (cname == "hamiltonian") {

      ptree pt = it.second;

      auto [eri_name, eri_type] = methods::get_eri_block(mpi_context, pt, mf_list,
                                                         thc_list, chol_list, "interaction");
      if(eri_type == "thc") {
        auto mf_name = std::get<0>(thc_list[eri_name]);
        // adds H0 if requested 
        methods::add_core_hamiltonian(*mf_list[mf_name],pt);
        // factorizes Vuv and adds it to the file. 
        // If requested, calculates and writes half-transformed integrals in THC format. 
        methods::add_thc_hamiltonian_components(*mf_list[mf_name],*std::get<1>(thc_list[eri_name]),pt);
      } else if(eri_type == "cholesky") {
        auto mf_name = std::get<0>(chol_list[eri_name]);
        // adds H0 if requested 
        methods::add_core_hamiltonian(*mf_list[mf_name],pt);
        // nothing to do, AFQMC code can read /Interaction directly
      } else {
        // right now this will abort if no mf is found. Find fix!
        auto mf_name = mf::get_mf(mpi_context, pt, mf_list);
        // adds H0 if requested 
        methods::add_core_hamiltonian(*mf_list[mf_name],pt);
      }

      // add interation term if requested (possible to add outside this block, 
      // only necessary to do here if changing storage formats 

      // add interacting wfn if requested (from MBPT calculation)

    } else if (cname == "wavefunction") {

      ptree pt = it.second;
      if (auto v = pt.get_value_optional<std::string>())
        utils::check(*v == "", "wavefunction reference not allowed at top level.");
      for (auto const& wfn_it : pt) {
        std::string wfn_type = wfn_it.first;
        ptree wfn_pt = wfn_it.second; 
        utils::check(!wfn_pt.empty(), "Every entry of \'wavefunction\' should be a node.");
        if (wfn_type == "mf") {
          auto mf_name = mf::get_mf(mpi_context,wfn_pt, mf_list);
          if(mpi_context->comm.root()) {
            auto output = io::get_value_with_default<std::string>(wfn_pt,"output","wfn.h5");
            h5::file file;
            try {
              file = h5::file(output, 'a');
            } catch(...) {
              APP_ABORT("Failed to open h5 file: {}, mode:a",output);
            }
            h5::group grp(file);
            methods::add_wavefunction(grp,*mf_list[mf_name],wfn_pt);
          }
          mpi_context->comm.barrier();
        } else
          APP_ABORT("Error: Invalid wavefunction type: {}",wfn_type);
      }

    } else if (cname == "wannier90") {
  
      ptree pt = it.second;
      if (auto v = pt.get_value_optional<std::string>())
        utils::check(*v == "", "wannier90 reference not allowed at top level.");
      for (auto const& wann_it : pt) {
        std::string wann_type = wann_it.first;
        ptree wann_pt = wann_it.second;
        utils::check(!wann_pt.empty(), "Every entry of \'wannier90\' should be a node.");
        if (wann_type == "append_win") {
          auto mf_name = mf::get_mf(mpi_context, wann_pt, mf_list);
          wannier::append_wannier90_win(*mf_list[mf_name], wann_pt); 
        } else if (wann_type == "converter") {
          auto mf_name = mf::get_mf(mpi_context, wann_pt, mf_list);
          wannier::to_wannier90(*mf_list[mf_name],wann_pt);
        } else if (wann_type == "library_mode") {
#if defined(ENABLE_WANNIER90)
          // only assumes *.win file with options 
          auto mf_name = mf::get_mf(mpi_context, wann_pt, mf_list);
          wannier::wannier90_library_mode(*mf_list[mf_name],wann_pt);
#else
          APP_ABORT("Error: wannier90.library_mode without wannier90 support. Recompile with ENABLE_WANNIER90=ON.");
#endif
        } else if (wann_type == "library_mode_from_nnkp") {
#if defined(ENABLE_WANNIER90)
          auto mf_name = mf::get_mf(mpi_context, wann_pt, mf_list);
          wannier::wannier90_library_mode_from_nnkp(*mf_list[mf_name],wann_pt);
#else
          APP_ABORT("Error: wannier90.library_mode without wannier90 support. Recompile with ENABLE_WANNIER90=ON."); 
#endif
        } else if (wann_type == "mlwf_h5") {
          auto mf_name = mf::get_mf(mpi_context, wann_pt, mf_list);
          wannier::mlwf_h5_from_wannier90_output(*mf_list[mf_name], wann_pt);
        } else
          APP_ABORT("Error: Invalid wannier90 type: {}",wann_type);
      }
      mpi_context->comm.barrier();

    } else {

      app_error("unknown calculation type: {} \n",cname.c_str());
      mpi3::environment::finalize();
      exit(1);	

    }
  } 
}

