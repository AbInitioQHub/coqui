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
#include <iomanip>
#include <fstream>
#include <ctime>

#include "configuration.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Timer.hpp"
#include "utilities/mpi_context.h"
#include "utilities/kpoint_utils.hpp"
#include "utilities/stl_utils.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"

#include "wannier/scdm.hpp"
#include "wannier/wan90.h"
#include "wannier/wan90_aux.hpp"

/** Utilities for Brillioun zone unfolding **/

namespace wannier {

/**
 * Generate wannier90 data files from MF object 
 * @param context - [INPUT]
 * @param mf - [INPUT] mean-field instance for all the metadata of the system
 * @param pt - [INPUT] property tree with input options 
 */
void to_wannier90(mf::MF &mf, ptree &pt)
{
  app_log(1, "*************************************************");
  app_log(1, "                Wannier90 Converter              "); 
  app_log(1, "*************************************************\n");

  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  // options
  auto write_mmn = io::get_value_with_default<bool>(pt,"write_mmn",true);
  auto write_amn = io::get_value_with_default<bool>(pt,"write_amn",true);
  auto write_eigv = io::get_value_with_default<bool>(pt,"write_eig",true);
  if( not( write_mmn or write_amn or write_eigv) ) return;

  auto& mpi = *(mf.mpi());

  utils::check(std::filesystem::exists(resolved_prefix+".nnkp"), 
               "Wannier90 nnkp file not found:{}",resolved_prefix+".nnkp");
  /*
   * Read nnkp file 
   */ 
  auto [kp_map, wann_kp, nnkpts, proj, band_list, auto_projections] = detail::read_nnkp(mpi,mf,resolved_prefix+".nnkp");

  if(write_mmn) {
    app_log(2, " - Computing orbital overlaps, Mmn"); 
    auto Mmn = detail::compute_mmn(mpi,mf,resolved_prefix,kp_map,wann_kp,nnkpts,band_list,false,true);
  }
  if(write_amn) {
    if(proj.size() > 0) {
      app_log(2, " - Computing initial projection matrix, Amn, with basis functions."); 
      auto Amn = detail::compute_amn_projections(mpi,mf,pt,kp_map,wann_kp,band_list,proj,false,true);
    } else if(auto_projections > 0) {
      app_log(2, " - Computing initial projection matrix, Amn, with SCDM."); 
      auto Amn = scdm(mpi,mf,pt,auto_projections,kp_map,wann_kp,band_list,false,true);
    } else 
      utils::check(false, "to_wannier90: No projections or auto_projections found in nnkp file.");
  }
  if(write_eigv) {
    auto eigv = detail::get_eig(mpi,mf,resolved_prefix,kp_map,band_list, true);
  }
  mpi.comm.barrier();
}

void wannier90_library_mode_from_nnkp(mf::MF &mf, ptree &pt)
{
  app_log(1, "*************************************************");
  app_log(1, "       Running Wannier90 in library-mode         "); 
  app_log(1, "      (assuming win and nnkp files exist)        "); 
  app_log(1, "*************************************************\n");

  auto& mpi = *(mf.mpi());
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto resolved_prefix = outdir + "/" + prefix;
  
  int nproj = 0;

  utils::check(std::filesystem::exists(resolved_prefix+".nnkp"), 
               "Wannier90 nnkp file not found:{}",resolved_prefix+".nnkp");
  utils::check(std::filesystem::exists(resolved_prefix+".win"), 
               "Wannier90 win file not found:{}",resolved_prefix+".win");

  /*
   * Read nnkp file 
   */ 
  auto [kp_map, wann_kp, nnkpts, proj, band_list, auto_projections] = detail::read_nnkp(mpi,mf,resolved_prefix+".nnkp");

  /*
   * Generate mmn, amn and eig files.
   */
  {
    app_log(2, " - Computing orbital overlaps, Mmn"); 
    auto Mmn = detail::compute_mmn(mpi,mf,resolved_prefix,kp_map,wann_kp,nnkpts,band_list,false,true);
  }
  {
    if(proj.size() > 0) {
      app_log(2, " - Computing initial projection matrix, Amn, with basis functions."); 
      nproj = proj.size();
      auto Amn = detail::compute_amn_projections(mpi,mf,pt,kp_map,wann_kp,band_list,proj,false,true);
    } else if(auto_projections > 0) {
      app_log(2, " - Computing initial projection matrix, Amn, with SCDM."); 
      nproj = auto_projections;
      auto Amn = scdm(mpi,mf,pt,auto_projections,kp_map,wann_kp,band_list,false,true);
    } else 
      utils::check(false, "to_wannier90: No projections or auto_projections found in nnkp file.");
  }
  { 
    auto eigv = detail::get_eig(mpi,mf,resolved_prefix,kp_map,band_list,true);
  }
  mpi.comm.barrier();

  /*
   * run wannier90 wannierization
   * Restricted to cases where nproj == num_wann. 
   * select_projections is not yet allowed. 
   */
  {
    auto [Pkam,wann_center,wann_spreads] = detail::wannier90_library_run_from_files(mpi, mf, pt, nnkpts.extent(1), nproj, kp_map, wann_kp, band_list,true);

    // write to file
  }

  app_log(1, "");
  app_log(1, "*************************************************");
  app_log(1, "                Done with Wannier90              "); 
  app_log(1, "*************************************************\n");
}

void wannier90_library_mode(mf::MF &mf, ptree &pt)
{
  app_log(1, "*************************************************");
  app_log(1, "       Running Wannier90 in library-mode         ");
  app_log(1, "*************************************************\n");

  auto& mpi = *(mf.mpi());
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto resolved_prefix = outdir + "/" + prefix;

  utils::check(std::filesystem::exists(resolved_prefix+".win"), 
               "Wannier90 win file not found:{}",resolved_prefix+".win");

  /*
   * 1. Read nnkp file 
   * 2. Generate mmn, amn and eig files.
   * 3. run wannier90 wannierization
   * 
   * Right now assumes that the number of projections is equal to the number of wannier functions.
   * Read select_projections to generalize this.
   */ 
  auto [Pkam,wann_center,wann_spreads] = detail::wannier90_library_run(mpi,mf,pt);
  mpi.comm.barrier();

  // write files 

  app_log(1, "");
  app_log(1, "*************************************************");
  app_log(1, "                Done with Wannier90              ");
  app_log(1, "*************************************************");

}

void append_wannier90_win(mf::MF &mf, ptree &pt)
{
  app_log(1, "*************************************************");
  app_log(1, "        Modifying Wannier90's *.win file         ");
  app_log(1, "*************************************************\n");

  auto& mpi = *(mf.mpi());
  if(mpi.comm.root()) {

    auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
    auto prefix = io::get_value<std::string>(pt, "prefix");
    auto resolved_prefix = outdir + "/" + prefix;
    utils::check(std::filesystem::exists(resolved_prefix+".win"), "Problems opening *win file:{}",resolved_prefix+".win");

    { // first check that input doesn't contain requested blocks
      auto file_data = utils::read_file_to_string(resolved_prefix+".win");

      if( io::get_value_with_default<bool>(pt,"atoms",true) ) 
        utils::check(file_data.find("begin atoms_cart") == std::string::npos and
                     file_data.find("begin atoms_frac") == std::string::npos,
                     "append_wannier90_win: atoms block already present. ");

      if( io::get_value_with_default<bool>(pt,"kpts",true) ) 
        utils::check(file_data.find("begin kpoints") == std::string::npos,
                     "append_wannier90_win: kpoints block already present. "); 

      if( io::get_value_with_default<bool>(pt,"cell",true) ) 
        utils::check(file_data.find("begin unit_cell_cart") == std::string::npos,
                     "append_wannier90_win: unit_cell_cart block already present. "); 
    }
  
    {
      std::ofstream out(resolved_prefix+".win", std::ios_base::app);
      utils::check(out.is_open(), "append_wannier90_win: Problems opening file: ",resolved_prefix+".win");    

      if( io::get_value_with_default<bool>(pt,"atoms",true) )  {
        auto species = mf.species();
        auto id = mf.atomic_id();
        auto pos = mf.atomic_positions();
        out<<"\n";
        out<<"begin atoms_cart \n";
        out<<"  Bohr  \n";
        for(int i=0; i<pos.extent(0); i++)
          out<<"  " <<species[id(i)] <<"  " 
             <<std::fixed <<std::setprecision(10) 
             <<std::setw(14)  <<pos(i,0) <<"  " 
             <<std::setw(14)  <<pos(i,1) <<"  " 
             <<std::setw(14)  <<pos(i,2) <<"\n";
        out<<"end atoms_cart\n\n";

      }
   
      if( io::get_value_with_default<bool>(pt,"kpts",true) ) {
        auto kp_grid = mf.kp_grid();
        auto kpts = mf.kpts_crystal();
        out<<"\n";
        out<<"mp_grid " <<kp_grid(0) <<" " <<kp_grid(1) <<" " <<kp_grid(2) <<"\n";
        out<<"begin kpoints \n";
        for(int i=0; i<kpts.extent(0); i++) 
          out<<"  " <<std::fixed <<std::setprecision(10) 
             <<std::setw(14) <<kpts(i,0) <<"  " 
             <<std::setw(14) <<kpts(i,1) <<"  " 
             <<std::setw(14) <<kpts(i,2) <<"\n";
        out<<"end kpoints\n\n";
      }

      if( io::get_value_with_default<bool>(pt,"cell",true) ) {
        out<<"\n";
        out<<"begin unit_cell_cart \n";
        out<<"  Bohr  \n";
        out<<std::setprecision(12) <<std::fixed
           <<"  " <<std::setw(18) <<mf.lattv(0,0) 
           <<"  " <<std::setw(18) <<mf.lattv(0,1) 
           <<"  " <<std::setw(18) <<mf.lattv(0,2) <<"\n"
           <<"  " <<std::setw(18) <<mf.lattv(1,0) 
           <<"  " <<std::setw(18) <<mf.lattv(1,1) 
           <<"  " <<std::setw(18) <<mf.lattv(1,2) <<"\n"
           <<"  " <<std::setw(18) <<mf.lattv(2,0) 
           <<"  " <<std::setw(18) <<mf.lattv(2,1) <<"  " 
           <<std::setw(18) <<mf.lattv(2,2) <<"\n";
        out<<"end unit_cell_cart\n\n";
      }
    }
  }
  mpi.comm.barrier();

  app_log(1, "");
  app_log(1, "*************************************************");
  app_log(1, "      Done appending Wannier90's *.win file      ");
  app_log(1, "*************************************************");
}

/**
 * Read Wannier90 standalone mode outputs and generate MLWF HDF5 file
 */
void mlwf_h5_from_wannier90_output(mf::MF &mf, ptree &pt)
{
  app_log(1, "*************************************************************");
  app_log(1, "   Reading Wannier90 Standalone Output & Generating HDF5   ");
  app_log(1, "*************************************************************\n");

  auto &mpi = *(mf.mpi());
  auto outdir = io::get_value_with_default<std::string>(pt, "outdir", "./");
  auto prefix = io::get_value<std::string>(pt, "prefix");
  auto resolved_prefix = outdir + "/" + prefix;

  utils::check(std::filesystem::exists(resolved_prefix + ".win"),
               "wan90.cpp::mlwf_h5_from_wannier90_output: Win file not found: {}.win",
               resolved_prefix);

  detail::mlwf_h5_from_wannier90_output_impl(mpi, mf, pt);
  mpi.comm.barrier();

  app_log(1, "");
  app_log(1, "*************************************************************");
  app_log(1, "     Wannier90 Standalone Output Successfully Processed     ");
  app_log(1, "*************************************************************");
}

}
