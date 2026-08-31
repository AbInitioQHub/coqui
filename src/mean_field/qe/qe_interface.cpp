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
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "nda/h5.hpp"
#include "qe_system.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/InputParser.hpp"
#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/concepts.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"
#include "mean_field/mf_source.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

namespace mf
{
namespace qe 
{

qe_system read_xml(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                   std::string outdir, std::string prefix,
                   int nbnd_overwrite,
                   bool no_q_sym) 
{
  app_log(3," Reading QE xml file.");

  InputParser parser;
// check that file exists and exit with a message otherwise
  try {
    parser.read(outdir+"/"+prefix+".xml");
  } catch (std::exception const& e) {
    std::cerr<<" Filename: " <<outdir+"/"+prefix+".xml" <<std::endl;
    APP_ABORT("Error parsing qe xml file. Check format.");
  }  

  auto pt = parser.get_root();

  /* 
   * atomic species
   */ 
  auto ntyp = io::get_value<int>(pt, "qes:espresso.output.atomic_species.ntyp");

  std::vector<std::string> species;
  for(auto const& it : io::find_child(pt, "qes:espresso.output.atomic_species")) {
    if(it.first == "species") 
      species.push_back(io::get_value<std::string>(it.second,"name"));   
  }
  utils::check(species.size() == ntyp, "Size mismatch."); 

  /* 
   * atomic structure 
   */ 
  auto nat = io::get_value<int>(pt, "qes:espresso.output.atomic_structure.nat"); 
  auto alat = io::get_value<double>(pt, "qes:espresso.output.atomic_structure.alat"); 
  double tpiba = 2.0 * 3.14159265358979 / alat;
 
  nda::array<double, 2> at_pos(nat,3); 
  nda::array<int, 1> at_ids(nat);
  for(auto const& it : io::find_child(pt, "qes:espresso.output.atomic_structure.atomic_positions")) {
    if(it.first == "atom") {
      auto index = io::get_value<int>(it.second, "index") - 1;
      auto name = io::get_value<std::string>(it.second, "name");
      if(auto p = std::ranges::find(species, name); p != species.end()) {
        at_ids(index) = std::distance(species.begin(),p);
        auto xyz = io::str2vec<double>( it.second.get_value<std::string>() );
        utils::check(xyz.size() == 3, "Error parsing atomic positions from QE xml file.");
        at_pos(index,0) = xyz[0];
        at_pos(index,1) = xyz[1];
        at_pos(index,2) = xyz[2];
      } else
        APP_ABORT("Error: Could not find atomic species in QE xml: "+name);
    }
  }

  nda::array<double, 2> lattv(3,3);
  {
    auto a1 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.atomic_structure.cell.a1"));
    utils::check(a1.size() == 3, "Error parsing cell vectors from QE xml file.");
    auto a2 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.atomic_structure.cell.a2"));
    utils::check(a2.size() == 3, "Error parsing cell vectors from QE xml file.");
    auto a3 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.atomic_structure.cell.a3"));
    utils::check(a3.size() == 3, "Error parsing cell vectors from QE xml file.");
    for(int i=0; i<3; i++) {
      lattv(0,i) = a1[i];
      lattv(1,i) = a2[i];
      lattv(2,i) = a3[i];
    }
  }

  /* 
   * basis set
   */ 
  auto ecrho = io::get_value<double>(pt, "qes:espresso.output.basis_set.ecutrho");
  auto npwx = io::get_value<int>(pt, "qes:espresso.output.basis_set.npwx"); 
  auto ngm = io::get_value<int>(pt, "qes:espresso.output.basis_set.ngm"); 
  auto ngms = io::get_value<int>(pt, "qes:espresso.output.basis_set.ngms"); 
  nda::array<int, 1> fft_mesh(3);
  fft_mesh(0) = io::get_value<int>(pt, "qes:espresso.output.basis_set.fft_grid.nr1"); 
  fft_mesh(1) = io::get_value<int>(pt, "qes:espresso.output.basis_set.fft_grid.nr2"); 
  fft_mesh(2) = io::get_value<int>(pt, "qes:espresso.output.basis_set.fft_grid.nr3"); 

  nda::array<double, 2> bg(3,3);
  {
    auto b1 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.basis_set.reciprocal_lattice.b1"));
    utils::check(b1.size() == 3, "Error parsing rec cell vectors from QE xml file.");
    auto b2 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.basis_set.reciprocal_lattice.b2"));
    utils::check(b2.size() == 3, "Error parsing rec cell vectors from QE xml file.");
    auto b3 = io::str2vec<double>(io::get_value<std::string>(pt, "qes:espresso.output.basis_set.reciprocal_lattice.b3"));
    utils::check(b3.size() == 3, "Error parsing rec cell vectors from QE xml file.");
    for(int i=0; i<3; i++) {
      bg(0,i) = b1[i]*tpiba;
      bg(1,i) = b2[i]*tpiba;
      bg(2,i) = b3[i]*tpiba;
    }
  }

  /*
   * magnetization: how does this compare to the same info in band structure???
   */ 
  //auto lsda = io::get_value<bool>(pt, "qes:espresso.output.magnetization.lsda"); 
  //auto noncolin = io::get_value<bool>(pt, "qes:espresso.output.magnetization.noncolin"); 
  //auto spinorbit = io::get_value<bool>(pt, "qes:espresso.output.magnetization.spinorbit"); 
  //auto magn = io::get_value<double>(pt, "qes:espresso.output.magnetization.total");  
  //auto abs_magn = io::get_value<double>(pt, "qes:espresso.output.magnetization.absolute");  
  

  /*
   * total_energy
   */ 
  std::map<std::string,double> total_energy;
  {
    std::string base = "qes:espresso.output.total_energy.";
    for( auto [i,v] : itertools::enumerate(std::vector<std::string>{"etot","eband","ehart","vtxc","etxc","ewald","demet"}) )
      total_energy[v] = io::get_value_with_default<double>(pt, base+v,0.0);  
  }

  /* 
   * band structure 
   */ 
  auto lsda = io::get_value<bool>(pt, "qes:espresso.output.band_structure.lsda"); 
  int nspin = (lsda?2:1);
  auto kp_grid = nda::array<int, 1>::zeros({3});
  kp_grid(0) = io::get_value_with_default<int>(pt,
		"qes:espresso.output.band_structure.starting_k_points.monkhorst_pack.nk1",0);
  kp_grid(1) = io::get_value_with_default<int>(pt,
		"qes:espresso.output.band_structure.starting_k_points.monkhorst_pack.nk2",0);
  kp_grid(2) = io::get_value_with_default<int>(pt,
		"qes:espresso.output.band_structure.starting_k_points.monkhorst_pack.nk3",0);
  auto nkpts = io::get_value<int>(pt, "qes:espresso.output.band_structure.nks");
  int nbnd = ( lsda ? io::get_value<int>(pt, "qes:espresso.output.band_structure.nbnd_up") : 
                      io::get_value<int>(pt, "qes:espresso.output.band_structure.nbnd"));
  // MAM: for now check that nbnd_up and nbnd_dw are the same, generalize if needed
  if( lsda ) {
    utils::check(nbnd == io::get_value<int>(pt, "qes:espresso.output.band_structure.nbnd_dw"), "Error: lsda limited to cases where nbnd_up == nbnw_dw for now.");
  }
  auto nelec = io::get_value<double>(pt, "qes:espresso.output.band_structure.nelec"); 
  auto noncolin = io::get_value<bool>(pt, "qes:espresso.output.band_structure.noncolin"); 
  auto spinorbit = io::get_value<bool>(pt, "qes:espresso.output.band_structure.spinorbit"); 
  auto efermi = io::get_value<double>(pt, "qes:espresso.output.band_structure.fermi_energy"); 

  // adjust nbnd_overwrite if needed
  if(nbnd_overwrite <= 0) nbnd_overwrite = nbnd;
  utils::check(nbnd_overwrite <= nbnd, "Error in qe_interface: nbnd_overwrite > nbnd: nbnd_overwrite:{}, nbnd:{}",nbnd_overwrite,nbnd);

  nda::array<double, 2> kpts(nkpts,3);
  nda::array<double, 1> k_weight(nkpts);
  nda::array<int, 1> npw(nkpts);
  nda::array<double, 3> eigval(nspin,nkpts, nbnd_overwrite);
  nda::array<double, 3> occ(nspin,nkpts, nbnd_overwrite);

  int nk=0;
  for(auto const& it : io::find_child(pt, "qes:espresso.output.band_structure")) {
    if(it.first == "ks_energies") {
      auto k = io::str2vec<double>(io::get_value<std::string>(it.second, "k_point"));
      utils::check(k.size() == 3, "Error parsing k_point from QE xml file.");
      for(int i=0; i<3; i++) kpts(nk,i) = k[i] * tpiba; 
      k_weight(nk) = io::get_value<double>(it.second,"k_point.weight");
      npw(nk) = io::get_value<int>(it.second,"npw");
      auto ev = io::str2vec<double>(io::get_value<std::string>(it.second, "eigenvalues"));
      utils::check(ev.size() == nspin*nbnd,"Error parsing eigenvalues from QE xml file.");
      auto o = io::str2vec<double>(io::get_value<std::string>(it.second, "occupations"));
      utils::check(o.size() == nspin*nbnd, "Error parsing occ vectors from QE xml file.");
      for(int it_=0, is=0; is<nspin; ++is) {
        for(int ib=0; ib<nbnd; ++ib, ++it_) {
          if(ib < nbnd_overwrite) {
            eigval(is, nk, ib) = ev[it_];
            occ(is, nk, ib) = o[it_];
          }
        }
      }
      nk++;
    }
  }
  // MAM: scale by 0.5 if nspin==1
  if(nspin==1)
    k_weight() *= 0.5;

  // symmetry
  auto noinv = io::get_value<bool>(pt, "qes:espresso.input.symmetry_flags.noinv");
  auto nsym = io::get_value<int>(pt, "qes:espresso.output.symmetries.nsym");     
  //auto nrot = io::get_value<int>(pt, "qes:espresso.output.symmetries.nrot");     
  //auto space_group = io::get_value<int>(pt, "qes:espresso.output.symmetries.space_group");     
  bool found_E = false;
  std::vector<utils::symm_op> symm_list;
  symm_list.reserve(nsym);
  nda::stack_array<double, 3, 3> R;
  nda::matrix<double> Rinv(3,3);
  nda::stack_array<double, 3> ft;
  for(auto const& it : io::find_child(pt, "qes:espresso.output.symmetries")) {
    if(it.first == "symmetry") {
      if(io::get_value<std::string>(it.second, "info") == "crystal_symmetry") {
        auto cls = io::get_value<std::string>(it.second, "info.class");
        //if(not noncolin) utils::check(cls != "not found","Error: crystal_symmetry with incorrect class:{}",cls);
        auto rot = io::str2vec<double>(io::get_value<std::string>(it.second, "rotation"));
        utils::check(rot.size() == 9, "Error parsing k_point from QE xml file.");
        auto tau = io::str2vec<double>(io::get_value<std::string>(it.second, "fractional_translation"));
        utils::check(tau.size() == 3, "Error parsing k_point from QE xml file.");
        for(int i=0,ij=0; i<3; i++) {
          ft(i) = tau[i];
          for(int j=0; j<3; j++,ij++) R(i,j) = rot[ij];
        }
        utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
                     " Error: Fractional translations not yet allowed. Use force_symmorphic=true.");
        // MAM: The rotation matrix is for vectors in the fft grid, already contains factors of lattv
        // S = inv(T(lattv)) * R * T(lattv), where r = T(lattv) * nfft
        // Then, to apply the rotation to vectors in the real space fft grid, we just need:
        // nr_new = S * nr,  (e.g. assuming r/nr are column vectors).
        // Similarly, for the fft grid in G space (or in k-space)
        // G = ng * recv, which leads to ng_new = ng * S, where G/ng are assumed row vectors.
        // So S can be applied to rotations in both r and G space.
        // Finally, S( inv(R) ) = inv(S)
        if( std::abs(
            std::pow(R(0,0)-1.0,2.0) + std::pow(R(0,1),2.0) + std::pow(R(0,2),2.0) +
            std::pow(R(1,0),2.0) + std::pow(R(1,1)-1.0,2.0) + std::pow(R(1,2),2.0) +
            std::pow(R(2,0),2.0) + std::pow(R(2,1),2.0) + std::pow(R(2,2)-1.0,2.0)) < 1e-12 ) {

          utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
                       "Error: Identity operation has non-zero ft:({},{},{})",ft(0),ft(1),ft(2));
          found_E = true;
          if(symm_list.size() == 0)
            symm_list.emplace_back(utils::symm_op{R,R,ft});
          else {
            symm_list.emplace_back(symm_list[0]);
            symm_list[0].R = R;
            symm_list[0].Rinv = R;
            symm_list[0].ft = ft;
          }
        } else {
          Rinv = R;
          nda::inverse3_in_place(Rinv);
          symm_list.emplace_back(utils::symm_op{R,Rinv,ft});
        }
      } 	 
    }
  }
  utils::check(found_E, "Error: Identity operation not found among symmetry list.");
  utils::check(symm_list.size() == nsym,
               "Error parsing symmetries from qe::xml. nsym:{}, # symmetries found:{}",nsym,symm_list.size());

  return qe_system(std::move(mpi),outdir,prefix,xml_input_type,no_q_sym,
		   alat,npwx,ngm,ngms,nelec,3,nspin,(noncolin?2:1),spinorbit,
		   ecrho,species,at_ids,at_pos,fft_mesh,
		   lattv,bg,kp_grid,kpts,k_weight,npw,eigval,occ,symm_list,efermi,not noinv);
}

qe_system read_h5(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi,
                  std::string outdir, 
                  std::string prefix,
                  int nbnd_overwrite)
{
  using nda::range;
  app_log(3," Reading QE h5 file.");

  h5::file file = h5::file(outdir+"/"+prefix+".coqui.h5", 'r');
  h5::group grp(file);
  utils::check(grp.has_subgroup("System"), "Error in read_h5: Missing System group in h5 file: {}",
               outdir+"/"+prefix+".coqui.h5");

  double alat = 1.0;                    // lattv units in QE, not used in this code.
  int nbnd = 0;                         // number of bands
  int natoms = 0;                       // # of atoms
  int nspecies = 0;                             // # of species
  int npwx = 0;                         // maximum number of pws in wfn grid
  int ngm = 0;                          // number of g vectors
  int ngms = 0;                         // number of g vectors on smooth grid
  double nelec = 0.0;                   // integrated number of electrons
  int ndims = 3;                        // number of dimensions
  int nspin = 0;                        // # of spins (1:closed/non-collinear, 2:collinear)
  int npol = 0;
  bool spinorbit = false;               // spin orbit coupling
  double ecrho = 0.0;
  double enuc=0.0;                      // nuclear energy
  double efermi = 0.0;

  nda::stack_array<int,3> fft_mesh;

  h5::group sgrp = grp.open_group("System"); 
  // unit cell
  h5::h5_read_attribute(sgrp, "number_of_atoms", natoms);
  h5::h5_read_attribute(sgrp, "number_of_species", nspecies);
  //if(sgrp.has_key("number_of_dimensions"))
  if( H5Aexists(h5::hid_t(sgrp),"number_of_dimensions") )
    h5::h5_read_attribute(sgrp, "number_of_dimensions", ndims);
  h5::h5_read_attribute(sgrp, "number_of_spins", nspin);
  if( H5Aexists(h5::hid_t(sgrp),"number_of_polarizations") )
    h5::h5_read_attribute(sgrp, "number_of_polarizations", npol);
  else
    npol = 1;
  h5::h5_read_attribute(sgrp, "nuclear_energy", enuc);
  if( H5Aexists(h5::hid_t(sgrp),"fermi_energy") )
    h5::h5_read_attribute(sgrp, "fermi_energy", efermi);
  h5::h5_read_attribute(sgrp, "number_of_elec", nelec);
  int lso = 0;
  if( H5Aexists(h5::hid_t(sgrp),"lspinorbit") )
    h5::h5_read_attribute(sgrp, "lspinorbit", lso);
  spinorbit = (lso!=0);
  auto species = std::vector<std::string>(nspecies);
  auto at_ids  = nda::array<int, 1>::zeros({natoms});
  auto at_pos  = nda::array<double, 2>::zeros({natoms, 3});
  h5::h5_read(grp, "System/species", species);
  nda::h5_read(grp, "System/atomic_id", at_ids);
  nda::h5_read(grp, "System/atomic_positions", at_pos);
  nda::array<double, 2> lattv(3,3);
  nda::array<double, 2> bg(3,3);
  nda::h5_read(grp, "System/lattice_vectors", lattv);
  nda::h5_read(grp, "System/reciprocal_vectors", bg);

  // MAM: temporary fix until new pw2coqui.f90 is in place, remove eventually
  // should be 0-based indexing always
  {
    int id_min = *std::min_element(at_ids.begin(),at_ids.end());
    utils::check(id_min==0 or id_min==1, 
                 "qe_interface::read_h5 Invalid atomic_ids array: min id:{}.",id_min);
    at_ids() -= id_min; 
  }

  // BZ info
  int nkpts;
  int nkpts_ibz;
  h5::group bgrp = sgrp.open_group("BZ"); 
  h5::h5_read_attribute(bgrp, "number_of_kpoints", nkpts);
  h5::h5_read_attribute(bgrp, "number_of_kpoints_ibz", nkpts_ibz);
  auto kpts = nda::array<double, 2>::zeros({nkpts_ibz,3});
  nda::h5_read(grp, "System/BZ/kpoints", kpts);
  auto kp_grid = nda::array<int, 1>::zeros({3});
  if( grp.has_key("System/BZ/kp_grid") )
    nda::h5_read(grp, "System/BZ/kp_grid", kp_grid);

  std::vector<utils::symm_op> symm_list;
  // symmetries
  {
    bool found_E = false;
    int nsym;
    nda::stack_array<double,3,3> R;
    nda::matrix<double> Rinv(3,3);
    nda::stack_array<double,3> ft;
    h5::group bsgrp = bgrp.open_group("Symmetries"); 
    h5::h5_read_attribute(bsgrp, "number_of_symmetries",nsym);
    symm_list.clear();
    symm_list.reserve(nsym);
    for(int i=0; i<nsym; i++) {
      nda::h5_read(sgrp, "BZ/Symmetries/s"+std::to_string(i)+"/R",R);
      nda::h5_read(sgrp, "BZ/Symmetries/s"+std::to_string(i)+"/ft",ft);
      utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
          " Error: Fractional translations not yet allowed. Use force_symmorphic=true.");
      if( std::abs(
            std::pow(R(0,0)-1.0,2.0) + std::pow(R(0,1),2.0) + std::pow(R(0,2),2.0) +
            std::pow(R(1,0),2.0) + std::pow(R(1,1)-1.0,2.0) + std::pow(R(1,2),2.0) +
            std::pow(R(2,0),2.0) + std::pow(R(2,1),2.0) + std::pow(R(2,2)-1.0,2.0)
          ) < 1e-12 ) {
        utils::check(std::abs(ft(0)*ft(0)+ft(1)*ft(1)+ft(2)*ft(2)) < 1e-12,
                     "Error: Identity operation has non-zero ft:({},{},{})",ft(0),ft(1),ft(2));
        found_E = true;
        if(symm_list.size() == 0)
          symm_list.emplace_back(utils::symm_op{R,R,ft});
        else {
          symm_list.emplace_back(symm_list[0]);
          symm_list[0].R = R;
          symm_list[0].Rinv = R;
          symm_list[0].ft = ft;
        }
      } else {
        Rinv = R;
        nda::inverse3_in_place(Rinv);
        symm_list.emplace_back(utils::symm_op{R,Rinv,ft});
      }
    }  
    utils::check(found_E, "Error: Identity operation not found among symmetry list.");
    utils::check(symm_list.size() == nsym,
                 "Error parsing symmetries from qe::xml. nsym:{}, # symmetries found:{}",
                 nsym,symm_list.size());
  }

  // Orbitals 
  h5::group ogrp = grp.open_group("Orbitals"); 
  { // some checks
    int dummy;
    h5::h5_read_attribute(ogrp, "number_of_spins", dummy);
    utils::check(dummy==nspin, "Inconsisten nspin - System:{}, Orbitals:{}",nspin,dummy);
    if( H5Aexists(h5::hid_t(ogrp),"number_of_polarizations") ) {
      h5::h5_read_attribute(ogrp, "number_of_polarizations", dummy);
      utils::check(dummy==npol, "Inconsisten npol - System:{}, Orbitals:{}",npol,dummy);
    } else {
      utils::check(1==npol, "Inconsisten npol - System:{}, Orbitals:{}",1,npol);
    }
    h5::h5_read_attribute(ogrp, "number_of_kpoints", dummy);
    utils::check(dummy==nkpts, "Inconsisten nkpts - System:{}, Orbitals:{}",nkpts,dummy);
    h5::h5_read_attribute(ogrp, "number_of_kpoints_ibz", dummy);
    utils::check(dummy==nkpts_ibz, "Inconsisten nkpts_ibz - System:{}, Orbitals:{}",nkpts_ibz,dummy);
  }

  nda::array<double, 1> k_weight;
  nda::h5_read(grp, "System/kpoint_weights", k_weight);
  utils::check( k_weight.extent(0) >= nkpts_ibz, "Error with k_weight: Incorrect dimensions in h5.");
  auto twist = nda::array<double, 1>::zeros({3});
  if( grp.has_key("System/twist") )
    nda::h5_read(grp, "System/twist", twist);

  h5::h5_read_attribute(ogrp, "number_of_bands", nbnd);
  h5::h5_read_attribute(ogrp, "ecutrho", ecrho);
  nda::h5_read(grp, "Orbitals/fft_mesh", fft_mesh);
  h5::h5_read_attribute(ogrp, "npwx", npwx);
  nda::array<int, 1> npw(npwx);
  nda::h5_read(grp, "Orbitals/npw", npw);

  // adjust nbnd_overwrite if needed
  if(nbnd_overwrite <= 0) nbnd_overwrite = nbnd;
  utils::check(nbnd_overwrite <= nbnd, "Error in qe_interface: nbnd_overwrite > nbnd: nbnd_overwrite:{}, nbnd:{}",nbnd_overwrite,nbnd);


  auto eigval = nda::array<double, 3>::zeros({nspin, nkpts_ibz, nbnd_overwrite});
  auto occ    = nda::array<double, 3>::zeros({nspin, nkpts_ibz, nbnd_overwrite});
  {
    nda::array<double, 3> e_;
    nda::h5_read(grp, "Orbitals/eigval", e_);
    utils::check( e_.extent(0) == nspin and 
                  e_.extent(1) >= nkpts_ibz and e_.extent(2) >= nbnd_overwrite, 
                  "Error with eigval: Incorrect dimensions in h5.");
    // Mapping from IBZ-FBZ is done by qe_system, so only provide IBZ data
    eigval() = e_(range(nspin),range(nkpts_ibz),range(nbnd_overwrite));
    nda::h5_read(grp, "Orbitals/occ", e_);
    utils::check( e_.extent(0) == nspin and 
                  e_.extent(1) >= nkpts_ibz and e_.extent(2) >= nbnd_overwrite, 
                  "Error with occ: Incorrect dimensions in h5.");
    occ() = e_(range(nspin),range(nkpts_ibz),range(nbnd_overwrite));
  }

  bool noinv = false;
  //if(sgrp.has_key("noinv")) 
  if(H5Aexists(h5::hid_t(sgrp),"noinv")) {
    int noinv_;
    h5::h5_read_attribute(sgrp, "noinv", noinv_);
    noinv = (noinv_==1);
  }

  std::map<std::string,double> total_energy;
  total_energy[std::string("ewald")] = enuc;

  if( mf::bz_symm::can_init_from_h5(sgrp) ) {
    mf::bz_symm bz(sgrp); 
    return qe_system(std::move(mpi),outdir,prefix,h5_input_type,false,
                     alat,npwx,ngm,ngms,nelec,ndims,nspin,npol,spinorbit,
                     ecrho,species,at_ids,at_pos,fft_mesh,
                     lattv,bg,kp_grid,kpts,k_weight,npw,eigval,occ,symm_list,efermi,bz);
  } else {
    return qe_system(std::move(mpi),outdir,prefix,h5_input_type,false,
                     alat,npwx,ngm,ngms,nelec,ndims,nspin,npol,spinorbit,
                     ecrho,species,at_ids,at_pos,fft_mesh,
                     lattv,bg,kp_grid,kpts,k_weight,npw,eigval,occ,symm_list,efermi,not noinv);
  } 
}

} // qe
} // mf
