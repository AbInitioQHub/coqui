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


#ifndef MEANFIELD_QE_QE_SYSTEM_H
#define MEANFIELD_QE_QE_SYSTEM_H

#include <map>
#include <string>
#include "IO/AppAbort.hpp"
#include "configuration.hpp"
#include <nda/nda.hpp>
#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/madelung_utils.hpp"
#include "utilities/concepts.hpp"

#include "mean_field/mf_source.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace mf
{
namespace qe
{

/* 
 * Class that contains details of QE calculations read from xml file.
 */
struct qe_system
{
  public:

  qe_system(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
            std::string out, std::string pre,
            mf_input_file_type_e input_file_type_,
            bool no_q_sym,
            double alat_, int npwx_, int ng,
            int ngms_, double nelec_, int ndims_, int nspin_,
            int npol_, bool spinorbit_,
            double ecrho,
            std::vector<std::string> sp,
            nda::array<int, 1> ids,
            nda::array<double, 2> pos,
            nda::array<int, 1> mesh,
            nda::array<double, 2> latt_,
            nda::array<double, 2> bg,
            nda::array<double, 1> kp_grid_,
            nda::array<double, 2> kpts_,
            nda::array<double, 1> k_weight_,
            nda::array<int, 1> npw_,
            nda::array<double, 3> eigval_,
            nda::array<double, 3> occ_,
            std::vector<utils::symm_op> symm_list_,
            double efermi_,
            bool use_trev):
        mpi(std::move(mpi_)),
        outdir(out),
        prefix(pre),
        filename(outdir+"/"+prefix+".coqui.h5"),
        input_file_type(input_file_type_),
        ndims(ndims_),
        nspin(nspin_),
        nspin_in_basis(nspin),      // no choice in this backend
        npol(npol_), 
        npol_in_basis(npol),
        nbnd(eigval_.shape()[2]), 
        nbnd_aux(0),
        natoms(pos.shape()[0]),
        nspecies(sp.size()),
        alat(alat_),
        nnr(mesh(0)*mesh(1)*mesh(2)),
        npwx(npwx_),
        ngm(ng),
        ngms(ngms_),
        nelec(nelec_),
        spinorbit(spinorbit_),
        species(sp),
        at_ids(ids),
        at_pos(pos),
        efermi(efermi_),
        ecutrho(ecrho),
        fft_mesh(mesh),
        latt(latt_),
        recv(bg),
        symm(mpi->comm,no_q_sym,latt,recv,kp_grid_,kpts_,symm_list_,use_trev)
  {
    decltype(nda::range::all) all;
    utils::check(ndims==2 or ndims==3, "Invalid ndims:{}",ndims);
    utils::check(at_ids.shape()[0] == natoms, "Wrong dimensions in qe_system constructor.");
    utils::check(at_pos.shape()[0] == natoms, "Wrong dimensions in qe_system constructor.");
    utils::check(at_pos.shape()[1] == 3, "Wrong dimensions in qe_system constructor.");
    utils::check(eigval_.shape()[0] == nspin, "Wrong dimensions in qe_system constructor.");
    utils::check(occ_.shape()[0] == nspin, "Wrong dimensions in qe_system constructor.");
    utils::check(occ_.shape()[2] == nbnd, "Wrong dimensions in qe_system constructor.");

    // adjust ,esh if needed
    fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,symm.symm_list,1e-4,"qe_system");
    fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,symm.symm_list,1e-4,"qe_system",true);

    int nkpts = symm.nkpts;
    int nkpts_ibz = symm.nkpts_ibz;
    k_weight = nda::array<double, 1>::zeros({nkpts_ibz});
    npw      = nda::array<int, 1>::zeros({nkpts});
    eigval   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd});
    eigval_aux   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd_aux});
    aux_weight   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd_aux});
    occ      = nda::array<double, 3>::zeros({nspin,nkpts,nbnd});
    for(int ik=0; ik<nkpts; ik++) {
      int ik_ibz = symm.kp_to_ibz(ik);
      npw(ik)            = npw_(ik_ibz);
      eigval(all,ik,all) = eigval_(all,ik_ibz,all);
      occ(all,ik,all)    = occ_(all,ik_ibz,all);
    } 
    k_weight() = k_weight_(nda::range(nkpts_ibz)); 
    if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 and mpi->comm.root())
      app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

    // compute Madelung constant
    auto rg = nda::range(ndims);
    madelung = -2 * utils::madelung(latt(rg,rg), recv(rg,rg), symm.kp_grid(rg), fft_mesh(rg), 1e-10);

    // temporary
    if( (nkpts_ibz < nkpts) and npol>1 )
      APP_ABORT("Error in qe_system: npol>1 not yet available with kpoint symmetries.");
  }

  qe_system(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
            std::string out, std::string pre,
            mf_input_file_type_e input_file_type_,
            [[maybe_unused]] bool no_q_sym,
            double alat_, int npwx_, int ng,
            int ngms_, double nelec_, int ndims_, int nspin_,
            int npol_, bool spinorbit_,
            double ecrho,
            std::vector<std::string> sp,
            nda::array<int, 1> ids,
            nda::array<double, 2> pos,
            nda::array<int, 1> mesh,
            nda::array<double, 2> latt_,
            nda::array<double, 2> bg,
            [[maybe_unused]] nda::array<double, 1> kp_grid_,
            [[maybe_unused]] nda::array<double, 2> kpts_,
            nda::array<double, 1> k_weight_,
            nda::array<int, 1> npw_,
            nda::array<double, 3> eigval_,
            nda::array<double, 3> occ_,
            [[maybe_unused]] std::vector<utils::symm_op> symm_list_,
            double efermi_,
            bz_symm const& bz_):
        mpi(std::move(mpi_)),
        outdir(out),
        prefix(pre),
        filename(outdir+"/"+prefix+".coqui.h5"),
        input_file_type(input_file_type_),
        ndims(ndims_),
        nspin(nspin_),
        npol(npol_),                  // no choice right now, hardcoded in QE
        npol_in_basis(npol),
        nbnd(eigval_.shape()[2]),
        nbnd_aux(0),
        natoms(pos.shape()[0]),
        nspecies(sp.size()),
        alat(alat_),
        nnr(mesh(0)*mesh(1)*mesh(2)),
        npwx(npwx_),
        ngm(ng),
        ngms(ngms_),
        nelec(nelec_),
        spinorbit(spinorbit_),
        species(sp),
        at_ids(ids),
        at_pos(pos),
        efermi(efermi_),
        ecutrho(ecrho),
        fft_mesh(mesh),
        latt(latt_),
        recv(bg),
        symm(bz_)
  {
    decltype(nda::range::all) all;
    utils::check(ndims==2 or ndims==3, "Invalid ndims:{}",ndims);
    utils::check(at_ids.shape()[0] == natoms, "Wrong dimensions in qe_system constructor.");
    utils::check(at_pos.shape()[0] == natoms, "Wrong dimensions in qe_system constructor.");
    utils::check(at_pos.shape()[1] == 3, "Wrong dimensions in qe_system constructor.");
    utils::check(eigval_.shape()[0] == nspin, "Wrong dimensions in qe_system constructor.");
    utils::check(occ_.shape()[0] == nspin, "Wrong dimensions in qe_system constructor.");
    utils::check(occ_.shape()[2] == nbnd, "Wrong dimensions in qe_system constructor.");

    // adjust ,esh if needed
    fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,symm.symm_list,1e-4,"qe_system");
    fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,symm.symm_list,1e-4,"qe_system",true);

    int nkpts = symm.nkpts;
    int nkpts_ibz = symm.nkpts_ibz;
    k_weight = nda::array<double, 1>::zeros({nkpts_ibz});
    npw      = nda::array<int, 1>::zeros({nkpts});
    eigval   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd});
    eigval_aux   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd_aux});
    aux_weight   = nda::array<double, 3>::zeros({nspin,nkpts,nbnd_aux});
    occ      = nda::array<double, 3>::zeros({nspin,nkpts,nbnd});
    for(int ik=0; ik<nkpts; ik++) {
      int ik_ibz = symm.kp_to_ibz(ik);
      npw(ik)            = npw_(ik_ibz);
      eigval(all,ik,all) = eigval_(all,ik_ibz,all);
      occ(all,ik,all)    = occ_(all,ik_ibz,all);
    } 
    k_weight() = k_weight_(nda::range(nkpts_ibz)); 
    if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 and mpi->comm.root() )
        app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

    // compute Madelung constant
    auto rg = nda::range(ndims);
    madelung = -2 * utils::madelung(latt(rg,rg), recv(rg,rg), symm.kp_grid(rg), fft_mesh(rg), 1e-10);

    // temporary
    if( (nkpts_ibz < nkpts) and npol>1 )
      APP_ABORT("Error in qe_system: npol>1 not yet available with kpoint symmetries.");
  }

  qe_system(qe_system const&) = default;
  qe_system(qe_system &&) = default;

  ~qe_system() = default;

  qe_system& operator=(qe_system const&) = default;
  qe_system& operator=(qe_system &&) = default;

  bz_symm const& bz() const { return symm; }

  void save(std::string fn) const {
    h5::file file = h5::file(fn, 'w');
    h5::group grp(file);
    save(grp);
  }

  void save(h5::group& grp) const
  { 
     h5::group sgrp = (grp.has_subgroup("System") ?
                       grp.open_group("System")    :
                       grp.create_group("System", true));
        
    //BZ
    symm.save(sgrp);

    // unit cell
    h5::h5_write_attribute(sgrp, "number_of_atoms", natoms);
    h5::h5_write_attribute(sgrp, "number_of_species", nspecies);
    h5::h5_write_attribute(sgrp, "number_of_dimensions", ndims);
    h5::h5_write_attribute(sgrp, "number_of_spins", nspin);
    h5::h5_write_attribute(sgrp, "number_of_elec", nelec);
    h5::h5_write_attribute(sgrp, "madelung_constant", madelung);
    h5::h5_write_attribute(sgrp, "nuclear_energy", enuc);
    h5::h5_write_attribute(sgrp, "fermi_energy", efermi);
    h5::h5_write_attribute(sgrp, "number_of_bands", nbnd);
    h5::h5_write(sgrp, "species", species);
    nda::h5_write(sgrp, "atomic_id", at_ids, false);
    nda::h5_write(sgrp, "atomic_positions", at_pos, false);
    nda::h5_write(sgrp, "lattice_vectors", latt, false);
    nda::h5_write(sgrp, "reciprocal_vectors", recv, false);

    nda::h5_write(sgrp, "kpoint_weights", k_weight, false);
        
    int nkpts = symm.nkpts;
    int nkpts_ibz = symm.nkpts_ibz;
    h5::group ogrp = (grp.has_subgroup("Orbitals") ?
                      grp.open_group("Oritals")    :
                      grp.create_group("Orbitals", true));
    h5::h5_write_attribute(ogrp, "number_of_spins", nspin);
    h5::h5_write_attribute(ogrp, "number_of_kpoints", nkpts);
    h5::h5_write_attribute(ogrp, "number_of_kpoints_ibz", nkpts_ibz);
    h5::h5_write_attribute(ogrp, "number_of_bands", nbnd);
    h5::h5_write_attribute(ogrp, "ecutrho", ecutrho);
    h5::h5_write_attribute(ogrp, "npwx", npwx);
    nda::h5_write(ogrp, "fft_mesh", fft_mesh, false);
    nda::h5_write(ogrp, "npw", npw, false);
        
    nda::h5_write(ogrp, "eigval", eigval, false);
    nda::h5_write(ogrp, "occ", occ, false);
  }

  // MPI handler for communicators
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

  // QE output directory
  std::string outdir = "";

  // QE prefix
  std::string prefix = "";

  // filename
  std::string filename = "";

  // type of input file, hardwired to h5
  mf_input_file_type_e input_file_type;  

  // basic info
  int ndims = 3;                        // number of dimensions
  int nspin = 0;                        // # of spins (1:closed/non-collinear, 2:collinear)
  int nspin_in_basis = 0;               // # number of spins in basis set
  int npol = 0;                         // # of polarization (1:closed/collinear, 2:non-collinear)
  int npol_in_basis = 0;                // # number of polarization in basis set
  int nbnd = 0;                         // actual number of bands in coqui calculation
  int nbnd_aux = 0;                     // number of auxiliary bands
  int natoms = 0;                       // # of atoms
  int nspecies = 0;		                // # of species
  double alat = 0.0;                    // lattv units in QE, not used in this code.
  // MAM: nnr should be long!
  int nnr = 0;                         // number of points in fft grid
  int npwx = 0;                         // maximum number of pws in wfn grid
  int ngm = 0;                          // number of g vectors
  int ngms = 0;                         // number of g vectors on smooth grid
  double nelec = 0.0;                   // integrated number of electrons
  bool spinorbit = false;               // spin orbit coupling
  std::vector<std::string> species;     // name/ids of species
  nda::array<int, 1> at_ids;            // array with the id of atoms in unit cell
  nda::array<double, 2> at_pos;	       	// array of atom positions
  double madelung = 0.0;                // madelung constant for finite-size corrections
  double enuc=0.0;                      // nuclear energy
  double efermi=0.0;                    // fermi energy

  bool orb_on_fft_grid = true;          // orbitals are stored on FFT grid or not
  double ecutrho = 0.0;                 // plane-wave cutoffs

  // fft grid	 
  nda::stack_array<int,3> fft_mesh;
   
  // lattice vectors
  // a_n[i] = latt(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
  nda::stack_array<double, 3, 3> latt;

  // reciprocal vectors
  // b_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
  nda::stack_array<double, 3, 3> recv;

  // weight of kpoint in averages
  nda::array<double, 1> k_weight;

  // kpt and Qpt information
  bz_symm symm;

  // number of plane waves at a given k-point 
  nda::array<int, 1> npw;

  // eigenvalues: [spin, kpoint, band]
  nda::array<double, 3> eigval;

  // auxiliary eigenvalues: [spin, kpoint, band]
  nda::array<double, 3> eigval_aux;

  // weights for auxliary basis 
  nda::array<double, 3> aux_weight; 

  // occupation numbers: [spin, kpoint, band]
  nda::array<double, 3> occ;

};

} // qe
} // mf

#endif
