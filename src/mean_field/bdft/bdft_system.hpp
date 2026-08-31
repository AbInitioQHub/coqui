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


#ifndef COQUI_BDFT_SYSTEM_HPP
#define COQUI_BDFT_SYSTEM_HPP

#include <string>
#include <filesystem>
#include <map>
#include "configuration.hpp"

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include "nda/h5.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "utilities/mpi_context.h"
#include "utilities/check.hpp"
#include "utilities/concepts.hpp"

#include "mean_field/mf_source.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace mf {
  namespace bdft {
    /**
     * Struct that contains details of a BDFT mean-field calculation, read from hdf5 files
     */
    struct bdft_system {
      public:

      bdft_system(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
                  std::string outdir_, std::string prefix_, int _nbnd) :
      mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_), filename(outdir_ + "/" + prefix_ + ".h5")
      {
        utils::check(_nbnd!=0, "Error in bdft_system: nbnd==0.");
        utils::check(std::filesystem::exists(filename),
		     " Error in bdft_system: filename: {} does not exist.",filename);
        initialize_from_h5(_nbnd);
      }

      template<class MF>
      bdft_system(MF const& mf, std::string fn, bool to_h5) : filename(fn)
      {
        initialize_from_MF(mf);
        if(to_h5) {
          h5::file file = h5::file(filename, 'w');
          h5::group grp(file);
          save(grp);
          _symm.save(grp);
        }
      }

      bdft_system(const bdft_system& ) = default;
      bdft_system(bdft_system&& ) = default;

      bdft_system& operator=(const bdft_system& ) = default;
      bdft_system& operator=(bdft_system&& ) = default;

      ~bdft_system() = default;

      auto const& bz() const { return _symm; }

      void save(std::string fn) const {
        h5::file file = h5::file(fn, 'w');
        h5::group grp(file);
        save(grp);
      } 

      // MPI handler for communicators
      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

      // bdft output directory
      std::string outdir = "";
      // bdft prefix
      std::string prefix = "";
      // name of file with input data
      std::string filename = "";
      // type of input file, hardwired to h5
      mf_input_file_type_e input_file_type = h5_input_type;    

      // kpt and Qpt information
      bz_symm _symm;

      // unit cell info
      int natoms = 0;
      // # of unique species
      int nspecies = 0;
      // # of electrons
      double nelec = 0.0;
      // name/ids of species
      std::vector<std::string> species;
      // array with the id(Z!?) of atoms in unit cell
      nda::array<int, 1> at_ids;
      // array of atom positions
      nda::array<double, 2> at_pos;
      // lattice vectors: a_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> latt;
      // reciprocal vectors: b_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> recv;
      // madelung constant for finite-size corrections
      double madelung = 0.0;
      // nuclear energy
      double enuc = 0.0;
      // fermi energy
      double efermi = 0.0;

      // weight of kpoint in averages
      nda::array<double, 1> k_weight;
 
      // basis info
      // # of primary single-particle basis
      int nbnd = 0;
      // # of auxiliary single-particle basis
      int nbnd_aux = 0;
      // orbitals are stored on fft grid or not
      bool orb_on_fft_grid = true;
      // plane-wave cutoff for charge density
      double ecutrho;
      // fft grid
      nda::stack_array<int, 3> fft_mesh;
      // number of points in fft grid
      int nnr;

      // mean-field info
      // # of dimensions
      int ndims = 3;
      // # of spin
      int nspin = 0;
      // # of spin in basis set
      int nspin_in_basis = 0;
      // # of polarizations 
      int npol = 0;
      // # of polarizations in basis set
      int npol_in_basis = 0;
      // eigenvalues: [nspin, kpoint, npol*band]
      nda::array<double, 3> eigval;
      // eigenvalues: [nspin, kpoint, npol*band]
      nda::array<double, 3> eigval_aux;
      // eigenvalues: [nspin, kpoint, npol*band]
      nda::array<double, 3> aux_weight;
      // occupation numbers: [nspin, kpoint, npol*band]
      nda::array<double, 3> occ;
      // expect SO term in pseudo?
      bool spinorbit = false;

      public:
      // dummy members, just to be consistent with qe_readonly class...
      // may use them in the future
      double alat = 0.0;

      void save(h5::group& grp) const
      {
        h5::group sgrp = (grp.has_subgroup("System") ?
                          grp.open_group("System")    :
                          grp.create_group("System", true));

        //BZ
        _symm.save(sgrp);

        // unit cell
        h5::h5_write_attribute(sgrp, "number_of_atoms", natoms);
        h5::h5_write_attribute(sgrp, "number_of_species", nspecies);
        h5::h5_write_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_write_attribute(sgrp, "number_of_spins", nspin);
        h5::h5_write_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_polarizations", npol);
        h5::h5_write_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_bands", nbnd);
        h5::h5_write_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_write_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_write_attribute(sgrp, "nuclear_energy", enuc);
        h5::h5_write_attribute(sgrp, "fermi_energy", efermi);
        h5::h5_write(sgrp, "species", species);
        nda::h5_write(sgrp, "atomic_id", at_ids, false);
        nda::h5_write(sgrp, "atomic_positions", at_pos, false);
        nda::h5_write(sgrp, "lattice_vectors", latt, false);
        nda::h5_write(sgrp, "reciprocal_vectors", recv, false);

        nda::h5_write(sgrp, "kpoint_weights", k_weight, false);

        int nkpts = _symm.nkpts; 
        int nkpts_ibz = _symm.nkpts_ibz; 
        h5::group ogrp = (grp.has_subgroup("Orbitals") ?
                          grp.open_group("Oritals")    :
                          grp.create_group("Orbitals", true));
        h5::h5_write_attribute(ogrp, "number_of_spins", nspin);
        h5::h5_write_attribute(ogrp, "number_of_spins_in_basis", nspin_in_basis);
        h5::h5_write_attribute(ogrp, "number_of_polarizations", npol);
        h5::h5_write_attribute(ogrp, "number_of_polarizations_in_basis", npol_in_basis);
        h5::h5_write_attribute(ogrp, "number_of_kpoints", nkpts);
        h5::h5_write_attribute(ogrp, "number_of_kpoints_ibz", nkpts_ibz);
        h5::h5_write_attribute(ogrp, "number_of_bands", nbnd);
        h5::h5_write_attribute(ogrp, "number_of_aux_bands", nbnd_aux);
        h5::h5_write_attribute(ogrp, "ecutrho", ecutrho);
        nda::h5_write(ogrp, "fft_mesh", fft_mesh, false);

        nda::h5_write(ogrp, "eigval", eigval, false);
        if(nbnd_aux>0) nda::h5_write(ogrp, "eigval_aux", eigval_aux, false);
        if(nbnd_aux>0) nda::h5_write(ogrp, "aux_weight", aux_weight, false);
        nda::h5_write(ogrp, "occ", occ, false);
      }

      private:
      void initialize_from_h5(int _nbnd = -1)
      {
        using nda::range;
        decltype(nda::range::all) all;
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);

        h5::group sgrp = grp.open_group("System"); 
        h5::group ogrp = grp.open_group("Orbitals"); 

        // unit cell
        h5::h5_read_attribute(sgrp, "number_of_atoms", natoms);
        h5::h5_read_attribute(sgrp, "number_of_species", nspecies);
        //if(sgrp.has_key("number_of_dimensions")) 
        if(H5Aexists(h5::hid_t(sgrp),"number_of_dimensions"))
          h5::h5_read_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_read_attribute(sgrp, "number_of_spins", nspin);
        //if(sgrp.has_key("number_of_spins_in_basis")) 
        if(H5Aexists(h5::hid_t(sgrp),"number_of_spins_in_basis"))
          h5::h5_read_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        else
          nspin_in_basis = nspin;
        utils::check( nspin > 0, "Error: nspin: {}",nspin);
        utils::check( nspin_in_basis <= nspin and nspin_in_basis > 0, "Error: nspin_in_basis: {}",nspin_in_basis);
        if(H5Aexists(h5::hid_t(sgrp),"number_of_polarizations"))
          h5::h5_read_attribute(sgrp, "number_of_polarizations", npol);
        else
          npol = 1;
        if(H5Aexists(h5::hid_t(sgrp),"number_of_polarizations_in_basis"))
          h5::h5_read_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        else
          npol_in_basis = npol;
        utils::check( npol > 0, "Error: npol: {}",npol);
        utils::check( npol_in_basis <= npol and npol_in_basis > 0, "Error: nspin_in_basis: {}",nspin_in_basis);

        h5::h5_read_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_read_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_read_attribute(sgrp, "nuclear_energy", enuc);
        if( H5Aexists(h5::hid_t(sgrp),"fermi_energy") )
          h5::h5_read_attribute(sgrp, "fermi_energy", efermi);
        species = std::vector<std::string>(nspecies);
        at_ids  = nda::array<int, 1>(natoms);
        at_pos  = nda::array<double, 2>(natoms, 3);
        h5::h5_read(grp, "System/species", species); 
        nda::h5_read(grp, "System/atomic_id", at_ids);
        nda::h5_read(grp, "System/atomic_positions", at_pos);
        nda::h5_read(grp, "System/lattice_vectors", latt);
        nda::h5_read(grp, "System/reciprocal_vectors", recv);

        // BZ
        _symm.initialize_from_h5(sgrp);  

        int nkpts = _symm.nkpts;
        int nkpts_ibz = _symm.nkpts_ibz;
        // BZ info
        k_weight = nda::array<double, 1>(nkpts);
        nda::h5_read(grp, "System/kpoint_weights", k_weight);
        if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 and mpi->comm.root() )
          app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

        // Orbitals 
        { // some checks
          int dummy;
          h5::h5_read_attribute(ogrp, "number_of_spins", dummy);
          utils::check(dummy==nspin, "Inconsisten nspin - System:{}, Orbitals:{}",nspin,dummy);
// MAM: add attribute check to nda
//          if(sgrp.has_key("number_of_spins_in_basis")) {
//            h5::h5_read_attribute(sgrp, "System/number_of_spins_in_basis", dummy);
//            utils::check(dummy==nspin_in_basis, 
//              "Inconsisten nspin_in_basis - System:{}, Orbitals:{}",nspin_in_basis,dummy);
//          }
          h5::h5_read_attribute(ogrp, "number_of_kpoints", dummy);
          utils::check(dummy==nkpts, "Inconsisten nkpts - System:{}, Orbitals:{}",nkpts,dummy);
          h5::h5_read_attribute(ogrp, "number_of_kpoints_ibz", dummy);
          utils::check(dummy==nkpts_ibz, "Inconsisten nkpts_ibz - System:{}, Orbitals:{}",nkpts_ibz,dummy);
        }
        h5::h5_read_attribute(ogrp, "number_of_bands", nbnd);
        utils::check(_nbnd!=0, "Error in bdft_system: nbnd==0.");
        if(_nbnd > 0) {
          utils::check(_nbnd <= nbnd, 
                       "Error in initialize_from_h5: nbnd:{} nbnd in h5:{}",_nbnd,nbnd);
          nbnd = _nbnd;
        }
          
// MAM: add attribute check to nda
//        if(ogrp.has_key("number_of_aux_bands")) 
//          h5::h5_read_attribute(ogrp, "number_of_aux_bands", nbnd_aux);
        h5::h5_read_attribute(ogrp, "ecutrho", ecutrho);
        nda::h5_read(grp, "Orbitals/fft_mesh", fft_mesh);
        // adjust ,esh if needed
        fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,_symm.symm_list,1e-4,"bdft_system");
        fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,_symm.symm_list,1e-4,"bdft_system",true);
        nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);

        // MAM: should this be size nspin_in_basis?
        //      In any case, if nspin_in_basis!=nspin, the basis can't be MO
        //      so occ and eigval are not meaningful anyway
        eigval = nda::array<double, 3>(nspin, nkpts, npol*nbnd);
        eigval_aux = nda::array<double, 3>(nspin, nkpts, npol*nbnd_aux);
        aux_weight = nda::array<double, 3>(nspin, nkpts, npol*nbnd_aux);
        occ    = nda::array<double, 3>(nspin, nkpts, npol*nbnd);
        {
          nda::array<double, 3> e_;
          nda::h5_read(grp, "Orbitals/eigval", e_);
          utils::check( e_.extent(0) == nspin and
                        e_.extent(1) >= nkpts_ibz and e_.extent(2) >= nbnd,
                        "Error with eigval: Incorrect dimensions in h5.");
          eigval(all,range(nkpts_ibz),all) = e_(range(nspin),range(nkpts_ibz),range(nbnd)); 
          nda::h5_read(grp, "Orbitals/occ", e_);
          utils::check( e_.extent(0) == nspin and
                        e_.extent(1) >= nkpts_ibz and e_.extent(2) >= nbnd,
                        "Error with eigval: Incorrect dimensions in h5.");
          occ(all,range(nkpts_ibz),all) = e_(range(nspin),range(nkpts_ibz),range(nbnd));   
          for(int ik=nkpts_ibz; ik<nkpts; ik++) {
            int ik_ibz = _symm.kp_to_ibz(ik);
            eigval(all,ik,all) = eigval(all,ik_ibz,all);
            occ(all,ik,all)    = occ(all,ik_ibz,all);
          }
        }

        if(nbnd_aux>0) {
          nda::h5_read(grp, "Orbitals/eigval_aux",eigval_aux);
          nda::h5_read(grp, "Orbitals/aux_weight", aux_weight);
        }

      }

      template<class MF>
      void initialize_from_MF(MF const& mf) 
      {
        mpi = mf.mpi();

        // BZ
        _symm.initialize_from_MF(mf);

        // unit cell
        natoms   = mf.number_of_atoms();
        nspecies = mf.number_of_species();
        ndims    = mf.ndims();
        nspin    = mf.nspin();
        nspin_in_basis    = mf.nspin_in_basis();
        npol    = mf.npol();
        npol_in_basis    = mf.npol_in_basis();
        nelec    = mf.nelec();
        species  = mf.species();
        at_ids   = mf.atomic_id();
        at_pos   = mf.atomic_positions();
        latt     = mf.lattv();
        recv     = mf.recv();
        madelung = mf.madelung();
        enuc     = 0;
        efermi   = mf.efermi();

        k_weight   = mf.k_weight();
        if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 )
          app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

        // Orbital info
        nbnd     = mf.nbnd();
        nbnd_aux = mf.nbnd_aux();
        ecutrho  = mf.ecutrho();
        fft_mesh = mf.fft_grid_dim();
        // adjust ,esh if needed
        fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,_symm.symm_list,1e-4,"bdft_system");
        fft_mesh = utils::generate_consistent_fft_mesh(fft_mesh,_symm.symm_list,1e-4,"bdft_system",true);
        nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);

        eigval     = mf.eigval(); 
        eigval_aux = mf.eigval_aux(); 
        aux_weight = mf.aux_weight(); 
        occ        = mf.occ();

      }

    }; // bdft_system

  } // bdft
} // mf

#endif //COQUI_PYSCF_SYSTEM_HPP
