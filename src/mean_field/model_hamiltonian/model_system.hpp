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


#ifndef COQUI_MODEL_SYSTEM_HPP
#define COQUI_MODEL_SYSTEM_HPP

#include <string>
#include <filesystem>
#include <map>
#include "configuration.hpp"

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include "nda/h5.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

#include "utilities/check.hpp"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/mf_source.hpp"
#include "mean_field/symmetry/bz_symmetry.hpp"

namespace mf {
  namespace model {
    /**
     * Struct that contains details of a Model Hamiltonian, read from hdf5 files
     */
    struct model_system {
      public:

      model_system(std::shared_ptr<utils::mpi_context_t<> > mpi_, std::string outdir_, std::string prefix_,
                  int _nbnd) : 
      mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_), filename(outdir_ + "/" + prefix_ + ".h5")
      {
        utils::check(_nbnd!=0, "Error in model_system: nbnd==0.");
        utils::check(std::filesystem::exists(filename),
		     " Error in model_system: filename: {} does not exist.",filename);
        initialize_from_h5(_nbnd);
      }

      model_system([[maybe_unused]] std::shared_ptr<utils::mpi_context_t<> > mpi_, std::string outdir_, std::string prefix_,
                   bz_symm symm, int nspin_, int npol_, double _nel,  
                   nda::MemoryArrayOfRank<4> auto const& h,
                   nda::MemoryArrayOfRank<4> auto const& ov,
                   nda::MemoryArrayOfRank<4> auto const& d,
                   nda::MemoryArrayOfRank<4> auto const& f,
                   double mad = 0,
                   double e0 = 0):
        mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_), filename(outdir_ + "/" + prefix_ + ".h5"),
        _symm(symm),
        nelec(_nel),
        madelung(mad),
        enuc(e0),
        efermi(0.0),
        k_weight(_symm.nkpts_ibz,double(_symm.nkpts)/_symm.nkpts_ibz),
        ndims(3),
        nspin(nspin_),
        nspin_in_basis(nspin),
        npol(npol_),
        npol_in_basis(std::max(1,npol)),
        nbnd(h.extent(2)/npol_in_basis),
        H0(h),
        S(ov),
        DM(d),
        FockM(f)
      {
        utils::check(nspin_in_basis == nspin, "nspin_in_basis != nspin not yet allowed",nspin_in_basis,nspin);
        utils::check(npol_in_basis == npol, "npol_in_basis != npol not yet allowed",npol_in_basis,npol);
        int nk = _symm.nkpts_ibz;
        utils::check(nelec > 0, "nelec=0 in model_system constructor.");
        utils::check( nspin == 1 or nspin == 2, "Invalid nspin:{}",nspin);
        utils::check(npol > 0, "npol=0 in model_system constructor.");
        utils::check(nspin_in_basis <= nspin, "nspin_in_basis ({}) > nspin ({}) in model_system constructor.",nspin_in_basis,nspin);
        utils::check(npol_in_basis <= npol, "npol_in_basis ({}) > npol ({}) in model_system constructor.",npol_in_basis,npol);
        auto shp = std::array<long,4>{nspin_in_basis,nk,npol_in_basis*nbnd,npol_in_basis*nbnd};
        utils::check(H0.shape() == shp, "Dimension mismatch.");
        utils::check(S.shape() == shp, "Dimension mismatch.");
        utils::check(DM.shape() == shp, "Dimension mismatch.");
        utils::check(FockM.shape() == shp, "Dimension mismatch.");
      } 

      model_system(const model_system& ) = default;
      model_system(model_system&& ) = default;

      model_system& operator=(const model_system& ) = default;
      model_system& operator=(model_system&& ) = default;

      ~model_system() = default;

      auto const& bz() const { return _symm; }

      void save(std::string fn) const {
        h5::file file = h5::file(fn, 'w');
        h5::group grp(file);
        save(grp);
      }

      // MPI handler for communicators
      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

      // model output directory
      std::string outdir = "";
      // model prefix
      std::string prefix = "";
      // name of file with input data
      std::string filename = "";
      // type of input file, hardwired to h5
      mf_input_file_type_e input_file_type = h5_input_type;    

      // kpt and Qpt information
      bz_symm _symm;

      // # of electrons
      double nelec = 0.0;
      // madelung constant for finite-size corrections
      double madelung = 0.0;
      // nuclear energy
      double enuc = 0.0;
      // fermi energy
      double efermi = 0.0;

      // weight of kpoint in averages
      nda::array<double, 1> k_weight;

      // # of dimensions
      int ndims = 3; // hardwired to 3d for now
      // # of spin
      int nspin = 0;
      // # of spin in basis set
      int nspin_in_basis = 0;
      // # of spin
      int npol = 0;
      // # of spin in basis set
      int npol_in_basis = 0;

      // basis info
      // # of primary single-particle basis
      int nbnd;


      // 1-body Hamiltonian
      nda::array<ComplexType, 4> H0;
      // overlap
      nda::array<ComplexType, 4> S;
      // dm
      nda::array<ComplexType, 4> DM;
      // fock matrix 
      nda::array<ComplexType, 4> FockM;

      /************************************************************************************/
      // Variables below exist for consistenty with other backends, but have no real meaning 
      // in a model hamiltonian
      // unit cell info
      int natoms = 0;
      // # of unique species
      int nspecies = 0;
      // name/ids of species
      std::vector<std::string> species;
      // array with the id(Z!?) of atoms in unit cell
      nda::array<int, 1> at_ids = nda::array<int, 1>(0);
      // array of atom positions
      nda::array<double, 2> at_pos = nda::array<double, 2>(0,0);
      // fft grid
      nda::stack_array<int, 3> fft_mesh = {0,0,0};
      // lattice vectors: a_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> latt = {{0,0,0},{0,0,0},{0,0,0}};
      // reciprocal vectors: b_n[i] = recv(n,i), for the ith (x,y,z) cartesian component of the nth (1,2,3) vector.
      nda::stack_array<double, 3, 3> recv = {{0,0,0},{0,0,0},{0,0,0}};
      // plane-wave cutoff for charge density
      double ecutrho = 0.0;
      // number of points in fft grid
      int nnr = 0;
      // # of auxiliary single-particle basis
      int nbnd_aux = 0;
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> eigval_aux = nda::array<double, 3>(0,0,0);
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> aux_weight = nda::array<double, 3>(0,0,0);
      // orbitals are stored on fft grid or not
      bool orb_on_fft_grid = false;
      // eigenvalues: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> eigval;
      // occupation numbers: [nspin_in_basis, kpoint, npol_in_basis*band]
      nda::array<double, 3> occ;
      bool spinorbit = false;
      /************************************************************************************/

      public:

      void save(h5::group& grp) const
      {
        h5::group sgrp = (grp.has_subgroup("System") ?
                          grp.open_group("System")    :
                          grp.create_group("System", true));

        //BZ
        _symm.save(sgrp);

        // unit cell
        h5::h5_write_attribute(sgrp, "number_of_spins", nspin);
        h5::h5_write_attribute(sgrp, "number_of_polarizations", npol);
        h5::h5_write_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_write_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        h5::h5_write_attribute(sgrp, "number_of_bands", nbnd);
        h5::h5_write_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_write_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_write_attribute(sgrp, "nuclear_energy", enuc);
        h5::h5_write_attribute(sgrp, "fermi_energy", efermi);

        nda::h5_write(sgrp, "kpoint_weights", k_weight, false);

        nda::h5_write(sgrp, "H0", H0, false);
        nda::h5_write(sgrp, "overlap", S, false);
        nda::h5_write(sgrp, "density_matrix", DM, false);
        nda::h5_write(sgrp, "fock_matrix", FockM, false);
      }

      private:
      void initialize_from_h5(int _nbnd = -1)
      {
        using nda::range;
        decltype(nda::range::all) all;
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);

        h5::group sgrp = grp.open_group("System"); 

        // BZ
        _symm.initialize_from_h5(sgrp);  

        if(sgrp.has_key("number_of_dimensions")) 
          h5::h5_read_attribute(sgrp, "number_of_dimensions", ndims);
        h5::h5_read_attribute(sgrp, "number_of_spins", nspin);
        utils::check( nspin == 1 or nspin == 2, "Invalid nspin:{}",nspin);
        if(sgrp.has_key("number_of_spins_in_basis")) 
          h5::h5_read_attribute(sgrp, "number_of_spins_in_basis", nspin_in_basis);
        else
          nspin_in_basis = nspin;
        utils::check( nspin_in_basis <= nspin and nspin_in_basis > 0, "Error: nspin_in_basis: {}",nspin_in_basis);
        h5::h5_read_attribute(sgrp, "number_of_polarizations", npol);
        if(sgrp.has_key("number_of_polarizations_in_basis"))
          h5::h5_read_attribute(sgrp, "number_of_polarizations_in_basis", npol_in_basis);
        else
          npol_in_basis = npol;
        utils::check( npol > 0, "Error: npol: {}",npol);
        utils::check( npol_in_basis <= npol and npol_in_basis > 0, "Error: npol_in_basis: {}",npol_in_basis);
        h5::h5_read_attribute(sgrp, "number_of_elec", nelec);
        h5::h5_read_attribute(sgrp, "madelung_constant", madelung);
        h5::h5_read_attribute(sgrp, "nuclear_energy", enuc);
        if( H5Aexists(h5::hid_t(sgrp),"fermi_energy") )
          h5::h5_read_attribute(sgrp, "fermi_energy", efermi);

        // temporary!
        utils::check(nspin_in_basis == nspin, "nspin_in_basis != nspin not yet allowed",nspin_in_basis,nspin);
        utils::check(npol_in_basis == npol, "npol_in_basis != npol not yet allowed",npol_in_basis,npol);

        //k_weight = nda::array<double, 1>(_symm.nkpts);
        nda::h5_read(grp, "System/kpoint_weights", k_weight);
        if( std::abs( nda::sum(k_weight) - 1.0 ) > 1e-6 and mpi->comm.root() )
          app_warning(" k_weight is not normalized. sum(k_weight): {}",nda::sum(k_weight)); 

        h5::h5_read_attribute(sgrp, "number_of_bands", nbnd);
        utils::check(_nbnd!=0, "Error in model_system: nbnd==0.");
        if(_nbnd > 0) {
          utils::check(_nbnd <= nbnd, 
                       "Error in initialize_from_h5: nbnd:{} nbnd in h5:{}",_nbnd,nbnd);
          nbnd = _nbnd;
        }
          
        int nkpts = _symm.nkpts;
        int nkpts_ibz = _symm.nkpts_ibz;
        utils::check(nkpts == nkpts_ibz, 
          "Error in model_hamiltonian::initialize_from_h5: K-point symmetry not yet implemented.");

        H0    = nda::array<ComplexType, 4>(nspin_in_basis, nkpts, npol_in_basis*nbnd, npol_in_basis*nbnd);
        H0()  = ComplexType(0.0); 
        S     = H0;
        DM    = H0;
        FockM = H0; 
        {
          nda::array<ComplexType, 4> e_;
          auto read_ = [&] (std::string name) {
              nda::h5_read(sgrp, name, e_);
              utils::check( e_.extent(0) == nspin_in_basis and
                            e_.extent(1) >= nkpts_ibz and 
                            e_.extent(2) >= npol_in_basis*nbnd and 
                            e_.extent(3) >= npol_in_basis*nbnd,
                            "Error with " + name +": Incorrect dimensions in h5.");
              return  e_(range(nspin_in_basis),range(nkpts_ibz),range(npol_in_basis*nbnd),range(npol_in_basis*nbnd)); 
            };
          H0(all,range(nkpts_ibz),all,all) = read_("H0"); 
          S(all,range(nkpts_ibz),all,all) = read_("overlap"); 
          DM(all,range(nkpts_ibz),all,all) = read_("density_matrix"); 
          FockM(all,range(nkpts_ibz),all,all) = read_("fock_matrix"); 
          for(int ik=nkpts_ibz; ik<nkpts; ik++) {
            int ik_ibz = _symm.kp_to_ibz(ik);
            H0(all,ik,all,all) = H0(all,ik_ibz,all,all);
            S(all,ik,all,all) = S(all,ik_ibz,all,all);
            DM(all,ik,all,all) = DM(all,ik_ibz,all,all);
            FockM(all,ik,all,all) = FockM(all,ik_ibz,all,all);
          }
        }

      }

    }; // model_system

  } // model
} // mf

#endif //COQUI_MODEL_SYSTEM_HPP
