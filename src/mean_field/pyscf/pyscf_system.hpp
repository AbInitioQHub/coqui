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


#ifndef COQUI_PYSCF_SYSTEM_HPP
#define COQUI_PYSCF_SYSTEM_HPP

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
  namespace pyscf {
    /**
     * Struct that contains details of a PySCF mean-field calculation, read from hdf5 files
     */
    struct pyscf_system {
    public:
      pyscf_system(std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi_,
                   std::string outdir_, std::string prefix_):
      mpi(std::move(mpi_)), outdir(outdir_), prefix(prefix_),
      filename(outdir_ + "/" + prefix_ + ".h5") {
        utils::check(std::filesystem::exists(filename),
		     " Error in pyscf_system: filename: {} does not exist.",filename);
        initialize_from_h5(mpi->comm);
      }

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
        _symm.save(sgrp);

        // unit cell
        h5::h5_write(sgrp, "number_of_atoms", natoms);
        h5::h5_write(sgrp, "number_of_species", nspecies);
        h5::h5_write(sgrp, "number_of_dimensions", ndims);
        h5::h5_write(sgrp, "number_of_spins", nspin);
        h5::h5_write(sgrp, "species", species);
        h5::h5_write(sgrp, "number_of_elec", nelec);
        nda::h5_write(sgrp, "atomic_id", at_ids, false);
        nda::h5_write(sgrp, "atomic_positions", at_pos, false);
        nda::h5_write(sgrp, "lattice_vectors", latt, false);
        nda::h5_write(sgrp, "reciprocal_vectors", recv, false);
        h5::h5_write(sgrp, "madelung_constant", madelung);
        h5::h5_write(sgrp, "nuclear_energy", enuc);
        h5::h5_write(sgrp, "fermi_energy", efermi);
        h5::h5_write(sgrp, "number_of_bands", nbnd);

        nda::h5_write(sgrp, "kpoint_weights", k_weight, false);

        int nkpts = _symm.nkpts;
        int nkpts_ibz = _symm.nkpts_ibz;
        h5::group ogrp = (grp.has_subgroup("Orbitals") ?
                          grp.open_group("Oritals")    :
                          grp.create_group("Orbitals", true));
        h5::h5_write(ogrp, "number_of_spins", nspin);
        h5::h5_write(ogrp, "number_of_kpoints", nkpts);
        h5::h5_write(ogrp, "number_of_kpoints_ibz", nkpts_ibz);
        h5::h5_write(ogrp, "number_of_bands", nbnd);
        h5::h5_write(ogrp, "ecutrho", ecutrho);
        nda::h5_write(ogrp, "fft_mesh", fft_mesh, false);

        nda::h5_write(ogrp, "eigval", eigval, false);
        nda::h5_write(ogrp, "occ", occ, false);
      }

    private:
      void initialize_from_h5(utils::Communicator auto& comm) 
      {
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);

        // unit cell
        h5::h5_read(grp, "natoms", natoms);
        h5::h5_read(grp, "nspecies", nspecies);
        species = std::vector<std::string>(nspecies);
        at_ids  = nda::array<int, 1>(natoms);
        at_pos  = nda::array<double, 2>(natoms, 3);
        //h5::h5_read(grp, "species", species); // CN: reading species is not ready yet. Will fix it easily later...
        h5::h5_read(grp, "nelec", nelec);
        nda::h5_read(grp, "at_ids", at_ids);
        nda::h5_read(grp, "at_pos", at_pos);
        nda::h5_read(grp, "latt", latt);
        nda::h5_read(grp, "recv", recv);
        h5::h5_read(grp, "madelung", madelung);
        h5::h5_read(grp, "enuc", enuc);
        if( H5Aexists(h5::hid_t(grp),"fermi_energy") )
          h5::h5_read(grp, "fermi_energy", efermi);

        // BZ info
        int nkpts;
        h5::h5_read(grp, "nkpts", nkpts);
        nda::array<int, 1> kp_grid_(3);
        nda::array<double, 2> kpts_(nkpts,3);
        k_weight = nda::array<double, 1>(nkpts);
        nda::h5_read(grp, "kpts", kpts_);
        nda::h5_read(grp, "k_weight", k_weight);
        if(grp.has_dataset("kp_grid"))
          nda::h5_read(grp, "kp_grid", kp_grid_);

        std::vector<utils::symm_op> symm_list_;
        { // add identity operation
          symm_list_.clear();
          nda::stack_array<double, 3, 3> R;
          nda::stack_array<double, 3> kp;
          R() = 0.0;
          R(0, 0) = 1.0;
          R(1, 1) = 1.0;
          R(2, 2) = 1.0;
          kp() = 0.0;
          symm_list_.emplace_back(utils::symm_op{R, R, kp});
        }
        _symm = bz_symm(comm,true,latt,recv,kp_grid_,kpts_,symm_list_,false);

        // mean-field calculation
        h5::h5_read(grp, "nbnd", nbnd);
        if(grp.has_dataset("number_of_dimensions"))
          h5::h5_read(grp, "number_of_dimensions", ndims);
        h5::h5_read(grp, "nspin", nspin);
        if(grp.has_dataset("nspin_in_basis"))
          h5::h5_read(grp, "nspin_in_basis", nspin_in_basis);
        else
          nspin_in_basis = nspin;
        utils::check( nspin_in_basis <= nspin, "Error: nspin_in_basis > nspin");
        npol = npol_in_basis = 1;
        if(grp.has_dataset("npol"))
          h5::h5_read(grp, "npol", npol);
        if(grp.has_dataset("npol_in_basis"))
          h5::h5_read(grp, "npol_in_basis", npol_in_basis);

        if (grp.has_subgroup("FFT")) {
          // fft info
          orb_on_fft_grid = true;
          h5::group fft_grp = grp.open_group("FFT");
          h5::h5_read(fft_grp, "ecut", ecutrho);
          nda::h5_read(fft_grp, "fft_mesh", fft_mesh);
          nnr = fft_mesh(0)*fft_mesh(1)*fft_mesh(2);
        } else {
          // Becke grid info
          orb_on_fft_grid = false;
          h5::group becke_grp = grp.open_group("BECKE");
          h5::h5_read(becke_grp, "number_of_rpoints", nnr);
          fft_mesh() = 1;
          fft_mesh(0) = nnr;

          // assign dummy values
          ecutrho = 1.0;
        }

        // TODO Add check if the pyscf input is in AO basis
        h5::group scf_grp = grp.open_group("SCF");
        eigval = nda::array<double, 3>(nspin, nkpts, nbnd);
        eigval_aux = nda::array<double, 3>(nspin, nkpts, nbnd_aux);
        aux_weight = nda::array<double, 3>(nspin, nkpts, nbnd_aux);
        occ    = nda::array<double, 3>(nspin, nkpts, nbnd);
        nda::h5_read(scf_grp, "eigval", eigval);
        nda::h5_read(scf_grp, "occ", occ);

      }

    public:
      pyscf_system(const pyscf_system& ) = default;
      pyscf_system(pyscf_system&& ) = default;

      pyscf_system& operator=(const pyscf_system& ) = default;
      pyscf_system& operator=(pyscf_system&& ) = default;

      ~pyscf_system() = default;

      bz_symm const& bz() const { return _symm; }

      // MPI handler for communicators
      std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> mpi;

      // name of file with input data
      std::string outdir = "";
      std::string prefix = "";
      std::string filename = "";
      mf_input_file_type_e input_file_type = h5_input_type;

      // unit cell info
      int natoms = 0;
      // # of unique species
      int nspecies = 0;
      // # of electrons
      double nelec = 0;
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

      // BZ info
      bz_symm _symm;
      // weight of kpoint in averages
      nda::array<double, 1> k_weight;

      // basis info
      // # of primary single-particle basis
      int nbnd = 0;
      // # of auxiliary single-particle basis
      int nbnd_aux = 0;
      // whether orbitals are stored on a fft grid
      bool orb_on_fft_grid = false;
      // plane-wave cutoff for AOs
      double ecutrho = 0;
      // fft grid
      nda::stack_array<int, 3> fft_mesh;
      // number of points in fft grid
      int nnr = 0;

      // # of dimensions
      int ndims = 3;
      // # of spin
      int nspin = 0;
      // # of spin in basis set (1: AO, 2:MO/UHF)
      int nspin_in_basis = 0;
      // # of polarizations (1:RHF/UHF, 2:GHF) 
      int npol = 1;
      // # of polarizations in basis set (1:AO, 2:MO/GHF)
      int npol_in_basis = 1;
      // eigenvalues: [nspin, kpoint, band]
      nda::array<double, 3> eigval;
      // auxiliary eigenvalues: [spin, kpoint, band]
      nda::array<double, 3> eigval_aux;
      // weights for auxliary basis 
      nda::array<double, 3> aux_weight;
      // occupation numbers: [nspin, kpoint, band]
      nda::array<double, 3> occ;

    public:
      // dummy members, just to be consistent with qe_readonly class...
      // may use them in the future
      bool spinorbit = false;
      double alat = 0.0;
      int ngm = -1;
      int ngms = -1;
    };

  } // pyscf
} // mf

#endif //COQUI_PYSCF_SYSTEM_HPP
