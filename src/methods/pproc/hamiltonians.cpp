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



#include "configuration.hpp"
#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"
#include "utilities/check.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"

#include "numerics/nda_functions.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "utilities/distributed_cholesky.hpp"

#include "methods/pproc/hamiltonians.h"
#include "methods/pproc/wavefunction_utils.h"
#include "methods/ERI/thc.h"
#include "methods/ERI/thc_reader_t.hpp"

namespace methods
{

/*
 * Writes second-quantized hamiltonians to file. 
 * 1. Adds non-interacting hamiltonian associated to the supplied MF object.
 * 2. If requested, adds a trial wave-function from mf.
 * Optional arguments:
 *  - "output" : "hamil.h5", name of output h5 file. 
 *  - "format" : "qmc", output format. 
 *  - "add_wavefunction" : "". Request a wavefunction block. 
 *       Choices:
 *         - "" : don't write wavefunction. 
 *         - "default" : default wave function, e.g. lowest 'nelec' orbitals from mf object.
 *         - "diag_F" : diagonalize fock matrix and take lowest 'nelec' orbitals.
 *         - "gw" : lowest 'nelec' orbitals from natural orbitals of converged SCF GW. 
 *  - "type" : "bare", type of second quantized hamiltonian. Choices: "bare" (add "gw"). 
 */
template<MEMORY_SPACE MEM>
void add_core_hamiltonian(mf::MF &mf, ptree const& pt)
{
  using Array_view_4D_t = memory::array_view<MEM, ComplexType, 4>;
  auto output = io::get_value_with_default<std::string>(pt,"output","hamil.h5");
  auto format = io::get_value_with_default<std::string>(pt,"format","qmc");
  auto type = io::get_value_with_default<std::string>(pt,"type","bare");
  auto add_wfn = io::get_value_with_default<std::string>(pt,"add_wavefunction","");

  auto mpi = mf.mpi();

  long nspins = mf.nspin();
  long nkpts_ibz = mf.nkpts_ibz();
  long nbnd = mf.nbnd();

  app_log(2, "*******************************");
  app_log(2, " Second-quantized 1-Body Hamiltonian  ");
  app_log(2, "*******************************");
  app_log(2, "output: {}",output);
  app_log(2, "format: {}",format);
  app_log(2, "type: {}",type);
  app_log(2, "add_wfn: {}",add_wfn);

  utils::check(format == "qmc", "Error in add_core_hamiltonian: Invalid format:{}", format); 

  if(type == "bare") {

    // write MF::System information 
   
    auto sH0_skij = math::shm::make_shared_array<Array_view_4D_t>(
          *mpi, {nspins, nkpts_ibz, nbnd, nbnd});
    auto psp = hamilt::make_pseudopot(mf);
    hamilt::set_H0(mf, psp.get(), sH0_skij);
    mpi->comm.barrier();
    if(mpi->comm.root()) {

      h5::file file(output, 'a');
      h5::group grp(file);

      if(!grp.has_subgroup("System")) mf.save_system(grp);

      h5::group sgrp = grp.open_group("System");
      auto H0 = sH0_skij.local();
      nda::h5_write(sgrp, "H0", H0); 

      if(add_wfn != "") add_wavefunction(grp,mf,pt);

    } // mpi->root()

  } else {    
    APP_ABORT("Error in write_core_hamiltonian: Invalid type:{}",type);
  }
  app_log(2, "\n");
 
}

template<MEMORY_SPACE MEM>
void add_thc_hamiltonian_components(mf::MF &mf,
                         thc_reader_t& thc, ptree const& pt)
{

  decltype(nda::range::all) all;
  auto output = io::get_value_with_default<std::string>(pt,"output","hamil.h5");
  auto format = io::get_value_with_default<std::string>(pt,"format","qmc");
  auto type = io::get_value_with_default<std::string>(pt,"type","bare");
  auto thresh = io::get_value_with_default<double>(pt,"thresh",1e-8);
  auto bsize = io::get_value_with_default<int>(pt,"chol_block_size",32);

  auto mpi = mf.mpi();

  bool add_half_transformed = false;
  ptree pt_coul; 
  for (auto const& child : pt) {
    if (child.first == "half_transformed_integrals") {
      add_half_transformed = true;
      pt_coul = child.second;
      break;
    }
  }

  long nspins = mf.nspin();
  long nkpts_ibz = mf.nkpts_ibz();
  long nkpts = mf.nkpts();
  long nqpts = mf.nqpts();
  long nqpts_ibz = mf.nqpts_ibz();
  long nbnd = mf.nbnd();
  auto Np = thc.Np();
  auto occ = mf.occ();

  app_log(2, "**************************************");
  app_log(2, " Adding THC components to Hamiltonian "); 
  app_log(2, "**************************************");
  app_log(2, "output: {}",output);
  app_log(2, "format: {}",format);
  app_log(2, "type: {}",type);

  utils::check(format == "qmc", "Error in add_thc_hamiltonian_components: Invalid format:{}", format); 

  long nx = utils::find_proc_grid_min_diff(mpi->comm.size(),1,1), ny = mpi->comm.size()/nx;
  auto dZ = thc.dZ(std::array<long,3>{1,nx,ny});
  long nI = dZ.global_shape()[1];

  using lArray_t = std::conditional_t<MEM==HOST_MEMORY, memory::host_array<ComplexType,2>,
                                      memory::unified_array<ComplexType,2>>;
  if(type == "bare") {

    if(mpi->comm.root()) {

      h5::file file(output, 'a');
      h5::group grp(file);
      
      h5::group igrp = (grp.has_subgroup("Interaction") ?
                        grp.open_group("Interaction")    :
                        grp.create_group("Interaction", true));

      h5::h5_write_attribute(igrp, "number_of_spins", nspins);
      h5::h5_write_attribute(igrp, "number_of_kpoints", nkpts);
      h5::h5_write_attribute(igrp, "number_of_kpoints_ibz", nkpts_ibz);
      h5::h5_write_attribute(igrp, "number_of_qpoints", nqpts);
      h5::h5_write_attribute(igrp, "number_of_qpoints_ibz", nqpts_ibz);
      h5::h5_write_attribute(igrp, "number_of_bands", nbnd);

      if(not add_half_transformed) math::nda::h5_write(igrp, "coulomb_matrix", dZ);
      // AFQMC ordering 
      nda::array<ComplexType,4> X_skiu(nspins,nkpts,nbnd,Np);
      for( auto is : nda::range(nspins) )
        for( auto ik : nda::range(nkpts) )
          X_skiu(is,ik,all,all) =  nda::transpose(thc.X(is,0,ik));
      nda::h5_write(igrp, "collocation_matrix", X_skiu, false);

    } else {

      h5::group grp;
      if(not add_half_transformed) math::nda::h5_write(grp, "coulomb_matrix", dZ);
      
    }
    mpi->comm.barrier();

    for( auto [iq,q] : itertools::enumerate(dZ.local_range(0)) ) {

      // this should be a stand alone function, for slicing distributed arrays 
      auto Z2D = dZ.local()(iq,all,all);
      memory::darray_view_t<lArray_t,mpi3::communicator> Ziq(std::addressof(mpi->comm),
                {nx,ny}, {nI,nI}, {dZ.origin()[1], dZ.origin()[2]}, 
                {dZ.block_size()[1],dZ.block_size()[2]}, Z2D);

      // returns Ziq = L * dagger(L)
      auto L = utils::distributed_cholesky(Ziq,thresh,bsize); 
      math::nda::redistribute(L,Ziq);

    }

    mpi->comm.barrier();
    if(mpi->comm.root()) {
      h5::file file(output, 'a');
      h5::group grp(file);
      h5::group igrp = grp.open_group("Interaction");
      math::nda::h5_write(igrp, "factorized_coulomb_matrix", dZ);
    } else {
      h5::group grp;
      math::nda::h5_write(grp, "factorized_coulomb_matrix", dZ);
    }

    if(add_half_transformed) {
      // should have a toml subblock with half_transformed input
      // right now assuming range, can also provide slater matrix/rotation
      if(type == "range") {
        // find standard range:
        long nocc_mf = 0;
        for(int is=0; is<nspins; is++) {
          for(int ik=0; ik<nkpts_ibz; ik++) {
            long n = long(std::round(std::accumulate(occ(is,ik,all).begin(),
                                              occ(is,ik,all).end(),double(0.0))));
            nocc_mf = std::max(n,nocc_mf);
          }
        }
        auto nIpts = io::get_value_with_default<int>(pt_coul,"nIpts",-1);
        auto nocc = io::get_value_with_default<int>(pt_coul,"nocc",int(nocc_mf));
        nda::range a_rng(nocc); 
        nda::range b_rng(nbnd); 
        // only correct right now if Vuv is real, otherwise need symmetric matrix element 
        methods::thc eri(std::addressof(mf),*mpi,pt_coul);
        auto [ri,Xa,Xb] = eri.interpolating_points<HOST_MEMORY>(0,nIpts,a_rng,b_rng);
        auto Muv = eri.evaluate<HOST_MEMORY>(ri,Xa,Xb,false,a_rng,b_rng); 

        if(mpi->comm.root()) {
          h5::file file(output, 'a');
          h5::group grp(file);
          h5::group igrp = grp.open_group("Interaction");
          math::nda::h5_write(igrp, "half_rotated_coulomb_matrix", std::get<0>(Muv));
          // already in AFQMC ordering, since not getting through thc_reader_t 
          math::nda::h5_write(igrp, "collocation_matrix_half_rotated", *Xb);
        } else {
          h5::group grp;
          math::nda::h5_write(grp, "half_rotated_coulomb_matrix", std::get<0>(Muv));
          math::nda::h5_write(grp, "collocation_matrix_half_rotated", *Xb);
        }
      } else {
        APP_ABORT(" Error in add_thc_hamiltonian_components: Invalid type in half_transformed_integrals - type:{}",type);
      }
    }

  } else {    
    APP_ABORT("Error in add_thc_hamiltonian_components: Invalid type:{}",type);
  }
  app_log(2, "\n");
  
}

template<MEMORY_SPACE MEM>
void add_cholesky_hamiltonian_components(mf::MF &mf,
                         chol_reader_t& chol, ptree const& pt)
{
  (void) mf; (void) chol; (void) pt;
}

using boost::mpi3::communicator;

template void 
add_core_hamiltonian<HOST_MEMORY>(mf::MF&,ptree const&);

template void 
add_thc_hamiltonian_components<HOST_MEMORY>(mf::MF&,thc_reader_t&,ptree const&);

template void 
add_cholesky_hamiltonian_components<HOST_MEMORY>(mf::MF&,chol_reader_t&,ptree const&);

}
