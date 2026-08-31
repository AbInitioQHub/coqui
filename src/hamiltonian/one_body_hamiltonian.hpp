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


#ifndef HAMILTONIAN_ONE_BODY_HAMILTONIAN_HPP
#define HAMILTONIAN_ONE_BODY_HAMILTONIAN_HPP

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"

#include "utilities/proc_grid_partition.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "mean_field/MF.hpp"

#include "methods/tools/chkpt_utils.h"

#include "hamiltonian/matrix_elements.h"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/gen_one_body_hamiltonian.icc"
#include "hamiltonian/pyscf_one_body_hamiltonian.icc"

namespace hamilt
{


/**
 * Non-interacting one-body hamiltonian associated with MF object in a distributed array
 * Only includes kinetic and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */ 
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H0(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,  
        nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1}, 
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(-1,-1), k_range, b_range, bz); 
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_range,b_range,psi,p3,p4);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismacth");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * Non-interacting Hamiltonian associated with a MF object in a shared memory array
 * Includes kinetic, pseudo-potential/external potential contributions
 * @return - A shared memory array of non-interacting Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_H0(mf::MF &mf, pseudopot *psp, math::shm::shared_array<Array_4D_t> &sH0_skij) {
  long np = sH0_skij.internode_comm()->size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 2048);

  if (sH0_skij.node_comm()->root()) {
    auto dH0  = hamilt::H0<HOST_MEMORY>(mf, *sH0_skij.internode_comm(), psp, 
                                        nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                        pgrid, bsize);
    auto H0_loc = sH0_skij.local();
    H0_loc(dH0.local_range(0), dH0.local_range(1), dH0.local_range(2), dH0.local_range(3)) = dH0.local();
  }
  sH0_skij.communicator()->barrier();
  sH0_skij.all_reduce();
}

/**
 * One-body hamiltonian associated with MF object in a distributed array
 * Includes kinetic, hartree and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param rhoij [input] - density matrix 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        nda::ArrayOfRank<4> auto const& nij,
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{ 
  using nij_type = decltype(nij);
  static_assert(memory::get_memory_space<nij_type>() == MEM, "Memory Space mismatch.");
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    nda::range b_rng(mf.nbnd());
    nda::range k_rng(mf.nkpts_ibz());
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(mf.nspin()),k_rng,b_rng,bz);
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_rng,b_rng,psi,p3,std::addressof(nij));
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * One-body hamiltonian associated with MF object in a distributed array
 * Includes kinetic, hartree and pseudo-potential/external potential contributions.
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator 
 * @param psp   [input] - pseudopotential object 
 * @param rhoij [input] - density matrix 
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto H(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        nda::ArrayOfRank<3> auto const& nii,
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in H0: Missing pseudopot object.");
    nda::range b_rng(mf.nbnd());
    nda::range k_rng(mf.nkpts_ibz());
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(mf.nspin()),k_rng,b_rng,bz);
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_H0<MEM>(mf,comm,psp,k_rng,b_rng,psi,std::addressof(nii),p4);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

/**
 * Fock matrix associated with a MF object. 
 * Includes Kinetic, pseudo-potential/external potential and HF/Vxc potential
 * @return - A distributed array of one-body Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto F(mf::MF& mf, boost::mpi3::communicator& comm, 
       nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1}, 
       std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048}, 
       bool evaluate = false)
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    auto eig = mf.eigval();
    if(evaluate or std::all_of( eig.begin(),  eig.end(), [] (auto&& v){ return v==0; })) {
      APP_ABORT("Error in hamilt::F(evaluate=true), disabled evaluation of Fock matrix without eigenvalues.");
    }
    {
      long nspin = mf.nspin();
      long nbnd = mf.nbnd();
      long nkpts = k_range.size();
      long M = b_range.size();
      utils::check(b_range.first() >= 0 and b_range.last() <= nbnd, "Band range out of bounds.");
      utils::check(k_range.first() >= 0 and k_range.last() <= mf.nkpts(), "K-point range out of bounds.");
      long np0 = std::accumulate(pgrid.cbegin(), pgrid.cend(), long(1), std::multiplies<>{});
      if( np0 == 0 ) {
        long sz = comm.size();
        long ps = (sz%nspin==0?nspin:1);
        long n_ = sz/ps;
        long pk = utils::find_proc_grid_max_rows(n_,nkpts);
        pgrid = {ps,pk,n_/pk,1};
      } 
      using larray = memory::array<MEM,ComplexType,4>;
      auto Fij = math::nda::make_distributed_array<larray>(comm,pgrid,{nspin,nkpts,M,M},
                  {bz[0],bz[1],bz[2],bz[2]});
      auto Floc = Fij.local();
      Floc = ComplexType(0.0);
      for( auto [is, s] : itertools::enumerate(Fij.local_range(0)))
        for( auto [ik, k] : itertools::enumerate(Fij.local_range(1)))
          for( auto [ia, a] : itertools::enumerate(Fij.local_range(2)))
            for( auto [ib, b] : itertools::enumerate(Fij.local_range(3)))
              if(a==b) Floc(is,ik,ia,ib) = eig(s,k+k_range.first(),a+b_range.first());
      comm.barrier();
      return Fij;
    }
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"Fock",comm,pgrid,bz);
  }
}

/**
 * One-body Hamiltonian associated with a MF object in a shared memory array
 * Includes Kinetic, pseudo-potential/external potential and HF/Vxc potential
 * @param exclude_H0 [INPUT] - exclude kinetic + pseudo/external potential or not
 * @return - A shared memory array of one-body Hamiltonian with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_fock(mf::MF &mf, pseudopot *psp, math::shm::shared_array<Array_4D_t> &sF_skij, bool exclude_H0=false) {
  long np = sF_skij.internode_comm()->size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 2048);

  if (sF_skij.node_comm()->root()) {
    auto dF  = hamilt::F<HOST_MEMORY>(mf, *sF_skij.internode_comm(), nda::range(mf.nkpts_ibz()),
                                      nda::range(mf.nbnd()), pgrid, bsize);
    if (exclude_H0) { 
      auto dH0 = hamilt::H0<HOST_MEMORY>(mf, *sF_skij.internode_comm(), psp, nda::range(mf.nkpts_ibz()),
                                         nda::range(mf.nbnd()), pgrid, bsize);
      dF.local() -= dH0.local();
    }
    auto F_loc = sF_skij.local();
    F_loc(dF.local_range(0), dF.local_range(1), dF.local_range(2), dF.local_range(3)) = dF.local();
  }
  sF_skij.communicator()->barrier();
  sF_skij.all_reduce();
}

/**
 * Compute the matrix elements of the Hartree potential in a distributed array
 * using PWs and FFT
 *
 * @param mf      [input] - mean-field object
 * @param comm    [input] - communicator
 * @param psp     [input] - pseudopotential object
 * @param nij     [input] - density matrix (nspin, nkpts, nbnd, nbnd)
 * @param k_range [input] - index range for k-points
 * @param b_range [input] - indenx range for orbitals
 * @param pgrid   [input] - processor grid for the distributed array
 * @param bz      [input] - block size for the distributed array
 * @return - A distributed array of the Hartree Hamiltonian
 *           with global shape = (nspin, k_range.size(), b_range.size(), b_range.size())
 */
template<MEMORY_SPACE MEM = HOST_MEMORY, nda::ArrayOfRank<4> Arr4_t>
auto Vhartree(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
        Arr4_t const& nij,
        nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
        std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in Vhartree: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    memory::array_view<MEM,ComplexType,3> *p3=nullptr;
    return detail::gen_Vhartree<MEM>(mf,comm,psp,k_range,b_range,psi,p3,std::addressof(nij),false);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(mf.orb_on_fft_grid(),
                 "Vhartree: The Hartree potential cannot be evaluated using FFT if mf.orb_on_fft_grid() == false");
    utils::check(false, "Vhartree: Hartree potential using FFT is not implemented for non-orthogonal basis yet!");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

template<MEMORY_SPACE MEM = HOST_MEMORY, nda::ArrayOfRank<3> Arr3_t>
auto Vhartree(mf::MF &mf, boost::mpi3::communicator &comm, pseudopot *psp,
              Arr3_t const& nii,
              nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
              std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    utils::check(psp != nullptr, "Error in Vhartree: Missing pseudopot object.");
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    memory::array_view<MEM,ComplexType,4> *p4=nullptr;
    return detail::gen_Vhartree<MEM>(mf,comm,psp,k_range,b_range,psi,std::addressof(nii),p4,false);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(mf.orb_on_fft_grid(),
                 "Vhartree: The Hartree potential cannot be evaluated using FFT if mf.orb_on_fft_grid() == false");
    utils::check(false, "Vhartree: Hartree potential using FFT is not implemented for non-orthogonal basis yet!");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"H0",comm,pgrid,bz);
  }
}

template<typename MPI_t>
void dump_hartree(MPI_t &mpi, mf::MF &mf, pseudopot *psp, std::string coqui_output, int scf_iter) {
  long np = mpi.comm.size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 2048);

  // logic of iteration
  if (scf_iter == -1) {
    std::string filename = coqui_output + ".mbpt.h5";
    h5::file file;
    try {
      file = h5::file(filename, 'r');
    } catch(...) {
      APP_ABORT("Failed to open h5 file: {}, mode:r",filename);
    }
    auto scf_grp = h5::group(file).open_group("scf");
    h5::h5_read(scf_grp, "final_iter", scf_iter);
  }
  int dm_iter = (scf_iter == 0)? scf_iter : scf_iter-1;

  using larray_view = memory::array_view<HOST_MEMORY,ComplexType,4>;
  using math::shm::make_shared_array;
  auto sDm_skij = make_shared_array<larray_view>(mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  methods::chkpt::read_dm(mpi.node_comm, coqui_output, dm_iter, sDm_skij);

  auto dVH = hamilt::Vhartree<HOST_MEMORY>(mf, mpi.comm, psp,
                                           sDm_skij.local(), nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                           pgrid, bsize);
  mpi.comm.barrier();

  std::string filename = coqui_output + ".mbpt.h5";
  app_log(2, "Dump the matrix elements of the Hartree potential: ");
  app_log(2, "  - h5 file: {}", filename);
  app_log(2, "  - h5 dataset = scf/iter{}/VH_skij\n", scf_iter);

  h5::group iter_grp;
  if (mpi.comm.root()) {
    h5::file file;
    try {
      file = h5::file(filename, 'a');
    } catch(...) {
      APP_ABORT("Failed to open h5 file: {}, mode:a",filename);
    }
    auto scf_grp = h5::group(file).open_group("scf");
    utils::check(scf_grp.has_subgroup("iter"+std::to_string(scf_iter)),
                 "dump_hartree: \"scf/iter{}\" does not exist!");

    iter_grp = scf_grp.open_group("iter"+std::to_string(scf_iter));
    math::nda::h5_write(iter_grp, "VH_skij", dVH);
  } else {
    math::nda::h5_write(iter_grp, "VH_skij", dVH);
  }
}

/**
 * Overlap matrix associated with a MF object in a distributed array
 * @return - A distributed array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto ovlp(mf::MF& mf, boost::mpi3::communicator& comm, 
          nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
          std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                               nda::range(-1,-1), k_range, b_range, bz); 
    return detail::gen_ovlp<MEM,false>(comm,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"ovlp",comm,pgrid,bz);
  }
}
/**
 * Diagonal elements of the overlap matrix associated with a MF object in a distributed array
 * @return - A distributed array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto ovlp_diagonal(mf::MF& mf, boost::mpi3::communicator& comm, 
          nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
          std::array<long,3> pgrid = {0}, std::array<long,3> bz = {1,1,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',
             {pgrid[0],pgrid[1],pgrid[2],1},nda::range(-1,-1), k_range, b_range,
             {bz[0],bz[1],bz[2],2048});
    return detail::gen_ovlp<MEM,true>(comm,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_diag_1B_from_file<MEM>(mf,"ovlp",comm,pgrid,bz);
  }
}

/**
 * Overlap matrix associated with a MF object in a shared memory array
 * @return - A shared memory array of overlap matrix with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_ovlp(mf::MF &mf, math::shm::shared_array<Array_4D_t> &sS_skij) {
  long np = sS_skij.internode_comm()->size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 2048);

  if (sS_skij.node_comm()->root()) {
    auto dS  = ovlp<HOST_MEMORY>(mf, *sS_skij.internode_comm(), nda::range(mf.nkpts_ibz()),
                                 nda::range(mf.nbnd()), pgrid, bsize);
    auto S_loc = sS_skij.local();
    S_loc(dS.local_range(0), dS.local_range(1), dS.local_range(2), dS.local_range(3)) = dS.local();
  }
  sS_skij.communicator()->barrier();
  sS_skij.all_reduce();
}

/**
 * Exchange-correlation hamiltonian associated with MF object in a distributed array
 * @param mf    [input] - mean-field object
 * @param comm  [input] - communicator
 * @param pgrid [input] - processor grid for the distributed array
 * @param bz    [input] - block size for the distributed array
 * @return - A distributed array of non-interacting one-body Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<MEMORY_SPACE MEM = HOST_MEMORY>
auto Vxc(mf::MF &mf, boost::mpi3::communicator &comm,
         nda::range k_range = {-1,-1}, nda::range b_range = {-1,-1},
         std::array<long,4> pgrid = {0}, std::array<long,4> bz = {1,1,2048,2048})
{
  if(k_range == nda::range{-1,-1}) k_range = nda::range(mf.nkpts_ibz());
  if(b_range == nda::range{-1,-1}) b_range = nda::range(mf.nbnd());
  // this is, unfortunately, code dependent, so fork here!
  if (mf.mf_type() == mf::qe_source or mf.mf_type() == mf::bdft_source) {
    using larray = memory::array<MEM,ComplexType,4>;
    auto psi = mf::read_distributed_orbital_set_ibz<larray>(mf,comm,'w',pgrid,
                                                            nda::range(-1,-1), k_range, b_range, bz);
    return detail::gen_Vxc<MEM>(mf,k_range,b_range,psi);
  } else {
    utils::check(mf.mf_type() == mf::pyscf_source, "Source mismatch");
    utils::check(k_range.size() == mf.nkpts_ibz(), "No k_range with pyscf backend yet.");
    utils::check(b_range.size() == mf.nbnd(), "No b_range with pyscf backend yet.");
    return detail::pyscf_read_1B_from_file<MEM>(mf,"Vxc",comm,pgrid,bz);
  }
}


/**
 * Exchange-correlation Hamiltonian associated with a MF object in a shared memory array
 * @return - A shared memory array of exchange-correlation Hamiltonian
 *           with global shape = (nspin, nkpts, nbnd, nbnd)
 */
template<nda::MemoryArrayOfRank<4> Array_4D_t>
void set_Vxc(mf::MF &mf, math::shm::shared_array<Array_4D_t> &sVxc_skij) {
  long np = sVxc_skij.internode_comm()->size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(4, "One-body Hamiltonian in distributed array: ");
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 2048);

  if (sVxc_skij.node_comm()->root()) {
    auto dVxc = hamilt::Vxc<HOST_MEMORY>(mf, *sVxc_skij.internode_comm(),
                                         nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                         pgrid, bsize);
    auto Vxc_loc = sVxc_skij.local();
    Vxc_loc(dVxc.local_range(0), dVxc.local_range(1), dVxc.local_range(2), dVxc.local_range(3)) = dVxc.local();
  }
  sVxc_skij.communicator()->barrier();
  sVxc_skij.all_reduce();
}

template<typename MPI_t>
void dump_vxc(MPI_t &mpi, mf::MF &mf, std::string coqui_output) {
  long np = mpi.comm.size();
  long np_s = (np % mf.nspin()== 0)? mf.nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf.nkpts_ibz(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, (mf.nbnd())/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 2048};
  app_log(2, "Evaluate the matrix elements of the exchange-correlation potential: ");
  app_log(2, "  - mean-field backend = {}", mf::mf_source_enum_to_string(mf.mf_type()));
  app_log(4, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(4, "  - bsize = ({}, {}, {}, {})", 1, 1, blk_i, 2048);
  app_log(2, "");
  auto dVxc = hamilt::Vxc<HOST_MEMORY>(mf, mpi.comm,
                                       nda::range(mf.nkpts_ibz()), nda::range(mf.nbnd()),
                                       pgrid, bsize);
  mpi.comm.barrier();

  std::string filename = coqui_output + ".mbpt.h5";
  app_log(2, "Dump the matrix elements of the exchange-correlation potential: ");
  app_log(2, "  - h5 file: {}", filename);
  app_log(2, "  - h5 dataset = system/Vxc_skij\n");

  h5::group sys_grp;
  if (mpi.comm.root()) {
    h5::file file;
    try {
      file = h5::file(filename, 'a');
    } catch(...) {
      APP_ABORT("Failed to open h5 file: {}, mode:a",filename);
    }
    h5::group grp(file);

    if (grp.has_subgroup("system")) {
      sys_grp = grp.open_group("system");
      long ns, nkpts, nkpts_ibz, npol, nbnd;
      h5::h5_read(sys_grp, "number_of_spins", ns);
      h5::h5_read(sys_grp, "number_of_kpoints", nkpts);
      h5::h5_read(sys_grp, "number_of_kpoints_ibz", nkpts_ibz);
      h5::h5_read(sys_grp, "number_of_orbitals", nbnd);
      h5::h5_read(sys_grp, "number_of_polarizations", npol);
      utils::check(ns == mf.nspin(), "dump_vxc: inconsistent \"nspin\" in coqui mean-field and {}", filename);
      utils::check(nkpts == mf.nkpts(), "dump_vxc: inconsistent \"nkpts\" in coqui mean-field and {}", filename);
      utils::check(nkpts_ibz == mf.nkpts_ibz(), "dump_vxc: inconsistent \"nkpts_ibz\" in coqui mean-field and {}",
                   filename);
      utils::check(nbnd == mf.nbnd(), "dump_vxc: inconsistent \"nbnd\" in coqui mean-field and {}", filename);
      utils::check(npol == mf.npol(), "dump_vxc: inconsistent \"npol\" in coqui mean-field and {}", filename);
    } else {
      sys_grp = grp.create_group("system");
    }

    math::nda::h5_write(sys_grp, "Vxc_skij", dVxc);
  } else {
    math::nda::h5_write(sys_grp, "Vxc_skij", dVxc);
  }
}

}

#endif
