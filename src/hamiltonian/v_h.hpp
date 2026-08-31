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


#ifndef HAMILTONIAN_ADD_V_H_HPP
#define HAMILTONIAN_ADD_V_H_HPP

#include <cmath>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/symmetry.hpp"
#include "potentials/potentials.hpp"

namespace hamilt
{

using utils::mpi_context_t;
using boost::mpi3::communicator;
using boost::mpi3::shared_communicator;

// TODO:
//   1. construct rho(r) from non-orthogonal basis
//   2. symmetrize rho(r)

/*
 * MAM: Implementation uses 2 copies of the charge density per core in a node. 
 *      If you need to save memory, redistribute to g-vector only distribution and
 *      work on segment of grid
 */
/**
 * Compute the Hartree potential via FFT using the diagonals of the density matrix
 * @tparam Arr - Type parameter for the Hartree potential
 * @param mpi  - [input] MPI handler
 * @param type - [input] Type of the Coulomb kernel
 * @param npol - [input] Number of polarizations
 * @param mesh - [input] FFT mesh
 * @param lattv - [input] Lattice vectors
 * @param recv - [input] Reciprocal vectors
 * @param k2g - [input] Mapping from the "wfc" to the "density" g-grid.
 * @param kpts - [input] K-points (nkpts, 3) in cartesian coordinates
 * @param kp_to_ibz - [input] Mapping from a k-point to its symmetry-related one
 * @param kp_trev   - [input] Whether time-reversal symmetry is needed in combination with the symmetry in kp_symm
 * @param kp_symm   - [input] Index of symmetry operation that connects kpts/kpts_crys to IRBZ
 * @param symm_list - [input] List of symmetry operations
 * @param nii - [input] Diagonals of the density matrix (s, k, a)
 * @param psi - [input] Single-particle basis in a distributed array (s,k,a,g),
 *                      where g is in the 'w' grid
 * @param svr  - [output] Hartree potential V_H(r) on a real-space mesh that is consistent with "mesh"
 */
template<nda::ArrayOfRank<1> Arr>
void v_h(mpi_context_t<communicator,shared_communicator> &mpi,
         pots::potential_t& vG,
         int npol,
         nda::stack_array<long, 3> const& mesh,
         nda::stack_array<double, 3, 3> const& lattv,
         nda::stack_array<double, 3, 3> const& recv,
         nda::ArrayOfRank<1> auto const& k2g_,
         nda::ArrayOfRank<2> auto const& kpts,
         nda::ArrayOfRank<1> auto const& kp_to_ibz,
         nda::ArrayOfRank<1> auto const& kp_trev,
         nda::ArrayOfRank<1> auto const& kp_symm,
         std::vector<utils::symm_op> const& symm_list,
         nda::ArrayOfRank<3> auto const& nii_, 
         math::nda::DistributedArrayOfRank<4> auto const& psi,
         bool symmetrize_rho_r,
         math::shm::shared_array<Arr>& svr)    // GPU!!!
{
  decltype(nda::range::all) all;
  constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(psi.local())>>();
#if defined(ENABLE_DEVICE)
  using nda::tensor::reduce;
  using nda::tensor::cutensor::cutensor_desc;
  using nda::tensor::cutensor::elementwise_binary;
  using nda::tensor::op::ID;
  using nda::tensor::op::MUL;
  using nda::tensor::op::CONJ;
  using nda::tensor::cutensor::to_cutensor;
#endif

  // copy to MEM if needed
  auto nii = memory::to_memory_space<MEM>(nii_());
  auto k2g = memory::to_memory_space<MEM>(k2g_());
  
  long nbnd_local = psi.local_shape()[2];
  long nk = kp_to_ibz.shape(0);
  long ngm = k2g.extent(0);
  auto ploc = psi.local();
  // non-collinear needs change!
  double ns_scl = ((psi.global_shape()[0]==1) and (npol==1)?2.0:1.0);
 
  long nnr = mesh(0)*mesh(1)*mesh(2);
  utils::check( psi.grid()[3] == 1, "Grid mismatch.");
  utils::check( psi.global_shape()[0] == nii.extent(0), "Shape mismatch.");
  utils::check( psi.global_shape()[1] == nii.extent(1), "Shape mismatch.");
  utils::check( psi.global_shape()[2] == nii.extent(2), "Shape mismatch.");
  utils::check( psi.global_shape()[3] == npol*ngm, "Shape mismatch.");
  utils::check( svr.size() == nnr, "Shape mismatch.");

  // vol = (2*pi)^3 / det(recv)
  // scl = 4*pi / vol / nk = 4*pi * det(recv) / (2*pi)^3 / nk = det(recv) / (2 * pi^2 * nk)
  RealType vol =   recv(0,0) * ( recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1) )
                - recv(1,0) * ( recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1) )
                + recv(2,0) * ( recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1) ) ;
  vol = 248.050213442399 / vol; 

  // n(r) = sum_s_k_a  conj( psi(s,k,a,r) ) nii(s,k,a) psi(s,k,a,r) 

  long nb_per_blk = 1;

  // tune later based on available memory
  if constexpr (MEM != HOST_MEMORY)
    nb_per_blk = nbnd_local;

  // no need to redistribute psi. Accumulate locally 
  memory::array<MEM, long, 1> k2g_rotate(k2g.shape(0));
  memory::array<MEM, ComplexType, 2> pr(nb_per_blk, nnr);
  memory::array<MEM, ComplexType, 1> nr(nnr);

  auto pr4d = nda::reshape(pr,std::array<long,4>{nb_per_blk,mesh(0),mesh(1),mesh(2)});
  math::nda::fft<true> F(pr4d);
  nda::array<ComplexType,1> *Xft = nullptr;
  nda::stack_array<double, 3> Gs;
  Gs() = 0.0;

  nr() = ComplexType(0.0);
  for( auto [is,s] : itertools::enumerate(psi.local_range(0)) ) {
    for( auto [ik_sym,k_sym] : itertools::enumerate(psi.local_range(1)) ) {
      for (long k = 0; k < nk; ++k) {

        if (kp_to_ibz(k) != k_sym)
          continue;

        // apply symmetry on the k2g mapping
        k2g_rotate() = k2g();
        if(kp_trev(k) or kp_symm(k) > 0) {
          utils::transform_k2g(kp_trev(k), symm_list[kp_symm(k)], Gs, mesh,
                                 kpts(k_sym, all), k2g_rotate, Xft);
        }
         
        for (auto p: nda::range(npol) ) {
          for (auto ia: nda::range(psi.local_range(2).first(), 
                                   psi.local_range(2).last(), nb_per_blk)) {

            long nb = std::min(nb_per_blk, psi.local_range(2).last() - ia);
            nda::range a_rng(ia,ia+nb);
            auto pr_l = pr(nda::range(nb),all);

            // w -> g
            pr_l() = ComplexType(0.0);
            nda::copy_select(true, 1, k2g_rotate, 
                             ComplexType(1.0), ploc(is, ik_sym, a_rng + (-psi.local_range(2).first()), 
                             nda::range(p*ngm,(p+1)*ngm)),
                             ComplexType(0.0), pr_l);

            // g -> r
            F.backward(pr4d);

            // accumulate
            if constexpr (MEM==HOST_MEMORY) {
              if (not kp_trev(k))
                for(auto [i,a] : itertools::enumerate(a_rng))
                  nr += nii(is, k_sym, a) * nda::conj(pr_l(i,all)) * pr_l(i,all);
              else
                for(auto [i,a] : itertools::enumerate(a_rng))
                  nr += nda::conj(nii(is, k_sym, a)) * nda::conj(pr_l(i,all)) * pr_l(i,all);
            } else {
#if defined(ENABLE_DEVICE)
              auto pr_t = to_cutensor(pr_l);
              auto nr_t = to_cutensor(nr);
              auto nii_a = nii(is, k_sym, a_rng);

              // p(a,r) = conj(p(a,r))*p(a,r)
              elementwise_binary(ComplexType(1.0),pr_t,CONJ,pr_l.data(),"ar",
                                 ComplexType(1.0),pr_t,ID,pr_l.data(),"ar",
                                 pr_l.data(),MUL);

              // n(r) += nii(a) * p(a,r)
              if(kp_trev(k))
                nda::tensor::contract( ComplexType(1.0), nda::conj(nii_a), "a",
                                       pr_l, "ar", ComplexType(1.0), nr, "r");                      
              else
                nda::tensor::contract( ComplexType(1.0), nii_a, "a",
                                       pr_l, "ar", ComplexType(1.0), nr, "r");                      
#else
              static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif
            }

          }
        }
      }
    } 
  }

  // coulomb interaction in pr 
  // MAM: avoiding additional use of shared memory to store v(G), computing on the fly, this is quick
  pr(0,all) = ComplexType(0.0);
  {
    auto [g0,g1] = itertools::chunk_range(1,nnr,mpi.comm.size(),mpi.comm.rank());
    nda::stack_array<double,3> G0 = {0.0,0.0,0.0};
      vG.evaluate_in_mesh(nda::range(g0,g1),pr(0,all),mesh,lattv,recv,G0,G0);
  }

  mpi.comm.reduce_in_place_n(nr.data(),nr.size(),std::plus<>{},0); 
  mpi.comm.reduce_in_place_n(pr.data(),nnr,std::plus<>{},0); 

  // if symmetrize, all_reduce and symm your segment [g0,g1]
  if (symmetrize_rho_r) {
    utils::check(false, "v_h: symmetrize rho(r) is not implemented yet!");
  } else {
    app_log(2, "  [WARNING] v_h: symmetrize_rho_r = false. The Hartree potential may not be fully symmetry-adapted.\n");
  }
 
  // calculate potential in root
  if( mpi.comm.root() ) {

    auto nr3d = nda::reshape(nr,std::array<long,3>{mesh(0),mesh(1),mesh(2)});
    math::nda::fft<false> Fn(nr3d,math::fft::FFT_MEASURE | math::fft::FFT_PRESERVE_INPUT);
    // r -> g 
    Fn.forward(nr3d);      

    if constexpr (MEM == HOST_MEMORY) {
      // vH(G) = vcoul(G) * n(G) -> pr(G)
      ComplexType nrm = ( ns_scl / ( vol * double(nk) ) );
      nr() *= (pr(0,all) * nrm);
    } else {
#if defined(ENABLE_DEVICE)
      auto pr_t = to_cutensor(pr(0,all));
      auto nr_t = to_cutensor(nr);

      // vH(G) = vcoul(G) * n(G) -> pr(G)
      ComplexType nrm = ( ns_scl / ( vol * double(nk) ) );
      elementwise_binary(nrm,nr_t,ID,nr.data(),"r",
                         ComplexType(1.0),pr_t,ID,pr.data(),"r",
                         nr.data(),MUL);
#endif      
    }

    // g -> r 
    Fn.backward(nr3d);

    // copy to shm buffer
    svr.local() = nr();
  } 

  // communicate among heads of nodes
  if(mpi.node_comm.root())  
    mpi.internode_comm.broadcast_n(svr.local().data(),nnr,0);
  mpi.comm.barrier();

}

/**
 * Calculates the Hartree potential V_H:
 *
 * V_H(r) = 1/(4*pi) \int \rho(r')/|r-r'| dr'
 *
 * 1. calculate charge density in the real space and fft to G space
 * 2. symmetrize density (?)
 * 3. multiply by Coulomb potential
 * 4. fft to real space
 *
 * Careful, vr should be assumed to be in shared memory!
 *
 * @tparam Arr - Type parameter for svr
 * @param mpi  - [input] MPI handler
 * @param type - [input] Type of the Coulomb kernel
 * @param npol - [input] Number of polarizations
 * @param mesh - [input] FFT mesh (Nx, Ny, Nz)
 * @param lattv - [input] Lattice vectors
 * @param recv - [input] Reciprocal vectors
 * @param k2g  - [input] Mapping from the "wavefunction" to the "density" g-grid.
 * @param kpts - [input] K-points (nkpts, 3) in cartesian coordinates
 * @param kp_to_ibz - [input] Mapping from a k-point to its symmetry-related one
 * @param kp_trev   - [input] Whether time-reversal symmetry is needed in combination with the symmetry in kp_symm
 * @param kp_symm   - [input] Index of symmetry operation that connects kpts/kpts_crys to IRBZ
 * @param symm_list - [input] List of symmetry operations
 * @param nij  - [input] Electron density (s, k, a, b)
 * @param psi  - [input] Single-particle basis (s, k, a, g) where g in on the "wavefunction" g-grid
 * @param svr  - [output] Hartree potential V_H(r) on a real-space mesh that is consistent with "mesh"
 */
template<nda::ArrayOfRank<1> Arr>
void v_h(mpi_context_t<communicator,shared_communicator> &mpi,
         pots::potential_t& vG,
         int npol,
         nda::stack_array<long, 3> const& mesh,
         nda::stack_array<double, 3, 3> const& lattv,
         nda::stack_array<double, 3, 3> const& recv,
         nda::ArrayOfRank<1> auto const& k2g_,
         nda::ArrayOfRank<2> auto const& kpts,
         nda::ArrayOfRank<1> auto const& kp_to_ibz,
         nda::ArrayOfRank<1> auto const& kp_trev,
         nda::ArrayOfRank<1> auto const& kp_symm,
         std::vector<utils::symm_op> const& symm_list,
         nda::ArrayOfRank<4> auto const& nij_,  
         math::nda::DistributedArrayOfRank<4> auto const& psi,
         bool symmetrize_rho_r,
         math::shm::shared_array<Arr>& svr)
{
#if defined(ENABLE_DEVICE)
  using nda::tensor::reduce;
  using nda::tensor::cutensor::cutensor_desc;
  using nda::tensor::cutensor::elementwise_binary;
  using nda::tensor::op::ID;
  using nda::tensor::op::MUL;
  using nda::tensor::cutensor::to_cutensor;
#endif

  using nda::range;
  decltype(range::all) all;
  constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(psi.local())>>();

  // copy to MEM if needed. 
  auto nij = memory::to_memory_space<MEM>(nij_());
  auto k2g = memory::to_memory_space<MEM>(k2g_());

  auto vr = svr.local();
  svr.set_zero(); 
  long nnr = mesh(0)*mesh(1)*mesh(2);
  long ngm = k2g.extent(0);
  //static bool first = true;
  double ns_scl = ((psi.global_shape()[0]==1) and (npol==1)?2.0:1.0);

  utils::check( psi.grid()[3] == 1, "Grid mismatch.");
  utils::check( psi.global_shape()[0] == nij.extent(0), "v_h: psi.global_shape()[0] != nij.extent(0)");
  utils::check( psi.global_shape()[1] == nij.extent(1), "v_h: psi.global_shape()[1] != nij.extent(1)");
  utils::check( psi.global_shape()[2] == nij.extent(2), "v_h: psi.global_shape()[2] != nij.extent(2)");
  utils::check( psi.global_shape()[2] == nij.extent(3), "v_h: psi.global_shape()[2] != nij.extent(3)");
  utils::check( psi.global_shape()[3] == npol*ngm, "v_h: psi.global_shape()[3] != npol*k2g.extent(0)");
  utils::check( mpi.comm == *(psi.communicator()), "Communicator mismatch.");
  utils::check( svr.size() == nnr, "v_h: vr.size ({}) != nnr ({})", vr.size(), nnr);

  // memory report if first==true if needed
  // MAM: if memory is a problem, e.g. large supercells, do not split over kpoints!

  long nk = kp_to_ibz.shape(0);
  long nk_ibz = psi.global_shape()[1];
  long nbnd = psi.global_shape()[2];
  long M_loc = psi.local_shape()[2];
  auto ploc = psi.local();

  // vol = (2*pi)^3 / det(recv)
  // scl = 4*pi / vol / nk = 4*pi * det(recv) / (2*pi)^3 / nk = det(recv) / (2 * pi^2 * nk)
  RealType vol = recv(0,0) * ( recv(1,1)*recv(2,2) - recv(1,2)*recv(2,1) )
                - recv(1,0) * ( recv(0,1)*recv(2,2) - recv(0,2)*recv(2,1) )
                + recv(2,0) * ( recv(0,1)*recv(1,2) - recv(0,2)*recv(1,1) ) ;
  vol = 248.050213442399 / vol;

  // communicator over k-points
  mpi3::communicator k_comm = mpi.comm.split(psi.origin()[0]*nk_ibz+psi.origin()[1],mpi.comm.rank());
  mpi3::communicator k_intercomm = mpi.comm.split(k_comm.rank(),mpi.comm.rank());
  utils::check(k_comm.size()*k_intercomm.size() == mpi.comm.size(),"Communicator partition mismatch.");

  // partition fft grid ahead
  auto [r0,r1] = itertools::chunk_range(0,nnr,k_comm.size(),k_comm.rank()); 
  long nnr_loc = r1-r0;
  {
    // check the partition of nnr is consistent in all k_intercomm
    long r_[2]; r_[0]=r0; r_[1]=r1;
    k_intercomm.all_reduce_in_place_n(r_,2,mpi3::max<>{});
    utils::check(r_[0] == r0 and r_[1] == r1, "Comm partition mismatch.");
  }

  memory::array<MEM,ComplexType,1> rho(nnr_loc);
  rho() = ComplexType(0.0);

  {
    // fft to r space and redistribute 
    // using distributed_array_view, to be able to reuse memory 
    // precalculate how much memory is needed
    long mem = nbnd*nnr_loc + std::max(M_loc*nnr, nbnd*nnr_loc);

    // buffer
    memory::array<MEM,ComplexType,1> buff(mem); 

    // distributed over fft grid
    memory::array_view<MEM,ComplexType,2> psi_r({nbnd,nnr_loc},buff.data());  
    // distributed over bands, for fft
    memory::array_view<MEM,ComplexType,2> psi_b({M_loc,nnr},psi_r.data()+psi_r.size());  
    memory::array_view<MEM,ComplexType,2> T({nbnd,nnr_loc},psi_b.data());      // temporary 

    using local_Array_t = memory::array<MEM,ComplexType,2>; 
    using math::nda::distributed_array_view;
    distributed_array_view<local_Array_t,decltype(k_comm)> dpsi_b(std::addressof(k_comm),
            {k_comm.size(),1},{nbnd,nnr},{psi.origin()[2],0},{1,1},psi_b);
    distributed_array_view<local_Array_t,decltype(k_comm)> dpsi_r(std::addressof(k_comm),
            {1,k_comm.size()},{nbnd,nnr},{0,r0},{1,1},psi_r);

    auto pb4d = nda::reshape(psi_b,std::array<long,4>{psi_b.extent(0),mesh(0),mesh(1),mesh(2)});
    math::nda::fft<true> F(pb4d);
    nda::array<ComplexType,1> *Xft = nullptr;
    nda::stack_array<double, 3> Gs;
    Gs() = 0.0;
    memory::array<MEM,long,1> k2g_rotate(k2g.shape(0));

    for( auto [is,s] : itertools::enumerate(psi.local_range(0)) ) {
      for( auto [ik_sym,k_sym] : itertools::enumerate(psi.local_range(1)) ) {
        for (long k = 0; k < nk; ++k) {

          if (kp_to_ibz(k) != k_sym)
            continue;

          // apply symmetry on the k2g mapping
          k2g_rotate() = k2g();
          if(kp_trev(k) or kp_symm(k) > 0) 
            utils::transform_k2g(kp_trev(k), symm_list[kp_symm(k)], Gs, mesh,
                                 kpts(k_sym,all), k2g_rotate, Xft);

          for( auto p : range(npol) ) {

            // w -> g
            psi_b() = ComplexType(0.0);
            nda::copy_select(true, 1, k2g_rotate, 
                             ComplexType(1.0), ploc(is, ik_sym, all, range(p*ngm,(p+1)*ngm)), 
                             ComplexType(0.0), psi_b);

            // g -> r
            F.backward(pb4d);

            // redistribute
            math::nda::redistribute(dpsi_b, dpsi_r);

            // make psi_r = conj(psi_r) if kp_trev(k) is true,
            // such that no conjugation is needed for nij in the next step
            if (kp_trev(k)) psi_r() = nda::conj(psi_r);

            //accumulate density: Only diagonal components in polarization
            nda::blas::gemm(nij(s, k_sym, all, all), psi_r, T);

            if constexpr ( MEM == HOST_MEMORY ) {
              for (auto ib : nda::range(nbnd) ) {
                // take kp_trev(k) into account here
                if ( not kp_trev(k) )
                  rho += nda::conj(psi_r(ib, all)) * T(ib, all);
                else
                  rho += psi_r(ib,all) * nda::conj(T(ib,all));
              }
            } else {
#if defined(ENABLE_DEVICE)
              // rho(r) += psi_r(i,r) * T(i,r)
              if( not kp_trev(k))
                nda::tensor::contract( ComplexType(1.0), nda::conj(psi_r), "ir",   
                                       T, "ir", ComplexType(1.0), rho, "r");   
              else
                nda::tensor::contract( ComplexType(1.0), psi_r, "ir",   
                                       nda::conj(T), "ir", ComplexType(1.0), rho, "r");    
#else         
              static_assert(MEM!=HOST_MEMORY,"Error: Device dispatch without device support.");
#endif     
            }

          } // pol
        } // k_sym
      } // k
    } // ispin
  }
  k_intercomm.reduce_in_place_n(rho.data(),nnr_loc,std::plus<>{},0);
  mpi.comm.barrier();

  // only allocated at root below
  memory::array<MEM, ComplexType, 1> nr;

  if(k_intercomm.root()) {

    // at this point, the set of mpi tasks with k_intercomm.root()==true contain the full density 
    if(mpi.comm.root()) nr = memory::array<MEM, ComplexType, 1>(nnr,ComplexType(0.0));
    nda::array<int,1> cnt(k_comm.size()),disp(k_comm.size()); 
    disp(0)=0;
    for( int n=0, icnt=0; n<k_comm.size(); ++n) {
      auto [r0_,r1_] = itertools::chunk_range(0,nnr,k_comm.size(),n);
      cnt(n) = int(r1_-r0_);
      icnt += cnt(n);
      if((n+1)<k_comm.size()) disp(n+1) = icnt;
    }
    k_comm.gatherv_n(rho.data(),nnr_loc,nr.data(),cnt.data(),disp.data(),0);

    if (symmetrize_rho_r) {
      utils::check(false, "v_h: symmetrize rho(r) is not implemented yet!");
    } else {
      app_log(2, "  [WARNING] v_h: symmetrize_rho_r = false. The Hartree potential may not be fully symmetry-adapted.\n");
    }

    // use vr to store potential and reduce onto mpi.comm.root
    nda::stack_array<double,3> G0 = {0.0,0.0,0.0};
    vG.evaluate_in_mesh(nda::range(r0,r1),vr,mesh,lattv,recv,G0,G0);
  }
  
  mpi.comm.barrier();
  if(mpi.node_comm.root())
    mpi.internode_comm.reduce_in_place_n(vr.data(),nnr,std::plus<>{},0);

  // calculate potential in root
  if( mpi.comm.root() ) {

    auto n3d = nda::reshape(nr,std::array<long,3>{mesh(0),mesh(1),mesh(2)});
    math::nda::fft<false> Fn(n3d,math::fft::FFT_MEASURE | math::fft::FFT_PRESERVE_INPUT);

    // r -> g 
    Fn.forward(n3d);

    if constexpr (MEM == HOST_MEMORY) {
      // vH(G) = vcoul(G) * n(G) -> pr(G)
      ComplexType nrm = ( ns_scl / ( vol * double(nk) ) );
      nr() *= (vr() * nrm);
    } else {
#if defined(ENABLE_DEVICE)
      auto vr_d = memory::to_memory_space<MEM>(vr()); 
      auto vr_t = to_cutensor(vr_d);
      auto nr_t = to_cutensor(nr);
      
      // vH(G) = vcoul(G) * n(G) -> nr(G)
      ComplexType nrm = ( ns_scl / ( vol * double(nk) ) );
      elementwise_binary(nrm,nr_t,ID,nr.data(),"r",
                         ComplexType(1.0),vr_t,ID,vr_d.data(),"r",
                         nr.data(),MUL);
#endif
    }

    // g -> r 
    Fn.backward(n3d);

    // copy to shm buffer
    vr() = nr();
  }

  // communicate among heads of nodes
  if(mpi.node_comm.root())
    mpi.internode_comm.broadcast_n(vr.data(),nnr,0);
  mpi.comm.barrier();

}

}

#endif
