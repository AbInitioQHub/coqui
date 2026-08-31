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


#ifndef HAMILTONIAN_ADD_VLOC_HPP
#define HAMILTONIAN_ADD_VLOC_HPP

#include <cmath>

#include "configuration.hpp"
#include "grids/g_grids.hpp"
#include "nda/concepts.hpp"

#include "arch/arch.h"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "utilities/kpoint_utils.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "hamiltonian/potentials.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/tensor.hpp"
#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/nda_functions.hpp"

namespace hamilt
{

// MAM: Not yet working for mf backends with npol_in_basis!=npol or nspin_in_basis!=nspin
/**
 * Adds local potential in "wavefunction" grid via the following steps:
 *   1. Apply the local potential in the real space "mesh"
 *   2. FFT to the "density" grid.
 *   3. Convert from the "density" space to the "wavefunction" space (w).
 *   Note: Number of spins and polarizations in local potential can be either 1 or
 *         consistent with psi/hpsi.
 *
 * @param mesh - [INPUT] FFT mesh
 * @param wfc_to_rho - [INPUT] Mapping from wfc g-grid to density g-grid.
 * @param vr   - [INPUT] Local potential vr(s,p*p,r), where r should be compatible with "mesh".
 *                       s: spin, p:polarization. 
 *                       s (p) must be either nspin (npol) or 1.   
 * @param psi  - [INPUT] Single-particle basis in a distributed array (s,k,a,p*g),
 *                       where g is in the 'w' grid.
 * @param hpsi - [INPUT] \hat{h} * psi where h is an arbitrary local operator
 *               [OUTPUT] (\hat{h} + vloc) * psi
 */
void add_vloc(int npol, 
              nda::ArrayOfRank<1> auto const& mesh,
              nda::MemoryArrayOfRank<1> auto const& wfc_to_rho_, 
              nda::MemoryArrayOfRank<3> auto const& vr_,
              math::nda::DistributedArrayOfRank<4> auto const& psi,
              math::nda::DistributedArrayOfRank<4> auto & hpsi)
{
  decltype(nda::range::all) all;
  constexpr auto MEM = memory::get_memory_space<std::decay_t<decltype(psi.local())>>();
  static_assert(MEM == memory::get_memory_space<std::decay_t<decltype(hpsi.local())>>(), "Memory mismatch.");

  // copy to MEM if needed
  auto wfc_to_rho = memory::to_memory_space<MEM>(wfc_to_rho_());
  auto vr = memory::to_memory_space<MEM>(vr_());

  utils::check(psi.global_shape() == hpsi.global_shape(), "add_vloc: Global shape mismatch");
  utils::check(psi.origin() == hpsi.origin(), "add_vloc: Origin mismatch");

  long nnr = mesh(0)*mesh(1)*mesh(2);
  long nspin_in_vr = vr.shape()[0];
  long npol2_in_vr = vr.shape()[1];
  long nbnd = psi.local_shape()[2];
  long ngm = wfc_to_rho.size();

  // check psi is consistent with 'w' grid
  utils::check(psi.grid()[3] == 1, "Processor grid mismatch");
  utils::check(psi.local_range(3) == nda::range{0,npol*ngm}, "Range mismatch");
  utils::check(vr.shape()[2] == nnr, "Size mismatch");
  utils::check(nspin_in_vr == psi.global_shape()[0] or nspin_in_vr == 1, "Error: add_vloc nspin inconsistency.");
  utils::check(npol2_in_vr == npol*npol or npol2_in_vr == 1, "Error: add_vloc npol inconsistency.");

  auto psiloc = psi.local();
  auto hpsiloc = hpsi.local();
  long nb_per_blk = 1;

  // tune later based on available memory
  if constexpr (MEM != HOST_MEMORY) 
    nb_per_blk = nbnd;     
    
  // local potential: apply in real space and fft back 
  // consider blocks over kpoints if too slow in GPUs
  memory::array<MEM, ComplexType, 2> psir(nb_per_blk,nnr);
  psir() = ComplexType(0.0);
  auto psir4D = nda::reshape(psir,std::array<long,4>{nb_per_blk,mesh(0),mesh(1),mesh(2)});
  math::nda::fft<true> F(psir4D);

  for( auto [is,s] : itertools::enumerate(psi.local_range(0)) ) {
    for (auto [ik, k]: itertools::enumerate(psi.local_range(1))) {
      for (auto ip: nda::range(npol)) {
        for (auto iq: nda::range(npol)) {
          auto vr_s_pq = vr( (nspin_in_vr==1?0:s), (npol2_in_vr==1?0:ip*npol+iq), all);
          if(npol2_in_vr==1 and ip!=iq) continue; // skip if npol2_in_vr==1 and off-diagonal
          for (auto ia: nda::range(0, nbnd, nb_per_blk)) {

            long nb = std::min(nb_per_blk, nbnd - ia);
            psir() = ComplexType(0.0);
            nda::range a_rng(ia, ia + nb);
            auto pska = psiloc(is, ik, a_rng, nda::range(iq*ngm,(iq+1)*ngm));
            auto hpska = hpsiloc(is, ik, a_rng, nda::range(ip*ngm,(ip+1)*ngm));
            auto psir2D = psir(nda::range(nb), all);

            // psir( wfc_to_rho(g) ) = psi(g)  expand to fft grid
            nda::copy_select(true, 1, wfc_to_rho, ComplexType(1.0), pska, ComplexType(0.0), psir2D);

            // psig -> psir
            F.backward(psir4D);

            // MAM: consider storing psir(iq) and  and bringing loop over ip inside

            // apply local potential
            if constexpr(MEM == HOST_MEMORY)
            {
              for (auto ib: nda::range(nb))
                psir(ib, all) *= vr_s_pq;
            } else {
#if defined(ENABLE_DEVICE)
              using nda::tensor::cutensor::cutensor_desc;
              using nda::tensor::cutensor::elementwise_binary;
              using nda::tensor::op::ID;
              using nda::tensor::op::MUL;
              cutensor_desc<ComplexType,1> a_t(vr_s_pq);
              cutensor_desc<ComplexType,2> b_t(psir2D);
              elementwise_binary(ComplexType(1.0),a_t,ID,vr_s_pq.data(),"r",
                                 ComplexType(1.0),b_t,ID,psir2D.data(),"ir",
                                 psir2D.data(),MUL);
#endif
            }

            // psir -> psig
            F.forward(psir4D);

            // hpsi( g ) += psig( wfc_to_rho(g) ): Accumulate on truncated grid
            nda::copy_select(false, 1, wfc_to_rho, ComplexType(1.0), psir2D, ComplexType(1.0), hpska);

          } // ia
        } // q
      } // p
    } // ik
  } // is
}

void add_vloc(int npol,
              nda::ArrayOfRank<1> auto const& mesh,
              nda::MemoryArrayOfRank<1> auto const& wfc_to_rho,
              nda::MemoryArrayOfRank<1> auto const& vr,
              math::nda::DistributedArrayOfRank<4> auto const& psi,
              math::nda::DistributedArrayOfRank<4> auto & hpsi) {

  auto vr_3D = nda::reshape(vr, std::array<long,3>{1, 1, vr.shape(0)});
  add_vloc(npol, mesh, wfc_to_rho, vr_3D, psi, hpsi);

}

namespace detail {

/**
 * Compute the mapping from "wfc" g-grid to a target fft g-grid, e.g. the density g-grid
 *
 * @tparam MF_t - Type parameter for mean-field handler
 * @param mpi - mpi context
 * @param mf - Mean-field handler
 * @param mesh - Target FFT mesh, e.g. density g-grid
 * @return
 */
template<typename MF_t>
auto make_wfc_to_rho(utils::mpi_context_t<boost::mpi3::communicator, 
                                          boost::mpi3::shared_communicator> &mpi,
                     MF_t &mf,
                     nda::ArrayOfRank<1> auto const &mesh) {
  using arr_t = math::shm::shared_array<nda::array_view<long, 1>>;
  using boost::mpi3::communicator;
  using boost::mpi3::shared_communicator;
  if (mf.has_wfc_grid()) {
    auto wfc_g = mf.wfc_truncated_grid();
    long ngm = wfc_g->size();
    arr_t swfc_to_rho(std::addressof(mpi.comm), std::addressof(mpi.internode_comm),
                      std::addressof(mpi.node_comm), std::array<long, 1>{ngm});
    if (mpi.comm.root()) {
      grids::map_truncated_grid_to_fft_grid(*wfc_g, mesh, swfc_to_rho.local());
      mpi.internode_comm.broadcast_n(swfc_to_rho.local().data(), ngm, 0);
    } else if (mpi.node_comm.root()) {
      mpi.internode_comm.broadcast_n(swfc_to_rho.local().data(), ngm, 0);
    }
    mpi.node_comm.barrier();
    return swfc_to_rho;
  } else {
    arr_t swfc_to_rho(std::addressof(mpi.comm), std::addressof(mpi.internode_comm),
                      std::addressof(mpi.node_comm), std::array<long, 1>{1});
    return swfc_to_rho;
  }
}

} // namespace details
} // namespace hamilt

#endif
