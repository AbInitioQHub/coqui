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


#ifndef COQUI_UNFOLD_BZ_H
#define COQUI_UNFOLD_BZ_H

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"

/** Utilities for Brillioun zone unfolding **/

namespace methods {
  using mpi_context_t = utils::mpi_context_t<>;

  /**
   * Unfold mbpt solutions of bdft from IBZ to the full BZ
   * @param context - [INPUT]
   * @param mf - [INPUT] mean-field instance for all the metadata of the system
   * @param scf_output - [INPUT/OUTPUT] h5 file where bdft solutions stored.
   *                     The unfolded solution will be written into the same file.
   */
  void unfold_bz(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string scf_output);
  /**
   * Unfolding BZ implementation details for dyson-type solutions
   * @param iter - [INPUT] which scf iteration for BZ unfolding
   */
  void unfold_dyson_solution(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, long iter);
  /**
   * Unfolding BZ implementation details for quasiparticle-type solutions
   * @param iter - [INPUT] which scf iteration for BZ unfolding
   */
  void unfold_qp_solution(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, long iter);

  /**
   * BZ unfolding for dynamic Hamiltonian, e.g. Green's function and self-energy
   * @param dataset - [INPUT] target dataset in the bdft h5 file
   * @return unfolding dynamic Hamiltonian in a shared-memory array: (t, s, k, i, j)
   */
  auto unfold_dynamic_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, std::string dataset)
  -> math::shm::shared_array<nda::array_view<ComplexType,5>>;
  /**
   * BZ unfolding for quasiparticle energies
   * @return unfolding qp energies in a shared-memory array: (s, k, i)
   */
  auto unfold_qp_energy(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, long iter)
  -> math::shm::shared_array<nda::array_view<ComplexType,3>>;
  /**
   * BZ unfolding for static Hamiltonian
   * @param dataset - [INPUT] target dataset in the bdft h5 file
   * @param include_H0 - [INPUT] whether to include the non-interacting Hamiltonian in the output
   * @return unfolding static Hamiltonian in a shared-memory array: (s, k, i, j)
   */
  auto unfold_1e_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf,
                             std::string filename, std::string dataset, bool include_H0=false)
  -> math::shm::shared_array<nda::array_view<ComplexType,4>>;

  /**
   * Unfold the wavefunctions of a mf instance w/ symmetry and overwrite the ones of
   * a mf instance w/o symmetry.
   *
   * Restrictions on mf_sym and mf_nosym:
   * 1. the full BZ and the number of spins should be consistent.
   * 2. mf_nosym.nbnd <= mf_sum.nbnd
   *
   * Note that k-points in mf_sym and mf_nosym are allowed to have shifts
   *
   * @param mf_sym   - [INPUT] qe mf system with space-group symmetries
   * @param mf_nosym - [INPUT/OUTPUT] qe mf system without space-group symmetries
   */
  void unfold_wfc(mf::MF& mf_sym, mf::MF& mf_nosym);
  void check_is_G(double ni_d, double nj_d, double nk_d);

} // methods


#endif //COQUI_UNFOLD_BZ_H
