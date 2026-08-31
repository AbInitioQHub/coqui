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


#include "mean_field/MF.hpp"
#include "methods/tools/chkpt_utils.h"
#include "mb_state.hpp"

namespace methods {

  MBState::MBState(std::shared_ptr<mpi_context_t> mpi_in, imag_axes_ft::IAFT &ft_in,
                   std::string prefix, bool restart_from_checkpoint):
  mpi(std::move(mpi_in)), ft(std::addressof(ft_in)), coqui_prefix(prefix) {
    if (restart_from_checkpoint) {
      app_log(1, "MBState: Restarting from checkpoint is not implemented yet.");
    }
  }

  MBState::MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
                   std::shared_ptr<mf::MF> &mf, std::string C_file, bool translate_home_cell,
                   bool restart_from_checkpoint):
    mpi(mf->mpi()), ft(std::addressof(ft_in)), coqui_prefix(prefix),
    proj_boson(std::in_place, *mf, C_file, translate_home_cell) {
      if (restart_from_checkpoint) {
        app_log(1, "MBState: Restarting from checkpoint is not implemented yet.");
      }
  }

  MBState::MBState(imag_axes_ft::IAFT &ft_in, std::string prefix,
                   std::shared_ptr<mf::MF> &mf, const nda::array<ComplexType, 5> &C_ksIai,
                   const nda::array<long, 3> &band_window, const nda::array<RealType, 2> &kpts_crys,
                   bool translate_home_cell, bool restart_from_checkpoint):
    mpi(mf->mpi()), ft(std::addressof(ft_in)), coqui_prefix(prefix),
    proj_boson(std::in_place, *mf, C_ksIai, band_window, kpts_crys, translate_home_cell) {
      if (restart_from_checkpoint) {
        app_log(1, "MBState: Restarting from checkpoint is not implemented yet.");
      }
  }

  void MBState::set_zero_local_polarizabilities() {
    using math::shm::make_shared_array;

    utils::check(proj_boson.has_value(),
                 "MBState::read_local_polarizabilities: proj_boson is not initialized.");

    long nw = ft->nw_b();
    long nw_half = (nw%2==0)? nw/2 : nw/2+1;
    long nImpOrbs = proj_boson.value().nImpOrbs();
    sPi_imp_wabcd.emplace(make_shared_array<nda::array_view<ComplexType, 5>>(*mpi, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs}));
    sPi_dc_wabcd.emplace(make_shared_array<nda::array_view<ComplexType, 5>>(*mpi, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs}));
  }

  bool MBState::read_local_polarizabilities(long weiss_b_iter) {

    set_zero_local_polarizabilities();

    // 1) Read pi_imp and pi_dc from chkpt file
    // 2) if the file does not contain them, we set them to std::nullopt
    bool pi_local_found = chkpt::read_pi_local(sPi_imp_wabcd.value(), sPi_dc_wabcd.value(), coqui_prefix+".mbpt.h5", weiss_b_iter);

    if (!pi_local_found) {
      sPi_imp_wabcd.reset();
      sPi_dc_wabcd.reset();
    } else {
      df_2e_iter = weiss_b_iter;
    }
    mpi->comm.barrier();
    return pi_local_found;
  }

  void MBState::set_local_polarizabilities(std::map<std::string, nda::array<ComplexType, 5>> local_polarizabilities) {

    set_zero_local_polarizabilities();

    if (mpi->node_comm.root()) {
      auto Pi_imp = sPi_imp_wabcd.value().local();
      auto Pi_dc = sPi_dc_wabcd.value().local();
      utils::check(Pi_imp.shape() == local_polarizabilities.at("imp").shape(),
                   "MBState::set_local_polarizabilities: Incorrect dimension for the provided pi_imp.");
      utils::check(Pi_dc.shape() == local_polarizabilities.at("dc").shape(),
                   "MBState::set_local_polarizabilities: Incorrect dimension for the provided pi_dc.");
      Pi_imp = local_polarizabilities.at("imp");
      Pi_dc = local_polarizabilities.at("dc");
    }
    mpi->comm.barrier();
  }

  void MBState::set_local_hf_potentials(std::map<std::string, nda::array<ComplexType, 4>> local_hf_potentials) {
    utils::check(proj_boson.has_value(),
                 "MBState::set_local_hf_potentials: proj_boson is not initialized.");
    Vhf_imp_sIab.emplace(local_hf_potentials.at("imp"));
    Vhf_dc_sIab.emplace(local_hf_potentials.at("dc"));
    utils::check(Vhf_imp_sIab.value().shape(2) == proj_boson.value().nImpOrbs() and
                 Vhf_imp_sIab.value().shape(3) == proj_boson.value().nImpOrbs(),
                 "MBState::set_local_hf_potentials: Incorrect dimension for the provided Vhf_imp_sIab.");
    utils::check(Vhf_dc_sIab.value().shape() == Vhf_imp_sIab.value().shape(),
                 "MBState::set_local_hf_potentials: Incorrect dimension for the provided Vhf_dc_sIab.");
  }

  void MBState::set_local_selfenergies(std::map<std::string, nda::array<ComplexType, 5>> local_selfenergies) {
    utils::check(proj_boson.has_value(),
                 "MBState::set_local_selfenergies: proj_boson is not initialized.");
    Sigma_imp_wsIab.emplace(local_selfenergies.at("imp"));
    Sigma_dc_wsIab.emplace(local_selfenergies.at("dc"));
    utils::check(Sigma_imp_wsIab.value().shape(3) == proj_boson.value().nImpOrbs() and
                 Sigma_imp_wsIab.value().shape(4) == proj_boson.value().nImpOrbs(),
                 "MBState::set_local_selfenergies: Incorrect dimension for the provided Sigma_imp_wsIab ({}, {}, {}, {} {}).",
                 Sigma_imp_wsIab.value().shape(0), Sigma_imp_wsIab.value().shape(1), Sigma_imp_wsIab.value().shape(2),
                 Sigma_imp_wsIab.value().shape(3), Sigma_imp_wsIab.value().shape(4));
    utils::check(Sigma_imp_wsIab.value().shape(0) == ft->nw_f(),
                 "MBState::set_local_selfenergies: Incorrect dimension for the provided Sigma_imp_wsIab:"
                 "Sigma_imp_wsIab.shape(0) = {} != ft->nw_f() = {}.",
                 Sigma_imp_wsIab.value().shape(0), ft->nw_f());
    utils::check(Sigma_dc_wsIab.value().shape() == Sigma_imp_wsIab.value().shape(),
                 "MBState::set_local_selfenergies: Incorrect dimension for the provided Sigma_dc_wsIab.");
  }

  bool MBState::has_local_selfenergies() {
    return Sigma_imp_wsIab.has_value() and Sigma_dc_wsIab.has_value() and Vhf_imp_sIab.has_value() and Vhf_dc_sIab.has_value();
  }



  /** Instantiation of public template **/

} // methods
