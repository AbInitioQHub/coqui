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


#include "nda/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/embed_t.h"

namespace methods {
  void embed_t::dmft_embed(MBState &mb_state, simple_dyson &dyson,
                           iter_scf::iter_scf_t *iter_solver,
                           bool qp_approx_mbpt, bool corr_only) {
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    utils::check(std::filesystem::exists(filename),
                 "embed_t::dmft_embed: checkpoint file, {}, does not exist!", filename);
    if (!qp_approx_mbpt)
      dmft_embed_impl(mb_state, dyson, iter_solver, corr_only);
    else
      dmft_embed_qp_impl(mb_state, dyson, iter_solver);
  }

  void embed_t::dmft_embed_logic(long gw_iter, long weiss_f_iter, long embed_iter, std::string filename) {
    app_log(2, "Checking the dataset in the checkpoint file {}\n", filename);
    utils::check(weiss_f_iter!=-1, "embed_t::dmft_embed_impl: "
                                   "\"downfold_1e\" is missing in {} at DMFT iteration {}. "
                                   "This means that you are initializing an embedding calculation "
                                   "using an unexpected workflow...",
                 filename, embed_iter);
    if (embed_iter==-1) {
      app_log(2, "The dataset \"embed\" does not exist in {}, \n"
                 "indicating that this is the initial DMFT iteration with the following inputs: \n\n"
                 "  a) Sigma_non-local from \"scf/iter{}\"\n\n"
                 "  b) Sigma_dc from \"downfold_1e/iter{}\"\n\n"
                 "  c) Sigma_imp from \"downfold_1e/iter{}\"\n\n"
                 "All the results will be written to \"embed\"/iter{}\".\n",
              filename, gw_iter, weiss_f_iter, weiss_f_iter, weiss_f_iter);
    } else {
      utils::check(embed_iter<=weiss_f_iter,
                   "dmft_embed: iteration mismatch between \"embed\" ({}) and \"downfold_1e\" ({}) groups. "
                   "This means that you are initializing an embedding calculation "
                   "using an unexpected workflow...", embed_iter, weiss_f_iter);
      app_log(2, "We are at DMFT iteration {} with the following inputs: \n\n"
                 "  a) Sigma_non-local from \"scf/iter{}\"\n\n"
                 "  b) Sigma_dc from \"downfold_1e/iter{}\"\n\n"
                 "  c) Sigma_imp from \"downfold_1e/iter{}\"\n\n"
                 "All the results will be written to \"embed/iter{}\".\n",
              weiss_f_iter, gw_iter, weiss_f_iter, weiss_f_iter, weiss_f_iter);
      if (embed_iter==weiss_f_iter)
        app_log(2, "  [WARNING] \"embed/iter{}\" already exists, and the dataset will be overwritten. \n"
                   "             Please check if this is what you want!\n",
                   weiss_f_iter);
    }
  }

  void embed_t::dmft_embed_impl(MBState &mb_state, simple_dyson &dyson,
                                iter_scf::iter_scf_t *iter_solver,
                                bool corr_only) {
    using math::shm::make_shared_array;
    for (auto &v: {"EMBED_TOTAL", "EMBED_ALLOC",
                   "EMBED_UPFOLD", "EMBED_DYSON", "EMBED_FIND_MU",
                   "EMBED_ITERATIVE", "EMBED_READ", "EMBED_WRITE"}) {
      _Timer.add(v);
    }

    _Timer.start("EMBED_TOTAL");
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto &ft = *mb_state.ft;
    auto &proj = mb_state.proj_boson.value().proj_fermi();
    auto nImps = proj.nImps();
    auto nImpOrbs = proj.nImpOrbs();

    _Timer.start("EMBED_READ");
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    utils::check(weiss_f_iter == weiss_b_iter,
                 "embed_t::dmft_embed_impl: inconsistent downfolding iterations "
                 "for fermionic ({}) and bosonic ({}) Weiss fields in {}",
                 weiss_f_iter, weiss_b_iter, filename);
    long embed_out_iter = (embed_iter > 0) ? embed_iter+1 : 1;
    _Timer.stop("EMBED_READ");

    ft.metadata_log();
    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20dmft%20embed
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┌┬┐┌─┐┌┬┐  ┌─┐┌┬┐┌┐ ┌─┐┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │││││├┤  │   ├┤ │││├┴┐├┤  ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  ─┴┘┴ ┴└   ┴   └─┘┴ ┴└─┘└─┘─┴┘\n");
    app_log(1, "  - CoQui checkpoint file:                      {}\n", filename);
    app_log(1, "    Non-local self-energy for embedding ");
    app_log(1, "      HDF5 group:                              scf");
    app_log(1, "      Iteration:                               {}\n", gw_iter);
    if (!mb_state.has_local_selfenergies()) {
      app_log(1, "    Local self-energy corrections");
      app_log(1, "      HDF5 group:                              downfold_1e");
      app_log(1, "      Iteration:                               {}\n", weiss_f_iter);
    } else {
      app_log(1, "    Local self-energy corrections provided directly through MBState\n"
                 "    (no read from checkpoint file)\n");
    }
    app_log(1, "    Embedded solution output");
    app_log(1, "      HDF5 group:                              embed");
    app_log(1, "      Iteration:                               {}", embed_out_iter);
    app_log(1, "      Embed correlated self-energy only:       {}\n", corr_only);
    if (proj.C_file() != "")
      app_log(1, "  - Transformation matrices:                   {}", proj.C_file());
    app_log(1, "  - Number of impurities:                      {}", nImps);
    app_log(1, "  - Number of local orbitals per impurity:     {}", nImpOrbs);
    app_log(1, "  - Range of primary orbitals for local basis: [{}, {})\n", proj.W_rng()[0].first(), proj.W_rng()[0].last());
    _context->comm.barrier();

    _Timer.start("EMBED_ALLOC");
    // Initialize MBState
    mb_state.sF_skij.emplace(make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sDm_skij.emplace(make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    auto& sVhf_skij = mb_state.sF_skij.value();
    auto& sDm_skij = mb_state.sDm_skij.value();
    auto& sG_tskij = mb_state.sG_tskij.value();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    double mu;
    _Timer.stop("EMBED_ALLOC");

    _Timer.start("EMBED_READ");
    // Read MBPT results
    utils::check(chkpt::read_scf(_context->node_comm, sVhf_skij, sSigma_tskij, mu, mb_state.coqui_prefix) == gw_iter,
                 "embed_t::dmft_embed_impl: "
                 "Inconsistent gw iterations. gw_iter ({}) is not the last iteration in {}",
                 gw_iter, filename);
    // Read chemical potential from the previous embedding iteration (if it exists)
    if (embed_iter > 0) {
      if (_context->comm.root()) {
        h5::file file(filename, 'r');
        auto iter_grp = h5::group(file).open_group("embed/iter"+std::to_string(embed_iter));
        h5::read(iter_grp, "mu", mu);
      }
      _context->comm.broadcast_n(&mu, 1, 0);
    }

    // Check local self-energy corrections in MBState
    bool sigma_local_given = false;
    if (!mb_state.has_local_selfenergies()) {

      long nw = ft.nw_f();
      mb_state.Sigma_imp_wsIab.emplace(nda::array<ComplexType, 5>(nw, _MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Sigma_dc_wsIab.emplace(nda::array<ComplexType, 5>(nw, _MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vhf_imp_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vhf_dc_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      if (_context->comm.root()) {
        // 2) if the file does not contain them, we set them to empty arrays
        // Return false if not "all" the local corrections present in the checkpoint file,
        // --> the local corrections will be set to zero.
        sigma_local_given = chkpt::read_sigma_local(mb_state.Sigma_imp_wsIab.value(),
                                                    mb_state.Sigma_dc_wsIab.value(),
                                                    mb_state.Vhf_imp_sIab.value(),
                                                    mb_state.Vhf_dc_sIab.value(),
                                                    filename, weiss_f_iter);
      }
      _context->comm.broadcast_n(&sigma_local_given, 1, 0);
      _context->comm.broadcast_n(mb_state.Sigma_imp_wsIab.value().data(), mb_state.Sigma_imp_wsIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Sigma_dc_wsIab.value().data(), mb_state.Sigma_dc_wsIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vhf_imp_sIab.value().data(), mb_state.Vhf_imp_sIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vhf_dc_sIab.value().data(), mb_state.Vhf_dc_sIab.value().size(), 0);
      if (sigma_local_given)
        app_log(1, "Successfully found the local self-energy corrections in the checkpoint h5 {}.\n", filename);
    } else {
      sigma_local_given = true;
    }
    _Timer.stop("EMBED_READ");

    _Timer.start("EMBED_UPFOLD");
    // upfold and add corrections from active spaces
    if (!corr_only) add_Vhf_correction(mb_state);
    add_Sigma_dyn_correction(mb_state);

    if (_context->node_comm.root()) {
      hermitize_in_tau(sVhf_skij.local(), "Fock matrix");
      hermitize_in_tau(sSigma_tskij.local(), "dynamic self-energy");
    }
    _context->comm.barrier();
    _Timer.stop("EMBED_UPFOLD");

    _Timer.start("EMBED_ITERATIVE");
    // if embed_iter is -1 -> mix with the previous gw results
    // if embed_iter != -1 -> mix with embed_iter-1
    double Vhf_conv, Sigma_conv;
    if (iter_solver != nullptr) {
      std::tie(Vhf_conv, Sigma_conv) = solve_iterative(
          *_context, *iter_solver, (embed_iter==-1)? gw_iter+1 : embed_iter+1,
          mb_state.coqui_prefix, sVhf_skij, sSigma_tskij, &ft,
          (embed_iter==-1)? std::array<std::string, 3>{"scf", "F_skij", "Sigma_tskij"} :
                            std::array<std::string, 3>{"embed", "F_skij", "Sigma_tskij"});
    }
    _Timer.stop("EMBED_ITERATIVE");

    _Timer.start("EMBED_FIND_MU");
    // find chemical potential
    mu = update_mu(mu, dyson, *_MF, ft, sVhf_skij, sSigma_tskij);
    _Timer.stop("EMBED_FIND_MU");

    _Timer.start("EMBED_DYSON");
    // solve Dyson for G and update mu
    dyson.solve_dyson(sDm_skij, sG_tskij, sVhf_skij, sSigma_tskij, mu);
    _Timer.stop("EMBED_DYSON");

    auto k_weight = _MF->k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(sDm_skij, sVhf_skij, dyson.sH0_skij(), k_weight, false);
    auto e_corr = eval_corr_energy(_context->comm, ft, sG_tskij, sSigma_tskij, k_weight);
    double e_tot_new = e_1e + e_hf + e_corr;
    app_log(1, "\nEnergy contributions");
    app_log(1, "----------------------");
    app_log(1, "  non-interacting (H0):        {} a.u.", e_1e);
    app_log(1, "  Hartree-Fock:                {} a.u.", e_hf);
    app_log(1, "  correlation:                 {} a.u.", e_corr);
    app_log(1, "  total energy:                {} a.u.\n", e_tot_new);
    
    if (iter_solver != nullptr && (embed_iter == -1 || embed_iter >= 1)) {
      app_log(1, "abs max diff of Fock matrix:   {}", Vhf_conv);
      app_log(1, "abs max diff of self-energy:   {}\n", Sigma_conv);
    }

    _Timer.start("EMBED_WRITE");
    // dump output to checkpoint file
    if (_context->comm.root()) {
      h5::file file(filename, 'a');
      auto grp = h5::group(file);
      auto embed_grp = (grp.has_subgroup("embed"))?
                       grp.open_group("embed") : grp.create_group("embed");
      auto iter_grp = (embed_grp.has_subgroup("iter"+std::to_string(embed_out_iter)))?
                      embed_grp.open_group("iter"+std::to_string(embed_out_iter)) :
                      embed_grp.create_group("iter"+std::to_string(embed_out_iter));

      h5::h5_write(embed_grp, "final_iter", embed_out_iter);
      h5::h5_write(iter_grp, "mbpt_source_grp", "scf");
      h5::h5_write(iter_grp, "mbpt_source_iter", gw_iter);
      nda::h5_write(iter_grp, "F_skij", sVhf_skij.local(), false);
      nda::h5_write(iter_grp, "Sigma_tskij", sSigma_tskij.local(), false);
      nda::h5_write(iter_grp, "G_tskij", sG_tskij.local(), false);
      h5::h5_write(iter_grp, "mu", mu);
    }
    _context->comm.barrier();
    _Timer.stop("EMBED_WRITE");

    _Timer.stop("EMBED_TOTAL");
    print_dmft_embed_timers();
  }

  void embed_t::dmft_embed_qp_impl(MBState &mb_state, simple_dyson &dyson,
                                   iter_scf::iter_scf_t *iter_solver) {
    using math::shm::make_shared_array;
    for( auto& v: {"EMBED_TOTAL", "EMBED_ALLOC",
                   "EMBED_UPFOLD", "EMBED_DYSON", "EMBED_FIND_MU",
                   "EMBED_ITERATIVE", "EMBED_READ", "EMBED_WRITE"} ) {
      _Timer.add(v);
    }

    _Timer.start("EMBED_TOTAL");
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto ft = *mb_state.ft;
    auto& proj = mb_state.proj_boson.value().proj_fermi();
    auto nImps = proj.nImps();
    auto nImpOrbs = proj.nImpOrbs();

    _Timer.start("EMBED_READ");
    // read current iteration based on downfold_1e/final_iter
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    _Timer.stop("EMBED_READ");

    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20dmft%20embed
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┌┬┐┌─┐┌┬┐  ┌─┐┌┬┐┌┐ ┌─┐┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │││││├┤  │   ├┤ │││├┴┐├┤  ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  ─┴┘┴ ┴└   ┴   └─┘┴ ┴└─┘└─┘─┴┘\n");
    app_log(1, "(applying static approximation to the non-local self-energy)\n");
    app_log(1, "  - CoQuí check-point file:                      {}", filename);
    app_log(1, "  - Transformation matrices:                   {}", proj.C_file());
    app_log(1, "  - Number of impurities:                      {}", nImps);
    app_log(1, "  - Number of local orbitals per impurity:     {}", nImpOrbs);
    app_log(1, "  - Range of primary orbitals for local basis: [{}, {})", proj.W_rng()[0].first(), proj.W_rng()[0].last());
    app_log(1, "  - GW iteration:                              {}", gw_iter);
    app_log(1, "  - Downfold_1e iteration:                     {}", weiss_f_iter);
    app_log(1, "  - Embed iteration:                           {}\n", embed_iter);
    ft.metadata_log();
    dmft_embed_logic(gw_iter, weiss_f_iter, embed_iter, filename);

    _Timer.start("EMBED_ALLOC");
    // Initialize MBState
    mb_state.sF_skij.emplace(make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sDm_skij.emplace(make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    auto sVcorr_skij = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    auto& sVhf_skij = mb_state.sF_skij.value();
    auto& sDm_skij = mb_state.sDm_skij.value();
    auto& sG_tskij = mb_state.sG_tskij.value();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    double mu;
    _Timer.stop("EMBED_ALLOC");

    _Timer.start("EMBED_READ");
    // Read components of effective QPGW Hamiltonian from "scf/iter{gw_iter}"
    chkpt::read_qp_hamilt_components(sVhf_skij, sVcorr_skij, mu, filename, gw_iter);

    // Read chemical potential from the previous embedding iteration (if it exists)
    if (embed_iter > 0) {
      if (_context->comm.root()) {
        h5::file file(filename, 'r');
        auto iter_grp = h5::group(file).open_group("embed/iter"+std::to_string(embed_iter));
        h5::read(iter_grp, "mu", mu);
      }
      _context->comm.broadcast_n(&mu, 1, 0);
    }

    // Read Vhf_imp, Vhf_dc, Sigma_imp, and Vcorr_dc
    bool sigma_local_given = false;
    if (!mb_state.Sigma_imp_wsIab or !mb_state.Vcorr_dc_sIab) {
      // 1) if the sigma_imp and Vcorr_dc are not set, we read them from the file
      // 2) if the file does not contain them, we set them to empty arrays
      mb_state.Sigma_imp_wsIab.emplace(nda::array<ComplexType, 5>(ft.nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vcorr_dc_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vhf_imp_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vhf_dc_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      if (_context->comm.root()) {
        sigma_local_given = chkpt::read_sigma_local(mb_state.Sigma_imp_wsIab.value(),
                                                    mb_state.Vcorr_dc_sIab.value(),
                                                    mb_state.Vhf_imp_sIab.value(),
                                                    mb_state.Vhf_dc_sIab.value(),
                                                    filename, weiss_f_iter);
      }
      _context->comm.broadcast_n(&sigma_local_given, 1, 0);
      _context->comm.broadcast_n(mb_state.Sigma_imp_wsIab.value().data(), mb_state.Sigma_imp_wsIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vcorr_dc_sIab.value().data(), mb_state.Vcorr_dc_sIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vhf_imp_sIab.value().data(), mb_state.Vhf_imp_sIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vhf_dc_sIab.value().data(), mb_state.Vhf_dc_sIab.value().size(), 0);
      if (sigma_local_given)
        app_log(2, "Found local self-energy corrections in the checkpoint file: {}", filename);
    } else {
      app_log(2, "Found local self-energy corrections already set in MBState.");
      sigma_local_given = true;
    }
    _Timer.stop("EMBED_READ");

    _Timer.start("EMBED_UPFOLD");
    // upfold and add corrections from active spaces
    {
      app_log(2, "Add local corrections from \"downfold_1e/iter{}\"", weiss_f_iter);
      add_Vhf_correction(mb_state);
      add_Vcorr_correction(sVcorr_skij, mb_state);
      add_Sigma_dyn_correction(mb_state, false);
    }
    _context->comm.barrier();
    _Timer.stop("EMBED_UPFOLD");

    _Timer.start("EMBED_ITERATIVE");
    double Vhf_conv = -1;
    double Vcorr_conv = -1;
    double Sigma_conv = -1;
    if (embed_iter >= 1) {
      iter_solver->metadata_log();
      if (_context->node_comm.root()) {
        h5::file file(filename, 'r');
        h5::group grp(file);

        std::string grp_name = "embed/iter"+std::to_string(weiss_f_iter-1);
        utils::check(grp.has_subgroup(grp_name), "damping_impl: {} does not exist in {}.",
                     grp_name, filename);
        auto emb_grp = grp.open_group("embed");
        Vhf_conv = iter_solver->solve(sVhf_skij.local(), "Vhf_skij", emb_grp, weiss_f_iter);
        Vcorr_conv = iter_solver->solve(sVcorr_skij.local(), "Vcorr_skij", emb_grp, weiss_f_iter);
        Sigma_conv = iter_solver->solve(sSigma_tskij.local(), "Sigma_tskij", emb_grp, weiss_f_iter);
      }
      _context->node_comm.broadcast_n(&Vhf_conv, 1, 0);
      _context->node_comm.broadcast_n(&Vcorr_conv, 1, 0);
      _context->node_comm.broadcast_n(&Sigma_conv, 1, 0);
    }
    _Timer.stop("EMBED_ITERATIVE");

    _Timer.start("EMBED_FIND_MU");
    if (sVcorr_skij.node_comm()->root()) sVcorr_skij.local() += sVhf_skij.local();
    sVcorr_skij.communicator()->barrier();

    // find chemical potential
    mu = update_mu(mu, dyson, *_MF, ft, sVcorr_skij, sSigma_tskij);
    _Timer.stop("EMBED_FIND_MU");

    _Timer.start("EMBED_DYSON");
    // solve Dyson for G and update mu
    dyson.solve_dyson(sDm_skij, sG_tskij, sVcorr_skij, sSigma_tskij, mu);

    if (sVcorr_skij.node_comm()->root()) sVcorr_skij.local() -= sVhf_skij.local();
    sVcorr_skij.communicator()->barrier();
    _Timer.stop("EMBED_DYSON");

    auto k_weight = _MF->k_weight();
    auto [e_1e, e_hf] = eval_hf_energy(sDm_skij, sVhf_skij, dyson.sH0_skij(), k_weight, false);
    auto e_corr = eval_corr_energy(_context->comm, ft, sG_tskij, sSigma_tskij, k_weight);
    double e_tot_new = e_1e + e_hf + e_corr;
    app_log(2, "\nEnergy contributions");
    app_log(2, "---------------------");
    app_log(2, "  non-interacting (H0):        {} a.u.", e_1e);
    app_log(2, "  Hartree-Fock:                {} a.u.", e_hf);
    app_log(2, "  correlation:                 {} a.u.", e_corr);
    app_log(2, "  total energy:                {} a.u.\n", e_tot_new);
    if (embed_iter >= 1) {
      app_log(2, "abs max diff of Vhf matrix:    {}", Vhf_conv);
      app_log(2, "abs max diff of Vcorr matrix:  {}", Vcorr_conv);
      app_log(2, "abs max diff of self-energy:   {}\n", Sigma_conv);
    }

    _Timer.start("EMBED_WRITE");
    // dump output to checkpoint file
    if (_context->comm.root()) {
      h5::file file(filename, 'a');
      auto grp = h5::group(file);
      auto embed_grp = (grp.has_subgroup("embed"))?
                       grp.open_group("embed") : grp.create_group("embed");
      auto iter_grp = (embed_grp.has_subgroup("iter"+std::to_string(weiss_f_iter)))?
                      embed_grp.open_group("iter"+std::to_string(weiss_f_iter)) :
                      embed_grp.create_group("iter"+std::to_string(weiss_f_iter));

      h5::h5_write(embed_grp, "final_iter", weiss_f_iter);
      nda::h5_write(iter_grp, "Vhf_skij", sVhf_skij.local(), false);
      nda::h5_write(iter_grp, "Vcorr_skij", sVcorr_skij.local(), false);

      sVhf_skij.local() += sVcorr_skij.local();
      nda::h5_write(iter_grp, "F_skij", sVhf_skij.local(), false);
      sVhf_skij.local() -= sVcorr_skij.local();

      nda::h5_write(iter_grp, "Sigma_tskij", sSigma_tskij.local(), false);
      nda::h5_write(iter_grp, "G_tskij", sG_tskij.local(), false);
      h5::h5_write(iter_grp, "mu", mu);
    }
    _context->comm.barrier();
    _Timer.stop("EMBED_WRITE");
    _Timer.stop("EMBED_TOTAL");
    print_dmft_embed_timers();
  }

  void embed_t::add_Vhf_correction(MBState &mb_state) {

    auto& sVhf_skij = mb_state.sF_skij.value();
    auto& proj = mb_state.proj_boson.value().proj_fermi();

    auto sVhf_correction_upfold = sVhf_skij;
    sVhf_correction_upfold.set_zero();

    nda::array<ComplexType, 4> Vhf_correction = mb_state.Vhf_imp_sIab.value() - mb_state.Vhf_dc_sIab.value();
    proj.upfold(sVhf_correction_upfold, Vhf_correction);

    if (sVhf_skij.node_comm()->root()) sVhf_skij.local() += sVhf_correction_upfold.local();
    sVhf_skij.communicator()->barrier();
  }

  void embed_t::add_Sigma_dyn_correction(MBState &mb_state, bool subtract_dc) {

    auto& proj = mb_state.proj_boson.value().proj_fermi();
    auto nImps = proj.nImps();
    auto nImpOrbs = proj.nImpOrbs();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();

    auto sSigma_correction_upfold = sSigma_tskij;
    sSigma_correction_upfold.set_zero();

    nda::array<ComplexType, 5> Sigma_imp_tsIab(mb_state.ft->nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    mb_state.ft->w_to_tau(mb_state.Sigma_imp_wsIab.value(), Sigma_imp_tsIab, imag_axes_ft::fermion);
    mb_state.ft->check_leakage(Sigma_imp_tsIab, imag_axes_ft::fermion, sSigma_tskij.communicator(), "impurity self-energy");

    if (subtract_dc) {
      nda::array<ComplexType, 5> Sigma_dc_tsIab(mb_state.ft->nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
      mb_state.ft->w_to_tau(mb_state.Sigma_dc_wsIab.value(), Sigma_dc_tsIab, imag_axes_ft::fermion);
      mb_state.ft->check_leakage(Sigma_dc_tsIab, imag_axes_ft::fermion, sSigma_tskij.communicator(), "DC self-energy");
      Sigma_imp_tsIab -= Sigma_dc_tsIab;
    }

    proj.upfold(sSigma_correction_upfold, Sigma_imp_tsIab);
    if (sSigma_tskij.node_comm()->root()) sSigma_tskij.local() += sSigma_correction_upfold.local();
    sSigma_tskij.communicator()->barrier();
  }

  template<nda::ArrayOfRank<4> Array_base_t>
  void embed_t::add_Vcorr_correction(sArray_t<Array_base_t> &sVcorr_skij,
                                     MBState &mb_state) {

    auto& proj = mb_state.proj_boson.value().proj_fermi();
    auto sVcorr_skij_upfold = sVcorr_skij;
    sVcorr_skij_upfold.set_zero();

    proj.upfold(sVcorr_skij_upfold, mb_state.Vcorr_dc_sIab.value());
    if (sVcorr_skij.node_comm()->root()) sVcorr_skij.local() -= sVcorr_skij_upfold.local();
    sVcorr_skij.communicator()->barrier();
  }

} // methods
