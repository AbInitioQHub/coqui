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

#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/dc_utilities.hpp"
#include "methods/embedding/embed_t.h"
#include "methods/ERI/thc_reader_t.hpp"
#include "mean_field/model_hamiltonian/model_utils.hpp"

namespace methods {

  auto embed_t::downfold_gloc(MBState& mb_state, bool force_real,
                              std::string g_grp, long g_iter)
  -> nda::array<ComplexType, 5> {
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┌┐┌┌─┐  ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║  │ ││││├┤───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┘└┘└─┘  └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    utils::check(std::filesystem::exists(filename),
                 "embed_t::downfolding: checkpoint file, {}, does not exist!", filename);

    using math::shm::make_shared_array;

    for (auto &v: {"DF_TOTAL", "DF_ALLOC",
                   "DF_DC", "DF_DOWNFOLD", "DF_UPFOLD",
                   "DF_FIND_MU", "DF_DYSON",
                   "DF_G_WEISS", "DF_READ", "DF_WRITE"}) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    auto mpi = mb_state.mpi;
    auto proj = mb_state.proj_boson.value().proj_fermi();
    auto ft = mb_state.ft;

    app_log(2, "Checking the dataset in the coqui checkpoint file...\n");
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    if (g_grp == "scf") {
      g_iter = (g_iter == -1)? gw_iter : g_iter;
      utils::check(g_iter <= gw_iter and g_iter >= 0,
                   "downfold_1e logic fail: the input dataset {}/iter{} does not exist!",
                   g_grp, g_iter);
    } else if (g_grp == "embed") {
      g_iter = (g_iter == -1)? embed_iter : g_iter;
      utils::check(g_iter <= embed_iter and g_iter > 0,
                   "downfold_1e logic fail: the input dataset {}/iter{} does not exist!",
                   g_grp, g_iter);
    } else
      utils::check(false, "downfold_1e logic fail: the input dataset {} does not exist!", g_grp);

    app_log(1, "  - CoQui checkpoint file                     = {}", filename);
    app_log(1, "  - Input Green's function ");
    app_log(1, "      HDF5 group                              = {}", g_grp);
    app_log(1, "      Iteration                               = {}", g_iter);
    if (proj.C_file() != "")
      app_log(1, "  - Transformation matrices                   = {}", proj.C_file());
    app_log(1, "  - Force real local Hamiltonian              = {}", force_real);
    app_log(1, "  - Number of impurities                      = {}", proj.nImps());
    app_log(1, "  - Number of local orbitals per impurity     = {}", proj.nImpOrbs());
    app_log(1, "  - Range of primary orbitals for local basis = [{}, {})\n",
            proj.W_rng()[0].first(), proj.W_rng()[0].last());
    ft->metadata_log();
    mpi->comm.barrier();

    // get Gloc
    _Timer.start("DF_READ");
    mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, filename, g_iter, g_grp));
    auto &sG_tskij = mb_state.sG_tskij.value();
    mpi->comm.barrier();
    _Timer.stop("DF_READ");

    _Timer.start("DF_DOWNFOLD");
    auto Gloc_tsIab = (force_real)? proj.downfold_loc<true>(sG_tskij, "Gloc") : proj.downfold_loc<false>(sG_tskij, "Gloc");
    ft->check_leakage(Gloc_tsIab, imag_axes_ft::fermion, std::addressof(mpi->comm), "Local Green's function");
    _Timer.stop("DF_DOWNFOLD");

    _Timer.stop("DF_TOTAL");
    print_downfold_mb_timers();

    return Gloc_tsIab;
  }

  void embed_t::downfolding(MBState &mb_state, ptree const& pt,
                            qp_params_t *qp_params, std::string format_type) {

    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    utils::check(std::filesystem::exists(filename),
                 "embed_t::downfolding: checkpoint file, {}, does not exist!", filename);

    std::string err = std::string("embed_t::downfolding: Incorrect input - ");
    auto update_dc = io::get_value_with_default<bool>(pt,"update_dc",true);
    auto dc_type = io::get_value<std::string>(
        pt, "dc_type", err+"dc_type. Valid types are \"hartree\", \"hf\", \"gw\", \"gw_dynamic_u\" and \"gw_mix_u\".");
    io::tolower(dc_type);

    if (qp_params!=nullptr) {
      utils::check(format_type == "default" or update_dc, "embed_t::downfolding: format_type!=default requires update_dc.");
      downfold_mb_solution_qp_impl(mb_state, *qp_params, update_dc, dc_type,
                                   io::get_value_with_default<bool>(pt, "force_real", true),
                                   format_type);
    } else {
      utils::check(format_type == "default", "embed_t::downfolding: qp_selfenergy=false requires format_type = default ");

      // g_weiss_type: Valid types are "dmft" and "gloc"
      auto g_weiss_type = io::get_value_with_default<std::string>(pt, "g_weiss_type", "dmft");
      io::tolower(g_weiss_type);

      std::array<double, 2> mixing{io::get_value_with_default<double>(pt,"dc_sigma_mixing",1.0),
                                   io::get_value_with_default<double>(pt,"g_weiss_mixing",1.0)};

      auto g_k_grp = io::get_value_with_default<std::string>(pt,"g_k_input", "");
      io::tolower(g_k_grp);
      if (g_k_grp=="mf") g_k_grp = "scf";
      auto g_k_iter = io::get_value_with_default<long>(pt, "g_k_input_iter", -1);

      downfold_mb_solution_impl(mb_state, update_dc, dc_type,
                                io::get_value_with_default<bool>(pt, "force_real", true),
                                g_k_grp, g_k_iter,
                                mixing, g_weiss_type);
    }
  }

  template<THC_ERI thc_t>
  void embed_t::hf_downfolding(std::string outdir, std::string prefix,
                      thc_t& eri, imag_axes_ft::IAFT &ft,
                      bool force_real, std::string hf_div_treatment) {

    prefix = outdir + "/" + prefix;
    downfold_hf_impl(prefix, eri, ft, force_real, hf_div_treatment);

  }

  void embed_t::downfold_hf_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                                  std::string filename) {
    app_log(2, "  Checking the dataset in the checkpoint file {}...\n", filename);
    utils::check(gw_iter == 0, "downfold_hf_impl: The iteration of scf != 0. \n"
                               "                  This implies the mean-field downfolding is \n"
                               "                  not set up properly. Please check! ");
    utils::check(weiss_b_iter == 1, "downfold_hf_impl: The iteration of downfolded ERIs != 1. \n"
                                    "                  This implies the mean-field downfolding is \n"
                                    "                  not set up properly. Please check! ");
    utils::check(weiss_f_iter == -1, "downfold_hf_impl: The iteration of one-body downfolding != -1. \n"
                                     "                  This implies the mean-field downfolding is \n"
                                     "                  not set up properly. Please check! ");
    utils::check(embed_iter == -1, "downfold_hf_impl: The iteration of embedding != -1. \n"
                                   "                  This implies the mean-field downfolding is \n"
                                   "                  not set up properly. Please check! ");
  }

  auto embed_t::gw_edmft_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                                  std::string filename, bool update_dc)
    -> std::tuple<long, std::string> {
    app_log(2, "Checking the dataset in the CoQuí checkpoint file {}\n", filename);
    utils::check(weiss_b_iter>0, "embed_t::gw_edmft_logic: weiss_b_iter <= 0, indicating "
                                 "no effective Coulomb interactions found in {}. "
                                 "Please run \"downfold_2e\" first. ", filename);
    long dc_iter = (embed_iter!=-1)? embed_iter : (gw_iter>0)? gw_iter-1 : gw_iter;
    std::string dc_src_grp = (embed_iter!=-1)? "embed" : "scf";
    if (embed_iter==-1) {
      utils::check(update_dc, "embed_t::gw_edmft_logic: embed_iter==-1 while update_dc=False. "
                              "You are trying to read dc self-energy from the previous iteration that "
                              "does not exist! ");
      app_log(2, "The dataset \"embed\" does not exist in {}, \n"
                 "indicating that this is the initial DMFT iteration starting from \n"
                 "the weakly correlated solution in \"scf/iter{}\". We will proceed \n"
                 "with the following inputs: \n\n"
                 "  a) Gloc for Sigma_dc from \"scf/iter{}\"\n\n"
                 "  b) Assume Sigma_imp = Sigma_dc\n\n"
                 "  c) Effective screened interactions from \"downfold_2e/iter{}\"\n\n"
                 "All the results will be written to \"downfold_1e/iter{}\".\n",
              filename, gw_iter, dc_iter, weiss_b_iter, gw_iter);
      if (weiss_f_iter!=-1)
        app_log(2, "  [WARNING] \"downfold_1e\" exists before the first DMFT iteration. \n"
                   "             The existing dataset will be overwritten. \n"
                   "             Please check if this is what you want!\n");
    } else {
      if (weiss_f_iter!=embed_iter) {
        utils::check(false, "embed_t::gw_edmft_logic: "
                            "\"downfold_1e/final_iter\" ({}) is mismatch with in \"embed/final_iter\" ({}). "
                            "This means that you are initializing an embedding calculation "
                            "using an unexpected workflow...",
                     weiss_f_iter, embed_iter);
      } else {
        app_log(2, "We are at DMFT iteration {} with the following inputs: \n", embed_iter+1);
        if (update_dc)
          app_log(2, "  a) Gloc for Sigma_dc from \"embed/iter{}\"\n", dc_iter);
        else
          app_log(2, "  a) Sigma_dc from \"downfold_1e/iter{}\"\n", weiss_f_iter);
        app_log(2, "  b) Sigma_imp from \"downfold_1e/iter{}\"\n\n"
                   "  c) Effective screened interactions from \"downfold_2e/iter{}\"\n\n"
                   "All the results will be written to \"downfold_1e/iter{}\".\n",
                weiss_f_iter, weiss_b_iter, weiss_f_iter+1);
      }
    }
    return std::tuple(dc_iter, dc_src_grp);
  }


  long embed_t::downfold_1e_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                                  std::string filename, bool update_dc) {
    app_log(2, "Checking the dataset in the checkpoint file {}\n", filename);
    utils::check(weiss_b_iter>0, "embed_t::downfold_1e_logic: weiss_b_iter <= 0, indicating "
                                 "no effective Coulomb interactions found in {}. Please run "
                                 "\"downfold_2e\" first. ", filename);
    long dc_iter = (embed_iter!=-1)? embed_iter : (gw_iter>0)? gw_iter-1 : gw_iter;
    if (embed_iter==-1) {
      utils::check(update_dc, "embed_t::downfold_1e_logic: embed_iter==-1 while update_dc=False. "
                              "You are trying to read dc self-energy from the previous iteration that "
                              "does not exist! ");
      app_log(2, "The dataset \"embed\" does not exist in {}, \n"
                 "indicating that this is the initial DMFT iteration starting from \n"
                 "the weakly correlated solution in \"scf/iter{}\". We will proceed \n"
                 "with the following inputs: \n\n"
                 "  a) Sigma_non-local from \"scf/iter{}\"\n\n"
                 "  b) Gloc for Sigma_dc from \"scf/iter{}\"\n\n"
                 "  c) Assume Sigma_imp = Sigma_dc\n\n"
                 "  d) Effective screened interactions from \"downfold_2e/iter{}\"\n\n"
                 "All the results will be written to \"downfold_1e/iter{}\".\n",
              filename, gw_iter, gw_iter, dc_iter, weiss_b_iter, gw_iter);
      if (weiss_f_iter!=-1)
        app_log(2, "  [WARNING] \"downfold_1e\" exists before the first DMFT iteration. \n"
                   "             The existing dataset will be overwritten. \n"
                   "             Please check if this is what you want!\n");
    } else {
      if (weiss_f_iter!=embed_iter) {
        utils::check(false, "embed_t::downfold_mb_solution_impl: "
                            "\"downfold_1e/final_iter\" ({}) is mismatch with in \"embed/final_iter\" ({}). "
                            "This means that you are initializing an embedding calculation "
                            "using an unexpected workflow...",
                     weiss_f_iter, embed_iter);
      } else {
        app_log(2, "We are at DMFT iteration {} with the following inputs: \n\n"
                   "  a) Sigma_gw from \"scf/iter{}\"\n", embed_iter+1, gw_iter);
        if (update_dc)
          app_log(2, "  b) Gloc for Sigma_dc from \"embed/iter{}\"\n", dc_iter);
        else
          app_log(2, "  b) Sigma_dc from \"downfold_1e/iter{}\"\n", weiss_f_iter);
        app_log(2, "  c) Sigma_imp from \"downfold_1e/iter{}\"\n\n"
                   "  d) Effective screened interactions from \"downfold_2e/iter{}\"\n\n"
                   "All the results will be written to \"downfold_1e/iter{}\".\n",
                weiss_f_iter, weiss_b_iter, weiss_f_iter+1);
      }
    }
    return dc_iter;
  }

  // FIXME This function requires "HDF5_USE_FILE_LOCKING=FALSE".
  // This function is responsible only for
  //   1) Read lattice Gk and downfold it to Gloc
  //   2) (optional) Calculate the dc self-energy
  //   3) Read impurity self-energy and evaluate delta using Gloc from step 1
  //   4) (optional) downfold various quantities to h5
  void embed_t::downfold_mb_solution_impl(
      MBState &mb_state, bool update_dc, std::string dc_type,
      bool force_real, std::string g_k_grp, long g_k_iter,
      std::array<double, 2> mixing,
      std::string g_weiss_type) {
    using math::shm::make_shared_array;

    std::vector<std::string> accept_dc_type = {"hf", "gw", "gw_dynamic_u",
                                               "gw_edmft", "gw_edmft_density"};
    utils::check(std::find(accept_dc_type.begin(), accept_dc_type.end(), dc_type) != accept_dc_type.end(),
                 "embed_t::downfold_mb_solution_impl: Invalid dc_type: {}. ", dc_type);

    for (auto &v: {"DF_TOTAL", "DF_ALLOC",
                   "DF_DC", "DF_DOWNFOLD", "DF_UPFOLD",
                   "DF_FIND_MU", "DF_DYSON",
                   "DF_G_WEISS", "DF_READ", "DF_WRITE"}) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    auto mpi  = mb_state.mpi;
    auto proj = mb_state.proj_boson.value().proj_fermi();
    auto ft   = mb_state.ft;
    auto nImps = proj.nImps();
    auto nImpOrbs = proj.nImpOrbs();

    _Timer.start("DF_READ");
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    auto dyson = simple_dyson(_MF, ft, mb_state.coqui_prefix);
    mpi->comm.barrier();
    _Timer.stop("DF_READ");

    utils::check(weiss_b_iter>0, "embed_t::gw_edmft_logic: weiss_b_iter <= 0, indicating "
                                 "no effective Coulomb interactions found in {}. "
                                 "Please run \"downfold_2e\" first. ", filename);

    if (g_k_grp=="") {
      // CNY: Here we change the default logic compared to the old implementation.
      // Old default logic: g_k_iter = (embed_iter != -1) ? embed_iter : (gw_iter > 0) ? gw_iter - 1 : gw_iter;
      g_k_grp = (embed_iter != -1) ? "embed" : "scf";
      g_k_iter = (embed_iter != -1) ? embed_iter : gw_iter;
    } else if (g_k_iter == -1) {
      g_k_iter = (g_k_grp == "embed")? embed_iter : gw_iter;
    }

    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┌┐┌┌─┐  ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║  │ ││││├┤───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┘└┘└─┘  └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  QoQuí checkpoint file:                     {}", filename);
    app_log(1, "    - Input Green's function");
    app_log(1, "      HDF5 group:                            {}", g_k_grp);
    app_log(1, "      iteration:                             {}", g_k_iter);
    app_log(1, "    - Output HDF5 group:                     downfold_1e/iter{}\n", g_k_iter+1);

    if (update_dc) {
      app_log(1, "  Double counting self-energy:");
      app_log(1, "    - Type:                                  {}", dc_type);
      app_log(1, "    - Input Green's function");
      app_log(1, "      HFD5 group:                            {}", g_k_grp);
      app_log(1, "      iteration:                             {}", g_k_iter);
      app_log(1, "    - Input interactions");
      app_log(1, "      HDF5 group:                            downfold_2e");
      app_log(1, "      iteration:                             {}", weiss_b_iter);
      app_log(1, "    - Mixing for the current iteration:      {}\n", mixing[0]);
    }

    app_log(1, "  Transformation matrices:                   {}", proj.C_file());
    app_log(1, "  Number of impurities:                      {}", proj.nImps());
    app_log(1, "  Number of local orbitals per impurity:     {}", proj.nImpOrbs());
    app_log(1, "  Range of primary orbitals for local basis: [{}, {})\n",
            proj.W_rng()[0].first(), proj.W_rng()[0].last());

    app_log(1, "  - force real local Hamiltonian:              {}", force_real);
    app_log(1, "  - g_weiss_type:                              {}\n", g_weiss_type);
    ft->metadata_log();
    mpi->comm.barrier();

    // get Gloc
    _Timer.start("DF_ALLOC");
    nda::array<ComplexType, 5> Gloc_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 4> Vhf_loc_sIab(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 5> Sigma_loc_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 4> Vhf_dc_sIab(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 5> Sigma_dc_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 5> g_weiss_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    double mu;
    {

      mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
          *mpi, {ft->nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
      mb_state.sF_skij.emplace(make_shared_array<Array_view_4D_t>(
          *mpi, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
      mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(
          *mpi, {ft->nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
      auto& sG_tskij = mb_state.sG_tskij.value();
      auto& sVhf_skij = mb_state.sF_skij.value();
      auto& sSigma_tskij = mb_state.sSigma_tskij.value();
      _Timer.stop("DF_ALLOC");

      _Timer.start("DF_READ");
      utils::check(chkpt::read_scf(mpi->node_comm, sVhf_skij, sSigma_tskij,
                                   mu, mb_state.coqui_prefix, g_k_grp, g_k_iter) == g_k_iter,
                   "embed_t::downfold_mb_solution_impl: "
                   "Inconsistent iterations - g_k_iter ({}) is not read in {} group. This should not happen!",
                   g_k_iter, g_k_grp);
      mpi->comm.barrier();

      h5::file file(filename, 'r');
      auto gh5 = h5::group(file);
      auto df_2e_grp = gh5.open_group("downfold_2e");
      auto iter_2e_grp = df_2e_grp.open_group("iter" + std::to_string(weiss_b_iter));
      auto iter_grp = gh5.open_group(g_k_grp).open_group("iter" + std::to_string(g_k_iter));

      if (iter_grp.has_dataset("G_tskij")) {
        if (mpi->node_comm.root()) {
          auto G_tskij = sG_tskij.local();
          nda::h5_read(iter_grp, "G_tskij", G_tskij);
        }
      } else
        compute_G_from_mf(iter_grp, *ft, sG_tskij);
      mpi->comm.barrier();
      _Timer.stop("DF_READ");

      _Timer.start("DF_DOWNFOLD");
      auto Gloc_tsIab = (force_real)? proj.downfold_loc<true>(sG_tskij, "Gloc") : proj.downfold_loc<false>(sG_tskij, "Gloc");
      Vhf_loc_sIab = (force_real)? proj.downfold_loc<true>(sVhf_skij, "Vhf_loc") : proj.downfold_loc<false>(sVhf_skij, "Vhf_loc");
      auto Sigma_tsIab = (force_real)? proj.downfold_loc<true>(sSigma_tskij, "Sigma_loc") : proj.downfold_loc<false>(sSigma_tskij, "Sigma_loc");
      ft->tau_to_w(Sigma_tsIab, Sigma_loc_wsIab, imag_axes_ft::fermion);
      _Timer.stop("DF_DOWNFOLD");

      _Timer.start("DF_DC");
      if (update_dc) {
        std::tie(Vhf_dc_sIab, Sigma_dc_wsIab) = double_counting(Gloc_tsIab, iter_2e_grp, dc_type, *ft);
      } else {
        auto df_1e_grp = gh5.open_group("downfold_1e");
        auto iter_1e_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
        app_log(1, "Reading double counting self-energy from \"downfold_1e/iter{}\"", weiss_f_iter);
        nda::h5_read(iter_1e_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
        nda::h5_read(iter_1e_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab);
      }
      _Timer.stop("DF_DC");

      // mixing
      if (mixing[0] < 1.0 and weiss_f_iter!=-1 and embed_iter != -1) {
        auto df_1e_grp = gh5.open_group("downfold_1e");
        auto iter_1e_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
        nda::array<ComplexType, 4> Vhf_prev_sIab;
        nda::array<ComplexType, 5> Sigma_prev_sIab;
        nda::h5_read(iter_1e_grp, "Vhf_dc_sIab", Vhf_prev_sIab);
        nda::h5_read(iter_1e_grp, "Sigma_dc_wsIab", Sigma_prev_sIab);
        Vhf_dc_sIab *= mixing[0];
        Vhf_dc_sIab += (1 - mixing[0]) * Vhf_prev_sIab;
        Sigma_dc_wsIab *= mixing[0];
        Sigma_dc_wsIab += (1 - mixing[0]) * Sigma_prev_sIab;
      }

      _Timer.start("DF_G_WEISS");
      ft->tau_to_w(Gloc_tsIab, Gloc_wsIab, imag_axes_ft::fermion);
      if (g_weiss_type == "dmft") {
        // Calculate the fermionic Weiss field:
        //     g_weiss(w)^{-1} = Gloc(w)^{-1} + vhf_imp + sigma_imp(w)
        // If weiss_f_iter==-1, we are in the 1st iteration of embedding and there is
        // no impurity self-energy. In that case, we assume Sigma_imp = Sigma_dc.
        //
        app_log(1, "\nEvaluating fermionic Weiss field with\n"
                   "  - Gloc from {}/iter{}", g_k_grp, g_k_iter);
        if (embed_iter != -1 and weiss_f_iter != -1)
          app_log(1, "  - Impurity self-energy from downfold_1e/iter{}", weiss_f_iter);
        else
          app_log(1, "  - Approximate impurity self-energy using double-counting self-energy\n");
        g_weiss_wsIab = (weiss_f_iter != -1 and embed_iter != -1) ?
                        compute_g_weiss(Gloc_wsIab, gh5, weiss_f_iter) : compute_g_weiss(Gloc_wsIab, Vhf_dc_sIab,
                                                                                         Sigma_dc_wsIab);
        // mixing for fermionic Weiss field
        if (mixing[1] < 1.0 and weiss_f_iter != -1 and embed_iter != -1) {
          app_log(1, "\nMixing the bosonic Weiss field with the previous iteration: {}\n", mixing[1]);
          auto df_1e_grp = gh5.open_group("downfold_1e");
          auto iter_1e_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
          nda::array<ComplexType, 5> g_weiss_prev_wsIab;
          nda::h5_read(iter_1e_grp, "g_weiss_wsIab", g_weiss_prev_wsIab);
          g_weiss_wsIab *= mixing[1];
          g_weiss_wsIab += (1 - mixing[1]) * g_weiss_prev_wsIab;
        }
      } else {
        app_log(1, "\nSetting fermionic Weiss field as Gloc.\n");
        g_weiss_wsIab() = Gloc_wsIab;
      }
    }
    mpi->comm.barrier();

    {
      nda::array<ComplexType, 5> g_weiss_tsIab(ft->nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
      ft->w_to_tau(g_weiss_wsIab, g_weiss_tsIab, imag_axes_ft::fermion);
      ft->check_leakage(g_weiss_tsIab, imag_axes_ft::fermion, std::addressof(mpi->comm), "Fermionic Weiss field");
    }
    auto H0_loc_sIab = (force_real)?
        proj.downfold_loc<true>(dyson.sH0_skij(), "H0_loc") :
        proj.downfold_loc<false>(dyson.sH0_skij(), "H0_loc");
    mpi->comm.barrier();
    _Timer.stop("DF_G_WEISS");

    //
    // Write all the results to 'downfold_1e' group
    //
    _Timer.start("DF_WRITE");
    if (mpi->comm.root()) {
      // update weiss_f_iter based on input G_tskij
      weiss_f_iter = g_k_iter+1;
      h5::file file(filename, 'a');
      auto grp = h5::group(file);
      auto weiss_f_grp = (grp.has_subgroup("downfold_1e"))?
                         grp.open_group("downfold_1e") : grp.create_group("downfold_1e");
      auto iter_grp = (weiss_f_grp.has_subgroup("iter"+std::to_string(weiss_f_iter)))?
                      weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter)) :
                      weiss_f_grp.create_group("iter"+std::to_string(weiss_f_iter));

      h5::h5_write(weiss_f_grp, "final_iter", weiss_f_iter);
      nda::h5_write(weiss_f_grp, "C_skIai", proj.C_skIai(), false);
      nda::h5_write(iter_grp, "H0_sIab", H0_loc_sIab, false);
      nda::h5_write(iter_grp, "Vhf_loc_sIab", Vhf_loc_sIab, false);
      nda::h5_write(iter_grp, "Sigma_loc_wsIab", Sigma_loc_wsIab, false);
      nda::h5_write(iter_grp, "Gloc_wsIab", Gloc_wsIab, false);
      nda::h5_write(iter_grp, "Vhf_dc_sIab", Vhf_dc_sIab, false);
      nda::h5_write(iter_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab, false);
      nda::h5_write(iter_grp, "g_weiss_wsIab", g_weiss_wsIab, false);
      h5::h5_write(iter_grp, "mu", mu);
      h5::h5_write(iter_grp, "dc_type", dc_type);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_WRITE");

    _Timer.stop("DF_TOTAL");
    print_downfold_mb_timers();
  }

  void embed_t::downfold_mb_solution_qp_impl(MBState &mb_state, qp_params_t &qp_params,
                                             bool update_dc, std::string dc_type,
                                             bool force_real, std::string format_type) {
    using math::shm::make_shared_array;
    std::vector<std::string> accept_dc_type = {"hf", "gw", "gw_dynamic_u"};
    utils::check(std::find(accept_dc_type.begin(), accept_dc_type.end(), dc_type) != accept_dc_type.end(),
                 "embed_t::downfold_mb_solution_qp_impl: Invalid dc_type: {}.", dc_type);

    for( auto& v: {"DF_TOTAL", "DF_ALLOC",
                   "DF_DC", "DF_DOWNFOLD", "DF_UPFOLD", "DF_DYSON", "DF_FIND_MU",
                   "DF_G_WEISS", "DF_READ", "DF_WRITE"} ) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    auto mpi  = mb_state.mpi;
    auto proj = mb_state.proj_boson.value().proj_fermi();
    auto ft   = mb_state.ft;
    auto nImps = proj.nImps();
    auto nImpOrbs = proj.nImpOrbs();

    _Timer.start("DF_READ");
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    auto dyson = simple_dyson(_MF, ft, mb_state.coqui_prefix);
    mpi->comm.barrier();
    _Timer.start("DF_READ");

    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┌┐┌┌─┐  ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║  │ ││││├┤───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┘└┘└─┘  └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "One-electron Hamiltonian downfolding for many-body solutions:");
    app_log(1, "(applying static approximation to the non-local self-energy)\n");
    app_log(1, "  - scf check-point file:                      {}", filename);
    app_log(1, "  - transformation matrices:                   {}", proj.C_file());
    app_log(1, "  - force real local Hamiltonian:              {}", force_real);
    app_log(1, "  - number of impurities:                      {}", nImps);
    app_log(1, "  - number of local orbitals per impurity:     {}", nImpOrbs);
    app_log(1, "  - range of primary orbitals for local basis: [{}, {})", proj.W_rng()[0].first(), proj.W_rng()[0].last());
    app_log(1, "  - gw iteration:                              {}", gw_iter);
    app_log(1, "  - downfold_1e iteration:                     {}", weiss_f_iter);
    app_log(1, "  - downfold_2e iteration:                     {}", weiss_b_iter);
    app_log(1, "  - embed iteration:                           {}", embed_iter);
    if (update_dc) {
      app_log(1, "  - update dc self-energy:                     {}", update_dc);
      app_log(1, "  - double counting type:                      {}\n", dc_type);
    } else
      app_log(1, "  - update dc self-energy:                     {}\n", update_dc);

    ft->metadata_log();
    long dc_iter = downfold_1e_logic(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter, filename, update_dc);
    mpi->comm.barrier();

    //
    // 1) Read GW results
    // 2) Find QP energies
    // 3) QP approx to dynamic self-energy
    // 4) Downfold Vhf_skij and Vcorr_skij
    //
    _Timer.start("DF_ALLOC");
    // Initialize MBState
    mb_state.sF_skij.emplace(make_shared_array<Array_view_4D_t>(
        *mpi, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *mpi, {ft->nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(
        *mpi, {ft->nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()}));
    auto& sVhf_skij = mb_state.sF_skij.value();
    auto& sG_tskij = mb_state.sG_tskij.value();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();
    double mu;

    auto sVcorr_skij = make_shared_array<Array_view_4D_t>(
        *mpi, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    auto sMO_skia = make_shared_array<Array_view_4D_t>(
        *mpi, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    auto sE_ska = make_shared_array<Array_view_3D_t>(
        *mpi, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd()});
    _Timer.stop("DF_ALLOC");

    _Timer.start("DF_READ");
    // Read GW results: Vhf_skij, sSigma_tskij, and mu
    utils::check(chkpt::read_scf(mpi->node_comm, sVhf_skij, sSigma_tskij, mu, mb_state.coqui_prefix) == gw_iter,
                 "embed_t::downfold_mb_solution_qp_impl: "
                 "Inconsistent gw iterations. gw_iter ({}) is not the last iteration in {}",
                 gw_iter, filename);
    mpi->comm.barrier();

    if (weiss_f_iter==-1) {
      // QP approximation for the dynamic GW self-energy
      auto psp = hamilt::make_pseudopot(*_MF);
      // a) Find MOs of the mean-field solution
      std::tie(sMO_skia, sE_ska) = get_mf_MOs(*mpi, *_MF, *psp);

      // b) compute qp energies; sbuff_tskij = Sigma_tskij, sbuff_skij = Vhf_skij
      if (sVhf_skij.node_comm()->root()) sVhf_skij.local() += dyson.H0();
      mpi->comm.barrier();
      solve_qp_eqn(sE_ska, sSigma_tskij, sVhf_skij, sMO_skia, mu, *ft, qp_params);

      // c) qp approximation for V_QPGW^{k}
      sVcorr_skij = qp_approx(sSigma_tskij,  sMO_skia, sE_ska, mu, *ft, qp_params);
      // this is not necessary but useful.
      double mu_qpgw = update_mu(mu, *_MF, sE_ska, ft->beta());

      if (sVhf_skij.node_comm()->root()) sVhf_skij.local() -= dyson.H0();
      mpi->comm.barrier();

      chkpt::write_qpgw_results(filename, gw_iter, sE_ska, sMO_skia, sVcorr_skij, mu_qpgw);
    } else {
      app_log(2, "Reading QPGW results from \"scf/iter{}\"", gw_iter);
      if (sVcorr_skij.node_comm()->root()) {
        auto E_ska = sE_ska.local();
        auto MO_skia = sMO_skia.local();
        auto Vcorr_skij = sVcorr_skij.local();
        h5::file file(filename, 'r');
        auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(gw_iter));
        utils::check(iter_grp.has_subgroup("qp_approx"),
                     "embed_t::dmft_embed_qp_impl: QPGW effective Hamiltonian does not exist in \"scf/iter{}\" at weiss_f_iter = {}. "
                     "This implies that your previous \"downfold_1e\" iteration does not finish properly. Please check!",
                     gw_iter, weiss_f_iter);
        auto qp_grp = iter_grp.open_group("qp_approx");
        nda::h5_read(qp_grp, "E_ska", E_ska);
        nda::h5_read(qp_grp, "MO_skia", MO_skia);
        if (qp_grp.has_dataset("Vcorr_skab"))
          nda::h5_read(qp_grp, "Vcorr_skab", Vcorr_skij); // CNY: backward compatibility
        else
          nda::h5_read(qp_grp, "Vcorr_skij", Vcorr_skij);
      }
    }

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
    if (!mb_state.Sigma_imp_wsIab or !mb_state.Vhf_imp_sIab) {
      // Read local self-energy corrections from the checkpoint file if they are not set in MBState.
      app_log(2, "MBState does not contain impurity self-energies, "
                 "trying to read them from the checkpoint file {} from", filename);
      app_log(2, "  - HDF5 group:           downfold_1e");
      app_log(2, "  - Iteration:            {}\n", weiss_f_iter);

      long nw = ft->nw_f();
      mb_state.Sigma_imp_wsIab.emplace(nda::array<ComplexType, 5>(nw, _MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      mb_state.Vhf_imp_sIab.emplace(nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs));
      if (_context->comm.root()) {
        // 2) if the file does not contain them, we set them to empty arrays
        // Return false if not "all" the local corrections present in the checkpoint file,
        // --> the local corrections will be set to zero.
        sigma_local_given = chkpt::read_sigma_local(mb_state.Sigma_imp_wsIab.value(),
                                                    mb_state.Vhf_imp_sIab.value(),
                                                    filename, weiss_f_iter);
      }
      _context->comm.broadcast_n(&sigma_local_given, 1, 0);
      _context->comm.broadcast_n(mb_state.Sigma_imp_wsIab.value().data(), mb_state.Sigma_imp_wsIab.value().size(), 0);
      _context->comm.broadcast_n(mb_state.Vhf_imp_sIab.value().data(), mb_state.Vhf_imp_sIab.value().size(), 0);
      if (sigma_local_given)
        app_log(2, "Found impurity self-energies in the checkpoint file {}", filename);
    } else {
      app_log(2, "Found impurity self-energies already set in MBState.");
      sigma_local_given = true;
    }
    mpi->comm.barrier();
    _Timer.stop("DF_READ");

    _Timer.start("DF_DOWNFOLD");
    auto Vhf_loc_sIab = (force_real)?
        proj.downfold_loc<true>(sVhf_skij, "Vhf_gw_loc") :
        proj.downfold_loc<false>(sVhf_skij, "Vhf_gw_loc");
    auto Vcorr_loc_sIab = (force_real)?
        proj.downfold_loc<true>(sVcorr_skij, "Vcorr_gw_loc") :
        proj.downfold_loc<false>(sVcorr_skij, "Vcorr_gw_loc");
    mpi->comm.barrier();
    _Timer.stop("DF_DOWNFOLD");

    _Timer.start("DF_DC");
    //
    // Calculate double counting terms using the Green's function from "/dc_src_grp/dc_iter"
    //
    // TODO When update_dc=true, what is the proper choice of mu? mu_gw or mu_emb?
    std::string dc_src_grp = (embed_iter!=-1)? "embed" : "scf";
    auto [Vhf_dc_sIab, Vcorr_dc_sIab, Sigma_dc_tsIab] = (update_dc)?
        double_counting_qp(mb_state.coqui_prefix, dc_type, dc_iter, dc_src_grp, weiss_b_iter, *ft,
                           mu, sMO_skia, sE_ska, qp_params, force_real, format_type) :
        read_double_counting_qp(filename, weiss_f_iter, *ft);
    nda::array<ComplexType, 5> Sigma_dc_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    ft->tau_to_w(Sigma_dc_tsIab, Sigma_dc_wsIab, imag_axes_ft::fermion);

    mb_state.Vhf_dc_sIab = Vhf_dc_sIab;
    mb_state.Vcorr_dc_sIab = Vcorr_dc_sIab;
    mb_state.Sigma_dc_wsIab = Sigma_dc_wsIab;

    mpi->comm.barrier();
    _Timer.stop("DF_DC");

    _Timer.start("DF_UPFOLD");
    // construct the Green's function with static approximation to the dynamic self-energy
    // Dyson equation for effective mean-field Hamiltonian with static approximation to the dynamic self-energy
    // Feff_skij = Vhf_skij + Vcorr_skij - Vhf_dc_skij - Vcorr_dc_skij + Vhf_imp_skij
    // Assume Vhf_imp_sIab = Vhf_dc_sIab -> only need to subtract Vcorr_dc_skij
    //
    // Sigma_tskij = Sigma_imp_tskij
    //
    sG_tskij.set_zero();
    sSigma_tskij.set_zero();

    // Vhf_skij = Vhf_skij + Vcorr_skij
    if (sVhf_skij.node_comm()->root()) sVhf_skij.local() += sVcorr_skij.local();
    mpi->comm.barrier();

    // Vhf_skij = Vhf_skij + Vcorr_skij - Vcorr_dc_skij
    proj.upfold(sVcorr_skij, Vcorr_dc_sIab);
    if (sVhf_skij.node_comm()->root()) sVhf_skij.local() -= sVcorr_skij.local();
    mpi->comm.barrier();

    // If weiss_f_iter==-1, we are in the 1st iteration of embedding and there is
    // no impurity self-energy.
    // In that case, we assume Sigma_imp = Sigma_dc and Vhf_imp = Vhf_dc.
    if (weiss_f_iter != -1 and embed_iter != -1) {
      app_log(2, "Add local corrections from \"downfold_1e/iter{}\"", weiss_f_iter);

      // Vhf_skij = Vhf_skij + Vcorr_skij - Vcorr_dc_skij + (Vhf_imp_skij - Vhf_dc_skij)
      add_Vhf_correction(mb_state);
      // Sigma_tskij = Sigma_imp_tskij
      add_Sigma_dyn_correction(mb_state, false);

    } else {
      // Sigma_tskij = sSigma_dc_tskij
      proj.upfold(sSigma_tskij, Sigma_dc_tsIab);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_UPFOLD");

    _Timer.start("DF_FIND_MU");
    mu = update_mu(mu, dyson, *_MF, *ft, sVhf_skij, sSigma_tskij);
    mpi->comm.barrier();
    _Timer.stop("DF_FIND_MU");

    _Timer.start("DF_DYSON");
    dyson.solve_dyson(sG_tskij, sVhf_skij, sSigma_tskij, mu);
    mpi->comm.barrier();
    _Timer.stop("DF_DYSON");

    _Timer.start("DF_DOWNFOLD");
    // reuse memory
    Sigma_dc_tsIab = (force_real)?
        proj.downfold_loc<true>(sG_tskij, "Gtot_loc") :
        proj.downfold_loc<false>(sG_tskij, "Gtot_loc");
    nda::array<ComplexType, 5> Gloc_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    ft->tau_to_w(Sigma_dc_tsIab, Gloc_wsIab, imag_axes_ft::fermion);
    auto H0_loc_sIab = (force_real)?
        proj.downfold_loc<true>(dyson.sH0_skij(), "H0_loc") :
        proj.downfold_loc<false>(dyson.sH0_skij(), "H0_loc");
    _Timer.start("DF_DOWNFOLD");

    //
    // Calculate fermionic Weiss field:
    //     g_weiss(w)^{-1} = Gloc(w)^{-1} + vhf_imp + sigma_imp(w)
    // If weiss_f_iter==-1, we are in the 1st iteration of embedding and there is
    // no impurity self-energy. In that case, we assume Sigma_imp = Sigma_dc.
    //
    _Timer.start("DF_G_WEISS");
    nda::array<ComplexType, 5> g_weiss_wsIab(ft->nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    if (weiss_f_iter!=-1 and embed_iter!=-1) {
      h5::file file(filename, 'r');
      auto gh5 = h5::group(file);
      g_weiss_wsIab = compute_g_weiss(Gloc_wsIab, gh5, weiss_f_iter);
    } else {
      g_weiss_wsIab = compute_g_weiss(Gloc_wsIab, Vhf_dc_sIab, Sigma_dc_wsIab);
    }
    H0_loc_sIab += (Vhf_loc_sIab + Vcorr_loc_sIab - Vhf_dc_sIab - Vcorr_dc_sIab);
    auto delta_wsIab = compute_hybridization(g_weiss_wsIab, H0_loc_sIab, mu, *ft);
    H0_loc_sIab -= (Vhf_loc_sIab + Vcorr_loc_sIab - Vhf_dc_sIab - Vcorr_dc_sIab);
    mpi->comm.barrier();
    _Timer.stop("DF_G_WEISS");

    //
    // Write all the results to 'downfold_1e' group
    //
    _Timer.start("DF_WRITE");
    if (mpi->comm.root()) {
      if( format_type == "default" ) {
        // update weiss_f_iter based on input G_tskij
        weiss_f_iter = (embed_iter!=-1)? embed_iter+1 : gw_iter;
        h5::file file(filename, 'a');
        auto grp = h5::group(file);
        auto weiss_f_grp = (grp.has_subgroup("downfold_1e"))?
                           grp.open_group("downfold_1e") : grp.create_group("downfold_1e");
        auto iter_grp = (weiss_f_grp.has_subgroup("iter"+std::to_string(weiss_f_iter)))?
                        weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter)) :
                        weiss_f_grp.create_group("iter"+std::to_string(weiss_f_iter));

        h5::h5_write(weiss_f_grp, "final_iter", (long)weiss_f_iter);
        nda::h5_write(weiss_f_grp, "C_skIai", proj.C_skIai(), false);
        nda::h5_write(iter_grp, "H0_sIab", H0_loc_sIab, false);
        nda::h5_write(iter_grp, "Vhf_gw_sIab", Vhf_loc_sIab, false);
        nda::h5_write(iter_grp, "Vcorr_gw_sIab", Vcorr_loc_sIab, false);
        nda::h5_write(iter_grp, "Gloc_wsIab", Gloc_wsIab, false);
        nda::h5_write(iter_grp, "Vhf_dc_sIab", Vhf_dc_sIab, false);
        nda::h5_write(iter_grp, "Vcorr_dc_sIab", Vcorr_dc_sIab, false);
        nda::h5_write(iter_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab, false);
        nda::h5_write(iter_grp, "g_weiss_wsIab", g_weiss_wsIab, false);
        nda::h5_write(iter_grp, "delta_wsIab", delta_wsIab, false);
        h5::h5_write(iter_grp, "mu", mu);
        h5::h5_write(iter_grp, "dc_type", dc_type);
      } else if( format_type == "model_static" ) {
        // write /System consistent with downfolded hamiltonian
        std::string model_filename = mb_state.coqui_prefix + ".model.h5";
        h5::file file(model_filename, 'a');
        auto grp = h5::group(file);
        auto sgrp = grp.create_group("System");
        // adds bz_symm checkpoint consistent with a single kpoint model at the gamma point
        mf::bz_symm::gamma_point_h5(sgrp);

        // unit cell
        h5::h5_write_attribute(sgrp, "number_of_spins", 1);
        h5::h5_write_attribute(sgrp, "number_of_spins_in_basis", 1);
        h5::h5_write_attribute(sgrp, "number_of_bands", nImpOrbs);
        h5::h5_write_attribute(sgrp, "number_of_elec", 1);
        h5::h5_write_attribute(sgrp, "madelung_constant", 0.0);
        h5::h5_write_attribute(sgrp, "nuclear_energy", 0.0);

        nda::h5_write(sgrp, "kpoint_weights", nda::array<double,1>(1,1.0), false);

        utils::check(nImps==1, "Error in hf_downfolding: nImps != 1 not yet implemented.");
        H0_loc_sIab() += (Vhf_loc_sIab() - Vhf_dc_sIab() + Vcorr_loc_sIab() - Vcorr_dc_sIab());
        nda::h5_write(sgrp, "H0", H0_loc_sIab, false);
        H0_loc_sIab() = ComplexType(0.0);
        auto Hab = H0_loc_sIab(0,0,nda::ellipsis{});
        nda::diagonal(Hab) = ComplexType(1.0);
        nda::h5_write(sgrp, "overlap", H0_loc_sIab, false);
        H0_loc_sIab() = ComplexType(0.0);
        // to do: provide proper density matrix and fock matrices from non-interacting solution
        nda::h5_write(sgrp, "density_matrix", H0_loc_sIab, false);
        nda::h5_write(sgrp, "fock_matrix", H0_loc_sIab, false);
      } else 
        APP_ABORT("Error in downfold_mb_solution_qp_impl: Unknown format type: {}",format_type);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_WRITE");

    _Timer.stop("DF_TOTAL");
    print_downfold_mb_timers();
  }

  template<THC_ERI thc_t>
  void embed_t::downfold_hf_impl(std::string prefix,
                                 thc_t& eri,
                                 imag_axes_ft::IAFT &ft,
                                 bool force_real,
                                 std::string hf_div_treatment) {
    using math::shm::make_shared_array;
    decltype(nda::range::all) all;

    for( auto& v: {"DF_TOTAL", "DF_ALLOC", "DF_READ",
                   "DF_DC", "DF_DOWNFOLD", "DF_WRITE"} ) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    auto nImps = _proj.value().nImps();
    auto nImpOrbs = _proj.value().nImpOrbs();
    std::string filename = prefix + ".mbpt.h5";
    utils::check(std::filesystem::exists(filename),
                 "embed_t::downfolding_hf_impl: checkpoint file, {}, does not exist!", filename);

    app_log(2, "  Frozen-core downfolding starting from a mean-field solution. \n");

    _Timer.start("DF_READ");
    auto [scf_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    downfold_hf_logic(scf_iter, weiss_f_iter, weiss_b_iter, embed_iter, filename);
    _context->comm.barrier();
    _Timer.stop("DF_READ");

    _Timer.start("DF_ALLOC");
    auto sVhf_skij = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    math::shm::shared_array<Array_view_4D_t>& sHeff_skij = sVhf_skij;
    auto sH0_skij = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    auto sS_skij = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    double mu = 0.0;
    // generates a new pseudopot object if not found in mf. Stores shared_ptr in mf and returns it.
    auto psp = hamilt::make_pseudopot(*_MF);
    hamilt::set_H0(*_MF, psp.get(), sH0_skij);
    hamilt::set_ovlp(*_MF, sS_skij);
    hamilt::set_fock(*_MF, psp.get(), sHeff_skij, false);

    // Obtains MO coefficients and energies from the given mean-field object
    auto sMO_skia = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    auto sE_ska = make_shared_array<Array_view_3D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd()});
    auto sDm_skij = make_shared_array<Array_view_4D_t>(
        *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
    update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
    mu = update_mu(mu, *_MF, sE_ska, ft.beta());
    update_Dm(sDm_skij, sMO_skia, sE_ska, mu, ft.beta());
    _Timer.stop("DF_ALLOC");

    // compute HF for the whole lattice problem
    solvers::hf_t hf(hf_div_treatment);
    hf.evaluate(sVhf_skij, sDm_skij.local(), eri, sS_skij.local());

    // downfold the HF solution
    _Timer.start("DF_DOWNFOLD");
    auto Vhf_loc_sIab = (force_real)?
        _proj.value().downfold_loc<true>(sVhf_skij, "Vhf_loc") :
        _proj.value().downfold_loc<false>(sVhf_skij, "Vhf_loc");
    auto H0_loc_sIab = (force_real)?
        _proj.value().downfold_loc<true>(sH0_skij, "H0_loc") :
        _proj.value().downfold_loc<false>(sH0_skij, "H0_loc");
    _context->comm.barrier();
    _Timer.stop("DF_DOWNFOLD");

    _Timer.start("DF_DC");
    // compute double counting contribution; scf_iter is always 0!
    auto Vhf_dc_sIab = double_counting_hf_bare(prefix, scf_iter, "scf", weiss_b_iter, force_real, "model_static");
    _Timer.stop("DF_DC");

    _Timer.start("DF_WRITE");

    if (_context->comm.root()) {
      // write /System consistent with downfolded hamiltonian
      std::string model_filename = prefix + ".model.h5";
      h5::file file(model_filename, 'a');
      auto grp = h5::group(file);
      auto sgrp = grp.create_group("System"); 
      // adds bz_symm checkpoint consistent with a single kpoint model at the gamma point
      mf::bz_symm::gamma_point_h5(sgrp);

      // unit cell
      h5::h5_write_attribute(sgrp, "number_of_spins", 1);
      h5::h5_write_attribute(sgrp, "number_of_spins_in_basis", 1);
      h5::h5_write_attribute(sgrp, "number_of_bands", nImpOrbs);
      h5::h5_write_attribute(sgrp, "number_of_elec", 1);
      h5::h5_write_attribute(sgrp, "madelung_constant", 0.0);
      h5::h5_write_attribute(sgrp, "nuclear_energy", 0.0);

      nda::h5_write(sgrp, "kpoint_weights", nda::array<double,1>(1,1.0), false);

      utils::check(nImps==1, "Error in hf_downfolding: nImps != 1 not yet implemented.");
      //H0_loc_sIab() += (Vhf_loc_sIab() - Vhf_dc_sIab()); 
      H0_loc_sIab() += Vhf_loc_sIab(); 
      nda::h5_write(sgrp, "H0", H0_loc_sIab, false);
      H0_loc_sIab() = ComplexType(0.0);
      auto Hab = H0_loc_sIab(0,0,all,all);
      nda::diagonal(Hab) = ComplexType(1.0); 
      nda::h5_write(sgrp, "overlap", H0_loc_sIab, false);
      H0_loc_sIab() = ComplexType(0.0);
      // to do: provide proper density matrix and fock matrices from non-interacting solution
      nda::h5_write(sgrp, "density_matrix", H0_loc_sIab, false);
      nda::h5_write(sgrp, "fock_matrix", H0_loc_sIab, false);
//      h5::h5_write(sgrp, "chemical_potential", mu);
    }
    _context->comm.barrier();
    _Timer.stop("DF_WRITE");

    _Timer.start("DF_TOTAL");
    print_downfold_hf_timers();
  }

  auto embed_t::double_counting_hf_bare(std::string prefix,
                                        long dc_iter, std::string dc_src_grp,
                                        long weiss_b_iter, bool force_real,
                                        std::string format_type)
  -> nda::array<ComplexType, 4> {
    h5::file file(prefix+".mbpt.h5", 'r');
    auto grp = h5::group(file);
    utils::check(format_type == "default" or format_type == "model_static", 
                 "Invalid format_type: {}",format_type);
    if(format_type == "default") {
      return double_counting_hf_bare(grp, grp, dc_iter, dc_src_grp, weiss_b_iter, force_real, format_type);
    } else { 
      h5::file fileV(prefix+".model.h5", 'r');
      auto grpV = h5::group(fileV);
      return double_counting_hf_bare(grp, grpV, dc_iter, dc_src_grp, weiss_b_iter, force_real, format_type);
    } 
  }

  auto embed_t::double_counting_hf_bare(h5::group &gh5_dc, h5::group &gh5_V, 
                                        long dc_iter, std::string dc_src_grp,
                                        long weiss_b_iter, bool force_real,
                                        std::string format_type)
    -> nda::array<ComplexType, 4> {
    decltype(nda::range::all) all;
    using math::shm::make_shared_array;

    auto nImps = _proj.value().nImps();
    auto nImpOrbs = _proj.value().nImpOrbs();

    nda::array<ComplexType, 4> Dm_sIab(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    {
      auto sDm_skij = make_shared_array<Array_view_4D_t>(
          *_context, {_MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
      auto Dm_skij = sDm_skij.local();
      auto dc_grp = gh5_dc.open_group(dc_src_grp);
      auto iter_grp = dc_grp.open_group("iter" + std::to_string(dc_iter));
      if (_context->node_comm.root()) {
        nda::h5_read(iter_grp, "Dm_skij", Dm_skij);
      }
      _context->node_comm.barrier();
      Dm_sIab = (force_real)?
          _proj.value().downfold_loc<true>(sDm_skij, "Dm_loc") :
          _proj.value().downfold_loc<false>(sDm_skij, "Dm_loc");
    }
    nda::array<ComplexType, 4> Vhf_dc_sIab;

    if(format_type == "default") {
      app_log(2, "Evaluating Hartree-Fock double counting with local density coming from "
                 "\"{}/iter{}\"", dc_src_grp, dc_iter);

      // read interactions
      auto sV_abcd = make_shared_array<Array_view_4D_t>(
          *_context, {nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
      auto V_abcd = sV_abcd.local();
      if (_context->node_comm.root()) {
        auto weiss_b_grp = gh5_V.open_group("downfold_2e");
        nda::h5_read(weiss_b_grp, "iter" + std::to_string(weiss_b_iter) + "/Vloc_abcd", V_abcd);
      }
      _context->node_comm.barrier();

      Vhf_dc_sIab = hartree_double_counting(Dm_sIab, V_abcd);
      Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_abcd);

    } else if(format_type == "model_static"){
    
      long factorization_type = 0;
      long _nChol = 0;
      if (_context->comm.root()) {
        if(gh5_V.has_key("Interaction/factorization_type")) {
          std::string ftype = "none";
          h5::h5_read(gh5_V, "Interaction/factorization_type", ftype);
          app_log(2,"factorization_type: {}",ftype);
          if(ftype.substr(0,8) == "cholesky") {
            factorization_type=1;
            auto l = h5::array_interface::get_dataset_info(gh5_V, 
                       "Interaction/Vq0");
            _nChol = l.lengths[0]; 
          } else if(ftype.substr(0,3) == "thc") {
            factorization_type=2;
          }
        }
      }
      _context->comm.broadcast_n(&factorization_type, 1);
      if(factorization_type == 1) _context->comm.broadcast_n(&_nChol, 1);

      if(factorization_type == 0) {
        // read interactions
        auto sV_abcd = make_shared_array<Array_view_4D_t>(
            *_context, {nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
        auto V_abcd = sV_abcd.local();
        if (_context->node_comm.root())
          nda::h5_read(gh5_V, "Interaction/Vq0", V_abcd);
        _context->node_comm.barrier();

        Vhf_dc_sIab = hartree_double_counting(Dm_sIab, V_abcd);
        Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_abcd);
      } else if(factorization_type == 1) {
        // read interactions
        auto [na,nb] = itertools::chunk_range(0,_nChol,_context->comm.size(),_context->comm.rank());
        if( nb > na ) { 
          nda::array<ComplexType,5> V(nb-na,1,1,nImpOrbs,nImpOrbs);
          nda::h5_read(gh5_V, "Interaction/Vq0", V, 
                       std::tuple{nda::range(na,nb),all,all,all,all});
          auto V_nab = nda::reshape(V, std::array<long,3>{nb-na, nImpOrbs, nImpOrbs});

          Vhf_dc_sIab = hartree_double_counting(Dm_sIab, V_nab);
          Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_nab);
        } else {
          Vhf_dc_sIab = nda::array<ComplexType, 4>(Dm_sIab.shape());
          Vhf_dc_sIab() = ComplexType(0.0);
        }
        _context->comm.all_reduce_in_place_n(Vhf_dc_sIab.data(),Vhf_dc_sIab.size(),std::plus<>{});
      } else if(factorization_type == 2) {
        APP_ABORT(" Finish implementation of factorization_type=thc in doble_counting_hf_bare");
      } else {
        APP_ABORT(" Error in doble_counting_hf_bare: Invalid factorization_type: {}",factorization_type);
      }
    
    } else
      APP_ABORT("Error in double_counting_hf_bare: Unknown format_type: {}",format_type);
    _context->comm.barrier();

    return Vhf_dc_sIab;
  }

  auto embed_t::double_counting(const nda::array<ComplexType, 5> &Gloc_tsIab,
                                h5::group &gh5,
                                std::string dc_type,
                                imag_axes_ft::IAFT &ft)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>> {
    using math::shm::make_shared_array;
    decltype(nda::range::all) all;

    auto nImps = _proj.value().nImps();
    auto nImpOrbs = _proj.value().nImpOrbs();

    // read local screened interactions
    auto sV_abcd = make_shared_array<Array_view_4D_t>(
        *_context, {nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
    auto V_abcd = sV_abcd.local();
    auto sUw0_abcd = make_shared_array<Array_view_4D_t>(
        *_context, {nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
    auto Uw0_abcd = sUw0_abcd.local();

    if (_context->node_comm.root()) {
      if (dc_type == "gw_edmft_density") {
        nda::array<ComplexType, 4> Vtmp;
        nda::h5_read(gh5, "Vloc_abcd", Vtmp);
        for (size_t i=0; i<nImpOrbs; ++i)
          for (size_t j=i; j<nImpOrbs; ++j) {
            V_abcd(i, i, j, j) = Vtmp(i, i, j, j);
            if (j > i) {
              V_abcd(j, j, i, i) = Vtmp(j, j, i, i);
              // pair-hopping
              V_abcd(i, j, i, j) = Vtmp(i, j, i, j);
              V_abcd(j, i, j, i) = Vtmp(j, i, j, i);
              // spin-flip
              V_abcd(i, j, j, i) = Vtmp(i, j, j, i);
              V_abcd(j, i, i, j) = Vtmp(j, i, i, j);
            }
          }
        nda::h5_read(gh5, "Uloc_wabcd", Vtmp,
                     std::tuple{0, all, all, all, all});
        for (size_t i=0; i<nImpOrbs; ++i)
          for (size_t j=i; j<nImpOrbs; ++j) {
            Uw0_abcd(i, i, j, j) = Vtmp(i, i, j, j);
            if (j > i) {
              Uw0_abcd(j, j, i, i) = Vtmp(j, j, i, i);
            }
          }
        Uw0_abcd += V_abcd;

      } else {
        nda::h5_read(gh5, "Vloc_abcd", V_abcd);
        nda::h5_read(gh5, "Uloc_wabcd", Uw0_abcd,
                     std::tuple{0, all, all, all, all});
        Uw0_abcd += V_abcd;
      }
    }
    _context->node_comm.barrier();

    // GW+EDMFT double counting, both static and dynamic contributions
    nda::array<ComplexType, 4> Dm_sIab(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    ft.tau_to_beta(Gloc_tsIab, Dm_sIab);
    Dm_sIab() *= -1;
    nda::array<ComplexType, 4> Vhf_dc_sIab = hartree_double_counting(Dm_sIab, Uw0_abcd);
    nda::array<ComplexType, 5> Sigma_dc_wsIab(ft.nw_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);

    if (dc_type == "hf") {

      Vhf_dc_sIab += exchange_double_counting(Dm_sIab, Uw0_abcd);

    } else if (dc_type == "gw") {

      Vhf_dc_sIab += exchange_double_counting(Dm_sIab, Uw0_abcd);
      Sigma_dc_wsIab = gw_double_counting_dmft<true>(_context->comm, Gloc_tsIab, Uw0_abcd, ft);

    }  else if (dc_type == "gw_dynamic_u") {

      // Static GW self-energy using W(w->infty) = bare V
      Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_abcd);
      // Dynamic GW self-energy
      long nw_half = (ft.nw_b()%2==0)? ft.nw_b()/2 : ft.nw_b()/2+1;
      auto sU_wabcd = make_shared_array<Array_view_5D_t>(
          *_context, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
      auto U_wabcd = sU_wabcd.local();
      if (_context->node_comm.root()) {
        nda::h5_read(gh5, "Uloc_wabcd", U_wabcd);
      }
      _context->node_comm.barrier();
      Sigma_dc_wsIab = gw_double_counting_dmft<true>(_context->comm, Gloc_tsIab, V_abcd, U_wabcd, ft);

    } else if (dc_type == "gw_edmft" or dc_type == "gw_edmft_density") {

      // Static GW self-energy using W(w->infty) = bare V
      Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_abcd);
      // Dynamic GW self-energy
      long nw_half = (ft.nw_b() % 2 == 0) ? ft.nw_b() / 2 : ft.nw_b() / 2 + 1;
      auto sW_wabcd = make_shared_array<Array_view_5D_t>(
          *_context, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
      auto W_wabcd = sW_wabcd.local();
      if (_context->node_comm.root()) {
        nda::h5_read(gh5, "Wloc_wabcd", W_wabcd);
      }
      _context->node_comm.barrier();
      Sigma_dc_wsIab = gw_edmft_double_counting<true>(_context->comm, Gloc_tsIab, W_wabcd, ft);

    } else {
      utils::check(false, "embed_t::double_counting: Invalid dc_type: {}", dc_type);
    }
    return std::tuple(Vhf_dc_sIab, Sigma_dc_wsIab);
  }

  auto embed_t::read_double_counting_qp(std::string filename, long weiss_f_iter, imag_axes_ft::IAFT &ft)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>> {

    auto nImps = _proj.value().nImps();
    auto nImpOrbs = _proj.value().nImpOrbs();

    app_log(2, "Reading double counting self-energy from \"downfold_1e/iter{}\"", weiss_f_iter);
    nda::array<ComplexType, 4> Vhf_dc_sIab;
    nda::array<ComplexType, 4> Vcorr_dc_sIab;
    nda::array<ComplexType, 5> Sigma_dc_wsIab;

    h5::file file(filename, 'r');
    auto weiss_f_grp = h5::group(file).open_group("downfold_1e");
    auto iter_grp = weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter));
    nda::h5_read(iter_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
    nda::h5_read(iter_grp, "Vcorr_dc_sIab", Vcorr_dc_sIab);
    nda::h5_read(iter_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab);

    nda::array<ComplexType, 5> Sigma_dc_tsIab(ft.nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    ft.w_to_tau(Sigma_dc_wsIab, Sigma_dc_tsIab, imag_axes_ft::fermion);

    return std::make_tuple(Vhf_dc_sIab, Vcorr_dc_sIab, Sigma_dc_tsIab);
  }

  auto embed_t::double_counting_qp(h5::group &gh5, h5::group &gh5_V, std::string prefix,
                                std::string dc_type, long dc_iter, std::string dc_src_grp,
                                long weiss_b_iter, imag_axes_ft::IAFT &ft,
                                double mu,
                                sArray_t<Array_view_4D_t> &sMO_skia, sArray_t<Array_view_3D_t> &sE_ska,
                                qp_params_t &qp_params, bool force_real, std::string format_type)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>> {
    app_log(2, "Evaluating double counting self-energy with Gloc coming from "
               "\"{}/iter{}\"", dc_src_grp, dc_iter);
    decltype(nda::range::all) all;

    auto nImps = _proj.value().nImps();
    auto nImpOrbs = _proj.value().nImpOrbs();

    // read Gloc
    nda::array<ComplexType, 5> Gloc_tsIab(ft.nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 4> Dm_sIab(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    {
      auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
          *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
      auto G_tskij = sG_tskij.local();

      auto dc_grp = gh5.open_group(dc_src_grp);
      auto iter_grp = dc_grp.open_group("iter" + std::to_string(dc_iter));
      if (iter_grp.has_dataset("G_tskij")) {
        if (_context->node_comm.root()) {
          nda::h5_read(iter_grp, "G_tskij", G_tskij);
        }
        _context->node_comm.barrier();
      } else
        compute_G_from_mf(iter_grp, ft, sG_tskij);
      Gloc_tsIab = (force_real)?
          _proj.value().downfold_loc<true>(sG_tskij, "Gloc_for_dc") :
          _proj.value().downfold_loc<false>(sG_tskij, "Gloc_for_dc");
      ft.tau_to_beta(Gloc_tsIab, Dm_sIab);
      Dm_sIab() *= -1;
    }

    // if format_type==model, read factorization type
    long factorization_type = 0;
    long _nChol = 0;
    if( format_type == "model_static" ) {
      utils::check(dc_type=="hf" or dc_type=="gw", "embed_t::double_counting_qp: Invalid dc_type:{} with format_type == model_static.",dc_type);
      if (_context->comm.root()) {
        std::string ftype = "none";
        h5::h5_read(gh5_V, "Interaction/factorization_type", ftype);
        if(ftype.substr(0,8) == "cholesky") {
          factorization_type=1;
          auto l = h5::array_interface::get_dataset_info(gh5_V,
                     "Interaction/Vq0");
          _nChol = l.lengths[0];
        } else if(ftype.substr(0,3) == "thc") {
          factorization_type=2;
        }
      }
      _context->comm.broadcast_n(&factorization_type, 1);
      if(factorization_type == 1) _context->comm.broadcast_n(&_nChol, 1);
    }

    nda::array<ComplexType, 4> Vhf_dc_sIab; //(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 4> Vcorr_dc_sIab; //(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 5> Sigma_dc_tsIab; //(ft.nt_f(), _MF->nspin(), nImps, nImpOrbs, nImpOrbs);
    if( format_type == "default" or factorization_type==0 ) {

      // read interactions
      nda::array<ComplexType, 4> V_abcd(nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
      nda::array<ComplexType, 4> Uw0_abcd(nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
      h5::group weiss_b_grp;
      if(format_type == "default") {
        weiss_b_grp = gh5_V.open_group("downfold_2e");
        nda::h5_read(weiss_b_grp, "iter" + std::to_string(weiss_b_iter) + "/Vloc_abcd", V_abcd);
        nda::h5_read(weiss_b_grp, "iter" + std::to_string(weiss_b_iter) + "/Uloc_wabcd", Uw0_abcd,
                     std::tuple{0, all, all, all, all});
        Uw0_abcd += V_abcd;
      } else {
        // already contains V_abcd
        nda::h5_read(gh5_V, "Interaction/Vq0", Uw0_abcd);
      }

      Vhf_dc_sIab = hartree_double_counting(Dm_sIab, Uw0_abcd);
      if (dc_type == "hf")
        Vhf_dc_sIab += exchange_double_counting(Dm_sIab, Uw0_abcd);
      else if (dc_type == "gw") {

        // Static GW self-energy using U(w=0)
        Vhf_dc_sIab += exchange_double_counting(Dm_sIab, Uw0_abcd);

        // Dynamic GW self-energy
        auto sSigma_dc_tskij = math::shm::make_shared_array<Array_view_5D_t>(
            *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
        Sigma_dc_tsIab = gw_double_counting_dmft<false>(_context->comm, Gloc_tsIab, Uw0_abcd, ft);
        _proj.value().upfold(sSigma_dc_tskij, Sigma_dc_tsIab); // upfold to the primary basis
        auto sVcorr_skij = qp_approx(sSigma_dc_tskij,  sMO_skia, sE_ska, mu, ft, qp_params); // static approximation
        // downfolding
        Vcorr_dc_sIab = (force_real)?
            _proj.value().downfold_loc<true>(sVcorr_skij, "Vcorr_dc_loc") :
            _proj.value().downfold_loc<false>(sVcorr_skij, "Vcorr_dc_loc");

      } else if (dc_type == "gw_dynamic_u" or dc_type == "gw_mix_u") {

        if (dc_type == "gw_dynamic_u") {
          // Static GW self-energy using W(w->infty) = bare V
          Vhf_dc_sIab += exchange_double_counting(Dm_sIab, V_abcd);
        } else {
          // Static GW self-energy using U(w=0)
          Vhf_dc_sIab += exchange_double_counting(Dm_sIab, Uw0_abcd);
        }

        // Dynamic GW self-energy
        auto sSigma_dc_tskij = math::shm::make_shared_array<Array_view_5D_t>(
            *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
        nda::array<ComplexType, 5> U_wabcd;
        nda::h5_read(weiss_b_grp, "iter" + std::to_string(weiss_b_iter) + "/Uloc_wabcd", U_wabcd);
        Sigma_dc_tsIab = gw_double_counting_dmft<false>(_context->comm, Gloc_tsIab, V_abcd, U_wabcd, ft);
        _proj.value().upfold(sSigma_dc_tskij, Sigma_dc_tsIab); // upfold to the primary basis
        auto sVcorr_skij = qp_approx(sSigma_dc_tskij,  sMO_skia, sE_ska, mu, ft, qp_params); // static approximation
        // downfolding
        Vcorr_dc_sIab = (force_real)?
            _proj.value().downfold_loc<true>(sVcorr_skij, "Vcorr_dc_loc") :
            _proj.value().downfold_loc<false>(sVcorr_skij, "Vcorr_dc_loc");

      } else
        utils::check(false, "embed_t::double_counting_qp: Invalid dc_type: {}", dc_type);

    } else if( format_type == "model_static" ) {

      if(factorization_type == 1) {

        // cholesky
        auto [na,nb] = itertools::chunk_range(0,_nChol,_context->comm.size(),_context->comm.rank());
        nda::array<ComplexType, 5> V;
        if( nb > na ) {
          nda::h5_read(gh5_V, "Interaction/Vq0", V, 
                       std::tuple{nda::range(na,nb),all,all,all,all});
          utils::check(V.extent(3) == V.extent(4) and V.extent(3) == nImpOrbs,"Dimension mismatch.");
          auto U_nab = nda::reshape(V, std::array<long,3>{nb-na, nImpOrbs, nImpOrbs});

          Vhf_dc_sIab = hartree_double_counting(Dm_sIab, U_nab);
          if (dc_type == "hf" or dc_type == "gw")
            Vhf_dc_sIab += exchange_double_counting(Dm_sIab, U_nab);

        } else {
          Vhf_dc_sIab = nda::array<ComplexType, 4>(_MF->nspin(), nImps, nImpOrbs, nImpOrbs);
          Vhf_dc_sIab() = ComplexType(0.0);
        }
        _context->comm.all_reduce_in_place_n(Vhf_dc_sIab.data(),Vhf_dc_sIab.size(),std::plus<>{});
        if(dc_type == "gw") {
          Sigma_dc_tsIab = gw_double_counting_dmft_cholesky<false>(_context, Gloc_tsIab, prefix+".model.h5", ft);
          auto sSigma_dc_tskij = math::shm::make_shared_array<Array_view_5D_t>(
                *_context, {ft.nt_f(), _MF->nspin(), _MF->nkpts_ibz(), _MF->nbnd(), _MF->nbnd()});
          _proj.value().upfold(sSigma_dc_tskij, Sigma_dc_tsIab); // upfold to the primary basis
          auto sVcorr_skij = qp_approx(sSigma_dc_tskij,  sMO_skia, sE_ska, mu, ft, qp_params); 
          // downfolding
          Vcorr_dc_sIab = (force_real)?
                _proj.value().downfold_loc<true>(sVcorr_skij, "Vcorr_dc_loc") :
                _proj.value().downfold_loc<false>(sVcorr_skij, "Vcorr_dc_loc");
        }
      } else if(factorization_type == 2) 
        APP_ABORT(" Finish implementation of factorization_type=thc in doble_counting_qp");
      else 
        APP_ABORT(" Error in doble_counting_qp: Invalid factorization_type: {}",factorization_type);

    } else 
      APP_ABORT("double_counting_qp: Invalid format_type: {}",format_type);

    return std::make_tuple(Vhf_dc_sIab, Vcorr_dc_sIab, Sigma_dc_tsIab);
  }

  auto embed_t::double_counting_qp(std::string prefix,
                                   std::string dc_type, long dc_iter, std::string dc_src_grp,
                                   long weiss_b_iter, imag_axes_ft::IAFT &ft,
                                   double mu,
                                   sArray_t<Array_view_4D_t> &sMO_skia, sArray_t<Array_view_3D_t> &sE_ska,
                                   qp_params_t &qp_params, bool force_real, std::string format_type)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>> {
    h5::file file(prefix+".mbpt.h5", 'r');
    auto grp = h5::group(file);
    utils::check(format_type == "default" or format_type == "model_static",
                 "Invalid format_type: {}",format_type);
    if(format_type == "default") {
      return double_counting_qp(grp, grp, prefix, dc_type, dc_iter, dc_src_grp,
                         weiss_b_iter, ft, mu, sMO_skia, sE_ska, qp_params, force_real, format_type);
    } else {
      h5::file fileV(prefix+".model.h5", 'r');
      auto grpV = h5::group(fileV);
      return double_counting_qp(grp, grpV, prefix, dc_type, dc_iter, dc_src_grp,
                         weiss_b_iter, ft, mu, sMO_skia, sE_ska, qp_params, force_real, format_type);
    }
  }

  auto embed_t::compute_g_weiss(const nda::array<ComplexType, 5> &Gloc_wsIab,
                                const nda::array<ComplexType, 4> &Vhf_imp_sIab,
                                const nda::array<ComplexType, 5> &Sigma_imp_wsIab)
  -> nda::array<ComplexType, 5> {
    auto nImpOrbs = _proj.value().nImpOrbs();
    nda::array<ComplexType, 5> g_wsIab(Gloc_wsIab.shape());
    nda::matrix<ComplexType> tmp_ab(nImpOrbs, nImpOrbs);
    for (size_t w=0; w<Gloc_wsIab.shape(0); ++w) {
      for (size_t s=0; s<_MF->nspin(); ++s) {
        // g(w) = [G(w)^{-1} + Sigma_imp]^{-1]
        nda::matrix_const_view<ComplexType> Gloc = Gloc_wsIab(w, s, 0, nda::ellipsis{});
        tmp_ab = nda::inverse(Gloc);
        tmp_ab += ( Vhf_imp_sIab(s,0,nda::ellipsis{}) + Sigma_imp_wsIab(w,s,0,nda::ellipsis{}) );
        g_wsIab(w,s,0,nda::ellipsis{}) = nda::inverse(tmp_ab);
      }
    }
    return g_wsIab;
  }

  auto embed_t::compute_g_weiss(const nda::array<ComplexType, 5> &Gloc_wsIab,
                                h5::group h5_grp, long weiss_f_iter)
  -> nda::array<ComplexType, 5> {

    nda::array<ComplexType, 4> Vhf_imp_sIab;
    nda::array<ComplexType, 5> Sigma_imp_sIab;

    auto weiss_f_grp = h5_grp.open_group("downfold_1e");
    auto iter_grp = weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter));
    nda::h5_read(iter_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    nda::h5_read(iter_grp, "Sigma_imp_wsIab", Sigma_imp_sIab);

    return compute_g_weiss(Gloc_wsIab, Vhf_imp_sIab, Sigma_imp_sIab);
  }

  auto embed_t::compute_hybridization(const nda::array<ComplexType, 5> &g_wsIab,
                                      const nda::array<ComplexType, 4> &t_sIab,
                                      double mu, imag_axes_ft::IAFT &ft)
  -> nda::array<ComplexType, 5> {
    auto nImpOrbs = _proj.value().nImpOrbs();

    nda::array<ComplexType, 5> delta_wsIab(g_wsIab.shape());
    nda::matrix<ComplexType> ginv(nImpOrbs, nImpOrbs);
    auto eye = nda::eye<ComplexType>(nImpOrbs);
    for (size_t n=0; n<g_wsIab.shape(0); ++n) {
      for (size_t s=0; s<g_wsIab.shape(1); ++s) {
        // delta(n) = (wn+mu)*I - t - g^{-1}
        ginv = g_wsIab(n,s,0,nda::ellipsis{});
        ginv = nda::inverse(ginv);

        ComplexType omega_mu = ft.omega( ft.wn_mesh()(n) ) + mu;
        delta_wsIab(n,s,0,nda::ellipsis{}) = omega_mu*eye - t_sIab(s,0,nda::ellipsis{}) - ginv;
      }
    }
    return delta_wsIab;

  }

} // methods

// instantiation of "public" templates
namespace methods {

template void embed_t::hf_downfolding(std::string, std::string,
  thc_reader_t&, imag_axes_ft::IAFT&, bool, std::string);

}
