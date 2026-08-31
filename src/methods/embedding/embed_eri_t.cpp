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


#include "mpi3/communicator.hpp"

#include <unordered_set>
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/nda_functions.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/GW/g0_div_utils.hpp"
#include "embed_eri_t.h"
#include "cholesky.hpp"

namespace methods {
  template<THC_ERI thc_t>
  auto embed_eri_t::compute_collation_impurity_basis(
      thc_t &thc, const projector_boson_t &proj_boson, nda::range u_rng)
  {
    using Array_6D_t = memory::array<HOST_MEMORY, ComplexType, 6>;
    decltype(nda::range::all) all;

    auto mpi = thc.mpi();
    auto C_skIai = proj_boson.C_skIai();
    auto W_rng = proj_boson.W_rng();
    auto [ns, nkpts, nImps, nImpOrbs, nOrbs_W] = C_skIai.shape();
    auto qsymms = _MF->qsymms();
    auto nsym = qsymms.size();

    // collation matrix (partition over Np by u_rng)
    using comm_t = std::decay_t<decltype(mpi->comm)>;
    memory::darray_t<Array_6D_t,comm_t> dT_skIPa(std::addressof(mpi->comm),
        {1,1,1,1,mpi->comm.size(),1},
        {nsym, ns, nkpts, nImps, thc.Np(), nImpOrbs},
        {nsym, ns, nkpts, nImps, u_rng.size(), nImpOrbs},
        {0,0,0,0,u_rng.first(),0},
        {nsym,ns, nkpts, nImps, u_rng.size(), nImpOrbs});
    auto T_skIPa = dT_skIPa.local();
    nda::array<ComplexType, 2> Cfull_jb(_MF->nbnd(), nImpOrbs);
    nda::array<ComplexType, 2> tmp_ib(_MF->nbnd(), nImpOrbs);

    using math::sparse::T;
    using math::sparse::csrmm;
    for (size_t isym=0; isym<nsym; ++isym) {
      for (size_t isk=0; isk<ns*nkpts; ++isk) {
        size_t is = isk / nkpts;
        size_t ik = isk % nkpts;
        for (size_t I=0; I<nImps; ++I) {
          if (isym==0) {
            // TskI_Pa = conj(Ck_bj) * X_Pj(k)
            nda::blas::gemm(thc.X(is, 0, ik)(u_rng, W_rng[I]),
                            nda::dagger(C_skIai(is, ik, I, nda::ellipsis{})),
                            T_skIPa(isym, is, ik, I, nda::ellipsis{}));
          } else {
            auto [cjg, D_ij] = _MF->symmetry_rotation(isym, ik);
            // D_ij * Cfull_jb = tmp_ib
            Cfull_jb() = 0.0;
            if(not cjg) {
              Cfull_jb(W_rng[I], all) = nda::conj(nda::transpose(C_skIai(is, ik, I, nda::ellipsis{})));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
            } else {
              Cfull_jb(W_rng[I], all) = nda::transpose(C_skIai(is, ik, I, nda::ellipsis{}));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
              tmp_ib = nda::conj(tmp_ib);
            }
            // X_Pi * tmp_ib = TskI_aP
            auto ikR = _MF->ks_to_k(isym, ik);
            nda::blas::gemm(thc.X(is, 0, ikR)(u_rng, all), tmp_ib, T_skIPa(isym, is,ik,I,nda::ellipsis{}));
          }
        }
      }
    }
    return dT_skIPa;
  }

  auto embed_eri_t::downfold_2e_logic(long gw_iter, [[maybe_unused]] long weiss_f_iter, long weiss_b_iter, long embed_iter)
    -> std::tuple<std::string, long> {
    
    app_log(2, "Use default logic to determine the input Green's function for downfolding.\n");
    
    long greens_func_iteration = (embed_iter!=-1)? embed_iter : gw_iter;
    std::string greens_func_source = (embed_iter!=-1)? "embed" : "scf";

    // Explain the default logic to the users 
    if (embed_iter==-1) {
      app_log(1, "  The dataset \"embed\" does not exist in the checkpoint file, \n"
                 "  indicating that this is the initial DMFT iteration based on \n"
                 "  the weakly correlated solution in \"scf/iter{}\". \n\n"
                 "    a) The downfolded Coulomb interactions will be calculated using the Green's function from"
                 " \"{}/iter{}\". \n\n"
                 "    b) All the results will be written to \"downfold_2e/iter{}\".\n",
              gw_iter, greens_func_source, greens_func_iteration, greens_func_iteration+1);
      if (weiss_b_iter!=-1) {
        app_log(1, "");
        app_log(1, "╔═══════════════════════════════════════════════════════╗");
        app_log(1, "║ [ WARNING ]                                           ║");
        app_log(1, "║ \"downfold_2e\" exists before the first DMFT iteration. ║");
        app_log(1, "║ The existing dataset will be overwritten.             ║");
        app_log(1, "║ Check if this is what you want!                       ║");
        app_log(1, "╚═══════════════════════════════════════════════════════╝\n");
      }
    } else {
     app_log(1, "  We are at DMFT iteration {}: \n\n"
                "    a) The downfolded Coulomb interactions will be calculated using the Green's function from"
                " \"{}/iter{}\". \n\n"
                "    b) All the results will be written to \"downfold_2e/iter{}\".\n",
             embed_iter+1, greens_func_source, greens_func_iteration, embed_iter+1);
    }

    return std::make_tuple(greens_func_source, greens_func_iteration);
  }

  void embed_eri_t::check_downfold_2e_logic(std::string greens_func_source, long greens_func_iteration, long gw_iter, long embed_iter) {
    app_log(2, "Checking the dataset in the CoQuí checkpoint file...\n");
      if (greens_func_source == "scf")
        utils::check(greens_func_iteration <= gw_iter and greens_func_iteration >= 0,
                     "downfold_2e logic fail: the input dataset {}/iter{} does not exist!",
                     greens_func_source, greens_func_iteration);
      else if (greens_func_source == "embed")
        utils::check(greens_func_iteration <= embed_iter and greens_func_iteration > 0,
                     "downfold_2e logic fail: the input dataset {}/iter{} does not exist!",
                     greens_func_source, greens_func_iteration);
      else
        utils::check(false, "downfold_2e logic fail: the input dataset {} does not exist!", greens_func_source);
  }

  template<bool return_wt, THC_ERI thc_t>
  auto embed_eri_t::compute_downfolded_coulomb_tensors(
    thc_t &eri, MBState &mb_state, std::string screen_type,
    bool force_permut_symm, bool force_real, imag_axes_ft::IAFT *ft,
    std::string greens_func_source, long greens_func_iteration, 
    bool write_to_hdf5, bool q_dependent_output)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> > {

    // Valid screen types (static to avoid repeated construction)
    static const std::unordered_set<std::string> valid_screen_types = {
      "gw_edmft", "gw_edmft_rpa", "gw_edmft_rpa_density", "gw_edmft_density",
      "rpa", "crpa", "crpa_ks", "crpa_vasp", "crpa_edmft", "crpa_edmft_density"
    };

    // Sanity checks
    utils::check(valid_screen_types.count(screen_type),
      "embed_eri_t::downfolding_wloc: invalid screen_type: {}. "
      "Acceptable options are \"gw_edmft\", \"gw_edmft_rpa\", \"gw_edmft_density\", "
      "\"rpa\", \"crpa\", \"crpa_ks\", \"crpa_vasp\", \"crpa_edmft\", \"crpa_edmft_density\".",
      screen_type);

    utils::check(_context == eri.mpi() && _context == mb_state.mpi,
      "embed_eri_t::downfolding_wloc: eri.mpi() and mb_state.mpi() must be the same as _context.");

    utils::check(mb_state.proj_boson.has_value(),
      "embed_eri_t::downfolding_wloc: mb_state.proj_boson must be set before calling downfolding_wloc.");

    // Add timers
    for (const auto &v : {"DF_TOTAL", "DF_READ", "DF_DOWNFOLD", "DF_SYMM", "DF_WRITE"}) {
      _Timer.add(v);
    }

    // Determine permutation symmetry
    auto permut_symm = determine_permut_symm(force_permut_symm, force_real);

    // Delegate to implementation
    return compute_downfolded_coulomb_tensors_impl<return_wt>(
      eri, mb_state, screen_type, permut_symm, *ft, 
      greens_func_source, greens_func_iteration, 
      write_to_hdf5, q_dependent_output);
  }

  template<bool return_wt>
  auto embed_eri_t::compute_downfolded_coulomb_tensors_impl(
    THC_ERI auto &eri, MBState &mb_state,
    std::string screen_type, std::string permut_symm, const imag_axes_ft::IAFT &ft, 
    std::string greens_func_source, long greens_func_iteration, 
    bool write_to_hdf5, bool q_dependent_output)
  -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> > {

    using math::shm::make_shared_array;

    _Timer.start("DF_TOTAL");

    ft.metadata_log();
    const std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);

    app_log(2, "Checking the dataset in the CoQuí checkpoint file {}...\n", filename);
    // Validate input group and iteration
    if (greens_func_source == "scf") {
      greens_func_iteration = (greens_func_iteration == -1) ? gw_iter : greens_func_iteration;
      utils::check(greens_func_iteration <= gw_iter && greens_func_iteration >= 0,
        "embed_eri_t.cpp::compute_downfolded_coulomb_tensors_impl: downfold_2e logic fail: the input dataset {}/iter{} does not exist!", greens_func_source, greens_func_iteration);
    } else if (greens_func_source == "embed") {
      greens_func_iteration = (greens_func_iteration == -1) ? embed_iter : greens_func_iteration;
      utils::check(greens_func_iteration <= embed_iter && greens_func_iteration > 0,
        "embed_eri_t.cpp::compute_downfolded_coulomb_tensors_impl: downfold_2e logic fail: the input dataset {}/iter{} does not exist!", greens_func_source, greens_func_iteration);
    } else {
      utils::check(false, "embed_eri_t.cpp::compute_downfolded_coulomb_tensors_impl: downfold_2e logic fail: the input dataset {} does not exist!", greens_func_source);
    }
    app_log(2, "Dataset check complete: input dataset {}/iter{} is valid.\n", greens_func_source, greens_func_iteration);

    auto mpi = eri.mpi();
    const auto &proj_boson = mb_state.proj_boson.value();
    const long nImpOrbs = proj_boson.nImpOrbs();

    // Banner gerenated at: http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20two-e%20downfold
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐   ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ││││ │───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ └┴┘└─┘   └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  - CoQui checkpoint file                     = {}", filename);
    app_log(1, "    - Input Green's function ");
    app_log(1, "      HDF5 group                              = {}", greens_func_source);
    app_log(1, "      Iteration                               = {}", greens_func_iteration);
    if (write_to_hdf5) {
      app_log(1, "    - Output HDF5 group                       = {}/iter{}/downfolded_model", greens_func_source, greens_func_iteration);
      if (q_dependent_output)
        app_log(1, "      Stored q-dependent result               = {}", q_dependent_output);
    }
    if (!proj_boson.C_file().empty())
      app_log(1, "  - Transformation matrices                   = {}", proj_boson.C_file());
    app_log(1, "  - Number of impurities                      = {}", proj_boson.nImps());
    app_log(1, "  - Number of local orbitals per impurity     = {}", proj_boson.nImpOrbs());
    app_log(1, "  - Range of primary orbitals for local basis = [{}, {})",
            proj_boson.W_rng()[0].first(), proj_boson.W_rng()[0].last());
    app_log(1, "  - Screening type                            = {}", screen_type);
    app_log(1, "  - Permutation symmetry                      = {}\n", permut_symm);

    // Read Green's function if needed
    _Timer.start("DF_READ");
    if (!mb_state.sG_tskij)
      mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, filename, greens_func_iteration, greens_func_source));
    _Timer.stop("DF_READ");

    // Compute local or q-dependent screened interaction with the choice of screening type
    _Timer.start("DF_DOWNFOLD");
    std::optional<nda::array<ComplexType, 5>> V_qabcd_opt;
    std::optional<nda::array<ComplexType, 6>> U_qwabcd_opt;
    nda::array<ComplexType, 4> V_abcd;
    nda::array<ComplexType, 2> eps_inv_head_wq;
    nda::array<ComplexType, 1> eps_inv_head_w;
    nda::array<ComplexType, 2> pi_head_wq;
    nda::array<ComplexType, 5> U_wabcd;
    if (q_dependent_output == false) {
      std::tie(V_abcd, U_wabcd, eps_inv_head_wq, eps_inv_head_w, pi_head_wq)
          = local_eri_impl<true>(mb_state, eri, ft, screen_type);
    } else {
      std::tie(V_qabcd_opt, U_qwabcd_opt, V_abcd, U_wabcd, 
        eps_inv_head_wq, eps_inv_head_w, pi_head_wq)
        = rpa_q_eri_impl<true>(mb_state, eri, ft, screen_type);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_DOWNFOLD");

    // Enforce permutation symmetries if needed
    int nts_half = (ft.nt_b() % 2 == 0) ? ft.nt_b() / 2 : ft.nt_b() / 2 + 1;
    nda::array<ComplexType, 5> U_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    ft.w_to_tau_PHsym(U_wabcd, U_tabcd);
    if (permut_symm != "none") {
      _Timer.start("DF_SYMM");
      apply_permut_symm(V_abcd, permut_symm, "bare interactions");
      apply_permut_symm(U_tabcd, permut_symm, "screened interactions");
      ft.tau_to_w_PHsym(U_tabcd, U_wabcd);
      _Timer.stop("DF_SYMM");
    }
    ft.check_leakage_PHsym(U_tabcd, imag_axes_ft::boson, std::addressof(mpi->comm), "Local screened interaction");

    // Print summary
    auto [V, Vp, J_pair_bare, J_spin_bare] = orbital_average_int(V_abcd);
    auto [U, Up, J_pair_scr, J_spin_scr] = orbital_average_int(U_wabcd(0, nda::ellipsis{}));
    constexpr double hartree_to_eV = 27.211386245988;
    app_log(1, "\nDownfolded Coulomb Elements Summary");
    app_log(1, "-------------------------------------");
    app_log(1, "bare interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", V * hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", Vp * hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", J_spin_bare * hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV", J_pair_bare * hartree_to_eV);
    app_log(1, "static screened interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", (V + U) * hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", (Vp + Up) * hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", (J_spin_bare + J_spin_scr) * hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV\n", (J_pair_bare + J_pair_scr) * hartree_to_eV);

    mpi->comm.barrier();
    _Timer.stop("DF_TOTAL");
    print_downfold_timers();


    // Optionally write results to HDF5
    if (write_to_hdf5) {
      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
        h5::file file(filename, 'a');
        auto grp = h5::group(file);
        // Find the input group (scf/iter{} or embed/iter{})
        std::string iter_group_name = greens_func_source + "/iter" + std::to_string(greens_func_iteration);
        if (!grp.has_subgroup(iter_group_name)) {
          utils::check(false, "embed_eri_t.cpp::compute_downfolded_coulomb_tensors_impl: Expected iteration group '{}' to exist in HDF5 file, but it does not. This indicates a logic or I/O error!", iter_group_name);
        }
        auto iter_grp = grp.open_group(iter_group_name);
        // Create or open the downfolded_model subgroup
        auto downfolded_model_grp = (iter_grp.has_subgroup("downfolded_model")) ?
          iter_grp.open_group("downfolded_model") : iter_grp.create_group("downfolded_model");
        // Write all downfolded results into downfolded_model
        nda::h5_write(downfolded_model_grp, "C_skIai", proj_boson.C_skIai(), false);
        nda::h5_write(downfolded_model_grp, "Vloc_abcd", V_abcd, false);
        nda::h5_write(downfolded_model_grp, "Uloc_wabcd", U_wabcd, false);
        nda::h5_write(downfolded_model_grp, "eps_inv_head_wq", eps_inv_head_wq, false);
        nda::h5_write(downfolded_model_grp, "eps_inv_head_w", eps_inv_head_w, false);
        nda::h5_write(downfolded_model_grp, "pi_head_wq", pi_head_wq, false);
        h5::h5_write(downfolded_model_grp, "permut_symm", permut_symm);
        h5::h5_write(downfolded_model_grp, "screening_type", screen_type);
        // Write q-dependent tensors if present
        if (q_dependent_output && V_qabcd_opt.has_value() && U_qwabcd_opt.has_value()) {
          nda::h5_write(downfolded_model_grp, "V_qabcd", V_qabcd_opt.value(), false);
          nda::h5_write(downfolded_model_grp, "U_qwabcd", U_qwabcd_opt.value(), false);
        }
      }
      mpi->comm.barrier();
      _Timer.stop("DF_WRITE");
    }

    if constexpr (return_wt)
      return std::make_tuple(V_abcd, U_tabcd);
    else
      return std::make_tuple(V_abcd, U_wabcd);
  }

  template<THC_ERI thc_t>
  void embed_eri_t::downfolding_edmft(
      thc_t &eri, MBState &mb_state, ptree const& pt, std::string screen_type) {

    // Validate screening type
    std::unordered_set<std::string> valid_screen_types = {
        "gw_edmft", "gw_edmft_density", "gw_edmft_rpa", "gw_edmft_rpa_density",
        "gw_edmft_zero_pi_imp", "gw_edmft_zero_pi_imp_density"
    };
    utils::check(valid_screen_types.count(screen_type),
               "embed_eri_t::downfolding: invalid screen_type: {}. ", screen_type);
    // Validate MPI context and projector
    utils::check(_context == eri.mpi() and _context == mb_state.mpi,
                 "embed_eri_t::downfolding_edmft: eri.mpi() and mb_state.mpi() must be the same as _context.");
    utils::check(mb_state.proj_boson.has_value(),
                 "embed_eri_t::downfolding_edmft: mb_state.proj_boson must be set before calling downfolding_edmft.");

    for( auto& v: {"DF_TOTAL", "DF_READ", "DF_DOWNFOLD", "DF_SYMM", "DF_WRITE"} ) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    _Timer.start("DF_READ");

    std::string err_prefix = std::string("downfolding_edmft - Incorrect input - ");
    auto greens_func_source = io::tolower_copy(io::get_value<std::string>(
      pt,"greens_func_source", err_prefix+"\"greens_func_source\" defines the source of input Green's function. "
                                          "Valid options are \"mf\", \"scf\", and \"embed\". "));
    if (greens_func_source == "mf") greens_func_source = "scf";
    auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);

    // Read doulbe-counting Green's function source and iteration
    auto dc_greens_func_source = io::tolower_copy(io::get_value_with_default<std::string>(pt,"dc_greens_func_source", ""));
    auto dc_greens_func_iteration = io::get_value_with_default<long>(pt, "dc_greens_func_iteration", -1);
    if (dc_greens_func_source == "") {
      dc_greens_func_source = greens_func_source;
      dc_greens_func_iteration = greens_func_iteration;
    }

    std::array<std::string, 2> g_grp{greens_func_source, dc_greens_func_source};
    std::array<long, 2> g_iter{greens_func_iteration, dc_greens_func_iteration};

    auto permut_symm = determine_permut_symm(
        io::get_value_with_default<bool>(pt, "permut_symm", true),
        io::get_value_with_default<bool>(pt, "force_real", true) );

    std::array<double, 2> mixing{io::get_value_with_default<double>(pt,"dc_pi_mixing",1.0),
                                 io::get_value_with_default<double>(pt,"u_weiss_mixing",1.0)};

    _Timer.stop("DF_READ");

    utils::check(_output_type == "default",
                 "Error in eri::downfolding: edmft_downfolding is only available with output_type=default.");

    downfold_edmft_impl(eri, mb_state, screen_type, permut_symm, g_grp, g_iter, mixing);
  }

  template<THC_ERI thc_t>
  void embed_eri_t::downfolding_crpa(
      thc_t &eri, MBState &mb_state, ptree const& pt,
      std::string screen_type,
      std::string factorization_type, double thresh) {

    std::unordered_set<std::string> valid_screen_types = {
      "bare", "crpa", "crpa_ks", "crpa_vasp", "crpa_edmft", "crpa_edmft_density"
    };
    utils::check(valid_screen_types.count(screen_type),
                 "embed_eri_t::downfolding_crpa: invalid screen_type: {}. "
                 "Acceptable options are \"bare\", \"crpa\", \"crpa_ks\", \"crpa_vasp\", "
                 "\"crpa_edmft\", \"crpa_edmft_density\".", screen_type);
    utils::check(_context == eri.mpi() and _context == mb_state.mpi,
                 "embed_eri_t::downfolding_crpa: eri.mpi() and mb_state.mpi() must be the same as _context.");
    utils::check(mb_state.proj_boson.has_value(),
                 "embed_eri_t::downfolding_crpa: mb_state.proj_boson must be set before calling downfolding_crpa.");;

    for( auto& v: {"DF_TOTAL", "DF_READ", "DF_DOWNFOLD", "DF_SYMM", "DF_WRITE"} ) {
      _Timer.add(v);
    }

    _Timer.start("DF_TOTAL");
    _Timer.start("DF_READ");

    // Read input parameters
    const std::string err_prefix = "downfolding_crpa - Incorrect input - ";
    auto& proj_boson = mb_state.proj_boson.value();
    long nImpOrbs = proj_boson.nImpOrbs();

    auto greens_func_source = io::tolower_copy(io::get_value<std::string>(
        pt,"greens_func_source", err_prefix + " greens_func_source. This parameter defines the source of input Green's function. "
                                              "Valid types are \"\", \"mf\", \"scf\", and \"embed\". "));
    if (greens_func_source=="mf") greens_func_source = "scf";
    auto greens_func_iteration = io::get_value_with_default<long>(pt, "greens_func_iteration", -1);
    auto q_dependent_output = io::get_value_with_default<bool>(pt,"q_dependent_output", false);

    auto permut_symm = determine_permut_symm(
        io::get_value_with_default<bool>(pt, "permut_symm", true),
        io::get_value_with_default<bool>(pt, "force_real", true));

    _Timer.stop("DF_READ");

    // switch to cholesky_from_4index in small problems. embed_cholesky requires nproc >= nImp*nImp
    if( (factorization_type == "cholesky" or factorization_type == "cholesky_high_memory" ) and
        (eri.mpi()->comm.size() > nImpOrbs*nImpOrbs ) ) // or nImpOrbs < 16) )
    {
      app_log(0," Requested cholesky decomposition of downfolded potential. For small problems or too many processors, we require factorization_type=cholesky_from_4index. Switching to this mode.");
      factorization_type = "cholesky_from_4index";
    }
    // check that no factorization is requested when _output_type == "default"
    if(_output_type == "default")
      utils::check(factorization_type == "none", "Error in eri::downfolding: output_type=default requires factorization_type == none.");
    if(factorization_type.substr(0,8) == "cholesky" or factorization_type.substr(0,3) == "thc")
      utils::check(_output_type.substr(0,5) == "model",
                   "Error: factorization_type=cholesky and thc require output_type=model");

    // dispatch based on screen_type and factorization_type
    if (screen_type == "bare") {
      utils::check(not q_dependent_output, "Error: q_dependent_output = true not implemented with bare interaction.");
      downfold_bare_impl(mb_state.coqui_prefix, eri, mb_state.proj_boson.value(),
                         factorization_type, permut_symm, thresh);
    } else {
     if( _output_type == "default" ) {
       downfold_crpa_impl(eri, mb_state, screen_type, factorization_type, permut_symm,
                          greens_func_source, greens_func_iteration, q_dependent_output, thresh);
      } else if( _output_type == "model_static" ) {
        utils::check(screen_type=="crpa", "Error in eri.downfolding: factorization_type=cholesky requires screen_type=bare or crpa. ");
        utils::check(not q_dependent_output, "Error in eri.downfolding: q_dependent_output = true is not yet implemented with factorization_type=cholesky.. ");
        downfold_screen_model_impl(eri, mb_state, screen_type, factorization_type, permut_symm,
                                   greens_func_source, greens_func_iteration, thresh);
      } else {
        APP_ABORT("Error in eri::downfolding: Invalid output_type: {}",_output_type);
      }
    }
  }

  void embed_eri_t::downfold_bare_impl(std::string output,
                                       THC_ERI auto &eri,
                                       const projector_boson_t &proj_boson,
                                       std::string factorization_type,
                                       std::string permut_symm,
                                       double thresh) {
    decltype(nda::range::all) all;
    auto mpi = eri.mpi();
    auto nImpOrbs = proj_boson.nImpOrbs();
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(output + ".mbpt.h5");
    auto [g_grp, g_iter] = downfold_2e_logic(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter);

    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20two-e%20downfold
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐   ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ││││ │───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ └┴┘└─┘   └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  CoQuí checkpoint file:                    {}", output + ".mbpt.h5");
    if (_output_type.substr(0,5) != "model")
      app_log(1, "    - Output HDF5 group:                     downfold_2e/iter{}",(embed_iter==-1)? g_iter+1 : embed_iter+1);
    else 
      app_log(1, "  Output file with model hamiltonian:        {}", output+".model.h5");
    app_log(1, "");
    app_log(1, "  Transformation matrices:                   {}", proj_boson.C_file());
    app_log(1, "  Number of impurities:                      {}", proj_boson.nImps());
    app_log(1, "  Number of local orbitals per impurity:     {}", proj_boson.nImpOrbs());
    app_log(1, "  Range of primary orbitals for local basis: [{}, {})",
            proj_boson.W_rng()[0].first(), proj_boson.W_rng()[0].last());
    app_log(1, "");
    app_log(1, "  factorization type:                        {}", factorization_type);
    app_log(1, "  permutation symmetry:                      {}\n", permut_symm);


    if(factorization_type == "none" or factorization_type == "cholesky_from_4index") {

      _Timer.start("DF_DOWNFOLD");
      // Construct B matrix from the local basis to product basis from the thc class
      auto sB_qIPab = (_MF->nqpts_ibz()==_MF->nqpts())?
          proj_boson.calc_bosonic_projector(eri) : proj_boson.calc_bosonic_projector_symm(eri);
      auto B_qIPab = sB_qIPab.local();
      auto V_abcd = downfold_V(eri, B_qIPab); 
      _Timer.stop("DF_DOWNFOLD");

      // enforce permutation symmetries
      if (permut_symm!="none") {
        _Timer.start("DF_SYMM");
        apply_permut_symm(V_abcd, permut_symm, "bare interactions");
        _Timer.stop("DF_SYMM");
      }

      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
        if(_output_type == "default") {
          weiss_b_iter = (embed_iter==-1)? g_iter+1 : embed_iter+1;
          std::string filename = output + ".mbpt.h5";
          h5::file file(filename, 'a');
          auto grp = h5::group(file);
          auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                            grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
          auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter)))?
                        downfold_grp.open_group("iter"+std::to_string(weiss_b_iter)) :
                        downfold_grp.create_group("iter"+std::to_string(weiss_b_iter));

          h5::h5_write(downfold_grp, "factorization_type", factorization_type);
          nda::h5_write(iter_grp, "Vloc_abcd", V_abcd, false);
          h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter);
          nda::h5_write(downfold_grp, "C_skIai", proj_boson.C_skIai(), false);
          h5::h5_write(iter_grp, "permut_symm", permut_symm);
        } else if(_output_type == "model_static") {
          {
            // write basic info to downfold_2e
            weiss_b_iter = (embed_iter==-1)? g_iter+1 : embed_iter+1;
            std::string filename = output + ".mbpt.h5";
            h5::file file(filename, 'a');
            auto grp = h5::group(file);
            auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                              grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
            auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter)))?
                          downfold_grp.open_group("iter"+std::to_string(weiss_b_iter)) :
                          downfold_grp.create_group("iter"+std::to_string(weiss_b_iter));

            h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter);
          }
          std::string filename = output + ".model.h5";
          h5::file file(filename, 'a');
          auto grp = h5::group(file);
          auto sgrp = grp.create_group("Interaction");
          
          if(factorization_type == "cholesky_from_4index") { 
            nda::array<int,1> piv(nImpOrbs*nImpOrbs+1);
            auto V3D = nda::reshape(V_abcd,std::array<long,3>{nImpOrbs*nImpOrbs,nImpOrbs,nImpOrbs});
            auto V2D = nda::reshape(V_abcd,std::array<long,2>{nImpOrbs*nImpOrbs,nImpOrbs*nImpOrbs});
            { // transpose into hermitian matrix
              nda::array<ComplexType,1> T(nImpOrbs*nImpOrbs);
              for( int a=0; a<nImpOrbs; ++a )
                for( int b=a+1; b<nImpOrbs; ++b ) {
                  T(all) = V3D(all,a,b);
                  V3D(all,a,b) = V3D(all,b,a);
                  V3D(all,b,a) = T();
                }
            }
            using U_type = nda::array<ComplexType,2>;
            auto U = utils::chol<true,U_type>(V2D,piv,thresh,true);
            // conjugate to get L*conj(L)
            U() = nda::conj(U());
            int nchol = U.extent(0);
            auto Vq = nda::reshape(U,std::array<long,5>{nchol,1,1,nImpOrbs,nImpOrbs});
            methods::chol_reader_t::add_meta_data(sgrp,nchol,thresh,1,1,1,nImpOrbs,
                nda::array<double, 2>::zeros({1,3}),nda::array<double, 2>::zeros({1,3}),
                nda::array<int, 2>::zeros({1,1}));
            h5::h5_write(sgrp, "factorization_type", "cholesky");
            nda::h5_write(sgrp, "Vq0", Vq, false);
          } else { 
            // not clear what this is yet, since it can't be read by anything right now!
            h5::h5_write(sgrp, "factorization_type", factorization_type);
            nda::h5_write(sgrp, "Vq0", V_abcd, false);
          } 
        } else {
          APP_ABORT("Unknown output_type:{} in downfold_bare",_output_type);
        }
      }
      mpi->comm.barrier();
      _Timer.stop("DF_WRITE");

    } else if(factorization_type == "cholesky" or 
              factorization_type == "cholesky_high_memory" )  {

      _Timer.start("DF_DOWNFOLD");
      auto dV_qPQ = eri.dZ({1, 1, mpi->comm.size()});
      app_log(1, "Treatment of long-wavelength divergence in V (bare): {}", _bare_div_treatment);
      auto div_factor = ( _bare_div_treatment == "ignore_g0") ? ComplexType(0.0) : ComplexType(1.0);
      auto V_nab = ( factorization_type=="cholesky" ? 
                     downfold_cholesky(eri, proj_boson, dV_qPQ, div_factor, thresh) :
                     downfold_cholesky_high_memory(eri, proj_boson, dV_qPQ, div_factor, thresh) );
      dV_qPQ.reset(); // return memory, since this can be large
      _Timer.stop("DF_DOWNFOLD");

      // enforce permutation symmetries
      if (permut_symm!="none" and mpi->comm.root())
        app_warning(" downfold_2e: Skipping application of permutation symmetry with factorization type = cholesky");

      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
        // write basic info to downfold_2e
        {
          weiss_b_iter = (embed_iter==-1)? g_iter+1 : embed_iter+1;
          std::string filename = output + ".mbpt.h5";
          h5::file file(filename, 'a');
          auto grp = h5::group(file);
          auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                            grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
          auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter)))?
                        downfold_grp.open_group("iter"+std::to_string(weiss_b_iter)) :
                        downfold_grp.create_group("iter"+std::to_string(weiss_b_iter));

          h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter);
        }
        std::string filename = output + ".model.h5";
        h5::file file(filename, 'a');
        auto grp = h5::group(file);
        auto sgrp = grp.create_group("Interaction");

        // temporary hack to make convention consistent wirh ERI::cholesky
        int nchol = V_nab.global_shape()[0];
        methods::chol_reader_t::add_meta_data(sgrp,nchol,thresh,1,1,1,nImpOrbs,
                nda::array<double, 2>::zeros({1,3}),nda::array<double, 2>::zeros({1,3}),
                nda::array<int, 2>::zeros({1,1}));
        h5::h5_write(sgrp, "factorization_type", factorization_type.substr(0,8));
        write_cholesky_embed(sgrp,"Vq0",V_nab,true,true);
      } else {
        auto grp = h5::group();
        write_cholesky_embed(grp,"Vq0",V_nab,true,true);
      }
      mpi->comm.barrier();
      _Timer.stop("DF_WRITE");

    } else if(factorization_type == "thc") {

      utils::check(_MF->nqpts() == 1, 
        "Error in downfold_2e: factorization_type=thc only available for single k-point calculations.");
      APP_ABORT("Error: factorization_type = thc not yet available.");

      _Timer.start("DF_DOWNFOLD");
      auto [X_au,V_uv] = downfold_V_thc(eri,thresh);
      _Timer.stop("DF_DOWNFOLD");

      // enforce permutation symmetries
      if (permut_symm!="none" and mpi->comm.root())
        app_warning(" downfold_2e: Skipping application of permutation symmetry with factorization type = thc");

      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
      }
      mpi->comm.barrier();
      _Timer.stop("DF_WRITE");

    } else 
      APP_ABORT("Error in downfold_2e: Unknown factorization_type: {}",factorization_type);

    _Timer.stop("DF_TOTAL");
    print_downfold_timers();

  }

  void embed_eri_t::downfold_edmft_impl(
      THC_ERI auto &eri, MBState &mb_state,
      std::string screen_type,
      std::string permut_symm,
      std::array<std::string, 2> g_grp,
      std::array<long, 2> g_iter,
      std::array<double, 2> mixing) {

    using math::shm::make_shared_array;

    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    // checking g_grp and g_iter exist in chkpt h5.
    if (g_iter[0]!=-1)
      check_downfold_2e_logic(g_grp[0], g_iter[0], gw_iter, embed_iter);
    else
      g_iter[0] = (g_grp[0]=="scf")? gw_iter : embed_iter;

    if (g_iter[1]!=-1)
      check_downfold_2e_logic(g_grp[1], g_iter[1], gw_iter, embed_iter);
    else
      g_iter[1] = (g_grp[1]=="scf")? gw_iter : embed_iter;

    auto mpi = eri.mpi();
    auto& ft = *mb_state.ft;
    auto& proj_boson = mb_state.proj_boson.value();
    long nImpOrbs = proj_boson.nImpOrbs();

    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20two-e%20downfold
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐   ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ││││ │───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ └┴┘└─┘   └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  QoQuí checkpoint file:                     {}", filename);
    app_log(1, "    - Input Green's function for rpa and dc polarizability ");
    app_log(1, "      HDF5 group:                            {}, {}", g_grp[0], g_grp[1]);
    app_log(1, "      iteration:                             {}, {}", g_iter[0], g_iter[1]);
    app_log(1, "    - Output HDF5 group:                     downfold_2e/iter{}", g_iter[1]+1);
    app_log(1, "");
    app_log(1, "  Transformation matrices:                   {}", proj_boson.C_file());
    app_log(1, "  Number of impurities:                      {}", proj_boson.nImps());
    app_log(1, "  Number of local orbitals per impurity:     {}", proj_boson.nImpOrbs());
    app_log(1, "  Range of primary orbitals for local basis: [{}, {})",
            proj_boson.W_rng()[0].first(), proj_boson.W_rng()[0].last());
    app_log(1, "");
    app_log(1, "  Screening type:                            {}", screen_type);
    app_log(1, "  Permutation symmetry:                      {}\n", permut_symm);
    ft.metadata_log();

    _Timer.start("DF_READ");
    // Check status of MBState
    // Read MB data for the evaluation of RPA polarizability
    if (!mb_state.sG_tskij)
      mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, filename, g_iter[0], g_grp[0]));
    auto& sG_tskij = mb_state.sG_tskij.value();

    bool density_only = (screen_type.find("density")==std::string::npos)? false : true;

    bool pi_local_given = (!mb_state.sPi_imp_wabcd or !mb_state.sPi_dc_wabcd)?
        mb_state.read_local_polarizabilities(weiss_b_iter) : true;

    if (pi_local_given) {
      auto& sPi_imp_wabcd = mb_state.sPi_imp_wabcd.value();
      int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
      nda::array<ComplexType, 5> X_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
      ft.w_to_tau_PHsym(sPi_imp_wabcd.local(), X_tabcd);
      ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson,
                           std::addressof(mpi->comm), "impurity polarizability");

      if (screen_type.find("zero_pi_imp")!=std::string::npos) {
        app_log(2, "Ignoring the input impurity polarizability since scree_type = {}\n", screen_type);
        mb_state.sPi_imp_wabcd.value().set_zero();
      }
    } else if (screen_type.find("zero_pi_imp")!=std::string::npos) {
      app_log(2, "Screen_type = {}", screen_type);
      app_log(2, "-> Evaluate the fully screened interaction with no local correction.\n");
    }
    mpi->comm.barrier();
    _Timer.stop("DF_READ");

    _Timer.start("DF_DONWFOLD");
    // evaluate local screened interaction W(iw) with given screen_type
    auto [V_abcd, W_wabcd, eps_inv_head_wq, eps_inv_head_w, pi_head_wq] =
        local_eri_impl<true>(mb_state, eri, ft, screen_type);

    nda::array<ComplexType, 5> U_wabcd(W_wabcd.shape());
    if (screen_type.find("zero_pi_imp")!=std::string::npos) {

      app_log(2, "Setting the bosonic Weiss field to Wloc\n");
      U_wabcd() = W_wabcd;

    } else if (screen_type.find("gw_edmft_rpa")!=std::string::npos or not pi_local_given) {

      auto G_tsIab = (permut_symm=="8-fold")?
          proj_boson.proj_fermi().downfold_loc<true>(sG_tskij, "Gloc for DC polarizability") :
          proj_boson.proj_fermi().downfold_loc<false>(sG_tskij, "Gloc for DC polarizability");

      auto sU_wabcd = u_bosonic_weiss_rpa(G_tsIab, W_wabcd, V_abcd, ft, density_only);
      U_wabcd() = sU_wabcd.local();

    } else {

      auto& sPi_imp_wabcd = mb_state.sPi_imp_wabcd.value();
      u_bosonic_weiss_edmft_in_place(W_wabcd, V_abcd, sPi_imp_wabcd);
      U_wabcd() = sPi_imp_wabcd.local();

      long nts = ft.nt_b();
      long nts_half = (nts%2==0)? nts/2 : nts/2+1;
      nda::array<ComplexType, 5> X_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
      ft.w_to_tau_PHsym(U_wabcd, X_tabcd);
      ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(mpi->comm), "bosonic weiss field");

    }

    // mixing bosonic Weiss fields
    if (mixing[1] < 1.0 and weiss_b_iter!=-1) {
      app_log(1, "\nMixing the bosonic Weiss field with the previous iteration: {}\n", mixing[1]);
      h5::file file(filename, 'r');
      auto df_2e_grp = h5::group(file).open_group("downfold_2e");
      auto iter_2e_grp = df_2e_grp.open_group("iter" + std::to_string(weiss_b_iter));
      nda::array<ComplexType, 5> U_wabcd_prev;
      nda::h5_read(iter_2e_grp, "Uloc_wabcd", U_wabcd_prev);
      U_wabcd *= mixing[1];
      U_wabcd += (1 - mixing[1]) * U_wabcd_prev;
    }
    mpi->comm.barrier();

    app_log(1, "\nEvaluating double counting polarizability with\n"
             "  - Gloc from {}/iter{}\n"
             "  - density-density only: {}",
          g_grp[1], g_iter[1], density_only);
    if (g_grp[0] != g_grp[1] or g_iter[0] != g_iter[1]) {
      mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, filename, g_iter[1], g_grp[1]));
    }
    auto& sG_tskij_dc = mb_state.sG_tskij.value();
    auto G_tsIab = (permut_symm=="8-fold")?
        proj_boson.proj_fermi().downfold_loc<true>(sG_tskij_dc, "Gloc for DC polarizability") :
        proj_boson.proj_fermi().downfold_loc<false>(sG_tskij_dc, "Gloc for DC polarizability");
    auto sPi_dc_wabcd_new = eval_Pi_rpa_dc<true>(*mpi, G_tsIab, ft, density_only);

    // mixing pi_dc with the previous value
    if (mb_state.sPi_dc_wabcd and mixing[0] < 1.0) {
      app_log(1, "  - Mixing for the current iteration: {}\n", mixing[0]);
      utils::check(mb_state.sPi_dc_wabcd->shape() == sPi_dc_wabcd_new.shape(),
                   "Error in downfold_2e: shape of Pi_dc given in MBState is different from the newly evaluated one.");
      sPi_dc_wabcd_new.win().fence();
      if (mpi->node_comm.root()) {
        auto Pi_dc = mb_state.sPi_dc_wabcd.value().local();
        auto Pi_dc_new = sPi_dc_wabcd_new.local();
        Pi_dc_new *= mixing[0];
        Pi_dc_new += (1 - mixing[0]) * Pi_dc;
      }
      sPi_dc_wabcd_new.win().fence();

    } else if (not mb_state.sPi_dc_wabcd and mixing[0] < 1.0) {
      app_log(1, "  - dc_pi_mixing < 1.0 while no Pi_dc is found in MBState. Ignoring mixing.\n");
    }
    mpi->comm.barrier();

    _Timer.stop("DF_DOWNFOLD");

    // enforce permutation symmetries
    if (permut_symm!="none") {
      _Timer.start("DF_SYMM");
      apply_permut_symm(V_abcd, permut_symm, "bare interactions");

      int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
      nda::array<ComplexType, 5> tmp_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);

      ft.w_to_tau_PHsym(U_wabcd, tmp_tabcd);
      apply_permut_symm(tmp_tabcd, permut_symm, "Bosonic weiss field");
      ft.tau_to_w_PHsym(tmp_tabcd, U_wabcd);

      ft.w_to_tau_PHsym(W_wabcd, tmp_tabcd);
      apply_permut_symm(tmp_tabcd, permut_symm, "screened interactions");
      ft.tau_to_w_PHsym(tmp_tabcd, W_wabcd);
      _Timer.stop("DF_SYMM");
    }

    auto[V, Vp, J_pair_bare, J_spin_bare] = orbital_average_int(V_abcd);
    auto[U, Up, J_pair_scr, J_spin_scr] = orbital_average_int(U_wabcd(0,nda::ellipsis{}));

    double hartree_to_eV = 27.211386245988;
    app_log(1, "\ndownfold_2e summary");
    app_log(1, "-------------------");
    app_log(1, "bare interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", V*hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", Vp*hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", J_spin_bare*hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV", J_pair_bare*hartree_to_eV);
    app_log(1, "static screened interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", (V+U)*hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", (Vp+Up)*hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", (J_spin_bare+J_spin_scr)*hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV\n", (J_pair_bare+J_pair_scr)*hartree_to_eV);

    _Timer.start("DF_WRITE");
    if (mpi->comm.root()) {
      long weiss_b_iter_ = g_iter[1]+1;
      h5::file file(filename, 'a');
      auto grp = h5::group(file);
      auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                        grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
      auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter_)))?
                    downfold_grp.open_group("iter"+std::to_string(weiss_b_iter_)) :
                    downfold_grp.create_group("iter"+std::to_string(weiss_b_iter_));

      h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter_);
      nda::h5_write(downfold_grp, "C_skIai", proj_boson.C_skIai(), false);
      h5::h5_write(iter_grp, "greens_func_source", g_grp[0]);
      h5::h5_write(iter_grp, "pi_rpa_iter", g_iter[0]);
      h5::h5_write(iter_grp, "pi_dc_input", g_grp[1]);
      h5::h5_write(iter_grp, "pi_dc_iter", g_iter[1]);
      nda::h5_write(iter_grp, "Gloc_tsIab", G_tsIab, false);
      nda::h5_write(iter_grp, "Wloc_wabcd", W_wabcd, false);
      nda::h5_write(iter_grp, "Vloc_abcd", V_abcd, false);
      nda::h5_write(iter_grp, "Uloc_wabcd", U_wabcd, false); // optional
      h5::h5_write(iter_grp, "screening_type", screen_type); // optional
      nda::h5_write(iter_grp, "Pi_dc_wabcd", sPi_dc_wabcd_new.local(), false); // optional
      h5::h5_write(iter_grp, "permut_symm", permut_symm);
      nda::h5_write(iter_grp, "eps_inv_head_wq", eps_inv_head_wq, false);
      nda::h5_write(iter_grp, "eps_inv_head_w", eps_inv_head_w, false);
      nda::h5_write(iter_grp, "pi_head_wq", pi_head_wq, false);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_WRITE");
    _Timer.stop("DF_TOTAL");
    print_downfold_timers();
  }

  void embed_eri_t::downfold_crpa_impl(THC_ERI auto &eri, MBState &mb_state, std::string screen_type,
                                       [[maybe_unused]] std::string factorization_type,
                                       std::string permut_symm,
                                       std::string greens_func_source, long greens_func_iteration,
                                       bool q_dependent_output,
                                       [[maybe_unused]] double thresh) {
    using math::shm::make_shared_array;

    utils::check(screen_type.substr(0,4)=="crpa", "embed_eri_t::downfold_crpa_impl: invalid screen_type = {}.", screen_type);    

    auto& ft = *mb_state.ft;
    bool density_only = (screen_type.find("density")==std::string::npos)? false : true;

    ft.metadata_log();
    std::string filename = mb_state.coqui_prefix + ".mbpt.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(filename);
    if (greens_func_source == "" or greens_func_iteration == -1)
      std::tie(greens_func_source, greens_func_iteration) = downfold_2e_logic(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter);
    else
      check_downfold_2e_logic(greens_func_source, greens_func_iteration, gw_iter, embed_iter);

    auto mpi = eri.mpi();
    auto& proj_boson = mb_state.proj_boson.value();
    long nImpOrbs = proj_boson.nImpOrbs();


    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20two-e%20downfold
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐   ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ││││ │───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ └┴┘└─┘   └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  CoQuí checkpoint file:                    {}", filename);
    app_log(1, "    - Input Green's function: ");
    app_log(1, "      HDF5 group:                            {}", greens_func_source);
    app_log(1, "      iteration:                             {}", greens_func_iteration);
    app_log(1, "    - Output HDF5 group:                     downfold_2e/iter{}", greens_func_iteration+1);
    app_log(1, "");
    app_log(1, "  Transformation matrices:                   {}", proj_boson.C_file());
    app_log(1, "  Number of impurities:                      {}", proj_boson.nImps());
    app_log(1, "  Number of local orbitals per impurity:     {}", proj_boson.nImpOrbs());
    app_log(1, "  Range of primary orbitals for local basis: [{}, {})",
            proj_boson.W_rng()[0].first(), proj_boson.W_rng()[0].last());
    app_log(1, "");
    app_log(1, "  Screening type:                            {}", screen_type);
    app_log(1, "  factorization type:                        {}", factorization_type);
    app_log(1, "  permutation symmetry:                      {}\n", permut_symm);
  

    _Timer.start("DF_READ");

    // Check status of MBState
    if (!mb_state.sG_tskij)
      mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, filename, greens_func_iteration, greens_func_source));
    auto& sG_tskij = mb_state.sG_tskij.value();

    bool pi_local_given = false;
    if (!mb_state.sPi_imp_wabcd or !mb_state.sPi_dc_wabcd) {
      // 1) if the pi_imp and pi_dc are not set, we read them from the file
      // 2) if the file does not contain them, we set them to empty arrays
      long nw = ft.nw_b();
      long nw_half = (nw%2==0)? nw/2 : nw/2+1;
      mb_state.sPi_imp_wabcd.emplace(make_shared_array<Array_view_5D_t>(
          *mpi, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs}));
      mb_state.sPi_dc_wabcd.emplace(make_shared_array<Array_view_5D_t>(
          *mpi, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs}));
      pi_local_given = chkpt::read_pi_local(mb_state.sPi_imp_wabcd.value(),
                                            mb_state.sPi_dc_wabcd.value(),
                                            filename, weiss_b_iter);
    } else {
      pi_local_given = true;
    }
    mpi->node_comm.broadcast_n(&pi_local_given, 1, 0);

    auto& sPi_imp_wabcd = mb_state.sPi_imp_wabcd.value();
    if (pi_local_given) {
      int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
      nda::array<ComplexType, 5> X_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
      ft.w_to_tau_PHsym(sPi_imp_wabcd.local(), X_tabcd);
      ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(mpi->comm), "impurity polarizability");
    }
    mpi->comm.barrier();

    _Timer.stop("DF_READ");

    _Timer.start("DF_DONWFOLD");

    std::optional<nda::array<ComplexType, 5>> V_qabcd_opt;
    std::optional<nda::array<ComplexType, 6>> U_wqabcd_opt;
    nda::array<ComplexType, 4> V_abcd;
    nda::array<ComplexType,2> eps_inv_head_wq;
    nda::array<ComplexType,1> eps_inv_head_w;
    nda::array<ComplexType,2> pi_head_wq;
    nda::array<ComplexType, 5> U_wabcd;
    // evaluate partially screened interaction u(iw) based on cRPA equations
    if (q_dependent_output == false) {
      std::tie(V_abcd, U_wabcd, eps_inv_head_wq, eps_inv_head_w, pi_head_wq)
          = local_eri_impl<true>(mb_state, eri, ft, screen_type);
    } else {
      std::tie(V_qabcd_opt, U_wqabcd_opt, V_abcd, U_wabcd, 
        eps_inv_head_wq, eps_inv_head_w, pi_head_wq)
        = rpa_q_eri_impl<true>(mb_state, eri, ft, screen_type);
    }
    auto G_tsIab = (permut_symm=="8-fold")?
                proj_boson.proj_fermi().downfold_loc<true>(sG_tskij, "Gloc") :
                proj_boson.proj_fermi().downfold_loc<false>(sG_tskij, "Gloc");

    app_log(1, "\nEvaluating double counting polarizability with\n"
               "  - Gloc from {}/iter{}\n"
               "  - density-density only: {}\n",
            greens_func_source, greens_func_iteration, density_only);
    auto sPi_dc_wabcd_new = eval_Pi_rpa_dc<true>(*mpi, G_tsIab, ft, density_only);
    mpi->comm.barrier();


    _Timer.stop("DF_DOWNFOLD");

    // enforce permutation symmetries
    _Timer.start("DF_SYMM");

    if (permut_symm!="none") {
      apply_permut_symm(V_abcd, permut_symm, "bare interactions");

      int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
      nda::array<ComplexType, 5> tmp_tabcd(nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);

      ft.w_to_tau_PHsym(U_wabcd, tmp_tabcd);
      apply_permut_symm(tmp_tabcd, permut_symm, "Bosonic weiss field");
      ft.tau_to_w_PHsym(tmp_tabcd, U_wabcd);

    }

    _Timer.stop("DF_SYMM");

    auto[V, Vp, J_pair_bare, J_spin_bare] = orbital_average_int(V_abcd);
    auto[U, Up, J_pair_scr, J_spin_scr] = orbital_average_int(U_wabcd(0,nda::ellipsis{}));

    double hartree_to_eV = 27.211386245988;
    app_log(1, "\ndownfold_2e summary (screening = {})", screen_type);
    app_log(1, "-------------------");
    app_log(1, "bare interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", V*hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", Vp*hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", J_spin_bare*hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV", J_pair_bare*hartree_to_eV);
    app_log(1, "static screened interactions (orbital-average):");
    app_log(1, "  - intra-orbital = {} eV", (V+U)*hartree_to_eV);
    app_log(1, "  - inter-orbital = {} eV", (Vp+Up)*hartree_to_eV);
    app_log(1, "  - Hund's coupling (spin-flip) = {} eV", (J_spin_bare+J_spin_scr)*hartree_to_eV);
    app_log(1, "  - Hund's coupling (pair-hopping) = {} eV\n", (J_pair_bare+J_pair_scr)*hartree_to_eV);

    _Timer.start("DF_WRITE");

    if (mpi->comm.root()) {
      long weiss_b_iter_ = greens_func_iteration+1;
      h5::file file(filename, 'a');
      auto grp = h5::group(file);
      auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                          grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
      auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter_)))?
                      downfold_grp.open_group("iter"+std::to_string(weiss_b_iter_)) :
                      downfold_grp.create_group("iter"+std::to_string(weiss_b_iter_));

      h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter_);
      nda::h5_write(downfold_grp, "C_skIai", proj_boson.C_skIai(), false);
      nda::h5_write(iter_grp, "Gloc_tsIab", G_tsIab, false);
      h5::h5_write(iter_grp, "greens_func_source", greens_func_source);
      h5::h5_write(iter_grp, "greens_func_iteration", greens_func_iteration);

      if (V_qabcd_opt) nda::h5_write(iter_grp, "V_qabcd", V_qabcd_opt.value(), false);
      if (U_wqabcd_opt) nda::h5_write(iter_grp, "U_wqabcd", U_wqabcd_opt.value(), false);

      nda::h5_write(iter_grp, "Vloc_abcd", V_abcd, false);
      nda::h5_write(iter_grp, "Uloc_wabcd", U_wabcd, false);
      h5::h5_write(iter_grp, "screening_type", screen_type);
      h5::h5_write(iter_grp, "permut_symm", permut_symm);
      nda::h5_write(iter_grp, "Pi_dc_wabcd", sPi_dc_wabcd_new.local(), false); // optional
      nda::h5_write(iter_grp, "eps_inv_head_wq", eps_inv_head_wq, false);
      nda::h5_write(iter_grp, "eps_inv_head_w", eps_inv_head_w, false);
      nda::h5_write(iter_grp, "pi_head_wq", pi_head_wq, false);
    }
    mpi->comm.barrier();

    _Timer.stop("DF_WRITE");
    _Timer.stop("DF_TOTAL");

    print_downfold_timers();
  }

  void embed_eri_t::downfold_screen_model_impl(
      THC_ERI auto &eri, MBState &mb_state, std::string screen_type,
      std::string factorization_type, std::string permut_symm,
      std::string greens_func_source, long greens_func_iteration, double thresh) {
    using math::shm::make_shared_array;
    utils::check(screen_type.substr(0,4)=="crpa", "downfold_crpa_impl: invalid screen_type = {}.", screen_type);

    auto& ft = *mb_state.ft;
    ft.metadata_log();
    std::string input_file = mb_state.coqui_prefix + ".mbpt.h5";
    std::string output_file = mb_state.coqui_prefix + ".model.h5";
    auto [gw_iter, weiss_f_iter, weiss_b_iter, embed_iter] = chkpt::read_input_iterations(input_file);
    if (greens_func_source == "" or greens_func_iteration == -1)
      std::tie(greens_func_source, greens_func_iteration) = downfold_2e_logic(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter);
    else
      check_downfold_2e_logic(greens_func_source, greens_func_iteration, gw_iter, embed_iter);

    auto mpi = eri.mpi();
    auto& proj_boson = mb_state.proj_boson.value();
    long nImpOrbs = proj_boson.nImpOrbs();


    // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20two-e%20downfold
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐   ┌─┐  ┌┬┐┌─┐┬ ┬┌┐┌┌─┐┌─┐┬  ┌┬┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ││││ │───├┤    │││ │││││││├┤ │ ││   ││\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ └┴┘└─┘   └─┘  ─┴┘└─┘└┴┘┘└┘└  └─┘┴─┘─┴┘\n");
    app_log(1, "  CoQuí input checkpoint file:              {}", input_file);
    app_log(1, "    - Input Green's function: ");
    app_log(1, "      HDF5 group:                            {}", greens_func_source);
    app_log(1, "      iteration:                             {}", greens_func_iteration);
    app_log(1, "  Output file with model hamiltonian:        {}", output_file);
    app_log(1, "");
    app_log(1, "  Transformation matrices:                   {}", proj_boson.C_file());
    app_log(1, "  Number of impurities:                      {}", proj_boson.nImps());
    app_log(1, "  Number of local orbitals per impurity:     {}", proj_boson.nImpOrbs());
    app_log(1, "  Range of primary orbitals for local basis: [{}, {})",
            proj_boson.W_rng()[0].first(), proj_boson.W_rng()[0].last());
    app_log(1, "");
    app_log(1, "  Screening type:                            {}", screen_type);
    app_log(1, "  factorization type:                        {}", factorization_type);
    app_log(1, "  permutation symmetry:                      {}\n", permut_symm);


    _Timer.start("DF_READ");
    // Check status of MBState
    if (!mb_state.sG_tskij)
      mb_state.sG_tskij.emplace(read_greens_function(*mpi, _MF, input_file, greens_func_iteration, greens_func_source));
    _Timer.stop("DF_READ");

    if (mpi->comm.root()) {
      // write basic info to downfold_2e, needed for logic in downfold_1e
      weiss_b_iter = greens_func_iteration+1;
      h5::file file(input_file, 'a');
      auto grp = h5::group(file);
      auto downfold_grp = (grp.has_subgroup("downfold_2e"))?
                          grp.open_group("downfold_2e") : grp.create_group("downfold_2e");
      auto iter_grp = (downfold_grp.has_subgroup("iter"+std::to_string(weiss_b_iter)))?
                      downfold_grp.open_group("iter"+std::to_string(weiss_b_iter)) :
                      downfold_grp.create_group("iter"+std::to_string(weiss_b_iter));

      h5::h5_write(downfold_grp, "final_iter", (long)weiss_b_iter);
    }

    _Timer.start("DF_DONWFOLD");
    if( factorization_type=="none" or factorization_type=="cholesky_from_4index" ) {

      auto[V_abcd, U_wabcd, eps_inv_head_wq, eps_inv_head_w, pi_head_wq] =
          local_eri_impl<true>(mb_state, eri, ft, screen_type);
      V_abcd += U_wabcd(0,nda::ellipsis{});

      // enforce permutation symmetries
      if (permut_symm!="none") {
        _Timer.start("DF_SYMM");
        apply_permut_symm(V_abcd, permut_symm, "static screened interactions");
        _Timer.stop("DF_SYMM");
      }

      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
        h5::file file(output_file, 'a');
        auto grp = h5::group(file);
        auto sgrp = grp.create_group("Interaction");

        if(factorization_type=="none") {
          h5::h5_write(sgrp, "factorization_type", factorization_type);
          nda::h5_write(sgrp,"Vq0",V_abcd,false);
        } else if(factorization_type=="cholesky_from_4index") {
          nda::array<int,1> piv(nImpOrbs*nImpOrbs+1);
          auto V3D = nda::reshape(V_abcd,std::array<long,3>{nImpOrbs*nImpOrbs,nImpOrbs,nImpOrbs});
          auto V2D = nda::reshape(V_abcd,std::array<long,2>{nImpOrbs*nImpOrbs,nImpOrbs*nImpOrbs});
          { // transpose into hermitian matrix
            nda::array<ComplexType,1> T(nImpOrbs*nImpOrbs);
            for( int a=0; a<nImpOrbs; ++a )
              for( int b=a+1; b<nImpOrbs; ++b ) {
                T() = V3D(nda::range::all,a,b);
                V3D(nda::range::all,a,b) = V3D(nda::range::all,b,a);
                V3D(nda::range::all,b,a) = T();
              }
          }
          using U_type = nda::array<ComplexType,2>;
          auto U = utils::chol<true,U_type>(V2D,piv,thresh,true);
          // conjugate to get L*conj(L)
          U() = nda::conj(U());
          int nchol = U.extent(0);
          auto Vq = nda::reshape(U,std::array<long,5>{nchol,1,1,nImpOrbs,nImpOrbs});
          methods::chol_reader_t::add_meta_data(sgrp,nchol,thresh,1,1,1,nImpOrbs,
                                                nda::array<double, 2>::zeros({1,3}),nda::array<double, 2>::zeros({1,3}),
                                                nda::array<int, 2>::zeros({1,1}));
          h5::h5_write(sgrp, "factorization_type", "cholesky");
          nda::h5_write(sgrp, "Vq0", Vq, false);
        }

        nda::h5_write(sgrp, "crpa_eps_inv_head_wq", eps_inv_head_wq, false);
        nda::h5_write(sgrp, "crpa_eps_inv_head_w", eps_inv_head_w, false);
        nda::h5_write(sgrp, "crpa_pi_head_wq", pi_head_wq, false);

      }
      _Timer.stop("DF_WRITE");

    } else if(factorization_type == "cholesky" or factorization_type=="cholesky_high_memory") {

      // evaluate local screened interaction W(iw)
      auto [U_nab, eps_inv_head_wq, eps_inv_head_w, pi_head_wq] =
          rpa_chol_eri_impl(mb_state, eri, ft, factorization_type, thresh, screen_type);

      // enforce permutation symmetries
      if (permut_symm!="none" and mpi->comm.root() )
        app_warning(" downfold_2e: Skipping application of permutation symmetry with factorization type = cholesky");

      _Timer.start("DF_WRITE");
      if (mpi->comm.root()) {
        h5::file file(output_file, 'a');
        auto grp = h5::group(file);
        auto sgrp = grp.create_group("Interaction");

        // temporary hack to make convention consistent wirh ERI::cholesky
        int nchol = U_nab.global_shape()[0];
        methods::chol_reader_t::add_meta_data(sgrp,nchol,thresh,1,1,1,nImpOrbs,
                                              nda::array<double, 2>::zeros({1,3}),nda::array<double, 2>::zeros({1,3}),
                                              nda::array<int, 2>::zeros({1,1}));
        h5::h5_write(sgrp, "factorization_type", factorization_type.substr(0,8));
        write_cholesky_embed(sgrp,"Vq0",U_nab,true,true);

        nda::h5_write(sgrp, "crpa_eps_inv_head_wq", eps_inv_head_wq, false);
        nda::h5_write(sgrp, "crpa_eps_inv_head_w", eps_inv_head_w, false);
        nda::h5_write(sgrp, "crpa_pi_head_wq", pi_head_wq, false);
      } else {
        auto grp = h5::group();
        write_cholesky_embed(grp,"Vq0",U_nab,true,true);
      }
      _Timer.stop("DF_WRITE");

    } else if(factorization_type == "thc") {
      APP_ABORT("Error in downfold_screen_model_impl: factorization_type=thc not yet implemented");
    } else {
      APP_ABORT("Error in downfold_screen_model_impl: Invalid factorization_type:{}",factorization_type);
    }
    mpi->comm.barrier();
    _Timer.stop("DF_TOTAL");
    print_downfold_timers();
  }

  auto embed_eri_t::u_bosonic_weiss_rpa(nda::array<ComplexType, 5> &G_tsIab,
                                        nda::array<ComplexType, 5> &W_wabcd,
                                        nda::array<ComplexType, 4> &V_abcd,
                                        const imag_axes_ft::IAFT &ft,
                                        bool density_only)
  -> sArray_t<Array_view_5D_t> {
    using math::shm::make_shared_array;
    app_log(1, "\n[==== EDMFT bosonic Weiss field routines start ====]\n");
    app_log(1, " * No impurity polarizability is found!");
    app_log(1, " * Will take Pi_imp(t) equals to the DC counting correction Pi0(t)=G(t)*G(-t).\n");

    auto sPi_imp_pb_wabcd = eval_Pi_rpa_dc(*_context, G_tsIab, ft, density_only);

    int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
    int nbnd = sPi_imp_pb_wabcd.shape()[1];
    nda::array<ComplexType, 5> X_tabcd(nts_half, nbnd, nbnd, nbnd, nbnd);
    ft.w_to_tau_PHsym(sPi_imp_pb_wabcd.local(), X_tabcd);
    // CNY: when Pi is almost negligible, e.g. 1e-10, the leakage could be very large, although it should be fine.
    ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(_context->comm), "impurity polarizability");

    auto sW_pb_wabcd = make_shared_array<Array_view_5D_t>(*_context, W_wabcd.shape());
    auto sV_pb_abcd = make_shared_array<Array_view_4D_t>(*_context, V_abcd.shape());
    sW_pb_wabcd.local() = to_product_basis(W_wabcd);
    sV_pb_abcd.local() = to_product_basis(V_abcd);
    _context->comm.barrier();

    dyson_for_u_weiss_in_place(sW_pb_wabcd, sV_pb_abcd, sPi_imp_pb_wabcd);
    ft.w_to_tau_PHsym(sPi_imp_pb_wabcd.local(), X_tabcd);
    ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(_context->comm), "bosonic weiss field");

    // transform back to the chemists' notation
    if (_context->node_comm.root()) {
      auto U_wabcd = sPi_imp_pb_wabcd.local();
      nda::array<ComplexType, 4> tmp(nbnd, nbnd, nbnd, nbnd);
      for (size_t w=0; w<sPi_imp_pb_wabcd.shape()[0]; ++w) {
        for (size_t a=0; a<nbnd; ++a) {
          for (size_t b=0; b<nbnd; ++b) {
            tmp(a,b,nda::ellipsis{}) = U_wabcd(w,b,a,nda::ellipsis{});
          }
        }
        U_wabcd(w,nda::ellipsis{}) = tmp;
      }
    }
    _context->node_comm.barrier();
    return sPi_imp_pb_wabcd;
  }

  void embed_eri_t::u_bosonic_weiss_edmft_in_place(
      const nda::array<ComplexType, 5> &W_wabcd,
      const nda::array<ComplexType, 4> &V_abcd,
      sArray_t<Array_view_5D_t> &sPi_imp_wabcd) {
    using math::shm::make_shared_array;
    app_log(1, "\n[==== EDMFT Bosonic Weiss Field Routines ====]\n");

    // convert to product basis
    auto sW_pb_wabcd = make_shared_array<Array_view_5D_t>(*_context, W_wabcd.shape());
    auto sV_pb_abcd = make_shared_array<Array_view_4D_t>(*_context, V_abcd.shape());
    sW_pb_wabcd.local() = to_product_basis(W_wabcd);
    sV_pb_abcd.local() = to_product_basis(V_abcd);
    _context->comm.barrier();

    // perform the Dyson equation for the Weiss field in place
    dyson_for_u_weiss_in_place(sW_pb_wabcd, sV_pb_abcd, sPi_imp_wabcd);

    // transform back to the chemists' notation
    if (_context->node_comm.root()) {
      auto U_wabcd = sPi_imp_wabcd.local();
      nda::array<ComplexType, 4> tmp(V_abcd.shape());
      for (size_t w=0; w<sPi_imp_wabcd.shape()[0]; ++w) {
        for (size_t a=0; a<V_abcd.shape(0); ++a) {
          for (size_t b=0; b<V_abcd.shape(0); ++b) {
            tmp(a,b,nda::ellipsis{}) = U_wabcd(w,b,a,nda::ellipsis{});
          }
        }
        U_wabcd(w,nda::ellipsis{}) = tmp;
      }
    }
    _context->node_comm.barrier();
  }

  auto embed_eri_t::u_bosonic_weiss_edmft(h5::group weiss_b_grp,
                                          nda::array<ComplexType, 5> &W_wabcd,
                                          nda::array<ComplexType, 4> &V_abcd,
                                          const imag_axes_ft::IAFT &ft)
  -> sArray_t<Array_view_5D_t> {
    using math::shm::make_shared_array;
    app_log(1, "\n[==== EDMFT Bosonic Weiss Field Routines ====]\n");
    app_log(1, " * Reading impurity polarizability from coqui h5.\n");

    auto sPi_imp_wabcd = make_shared_array<Array_view_5D_t>(*_context, W_wabcd.shape());
    if (_context->node_comm.root()) {
      auto Pi_imp = sPi_imp_wabcd.local();
      nda::h5_read(weiss_b_grp, "Pi_imp_wabcd", Pi_imp);
    }
    _context->comm.barrier();

    int nts_half = (ft.nt_b()%2==0)? ft.nt_b()/2 : ft.nt_b()/2+1;
    int nbnd = sPi_imp_wabcd.shape()[1];
    nda::array<ComplexType, 5> X_tabcd(nts_half, nbnd, nbnd, nbnd, nbnd);
    ft.w_to_tau_PHsym(sPi_imp_wabcd.local(), X_tabcd);
    ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(_context->comm), "impurity polarizability");

    auto sW_pb_wabcd = make_shared_array<Array_view_5D_t>(*_context, W_wabcd.shape());
    auto sV_pb_abcd = make_shared_array<Array_view_4D_t>(*_context, V_abcd.shape());
    sW_pb_wabcd.local() = to_product_basis(W_wabcd);
    sV_pb_abcd.local() = to_product_basis(V_abcd);
    _context->comm.barrier();

    dyson_for_u_weiss_in_place(sW_pb_wabcd, sV_pb_abcd, sPi_imp_wabcd);
    ft.w_to_tau_PHsym(sPi_imp_wabcd.local(), X_tabcd);
    ft.check_leakage_PHsym(X_tabcd, imag_axes_ft::boson, std::addressof(_context->comm), "bosonic weiss field");

    // transform back to the chemists' notation
    if (_context->node_comm.root()) {
      auto U_wabcd = sPi_imp_wabcd.local();
      nda::array<ComplexType, 4> tmp(nbnd, nbnd, nbnd, nbnd);
      for (size_t w=0; w<sPi_imp_wabcd.shape()[0]; ++w) {
        for (size_t a=0; a<nbnd; ++a) {
          for (size_t b=0; b<nbnd; ++b) {
            tmp(a,b,nda::ellipsis{}) = U_wabcd(w,b,a,nda::ellipsis{});
          }
        }
        U_wabcd(w,nda::ellipsis{}) = tmp;
      }
    }
    _context->node_comm.barrier();
    return sPi_imp_wabcd;
  }

  template<bool return_eps_inv, THC_ERI thc_t>
  auto embed_eri_t::local_eri_impl(MBState &mb_state, thc_t &thc,
                                   const imag_axes_ft::IAFT &ft,
                                   std::string screen_type) {

    app_log(1, "\n[==== Local Screened Interactions Routines ====]\n");
    auto mpi = thc.mpi();
    auto& proj_boson = mb_state.proj_boson.value();

    // Construct B matrix from the local basis to a product basis from the thc class
    auto sB_qIPab = (_MF->nqpts_ibz()==_MF->nqpts())?
                    proj_boson.calc_bosonic_projector(thc) :
                    proj_boson.calc_bosonic_projector_symm(thc);
    auto B_qIPab = sB_qIPab.local();

    // Bare interactions
    app_log(1, "Downfolding the Bare Coulomb Interactions\n"
               "-----------------------------------------\n");
    auto V_abcd = downfold_V(thc, B_qIPab);

    // Dynamical screened interactions
    app_log(1, "Downfolding the Dynamic Screened Interactions\n"
               "-----------------------------------------\n");
    app_log(1, "  Screening = {}\n", screen_type);
    solvers::scr_coulomb_t scr_coulomb(&ft, screen_type, _div_treatment);
    auto dPi_tqPQ = scr_coulomb.eval_Pi_qdep(mb_state, thc);

    auto[w_pgrid, w_bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(mpi->comm.size(), _MF->nqpts_ibz(), ft.nw_b(), thc.Np());
    auto dW_wqPQ = scr_coulomb.tau_to_w(dPi_tqPQ, w_pgrid, w_bsize, true);
    auto pi_head_wq = solvers::div_utils::head_from_prod_basis(dW_wqPQ, thc);

    // Dyson for screened interaction. Here we assume particle-hole symmetry
    scr_coulomb.dyson_W_in_place(dW_wqPQ, thc);
    auto [eps_inv_head_wq, eps_inv_head_w] = solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *_MF, _div_treatment);
    auto W_wabcd = downfold_W(thc, dW_wqPQ, B_qIPab, eps_inv_head_w);

    if constexpr (return_eps_inv) {
      return std::make_tuple(std::move(V_abcd), std::move(W_wabcd), std::move(eps_inv_head_wq),
                             std::move(eps_inv_head_w), std::move(pi_head_wq));
    } else {
      return std::make_tuple(std::move(V_abcd), std::move(W_wabcd));
    }
  }

  template<bool return_eps_inv, THC_ERI thc_t>
  auto embed_eri_t::rpa_q_eri_impl(MBState &mb_state, thc_t &thc, const imag_axes_ft::IAFT &ft,
                                   std::string screen_type) {
    app_log(1, "\nQ-dependent downfolded screened interactions routines begin:\n"
               "------------------------------------------------------------\n");
    auto mpi = thc.mpi();
    auto& proj_boson = mb_state.proj_boson.value();
    long nImpOrbs = proj_boson.nImpOrbs();

    // Construct B matrix from the local basis to a product basis from the thc class
    auto sB_qIPab = (_MF->nqpts_ibz()==_MF->nqpts())?
                    proj_boson.calc_bosonic_projector(thc) : proj_boson.calc_bosonic_projector_symm(thc);
    auto B_qIPab = sB_qIPab.local();

    // Bare interactions
    app_log(1, "Downfolding the bare Coulomb interactions...\n");
    auto V_qabcd = downfold_Vq(thc, B_qIPab);

    // Dynamical screened interactions
    app_log(1, "Downfolding the dynamic screened interactions with screening type = {}.\n", screen_type);
    solvers::scr_coulomb_t scr_coulomb(&ft, screen_type, _div_treatment);
    auto dPi_tqPQ = scr_coulomb.eval_Pi_qdep(mb_state, thc);

    auto[w_pgrid, w_bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(mpi->comm.size(), _MF->nqpts_ibz(), ft.nw_b(), thc.Np());
    auto dW_wqPQ = scr_coulomb.tau_to_w(dPi_tqPQ, w_pgrid, w_bsize, true);
    auto pi_head_wq = solvers::div_utils::head_from_prod_basis(dW_wqPQ, thc);

    // Dyson for screened interaction
    // FIXME We assume particle-hole symmetry. This may not always be the case!
    scr_coulomb.dyson_W_in_place(dW_wqPQ, thc);
    auto [eps_inv_head_wq, eps_inv_head_w] = solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *_MF, _div_treatment);
    auto W_wqabcd = downfold_Wq(thc, dW_wqPQ, B_qIPab);

    // projection for local quantities
    nda::array<ComplexType, 4> V_abcd(nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 5> W_wabcd(W_wqabcd.shape(0), nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    for (size_t iq_full=0; iq_full < _MF->nqpts(); ++iq_full) {
      V_abcd += V_qabcd(iq_full, nda::ellipsis{});
      W_wabcd += W_wqabcd(nda::range::all, iq_full, nda::ellipsis{});
    }
    V_abcd() /= _MF->nqpts();
    W_wabcd() /= _MF->nqpts();

    // finite-size correction to V_abcd
    V_div_correction(V_abcd, B_qIPab, thc);
    // finite-size correction to W_abcd(w)
    W_div_correction(thc, W_wabcd, B_qIPab, eps_inv_head_w);

    if constexpr (return_eps_inv) {
      return std::make_tuple(std::move(V_qabcd), std::move(W_wqabcd), std::move(V_abcd), std::move(W_wabcd),
                             std::move(eps_inv_head_wq), std::move(eps_inv_head_w), std::move(pi_head_wq));
    } else {
      return std::make_tuple(std::move(V_qabcd), std::move(W_wqabcd), std::move(V_abcd), std::move(W_wabcd));
    }
  }

  auto embed_eri_t::rpa_chol_eri_impl(MBState &mb_state, THC_ERI auto &thc, const imag_axes_ft::IAFT &ft,
                                 std::string factorization_type, double thresh,
                                 std::string screen_type) {
    app_log(1, "\nCholesky decomposed local screened interactions routines begin:\n"
               "---------------------------------------------------------------\n");

    utils::check(screen_type.substr(0,4)=="crpa", "embed_eri_t::rpa_q_eri_impl: non-crpa type screening is not supported.");
    auto mpi = thc.mpi();

    // Dynamical screened interactions
    app_log(1, "Downfolding the dynamic screened interactions with screening type = {}.\n", screen_type);
    solvers::scr_coulomb_t scr_coulomb(&ft, screen_type, _div_treatment);
    auto dPi_tqPQ = scr_coulomb.eval_Pi_qdep(mb_state, thc);

    auto[w_pgrid, w_bsize] = solvers::scr_coulomb_t::W_omega_proc_grid(mpi->comm.size(), _MF->nqpts_ibz(), ft.nw_b(), thc.Np());
    auto dW_wqPQ = scr_coulomb.tau_to_w(dPi_tqPQ, w_pgrid, w_bsize, true);
    auto pi_head_wq = solvers::div_utils::head_from_prod_basis(dW_wqPQ, thc);

    // Dyson for screened interaction
    // FIXME We assume particle-hole symmetry. This may not always be the case!
    scr_coulomb.dyson_W_in_place(dW_wqPQ, thc);
    auto [eps_inv_head_wq, eps_inv_head_w] = solvers::div_utils::eps_inv_head_w(dW_wqPQ, thc, *_MF, _div_treatment);

    auto dV_qPQ = thc.dZ({1, 1, mpi->comm.size()});
    // dV_qPQ = dV_qPQ + dW_wqPQ[0,...]
    {
      // lazy for now, add redistribute routine that can operate on a submatrix, or eval_W_selected_frequencies
      math::nda::redistribute_in_place(dW_wqPQ,{1,1,1,mpi->comm.size()},
                     {dW_wqPQ.global_shape()[0],dW_wqPQ.global_shape()[1],dW_wqPQ.global_shape()[2],dV_qPQ.block_size()[2]}); 
      utils::check(dV_qPQ.local_shape()[0] == dW_wqPQ.local_shape()[1] and
                   dV_qPQ.local_shape()[1] == dW_wqPQ.local_shape()[2] and
                   dV_qPQ.local_shape()[2] == dW_wqPQ.local_shape()[3] and
                   dV_qPQ.origin()[0] == dW_wqPQ.origin()[1] and
                   dV_qPQ.origin()[1] == dW_wqPQ.origin()[2] and
                   dV_qPQ.origin()[2] == dW_wqPQ.origin()[3], 
                   "Error in rpa_chol_eri_impl: Inconsistent data distribution, should not happen. \n");
      dV_qPQ.local() += dW_wqPQ.local()(0,nda::ellipsis{});
    }
    dW_wqPQ.reset();
    app_log(1, "Treatment of long-wavelength divergence in V (bare): {}", _bare_div_treatment);
    app_log(1, "Treatment of long-wavelength divergence in W: {}", _div_treatment);
    auto div_factor = ( _bare_div_treatment == "ignore_g0" )? ComplexType(0.0) : ComplexType(1.0);
    div_factor += ( _div_treatment == "ignore_g0" ) ? ComplexType(0.0) : eps_inv_head_w(0);
    auto W_nab = ( factorization_type=="cholesky" ?
                   downfold_cholesky(thc, mb_state.proj_boson.value(), dV_qPQ, div_factor, thresh) :
                   downfold_cholesky_high_memory(thc, mb_state.proj_boson.value(), dV_qPQ, div_factor, thresh) );
    dV_qPQ.reset(); // return memory, since this can be large

    return std::make_tuple(std::move(W_nab), std::move(eps_inv_head_wq),
                           std::move(eps_inv_head_w), std::move(pi_head_wq));
  }

  void embed_eri_t::dyson_for_u_weiss_in_place(sArray_t<Array_view_5D_t> &sW_pb_wabcd, sArray_t<Array_view_4D_t> &sV_pb_abcd,
                                               sArray_t<Array_view_5D_t> &sPi_imp_pb_wabcd) {
    int nw_half = sW_pb_wabcd.shape()[0];
    int nImpOrbs = sW_pb_wabcd.shape()[1];
    int nImpOrbs2 = nImpOrbs*nImpOrbs;

    auto W_pb_wabcd = sW_pb_wabcd.local();
    auto V_pb_abcd = sV_pb_abcd.local();
    auto Pi_pb_wabcd = sPi_imp_pb_wabcd.local();

    auto W_ab_cd = nda::reshape(W_pb_wabcd, shape_t<3>{nw_half, nImpOrbs2, nImpOrbs2});
    auto Pi_ab_cd = nda::reshape(Pi_pb_wabcd, shape_t<3>{nw_half, nImpOrbs2, nImpOrbs2});
    auto V_ab_cd = nda::reshape(V_pb_abcd, shape_t<2>{nImpOrbs2, nImpOrbs2});

    int node_rank = sPi_imp_pb_wabcd.node_comm()->rank();
    int node_size = sPi_imp_pb_wabcd.node_comm()->size();
    sPi_imp_pb_wabcd.win().fence();
    if (node_rank < nw_half) {
      nda::matrix <ComplexType> tmp(nImpOrbs2, nImpOrbs2);
      nda::matrix <ComplexType> Wfull(nImpOrbs2, nImpOrbs2);
      for (size_t w = node_rank; w < nw_half; w += node_size) {
        Wfull() = W_ab_cd(w, nda::ellipsis{}) + V_ab_cd;
        tmp() = ComplexType(1.0);
        // tmp = I + Pi*W
        nda::blas::gemm(ComplexType(1.0), Pi_ab_cd(w, nda::ellipsis{}), Wfull, ComplexType(1.0), tmp);
        // tmp = [I + Pi*W]^{-1}
        tmp = nda::inverse(tmp);
        // u = W[I + Pi*W]^{-1}
        nda::blas::gemm(Wfull, tmp, Pi_ab_cd(w, nda::ellipsis{}));
        // u -= v
        Pi_ab_cd(w, nda::ellipsis{}) -= V_ab_cd;
      }
    }
    sPi_imp_pb_wabcd.win().fence();
  }

  template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
  auto embed_eri_t::downfold_V(thc_t &thc, const B_t &B_qIPab)
      -> nda::array<ComplexType, 4>
  {
    // B_qIPab lives in the full MP mesh
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    auto nqpts_ibz = _MF->nqpts_ibz();

    auto comm = thc.mpi()->comm;
    int np = comm.size();
    long nqpools = utils::find_proc_grid_max_npools(np, nqpts_ibz, 0.2);
    np /= nqpools;
    long np_P = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_Q = np / np_P;
    utils::check(nqpools > 0 and nqpools <= nqpts_ibz,
                 "embed_eri_t::downfold_V: nqpools <= 0 or nqpools > nqpts_ibz. nqpools = {}", nqpools);
    utils::check(comm.size() % nqpools == 0, "embed_eri_t::downfold_V: gcomm.size() % nqpools != 0");

    auto dV_qPQ = thc.dZ({nqpools, np_P, np_Q});
    auto [nq_loc, NP_loc, NQ_loc] = dV_qPQ.local_shape();
    auto Q_rng = dV_qPQ.local_range(2);
    auto [q_origin, P_origin, Q_origin] = dV_qPQ.origin();

    nda::array<ComplexType, 4> V_cdab(nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    auto V_cd_ab = nda::reshape(V_cdab, shape_t<2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});

    // V_cdab = conj(B_qPdc) * [ V_qPQ ] * B_qQab
    nda::array<ComplexType, 2> V_PQ_loc(NP_loc, NQ_loc);
    nda::array<ComplexType, 2> T_P_ab(NP_loc, nImpOrbs*nImpOrbs);
    nda::array<ComplexType, 3> B_cdP_conj(nImpOrbs, nImpOrbs, NP_loc);
    auto B_cd_P_conj = nda::reshape(B_cdP_conj, shape_t<2>{nImpOrbs*nImpOrbs, NP_loc});

    // Bare interactions
    for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
      // iq lives in IBZ
      size_t iq = q_origin + iq_loc;

      // loop over all symmetry-related q-points for iq
      for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
        if (_MF->qp_to_ibz(iq_full)!=iq) continue;
        auto B_Q_ab = nda::reshape(B_qIPab(iq_full, 0, Q_rng, nda::ellipsis{}),
                                   shape_t<2>{NQ_loc, nImpOrbs*nImpOrbs});
        if (_MF->qp_trev(iq_full)) {
          V_PQ_loc = nda::conj( dV_qPQ.local()(iq_loc, nda::ellipsis{}) );
          nda::blas::gemm(V_PQ_loc, B_Q_ab, T_P_ab);
        } else {
          V_PQ_loc = dV_qPQ.local()(iq_loc, nda::ellipsis{});
          nda::blas::gemm(V_PQ_loc, B_Q_ab, T_P_ab);
        }

        for (size_t P = 0; P < NP_loc; ++P) {
          auto B_dc = B_qIPab(iq_full, 0, P_origin+P, nda::ellipsis{});
          B_cdP_conj(nda::range::all, nda::range::all, P) = nda::conj(nda::transpose(B_dc));
        }
        nda::blas::gemm(ComplexType(1.0), B_cd_P_conj, T_P_ab, ComplexType(1.0), V_cd_ab);
      }
    }
    comm.all_reduce_in_place_n(V_cdab.data(), V_cdab.size(), std::plus<>{});
    V_cdab() /= (nqpts);

    // finite-size correction to V_cdab
    V_div_correction(V_cdab, B_qIPab, thc);

    return V_cdab;
  }

  template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
  auto embed_eri_t::downfold_Vq(thc_t &thc, const B_t &B_qIPab)
  -> nda::array<ComplexType, 5>
  {
    // B_qIPab lives in the full MP mesh
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    auto nqpts_ibz = _MF->nqpts_ibz();

    auto comm = thc.mpi()->comm;
    int np = comm.size();
    long nqpools = utils::find_proc_grid_max_npools(np, nqpts_ibz, 0.2);
    np /= nqpools;
    long np_P = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_Q = np / np_P;
    utils::check(nqpools > 0 and nqpools <= nqpts_ibz,
                 "embed_eri_t::downfold_Vq: nqpools <= 0 or nqpools > nqpts_ibz. nqpools = {}", nqpools);
    utils::check(comm.size() % nqpools == 0, "embed_eri_t::downfold_Vq: gcomm.size() % nqpools != 0");

    auto dV_qPQ = thc.dZ({nqpools, np_P, np_Q});
    auto [nq_loc, NP_loc, NQ_loc] = dV_qPQ.local_shape();
    auto Q_rng = dV_qPQ.local_range(2);
    auto [q_origin, P_origin, Q_origin] = dV_qPQ.origin();

    nda::array<ComplexType, 5> V_qcdab(nqpts, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);

    // V_qcdab = conj(B_qPdc) * [ V_qPQ ] * B_qQab
    nda::array<ComplexType, 2> V_PQ_loc(NP_loc, NQ_loc);
    nda::array<ComplexType, 2> T_P_ab(NP_loc, nImpOrbs*nImpOrbs);
    nda::array<ComplexType, 3> B_cdP_conj(nImpOrbs, nImpOrbs, NP_loc);
    auto B_cd_P_conj = nda::reshape(B_cdP_conj, shape_t<2>{nImpOrbs*nImpOrbs, NP_loc});

    // Bare interactions
    for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
      // iq lives in IBZ
      size_t iq = q_origin + iq_loc;

      // loop over all symmetry-related q-points for iq
      for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
        if (_MF->qp_to_ibz(iq_full)!=iq) continue;
        auto B_Q_ab = nda::reshape(B_qIPab(iq_full, 0, Q_rng, nda::ellipsis{}),
                                   shape_t<2>{NQ_loc, nImpOrbs*nImpOrbs});
        if (_MF->qp_trev(iq_full)) {
          V_PQ_loc = nda::conj( dV_qPQ.local()(iq_loc, nda::ellipsis{}) );
          nda::blas::gemm(V_PQ_loc, B_Q_ab, T_P_ab);
        } else {
          V_PQ_loc = dV_qPQ.local()(iq_loc, nda::ellipsis{});
          nda::blas::gemm(V_PQ_loc, B_Q_ab, T_P_ab);
        }

        for (size_t P = 0; P < NP_loc; ++P) {
          auto B_dc = B_qIPab(iq_full, 0, P_origin+P, nda::ellipsis{});
          B_cdP_conj(nda::range::all, nda::range::all, P) = nda::conj(nda::transpose(B_dc));
        }
        auto Vq_2D = nda::reshape(V_qcdab(iq_full, nda::ellipsis{}), shape_t<2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
        nda::blas::gemm(ComplexType(1.0), B_cd_P_conj, T_P_ab, ComplexType(1.0), Vq_2D);
      }
    }
    comm.all_reduce_in_place_n(V_qcdab.data(), V_qcdab.size(), std::plus<>{});

    return V_qcdab;
  }

  template<THC_ERI thc_t> 
  auto embed_eri_t::downfold_cholesky(thc_t &thc, const projector_boson_t &proj_boson,
                                      math::nda::DistributedArrayOfRank<3> auto &dV_qPQ,
                                      ComplexType div_correction_factor,
                                      double thresh)
  {
    constexpr MEMORY_SPACE MEM = HOST_MEMORY;
    auto mpi = thc.mpi();
    bool root = mpi->comm.root();
    utils::check( dV_qPQ.grid() == std::array<long,3>{1,1,mpi->comm.size()},
                   " Error in downfold_cholesky: Incorrect processor grid in V_qPQ.");

    auto dT_skIPa = compute_collation_impurity_basis(thc, proj_boson, dV_qPQ.local_range(2));
    auto T_skIPa = dT_skIPa.local();

    nda::array<ComplexType, 2> BB_ab;
    {
      auto [nsym, ns, nkpts, nImps, NQ, nImpOrbs] = T_skIPa.shape();
      BB_ab = nda::array<ComplexType, 2>::zeros({nImpOrbs,nImpOrbs});
      // BB(a,b) = e0 * sum_skP conj(T_skIPa(0,sk,I,P,a)) * T_skIPa(0,sk,I,P,b) * chi_head(0,P)
      auto chi_head = thc.basis_head()(0, dT_skIPa.local_range(4));
      ComplexType e0(std::sqrt(_MF->madelung()*div_correction_factor)/nkpts);
      long I=0;
      for (long is = 0; is < ns; ++is) 
        for (long ik = 0; ik < nkpts; ++ik) 
          for (long P = 0; P < NQ; ++P)
            nda::blas::gerc(ComplexType(e0)*chi_head(P),T_skIPa(0, is, ik, I, P, nda::range::all),
                            T_skIPa(0, is, ik, I, P, nda::range::all), BB_ab);
      mpi->comm.reduce_in_place_n(BB_ab.data(),BB_ab.size(),std::plus<>{},0);
      if(not root)
        BB_ab = nda::array<ComplexType, 2>(0,0);
    }

    /********************************************************
     *               Diagonal based on T_skIPa              * 
     ********************************************************/
    auto diag_V = [&](nda::MemoryArrayOfRank<1> auto && D) {

      _Timer.start("D1");
      using value_type = typename std::decay_t<decltype(D)>::value_type;
      decltype(nda::range::all) all;

      auto qsymms = _MF->qsymms();
      auto [nq, NP_loc, NQ_loc] = dV_qPQ.local_shape();
      auto Vloc = dV_qPQ.local();
      auto Q_rng = dV_qPQ.local_range(2);
      auto NP_tot = dV_qPQ.global_shape()[1];

      auto [nsym, ns, nkpts, nImps, NQ, nImpOrbs] = T_skIPa.shape();
      auto nqpts = _MF->nqpts();
      value_type norm(1.0/(nqpts*nkpts*nkpts));

      utils::check( D.size() == nImpOrbs*nImpOrbs, "Shape mismatch in diag_f.");

      // size correction
      if(root) {
        auto BB = nda::reshape(BB_ab, shape_t<1>{nImpOrbs*nImpOrbs});
        D() = nda::conj(BB)*BB;
      } else {
        D() = value_type(0.0);
      }

      memory::array<MEM,ComplexType,2> T(nImpOrbs*nImpOrbs, NP_tot);
      memory::array<MEM,ComplexType,2> Bq(NQ_loc, nImpOrbs*nImpOrbs);
      auto Bq3D = nda::reshape(Bq, shape_t<3>{NQ_loc, nImpOrbs, nImpOrbs});

      // Bare interactions 
      for (size_t iq = 0; iq < nqpts; ++iq) {
        // iq lives in IBZ

        // loop over all symmetry-related q-points for iq
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;

          Bq() = ComplexType(0.0);
          auto sym_it = std::find(qsymms.begin(), qsymms.end(), _MF->qp_symm(iq_full));
          auto isym = std::distance(qsymms.begin(), sym_it);
          // Bqab
          //for (long I = 0; I < nImps; ++I) {
            long I = 0;
            for (long isk = 0; isk < ns*nkpts; ++isk) {
              long is = isk / nkpts;
              long ik = isk % nkpts;
              if (_MF->qp_trev(iq_full)) {
                long ikmq = _MF->qk_to_k2(_MF->kp_trev_pair(iq_full), ik);
                for (long P = 0; P < NQ; ++P)
                  nda::blas::gerc(ComplexType(1.0),T_skIPa(isym, is, ik, I, P, nda::range::all),
                            T_skIPa(isym, is, ikmq, I, P, nda::range::all), Bq3D(P,nda::ellipsis{}));
              } else {
                long ikmq = _MF->qk_to_k2(iq_full, ik);
                for (long P = 0; P < NQ; ++P)
                  nda::blas::gerc(ComplexType(1.0),T_skIPa(isym, is, ikmq, I, P, nda::range::all),
                            T_skIPa(isym, is, ik, I, P, nda::range::all), Bq3D(P,nda::ellipsis{}));
              }
            }
          //} 

          if (_MF->qp_trev(iq_full)) {
            nda::blas::gemm(value_type(1.0), nda::transpose(Bq),
                                             nda::dagger(Vloc(iq, nda::ellipsis{})),
                            value_type(0.0), T);
          } else {
            nda::blas::gemm(value_type(1.0), nda::transpose(Bq),
                                             nda::transpose(Vloc(iq, nda::ellipsis{})),
                            value_type(0.0), T);
          }
          mpi->comm.all_reduce_in_place_n(T.data(),T.size(),std::plus<>{});
          for( long ab=0; ab<nImpOrbs*nImpOrbs; ++ab )
            D(ab) += norm * nda::blas::dotc(Bq(all,ab), T(ab,Q_rng));
        }
      }
      _Timer.stop("D1");

    };

    /********************************************************
     *                Column based on T_skIPa                *
     ********************************************************/
    // Given (u,v): D(a,b) = V_abcd(b,a,u,v) = conj(V_abcd(v,u,a,b)),
    //              where the second expression is used to avoid needing to transpose
    // The cholesky routine expects a column of the matrix and produces a decomposition L*dagger(L)
    // To be consistent with the current definition of V_abcd = (ba|cd) = sum_n conj(Lnba) * Lncd,
    // col_V returns conj(V_abcd(b,a,u,v)) = V_abcd(v,u,a,b), which will lead to V_abcd = sum_n Lnba * conj(Lncd)
    auto col_V = [&](nda::MemoryArrayOfRank<1> auto && index,
                     nda::MemoryArrayOfRank<2> auto && D) {

      _Timer.start("D2");
      using value_type = typename std::decay_t<decltype(D)>::value_type;
      decltype(nda::range::all) all;

      auto qsymms = _MF->qsymms();
      auto [nq, NP_loc, NQ_loc] = dV_qPQ.local_shape();
      auto Vloc = dV_qPQ.local();
      auto Q_rng = dV_qPQ.local_range(2);
      auto NP_tot = dV_qPQ.global_shape()[1];

      auto [nsym, ns, nkpts, nImps, NQ, nImpOrbs] = T_skIPa.shape();

      auto nqpts = _MF->nqpts();
      auto nqpts_ibz = _MF->nqpts_ibz();

      utils::check( nq == nqpts_ibz, "Shape mismatch in col_V.");
      utils::check( index.size() == D.extent(0), "Shape mismatch in col_V.");
      utils::check( D.extent(1) == nImpOrbs*nImpOrbs, "Shape mismatch in col_V.");
      D() = value_type(0.0);

      memory::array<MEM,ComplexType,2> Bq(NQ_loc, nImpOrbs*nImpOrbs);
      Bq() = ComplexType(0.0);
      auto Bq3D = nda::reshape(Bq, shape_t<3>{NQ_loc, nImpOrbs, nImpOrbs});

      memory::array<MEM,ComplexType,2> B(index.size(), NP_tot);
      memory::array<MEM,ComplexType,2> T(index.size(), NQ_loc);
      value_type norm(1.0/(nqpts*nkpts*nkpts));

      // size correction
      if(root) {
        auto BB = nda::reshape(BB_ab, shape_t<1>{nImpOrbs*nImpOrbs});
        for( auto [in,n] : itertools::enumerate(index) )
          D(in,all) += nda::conj(BB(n))*BB;
      }

      // Bare interactions
      for (size_t iq = 0; iq < nqpts_ibz; ++iq) {
        // iq lives in IBZ

        // loop over all symmetry-related q-points for iq
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;

          Bq() = ComplexType(0.0);
          auto sym_it = std::find(qsymms.begin(), qsymms.end(), _MF->qp_symm(iq_full));
          auto isym = std::distance(qsymms.begin(), sym_it);
          // Bqab
          //for (long I = 0; I < nImps; ++I) {
            long I = 0;
            for (long isk = 0; isk < ns*nkpts; ++isk) {
              long is = isk / nkpts;
              long ik = isk % nkpts;
              long ikmq = (!_MF->qp_trev(iq_full))? _MF->qk_to_k2(iq_full, ik) : 
                                                    _MF->qk_to_k2(_MF->kp_trev_pair(iq_full), ik);

              for (long P = 0; P < NQ; ++P)
                nda::blas::gerc(ComplexType(1.0),T_skIPa(isym, is, ikmq, I, P, nda::range::all),
                          T_skIPa(isym, is, ik, I, P, nda::range::all), Bq3D(P,nda::ellipsis{}));
            }
            if (_MF->qp_trev(iq_full)) {
              for (size_t P=0; P<NQ; ++P)
                Bq3D(P, nda::ellipsis{}) =
                  nda::conj(nda::transpose(Bq3D(P, nda::ellipsis{})));
            }
          //}

          if (_MF->qp_trev(iq_full)) {
            B() = ComplexType(0.0);
            for( auto [in,n] : itertools::enumerate(index) )
              B(in,Q_rng) = Bq(all,n);
            mpi->comm.all_reduce_in_place_n(B.data(),B.size(),std::plus<>{});
            nda::blas::gemm(value_type(1.0), B, Vloc(iq, nda::ellipsis{}),
                            value_type(0.0), T);
            T = nda::conj(T);
            nda::blas::gemm(norm, T, Bq, value_type(1.0), D);
          } else {
            B() = ComplexType(0.0);
            for( auto [in,n] : itertools::enumerate(index) )
              B(in,Q_rng) = nda::conj(Bq(all,n));
            mpi->comm.all_reduce_in_place_n(B.data(),B.size(),std::plus<>{});
            nda::blas::gemm(value_type(1.0), B, Vloc(iq, nda::ellipsis{}),
                            value_type(0.0), T);
            nda::blas::gemm(norm, T, Bq, value_type(1.0), D);
          }
        }
      }
      // don't conjugate, to return conj(V_abcd(b,a,u,v))
      //D() = nda::conj(D); 
      _Timer.stop("D2");

    };

    auto nImpOrbs = T_skIPa.shape(5);
    auto dLnab = embed_cholesky<MEM>(mpi->comm,nImpOrbs,diag_V,col_V,thresh,1);

    return dLnab;
  }

  template<THC_ERI thc_t>
  auto embed_eri_t::downfold_cholesky_high_memory(thc_t &thc, const projector_boson_t &proj_boson,
                                                  math::nda::DistributedArrayOfRank<3> auto &dV_qPQ,
                                                  ComplexType div_correction_factor,
                                                  double thresh)
  {
    constexpr MEMORY_SPACE MEM = HOST_MEMORY;
    auto mpi = thc.mpi();
    bool root = mpi->comm.root();
    utils::check( dV_qPQ.grid() == std::array<long,3>{1,1,mpi->comm.size()},
                   " Error in downfold_cholesky: Incorrect processor grid in V_qPQ.");

    auto sB_qIPab = (_MF->nqpts_ibz()==_MF->nqpts())?
        proj_boson.calc_bosonic_projector(thc) : proj_boson.calc_bosonic_projector_symm(thc);
    auto B_qIPab = sB_qIPab.local();

    nda::array<ComplexType, 2> BB_ab;
    if(root)
    { 
      auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
      BB_ab = nda::array<ComplexType, 2>(nImpOrbs,nImpOrbs);
      auto BB = nda::reshape(BB_ab, shape_t<1>{nImpOrbs*nImpOrbs});
      auto B_4D = nda::reshape(B_qIPab, shape_t<4>{nqpts, nImps, NP, nImpOrbs*nImpOrbs});
      auto chi_head = thc.basis_head()(0, nda::range::all);
      ComplexType e0(std::sqrt(_MF->madelung()) * div_correction_factor);
      nda::blas::gemv(e0,nda::transpose(B_4D(0,0,nda::range::all,nda::range::all)),
                            chi_head,ComplexType(0.0),BB);
    }

    auto diag_V = [&](nda::MemoryArrayOfRank<1> auto && D) {

      using value_type = typename std::decay_t<decltype(D)>::value_type;
      decltype(nda::range::all) all;

      auto [nq_loc, NP_loc, NQ_loc] = dV_qPQ.local_shape();
      auto [q_origin, P_origin, Q_origin] = dV_qPQ.origin();
      auto Vloc = dV_qPQ.local();
      auto P_rng = dV_qPQ.local_range(1);
      auto Q_rng = dV_qPQ.local_range(2);

      auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
      auto B_4D = nda::reshape(B_qIPab, shape_t<4>{nqpts, nImps, NP, nImpOrbs*nImpOrbs});
      value_type norm(1.0/nqpts);

      utils::check( D.size() == nImpOrbs*nImpOrbs, "Shape mismatch in diag_f.");

      // size correction
      if(root) {
        auto BB = nda::reshape(BB_ab, shape_t<1>{nImpOrbs*nImpOrbs});
        D() = nda::conj(BB)*BB;
      } else {
        D() = value_type(0.0);
      }

      memory::array<MEM,ComplexType,2> T(nImpOrbs*nImpOrbs, NQ_loc);
      // Bare interactions 
      for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
        // iq lives in IBZ
        size_t iq = q_origin + iq_loc;

        // loop over all symmetry-related q-points for iq
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;
          if (_MF->qp_trev(iq_full)) {
            // postponing conjugation to dotc
            nda::blas::gemm(value_type(1.0), nda::transpose(B_4D(iq_full,0,P_rng,all)), Vloc(iq_loc, nda::ellipsis{}), value_type(0.0), T);
            for( long ab=0; ab<nImpOrbs*nImpOrbs; ++ab )
              D(ab) += norm * nda::blas::dotc(T(ab,all), B_4D(iq_full,0,Q_rng,ab));
          } else {
            nda::blas::gemm(value_type(1.0), nda::dagger(B_4D(iq_full,0,P_rng,all)), Vloc(iq_loc, nda::ellipsis{}), value_type(0.0), T);
            for( long ab=0; ab<nImpOrbs*nImpOrbs; ++ab )
              D(ab) += norm * nda::blas::dot(T(ab,all), B_4D(iq_full,0,Q_rng,ab));
          }
        }
      }

    };

    // Given (u,v): D(a,b) = V_abcd(b,a,u,v) = conj(V_abcd(v,u,a,b)),
    //              where the second expression is used to avoid needing to transpose
    auto col_V = [&](nda::MemoryArrayOfRank<1> auto && index,
                     nda::MemoryArrayOfRank<2> auto && D) {

      using value_type = typename std::decay_t<decltype(D)>::value_type;

      decltype(nda::range::all) all;

      auto [nq_loc, NP_loc, NQ_loc] = dV_qPQ.local_shape();
      auto [q_origin, P_origin, Q_origin] = dV_qPQ.origin();
      auto Vloc = dV_qPQ.local();
      auto P_rng = dV_qPQ.local_range(2);
      auto Q_rng = dV_qPQ.local_range(2);

      auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
      auto B_4D = nda::reshape(B_qIPab, shape_t<4>{nqpts, nImps, NP, nImpOrbs*nImpOrbs});

      utils::check( index.size() == D.extent(0), "Shape mismatch in col_V.");
      utils::check( D.extent(1) == nImpOrbs*nImpOrbs, "Shape mismatch in col_V.");
      D() = value_type(0.0);

      memory::array<MEM,ComplexType,2> B(index.size(), NP_loc);
      memory::array<MEM,ComplexType,2> T(index.size(), NQ_loc);
      value_type norm(1.0/nqpts);

      // size correction
      if(root) {
        auto BB = nda::reshape(BB_ab, shape_t<1>{nImpOrbs*nImpOrbs});
        for( auto [in,n] : itertools::enumerate(index) )
          D(in,all) += nda::conj(BB(n))*BB;
      }

      // Bare interactions 
      for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
        // iq lives in IBZ
        size_t iq = q_origin + iq_loc;

        // loop over all symmetry-related q-points for iq
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;
          if (_MF->qp_trev(iq_full)) {
            for( auto [in,n] : itertools::enumerate(index) )
              B(in,all) = B_4D(iq_full,0,P_rng,n);
            nda::blas::gemm(value_type(1.0), B, Vloc(iq_loc, nda::ellipsis{}),
                            value_type(0.0), T);
            T = nda::conj(T);
            nda::blas::gemm(norm, T, B_4D(iq_full,0,Q_rng,all),
                            value_type(1.0), D);
          } else {
            for( auto [in,n] : itertools::enumerate(index) )
              B(in,all) = nda::conj(B_4D(iq_full,0,P_rng,n));
            nda::blas::gemm(value_type(1.0), B, Vloc(iq_loc, nda::ellipsis{}),
                            value_type(0.0), T);
            nda::blas::gemm(norm, T, B_4D(iq_full,0,Q_rng,all),
                            value_type(1.0), D);
          }
        }
      }

    };

    auto nImpOrbs = B_qIPab.shape(3);
    auto Lnab = embed_cholesky<MEM>(mpi->comm,nImpOrbs,diag_V,col_V,thresh,32);

    return Lnab;
  }

  template<THC_ERI thc_t>
  auto embed_eri_t::downfold_V_thc(thc_t &thc, double thresh)
  {
    (void) thc;
    (void) thresh;
    nda::array<ComplexType, 2> X_au(1,1);
    nda::array<ComplexType, 2> V_uv(1,1);
    return std::make_tuple(std::move(X_au),std::move(V_uv));
  }

  template<THC_ERI thc_t, nda::MemoryArray Array_4D_t, typename communicator_t,
      nda::ArrayOfRank<5> B_t>
  auto embed_eri_t::downfold_W(thc_t &thc, memory::darray_t<Array_4D_t, communicator_t> &dW_wqPQ,
                          const B_t &B_qIPab,
                          const nda::array<ComplexType, 1> &eps_inv_head)
      -> nda::array<ComplexType, 5>
  {
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    auto [nw_loc, nq_loc, NP_loc, NQ_loc] = dW_wqPQ.local_shape();
    auto Q_rng = dW_wqPQ.local_range(3);
    auto w_origin = dW_wqPQ.origin()[0];
    auto q_origin = dW_wqPQ.origin()[1];
    auto P_origin = dW_wqPQ.origin()[2];
    auto nw_half = dW_wqPQ.global_shape()[0];

    nda::array<ComplexType, 5> W_wcdab(nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    W_wcdab() = ComplexType(0.0);

    // W_wcdab = conj(B_qPdc) * [ W_wqPQ ] * B_qQab
    nda::array<ComplexType, 2> T_P_ab(NP_loc, nImpOrbs*nImpOrbs);
    nda::array<ComplexType, 3> B_cdP_conj(nImpOrbs, nImpOrbs, NP_loc);
    nda::array<ComplexType, 2> W_PQ(NP_loc, NQ_loc);
    auto B_cd_P_conj = nda::reshape(B_cdP_conj, shape_t<2>{nImpOrbs*nImpOrbs, NP_loc});

    auto W_loc = dW_wqPQ.local();
    for (size_t iw_loc = 0; iw_loc < nw_loc; ++iw_loc) {
      size_t iw = w_origin + iw_loc;
      auto W_cd_ab = nda::reshape(W_wcdab(iw, nda::ellipsis()), shape_t<2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
      for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
        size_t iq = q_origin + iq_loc; // iq lives inside IBZ

        // loop over all symmetry-related q-points
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;
          auto B_Q_ab = nda::reshape(B_qIPab(iq_full, 0, Q_rng, nda::ellipsis{}), shape_t<2>{NQ_loc, nImpOrbs*nImpOrbs});
          if (_MF->qp_trev(iq_full))
            W_PQ = nda::conj( W_loc(iw_loc, iq_loc, nda::ellipsis()) );
          else
            W_PQ = W_loc(iw_loc, iq_loc, nda::ellipsis{});
          nda::blas::gemm(W_PQ, B_Q_ab, T_P_ab);

          for (size_t P = 0; P < NP_loc; ++P) {
            auto B_dc = B_qIPab(iq_full, 0, P_origin+P, nda::ellipsis{});
            B_cdP_conj(nda::range::all, nda::range::all, P) = nda::conj(nda::transpose(B_dc));
          }
          nda::blas::gemm(ComplexType(1.0), B_cd_P_conj, T_P_ab, ComplexType(1.0), W_cd_ab);
        }
      }
    }
    dW_wqPQ.communicator()->all_reduce_in_place_n(W_wcdab.data(), W_wcdab.size(), std::plus<>{});
    W_wcdab() /= nqpts;

    // finite-size correction to W_cdab(w)
    W_div_correction(thc, W_wcdab, B_qIPab, eps_inv_head);

    return W_wcdab;
  }

  template<THC_ERI thc_t, nda::MemoryArray Array_4D_t, typename communicator_t, nda::ArrayOfRank<5> B_t>
  auto embed_eri_t::downfold_Wq([[maybe_unused]] thc_t &thc, memory::darray_t<Array_4D_t, communicator_t> &dW_wqPQ,
                                const B_t &B_qIPab)
  -> nda::array<ComplexType, 6> {
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    auto nw_half = dW_wqPQ.global_shape()[0];
    auto [nw_loc, nq_loc, NP_loc, NQ_loc] = dW_wqPQ.local_shape();
    auto Q_rng = dW_wqPQ.local_range(3);
    auto w_origin = dW_wqPQ.origin()[0];
    auto q_origin = dW_wqPQ.origin()[1];
    auto P_origin = dW_wqPQ.origin()[2];

    nda::array<ComplexType, 6> W_wqcdab(nw_half, nqpts, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
    W_wqcdab() = ComplexType(0.0);

    // W_wqcdab = conj(B_qPdc) * [ W_wqPQ ] * B_qQab
    nda::array<ComplexType, 2> T_P_ab(NP_loc, nImpOrbs*nImpOrbs);
    nda::array<ComplexType, 3> B_cdP_conj(nImpOrbs, nImpOrbs, NP_loc);
    nda::array<ComplexType, 2> W_PQ(NP_loc, NQ_loc);
    auto B_cd_P_conj = nda::reshape(B_cdP_conj, shape_t<2>{nImpOrbs*nImpOrbs, NP_loc});

    auto W_loc = dW_wqPQ.local();
    for (size_t iw_loc = 0; iw_loc < nw_loc; ++iw_loc) {
      size_t iw = w_origin + iw_loc;
      //auto W_cd_ab = nda::reshape(W_wcdab(iw, nda::ellipsis()), shape_t<2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
      for (size_t iq_loc = 0; iq_loc < nq_loc; ++iq_loc) {
        size_t iq = q_origin + iq_loc; // iq lives inside IBZ

        // loop over all symmetry-related q-points
        for (size_t iq_full=0; iq_full<nqpts; ++iq_full) {
          if (_MF->qp_to_ibz(iq_full)!=iq) continue;
          auto B_Q_ab = nda::reshape(B_qIPab(iq_full, 0, Q_rng, nda::ellipsis{}), shape_t<2>{NQ_loc, nImpOrbs*nImpOrbs});
          if (_MF->qp_trev(iq_full))
            W_PQ = nda::conj( W_loc(iw_loc, iq_loc, nda::ellipsis()) );
          else
            W_PQ = W_loc(iw_loc, iq_loc, nda::ellipsis{});
          nda::blas::gemm(W_PQ, B_Q_ab, T_P_ab);

          for (size_t P = 0; P < NP_loc; ++P) {
            auto B_dc = B_qIPab(iq_full, 0, P_origin+P, nda::ellipsis{});
            B_cdP_conj(nda::range::all, nda::range::all, P) = nda::conj(nda::transpose(B_dc));
          }
          auto W_wq_2D = nda::reshape(W_wqcdab(iw, iq_full, nda::ellipsis{}), shape_t<2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
          nda::blas::gemm(ComplexType(1.0), B_cd_P_conj, T_P_ab, ComplexType(1.0), W_wq_2D);
        }
      }
    }
    dW_wqPQ.communicator()->all_reduce_in_place_n(W_wqcdab.data(), W_wqcdab.size(), std::plus<>{});

    return W_wqcdab;
  }

  template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
  void embed_eri_t::V_div_correction(nda::array<ComplexType, 4> &V_cdab,
                                const B_t &B_qIPab, thc_t &thc) {
    auto div_string = _bare_div_treatment;
    if (div_string == "ignore_g0") {
      app_log(1, "No finite-size correction for the long-wavelength divergence in "
                 "the local bare V. ");
      return;
    }
    app_log(1, "  Treatment of long-wavelength divergence in bare V: {}\n"
               "    - madelung = {}\n",
          _bare_div_treatment, _MF->madelung());
    auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
    nda::array<ComplexType, 2> BB_ab(nImpOrbs, nImpOrbs);
    nda::array<ComplexType, 2> BB_cd_conj(nImpOrbs, nImpOrbs);
    auto chi_head = thc.basis_head()(0, nda::range::all);
    for (size_t P = 0; P < NP; ++P) {
      BB_ab() += B_qIPab(0, 0, P, nda::ellipsis{}) * chi_head(P);
    }
    BB_cd_conj = nda::conj(nda::transpose(BB_ab));

    auto V_cd_ab_2D = nda::reshape(V_cdab, shape_t<2>{nImpOrbs * nImpOrbs, nImpOrbs * nImpOrbs});
    auto BB_ab_1D = nda::reshape(BB_ab, shape_t<1>{nImpOrbs * nImpOrbs});
    auto BB_cd_conj_1D = nda::reshape(BB_cd_conj, shape_t<1>{nImpOrbs * nImpOrbs});
    V_cd_ab_2D += _MF->madelung() * nda::blas::outer_product(BB_cd_conj_1D, BB_ab_1D);
  }

  template<THC_ERI thc_t>
  void embed_eri_t::W_div_correction(thc_t &thc,
                                nda::array<ComplexType, 5> &W_wcdab,
                                const nda::array<ComplexType, 5> &B_qIPab,
                                const nda::array<ComplexType, 1> &eps_inv_head) {
    app_log(1, "  Treatment of long-wavelength divergence in W: {}", _div_treatment);
    if (_div_treatment == "ignore_g0") {
      return;
    } else if (_div_treatment.find("gygi") != std::string::npos) {
      app_log(1, "    - madelung = {}", _MF->madelung());

      auto [nqpts, nImps, NP, nImpOrbs, nImpOrbs2] = B_qIPab.shape();
      auto nw = W_wcdab.shape(0);

      nda::array<ComplexType, 2> BB_ab(nImpOrbs, nImpOrbs);
      nda::array<ComplexType, 2> BB_cd_conj(nImpOrbs, nImpOrbs);
      auto chi_head = thc.basis_head()(0, nda::range::all);
      for (size_t P = 0; P < NP; ++P) {
        BB_ab() += B_qIPab(0, 0, P, nda::ellipsis{}) * chi_head(P);
      }
      BB_cd_conj = nda::conj(nda::transpose(BB_ab));

      auto W_w_cd_ab_3D = nda::reshape(W_wcdab, shape_t<3>{nw, nImpOrbs * nImpOrbs, nImpOrbs * nImpOrbs});
      auto BB_ab_1D = nda::reshape(BB_ab, shape_t<1>{nImpOrbs * nImpOrbs});
      auto BB_cd_conj_1D = nda::reshape(BB_cd_conj, shape_t<1>{nImpOrbs * nImpOrbs});
      for (size_t n = 0; n < nw; ++n) {
        W_w_cd_ab_3D(n, nda::ellipsis{}) +=
            _MF->madelung() * eps_inv_head(n) * nda::blas::outer_product(BB_cd_conj_1D, BB_ab_1D);
      }
    } else {
      utils::check(false, "Unsupported divergence treatment: {}", _div_treatment);
    }
    app_log(1, "");
  }

  template<nda::MemoryArrayOfRank<4> Array_4D_t>
  auto embed_eri_t::orbital_average_int(const Array_4D_t &V_abcd)
  -> std::tuple<ComplexType, ComplexType, ComplexType, ComplexType> {

    auto nOrbs = V_abcd.shape(0);
    ComplexType U = 0.0;
    ComplexType Up = 0.0;
    ComplexType J_pair = 0.0;
    ComplexType J_spin = 0.0;

    // intra-orbital U
    for (size_t i=0; i<nOrbs; ++i)
      U += V_abcd(i,i,i,i);
    U /= nOrbs;

    if (nOrbs == 1)
      return std::make_tuple(U, Up, J_pair, J_spin);

    // inter-orbital U
    for (size_t i=0; i<nOrbs; ++i) {
      for (size_t j=i+1; j<nOrbs; ++j) {
        if (i != j)
          Up += V_abcd(i,i,j,j);
      }
    }
    Up /= (nOrbs * (nOrbs-1)) / 2;

    // Hund's coupling J: pair-hopping
    for (size_t i=0; i<nOrbs; ++i) {
      for (size_t j=i+1; j<nOrbs; ++j) {
        if (i != j)
          J_pair += V_abcd(i,j,i,j);
      }
    }
    J_pair /= (nOrbs * (nOrbs-1)) / 2;

    // Hund's coupling J: spin-flip
    for (size_t i=0; i<nOrbs; ++i) {
      for (size_t j=i+1; j<nOrbs; ++j) {
        if (i != j)
          J_spin += V_abcd(i,j,j,i);
      }
    }
    J_spin /= (nOrbs * (nOrbs-1)) / 2;

    return std::make_tuple(U, Up, J_pair, J_spin);
  }

} // methods

// instantiation of "public" templates
namespace methods {

  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
  embed_eri_t::compute_downfolded_coulomb_tensors<true>(thc_reader_t&, MBState &, std::string, bool, bool, imag_axes_ft::IAFT *, std::string, long, bool, bool);
  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >
  embed_eri_t::compute_downfolded_coulomb_tensors<false>(thc_reader_t&, MBState &, std::string, bool, bool, imag_axes_ft::IAFT *, std::string, long, bool, bool);

  template void embed_eri_t::downfolding_edmft(thc_reader_t&, MBState&, ptree const&, std::string);

  template void embed_eri_t::downfolding_crpa(thc_reader_t&, MBState&, ptree const&, std::string, std::string, double);

}
