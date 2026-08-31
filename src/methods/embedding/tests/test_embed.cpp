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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "utilities/test_common.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/MBPT_drivers.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/embedding/embed_eri_t.h"
#include "methods/embedding/embed_t.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;

  TEST_CASE("downfold_1e_mb", "[methods][embed][df_1e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    double beta = 1000.0;
    double wmax = 120.0;

    std::string coqui_prefix = "downfold_1e_mb";
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    iter_scf::iter_scf_t iter_sol("damping");

    auto downfold = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file, std::string dc_type,
        std::array<double, 6> &refs, double eps) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi_smallest_q", coqui_prefix);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "gygi_smallest_q");
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);

      auto psp = hamilt::make_pseudopot(*mf);
      write_mf_data(*mf, ft, *psp, coqui_prefix);
      mpi->comm.barrier();

      // cRPA from DFT Green's function
      MBState mb_state(ft, coqui_prefix, mf, wannier_file, true);
      ptree pt;
      pt.put("permut_symm", true);
      pt.put("force_real", true);
      pt.put("greens_func_source", "mf");
      embed_eri_t embed_2e(*mf, "gygi_smallest_q");
      embed_2e.downfolding_crpa(thc, mb_state, pt, "crpa");

      // Single-shot GW based on DFT Green's function
      [[maybe_unused]] auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                                      solvers::mb_solver_t(&hf,&gw,&scr_eri),
                                                      &iter_sol, 1, true, 1e-9, false);

      // DC from DFT Green's function; Fermionic Weiss field from DFT Green's function
      embed_t embed_1e(*mf, wannier_file, true);
      ptree pt_1e;
      pt_1e.put("update_dc", true);
      pt_1e.put("force_real", true);
      pt_1e.put("dc_type", dc_type);
      pt_1e.put("g_k_input", "scf");
      pt_1e.put("g_k_input_iter", 0);
      embed_1e.downfolding(mb_state, pt_1e);

      // check downfolded Hamiltonian
      std::string fname = coqui_prefix+".mbpt.h5";
      nda::array<ComplexType, 4> Vhf_gw_sIab;
      nda::array<ComplexType, 4> Vhf_dc_sIab;
      nda::array<ComplexType, 5> Sigma_gw_wsIab;
      nda::array<ComplexType, 5> Sigma_dc_wsIab;
      {
        h5::file file(fname, 'r');
        auto iter_grp = h5::group(file).open_group("downfold_1e/iter1");
        nda::h5_read(iter_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
        nda::h5_read(iter_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab);
      }
      app_log(2, "Vhf_dc_sIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Vhf_dc_sIab(0,0,0,0).real(), Vhf_dc_sIab(0,0,1,1).real(), Vhf_dc_sIab(0,0,0,1).real());
      VALUE_EQUAL(Vhf_dc_sIab(0,0,0,0), refs[0], eps);
      VALUE_EQUAL(Vhf_dc_sIab(0,0,1,1), refs[1], eps);
      VALUE_EQUAL(Vhf_dc_sIab(0,0,0,1), refs[2], eps);

      nda::array<ComplexType, 5> Sigma_tsIab(ft.nt_f(), Sigma_dc_wsIab.shape(1),
                                             Sigma_dc_wsIab.shape(2), Sigma_dc_wsIab.shape(3),
                                             Sigma_dc_wsIab.shape(4));
      ft.w_to_tau(Sigma_dc_wsIab, Sigma_tsIab, imag_axes_ft::fermion);
      app_log(2, "Sigma_dc_tsIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Sigma_tsIab(ft.nt_f()-1,0,0,0,0).real(), Sigma_tsIab(ft.nt_f()-1,0,0,1,1).real(),
              Sigma_tsIab(ft.nt_f()-1,0,0,0,1).real());
      VALUE_EQUAL(Sigma_tsIab(ft.nt_f()-1,0,0,0,0), refs[3], eps);
      VALUE_EQUAL(Sigma_tsIab(ft.nt_f()-1,0,0,1,1), refs[4], eps);
      VALUE_EQUAL(Sigma_tsIab(ft.nt_f()-1,0,0,0,1), refs[5], eps);
      mpi->comm.barrier();

      if (mpi->comm.root()) remove(fname.c_str());
      mpi->comm.barrier();
    };

    // the references are obtained in "sym_gw_dynamic_dc" cases
    SECTION("sym_gw_dynamic_dc") {
      std::array<double,6> refs = {0.994863460627, 0.392527791391, 0.003913713921,
                                    -0.205980066335, -0.086783668184, -0.000961596821};
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file, "gw_dynamic_u", refs, 1e-6);
    }
    SECTION("nosym_gw_dynamic_dc") {
      std::array<double,6> refs = {0.994863460627, 0.392527791391, 0.003913713921,
                                    -0.205980066335, -0.086783668184, -0.000961596821};
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file, "gw_dynamic_u", refs, 1e-5);
    }
  }

TEST_CASE("downfold_1e_mb_qp", "[methods][embed][df_1e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    double beta = 1000.0;
    double wmax = 1.2;

    std::string coqui_prefix = "downfold_1e_mb";
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");
    iter_scf::iter_scf_t iter_sol("damping");

    auto downfold = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file, std::string dc_type,
        std::array<double, 12> &refs, double eps) {

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi_smallest_q", coqui_prefix);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "gygi_smallest_q");
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(0, "", "incore", "", "bdft",
                                                 1e-8, mf->ecutrho(), 1, 1024, 10, 0.4));
      auto eri = mb_eri_t(thc, thc);
      auto psp = hamilt::make_pseudopot(*mf);
      write_mf_data(*mf, ft, *psp, coqui_prefix);
      mpi->comm.barrier();

      MBState mb_state(ft, coqui_prefix, mf, wannier_file, true);
      ptree pt;
      pt.put("permut_symm", true);
      pt.put("force_real", true);
      pt.put("greens_func_source", "mf");
      embed_eri_t embed_2e(*mf, "gygi_smallest_q");
      embed_2e.downfolding_crpa(thc, mb_state, pt, "crpa");

      [[maybe_unused]] auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                                      solvers::mb_solver_t(&hf,&gw,&scr_eri),
                                                      &iter_sol, 1, true, 1e-9, false);

      qp_params_t qp_params("sc", "pade", 18, 1e-8, 1e-8, "qpscf", false, "qp_energy");
      embed_t embed_1e(*mf, wannier_file, true);
      ptree pt_1e;
      pt_1e.put("update_dc", true);
      pt_1e.put("force_real", true);
      pt_1e.put("dc_type", dc_type);
      embed_1e.downfolding(mb_state, pt_1e, &qp_params);

      // check downfolded Hamiltonian
      std::string fname = coqui_prefix+".mbpt.h5";
      nda::array<ComplexType, 4> Vhf_gw_sIab;
      nda::array<ComplexType, 4> Vhf_dc_sIab;
      nda::array<ComplexType, 4> Vcorr_gw_sIab;
      nda::array<ComplexType, 4> Vcorr_dc_sIab;
      {
        h5::file file(fname, 'r');
        auto iter_grp = h5::group(file).open_group("downfold_1e/iter1");
        nda::h5_read(iter_grp, "Vhf_gw_sIab", Vhf_gw_sIab);
        nda::h5_read(iter_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
        nda::h5_read(iter_grp, "Vcorr_gw_sIab", Vcorr_gw_sIab);
        nda::h5_read(iter_grp, "Vcorr_dc_sIab", Vcorr_dc_sIab);
      }
      app_log(2, "Vhf_gw_sIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Vhf_gw_sIab(0,0,0,0).real(), Vhf_gw_sIab(0,0,1,1).real(), Vhf_gw_sIab(0,0,0,1).real());
      app_log(2, "Vhf_dc_sIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Vhf_dc_sIab(0,0,0,0).real(), Vhf_dc_sIab(0,0,1,1).real(), Vhf_dc_sIab(0,0,0,1).real());
      app_log(2, "Vcorr_gw_sIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Vcorr_gw_sIab(0,0,0,0).real(), Vcorr_gw_sIab(0,0,1,1).real(), Vcorr_gw_sIab(0,0,0,1).real());
      app_log(2, "Vcorr_dc_sIab: {0:.12f}, {1:.12f}, {2:.12f}",
              Vcorr_dc_sIab(0,0,0,0).real(), Vcorr_dc_sIab(0,0,1,1).real(), Vcorr_dc_sIab(0,0,0,1).real());
      VALUE_EQUAL(Vhf_gw_sIab(0,0,0,0), refs[0], eps);
      VALUE_EQUAL(Vhf_gw_sIab(0,0,1,1), refs[1], eps);
      VALUE_EQUAL(Vhf_gw_sIab(0,0,0,1), refs[2], eps);

      VALUE_EQUAL(Vhf_dc_sIab(0,0,0,0), refs[3], eps);
      VALUE_EQUAL(Vhf_dc_sIab(0,0,1,1), refs[4], eps);
      VALUE_EQUAL(Vhf_dc_sIab(0,0,0,1), refs[5], eps);

      VALUE_EQUAL(Vcorr_gw_sIab(0,0,0,0), refs[6], eps);
      VALUE_EQUAL(Vcorr_gw_sIab(0,0,1,1), refs[7], eps);
      VALUE_EQUAL(Vcorr_gw_sIab(0,0,0,1), refs[8], eps);

      VALUE_EQUAL(Vcorr_dc_sIab(0,0,0,0), refs[9], eps);
      VALUE_EQUAL(Vcorr_dc_sIab(0,0,1,1), refs[10], eps);
      VALUE_EQUAL(Vcorr_dc_sIab(0,0,0,1), refs[11], eps);
      mpi->comm.barrier();

      if (mpi->comm.root()) remove(fname.c_str());
      mpi->comm.barrier();
    };

    // the references are obtained using
    //    a) isdf threshold = 1e-8, chol_blk = 1
    //    b) Pade with Nfit = 18, eta = 1e-8, qp-eqn threshold = 1e-8
    //    c) with space-group symmetries activated.
    // AC seems to amplify the error coming from DFT w/ and w/o symmetry, resulting
    // in errors ~ 1e-3. The Wannier functions in the two cases are therefore not
    // exactly the same.
    // This is mainly because of the presence of very deep orbital (Li: 1s).
    SECTION("sym_gw_dc") {
      std::array<double,12> refs = {0.123474085511, -0.543810434672, 0.004800884640,
                                    1.345183158146, 0.613449576594, 0.004843167376,
                                    0.237810227112, 0.079665538571, 0.000517075666,
                                    0.0, 0.0, 0.0};
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file, "gw", refs, 1e-3);
    }

    SECTION("nosym_gw_dc") {
      std::array<double,12> refs = {0.123474085511, -0.543810434672, 0.004800884640,
                                    1.345183158146, 0.613449576594, 0.004843167376,
                                    0.237810227112, 0.079665538571, 0.000517075666,
                                    0.0, 0.0, 0.0};
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file, "gw", refs, 1e-3);
    }
  }

  TEST_CASE("downfold_gloc", "[methods][embed][df_1e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    double beta = 1000.0;
    double wmax = 3.0;

    std::string coqui_prefix = "downfold_gloc";
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");

    auto downfold = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file,
        std::array<double, 3> &refs, double eps) {
      auto psp = hamilt::make_pseudopot(*mf);
      write_mf_data(*mf, ft, *psp, coqui_prefix);
      mpi->comm.barrier();

      MBState mb_state(ft, coqui_prefix, mf, wannier_file, false);
      embed_t embed(*mf);
      auto gloc_tsIab = embed.downfold_gloc(mb_state, true, "scf", 0);

      app_log(2, "gloc: {0:.12f}, {1:.12f}, {2:.12f}",
              gloc_tsIab(ft.nt_f()-1,0,0,0,0).real(),
              gloc_tsIab(ft.nt_f()-1,0,0,1,1).real(),
              gloc_tsIab(ft.nt_f()-1,0,0,2,2).real());
      VALUE_EQUAL(gloc_tsIab(ft.nt_f()-1,0,0,0,0), refs[0], eps);
      VALUE_EQUAL(gloc_tsIab(ft.nt_f()-1,0,0,1,1), refs[1], eps);
      VALUE_EQUAL(gloc_tsIab(ft.nt_f()-1,0,0,2,2), refs[2], eps);
      mpi->comm.barrier();

      if (mpi->comm.root()) {
        std::string filename = coqui_prefix + ".mbpt.h5";
        remove(filename.c_str());
      }
      mpi->comm.barrier();
    };

    SECTION("sym_svo") {
      std::array<double,3> refs = {-0.166663724238, -0.166663724238, -0.166663724238};
      auto [outdir, prefix] = utils::utest_filename("qe_svo222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_svo222_sym"));
      std::string wannier_file = outdir + "/../mlwf/svo.mlwf.h5";
      downfold(mf, wannier_file, refs, 1e-10);
    }
  }

  TEST_CASE("downfold_Gloc_to_h5", "[methods][embed]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::ir_basis, "high", false);

    auto downfold = [&](mf::MF &mf, std::string wannier_file) {
      // write dft data
      auto psp = hamilt::make_pseudopot(mf);
      write_mf_data(mf, ft, *psp, "gloc");
      mpi->comm.barrier();

      // Read dft Green's function
      auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
          *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});

      projector_t proj(mf, wannier_file);
      {
        h5::file file("gloc.mbpt.h5", 'r');
        auto iter_grp = h5::group(file).open_group("scf/iter0");
        compute_G_from_mf(iter_grp, ft, sG_tskij);
      }
      mpi->comm.barrier();

      // Downfolding
      auto Gloc = proj.downfold_loc(sG_tskij, "Gloc");
      app_log(2, "Gloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}, {4:.12f}",
              Gloc(0,0,0,0,0).real(), Gloc(0,0,0,1,1).real(),
              Gloc(ft.nt_f()-1,0,0,0,0).real(), Gloc(ft.nt_f()-1,0,0,1,1).real(),
              Gloc(ft.nt_f()-1,0,0,0,1).real());
      VALUE_EQUAL(Gloc(0,0,0,0,0), 0.0, 1e-8);
      VALUE_EQUAL(Gloc(0,0,0,1,1), 0.0, 1e-8);
      VALUE_EQUAL(Gloc(ft.nt_f()-1,0,0,0,0), -0.988929386332, 1e-8);
      VALUE_EQUAL(Gloc(ft.nt_f()-1,0,0,1,1), -0.999056693835, 1e-8);
      VALUE_EQUAL(Gloc(ft.nt_f()-1,0,0,0,1), -0.000017119016, 1e-8);
      mpi->comm.barrier();

      if (mpi->comm.root()) remove("gloc.mbpt.h5");
      mpi->comm.barrier();
    };

    SECTION("sym") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = mf::default_MF(mpi, "qe_lih222_sym");
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file);
    }
    SECTION("nosym") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = mf::default_MF(mpi, "qe_lih222");
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold(mf, wannier_file);
    }

  }

  TEST_CASE("downfold_coulomb", "[methods][embed][df_2e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    std::string coqui_prefix = "downfold_coulomb";

    auto downfold = [&](
    std::shared_ptr<mf::MF> &mf, std::string wannier_file,
    std::array<double, 6> &refs, double eps) {

      int nIpts = mf->nbnd() * 15;
      std::string cd_dir = "";
      std::string storage = "incore";
      std::string save = "";
      std::string format = "bdft";
      double thresh = 1e-10;
      double ecut = mf->ecutrho();
      int chol_block_size = 1;
      int matrix_block_size = 1024;
      thc_reader_t thc(
        mf,
        make_thc_reader_ptree(
          nIpts, cd_dir, storage, save, format, thresh, ecut,
          chol_block_size, matrix_block_size)
      );

      ptree pt;
      pt.put("prefix", coqui_prefix);
      pt.put("wannier_file", wannier_file);
      pt.put("greens_func_source", "mf");
      pt.put("screen_type", "crpa");
      pt.put("div_treatment", "gygi_smallest_q");
      pt.put("beta", 1000.0);
      pt.put("wmax", 3.0);
      auto [Vloc, Wloc_w] = downfold_coulomb_with_projector_from_h5(thc, pt);

      // alpha = 10
      // Vloc: 0.602394239233, 0.552158246110, 0.552161743535
      // Wloc_w0: -0.420167997680, -0.411211138381, -0.411203307865
      // alpha = 15
      // Vloc: 0.602063603449, 0.552015228505, 0.552021449904
      // Wloc_w0: -0.420102061356, -0.411297369322, -0.411298486199
      // alpha = 20
      // Vloc: 0.602097182891, 0.552027879790, 0.552026756400
      //Wloc_w0: -0.420151990687, -0.411333804347, -0.411330635493
      app_log(2, "Vloc: {0:.12f}, {1:.12f}, {2:.12f}",
              Vloc(0, 0, 0, 0).real(), Vloc(0, 0, 1, 1).real(), Vloc(0, 0, 2, 2).real());

      VALUE_EQUAL(Vloc(0, 0, 0, 0), refs[0], eps);
      VALUE_EQUAL(Vloc(0, 0, 1, 1), refs[1], eps);
      VALUE_EQUAL(Vloc(0, 0, 2, 2), refs[2], eps);

      app_log(2, "Wloc_w0: {0:.12f}, {1:.12f}, {2:.12f}", 
              Wloc_w(0, 0, 0, 0, 0).real(), Wloc_w(0, 0, 0, 1, 1).real(), Wloc_w(0, 0, 0, 2, 2).real());

      VALUE_EQUAL(Wloc_w(0, 0, 0, 0, 0), refs[3], eps);
      VALUE_EQUAL(Wloc_w(0, 0, 0, 1, 1), refs[4], eps);
      VALUE_EQUAL(Wloc_w(0, 0, 0, 2, 2), refs[5], eps);

      mpi->comm.barrier();

      if (mpi->comm.root()) {
        std::string filename = coqui_prefix + ".mbpt.h5";
        remove(filename.c_str());
      }
    };

    SECTION("sym_svo") {
      // references obtatined from N_mu = 20 * N_orbs
      std::array<double,6> refs = {0.602097182891, 0.552027879790, 0.552026756400,
                                   -0.420151990687, -0.411333804347, -0.411330635493};
      auto [outdir, prefix] = utils::utest_filename("qe_svo222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_svo222_sym"));
      std::string wannier_file = outdir + "/../mlwf/svo.mlwf.h5";
      downfold(mf, wannier_file, refs, 1e-4);
    }
  }

  TEST_CASE("downfold_2e_crpa", "[methods][embed][df_2e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto downfold_crpa = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file) {
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));

      std::string prefix = "coqui";
      imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::ir_basis, "high", true);
      simple_dyson dyson(mf.get(), &ft);
      write_mf_data(*mf, ft, dyson, prefix);
      mpi->comm.barrier();

      nda::array<ComplexType, 4> Vloc;
      nda::array<ComplexType, 5> Wloc;
      nda::array<ComplexType, 5> Uloc;
      // downfold_2e with crpa mode
      MBState mb_state(ft, prefix, mf, wannier_file, true);
      ptree pt;
      pt.put("permut_symm", true);
      pt.put("force_real", true);
      pt.put("greens_func_source", "");
      embed_eri_t embed_2e(*mf, "gygi_smallest_q");
      embed_2e.downfolding_crpa(thc, mb_state, pt, "crpa");
      mpi->comm.barrier();

      long iter;
      h5::file file(prefix+".mbpt.h5", 'r');
      auto df_grp = h5::group(file).open_group("downfold_2e");
      h5::h5_read(df_grp, "final_iter", iter);
      auto iter_grp = df_grp.open_group("iter"+std::to_string(iter));
      nda::h5_read(iter_grp, "Vloc_abcd", Vloc);
      nda::h5_read(iter_grp, "Uloc_wabcd", Uloc);

      app_log(2, "Vloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Vloc(0,0,0,0).real(), Vloc(0,1,0,1).real(),
              Vloc(1,1,1,1).real(), Vloc(0,0,1,1).real());
      app_log(2, "Uloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Uloc(0,0,0,0,0).real(), Uloc(0,0,1,0,1).real(),
              Uloc(0,1,1,1,1).real(), Uloc(0,0,0,1,1).real());
      VALUE_EQUAL(Vloc(0,0,0,0), 1.416143628383, 1e-5);
      VALUE_EQUAL(Vloc(0,1,0,1), 0.000042865260, 1e-5);
      VALUE_EQUAL(Vloc(1,1,1,1), 0.555000665573, 1e-5);
      VALUE_EQUAL(Vloc(0,0,1,1), 0.254836731135, 1e-5);

      VALUE_EQUAL(Uloc(0,0,0,0,0), -0.350315268764, 1e-5);
      VALUE_EQUAL(Uloc(0,0,1,0,1), -0.000005850911, 1e-5);
      VALUE_EQUAL(Uloc(0,1,1,1,1), -0.220910992415, 1e-5);
      VALUE_EQUAL(Uloc(0,0,0,1,1), -0.115140041097, 1e-5);
      mpi->comm.barrier();

      if (mpi->comm.root()) {
        std::string filename = prefix + ".mbpt.h5";
        remove(filename.c_str());
      }
      mpi->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold_crpa(mf, wannier_file);
    }

    SECTION("sym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold_crpa(mf, wannier_file);
    }
  }

  TEST_CASE("compute_downfolded_coulomb_tensors", "[methods][embed][df_2e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto test_compute = [&](std::shared_ptr<mf::MF> &mf, std::string wannier_file) {
      int nIpts = mf->nbnd() * 20;
      std::string cd_dir = "";
      std::string storage = "incore";
      std::string save = "";
      std::string format = "bdft";
      double thresh = 1e-10;
      double ecut = mf->ecutrho();
      int chol_block_size = 1;
      int matrix_block_size = 1024;
      thc_reader_t thc(
        mf,
        make_thc_reader_ptree(
          nIpts, cd_dir, storage, save, format, thresh, ecut,
          chol_block_size, matrix_block_size)
      );

      std::string prefix = "coqui";
      double beta = 1000.0;
      double wmax = 1.2;
      std::string precision = "high";
      bool verbose = true;
      imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, precision, verbose);
      simple_dyson dyson(mf.get(), &ft);
      write_mf_data(*mf, ft, dyson, prefix);
      mpi->comm.barrier();

      bool translate_home_cell = true;
      MBState mb_state(ft, prefix, mf, wannier_file, translate_home_cell);
      bool force_permut_symm = true;
      bool force_real = true;
      bool write_to_hdf5 = false;
      bool q_dependent_output = false;
      std::string greens_func_source = "scf";
      long greens_func_iteration = 0;

      embed_eri_t embed_2e(*mf, "gygi_smallest_q");

      // --- RPA ---
      std::string screen_type_rpa = "rpa";
      auto [Vloc_rpa, Wloc_rpa] = embed_2e.compute_downfolded_coulomb_tensors(
        thc, mb_state, screen_type_rpa, force_permut_symm, force_real, &ft,
        greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output);

      app_log(2, "[RPA] Vloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Vloc_rpa(0,0,0,0).real(), Vloc_rpa(0,1,0,1).real(),
              Vloc_rpa(1,1,1,1).real(), Vloc_rpa(0,0,1,1).real());
      app_log(2, "[RPA] Wloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Wloc_rpa(0,0,0,0,0).real(), Wloc_rpa(0,0,1,0,1).real(),
              Wloc_rpa(0,1,1,1,1).real(), Wloc_rpa(0,0,0,1,1).real());

      // Compare Vloc_rpa to Vloc from downfold_2e_crpa
      VALUE_EQUAL(Vloc_rpa(0,0,0,0), 1.416143628383, 1e-5);
      VALUE_EQUAL(Vloc_rpa(0,1,0,1), 0.000042865260, 1e-5);
      VALUE_EQUAL(Vloc_rpa(1,1,1,1), 0.555000665573, 1e-5);
      VALUE_EQUAL(Vloc_rpa(0,0,1,1), 0.254836731135, 1e-5);
              
      // Compare Wloc_rpa to Wloc from downfold_2e_crpa
      VALUE_EQUAL(Wloc_rpa(0,0,0,0,0), -0.350315268764, 1e-5);
      VALUE_EQUAL(Wloc_rpa(0,0,1,0,1), -0.000005850911, 1e-5);
      VALUE_EQUAL(Wloc_rpa(0,1,1,1,1), -0.220910992415, 1e-5);
      VALUE_EQUAL(Wloc_rpa(0,0,0,1,1), -0.115140041097, 1e-5);

      // --- cRPA ---
      std::string screen_type_crpa = "crpa";
      auto [Vloc_crpa, Wloc_crpa] = embed_2e.compute_downfolded_coulomb_tensors(
        thc, mb_state, screen_type_crpa, force_permut_symm, force_real, &ft,
        greens_func_source, greens_func_iteration, write_to_hdf5, q_dependent_output);

      app_log(2, "[cRPA] Vloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Vloc_crpa(0,0,0,0).real(), Vloc_crpa(0,1,0,1).real(),
              Vloc_crpa(1,1,1,1).real(), Vloc_crpa(0,0,1,1).real());
      app_log(2, "[cRPA] Wloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Wloc_crpa(0,0,0,0,0).real(), Wloc_crpa(0,0,1,0,1).real(),
              Wloc_crpa(0,1,1,1,1).real(), Wloc_crpa(0,0,0,1,1).real());

      // Compare Vloc_rpa to Vloc from downfold_2e_crpa
      VALUE_EQUAL(Vloc_crpa(0,0,0,0), 1.416143628383, 1e-5);
      VALUE_EQUAL(Vloc_crpa(0,1,0,1), 0.000042865260, 1e-5);
      VALUE_EQUAL(Vloc_crpa(1,1,1,1), 0.555000665573, 1e-5);
      VALUE_EQUAL(Vloc_crpa(0,0,1,1), 0.254836731135, 1e-5);

      // Compare Wloc_crpa to Uloc from downfold_2e_crpa
      VALUE_EQUAL(Wloc_crpa(0,0,0,0,0), -0.350315268764, 1e-5);
      VALUE_EQUAL(Wloc_crpa(0,0,1,0,1), -0.000005850911, 1e-5);
      VALUE_EQUAL(Wloc_crpa(0,1,1,1,1), -0.220910992415, 1e-5);
      VALUE_EQUAL(Wloc_crpa(0,0,0,1,1), -0.115140041097, 1e-5);

      mpi->comm.barrier();

      if (mpi->comm.root()) {
        std::string filename = prefix + ".mbpt.h5";
        remove(filename.c_str());
      }
      mpi->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      test_compute(mf, wannier_file);
    }

    SECTION("sym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      test_compute(mf, wannier_file);
    }
  }

  TEST_CASE("downfold_2e_edmft", "[methods][embed][df_2e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto downfold_edmft = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file) {
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));

      std::string prefix = "coqui";
      imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::ir_basis, "high", true);
      simple_dyson dyson(mf.get(), &ft);
      write_mf_data(*mf, ft, dyson, prefix);
      mpi->comm.barrier();

      // downfold_2e with edmft mode
      MBState mb_state(ft, prefix, mf, wannier_file, true);
      ptree pt;
      pt.put("permut_symm", true);
      pt.put("force_real", true);
      // FIXME ?
      pt.put("greens_func_source", "mf");
      embed_eri_t embed_2e(*mf, "gygi_smallest_q");
      embed_2e.downfolding_edmft(thc, mb_state, pt, "gw_edmft");
      mpi->comm.barrier();

      nda::array<ComplexType, 4> Vloc;
      nda::array<ComplexType, 5> Wloc;
      nda::array<ComplexType, 5> Uloc;
      long iter;
      h5::file file(prefix+".mbpt.h5", 'r');
      auto df_grp = h5::group(file).open_group("downfold_2e");
      h5::h5_read(df_grp, "final_iter", iter);
      auto iter_grp = df_grp.open_group("iter"+std::to_string(iter));
      nda::h5_read(iter_grp, "Vloc_abcd", Vloc);
      nda::h5_read(iter_grp, "Wloc_wabcd", Wloc);
      nda::h5_read(iter_grp, "Uloc_wabcd", Uloc);

      app_log(2, "Vloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Vloc(0,0,0,0).real(), Vloc(0,1,0,1).real(),
              Vloc(1,1,1,1).real(), Vloc(0,0,1,1).real());
      app_log(2, "Wloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Wloc(0,0,0,0,0).real(), Wloc(0,0,1,0,1).real(),
              Wloc(0,1,1,1,1).real(), Wloc(0,0,0,1,1).real());
      app_log(2, "Uloc: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              Uloc(0,0,0,0,0).real(), Uloc(0,0,1,0,1).real(),
              Uloc(0,1,1,1,1).real(), Uloc(0,0,0,1,1).real());
      VALUE_EQUAL(Vloc(0,0,0,0), 1.416143628300, 1e-5);
      VALUE_EQUAL(Vloc(0,1,0,1), 0.000042865260, 1e-5);
      VALUE_EQUAL(Vloc(1,1,1,1), 0.555000665655, 1e-5);
      VALUE_EQUAL(Vloc(0,0,1,1), 0.254836731163, 1e-5);

      VALUE_EQUAL(Wloc(0,0,0,0,0), -0.350314225326, 1e-5);
      VALUE_EQUAL(Wloc(0,0,1,0,1), -0.000005850911, 1e-5);
      VALUE_EQUAL(Wloc(0,1,1,1,1), -0.220909949584, 1e-5);
      VALUE_EQUAL(Wloc(0,0,0,1,1), -0.115138998264, 1e-5);

      VALUE_EQUAL(Uloc(0,0,0,0,0), -0.350314225326, 1e-5);
      VALUE_EQUAL(Uloc(0,0,1,0,1), -0.000005850911, 1e-5);
      VALUE_EQUAL(Uloc(0,1,1,1,1), -0.220909949584, 1e-5);
      VALUE_EQUAL(Uloc(0,0,0,1,1), -0.115138998264, 1e-5);
      mpi->comm.barrier();

      if (mpi->comm.root()) {
        remove((prefix+".mbpt.h5").c_str());
      }
      mpi->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir+"/lih_wan.h5";
      downfold_edmft(mf, wannier_file);
    }

    SECTION("sym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir+"/lih_wan.h5";
      downfold_edmft(mf, wannier_file);
    }
  }

  TEST_CASE("downfold_model_cholesky", "[methods][embed][df_2e]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto downfold_chol = [&](
        std::shared_ptr<mf::MF> &mf, std::string wannier_file) {

      std::string prefix = "coqui";
      imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::ir_basis, "high", true);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi_smallest_q", prefix);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "gygi_smallest_q");
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);

      { // base calculation with no decomposition
        write_mf_data(*mf, ft, dyson, prefix);
        mpi->comm.barrier();

        // downfold_2e with bare mode
        MBState mb_state(ft, prefix, mf, wannier_file, true);
        ptree pt;
        pt.put("permut_symm", true);
        pt.put("force_real", false);
        pt.put("greens_func_source", "");
        embed_eri_t embed_2e(*mf, "gygi_smallest_q");
        embed_2e.downfolding_crpa(thc, mb_state, pt, "crpa", "none", 1e-8);
        mpi->comm.barrier();

        iter_scf::iter_scf_t iter_sol("damping");
        [[maybe_unused]] auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                                        solvers::mb_solver_t(&hf,&gw,&scr_eri),
                                                        &iter_sol, 1, true, 1e-9, false);

        qp_params_t qp_params("sc", "pade", 18, 1e-8, 1e-8, "qpscf", false, "qp_energy");
        embed_t embed(*mf, wannier_file, true);
        ptree pt_1e;
        pt_1e.put("update_dc", true);
        pt_1e.put("dc_type", "gw");
        pt_1e.put("force_real", false);
        embed.downfolding(mb_state, pt_1e, &qp_params);
        mpi->comm.barrier();
      }

      { // bare interaction with no factorization 
        write_mf_data(*mf, ft, dyson, prefix+".bare");
        mpi->comm.barrier();

        // downfold_2e with bare mode
        MBState mb_state(ft, prefix+".bare", mf, wannier_file, true);
        ptree pt;
        pt.put("permut_symm", true);
        pt.put("force_real", false);
        pt.put("greens_func_source", "");
        embed_eri_t embed_2e(*mf, "gygi_smallest_q", "gygi", "model_static");
        embed_2e.downfolding_crpa(thc, mb_state, pt, "bare", "none", 1e-8);
        mpi->comm.barrier();

        embed_t embed(*mf, wannier_file, true);
        embed.hf_downfolding("./", prefix + ".bare", thc, ft, false, "gygi");
        mpi->comm.barrier();
      }

      { // bare interaction with cholesky decomposition 
        write_mf_data(*mf, ft, dyson, prefix+".bare.chol");
        mpi->comm.barrier();

        // downfold_2e with bare mode
        MBState mb_state(ft, prefix+".bare.chol", mf, wannier_file, true);
        ptree pt;
        pt.put("permut_symm", true);
        pt.put("force_real", false);
        pt.put("greens_func_source", "");
        embed_eri_t embed_2e(*mf, "gygi_smallest_q", "gygi", "model_static");
        embed_2e.downfolding_crpa(thc, mb_state, pt, "bare", "cholesky", 1e-8);
        mpi->comm.barrier();

        embed_t embed(*mf, wannier_file, true);
        embed.hf_downfolding("./", prefix + ".bare.chol", thc, ft, false, "gygi");
        mpi->comm.barrier();
      }

      { // crpa screening with cholesky decomposition 
        write_mf_data(*mf, ft, dyson, prefix+".crpa.chol");
        mpi->comm.barrier();

        // downfold_2e with crpa mode
        MBState mb_state(ft, prefix+".crpa.chol", mf, wannier_file, true);
        ptree pt;
        pt.put("permut_symm", true);
        pt.put("force_real", false);
        pt.put("greens_func_source", "");
        embed_eri_t embed_2e(*mf, "gygi_smallest_q", "gygi", "model_static");
        embed_2e.downfolding_crpa(thc, mb_state, pt, "crpa", "cholesky", 1e-8);
        mpi->comm.barrier();

        iter_scf::iter_scf_t iter_sol("damping");
        [[maybe_unused]] auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                                        solvers::mb_solver_t(&hf,&gw,&scr_eri),
                                                        &iter_sol, 1, true, 1e-9, false);
        mpi->comm.barrier();

        qp_params_t qp_params("sc", "pade", 18, 1e-8, 1e-8, "qpscf", false, "qp_energy");
        mpi->comm.barrier();
        embed_t embed(*mf, wannier_file, true);
        ptree pt_1e;
        pt_1e.put("update_dc", true);
        pt_1e.put("dc_type", "gw");
        pt_1e.put("force_real", false);
        embed.downfolding(mb_state, pt_1e, &qp_params, "model_static");
        mpi->comm.barrier();
      }

      if(mpi->comm.root()) {

        nda::array<ComplexType, 4> Vloc_abcd;
        nda::array<ComplexType, 4> Uloc_abcd;
        nda::array<ComplexType, 4> Hgw_ref; 
        {
          h5::file file(prefix+".mbpt.h5", 'r');
          auto grp = h5::group(file);
          long iter = 1;
          nda::h5_read(grp, "downfold_2e/iter"+std::to_string(iter)+"/Vloc_abcd", Vloc_abcd);
          nda::array<ComplexType, 5> U_; 
          nda::h5_read(grp, "downfold_2e/iter"+std::to_string(iter)+"/Uloc_wabcd", U_);
          // add static contribution
          Uloc_abcd.resize(Vloc_abcd.shape());
          Uloc_abcd() = (Vloc_abcd() + U_(0,nda::ellipsis{}));

          nda::array<ComplexType, 4> h; 
          nda::h5_read(grp, "downfold_1e/iter"+std::to_string(iter)+"/H0_sIab",h);
          Hgw_ref = h; 
          nda::h5_read(grp, "downfold_1e/iter"+std::to_string(iter)+"/Vhf_gw_sIab",h);
          Hgw_ref() += h(); 
          nda::h5_read(grp, "downfold_1e/iter"+std::to_string(iter)+"/Vhf_dc_sIab",h); 
          Hgw_ref() -= h(); 
          nda::h5_read(grp, "downfold_1e/iter"+std::to_string(iter)+"/Vcorr_gw_sIab",h);
          Hgw_ref() += h(); 
          nda::h5_read(grp, "downfold_1e/iter"+std::to_string(iter)+"/Vcorr_dc_sIab",h); 
          Hgw_ref() -= h(); 
        }

        nda::array<ComplexType, 4> V_abcd;
        nda::array<ComplexType, 4> Hhf_ref;
        {
          h5::file file(prefix+".bare.model.h5", 'r');
          auto grp = h5::group(file);
          nda::h5_read(grp, "Interaction/Vq0", V_abcd);
          nda::h5_read(grp, "System/H0",Hhf_ref);
        }


        nda::array<ComplexType, 5> V5d;
        nda::array<ComplexType, 4> Hhf; 
        {
          h5::file file(prefix+".bare.chol.model.h5", 'r');
          auto grp = h5::group(file);
          nda::h5_read(grp, "Interaction/Vq0", V5d);
          nda::h5_read(grp, "System/H0",Hhf);
        }

        nda::array<ComplexType, 5> U5d;
        nda::array<ComplexType, 4> Hgw;       
        {
          h5::file file(prefix+".crpa.chol.model.h5", 'r');
          auto grp = h5::group(file);
          nda::h5_read(grp, "Interaction/Vq0", U5d);
          nda::h5_read(grp, "System/H0",Hgw);
        }

        {
          auto dH = Hhf_ref(0,0,nda::ellipsis{})-Hhf(0,0,nda::ellipsis{});
          auto fH0 = nda::frobenius_norm(dH)/double(dH.size());
          VALUE_EQUAL(fH0, 0.0);
          app_log(2,"downfold_model_cholesky hf: dH:{}",fH0);
        }

        {
          auto dH = Hgw_ref(0,0,nda::ellipsis{})-Hgw(0,0,nda::ellipsis{});
          auto fH0 = nda::frobenius_norm(dH)/double(dH.size());
          VALUE_EQUAL(fH0, 0.0);
          app_log(2,"downfold_model_cholesky gw: dH:{}",fH0);
        }
 
        long nI = Vloc_abcd.extent(0);

        // compare Vloc_abcd and V_abcd
        {
          auto Vloc_ab_cd = nda::reshape(Vloc_abcd, std::array<long,2>{nI*nI,nI*nI}); 
          auto V_ab_cd = nda::reshape(V_abcd, std::array<long,2>{nI*nI,nI*nI}); 
          auto mse = nda::frobenius_norm(Vloc_ab_cd()-V_ab_cd())/double(nI*nI*nI*nI);
          auto me = nda::sum(Vloc_ab_cd()-V_ab_cd())/double(nI*nI*nI*nI);
          app_log(2,"bare interaction no factorization: mse:{} me:{}",mse,me);
        }
        
        REQUIRE( std::array<long,4>{nI,nI,nI,nI} == Vloc_abcd.shape() );
        {
          long nP = V5d.extent(0);
          REQUIRE( std::array<long,5>{nP,1,1,nI,nI} == V5d.shape() );
          auto Vloc_nab = nda::reshape(V5d, std::array<long,2>{nP,nI*nI}); 
          auto Vloc_ab_cd = nda::reshape(Vloc_abcd, std::array<long,2>{nI*nI,nI*nI}); 
          nda::array<ComplexType, 2> Vloc_ncd(Vloc_nab); 
          for(long a=0, ab=0; a<nI; a++) 
            for(long b=0; b<nI; b++, ab++)
              Vloc_ncd(nda::range::all,ab) = nda::conj(Vloc_nab(nda::range::all,b*nI+a)); 

          nda::array<ComplexType, 2> V_(nI*nI,nI*nI); 
          nda::blas::gemm(ComplexType(1.0),nda::transpose(Vloc_nab),Vloc_ncd,ComplexType(0.0),V_);
 
          nda::blas::gemm(ComplexType(1.0),nda::transpose(Vloc_nab),Vloc_ncd,ComplexType(-1.0),Vloc_ab_cd);
          auto mse = nda::frobenius_norm(Vloc_ab_cd)/double(nI*nI*nI*nI); 
          auto me = nda::sum(Vloc_ab_cd)/double(nI*nI*nI*nI); 
          app_log(2,"bare interaction with cholesky: mse:{} me:{}",mse,me);
          VALUE_EQUAL(mse, 0.0);
          VALUE_EQUAL(me, 0.0);
        }
        {
          long nP = U5d.extent(0);
          REQUIRE( std::array<long,5>{nP,1,1,nI,nI} == U5d.shape() );
          auto Vloc_nab = nda::reshape(U5d, std::array<long,2>{nP,nI*nI});
          auto Vloc_ab_cd = nda::reshape(Uloc_abcd, std::array<long,2>{nI*nI,nI*nI});
          nda::array<ComplexType, 2> Vloc_ncd(Vloc_nab);
          for(long a=0, ab=0; a<nI; a++)
            for(long b=0; b<nI; b++, ab++)
              Vloc_ncd(nda::range::all,ab) = nda::conj(Vloc_nab(nda::range::all,b*nI+a));

          nda::array<ComplexType, 2> V_(nI*nI,nI*nI);
          nda::blas::gemm(ComplexType(1.0),nda::transpose(Vloc_nab),Vloc_ncd,ComplexType(0.0),V_);

          nda::blas::gemm(ComplexType(1.0),nda::transpose(Vloc_nab),Vloc_ncd,ComplexType(-1.0),Vloc_ab_cd);
          auto mse = nda::frobenius_norm(Vloc_ab_cd)/double(nI*nI*nI*nI);
          auto me = nda::sum(Vloc_ab_cd)/double(nI*nI*nI*nI);
          app_log(2,"crpa interaction with cholesky: mse:{} me:{}",mse,me);
          VALUE_EQUAL(mse, 0.0);
          VALUE_EQUAL(me, 0.0);
        }
        std::string filename = prefix + ".mbpt.h5";
        remove(filename.c_str());
        filename = prefix + ".bare.model.h5";
        remove(filename.c_str());
        filename = prefix + ".bare.chol.model.h5";
        remove(filename.c_str());
        filename = prefix + ".crpa.chol.model.h5";
        remove(filename.c_str());
        filename = prefix + ".bare.mbpt.h5";
        remove(filename.c_str());
        filename = prefix + ".bare.chol.mbpt.h5";
        remove(filename.c_str());
        filename = prefix + ".crpa.chol.mbpt.h5";
        remove(filename.c_str());
      }

      mpi->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold_chol(mf, wannier_file);
    }

    SECTION("sym_qe") {
      auto [outdir, prefix] = utils::utest_filename("qe_lih222_sym");
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih222_sym"));
      std::string wannier_file = outdir + "/lih_wan.h5";
      downfold_chol(mf, wannier_file);
    }
  }


} // bdft_tests
