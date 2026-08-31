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

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"
#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"

#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"

// TODO add lih_223 unit tests

namespace bdft_tests {

  using namespace methods;

  TEST_CASE("thc_g0w0_qe_bdft", "[methods][thc][gw][qe][bdft]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis, "high");
    std::string output = "coqui";

    auto solve_thc_g0w0 = [&](std::shared_ptr<mf::MF> &mf) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*24, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params("sc", "pade", 18, 0.0001, 1e-8, "evscf");
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      [[maybe_unused]] double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                      solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol, 1, false, 1e-8);
      mpi_context->comm.barrier();

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file(output+".mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        auto iter_grp = scf_grp.open_group("iter1");
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      /**
       * Reference value is obtained from Chol-G0W0 with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=24 for this system in the presence of AC.
       **/
      int homo = int(mf->nelec()/2 - 1);
      int lumo = int(mf->nelec()/2);
      app_log(2, "E_ska at k = 0: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              E_ska(0,0,homo-1).real(), E_ska(0,0,homo).real(),
              E_ska(0,0,lumo).real(), E_ska(0,0,lumo+1).real());
      VALUE_EQUAL(E_ska(0,0,homo-1).real(), -1.959166853350, 1e-5);
      VALUE_EQUAL(E_ska(0,0,homo).real(), -0.343590135344, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo).real(), 0.769452793794, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo+1).real(), 0.819356108320, 1e-5);

      app_log(2, "E_ska at k = 1: {0:.12f}, {1:.12f}, {2:.12f}, {3:.12f}",
              E_ska(0,1,homo-1).real(), E_ska(0,1,homo).real(),
              E_ska(0,1,lumo).real(), E_ska(0,1,lumo+1).real());
      VALUE_EQUAL(E_ska(0,1,homo-1).real(), -1.949608656698, 1e-5);
      VALUE_EQUAL(E_ska(0,1,homo).real(), -0.234561625134, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo).real(), 0.332168314756, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo+1).real(), 0.691491471197, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym_qe") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_g0w0(mf);
    }
    SECTION("sym_qe") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_g0w0(mf);
    }
    SECTION("nosym_bdft") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "bdft_lih222"));
      solve_thc_g0w0(mf);
    }
    SECTION("sym_bdft") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "bdft_lih222_sym"));
      solve_thc_g0w0(mf);
    }
  }

  TEST_CASE("thc_gw_qe", "[methods][thc][gw][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_gw = [&](
      std::shared_ptr<mf::MF> &mf, double wmax, bool chol_eri_hf=false) {

      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::ir_basis, "high");
      std::string output = "coqui";

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);

      /**
       * Reference value is obtained from Chol-GW with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      double e_hf;
      double e_corr;
      if (!chol_eri_hf) {
        thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                   1e-10, mf->ecutrho(), 1, 1024));
        auto eri = mb_eri_t(thc, thc);
        iter_scf::iter_scf_t iter_sol("damping");
        std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                          solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                          1, false, 1e-9, true);
      } else {
        thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                   1e-10, mf->ecutrho(), 1, 1024));
        chol_reader_t chol(mf, methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, "./"));
        iter_scf::iter_scf_t iter_sol("damping");
        auto eri = mb_eri_t(chol, thc);
        std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                          solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                          1, false, 1e-9, true);
      }

      VALUE_EQUAL(e_hf, -4.224737908935479, 1e-5);
      VALUE_EQUAL(e_corr, -0.11256940748889475, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
        if (chol_eri_hf) {
          remove("chol_info.h5");
          for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
            std::string fname = "Vq"+std::to_string(ik)+".h5";
            remove(fname.c_str());
          }
        }
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_gw(mf, 1.2);
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_gw(mf, 1.2);
      solve_thc_gw(mf, 12.0);
    }
    SECTION("nosym_mix_eri") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_gw(mf, 1.2, true);
    }

  }

  TEST_CASE("thc_rpa_qe", "[methods][thc][rpa][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_rpa = [&](std::shared_ptr<mf::MF> &mf, double wmax) {
      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::ir_basis, "high");

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "gygi_smallest_q");

      /**
       * Reference value is obtained from Chol-RPA with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw));
      VALUE_EQUAL(e_rpa, -0.07295472568310496, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_rpa(mf, 1.2);
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_rpa(mf, 1.2);
      solve_thc_rpa(mf, 12.0);
    }
  }


  TEST_CASE("thc_gw_pyscf", "[methods][thc][gw][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string output = "coqui";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source));
    imag_axes_ft::IAFT ft(1000, 12.0, imag_axes_ft::ir_basis, "high");
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", output);

    /**
     * Reference value is obtained from Chol-GW with ERIs converge to 1e-10.
     * The accuracy is roughly 1e-6 at alpha=25 for this system.
     **/
    { // incore thc-gw
      MBState mb_state(mpi_context, ft, output);
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, true);

      // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
      VALUE_EQUAL(e_hf, 0.9096946909052888, 1e-6);
      VALUE_EQUAL(e_corr, -0.11439719195215467, 1e-6);
      mpi_context->comm.barrier();
    }

    { // outcore thc-gw from the precomputed thc-eri
      MBState mb_state(mpi_context, ft, output);
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, "outcore", "./thc_eri.h5");
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, true);

      // Reference value is obtained from Chol-GW with ERIs converge to 1e-10
      VALUE_EQUAL(e_hf, 0.9096946909052888, 1e-6);
      VALUE_EQUAL(e_corr, -0.11439719195215467, 1e-6);
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove((output+".mbpt.h5").c_str());
      }
    }
  }

  TEST_CASE("thc_rpa_pyscf", "[methods][thc][rpa][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf::pyscf_source));
    imag_axes_ft::IAFT ft(1000, 12.0, imag_axes_ft::ir_basis, "high");
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "gygi_smallest_q");
    solvers::mb_solver_t mb_solver(&hf, &gw);

    { // incore thc-rpa
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);
      VALUE_EQUAL(e_rpa, -0.06481111309877628, 1e-6);
      mpi_context->comm.barrier();
    }

    { // outcore thc-rpa
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "outcore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, "bdft");
      double e_rpa = rpa_loop(mb_state, dyson, eri, ft, mb_solver);
      VALUE_EQUAL(e_rpa, -0.06481111309877628, 1e-6);
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    }
  }

  TEST_CASE("thc_gw_mol", "[methods][thc][gw][pyscf][mol]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    std::string output = "coqui";
    imag_axes_ft::IAFT ft(2000, 6.0, imag_axes_ft::ir_basis, "high");

    auto solve_gdf_thc_gw = [&](std::shared_ptr<mf::MF> &mf, std::string gdf_dir) {
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      /**
       * References are obtained from the same GDF-THC-GW with alpha=12
       * The accuracy is roughly 1e-4 at Np=280 (alpha~11.67) for this system.
       **/
      thc_reader_t thc(mf, make_thc_reader_ptree(280, gdf_dir, "incore", "", "bdft",
                                                 0.0, mf->ecutrho(), 1, 1024));
      simple_dyson dyson(mf.get(), &ft);
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, output);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, -84.66602711500559, 1e-4);
      VALUE_EQUAL(e_corr, -0.41696395032933564, 1e-4);

      if (mpi_context->comm.root())
        remove((output+".mbpt.h5").c_str());
      mpi_context->comm.barrier();
    };

    std::string gdf_dir = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/h2o_mol/gdf_eri/";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2o_mol"));
    solve_gdf_thc_gw(mf, gdf_dir);
  }

#ifdef ENABLE_DLR
  TEST_CASE("thc_gw_dlr_vs_ir", "[methods][thc][gw][qe][iaft][dlr][ir]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto solve_thc_gw = [&](std::shared_ptr<mf::MF> &mf, double wmax) {

      imag_axes_ft::IAFT ft(1000, wmax, imag_axes_ft::dlr_basis, "high");
      std::string output = "coqui";

      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      simple_dyson dyson(mf.get(), &ft);
      MBState mb_state(mpi_context, ft, output);

      /**
       * Reference value is obtained from Chol-GW with ERIs converge to 1e-10 with IR basis.
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      double e_hf;
      double e_corr;
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd() * 20, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      std::tie(e_hf, e_corr) = scf_loop(mb_state, dyson, eri, ft,
                                        solvers::mb_solver_t(&hf,&gw,&scr_eri), &iter_sol,
                                        1, false, 1e-9, true);

      VALUE_EQUAL(e_hf, -4.224737908935479, 1e-5);
      VALUE_EQUAL(e_corr, -0.11256940748889475, 1e-5);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove((output+".mbpt.h5").c_str());
      }
      mpi_context->comm.barrier();
    };

    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_gw(mf, 10.0);
      solve_thc_gw(mf, 100.0);
    }
  }
#endif

} // bdft_tests
