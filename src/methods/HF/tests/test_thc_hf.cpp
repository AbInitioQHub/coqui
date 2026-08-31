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

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_communicator.hpp"
#include "utilities/mpi_context.h"

#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/default_MF.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/simple_dyson.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_driver.hpp"

namespace bdft_tests {

  using namespace methods;
  using mpi_context_t = utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator>;
  using array_view_4d_t = nda::array_view<ComplexType, 4>;
  using math::shm::make_shared_array;
  using math::shm::shared_array;

  TEST_CASE("thc_hf_qe_components", "[methods][thc][hf][qe]") {

    auto& mpi = utils::make_unit_test_mpi_context();

    auto eval_thc_hf = [&](std::shared_ptr<mf::MF> &mf) {
      thc_reader_t thc(mf, methods::make_thc_reader_ptree(0.0, "", "incore", "", "bdft", 1e-5,
                                                      0.4*mf->ecutrho()));
      solvers::hf_t hf("ignore_g0");

      long nspin = mf->nspin();
      long nkpts_ibz = mf->nkpts_ibz();
      long nbnd = mf->nbnd();
      auto mfocc = mf->occ();

      auto sS_skij = make_shared_array<array_view_4d_t>(*mpi, {nspin, nkpts_ibz, nbnd, nbnd});
      hamilt::set_ovlp(*mf, sS_skij);

      auto sJ = make_shared_array<array_view_4d_t>(*mpi, {nspin, nkpts_ibz, nbnd, nbnd});
      auto sK = make_shared_array<array_view_4d_t>(*mpi, {nspin, nkpts_ibz, nbnd, nbnd});
      auto sJK = make_shared_array<array_view_4d_t>(*mpi, {nspin, nkpts_ibz, nbnd, nbnd});


      nda::array<ComplexType,4> occ4d(nspin,nkpts_ibz,nbnd,nbnd);
      occ4d() = ComplexType(0.0);
      for( int s=0; s<nspin; s++ )
        for( int k=0; k<nkpts_ibz; k++ ) {
          for( int a=0; a<nbnd; a++ )
            occ4d(s,k,a,a) = mfocc(s,k,a);
        }
      hf.evaluate(sJ, occ4d, thc, sS_skij.local(), true, false);
      hf.evaluate(sK, occ4d, thc, sS_skij.local(), false, true);
      hf.evaluate(sJK, occ4d, thc, sS_skij.local(), true, true);

      app_log(2, "J(0,0,0,0) = {0:.8f}, K(0,0,0,0) = {1:.8f}, J+K(0,0,0,0) = {2:.8f}",
              sJ.local()(0, 0, 0, 0).real(), sK.local()(0, 0, 0, 0).real(), sJK.local()(0, 0, 0, 0).real());
      app_log(2, "J(0,0,0,1) = {0:.8f}, K(0,0,0,1) = {1:.8f}, J+K(0,0,0,1) = {2:.8f}",
              sJ.local()(0, 0, 0, 1).real(), sK.local()(0, 0, 0, 1).real(), sJK.local()(0, 0, 0, 1).real());

      if(mpi->node_comm.root())
        sJ.local() += sK.local();
      mpi->node_comm.barrier();
      utils::ARRAY_EQUAL(sJ.local(), sJK.local(), 1e-8);
    };

    SECTION("223_nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223"));
      eval_thc_hf(mf);
    }
    SECTION("223_sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_sym"));
      eval_thc_hf(mf);
    }
    SECTION("223_inv") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_lih223_inv"));
      eval_thc_hf(mf);
    }
    SECTION("GaAs222_so") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_GaAs222_so"));
      eval_thc_hf(mf);
    }

  }

  TEST_CASE("thc_hf_qe", "[methods][thc][hf][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis);

    auto solve_thc_hf = [&](
        std::shared_ptr<mf::MF> &mf, std::string cd_dir, double e0) {
      bool ls_thc = (cd_dir=="")? false : true;
      solvers::hf_t hf;
      /**
       * References are obtained from chol-hf with Cholesky tolerance = 1e-10
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      if (ls_thc)
        chol_reader_t chol_reader(mf,
                methods::make_chol_reader_ptree(1e-10, mf->ecutrho(), 32, cd_dir, "chol_info.h5", chol_reading_type_e::each_q));

      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, (ls_thc)? cd_dir : "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      simple_dyson dyson(mf.get(), &ft);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, e0, 1e-5);

      if (mpi_context->comm.root()) {
        remove("./bdft.mbpt.h5");
        if (ls_thc) {
          std::string info_name = cd_dir + "/chol_info.h5";
          remove(info_name.c_str());
          for (size_t ik = 0; ik < mf->nqpts(); ++ik) {
            std::string fname = cd_dir+"/Vq"+std::to_string(ik)+".h5";
            remove(fname.c_str());
          }
        }
      }
      mpi_context->comm.barrier();
    };

    SECTION("222_nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_hf(mf,"",-4.2818278244126935);
    }
    SECTION("222_sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_hf(mf,"",-4.2818278244126935);
    }
    SECTION("223_nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223"));
      solve_thc_hf(mf,"",-4.287485045424232);
    }

    SECTION("223_sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_sym"));
      solve_thc_hf(mf,"",-4.287485045424232);
    }

    SECTION("223_inv") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_inv"));
      solve_thc_hf(mf,"",-4.287485045424232);
    }
    SECTION("ls_thc_nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_hf(mf, "./",-4.2818278244126935);
    }

  }

  TEST_CASE("thc_qphf_qe", "[methods][thc][hf][qe]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 1.2, imag_axes_ft::ir_basis);

    auto solve_thc_qphf = [&](std::shared_ptr<mf::MF> &mf) {
      solvers::hf_t hf;
      /**
       * References are obtained from chol-hf with Cholesky tolerance = 1e-10
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params;
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                solvers::mb_solver_t(&hf), &iter_sol,
                                1, false, 1e-9);
      VALUE_EQUAL(e_hf, -4.2818278244126935, 1e-5);

      nda::array<ComplexType, 3> E_ska;
      {
        h5::file file("./bdft.mbpt.h5", 'r');
        auto scf_grp = h5::group(file).open_group("scf");
        auto iter_grp = scf_grp.open_group("iter1");
        nda::h5_read(iter_grp, "E_ska", E_ska);
      }
      mpi_context->comm.barrier();

      /**
       * Reference value is obtained from chol-hf with ERIs converge to 1e-10.
       * The accuracy is roughly 1e-5 at alpha=20 for this system
       **/
      int homo = int(mf->nelec()/2 - 1);
      int lumo = int(mf->nelec()/2);
      VALUE_EQUAL(E_ska(0,0,homo-1).real(), -2.127141508, 1e-5);
      VALUE_EQUAL(E_ska(0,0,homo).real(), -0.3701777979, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo).real(), 0.8441697097, 1e-5);
      VALUE_EQUAL(E_ska(0,0,lumo+1).real(), 0.8911358066, 1e-5);

      VALUE_EQUAL(E_ska(0,1,homo-1).real(), -2.1247920543, 1e-5);
      VALUE_EQUAL(E_ska(0,1,homo).real(), -0.2494370989, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo).real(), 0.3769966973, 1e-5);
      VALUE_EQUAL(E_ska(0,1,lumo+1).real(), 0.7321295428, 1e-5);

      if (mpi_context->comm.root()) {
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    };

    SECTION("nosym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
      solve_thc_qphf(mf);
    }
    SECTION("sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_qphf(mf);
    }
  }

  TEST_CASE("thc_hf_pyscf", "[methods][thc][hf][pyscf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_si222"));
    imag_axes_ft::IAFT ft(5000, 2.4, imag_axes_ft::ir_basis);
    solvers::hf_t hf;

    /**
     * References are obtained from PySCF in the zero-temperature formalism.
     * The accuracy is roughly 1e-6 at alpha=25 for this system.
     **/
    { // incore thc-hf from Dyson scf
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      simple_dyson dyson(mf.get(), &ft);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, 0.8730537612681228, 1e-6);
      mpi_context->comm.barrier();

      nda::array<ComplexType, 4> F;
      nda::array<ComplexType, 4> H0;
      {
        int it;
        h5::file file("bdft.mbpt.h5", 'r');
        h5::group grp(file);
        auto sys_grp = grp.open_group("system");
        nda::h5_read(sys_grp, "H0_skij", H0);

        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", it);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(it));
        nda::h5_read(iter_grp, "F_skij", F);
      }

      auto F_2D = nda::reshape(F, shape_t<2>{mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd()});
      auto H0_2D = nda::reshape(H0, shape_t<2>{mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd()});
      nda::matrix<ComplexType> H0pF(mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd());
      H0pF = H0_2D + F_2D;
      auto norm = nda::frobenius_norm(H0pF);
      VALUE_EQUAL(norm, 2.7073865879, 1e-6);
      mpi_context->comm.barrier();
    }

    { // outcore thc-hf from Dyson scf
      simple_dyson dyson(mf.get(), &ft);
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "outcore", "./thc_eri.h5", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     1, false, 1e-9, false);
      mpi_context->comm.barrier();
      VALUE_EQUAL(e_hf, 0.8730537612681228, 1e-6);

      nda::array<ComplexType, 4> F;
      nda::array<ComplexType, 4> H0;
      {
        int it;
        h5::file file("bdft.mbpt.h5", 'r');
        h5::group grp(file);
        auto sys_grp = grp.open_group("system");
        nda::h5_read(sys_grp, "H0_skij", H0);

        auto scf_grp = grp.open_group("scf");
        h5::h5_read(scf_grp, "final_iter", it);
        auto iter_grp = scf_grp.open_group("iter" + std::to_string(it));
        nda::h5_read(iter_grp, "F_skij", F);
      }
      auto F_2D = nda::reshape(F, shape_t<2>{mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd()});
      auto H0_2D = nda::reshape(H0, shape_t<2>{mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd()});
      nda::matrix<ComplexType> H0pF(mf->nspin()*mf->nkpts_ibz(), mf->nbnd()*mf->nbnd());
      H0pF = H0_2D + F_2D;
      auto norm = nda::frobenius_norm(H0pF);
      VALUE_EQUAL(norm, 2.7073865879, 1e-6);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) {
        remove("./thc_eri.h5");
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    }

    { // incore thc-hf from QP scf
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*25, "", "incore", "", "bdft",
                                                 1e-10, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      qp_params_t qp_params;
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                solvers::mb_solver_t(&hf), &iter_sol,
                                1, false, 1e-9);
      VALUE_EQUAL(e_hf, 0.8730537612681228, 1e-6);
      mpi_context->comm.barrier();

      if (mpi_context->comm.root()) {
        remove("./bdft.mbpt.h5");
      }
      mpi_context->comm.barrier();
    }

  }

  TEST_CASE("thc_hf_mol", "[methods][thc][hf][pyscf][mol]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(2000, 6.0, imag_axes_ft::ir_basis);

    auto solve_gdf_thc_hf = [&](std::shared_ptr<mf::MF> &mf, std::string gdf_dir) {
      solvers::hf_t hf;
      /**
       * References are obtained from pyscf RHF with the same GDF-ERIs
       * The accuracy is roughly 1e-4 at Np=280 (alpha~11.67) for this system.
       **/
      thc_reader_t thc(mf, make_thc_reader_ptree(280, gdf_dir, "incore", "", "bdft",
                                                 0.0, mf->ecutrho(), 1, 1024));
      auto eri = mb_eri_t(thc, thc);
      simple_dyson dyson(mf.get(), &ft);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "bdft");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, -84.85778159535779, 1e-4);

      if (mpi_context->comm.root())
        remove("./bdft.mbpt.h5");
      mpi_context->comm.barrier();
    };

    std::string gdf_dir = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/h2o_mol/gdf_eri/";
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "pyscf_h2o_mol"));
    solve_gdf_thc_hf(mf, gdf_dir);
  }

#ifdef ENABLE_DLR
  TEST_CASE("thc_hf_dlr_vs_ir", "[methods][thc][hf][qe][iaft][dlr][ir]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    imag_axes_ft::IAFT ft(1000, 10.0, imag_axes_ft::dlr_basis);

    auto solve_thc_hf = [&](std::shared_ptr<mf::MF> &mf, double e0) {
      solvers::hf_t hf;
      /**
       * References are obtained from chol-hf with Cholesky tolerance = 1e-10 wiht IR basis
       * The accuracy is roughly 1e-5 at alpha=20 for this system.
       **/
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*20, "", "incore", "", "bdft", 1e-10, mf->ecutrho(),
                       1, 1024));
      auto eri = mb_eri_t(thc, thc);
      simple_dyson dyson(mf.get(), &ft);
      iter_scf::iter_scf_t iter_sol("damping");
      MBState mb_state(mpi_context, ft, "coqui");
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf), &iter_sol,
                                     1, false, 1e-9, false);
      VALUE_EQUAL(e_hf, e0, 1e-5);

      if (mpi_context->comm.root()) {
        remove("./coqui.mbpt.h5");
      }
      mpi_context->comm.barrier();
    };

    SECTION("222_sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222_sym"));
      solve_thc_hf(mf, -4.2818278244126935);
    }

    SECTION("223_sym") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih223_sym"));
      solve_thc_hf(mf, -4.287485045424232);
    }
  }
#endif

} // bdft_tests
