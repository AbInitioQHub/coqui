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

#include "utilities/test_common.hpp"
#include "methods/tests/test_common.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/mf_utils.hpp"
#include "mean_field/default_MF.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "numerics/iter_scf/iter_scf_utils.hpp"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace methods;
  using utils::mpi_context_t;

  TEST_CASE("MO", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);
    double beta = 1000.0;

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);

    auto E_ski_ref = mf.eigval();
    ARRAY_EQUAL(sE_ski.local(), E_ski_ref, 1e-8);

    double Nelec = compute_Nelec(0.2, mf, sE_ski, beta);
    VALUE_EQUAL(Nelec, 8.0, 1e-9);
  }

  TEST_CASE("mu", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);
    double beta = 1000.0;

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);

    double mu_bisection = update_mu(0.0, mf, sE_ski, beta, 1e-9, "bisection");
    VALUE_EQUAL(mu_bisection, 0.175, 1e-9);

    double mu_midpoint = update_mu(0.0, mf, sE_ski, beta, 1e-9, "midpoint");
    VALUE_EQUAL(compute_Nelec(mu_midpoint, mf, sE_ski, beta), mf.nelec(), 1e-9);

    double mu_default = update_mu(0.0, mf, sE_ski, beta);
    VALUE_EQUAL(mu_default, mu_bisection, 1e-12);
  }

  TEST_CASE("qp_Dm_and_G", "[methods_scf]") {
    auto& context = utils::make_unit_test_mpi_context();

    std::string source_path = PROJECT_SOURCE_DIR;
    std::string filepath = source_path + "/tests/unit_test_files/pyscf/si_kp222_krhf/";
    auto mf = mf::make_MF(context, mf::pyscf_source, filepath, "pyscf");
    hamilt::pseudopot psp(mf);

    double beta = 1000;
    double wmax = 1.2;
    imag_axes_ft::IAFT ft(beta, wmax, imag_axes_ft::ir_basis, "high");

    auto [sMO_skij, sE_ski] = get_mf_MOs(*context, mf, psp);
    double mu = 0.175;
    double Nelec = compute_Nelec(mu, mf, sE_ski, ft.beta());
    VALUE_EQUAL(Nelec, 8.0, 1e-8);

    auto sDm_skij = math::shm::make_shared_array<Array_view_4D_t>(
        *context, {mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()});
    auto sG_tskij = math::shm::make_shared_array<Array_view_5D_t>(
        *context, {ft.nt_f(), mf.nspin(), mf.nkpts(), mf.nbnd(), mf.nbnd()});
    update_Dm(sDm_skij, sMO_skij, sE_ski, mu, ft.beta());
    update_G(sG_tskij, sMO_skij, sE_ski, mu, ft);
    ft.check_leakage(sG_tskij, imag_axes_ft::fermion, "Green's function");

    h5::file file(filepath+"/hf_Gw_Gt_beta1000_wmax1.2_high.h5", 'r');
    h5::group grp(file);

    nda::array<std::complex<double>, 5> G_tskij_ref;
    nda::array<std::complex<double>, 4> Dm_skij_ref;
    nda::h5_read(grp, "Dm", Dm_skij_ref);
    nda::h5_read(grp, "Gt", G_tskij_ref);

    ARRAY_EQUAL(sG_tskij.local(), G_tskij_ref, 1e-8);
    ARRAY_EQUAL(sDm_skij.local(), Dm_skij_ref, 1e-8);
  }

  TEST_CASE("qp_iterative_solver_diis_vs_damping", "[methods_scf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    solvers::hf_t hf;

    auto check_qphf_diis_vs_damping = [&](const std::string &mf_key,
                                          int thc_prefactor,
                                          double thc_tol,
                                          double e_hf_ref,
                                          double tol,
                                          std::string commutator_type,
                                          const std::string &name_tag) {
      imag_axes_ft::IAFT ft(1000.0, 1.2, imag_axes_ft::dlr_basis);
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, mf_key));

      auto run_qphf = [&](iter_scf::iter_scf_t &iter_sol, const std::string &output) -> double {
        thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*thc_prefactor, "", "incore", "", output,
                                                   thc_tol, mf->ecutrho(), 1, 1024));
        auto eri = mb_eri_t(thc, thc);
        qp_params_t qp_params;
        MBState mb_state(mpi_context, ft, output);
        double e_hf = qp_scf_loop(mb_state, eri, ft, qp_params,
                                  solvers::mb_solver_t(&hf), &iter_sol,
                                  50, false, 1e-7);
        mpi_context->comm.barrier();
        if (mpi_context->comm.root()) std::remove((output + ".mbpt.h5").c_str());
        mpi_context->comm.barrier();
        return e_hf;
      };

      iter_scf::iter_scf_t damp_sol(iter_scf::damp_t(0.5));
      iter_scf::iter_scf_t diis_sol(iter_scf::diis_t(0.5, 6, 2, commutator_type));

      double e_damping = run_qphf(damp_sol, "qphf_" + name_tag + "_damping_test");
      double e_diis    = run_qphf(diis_sol, "qphf_" + name_tag + "_diis_test");

      VALUE_EQUAL(e_damping, e_hf_ref, tol, tol);
      VALUE_EQUAL(e_diis,    e_hf_ref, tol, tol);
    };

    SECTION("qe_lih222") {
      // Orthogonal-basis regression.
      check_qphf_diis_vs_damping("qe_lih222", 12, 1e-12,
                                 -4.284215374096246, 1e-6, "commutator", "lih222");
    }

    SECTION("qe_lih222_vector_diff") {
      // Orthogonal-basis regression.
      check_qphf_diis_vs_damping("qe_lih222", 12, 1e-12,
                                 -4.284215374096246, 1e-6, "vector_diff", "lih222");
    }

    SECTION("pyscf_si222") {
      // Non-orthogonal-basis regression.
      check_qphf_diis_vs_damping("pyscf_si222", 12, 1e-10,
                                 0.8731465661058635, 1e-6, "commutator", "si222");
    }
  }

  TEST_CASE("dyson_scf_gw_diis_vs_damping", "[methods_scf]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();

    // Cheap setup for regression: lower beta/wmax and compact THC factors.
    imag_axes_ft::IAFT ft(200.0, 0.8, imag_axes_ft::dlr_basis);
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));

    auto run_dyson_gw = [&](iter_scf::iter_scf_t &iter_sol, const std::string &output)
        -> std::pair<double, double> {
      simple_dyson dyson(mf.get(), &ft);
      solvers::hf_t hf;
      solvers::gw_t gw(&ft, "ignore_g0", output);
      solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
      thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*8, "", "incore", "", output,
                                                 1e-8, mf->ecutrho(), 1, 512));
      auto eri = mb_eri_t(thc, thc);
      MBState mb_state(mpi_context, ft, output);
      auto [e_hf, e_corr] = scf_loop(mb_state, dyson, eri, ft,
                                     solvers::mb_solver_t(&hf, &gw, &scr_eri), &iter_sol,
                                     50, false, 1e-7, true);
      mpi_context->comm.barrier();
      if (mpi_context->comm.root()) std::remove((output + ".mbpt.h5").c_str());
      mpi_context->comm.barrier();
      return {e_hf, e_corr};
    };

    iter_scf::iter_scf_t damp_sol(iter_scf::damp_t(0.5));
    iter_scf::iter_scf_t diis_sol(iter_scf::diis_t(0.5, 6, 2));

    auto [e_hf_damp, e_corr_damp] = run_dyson_gw(damp_sol, "dyson_gw_damping_test");
    auto [e_hf_diis, e_corr_diis] = run_dyson_gw(diis_sol, "dyson_gw_diis_test");

    // Damping reference for this cheap setup; DIIS should converge to the same state.
    constexpr double e_hf_ref = -0.41698958999621144 + -3.7523191620415846;
    constexpr double e_corr_ref = -0.10985159447702234;
    constexpr double tol = 1e-6;
    VALUE_EQUAL(e_hf_damp, e_hf_ref, tol, tol);
    VALUE_EQUAL(e_corr_damp, e_corr_ref, tol, tol);
    VALUE_EQUAL(e_hf_diis, e_hf_ref, tol, tol);
    VALUE_EQUAL(e_corr_diis, e_corr_ref, tol, tol);
  }



} // bdft_tests
