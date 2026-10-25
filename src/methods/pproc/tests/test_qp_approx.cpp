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

#include <cstdio>

#include "utilities/test_common.hpp"
#include "mean_field/default_MF.hpp"
#include "nda/h5.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/eri_utils.hpp"
#include "methods/SCF/scf_driver.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/qp/linearized_qp.hpp"
#include "methods/pproc/pproc_t.h"

namespace bdft_tests {

  using namespace methods;

  /**
   * One Dyson G0W0 iteration of LiH 2x2x2 on ft, written to {prefix}.mbpt.h5:
   * system/H0_skij and scf/iter1/{F_skij, Sigma_tskij, mu}, the input of every routine below.
   */
  void write_g0w0_checkpoint(auto& mpi_context,
                             imag_axes_ft::IAFT& ft, std::string const& prefix) {
    auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi_context, "qe_lih222"));
    simple_dyson dyson(mf.get(), &ft);
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", prefix);
    solvers::scr_coulomb_t scr_eri(&ft, "rpa", "ignore_g0");
    thc_reader_t thc(mf, make_thc_reader_ptree(mf->nbnd()*8, "", "incore", "", prefix,
                                               1e-8, mf->ecutrho(), 1, 512));
    auto eri = mb_eri_t(thc, thc);
    iter_scf::iter_scf_t damp(iter_scf::damp_t(1.0));
    MBState mb_state(mpi_context, ft, prefix);
    scf_loop(mb_state, dyson, eri, ft, solvers::mb_solver_t(&hf, &gw, &scr_eri), &damp,
             1, false, 1e-8, true);
    mpi_context->comm.barrier();
  }

  // band_interpolation's request for one scheme; the other members are not read by the routines
  // under test
  qp_context_t lqp_request(lqp::params_t const& lqp) {
    qp_context_t qp; qp.qp_approx = "lqp"; qp.lqp = lqp;
    return qp;
  }
  qp_context_t qp_eqn_request(qp_eqn::params_t const& qp_eqn) {
    qp_context_t qp; qp.qp_approx = "qp_eqn"; qp.qp_eqn = qp_eqn;
    return qp;
  }

  void remove_checkpoint(auto& mpi_context,
                         std::string const& fname) {
    mpi_context->comm.barrier();
    if (mpi_context->comm.root()) std::remove(fname.c_str());
    mpi_context->comm.barrier();
  }

  TEST_CASE("pproc_linearized_qp_checkpoint", "[methods][mbpt][pproc][lqp]") {
    // linearized_qp_checkpoint writes the result under the group it read from, 
    // and keeps nothing of an earlier run in qp_approx/.
    //
    // Setup: one G0W0 iteration of LiH. 
    // 
    // Cases (2-5 continue on the file left by the case before, so they run in order):
    //   1. Mesh, single fit at n_fit = 4.
    //      - qp_approx/{E_ska, Heff_skij, Zqp_ska} and qp_approx/lqp/Z_skab match the kernel.
    //      - lqp/ stores the requested controls, fit_order = -1 stored as 7.
    //      - scheme is "lqp" and there is no ladder/.
    //   2. Mesh, ladder up to n_fit = 6.
    //      - qp_approx/ holds the ladder's last fit; lqp/n_fit and lqp/ladder/n_accepted match.
    //   3. Mesh, single fit after the ladder.
    //      - The stale ladder/ is gone.
    //   4. Mesh, single fit after a qp_eqn band_interpolation run.
    //      - The leftovers, written by hand, are gone: qp_approx/qp_eqn/ and ac_solver, and the
    //        bands interpolated from the previous Heff_skij, qp_approx/wannier_inter/.
    //   5. Kpath, the same arrays placed under wannier_inter/.
    //      - wannier_inter/qp_approx/ holds the same result in the same layout.
    using utils::ARRAY_EQUAL;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::dlr_basis);
    std::string prefix = "pproc_lqp_checkpoint", fname = prefix + ".mbpt.h5";
    write_g0w0_checkpoint(mpi_context, ft, prefix);

    nda::array<ComplexType, 4> F, H0;
    nda::array<ComplexType, 5> Sigma;
    double mu;
    {
      h5::file file(fname, 'r');
      auto root = h5::group(file);
      nda::h5_read(root.open_group("system"), "H0_skij", H0);
      auto it = root.open_group("scf/iter1");
      nda::h5_read(it, "F_skij", F); 
      nda::h5_read(it, "Sigma_tskij", Sigma);
      h5::h5_read(it, "mu", mu);
    }
    F += H0;
    lqp::params_t const params{.n_fit = 4, .fit_order = -1, .symmetric_window = true,
                               .fit_resid_tol = 1e-8};
    auto res = lqp::to_owned(lqp::linearized_qp_solve(*mpi_context, F, Sigma, mu, ft, params,
                                                      lqp::on_failure_e::abort));

    // 1. consistency between linearized_qp_checkpoint (single fit) and linearized_qp_solve. 
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, params, 0);
    {
      nda::array<double, 3> E, Zqp;
      nda::array<ComplexType, 4> Heff, Z;
      std::string scheme;
      int n_fit, fit_order;
      bool symmetric_window;
      double fit_resid_tol;
      h5::file file(fname, 'r');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      auto lg = qg.open_group("lqp");
      nda::h5_read(qg, "E_ska", E); nda::h5_read(qg, "Heff_skij", Heff);
      nda::h5_read(qg, "Zqp_ska", Zqp); h5::h5_read(qg, "scheme", scheme);
      nda::h5_read(lg, "Z_skab", Z); h5::h5_read(lg, "n_fit", n_fit);
      h5::h5_read(lg, "fit_order", fit_order); h5::h5_read(lg, "symmetric_window", symmetric_window);
      h5::h5_read(lg, "fit_resid_tol", fit_resid_tol);
      ARRAY_EQUAL(E, res.E_ska, 1e-10);
      ARRAY_EQUAL(Heff, res.Hqp_skab, 1e-10);
      ARRAY_EQUAL(Zqp, res.Zqp_ska, 1e-10);
      ARRAY_EQUAL(Z, res.Z_skab, 1e-10);
      CHECK(scheme == "lqp");
      CHECK(n_fit == 4); CHECK(fit_order == 7); CHECK(symmetric_window);
      CHECK(fit_resid_tol == 1e-8);
      CHECK(not lg.has_subgroup("ladder"));
    }
    mpi_context->comm.barrier();

    // 2. consistency between linearized_qp_checkpoint (ladder) and linearized_qp_ladder. 
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, params, 6);
    auto lad = lqp::to_owned(lqp::linearized_qp_ladder(*mpi_context, F, Sigma, mu, ft, 6,
                                                       params.fit_resid_tol,
                                                       params.symmetric_window));
    {
      nda::array<double, 3> E;
      int n_fit, n_accepted;
      h5::file file(fname, 'r');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      nda::h5_read(qg, "E_ska", E);
      h5::h5_read(qg.open_group("lqp"), "n_fit", n_fit);
      h5::h5_read(qg.open_group("lqp/ladder"), "n_accepted", n_accepted);
      ARRAY_EQUAL(E, lad.last.E_ska, 1e-10);
      CHECK(n_fit == lad.last.n_fit);
      CHECK(n_accepted == lad.n_accepted);
    }
    mpi_context->comm.barrier();

    // 3. single fit after the ladder: lqp group is recreated and no ladder group survives. 
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, params, 0);
    {
      h5::file file(fname, 'r');
      CHECK(not h5::group(file).open_group("scf/iter1/qp_approx/lqp").has_subgroup("ladder"));
    }
    mpi_context->comm.barrier();

    // 4. a qp_eqn band_interpolation run's leftovers should be overwritten.
    if (mpi_context->comm.root()) {
      h5::file file(fname, 'a');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      h5::h5_write(qg.create_group("qp_eqn"), "solver", std::string("sc"));
      h5::h5_write(qg, "ac_solver", std::string("sc"));
      nda::h5_write(qg.create_group("wannier_inter"), "E_ska", res.E_ska, false);
    }
    mpi_context->comm.barrier();
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, params, 0);
    {
      h5::file file(fname, 'r');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      CHECK(not qg.has_subgroup("qp_eqn"));
      CHECK(not qg.has_dataset("ac_solver"));
      CHECK(not qg.has_subgroup("wannier_inter"));
    }
    mpi_context->comm.barrier();

    // 5. kpath mode 
    if (mpi_context->comm.root()) {
      h5::file file(fname, 'a');
      auto wg = h5::group(file).open_group("scf/iter1").create_group("wannier_inter");
      nda::h5_write(wg, "F_skab", F, false);              // F already includes H0 here
      nda::h5_write(wg, "Sigma_tskab", Sigma, false);
    }
    mpi_context->comm.barrier();
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "interpolated", "scf", 1, params, 0);
    {
      nda::array<double, 3> E; nda::array<ComplexType, 4> Heff, Z;
      h5::file file(fname, 'r');
      auto kg = h5::group(file).open_group("scf/iter1/wannier_inter/qp_approx");
      nda::h5_read(kg, "E_ska", E); nda::h5_read(kg, "Heff_skab", Heff);
      nda::h5_read(kg.open_group("lqp"), "Z_skab", Z);
      ARRAY_EQUAL(E, res.E_ska, 1e-10);
      ARRAY_EQUAL(Heff, res.Hqp_skab, 1e-10);
      ARRAY_EQUAL(Z, res.Z_skab, 1e-10);
    }
    remove_checkpoint(mpi_context, fname);
  }

  TEST_CASE("pproc_qp_approx_from_dyson", "[methods][mbpt][pproc][qp_eqn]") {
    // qp_approx_from_dyson with qp_approx = "qp_eqn" replaces an earlier lqp result,
    // stores the controls it ran with, and gives the correct orbital energies
    //
    // Setup: one G0W0 iteration of LiH on an IR mesh, with an lqp result already written.
    //
    // Cases:
    //   1. Layout.
    //      - scheme is "qp_eqn"; qp_approx/qp_eqn/ holds the requested solver, ac_alg, ac_nfit,
    //        eta and tol.
    //      - The lqp result (Zqp_ska, lqp/) is gone.
    //   2. Energies.
    //      - E_ska of the four lowest bands at the first IBZ k-point matches values pinned on
    //        2026-10-24.
    using utils::VALUE_EQUAL;
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::ir_basis, "high");
    std::string prefix = "pproc_qp_eqn", fname = prefix + ".mbpt.h5";
    write_g0w0_checkpoint(mpi_context, ft, prefix);
    lqp::params_t const lqp_params{.n_fit = 4, .fit_order = -1, .symmetric_window = true,
                                   .fit_resid_tol = 1e-8};
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, lqp_params, 0);

    qp_eqn::params_t const params{.solver = "sc", .ac_alg = "pade", .ac_nfit = 18,
                                  .eta = 1e-4, .tol = 1e-8, .off_diag_mode = "fermi"};
    pproc_t pp(*mpi_context, prefix, "./");
    pp.qp_approx_from_dyson(qp_eqn_request(params), "scf", 1);
    mpi_context->comm.barrier();
    {
      nda::array<double, 3> E;
      std::string scheme, solver, ac_alg;
      int ac_nfit;
      double eta, tol;
      h5::file file(fname, 'r');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      auto pg = qg.open_group("qp_eqn");
      h5::h5_read(qg, "scheme", scheme);
      nda::h5_read(qg, "E_ska", E);
      h5::h5_read(pg, "solver", solver);
      h5::h5_read(pg, "ac_alg", ac_alg);
      h5::h5_read(pg, "ac_nfit", ac_nfit);
      h5::h5_read(pg, "eta", eta);
      h5::h5_read(pg, "tol", tol);
      // 1.
      CHECK(scheme == "qp_eqn");
      CHECK(solver == "sc");
      CHECK(ac_alg == "pade");
      CHECK(ac_nfit == 18);
      CHECK(eta == 1e-4); CHECK(tol == 1e-8);
      CHECK(not qg.has_dataset("Zqp_ska"));
      CHECK(not qg.has_subgroup("lqp"));
      // 2. pinned 2026-10-24; the same to ~2e-15 on 1, 2 and 4 ranks. The tolerance is 100x
      //    qp_eqn.tol: bisection stops at the first midpoint with |residual| <= tol, so a
      //    perturbation far below it may stop at a neighbouring midpoint, ~tol away; a broken
      //    input or continuation moves the energies by >= 1e-4 Ha.
      std::array<double, 4> const E_pinned = {-1.955088743564202e+00, -3.438826725760254e-01,
                                               7.690725924379263e-01,  8.093076551998136e-01};
      for (long a = 0; a < 4; ++a) VALUE_EQUAL(E(0, 0, a), E_pinned[a], 1e-6, 1e-6);
    }
    remove_checkpoint(mpi_context, fname);
  }

  TEST_CASE("pproc_qp_approx_reusable", "[methods][mbpt][pproc][lqp][qp_eqn]") {
    // qp_approx_reusable accepts a stored result only when its scheme and every stored
    // control match the request. 
    //
    // Setup: one G0W0 iteration of LiH
    //
    // Cases (2-4 continue on the file left by the case before, so they run in order):
    //   1. After one lqp fit at n_fit = 4.
    //      - Reusable: the same request, and the same with fit_order = 7 (what -1 resolves to).
    //      - Not reusable: another n_fit, fit_order, fit_resid_tol or symmetric_window; qp_eqn.
    //   2. After the ladder up to n_fit = 6, stored as a single fit at its last n_fit.
    //      - Reusable at that n_fit; not at the one below.
    //   3. After a qp_eqn run.
    //      - Reusable: the same request.
    //      - Not reusable: another eta or solver; lqp.
    //   4. The same file without scheme and qp_eqn/, like a checkpoint written before controls
    //      were stored.
    //      - Not reusable, even for the request of case 3.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 7.0, imag_axes_ft::dlr_basis);
    std::string prefix = "pproc_reusable", fname = prefix + ".mbpt.h5";
    write_g0w0_checkpoint(mpi_context, ft, prefix);
    auto reusable = [&fname](qp_context_t const& requested) {
      h5::file file(fname, 'r');
      return pproc_t::qp_approx_reusable(h5::group(file).open_group("scf/iter1"), requested);
    };
    lqp::params_t const lqp_params{.n_fit = 4, .fit_order = -1, .symmetric_window = true,
                                   .fit_resid_tol = 1e-8};
    qp_eqn::params_t const qp_eqn_params{.solver = "sc", .ac_alg = "pade", .ac_nfit = 18,
                                         .eta = 1e-4, .tol = 1e-8, .off_diag_mode = "fermi"};

    // 1.
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, lqp_params, 0);
    auto with = [&lqp_params](auto&& change) {
      lqp::params_t p = lqp_params; change(p); return lqp_request(p);
    };
    CHECK(reusable(lqp_request(lqp_params)));
    CHECK(reusable(with([](auto& p) { p.fit_order = 7; })));
    CHECK(not reusable(with([](auto& p) { p.n_fit = 5; })));
    CHECK(not reusable(with([](auto& p) { p.fit_order = 5; })));
    CHECK(not reusable(with([](auto& p) { p.fit_resid_tol = 1e-6; })));
    CHECK(not reusable(with([](auto& p) { p.symmetric_window = false; })));
    CHECK(not reusable(qp_eqn_request(qp_eqn_params)));
    mpi_context->comm.barrier();

    // 2.
    pproc_t::linearized_qp_checkpoint(*mpi_context, fname, "mesh", "scf", 1, lqp_params, 6);
    int last_n_fit;
    {
      h5::file file(fname, 'r');
      h5::h5_read(h5::group(file).open_group("scf/iter1/qp_approx/lqp"), "n_fit", last_n_fit);
    }
    REQUIRE(last_n_fit > 2);
    CHECK(reusable(with([last_n_fit](auto& p) { p.n_fit = last_n_fit; })));
    CHECK(not reusable(with([last_n_fit](auto& p) { p.n_fit = last_n_fit - 1; })));
    mpi_context->comm.barrier();

    // 3.
    pproc_t pp(*mpi_context, prefix, "./");
    pp.qp_approx_from_dyson(qp_eqn_request(qp_eqn_params), "scf", 1);
    mpi_context->comm.barrier();
    qp_eqn::params_t other_eta = qp_eqn_params;    other_eta.eta = 2e-4;
    qp_eqn::params_t other_solver = qp_eqn_params; other_solver.solver = "linearized";
    CHECK(reusable(qp_eqn_request(qp_eqn_params)));
    CHECK(not reusable(qp_eqn_request(other_eta)));
    CHECK(not reusable(qp_eqn_request(other_solver)));
    CHECK(not reusable(lqp_request(lqp_params)));
    mpi_context->comm.barrier();

    // 4.
    if (mpi_context->comm.root()) {
      h5::file file(fname, 'a');
      auto qg = h5::group(file).open_group("scf/iter1/qp_approx");
      qg.unlink("scheme"); qg.unlink("qp_eqn");
    }
    mpi_context->comm.barrier();
    CHECK(not reusable(qp_eqn_request(qp_eqn_params)));
    remove_checkpoint(mpi_context, fname);
  }

} // bdft_tests
