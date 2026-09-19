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

#include <cmath>
#include <random>
#include <set>
#include <sstream>
#include <vector>

#include <boost/property_tree/json_parser.hpp>

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/test_common.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/linalg.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/qp/linearized_qp.hpp"
#include "methods/SCF/qp/qp_params_utils.hpp"

namespace bdft_tests {

  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  using namespace methods;

  namespace {
    // Sigma(iw) = sum_p V_p / (iw - e_p), V_p = u_p u_p^dag with complex u_p:
    // Hermitian, complex off-diagonal, positive semidefinite residues.
    struct pole_model_t {
      std::vector<double> e;
      std::vector<nda::array<ComplexType, 2>> V;
      long n;
    };

    pole_model_t make_pole_model(long n, int npole, unsigned seed) {
      pole_model_t m; m.n = n;
      std::mt19937 gen(seed);
      std::uniform_real_distribution<double> u(-1.0, 1.0);
      for (int p = 0; p < npole; ++p) {
        m.e.push_back((p % 2 == 0 ? 1.0 : -1.0) * (8.0 + 2.0 * p));   // |e| in [8, 8 + 2 npole]
        nda::array<ComplexType, 1> vec(n);
        for (long a = 0; a < n; ++a) vec(a) = ComplexType(u(gen), u(gen));
        nda::array<ComplexType, 2> V(n, n);
        for (long a = 0; a < n; ++a)
          for (long b = 0; b < n; ++b) V(a, b) = vec(a) * std::conj(vec(b));
        m.V.push_back(V);
      }
      return m;
    }

    // Sigma at CoQui Matsubara indices n_idx (w_n = n pi / beta)
    nda::array<ComplexType, 3> sigma_on_window(pole_model_t const& m, nda::array<long, 1> const& n_idx,
                                               double beta) {
      nda::array<ComplexType, 3> S(n_idx.shape(0), m.n, m.n);
      S() = 0.0;
      for (long j = 0; j < n_idx.shape(0); ++j) {
        ComplexType iw(0.0, double(n_idx(j)) * M_PI / beta);
        for (size_t p = 0; p < m.e.size(); ++p)
          S(j, nda::ellipsis{}) += m.V[p] / (iw - m.e[p]);
      }
      return S;
    }

    std::pair<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>> exact_AB(pole_model_t const& m) {
      nda::array<ComplexType, 2> A(m.n, m.n), B(m.n, m.n);
      A() = 0.0; B() = 0.0;
      for (size_t p = 0; p < m.e.size(); ++p) {
        A -= m.V[p] / m.e[p];
        B -= m.V[p] / (m.e[p] * m.e[p]);
      }
      return {A, B};
    }

    // Sigma(tau) on the IAFT fermionic mesh: a pole at e with weight V is V g_e(tau), with the
    // same g_e that update_G / compute_G0 use (qp_scf_common.cpp). tau_mesh() is x in [-1, 1].
    nda::array<ComplexType, 3> sigma_tau(pole_model_t const& m, imag_axes_ft::IAFT const& ft) {
      double beta = ft.beta();
      auto x = ft.tau_mesh();
      long nt = x.shape(0);
      nda::array<ComplexType, 3> S(nt, m.n, m.n);
      S() = 0.0;
      for (long it = 0; it < nt; ++it) {
        double tau = (x(it) + 1.0) * beta / 2.0;
        for (size_t p = 0; p < m.e.size(); ++p) {
          double e = m.e[p];
          double g = (e > 0) ? -std::exp(-e * tau) / (1.0 + std::exp(-e * beta))
                             : -std::exp(e * (beta - tau)) / (1.0 + std::exp(e * beta));
          S(it, nda::ellipsis{}) += m.V[p] * g;
        }
      }
      return S;
    }

    nda::array<ComplexType, 2> random_hermitian(long n, unsigned seed, double scale = 1.0) {
      std::mt19937 gen(seed); std::uniform_real_distribution<double> u(-1.0, 1.0);
      nda::array<ComplexType, 2> X(n, n);
      for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) X(a, b) = ComplexType(u(gen), u(gen));
      nda::array<ComplexType, 2> H(n, n);
      for (long a = 0; a < n; ++a)
        for (long b = 0; b < n; ++b) H(a, b) = 0.5 * scale * (X(a, b) + std::conj(X(b, a)));
      return H;
    }
    // negative semidefinite B = -Y Y^dag * scale (causal slope)
    nda::array<ComplexType, 2> causal_B(long n, unsigned seed, double scale = 0.3) {
      std::mt19937 gen(seed); std::uniform_real_distribution<double> u(-1.0, 1.0);
      nda::array<ComplexType, 2> Y(n, n);
      for (long a = 0; a < n; ++a) for (long b = 0; b < n; ++b) Y(a, b) = ComplexType(u(gen), u(gen));
      nda::array<ComplexType, 2> B(n, n);
      nda::blas::gemm(Y, nda::dagger(Y), B);
      B *= -scale;
      return B;
    }
    nda::array<ComplexType, 2> random_unitary(long n, unsigned seed) {
      auto H = random_hermitian(n, seed);
      auto [w, U] = nda::linalg::eigenelements(H);   // columns of U are orthonormal
      nda::array<ComplexType, 2> Ua(U);
      return Ua;
    }

    void check_fit_operator(imag_axes_ft::IAFT const& ft) {
      lqp::fit_params_t p; p.n_fit = 3;                 // 6 nodes, exact order 5
      auto op = lqp::make_fit_operator(ft, p);
      REQUIRE(op.n_idx.shape(0) == 6);
      REQUIRE(op.M.shape(0) == 6); REQUIRE(op.M.shape(1) == 6);
      REQUIRE(op.W.shape(0) == 6); REQUIRE(op.W.shape(1) == 6);
      REQUIRE(op.w_sel.shape(0) == 6);
      auto wn = ft.wn_mesh_f();
      for (long j = 0; j < 6; ++j) CHECK(long(wn(op.w_sel(j))) == op.n_idx(j));  // selection, not interpolation
      CHECK(op.exact);
      VALUE_EQUAL(op.omega_scale, 5.0 * M_PI / ft.beta(), 1e-14, 1e-14);
      // W M = 1 for the square system
      nda::array<ComplexType, 2> WM(6, 6);
      nda::blas::gemm(op.W, op.M, WM);
      ARRAY_EQUAL(WM, nda::eye<ComplexType>(6), 1e-10);
      // M_{j p} = (i x_j)^p with x_j = n_j / n_max
      VALUE_EQUAL(op.M(5, 1), ComplexType(0.0, 1.0), 1e-14, 1e-14);
      VALUE_EQUAL(op.M(2, 2), ComplexType(-1.0/25.0, 0.0), 1e-14, 1e-14);
      // parity decoupling on a symmetric window: even and odd columns orthogonal
      ComplexType g = 0.0;
      for (int j = 0; j < 6; ++j) g += std::conj(op.M(j, 0)) * op.M(j, 1);
      CHECK(std::abs(g) < 1e-14);

      // over-determined: 6 nodes, order 3
      lqp::fit_params_t q; q.n_fit = 3; q.fit_order = 3;
      auto oq = lqp::make_fit_operator(ft, q);
      CHECK_FALSE(oq.exact);
      REQUIRE(oq.W.shape(0) == 4); REQUIRE(oq.W.shape(1) == 6);
      nda::array<ComplexType, 2> WMq(4, 4);
      nda::blas::gemm(oq.W, oq.M, WMq);
      ARRAY_EQUAL(WMq, nda::eye<ComplexType>(4), 1e-10);   // left inverse
    }
  } // anonymous namespace

  TEST_CASE("lqp_fit_window", "[methods_qp]") {
    auto n = lqp::fit_window(3, true);
    REQUIRE(n.shape(0) == 6);
    CHECK(n(0) == -5); CHECK(n(1) == -3); CHECK(n(2) == -1);
    CHECK(n(3) ==  1); CHECK(n(4) ==  3); CHECK(n(5) ==  5);
    auto np = lqp::fit_window(3, false);
    REQUIRE(np.shape(0) == 3);
    CHECK(np(0) == 1); CHECK(np(2) == 5);
  }

  TEST_CASE("lqp_fit_operator", "[methods_qp]") {
    SECTION("dlr") {
      imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
      check_fit_operator(ft);
    }
    SECTION("ir") {
      // the kernel selects nodes from the sampling mesh, so it must be backend-agnostic
      imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::ir_basis);
      check_fit_operator(ft);
    }
  }

  TEST_CASE("lqp_coefficients_recover_analytic_A_and_B", "[methods_qp]") {
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    lqp::fit_params_t p; p.n_fit = 6;                   // 12 nodes, order 11
    auto op = lqp::make_fit_operator(ft, p);
    auto m  = make_pole_model(3, 4, 11);
    auto S  = sigma_on_window(m, op.n_idx, ft.beta());
    lqp::fit_diagnostics_t d;
    auto coeffs = lqp::low_freq_coefficients(S, op, d);
    REQUIRE(coeffs.shape(0) == 12); REQUIRE(coeffs.shape(1) == 3);
    auto [A, B] = exact_AB(m);
    ARRAY_EQUAL(coeffs(0, nda::ellipsis{}), A, 1e-8);
    ARRAY_EQUAL(coeffs(1, nda::ellipsis{}), B, 1e-6);
    CHECK(d.resid < 1e-10);            // exact interpolation
    CHECK(d.anti_herm_A < 1e-10);      // Hermitian by construction on +/- nodes
    CHECK(d.herm_data < 1e-12);        // S(iw)^dag == S(-iw) for the model
    // off-diagonal of A is genuinely complex for this model
    CHECK(std::abs(A(0, 1).imag()) > 1e-3);
  }

  TEST_CASE("lqp_coefficients_zero_sigma", "[methods_qp]") {
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    lqp::fit_params_t p; p.n_fit = 3;
    auto op = lqp::make_fit_operator(ft, p);
    nda::array<ComplexType, 3> S(op.n_idx.shape(0), 2, 2); S() = 0.0;
    lqp::fit_diagnostics_t d;
    auto c = lqp::low_freq_coefficients(S, op, d);
    ARRAY_EQUAL(c(0, nda::ellipsis{}), nda::zeros<ComplexType>(2, 2), 1e-14);
    ARRAY_EQUAL(c(1, nda::ellipsis{}), nda::zeros<ComplexType>(2, 2), 1e-14);
    CHECK(d.resid == 0.0);
  }

  TEST_CASE("lqp_matrix_generalized_eigenproblem_and_sum_rule", "[methods_qp]") {
    long n = 5;
    auto K = random_hermitian(n, 1);
    auto B = causal_B(n, 2);
    auto r = lqp::linearized_qp_matrix(K, B);
    REQUIRE(r.valid);
    REQUIRE(r.E.shape(0) == n);
    CHECK(r.min_eig >= 1.0 - 1e-12);                   // 1 - B >= 1 for causal B
    CHECK(std::abs(r.sumrule) < 1e-12);                 // sum Zqp == Tr Z
    for (long l = 0; l < n; ++l) { CHECK(r.Zqp(l) > 0.0); CHECK(r.Zqp(l) <= 1.0 + 1e-12); }
    // Z (1 - B) == 1: Z really is the inverse the kernel claims
    nda::array<ComplexType, 2> ImB(n, n), ZImB(n, n), eye(n, n);
    ImB = -B; eye() = 0.0;
    for (long a = 0; a < n; ++a) { ImB(a, a) += 1.0; eye(a, a) = 1.0; }
    nda::blas::gemm(r.Z, ImB, ZImB);
    ARRAY_EQUAL(ZImB, eye, 1e-12);
    // psi = Z^1/2 v solves K psi = eps (1 - B) psi. The kernel no longer keeps Z^1/2, so build
    // it here from Z's eigendecomposition.
    auto [w, U] = nda::linalg::eigenelements(r.Z);
    nda::array<ComplexType, 2> Uw(n, n), Zhalf(n, n), Psi(n, n), lhs(n, n), rhs(n, n);
    for (long l = 0; l < n; ++l) Uw(nda::range::all, l) = U(nda::range::all, l) * std::sqrt(w(l));
    nda::blas::gemm(Uw, nda::dagger(U), Zhalf);
    nda::blas::gemm(Zhalf, r.V, Psi);
    nda::blas::gemm(K, Psi, lhs);
    nda::blas::gemm(ImB, Psi, rhs);
    for (long l = 0; l < n; ++l) rhs(nda::range::all, l) *= r.E(l);
    ARRAY_EQUAL(lhs, rhs, 1e-11);
  }

  TEST_CASE("lqp_matrix_scalar_B_scales_energies", "[methods_qp]") {
    long n = 4; double b = -0.5;
    auto K = random_hermitian(n, 3);
    nda::array<ComplexType, 2> B(n, n); B() = 0.0;
    for (long a = 0; a < n; ++a) B(a, a) = b;
    auto r = lqp::linearized_qp_matrix(K, B);
    auto eK = nda::linalg::eigenvalues(K);
    double z = 1.0 / (1.0 - b);
    for (long l = 0; l < n; ++l) {
      VALUE_EQUAL(r.Zqp(l), z, 1e-12, 1e-12);
      VALUE_EQUAL(r.E(l), z * eK(l), 1e-11, 1e-11);
    }
  }

  TEST_CASE("lqp_matrix_gauge_invariance", "[methods_qp]") {
    long n = 5;
    auto K = random_hermitian(n, 4); auto B = causal_B(n, 5); auto U = random_unitary(n, 6);
    auto r0 = lqp::linearized_qp_matrix(K, B);
    nda::array<ComplexType, 2> tmp(n, n), Ku(n, n), Bu(n, n);
    nda::blas::gemm(K, U, tmp); nda::blas::gemm(nda::dagger(U), tmp, Ku);
    nda::blas::gemm(B, U, tmp); nda::blas::gemm(nda::dagger(U), tmp, Bu);
    auto r1 = lqp::linearized_qp_matrix(Ku, Bu);
    ARRAY_EQUAL(r0.E, r1.E, 1e-11);
    ARRAY_EQUAL(r0.Zqp, r1.Zqp, 1e-11);
  }

  TEST_CASE("lqp_matrix_degenerate_levels_keep_the_sum_rule", "[methods_qp]") {
    // K = 0 makes H_QP = 0, fully degenerate: the eigensolver returns an arbitrary orthonormal
    // basis and the individual <v|Z|v> are basis-dependent by construction (no blocking is
    // attempted). What must hold regardless: the sum rule, and every weight lying between the
    // extreme eigenvalues of Z (the diagonal of a Hermitian matrix is majorized by its spectrum).
    long n = 3;
    nda::array<ComplexType, 2> K(n, n); K() = 0.0;
    auto B = causal_B(n, 8);
    auto r = lqp::linearized_qp_matrix(K, B);
    REQUIRE(r.valid);
    CHECK(std::abs(r.sumrule) < 1e-12);
    auto zev = nda::linalg::eigenvalues(r.Z);
    for (long l = 0; l < n; ++l) { CHECK(r.Zqp(l) >= zev(0) - 1e-12); CHECK(r.Zqp(l) <= zev(n - 1) + 1e-12); }
  }

  TEST_CASE("lqp_matrix_flags_non_positive_definite", "[methods_qp]") {
    long n = 2;
    auto K = random_hermitian(n, 8);
    nda::array<ComplexType, 2> B(n, n); B() = 0.0; B(0, 0) = 1.5;   // 1 - B has eigenvalue -0.5
    auto r = lqp::linearized_qp_matrix(K, B);
    CHECK_FALSE(r.valid);
    VALUE_EQUAL(r.min_eig, -0.5, 1e-12, 1e-12);
  }

  TEST_CASE("lqp_solve_point_single_pole", "[methods_qp]") {
    // scalar pole e = 10, g^2 = 9: Z = 1 / (1 + g^2/e^2), A = -g^2/e, B = -g^2/e^2
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    lqp::fit_params_t p; p.n_fit = 6;
    auto op = lqp::make_fit_operator(ft, p);
    pole_model_t m; m.n = 1; m.e = {10.0};
    nda::array<ComplexType, 2> V(1, 1); V(0, 0) = 9.0; m.V = {V};
    auto St = sigma_tau(m, ft);
    nda::array<ComplexType, 2> F(1, 1); F(0, 0) = 0.3;
    double mu = 0.2;
    auto r = lqp::solve_point(F, St, mu, ft, op, p);
    VALUE_EQUAL(r.qp.Zqp(0), 1.0 / (1.0 + 9.0 / 100.0), 1e-6, 1e-6);
    VALUE_EQUAL(r.A(0, 0).real(), -9.0 / 10.0, 1e-6, 1e-6);
    VALUE_EQUAL(r.B(0, 0).real(), -9.0 / 100.0, 1e-6, 1e-6);
    // E = Z (F + A - mu)
    VALUE_EQUAL(r.qp.E(0), r.qp.Zqp(0) * (0.3 - 0.9 - 0.2), 1e-6, 1e-6);
    CHECK(r.diagnostics.resid < 1e-8);
  }

  TEST_CASE("lqp_solve_full_arrays_matches_points", "[methods_qp]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    lqp::fit_params_t p; p.n_fit = 6;
    auto op = lqp::make_fit_operator(ft, p);
    long ns = 1, nk = 3, n = 3;
    long nt = ft.nt_f();
    nda::array<ComplexType, 4> F(ns, nk, n, n);
    nda::array<ComplexType, 5> S(nt, ns, nk, n, n);
    std::vector<pole_model_t> models;
    for (long k = 0; k < nk; ++k) {
      models.push_back(make_pole_model(n, 3, 20 + k));
      F(0, k, nda::ellipsis{}) = random_hermitian(n, 30 + k, 0.5);
      auto St = sigma_tau(models.back(), ft);
      for (long it = 0; it < nt; ++it) S(it, 0, k, nda::ellipsis{}) = St(it, nda::ellipsis{});
    }
    double mu = 0.1;
    auto res = lqp::linearized_qp_solve(mpi_context->comm, F, S, mu, ft, p);
    REQUIRE(res.E_ska.shape(1) == nk);
    for (long k = 0; k < nk; ++k) {
      auto St = sigma_tau(models[k], ft);
      auto r = lqp::solve_point(F(0, k, nda::ellipsis{}), St, mu, ft, op, p);
      ARRAY_EQUAL(res.E_ska(0, k, nda::range::all),  r.qp.E,   1e-12);
      ARRAY_EQUAL(res.Zqp_ska(0, k, nda::range::all), r.qp.Zqp, 1e-12);
      ARRAY_EQUAL(res.Z_skab(0, k, nda::ellipsis{}), r.qp.Z,   1e-12);
      ARRAY_EQUAL(res.Hqp_skab(0, k, nda::ellipsis{}), r.qp.Hqp, 1e-12);
      auto [A, B] = exact_AB(models[k]);
      ARRAY_EQUAL(r.B, B, 1e-6);                       // the fit itself, per point
    }
    CHECK(res.n_fit == 6); CHECK(res.fit_order == 11);
  }

  /*
   * read_qp_block: the two-axis input (qp_approx + the ac/lqp blocks) and the deprecated
   * qp_type, which used to select the family and the AC solver with one string.
   */
  TEST_CASE("lqp_max_n_fit_on_mesh_is_the_contiguous_odd_run", "[methods_qp]") {
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    int nf = lqp::max_n_fit_on_mesh(ft, true);
    auto wn = ft.wn_mesh_f();
    std::set<long> on(wn.begin(), wn.end());
    REQUIRE(nf >= 2);
    for (int k = 1; k <= nf; ++k) { CHECK(on.count(2 * k - 1)); CHECK(on.count(-(2 * k - 1))); }
    CHECK(not (on.count(2 * (nf + 1) - 1) and on.count(-(2 * (nf + 1) - 1))));
    // one step beyond the ceiling is exactly where make_fit_operator refuses
    lqp::fit_params_t p; p.n_fit = nf;
    CHECK_NOTHROW(lqp::make_fit_operator(ft, p));
  }

  TEST_CASE("lqp_ladder_converges_to_the_analytic_Z", "[methods_qp]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    long ns = 1, nk = 2, n = 3, nt = ft.nt_f();
    nda::array<ComplexType, 4> F(ns, nk, n, n);
    nda::array<ComplexType, 5> S(nt, ns, nk, n, n);
    std::vector<pole_model_t> models;
    for (long k = 0; k < nk; ++k) {
      models.push_back(make_pole_model(n, 3, 40 + k));
      F(0, k, nda::ellipsis{}) = random_hermitian(n, 50 + k, 0.5);
      auto St = sigma_tau(models.back(), ft);
      for (long it = 0; it < nt; ++it) S(it, 0, k, nda::ellipsis{}) = St(it, nda::ellipsis{});
    }
    double mu = 0.1;
    lqp::fit_params_t p;                                 // fit_resid_tol = 1e-8 is the gate
    auto lad = lqp::linearized_qp_ladder(mpi_context->comm, F, S, mu, ft, p, 10);
    REQUIRE(lad.n_accepted >= 3);
    CHECK(lad.n_fit_history(0) == 2);
    CHECK(lad.n_fit_history(lad.n_accepted - 1) == lad.last.n_fit);
    CHECK(lad.Zqp_history.shape(0) == lad.n_accepted);
    // the last rung IS a plain solve at that n_fit
    lqp::fit_params_t q = p; q.n_fit = lad.last.n_fit; q.fit_order = -1;
    auto single = lqp::linearized_qp_solve(mpi_context->comm, F, S, mu, ft, q);
    ARRAY_EQUAL(lad.last.E_ska, single.E_ska, 1e-12);
    ARRAY_EQUAL(lad.last.Zqp_ska, single.Zqp_ska, 1e-12);
    // and it reproduces the closed form built from the exact A, B
    for (long k = 0; k < nk; ++k) {
      auto [A, B] = exact_AB(models[k]);
      nda::array<ComplexType, 2> K(F(0, k, nda::ellipsis{}) + A);
      for (long a = 0; a < n; ++a) K(a, a) -= mu;
      auto ex = lqp::linearized_qp_matrix(K, B);
      for (long a = 0; a < n; ++a) VALUE_EQUAL(lad.last.Zqp_ska(0, k, a), ex.Zqp(a), 1e-5, 1e-5);
      CHECK(lad.err_fit_sk(0, k) >= 0.0); CHECK(lad.err_fit_sk(0, k) < 1e-4);
      CHECK(lad.dHqp_sk(0, k) >= 0.0);
    }
    CHECK(lad.stopped_n_fit == 0);                       // the gate never fired at n_fit <= 10
    // whether the mesh or n_fit_max ended the climb depends on the IAFT's precision preset
    CHECK(lad.mesh_limited == (lad.n_fit_mesh_max < 10));
    CHECK(lad.last.n_fit == std::min(10, lad.n_fit_mesh_max));
  }

  TEST_CASE("lqp_ladder_is_capped_by_the_sampling_mesh", "[methods_qp]") {
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    long ns = 1, nk = 1, n = 3, nt = ft.nt_f();
    nda::array<ComplexType, 4> F(ns, nk, n, n);
    nda::array<ComplexType, 5> S(nt, ns, nk, n, n);
    auto m = make_pole_model(n, 3, 61);
    F(0, 0, nda::ellipsis{}) = random_hermitian(n, 62, 0.5);
    auto St = sigma_tau(m, ft);
    for (long it = 0; it < nt; ++it) S(it, 0, 0, nda::ellipsis{}) = St(it, nda::ellipsis{});
    lqp::fit_params_t p;
    auto lad = lqp::linearized_qp_ladder(mpi_context->comm, F, S, 0.0, ft, p, 60);
    int nf_mesh = lqp::max_n_fit_on_mesh(ft, true);
    CHECK(lad.mesh_limited);
    CHECK(lad.n_fit_mesh_max == nf_mesh);
    REQUIRE(lad.n_accepted > 0);
    CHECK(lad.last.n_fit <= nf_mesh);                    // never walked off the mesh
    CHECK(lad.n_fit_history(lad.n_accepted - 1) <= nf_mesh);
  }

  TEST_CASE("qp_input_blocks", "[methods_qp]") {
    auto parse = [](std::string const& json) {
      std::istringstream in(json);
      ptree pt;
      boost::property_tree::read_json(in, pt);
      qp_params_t p;
      read_qp_block(pt, p);
      return p;
    };

    SECTION("defaults are untouched by an empty block") {
      auto p = parse("{}");
      CHECK(p.qp_approx == "qp_eqn");
      CHECK(p.qp_eqn.solver == "sc");
      CHECK(p.lqp.n_fit == 6);
    }
    SECTION("nested blocks") {
      auto p = parse(R"({"qp_eqn": {"solver": "sc_newton", "ac_nfit": "30", "eta": "0.01"}, "lqp": {"n_fit": "4"}})");
      CHECK(p.qp_approx == "qp_eqn");
      CHECK(p.qp_eqn.solver == "sc_newton");
      CHECK(p.qp_eqn.ac_nfit == 30);
      CHECK(p.qp_eqn.eta == Approx(0.01));
      CHECK(p.lqp.n_fit == 4);          // parsed even though qp_eqn is the selected family
    }
    SECTION("qp_approx selects the family, the blocks stay independent") {
      auto p = parse(R"({"qp_approx": "lqp", "lqp": {"n_fit": "8", "fit_order": "5"}})");
      CHECK(p.qp_approx == "lqp");
      CHECK(p.lqp.n_fit == 8);
      CHECK(p.lqp.fit_order == 5);
      CHECK(p.qp_eqn.solver == "sc");       // untouched default
    }
    SECTION("deprecated qp_type maps onto (qp_approx, qp_eqn.solver)") {
      auto p = parse(R"({"qp_type": "sc_newton"})");
      CHECK(p.qp_approx == "qp_eqn");
      CHECK(p.qp_eqn.solver == "sc_newton");
      auto q = parse(R"({"qp_type": "linearized"})");
      CHECK(q.qp_approx == "qp_eqn");
      CHECK(q.qp_eqn.solver == "linearized");
      auto r = parse(R"({"qp_type": "lqp"})");
      CHECK(r.qp_approx == "lqp");
    }
    SECTION("the new keys win over the deprecated one") {
      auto p = parse(R"({"qp_type": "lqp", "qp_approx": "qp_eqn", "qp_eqn": {"solver": "sc_bisection"}})");
      CHECK(p.qp_approx == "qp_eqn");
      CHECK(p.qp_eqn.solver == "sc_bisection");
    }
  }

} // bdft_tests
