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
    // A self-energy with known low-frequency coefficients: Sigma(iw) = sum_p V_p / (iw - e_p).
    struct pole_model_t {
      std::vector<double> e;                         // pole positions e_p [Ha]
      std::vector<nda::array<ComplexType, 2>> V;     // residues V_p, (norb, norb)
      long norb;
    };

    /**
     * A causal pole model for norb orbitals, reproducible from seed: 
     * 
     * Pole p sits at e_p = +-(8 + 2p) Ha, alternating above and below the Fermi level. 
     * Each residue is V_p = u_p u_p^dag with u_p of random complex entries, so Sigma is 
     * causal and complex off-diagonal. 
     * 
     * The linearized QP solution has the following closed forms: 
     * A = Sigma(0) = -sum_p V_p / e_p, B = dSigma/d(iw)|_0 = -sum_p V_p / e_p^2
     * @param norb  - [INPUT] number of orbitals
     * @param npole - [INPUT] number of poles
     * @param seed  - [INPUT] seed of the random residues; the same seed gives the same model
     */
    pole_model_t make_pole_model(long norb, int npole, unsigned seed) {
      pole_model_t model; model.norb = norb;
      std::mt19937 gen(seed);
      std::uniform_real_distribution<double> uniform(-1.0, 1.0);
      for (int p = 0; p < npole; ++p) {
        model.e.push_back((p % 2 == 0 ? 1.0 : -1.0) * (8.0 + 2.0 * p));
        nda::array<ComplexType, 1> u_p(norb);
        for (long a = 0; a < norb; ++a) u_p(a) = ComplexType(uniform(gen), uniform(gen));
        nda::array<ComplexType, 2> V(norb, norb);
        for (long a = 0; a < norb; ++a)
          for (long b = 0; b < norb; ++b) V(a, b) = u_p(a) * std::conj(u_p(b));
        model.V.push_back(V);
      }
      return model;
    }

    // Sigma at CoQui Matsubara indices n_idx (w_n = n pi / beta)
    nda::array<ComplexType, 3> sigma_on_window(pole_model_t const& model,
                                               nda::array<long, 1> const& n_idx, double beta) {
      nda::array<ComplexType, 3> Sigma_wab(n_idx.shape(0), model.norb, model.norb);
      Sigma_wab() = 0.0;
      for (long j = 0; j < n_idx.shape(0); ++j) {
        ComplexType iw(0.0, double(n_idx(j)) * M_PI / beta);
        for (size_t p = 0; p < model.e.size(); ++p)
          Sigma_wab(j, nda::ellipsis{}) += model.V[p] / (iw - model.e[p]);
      }
      return Sigma_wab;
    }

    // Closed form of A = Sigma(0) and B = dSigma/d(iw) at w = 0 of the model
    std::pair<nda::array<ComplexType, 2>, nda::array<ComplexType, 2>>
    exact_AB(pole_model_t const& model) {
      nda::array<ComplexType, 2> A(model.norb, model.norb), B(model.norb, model.norb);
      A() = 0.0; B() = 0.0;
      for (size_t p = 0; p < model.e.size(); ++p) {
        A -= model.V[p] / model.e[p];
        B -= model.V[p] / (model.e[p] * model.e[p]);
      }
      return {A, B};
    }

    // Sigma(tau) on the IAFT fermionic mesh constructed from a pole model: 
    //     Sigma(tau) = sum_p V_p g_{e_p}(tau), 0 <= tau <= beta,
    // where 
    //     g_e(tau) = -exp(-e tau) / (1 + exp(-e beta)) = -exp(e (beta - tau)) / (1 + exp(e beta)),
    // 
    // The first is used for e > 0 and the second for e < 0, to avoid exponential overflows. 
    nda::array<ComplexType, 3> sigma_tau(pole_model_t const& model, imag_axes_ft::IAFT const& ft) {
      double beta = ft.beta();
      auto x = ft.tau_mesh();
      long nt = x.shape(0);
      nda::array<ComplexType, 3> Sigma_tab(nt, model.norb, model.norb);
      Sigma_tab() = 0.0;
      for (long it = 0; it < nt; ++it) {
        double tau = (x(it) + 1.0) * beta / 2.0;
        for (size_t p = 0; p < model.e.size(); ++p) {
          double e = model.e[p];
          double g = (e > 0) ? -std::exp(-e * tau) / (1.0 + std::exp(-e * beta))
                             : -std::exp(e * (beta - tau)) / (1.0 + std::exp(e * beta));
          Sigma_tab(it, nda::ellipsis{}) += model.V[p] * g;
        }
      }
      return Sigma_tab;
    }

    // H = scale (X + X^dag) / 2 with X of random complex entries in [-1, 1] + i [-1, 1]
    nda::array<ComplexType, 2> random_hermitian_matrix(long norb, unsigned seed,
                                                       double scale = 1.0) {
      std::mt19937 gen(seed); std::uniform_real_distribution<double> uniform(-1.0, 1.0);
      nda::array<ComplexType, 2> X(norb, norb);
      for (long a = 0; a < norb; ++a)
        for (long b = 0; b < norb; ++b) X(a, b) = ComplexType(uniform(gen), uniform(gen));
      nda::array<ComplexType, 2> H(norb, norb);
      for (long a = 0; a < norb; ++a)
        for (long b = 0; b < norb; ++b) H(a, b) = 0.5 * scale * (X(a, b) + std::conj(X(b, a)));
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
    
    nda::array<ComplexType, 2> random_unitary(long norb, unsigned seed) {
      auto H = random_hermitian_matrix(norb, seed);
      auto [w, U] = nda::linalg::eigenelements(H);   // columns of U are orthonormal
      nda::array<ComplexType, 2> Ua(U);
      return Ua;
    }


    // The full-array inputs of linearized_qp_solve / linearized_qp_ladder for one spin and one
    // pole model per k-point: F_skab (1, nk, norb, norb), random Hermitian per k-point from
    // seed_F + k, and Sigma_tskab (nt, 1, nk, norb, norb), each model's Sigma(tau) on ft's mesh.
    std::pair<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
    make_full_array_inputs(std::vector<pole_model_t> const& models, imag_axes_ft::IAFT const& ft,
                           unsigned seed_F) {
      long nk = models.size(), norb = models[0].norb, nt = ft.nt_f();
      nda::array<ComplexType, 4> F(1, nk, norb, norb);
      nda::array<ComplexType, 5> Sigma(nt, 1, nk, norb, norb);
      for (long k = 0; k < nk; ++k) {
        F(0, k, nda::ellipsis{}) = random_hermitian_matrix(norb, seed_F + k, 0.5);
        auto Sigma_tab = sigma_tau(models[k], ft);
        for (long it = 0; it < nt; ++it)
          Sigma(it, 0, k, nda::ellipsis{}) = Sigma_tab(it, nda::ellipsis{});
      }
      return {std::move(F), std::move(Sigma)};
    }
  } // anonymous namespace

  TEST_CASE("lqp_fit_window", "[methods_qp]") {
    // The window is the lowest odd Matsubara indices: +-1, +-3, ..., +-(2 n_fit - 1) in ascending
    // order with symmetric_window, the positive half otherwise.
    auto window = lqp::fit_window(3, true);
    REQUIRE(window.shape(0) == 6);
    CHECK(window(0) == -5); CHECK(window(1) == -3); CHECK(window(2) == -1);
    CHECK(window(3) ==  1); CHECK(window(4) ==  3); CHECK(window(5) ==  5);
    auto window_pos = lqp::fit_window(3, false);
    REQUIRE(window_pos.shape(0) == 3);
    CHECK(window_pos(0) == 1); CHECK(window_pos(2) == 5);
  }

  TEST_CASE("lqp_make_fit_operator", "[methods_qp]") {
    // Checked: the selected indices, the frequency scale, two entries of the 
    // scaled design matrix (i x_j)^p, the decoupling of even and odd powers on 
    // a symmetric window, and the shape of an over-determined (least-squares) operator.
    auto check_fit_operator = [](imag_axes_ft::IAFT const& ft) {
      lqp::params_t params; params.n_fit = 3;       // 6 nodes, exact order 5
      auto op = lqp::make_fit_operator(ft, params);
      auto n_idx = lqp::fit_window(params.n_fit, params.symmetric_window);
      REQUIRE(op.M.shape(0) == 6); REQUIRE(op.M.shape(1) == 6);   // square system
      REQUIRE(op.w_sel.shape(0) == 6);
      auto wn = ft.wn_mesh_f();
      // check selection of window nodes. 
      for (long j = 0; j < 6; ++j) CHECK(long(wn(op.w_sel(j))) == n_idx(j));
      // check the omega scale factor 
      VALUE_EQUAL(op.omega_scale, 5.0 * M_PI / ft.beta(), 1e-14, 1e-14);
      // M_{j p} = (i x_j)^p with x_j = n_j / n_max
      VALUE_EQUAL(op.M(5, 1), ComplexType(0.0, 1.0), 1e-14, 1e-14);
      VALUE_EQUAL(op.M(2, 2), ComplexType(-1.0/25.0, 0.0), 1e-14, 1e-14);
      // parity decoupling on a symmetric window: even and odd columns orthogonal
      ComplexType overlap = 0.0;
      for (int j = 0; j < 6; ++j) overlap += std::conj(op.M(j, 0)) * op.M(j, 1);
      CHECK(std::abs(overlap) < 1e-14);

      // over-determined: 6 nodes, order 3
      lqp::params_t params_ls; params_ls.n_fit = 3; params_ls.fit_order = 3;
      auto op_ls = lqp::make_fit_operator(ft, params_ls);
      REQUIRE(op_ls.M.shape(0) == 6); REQUIRE(op_ls.M.shape(1) == 4);   // not square: least squares
    };

    SECTION("dlr") {
      imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
      check_fit_operator(ft);
    }
    SECTION("ir") {
      imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::ir_basis);
      check_fit_operator(ft);
    }
  }

  TEST_CASE("lqp_low_freq_coefficients", "[methods_qp]") {
    // The fit returns the Taylor coefficients of Sigma(iw) at w = 0, A = Sigma(0) and
    // B = dSigma/d(iw), unscaled and Hermitized, with the diagnostics of the fit.
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);

    // 1. A pole model with complex-Hermitian residues, sampled exactly on a 12-node window:
    //    A and B match their closed forms. 
    {
      long norb = 3; int npole = 4; unsigned seed = 11;
      lqp::params_t params; params.n_fit = 6;       // 12 nodes, order 11
      auto op = lqp::make_fit_operator(ft, params);
      auto model = make_pole_model(norb, npole, seed);
      auto Sigma_wab = sigma_on_window(
        model, lqp::fit_window(params.n_fit, params.symmetric_window), ft.beta());

      auto fit = lqp::low_freq_coefficients(Sigma_wab, op, params);
      REQUIRE(fit.A.shape(0) == norb); REQUIRE(fit.A.shape(1) == norb);
      REQUIRE(fit.B.shape(0) == norb); REQUIRE(fit.B.shape(1) == norb);
      // Tolerances: Sigma's poles sit at |e_p| >= 8 Ha, so its Taylor series about w = 0
      // converges for |z| < 8 Ha; on the window (|w| <= 0.35 Ha) the terms fall off like
      // (0.35/8)^k, and the order-11 fit misses only ~(0.35/8)^12 ~ 1e-16. What remains is
      // round-off, measured at <= 2e-15 for A, B, resid and anti_herm_A over 200 models; 1e-12
      // leaves a ~1e3 margin.
      auto [A, B] = exact_AB(model);
      ARRAY_EQUAL(fit.A, A, 1e-12);
      ARRAY_EQUAL(fit.B, B, 1e-12);
      CHECK(fit.resid < 1e-12);
      CHECK(fit.anti_herm_A < 1e-12);            // Hermitian by construction on +/- nodes
      CHECK(fit.anti_herm_Sigma < 1e-12);        // S(iw)^dag == S(-iw) for the model
    }
    // 2. Sigma = 0: A, B and the residual are exactly zero (no division by |Sigma|).
    {
      lqp::params_t params; params.n_fit = 3;
      auto op = lqp::make_fit_operator(ft, params);
      nda::array<ComplexType, 3> Sigma_wab(op.w_sel.shape(0), 2, 2); Sigma_wab() = 0.0;

      auto fit = lqp::low_freq_coefficients(Sigma_wab, op, params);
      ARRAY_EQUAL(fit.A, nda::zeros<ComplexType>(2, 2), 1e-14);
      ARRAY_EQUAL(fit.B, nda::zeros<ComplexType>(2, 2), 1e-14);
      CHECK(fit.resid == 0.0);
    }
  }

  TEST_CASE("lqp_solve_linearized_dyson", "[methods_qp]") {
    // Given K and B, the solver returns Z = (1 - B)^-1, the eigenpairs of H_QP = Z^1/2 K Z^1/2
    // and the pole weights <v|Z|v>; it flags 1 - B that is not positive definite.

    // 1. Random Hermitian K, causal B: Z is the inverse of 1 - B, min_eig_Zinv is the smallest
    //    eigenvalue of 1 - B, the pole weights are the diagonal of Z in the H_QP eigenbasis
    //    (diag(V^dag Z V), which a wrong basis or conjugation would miss), the sum rule
    //    sum(Zqp) = Tr Z holds, and psi = Z^1/2 v solves K psi = E (1 - B) psi.
    {
      long norb = 5;
      auto K = random_hermitian_matrix(norb, 1);
      auto B = causal_B(norb, 2);
      
      lqp::result_kpoint_t results; 
      lqp::solve_linearized_dyson(K, B, results);
      
      REQUIRE(results.status == 0);
      REQUIRE(results.E.shape(0) == norb);
      CHECK(std::abs(results.sumrule) < 1e-12);          // sum Zqp == Tr Z
      nda::array<ComplexType, 2> ImB(norb, norb), ZImB(norb, norb), eye(norb, norb);
      ImB = -B; eye() = 0.0;
      for (long a = 0; a < norb; ++a) { 
        ImB(a, a) += 1.0; 
        eye(a, a) = 1.0; 
      }
      nda::blas::gemm(results.Z, ImB, ZImB);
      ARRAY_EQUAL(ZImB, eye, 1e-12);
      // the smallest eigenvalue of 1 - B, which is >= 1 here since B is negative semidefinite
      VALUE_EQUAL(results.min_eig_Zinv, nda::linalg::eigenvalues(ImB)(0), 1e-12, 1e-12);
      // the pole weights are the diagonal of Z in the H_QP eigenbasis
      nda::array<ComplexType, 2> ZV(norb, norb), VdZV(norb, norb);
      nda::blas::gemm(results.Z, results.V, ZV);
      nda::blas::gemm(nda::dagger(results.V), ZV, VdZV);
      for (long l = 0; l < norb; ++l) VALUE_EQUAL(results.Zqp(l), VdZV(l, l).real(), 1e-12, 1e-12);
      // the kernel does not keep Z^1/2, so build it here from Z's eigendecomposition
      auto [w, U] = nda::linalg::eigenelements(results.Z);
      nda::array<ComplexType, 2> Uw(norb, norb), Zhalf(norb, norb), Psi(norb, norb),
                                 lhs(norb, norb), rhs(norb, norb);
      for (long l = 0; l < norb; ++l) Uw(nda::range::all, l) = U(nda::range::all, l) * std::sqrt(w(l));
      nda::blas::gemm(Uw, nda::dagger(U), Zhalf);
      nda::blas::gemm(Zhalf, results.V, Psi);
      nda::blas::gemm(K, Psi, lhs);
      nda::blas::gemm(ImB, Psi, rhs);
      for (long l = 0; l < norb; ++l) rhs(nda::range::all, l) *= results.E(l);
      ARRAY_EQUAL(lhs, rhs, 1e-11);
    }
    // 2. Scalar B = b: Z = 1/(1 - b) multiplies the eigenvalues of K, E = Z eig(K), rather than
    //    shifting them, and every pole weight is that Z.
    {
      long norb = 4; double b = -0.5;
      auto K = random_hermitian_matrix(norb, 3);
      nda::array<ComplexType, 2> B(norb, norb); B() = 0.0;
      for (long a = 0; a < norb; ++a) B(a, a) = b;
      lqp::result_kpoint_t results; lqp::solve_linearized_dyson(K, B, results);
      auto eig_K = nda::linalg::eigenvalues(K);
      double Z_scalar = 1.0 / (1.0 - b);
      for (long l = 0; l < norb; ++l) {
        VALUE_EQUAL(results.Zqp(l), Z_scalar, 1e-12, 1e-12);
        VALUE_EQUAL(results.E(l), Z_scalar * eig_K(l), 1e-11, 1e-11);
      }
    }
    // 3. Gauge invariance: rotating K and B by one unitary U leaves the energies and the pole
    //    weights unchanged, since both are basis-independent.
    {
      long norb = 5;
      auto K = random_hermitian_matrix(norb, 4); 
      auto B = causal_B(norb, 5); 
      auto U = random_unitary(norb, 6);
      
      lqp::result_kpoint_t results; 
      lqp::solve_linearized_dyson(K, B, results);

      nda::array<ComplexType, 2> tmp(norb, norb), K_u(norb, norb), B_u(norb, norb);
      nda::blas::gemm(K, U, tmp); nda::blas::gemm(nda::dagger(U), tmp, K_u);
      nda::blas::gemm(B, U, tmp); nda::blas::gemm(nda::dagger(U), tmp, B_u);
      
      lqp::result_kpoint_t results_u; 
      lqp::solve_linearized_dyson(K_u, B_u, results_u);

      ARRAY_EQUAL(results.E, results_u.E, 1e-11);
      ARRAY_EQUAL(results.Zqp, results_u.Zqp, 1e-11);
    }
    // 4. 1 - B with eigenvalue -0.5: status 2, with that eigenvalue reported as min_eig_Zinv.
    {
      long norb = 2;
      auto K = random_hermitian_matrix(norb, 8);
      nda::array<ComplexType, 2> B(norb, norb); B() = 0.0; B(0, 0) = 1.5;
      
      lqp::result_kpoint_t results;
      lqp::solve_linearized_dyson(K, B, results);
      
      CHECK(results.status == 2);
      VALUE_EQUAL(results.min_eig_Zinv, -0.5, 1e-12, 1e-12);
    }
  }

  TEST_CASE("lqp_linearized_qp_kpoint", "[methods_qp]") {
    // From Sigma(tau) to the quasiparticle energy at one k-point, through the IAFT transform.
    // A scalar pole e = 10 with weight g^2 = 9 has A = -g^2/e = -0.9 and B = -g^2/e^2, so
    // Z = 1/(1 + g^2/e^2) pins B, and E = Z (F + A - mu) + mu pins A and the convention that
    // energies are absolute (linearized about mu, mu added back).
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    
    lqp::params_t params; params.n_fit = 6;
    auto op = lqp::make_fit_operator(ft, params);
    
    pole_model_t model; model.norb = 1; model.e = {10.0};
    
    nda::array<ComplexType, 2> V(1, 1); V(0, 0) = 9.0;
    model.V = {V};
    
    auto Sigma_tab = sigma_tau(model, ft);
    nda::array<ComplexType, 2> F(1, 1); F(0, 0) = 0.3;
    double mu = 0.2;
    auto results = lqp::linearized_qp_kpoint(F, Sigma_tab, mu, ft, op, params);
    
    VALUE_EQUAL(results.Zqp(0), 1.0 / (1.0 + 9.0 / 100.0), 1e-6, 1e-6);
    VALUE_EQUAL(results.E(0), results.Zqp(0) * (0.3 - 0.9 - 0.2) + 0.2, 1e-6, 1e-6);
    
    CHECK(results.resid < 1e-8);
  }

  TEST_CASE("lqp_linearized_qp_solve", "[methods_qp]") {
    // The full-array driver returns, at every (s, k), exactly what linearized_qp_kpoint returns
    // there (to 1e-12), whatever the number of ranks the points are distributed over.
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    
    lqp::params_t params; params.n_fit = 6;
    auto op = lqp::make_fit_operator(ft, params);
    
    double mu = 0.1;

    // Three k-points, one pole model each: 
    //    - comparison between the full-array driver and the per-k-point driver
    //    - compare to the analytic A and B
    //    - on the 16 ranks ctest uses, 13 ranks own no point, yet every rank must take part in
    //      the shared-memory fences and the internode reduction and return the complete result
    {
      long nk = 3, norb = 3; int npole = 3; unsigned seed = 20, seed_F = 30;
      std::vector<pole_model_t> models;
      for (long k = 0; k < nk; ++k) models.push_back(make_pole_model(norb, npole, seed + k));
      auto [F, Sigma] = make_full_array_inputs(models, ft, seed_F);
      
      auto res = lqp::to_owned(lqp::linearized_qp_solve(*mpi_context, F, Sigma, mu, ft, params));
      REQUIRE(res.E_ska.shape(1) == nk); 
      REQUIRE(res.E_ska.shape(2) == norb);
      
      for (long k = 0; k < nk; ++k) {
        auto results = lqp::linearized_qp_kpoint(F(0, k, nda::ellipsis{}),
                                                 sigma_tau(models[k], ft), mu, ft, op, params);
        ARRAY_EQUAL(res.E_ska(0, k, nda::range::all),   results.E,   1e-12);
        ARRAY_EQUAL(res.Zqp_ska(0, k, nda::range::all), results.Zqp, 1e-12);
        ARRAY_EQUAL(res.Z_skab(0, k, nda::ellipsis{}),   results.Z,   1e-12);
        ARRAY_EQUAL(res.Hqp_skab(0, k, nda::ellipsis{}), results.Hqp, 1e-12);
        auto [A, B] = exact_AB(models[k]);
        nda::matrix<ComplexType> ImB(-B);
        for (long a = 0; a < norb; ++a) ImB(a, a) += 1.0;
        ARRAY_EQUAL(results.Z, nda::inverse(ImB), 1e-6);
      }
      CHECK(res.n_fit == params.n_fit);
      CHECK(res.fit_order == 2 * params.n_fit - 1);   // fit_order = -1 resolved: exactly determined
    }
  }

  TEST_CASE("lqp_linearized_qp_ladder", "[methods_qp]") {
    // The ladder climbs n_fit = 2, 3, ... up to min(n_fit_max, the mesh ceiling) while the fit
    // residual stays below the gate; its last accepted fit is a plain solve at that n_fit.
    // (Its aborts, when no fit is accepted or n_fit_max < 2, are not observable here.)
    auto& mpi_context = utils::make_unit_test_mpi_context();
    imag_axes_ft::IAFT ft(100.0, 40.0, imag_axes_ft::dlr_basis);
    
    lqp::params_t params;                            // fit_resid_tol = 1e-8

    // 1. Two k-points, n_fit_max = 6, below the mesh ceiling (8 for this "medium" DLR mesh), so
    //    n_fit_max sets the cap. The model's poles are far from every window (|e| >= 8 Ha, the
    //    n_fit = 6 window ends at 0.35 Ha), so fitting up to n_fit_max = 6 is accepted with
    //    fit_resid_tol = 1e-8.
    {
      long nk = 2, norb = 3;
      int npole = 3, n_fit_max = 6;
      unsigned seed = 40, seed_F = 50;
      double mu = 0.1;
      
      std::vector<pole_model_t> models;
      for (long k = 0; k < nk; ++k) models.push_back(make_pole_model(norb, npole, seed + k));
      auto [F, Sigma] = make_full_array_inputs(models, ft, seed_F);
      
      auto lad = lqp::to_owned(lqp::linearized_qp_ladder(
        *mpi_context, F, Sigma, mu, ft, n_fit_max, params.fit_resid_tol, params.symmetric_window));
      
      REQUIRE(lad.n_accepted == lad.n_fit_cap - 1);   // every fit up to the cap accepted
      CHECK(lad.last.n_fit == lad.n_accepted + 1);     // accepted fits are n_fit = 2, 3, ...
      CHECK(lad.resid_history.shape(0) == lad.n_accepted);
      
      lqp::params_t params_nf = params;
      params_nf.n_fit = lad.last.n_fit;
      params_nf.fit_order = -1;
      auto single = lqp::to_owned(lqp::linearized_qp_solve(
        *mpi_context, F, Sigma, mu, ft, params_nf));

      // consistency between the last accepted fit and a single solve at that n_fit
      ARRAY_EQUAL(lad.last.E_ska, single.E_ska, 1e-12);
      ARRAY_EQUAL(lad.last.Zqp_ska, single.Zqp_ska, 1e-12);

      // consistency between the last accepted fit and the exact A and B. 
      for (long k = 0; k < nk; ++k) {
        auto [A, B] = exact_AB(models[k]);
        nda::array<ComplexType, 2> K(F(0, k, nda::ellipsis{}) + A);
        for (long a = 0; a < norb; ++a) K(a, a) -= mu;
        
        lqp::result_kpoint_t exact;
        lqp::solve_linearized_dyson(K, B, exact);

        // Tolerances: measured 1.4e-8 for Zqp, 2.7e-10 for err_fit_sk and 1.4e-10 for dHqp_sk.
        // The floor is the tau -> iw transform of the "medium" DLR mesh, not the window fit
        for (long a = 0; a < norb; ++a)
          VALUE_EQUAL(lad.last.Zqp_ska(0, k, a), exact.Zqp(a), 1e-6, 1e-6);
        CHECK(lad.err_fit_sk(0, k) < 1e-8);
        CHECK(lad.dHqp_sk(0, k) < 1e-8);
      }
      CHECK(lad.n_fit_cap == std::min(n_fit_max, lqp::max_n_fit_on_mesh(ft, true)));
      CHECK(lad.last.n_fit == lad.n_fit_cap);
      CHECK(lad.stopped_resid == 0.0);                  // zero because the cap was reached. 
    }
    // 2. n_fit_max = 60, beyond the mesh: the cap is the mesh ceiling and the climb never
    //    walks off the mesh. Climbing to the cap also builds the fit operator at
    //    n_fit = max_n_fit_on_mesh(), so it checks that the two agree on where the mesh ends
    //    (make_fit_operator would abort one step beyond).
    {
      long norb = 3; int npole = 3, n_fit_max = 60; unsigned seed = 61, seed_F = 62;
      std::vector<pole_model_t> models = {make_pole_model(norb, npole, seed)};
      auto [F, Sigma] = make_full_array_inputs(models, ft, seed_F);
      auto lad = lqp::to_owned(lqp::linearized_qp_ladder(*mpi_context, F, Sigma, 0.0, ft, n_fit_max,
                                                         params.fit_resid_tol,
                                                         params.symmetric_window));
      int n_fit_mesh = lqp::max_n_fit_on_mesh(ft, true);
      CHECK(lad.n_fit_cap == n_fit_mesh);
      CHECK(lad.last.n_fit <= n_fit_mesh);
    }
  }

  TEST_CASE("lqp_read_qp_block", "[methods_qp]") {
    // read_qp_block reads the two-axis input -- qp_approx plus one nested block per family,
    // [.qp_eqn] and [.lqp] -- into the call site's defaults, and nothing else: the flat
    // spellings that predate the blocks are converted by the Python entry points
    // (coqui.utils.qp_input) and ignored here.
    // parses json into the given call-site defaults (the struct's own defaults if none)
    auto parse = [](std::string const& json, qp_context_t qp_params = {}) {
      std::istringstream in(json);
      ptree pt;
      boost::property_tree::read_json(in, pt);
      read_qp_block(pt, qp_params);
      return qp_params;
    };

    SECTION("call-site defaults are kept when absent and overridden when present") {
      // presets that differ from the struct's defaults, as evgw (eta = pi/beta) and the
      // downfolding drivers (ac_nfit = 30) set them: a parser that reset absent fields to the
      // struct's defaults would fail here
      qp_context_t call_site;
      call_site.qp_eqn.eta = 0.123; call_site.qp_eqn.ac_nfit = 30; call_site.lqp.n_fit = 9;
      auto qp_params = parse(R"({"qp_eqn": {"eta": "0.5"}})", call_site);
      CHECK(qp_params.qp_eqn.eta == Approx(0.5));            // present: overridden
      CHECK(qp_params.qp_eqn.ac_nfit == 30);                  // absent: preset kept
      CHECK(qp_params.lqp.n_fit == 9);                        // absent: preset kept
      CHECK(qp_params.qp_approx == "qp_eqn");                 // absent: struct default kept
      CHECK(qp_params.qp_eqn.solver == "sc");
    }
    SECTION("nested blocks") {
      // every key of both blocks except qp_eqn.ac_alg: "pade" is the only analytic continuation
      // implemented and also the default, so no valid value tells a read from the default.
      // Mixed-case strings check the lower-casing of solver and off_diag_mode.
      auto qp_params = parse(R"({"qp_eqn": {"solver": "SC_Newton", "ac_nfit": "30",)"
                             R"( "eta": "0.01", "tol": "1e-6", "off_diag_mode": "QP_Energy"},)"
                             R"( "lqp": {"n_fit": "4", "fit_order": "5", "fit_resid_tol": "1e-10"}})");
      CHECK(qp_params.qp_approx == "qp_eqn");
      CHECK(qp_params.qp_eqn.solver == "sc_newton");
      CHECK(qp_params.qp_eqn.ac_nfit == 30);
      CHECK(qp_params.qp_eqn.eta == Approx(0.01));
      CHECK(qp_params.qp_eqn.tol == Approx(1e-6));
      CHECK(qp_params.qp_eqn.off_diag_mode == "qp_energy");
      CHECK(qp_params.lqp.n_fit == 4);          // parsed even though qp_eqn is the selected family
      CHECK(qp_params.lqp.fit_order == 5);
      CHECK(qp_params.lqp.fit_resid_tol == Approx(1e-10));
    }
    SECTION("qp_approx selects the family, the blocks stay independent") {
      auto qp_params = parse(R"({"qp_approx": "LQP", "lqp": {"n_fit": "8", "fit_order": "5"}})");
      CHECK(qp_params.qp_approx == "lqp");
      CHECK(qp_params.lqp.n_fit == 8);
      CHECK(qp_params.lqp.fit_order == 5);
      CHECK(qp_params.qp_eqn.solver == "sc");       // untouched default
    }
    SECTION("only the nested form is read") {
      // qp_type and the flat spellings reaching C++ are ignored, so the call site's defaults
      // stand.
      auto qp_params = parse(R"({"qp_type": "lqp", "ac_alg": "foo", "Nfit": "30", "eta": "0.5",)"
                             R"( "off_diag_mode": "qp_energy", "qp_eta": "0.5", "qp_Nfit": "30",)"
                             R"( "qp_tol": "0.5", "n_fit": "2"})");
      qp_context_t defaults;
      CHECK(qp_params.qp_approx == defaults.qp_approx);
      CHECK(qp_params.qp_eqn.solver == defaults.qp_eqn.solver);
      CHECK(qp_params.qp_eqn.ac_alg == defaults.qp_eqn.ac_alg);
      CHECK(qp_params.qp_eqn.ac_nfit == defaults.qp_eqn.ac_nfit);
      CHECK(qp_params.qp_eqn.eta == defaults.qp_eqn.eta);
      CHECK(qp_params.qp_eqn.tol == defaults.qp_eqn.tol);
      CHECK(qp_params.qp_eqn.off_diag_mode == defaults.qp_eqn.off_diag_mode);
      CHECK(qp_params.lqp.n_fit == defaults.lqp.n_fit);
    }
  }

} // bdft_tests
