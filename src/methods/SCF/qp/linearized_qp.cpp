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

#include <cmath>
#include <set>
#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <format>
#include "nda/linalg.hpp"
#include "nda/linalg/det_and_inverse.hpp"
#include "nda/lapack.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include <memory>
#include "methods/SCF/qp/linearized_qp.hpp"

namespace methods::lqp {

namespace {
  // 0.5 (X + X^dag) in place; returns |anti-Hermitian part| / |X| (0 if X == 0)
  double hermitize(nda::array_view<ComplexType, 2> mat) {
    long norb = mat.shape(0);
    double anti = 0.0, scale = 0.0;
    for (long a = 0; a < norb; ++a) {
      for (long b = a; b < norb; ++b) {
        ComplexType herm_ab = 0.5 * (mat(a, b) + std::conj(mat(b, a)));
        ComplexType anti_ab = 0.5 * (mat(a, b) - std::conj(mat(b, a)));
        anti  = std::max(anti, std::abs(anti_ab));
        scale = std::max(scale, std::abs(mat(a, b)));
        mat(a, b) = herm_ab; mat(b, a) = std::conj(herm_ab);
      }
    }
    return (scale > 0.0) ? anti / scale : 0.0;
  }

  /**
   * The abort policy of on_failure_e::abort. 
   *   status 0 - returns without effect
   *   status 1 - terminate the program: The exactly-determined fit does not interpolates its nodes
   *              within the relative residual.
   *   status 2 - terminate the program: 1 - B is not positive definite.  
   * @param results - [INPUT] outcome of one k-point (lqp_kpoint_impl())
   * @param op      - [INPUT] fit operator the k-point was solved with; fit_order and cond are
   *                  quoted
   * @param params  - [INPUT] fit controls; n_fit and fit_resid_tol are quoted
   * @param is, ik  - [INPUT] spin and k index of the failed point, added to the message as
   *                  "at (s=.., k=..)" when >= 0; the defaults (-1) omit it, for a caller that
   *                  handles a single k-point
   */
  void abort_on_failure(result_kpoint_t const& results, fit_operator_t const& op,
                        params_t const& params,
                        long is = -1, long ik = -1) {
    if (results.status == 0) return;
    std::string where;
    if (is >= 0) where = "at (s=" + std::to_string(is) + ", k=" + std::to_string(ik) + ") ";
    utils::check(results.status != 1,
                 "linearized_qp.cpp::linearized_qp_kpoint: {}, the exactly-determined fit (n_fit={}, "
                 "fit_order={}) no longer interpolates the data: relative residual {:.2e} > {:.0e} "
                 "(cond {:.2e}). Lower n_fit.",
                 where, params.n_fit, op.fit_order, results.resid, params.fit_resid_tol, op.cond);
    utils::check(results.status != 2,
                 "linearized_qp.cpp::linearized_qp_kpoint: {}, 1 - B is not positive definite (min "
                 "eigenvalue {:.3e}); Sigma(iw) is not causal there or the fit failed. Not "
                 "regularized.",
                 where, results.min_eig_Zinv);
  }

  /**
   * Sums the unshared arrays of a shared_result_t over every rank of comm, in member order
   * (all_reduce is collective). Every rank holds its own copy of these, zero except at the
   * (s, k) points it computed, so the sum over all ranks fills every point exactly once. The
   * node-shared matrices are not touched here: they are reduced by their own
   * shared_array::all_reduce().
   */
  void all_reduce_unshared_arrays(shared_result_t& res, boost::mpi3::communicator& comm) {
    auto sum = [&comm](auto& arr) {
      comm.all_reduce_in_place_n(arr.data(), arr.size(), std::plus<>{});
    };
    for (auto* arr : {&res.E_ska, &res.Zqp_ska}) sum(*arr);
    for (auto* arr : {&res.min_eig_Zinv_sk, &res.resid_sk, &res.anti_herm_A_sk, &res.anti_herm_B_sk,
                      &res.anti_herm_Sigma_sk, &res.sumrule_sk})
      sum(*arr);
    sum(res.status_sk);
  }

  /**
   * The table both linearized_qp_solve() and linearized_qp_ladder() log: a title line, then the
   * column header of log_fit_row().
   */
  void log_fit_table_header(std::string const& title) {
    app_log(1, "  {}\n"
               "    n_fit  order  w_max [Ha]  max resid   min Zqp  mean Zqp   max Zqp", title);
  }

  /**
   * One row of the table: the fit window (n_fit, fit_order, highest node w_max), the largest fit
   * residual and the range and mean of the pole weights over every (s, k, band).
   */
  void log_fit_row(shared_result_t const& res, double beta) {
    app_log(1, "    {:5d}  {:5d}  {:10.4f}  {:9.1e}  {:8.6f}  {:8.6f}  {:8.6f}", res.n_fit,
            res.fit_order, (2 * res.n_fit - 1) * M_PI / beta, nda::max_element(res.resid_sk),
            nda::min_element(res.Zqp_ska), nda::sum(res.Zqp_ska) / double(res.Zqp_ska.size()),
            nda::max_element(res.Zqp_ska));
  }

  /**
   * The line after the table: the fit the result comes from and the diagnostics the table does
   * not show; failed points only when there are any (on_failure_e::report).
   * @param extra - appended to the line (the ladder's truncation-error estimate)
   */
  void log_fit_summary(shared_result_t const& res, std::string const& extra) {
    auto const& st = res.status_sk;
    long n_failed = std::count_if(st.begin(), st.end(), [](long x) { return x != 0; });
    app_log(1, "    result: n_fit = {}, min eig(1 - B) = {:.4f}, max |sum rule| = {:.1e}{}{}\n",
            res.n_fit, nda::min_element(res.min_eig_Zinv_sk),
            nda::max_element(nda::abs(res.sumrule_sk)),
            (n_failed > 0)? std::format(", failed points: {}", n_failed) : std::string(""),
            extra);
  }

} // anonymous namespace

nda::array<long, 1> fit_window(int n_fit, bool symmetric_window) {
  utils::check(n_fit >= 1, "linearized_qp.cpp::fit_window: n_fit must be >= 1, got {}", n_fit);
  long nn = symmetric_window ? 2 * n_fit : n_fit;
  nda::array<long, 1> n_idx(nn);
  if (symmetric_window) {
    // -(2 n_fit - 1), ..., -3, -1, 1, 3, ..., (2 n_fit - 1)  (ascending, as the Python reference)
    for (int j = 0; j < n_fit; ++j) {
      n_idx(j)         = -(2 * (n_fit - j) - 1);
      n_idx(n_fit + j) =   2 * j + 1;
    }
  } else {
    for (int j = 0; j < n_fit; ++j) n_idx(j) = 2 * j + 1;
  }
  return n_idx;
}

fit_operator_t make_fit_operator(imag_axes_ft::IAFT const& ft, params_t const& params) {
  fit_operator_t op;
  auto n_idx = fit_window(params.n_fit, params.symmetric_window);
  long nn    = n_idx.shape(0);
  int  P   = (params.fit_order < 0) ? int(nn) - 1 : params.fit_order;
  utils::check(P >= 1, "linearized_qp.cpp::make_fit_operator: fit_order must be >= 1 (the slope is "
                       "A^(1)), got {}", P);
  utils::check(nn >= P + 1, "linearized_qp.cpp::make_fit_operator: {} fit nodes cannot determine {} "
                            "parameters; raise n_fit or lower fit_order.", nn, P + 1);
  op.fit_order = P;

  // the window's nodes are rows of the IAFT fermionic sampling mesh (IR and DLR both contain the
  // contiguous low odd n)
  auto wn_mesh = ft.wn_mesh_f();
  op.w_sel = nda::array<long, 1>(nn);
  for (long j = 0; j < nn; ++j) {
    auto it = std::find(wn_mesh.begin(), wn_mesh.end(), n_idx(j));
    utils::check(it != wn_mesh.end(), "linearized_qp.cpp::make_fit_operator: window node n = {} is not on "
                                 "the IAFT fermionic sampling mesh; the largest admissible n_fit is {}.",
                 n_idx(j), max_n_fit_on_mesh(ft, params.symmetric_window));
    op.w_sel(j) = it - wn_mesh.begin();
  }

  // scaled design matrix M_{jp} = (i x_j)^p, x_j = n_j / n_max
  long n_max = 0;
  for (long j = 0; j < nn; ++j) n_max = std::max(n_max, std::abs(n_idx(j)));
  op.omega_scale = double(n_max) * M_PI / ft.beta();
  op.M = nda::array<ComplexType, 2>(nn, P + 1);
  for (long j = 0; j < nn; ++j) {
    ComplexType ix(0.0, double(n_idx(j)) / double(n_max));
    ComplexType x_pow = 1.0;
    for (int q = 0; q <= P; ++q) { 
      op.M(j, q) = x_pow;
      x_pow *= ix; 
    }
  }

  // condition number from the singular values gelss returns (F layout required); the
  // right-hand side is a dummy, the solve itself happens per call in low_freq_coefficients
  nda::matrix<ComplexType, nda::F_layout> Mf(op.M), Bf(nn, 1);
  Bf() = 0.0;
  nda::vector<double> sing_vals(std::min<long>(nn, P + 1));
  int rank = 0;
  utils::check(nda::lapack::gelss(Mf, Bf, sing_vals, 0.0, rank) == 0,
               "linearized_qp.cpp::make_fit_operator: gelss failed");
  op.cond = sing_vals(0) / sing_vals(sing_vals.shape(0) - 1);
  
  return op;
}

low_freq_fit_t low_freq_coefficients(nda::array_const_view<ComplexType, 3> Sigma_wab,
                                     fit_operator_t const& op, params_t const& params) {
  long nn = Sigma_wab.shape(0), norb = Sigma_wab.shape(1);
  utils::check(nn == op.w_sel.shape(0) and Sigma_wab.shape(2) == norb,
               "linearized_qp.cpp::low_freq_coefficients: Sigma_wab shape ({}, {}, {}) does not match "
               "the fit window of {} nodes", nn, norb, Sigma_wab.shape(2), op.w_sel.shape(0));
  int P = op.fit_order;
  low_freq_fit_t fit;

  // scale of the data, for the relative diagnostics below
  double smax = 0.0;
  for (auto const& elem : Sigma_wab) smax = std::max(smax, std::abs(elem));

  // Hermiticity of the data on a symmetric window: S(-iw) == S(iw)^dag, relative to max |S|
  double anti = 0.0;
  if (params.symmetric_window) {
    for (long j = 0; j < nn / 2; ++j) {
      long jm = j, jp = nn - 1 - j;   // fit_window is ascending: n(jm) == -n(jp)
      for (long a = 0; a < norb; ++a)
        for (long b = 0; b < norb; ++b)
          anti = std::max(anti, std::abs(Sigma_wab(jp, a, b) - std::conj(Sigma_wab(jm, b, a))));
    }
  }
  fit.anti_herm_Sigma = (smax > 0.0) ? anti / smax : 0.0;

  // one least-squares solve for all norb^2 elements: 
  //     M (nn, P+1) c (P+1, norb^2) = S (nn, norb^2).
  // Solved per call by gelss (backward stable: residual ~ eps regardless of cond(M)) rather than
  // by applying a precomputed pseudo-inverse of M, whose residual grows like eps * cond(M). 
  // rcond mirrors numpy.linalg.lstsq(rcond=None): eps * max(nn, P+1).
  nda::array<ComplexType, 3> S3(Sigma_wab);                     // contiguous, any input layout
  auto S2 = nda::reshape(S3, std::array<long, 2>{nn, norb * norb});    // (nn, norb^2) view of S3
  nda::matrix<ComplexType, nda::F_layout> Mf(op.M), Bf(S2);
  nda::vector<double> sing_vals(std::min<long>(nn, P + 1));
  int rank = 0;
  double rcond = std::numeric_limits<double>::epsilon() * double(std::max<long>(nn, P + 1));
  utils::check(nda::lapack::gelss(Mf, Bf, sing_vals, rcond, rank) == 0,
               "linearized_qp.cpp::low_freq_coefficients: gelss failed");
  nda::array<ComplexType, 2> c2 = Bf(nda::range(P + 1), nda::range::all);   // the solution rows

  // residual of the fit, relative to the data scale
  nda::array<ComplexType, 2> resid_mat(nn, norb * norb);
  nda::blas::gemm(op.M, c2, resid_mat);
  resid_mat -= S2;
  double rmax = 0.0;
  for (auto const& elem : resid_mat) rmax = std::max(rmax, std::abs(elem));
  fit.resid = (smax > 0.0) ? rmax / smax : 0.0;

  // unscale and Hermitize orders 0 and 1, A^(p) = c^(p) / omega_scale^p (P >= 1 is guaranteed by
  // make_fit_operator); the higher orders are discarded. coeffs is a view of c2, so A and B are
  // the only copies, and returning fit moves them (or elides the move) without copying.
  auto coeffs = nda::reshape(c2, std::array<long, 3>{P + 1, norb, norb});
  fit.A = coeffs(0, nda::ellipsis{});
  fit.B = coeffs(1, nda::ellipsis{}) / op.omega_scale;
  fit.anti_herm_A = hermitize(fit.A());
  fit.anti_herm_B = hermitize(fit.B());
  return fit;
}

void solve_linearized_dyson(nda::array_const_view<ComplexType, 2> K_ab,
                            nda::array_const_view<ComplexType, 2> B_ab,
                            result_kpoint_t& results) {
  long norb = K_ab.shape(0);
  utils::check(K_ab.shape(1) == norb and B_ab.shape(0) == norb and B_ab.shape(1) == norb,
               "linearized_qp.cpp::solve_linearized_dyson: K and B must be square and of equal "
               "size");
  nda::array<ComplexType, 2> tmp(norb, norb);

  // 1 - B, Hermitized (B is Hermitian only up to the fit's round-off)
  tmp = -B_ab;
  for (long a = 0; a < norb; ++a) tmp(a, a) += 1.0;
  hermitize(tmp());
  auto [w, U] = nda::linalg::eigenelements(tmp);          // ascending eigenvalues, columns
  results.min_eig_Zinv = w(0);
  // caller aborts or reports (s,k)
  if (results.min_eig_Zinv <= 0.0) { results.status = 2; return; }

  // Z = U diag(1/w) U^dag,  Z^1/2 = U diag(1/sqrt w) U^dag
  nda::array<ComplexType, 2> Zhalf(norb, norb);            // construction intermediate, not kept
  results.Z = nda::array<ComplexType, 2>(norb, norb);
  for (long l = 0; l < norb; ++l) tmp(nda::range::all, l) = U(nda::range::all, l) * (1.0 / w(l));
  nda::blas::gemm(tmp, nda::dagger(U), results.Z);
  for (long l = 0; l < norb; ++l) tmp(nda::range::all, l) = U(nda::range::all, l) / std::sqrt(w(l));
  nda::blas::gemm(tmp, nda::dagger(U), Zhalf);
  hermitize(results.Z()); hermitize(Zhalf());

  // H_QP = Z^1/2 K Z^1/2, Hermitized. K needs no Hermitization of its own: with Z^1/2 Hermitian,
  // the Hermitian part of Z^1/2 K Z^1/2 is exactly Z^1/2 [(K + K^dag)/2] Z^1/2.
  results.Hqp = nda::array<ComplexType, 2>(norb, norb);
  nda::blas::gemm(K_ab, Zhalf, tmp);
  nda::blas::gemm(Zhalf, tmp, results.Hqp);
  hermitize(results.Hqp());
  auto [E, V] = nda::linalg::eigenelements(results.Hqp);
  results.E = std::move(E); results.V = std::move(V);

  // Pole weights: the diagonal of Z in the H_QP eigenbasis, <v_l|Z|v_l> = sum_a conj(V_al) (Z V)_al
  nda::blas::gemm(results.Z, results.V, tmp);
  results.Zqp = nda::array<double, 1>(norb);
  for (long l = 0; l < norb; ++l) {
    ComplexType zqp_l = 0.0;
    for (long a = 0; a < norb; ++a) zqp_l += std::conj(results.V(a, l)) * tmp(a, l);
    results.Zqp(l) = zqp_l.real();
  }
  double trZ = 0.0;
  for (long a = 0; a < norb; ++a) trZ += results.Z(a, a).real();
  results.sumrule = nda::sum(results.Zqp) - trZ;
}

/**
 * linearized_qp_kpoint() under on_failure_e::report: the linearized quasiparticle approx. at
 * one (spin, k), a failure recorded in status rather than aborting. Separate from
 * linearized_qp_kpoint() because it returns early at status 1 and 2, and the abort policy must
 * see every exit.
 *   1. transform Sigma(tau) to the window nodes of the IAFT fermionic Matsubara mesh, one row
 *      of the transform per node (op.w_sel)
 *   2. fit the window (low_freq_coefficients): A = Sigma(0), B = dSigma/d(iw)|_0, fit diagnostics;
 *      an exactly-determined fit whose residual exceeds params.fit_resid_tol stops here (status 1)
 *   3. K = F + A - mu; solve the linearized Dyson equation (solve_linearized_dyson), which stops
 *      with status 2 when 1 - B is not positive definite
 *   4. shift E and diag(H_QP) by +mu, so the result is in absolute energies
 * @param F_ab      - [INPUT] static one-body matrix including H0, (norb, norb)
 * @param Sigma_tab - [INPUT] dynamic self-energy on ft's fermionic tau mesh, (nt, norb, norb);
 *                    contiguous (checked by the IAFT transform)
 * @param mu        - [INPUT] chemical potential [Ha]; the linearization point
 * @param ft        - [INPUT] imaginary-axis transform driver Sigma_tab is sampled on
 * @param op        - [INPUT] fit operator from make_fit_operator(ft, params)
 * @param params    - [INPUT] fit controls; fit_resid_tol is the gate applied in step 2
 * @return the fit members (always), the status, and for status 0 the solved problem; for status 2
 *         only min_eig_Zinv of the matrix-problem members
 * @note Aborts (utils::check) only on inputs whose shapes disagree with each other or with ft,
 *       which are programming errors rather than outcomes.
 */
static result_kpoint_t lqp_kpoint_impl(nda::array_const_view<ComplexType, 2> F_ab,
                                       nda::array_const_view<ComplexType, 3> Sigma_tab,
                                       double mu, imag_axes_ft::IAFT const& ft,
                                       fit_operator_t const& op, params_t const& params) {
  long nt = Sigma_tab.shape(0), norb = Sigma_tab.shape(1);
  utils::check(nt == ft.nt_f(),
               "linearized_qp.cpp::lqp_kpoint_impl: Inconsistent tau samplings: "
               "Sigma_tab has {} tau points, IAFT has {}", nt, ft.nt_f());
  utils::check(F_ab.shape(0) == norb and F_ab.shape(1) == norb and Sigma_tab.shape(2) == norb,
               "linearized_qp.cpp::lqp_kpoint_impl: Inconsistent orbital dimensions: "
               "F_ab has {}x{} orbitals, Sigma_tab has {} orbitals",
               F_ab.shape(0), F_ab.shape(1), Sigma_tab.shape(2));
  
  long nn = op.w_sel.shape(0);

  // tau -> iw at the window nodes only. 
  nda::array<ComplexType, 3> Sigma_wab(nn, norb, norb);
  for (long j = 0; j < nn; ++j)
    ft.tau_to_w(Sigma_tab, Sigma_wab(j, nda::ellipsis{}), imag_axes_ft::fermion, op.w_sel(j));

  // step 1: A = Sigma(w=0), B = dSigma/d(iw)|_{w=0} and the fit diagnostics
  auto fit = low_freq_coefficients(Sigma_wab, op, params);
  result_kpoint_t results;
  results.resid = fit.resid;
  results.anti_herm_A = fit.anti_herm_A;
  results.anti_herm_B = fit.anti_herm_B;
  results.anti_herm_Sigma = fit.anti_herm_Sigma;
  bool exact = (op.M.shape(0) == op.M.shape(1));   // the polynomial interpolates its nodes
  if (exact and results.resid > params.fit_resid_tol) { 
    // status 1: the exactly-determined fit failed the residual gate, omit the step 2. 
    results.status = 1; return results; 
  }

  // K = F + A - mu
  nda::array<ComplexType, 2> K(norb, norb);
  K = F_ab + fit.A;           
  for (long a = 0; a < norb; ++a) K(a, a) -= mu;

  solve_linearized_dyson(K, fit.B, results);
  if (results.status != 0) return results;
  // shift the orbital energies by +mu, so the result is in absolute energies
  results.E += mu;
  for (long a = 0; a < norb; ++a) results.Hqp(a, a) += mu;

  return results;
}

result_kpoint_t linearized_qp_kpoint(nda::array_const_view<ComplexType, 2> F_ab,
                                     nda::array_const_view<ComplexType, 3> Sigma_tab,
                                     double mu, imag_axes_ft::IAFT const& ft,
                                     fit_operator_t const& op, params_t const& params,
                                     on_failure_e on_failure) {
  auto results = lqp_kpoint_impl(F_ab, Sigma_tab, mu, ft, op, params);
  if (on_failure == on_failure_e::abort) abort_on_failure(results, op, params);
  return results;
}

shared_result_t::shared_result_t(
  utils::mpi_context_t<boost::mpi3::communicator>& mpi, 
  long ns, long nk, long norb) : 
  Z_skab(mpi, {ns, nk, norb, norb}), 
  Hqp_skab(mpi, {ns, nk, norb, norb}),
  V_skab(mpi, {ns, nk, norb, norb}), 
  E_ska(nda::zeros<double>(ns, nk, norb)),
  Zqp_ska(nda::zeros<double>(ns, nk, norb)), 
  min_eig_Zinv_sk(nda::zeros<double>(ns, nk)),
  resid_sk(nda::zeros<double>(ns, nk)), 
  anti_herm_A_sk(nda::zeros<double>(ns, nk)),
  anti_herm_B_sk(nda::zeros<double>(ns, nk)), 
  anti_herm_Sigma_sk(nda::zeros<double>(ns, nk)),
  sumrule_sk(nda::zeros<double>(ns, nk)), 
  status_sk(nda::zeros<long>(ns, nk)) {}

result_t to_owned(shared_result_t const& shared) {
  result_t owned;
  owned.Z_skab   = shared.Z_skab.local();
  owned.Hqp_skab = shared.Hqp_skab.local();
  owned.V_skab   = shared.V_skab.local();
  owned.E_ska    = shared.E_ska;
  owned.Zqp_ska  = shared.Zqp_ska;
  owned.min_eig_Zinv_sk = shared.min_eig_Zinv_sk;
  owned.resid_sk   = shared.resid_sk;
  owned.anti_herm_A_sk     = shared.anti_herm_A_sk;
  owned.anti_herm_B_sk     = shared.anti_herm_B_sk;
  owned.anti_herm_Sigma_sk = shared.anti_herm_Sigma_sk;
  owned.sumrule_sk = shared.sumrule_sk;
  owned.status_sk  = shared.status_sk;
  owned.mu = shared.mu; owned.n_fit = shared.n_fit; owned.fit_order = shared.fit_order;
  return owned;
}

ladder_result_t to_owned(shared_ladder_result_t const& shared) {
  ladder_result_t owned;
  owned.last = to_owned(shared.last);
  owned.err_fit_sk    = shared.err_fit_sk;
  owned.dHqp_sk       = shared.dHqp_sk;
  owned.resid_history = shared.resid_history;
  owned.n_accepted     = shared.n_accepted;
  owned.n_fit_cap      = shared.n_fit_cap;
  owned.stopped_resid  = shared.stopped_resid;
  return owned;
}

/**
 * linearized_qp_solve() without its log: the ladder solves every n_fit through this and logs one
 * row of its own table per fit instead.
 */
static shared_result_t solve_points(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                    nda::array_const_view<ComplexType, 4> F_skab,
                                    nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                    double mu, imag_axes_ft::IAFT const& ft,
                                    params_t const& params, on_failure_e on_failure) {
  long ns = F_skab.shape(0), nk = F_skab.shape(1), norb = F_skab.shape(2);
  utils::check(F_skab.shape(3) == norb,
               "linearized_qp.cpp::linearized_qp_solve: F_skab must be square in its last two "
               "indices, got ({}, {}).", norb, F_skab.shape(3));
  utils::check(Sigma_tskab.shape(0) == ft.nt_f() and Sigma_tskab.shape(1) == ns and
               Sigma_tskab.shape(2) == nk and Sigma_tskab.shape(3) == norb and
               Sigma_tskab.shape(4) == norb,
               "linearized_qp.cpp::linearized_qp_solve: Sigma_tskab shape is inconsistent with "
               "F_skab and the IAFT tau mesh ({} points).", ft.nt_f());
  
  auto op = make_fit_operator(ft, params);

  shared_result_t res(mpi, ns, nk, norb);
  res.mu = mu;
  res.n_fit = params.n_fit;
  res.fit_order = op.fit_order;
  auto Z = res.Z_skab.local(), Hqp = res.Hqp_skab.local(), V = res.V_skab.local();

  // Applies f to the three node-shared matrices in member order. The order is the same on every
  // rank, which the window operations below (fence, all_reduce) require: they are collective over
  // the node and the internode communicators.
  auto for_each_shared_array = [&res](auto&& f) {
    for (auto* arr : {&res.Z_skab, &res.Hqp_skab, &res.V_skab}) f(*arr);
  };
  // win().fence() brackets the writes of the ranks of one node to the shared matrices
  auto fence = [](auto& arr) { arr.win().fence(); };
  for_each_shared_array(fence);
  for (long sk = mpi.comm.rank(); sk < ns * nk; sk += mpi.comm.size()) {
    long is = sk / nk, ik = sk % nk;
    // the (t, is, ik, ...) slice of a 5-D array is not contiguous once ns*nk > 1
    auto Sigma_tab = nda::make_regular(Sigma_tskab(nda::range::all, is, ik, nda::ellipsis{}));
    auto results = linearized_qp_kpoint(F_skab(is, ik, nda::ellipsis{}), Sigma_tab, mu, ft, op,
                                        params, on_failure_e::report);
    if (on_failure == on_failure_e::abort) abort_on_failure(results, op, params, is, ik);
    res.status_sk(is, ik)          = results.status;
    res.resid_sk(is, ik)           = results.resid;
    res.anti_herm_A_sk(is, ik)     = results.anti_herm_A;
    res.anti_herm_B_sk(is, ik)     = results.anti_herm_B;
    res.anti_herm_Sigma_sk(is, ik) = results.anti_herm_Sigma;
    
    // fitting residual is too large. Did not run step 2. 
    if (results.status == 1) continue;                  
    
    res.min_eig_Zinv_sk(is, ik) = results.min_eig_Zinv;
    // 1 - B is not positive definite, results contains only min_eig_Zinv. 
    if (results.status == 2) continue;                  

    Z(is, ik, nda::ellipsis{})   = results.Z;
    Hqp(is, ik, nda::ellipsis{}) = results.Hqp;
    V(is, ik, nda::ellipsis{})   = results.V;
    res.E_ska(is, ik, nda::range::all)   = results.E;
    res.Zqp_ska(is, ik, nda::range::all) = results.Zqp;
    res.sumrule_sk(is, ik)               = results.sumrule;
  }
  for_each_shared_array(fence);
  // sum the shared matrices over nodes and the rest over ranks; exact because each (s,k) is
  // written by exactly one rank
  for_each_shared_array([](auto& arr) { arr.all_reduce(); });
  all_reduce_unshared_arrays(res, mpi.comm);
  return res;
}

shared_result_t linearized_qp_solve(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                    nda::array_const_view<ComplexType, 4> F_skab,
                                    nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                    double mu, imag_axes_ft::IAFT const& ft,
                                    params_t const& params, on_failure_e on_failure) {
  auto res = solve_points(mpi, F_skab, Sigma_tskab, mu, ft, params, on_failure);
  // the ladder's table, with this fit as its one row
  log_fit_table_header(std::format("Linearized QP: n_fit = {}", res.n_fit));
  log_fit_row(res, ft.beta());
  log_fit_summary(res, "");
  return res;
}

int max_n_fit_on_mesh(imag_axes_ft::IAFT const& ft, bool symmetric_window) {
  auto wn = ft.wn_mesh_f();
  std::set<long> on_mesh(wn.begin(), wn.end());
  int nf = 0;
  while (true) {
    long n = 2 * (nf + 1) - 1;
    if (not on_mesh.count(n) or (symmetric_window and not on_mesh.count(-n))) return nf;
    ++nf;
  }
}

shared_ladder_result_t linearized_qp_ladder(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                            nda::array_const_view<ComplexType, 4> F_skab,
                                            nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                            double mu, imag_axes_ft::IAFT const& ft,
                                            int n_fit_max, double fit_resid_tol,
                                            bool symmetric_window) {
  long ns = F_skab.shape(0), nk = F_skab.shape(1), norb = F_skab.shape(2);
  // the window of every fit must lie on the IAFT's fermionic sampling mesh
  int n_fit_mesh = max_n_fit_on_mesh(ft, symmetric_window);
  int n_fit_cap = std::min(n_fit_max, n_fit_mesh);
  utils::check(n_fit_cap >= 2,
               "linearized_qp.cpp::linearized_qp_ladder: the ladder starts at n_fit = 2, but "
               "n_fit_max = {} and the IAFT sampling mesh admits n_fit <= {}.",
               n_fit_max, n_fit_mesh);

  std::optional<shared_result_t> last;
  auto err_fit_sk = nda::zeros<double>(ns, nk), dHqp_sk = nda::zeros<double>(ns, nk);
  std::vector<double> residual_history;
  double stopped_resid = 0.0;
  // one row per accepted fit: how the pole weights converge with n_fit
  log_fit_table_header(std::format("Linearized QP ladder: n_fit = 2, ..., {}, max fit residual "
                                   "<= {:.0e}", n_fit_cap, fit_resid_tol));
  for (int nf = 2; nf <= n_fit_cap; ++nf) {
    // exactly determined (the fit interpolates its nodes); the residual gate is applied below,
    // to the max over all points, so the per-point gate is switched off
    params_t const params_nf{.n_fit = nf, .fit_order = -1,
                             .symmetric_window = symmetric_window,
                             .fit_resid_tol = std::numeric_limits<double>::infinity()};
    // a fit with a failed point is reported and the ladder is stopped. 
    auto cand = solve_points(mpi, F_skab, Sigma_tskab, mu, ft, params_nf, on_failure_e::report);
    double resid_max = nda::max_element(cand.resid_sk);
    if (resid_max > fit_resid_tol) {
      app_log(1, "    {:5d}  rejected: max fit residual {:.1e} > {:.0e}", nf, resid_max,
              fit_resid_tol);
      stopped_resid = resid_max;
      break;
    }
    
    // check fitting statuses over all (s, k): a nonzero status is 1 - B not positive definite. 
    long n_failed = std::count_if(cand.status_sk.begin(), cand.status_sk.end(),
                                  [](long st) { return st != 0; });
    if (n_failed > 0) {
      app_log(1, "    {:5d}  rejected: 1 - B not positive definite at {} (s, k) point(s)", nf,
              n_failed);
      break;
    }

    if (last) {
      // estimate the truncation error via the difference between the current and the previous Z and H_QP:
      // ||Z(N) - Z(N-1)||_2 and max |H_QP(N) - H_QP(N-1)|
      auto Z_new = cand.Z_skab.local(), Z_old = last->Z_skab.local();
      auto H_new = cand.Hqp_skab.local(), H_old = last->Hqp_skab.local();
      nda::array<ComplexType, 2> dZ(norb, norb);
      err_fit_sk = 0.0; dHqp_sk = 0.0;
      for (long sk = mpi.comm.rank(); sk < ns * nk; sk += mpi.comm.size()) {
        long is = sk / nk, ik = sk % nk;
        dZ = Z_new(is, ik, nda::ellipsis{}) - Z_old(is, ik, nda::ellipsis{});
        hermitize(dZ());
        auto ev = nda::linalg::eigenvalues(dZ);
        // take the largest eigenvalue in "magnitude". 
        err_fit_sk(is, ik) = std::max(std::abs(ev(0)), std::abs(ev(norb - 1)));
        double dmax = 0.0;
        for (long a = 0; a < norb; ++a)
          for (long b = 0; b < norb; ++b)
            dmax = std::max(dmax, std::abs(H_new(is, ik, a, b) - H_old(is, ik, a, b)));
        dHqp_sk(is, ik) = dmax;
      }
      // exact because each (s,k) is written by exactly one rank
      for (auto* arr : {&err_fit_sk, &dHqp_sk})
        mpi.comm.all_reduce_in_place_n(arr->data(), arr->size(), std::plus<>{});
    }
    residual_history.push_back(resid_max);
    log_fit_row(cand, ft.beta());
    last = std::move(cand);
  }
  // the same outcome on every rank (the residuals and statuses are all-reduced), so every rank
  // aborts or none does
  utils::check(last.has_value(),
               "linearized_qp.cpp::linearized_qp_ladder: no fit accepted: n_fit = 2 {}; the "
               "input is not usable.",
               (stopped_resid > 0.0)
                 ? std::format("fails the residual gate (relative residual {:.2e} > {:.0e})",
                               stopped_resid, fit_resid_tol)
                 : std::string("has 1 - B not positive definite at some (s, k) point"));
  int n_accepted = residual_history.size();
  nda::array<double, 1> resid_history(n_accepted);
  std::copy(residual_history.begin(), residual_history.end(), resid_history.begin());
  
  log_fit_summary(*last, (n_accepted >= 2)
      ? std::format(",\n    truncation error: max ||Z(N) - Z(N-1)||_2 = {:.2e}, "
                    "max |H_QP(N) - H_QP(N-1)| = {:.2e} Ha",
                    nda::max_element(err_fit_sk), nda::max_element(dHqp_sk))
      : std::string(""));
  
  return shared_ladder_result_t{.last = std::move(*last), .err_fit_sk = std::move(err_fit_sk),
                                .dHqp_sk = std::move(dHqp_sk),
                                .resid_history = std::move(resid_history),
                                .n_accepted = n_accepted, .n_fit_cap = n_fit_cap,
                                .stopped_resid = stopped_resid};
}

} // namespace methods::lqp
