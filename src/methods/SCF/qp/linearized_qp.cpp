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
  double hermitize(nda::array_view<ComplexType, 2> X) {
    long n = X.shape(0);
    double anti = 0.0, scale = 0.0;
    for (long a = 0; a < n; ++a) {
      for (long b = a; b < n; ++b) {
        ComplexType h = 0.5 * (X(a, b) + std::conj(X(b, a)));
        ComplexType d = 0.5 * (X(a, b) - std::conj(X(b, a)));
        anti  = std::max(anti, std::abs(d));
        scale = std::max(scale, std::abs(X(a, b)));
        X(a, b) = h; X(b, a) = std::conj(h);
      }
    }
    return (scale > 0.0) ? anti / scale : 0.0;
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

fit_operator_t make_fit_operator(imag_axes_ft::IAFT const& ft, fit_params_t const& p) {
  fit_operator_t op;
  op.n_idx = fit_window(p.n_fit, p.symmetric_window);
  long nn  = op.n_idx.shape(0);
  int  P   = (p.fit_order < 0) ? int(nn) - 1 : p.fit_order;
  utils::check(P >= 1, "linearized_qp.cpp::make_fit_operator: fit_order must be >= 1 (the slope is "
                       "A^(1)), got {}", P);
  utils::check(nn >= P + 1, "linearized_qp.cpp::make_fit_operator: {} fit nodes cannot determine {} "
                            "parameters; raise n_fit or lower fit_order.", nn, P + 1);
  op.fit_order = P;
  op.exact     = (nn == P + 1);

  // window nodes are rows of the IAFT fermionic sampling mesh (IR and DLR both contain the
  // contiguous low odd n; see the design spec, D3) -- select, do not interpolate
  auto wn = ft.wn_mesh_f();
  long nw = wn.shape(0);
  op.w_sel = nda::array<long, 1>(nn);
  long n_on_mesh = 0;                       // largest odd n with 1..n all on the mesh
  for (long n = 1; ; n += 2) {
    bool found = false;
    for (long iw = 0; iw < nw; ++iw) if (long(wn(iw)) == n) { found = true; break; }
    if (!found) break;
    n_on_mesh = n;
  }
  for (long j = 0; j < nn; ++j) {
    long iw = -1;
    for (long k = 0; k < nw; ++k) if (long(wn(k)) == op.n_idx(j)) { iw = k; break; }
    utils::check(iw >= 0, "linearized_qp.cpp::make_fit_operator: window node n = {} is not on the IAFT "
                          "fermionic sampling mesh (contiguous run is |n| <= {}); the largest admissible "
                          "n_fit is {}.", op.n_idx(j), n_on_mesh, (n_on_mesh + 1) / 2);
    op.w_sel(j) = iw;
  }

  // scaled design matrix M_{jp} = (i x_j)^p, x_j = n_j / n_max
  long n_max = 0;
  for (long j = 0; j < nn; ++j) n_max = std::max(n_max, std::abs(op.n_idx(j)));
  op.omega_scale = double(n_max) * M_PI / ft.beta();
  op.M = nda::array<ComplexType, 2>(nn, P + 1);
  for (long j = 0; j < nn; ++j) {
    ComplexType ix(0.0, double(op.n_idx(j)) / double(n_max));
    ComplexType v = 1.0;
    for (int q = 0; q <= P; ++q) { op.M(j, q) = v; v *= ix; }
  }

  // pseudo-inverse by gelss with an identity right-hand side (F layout required)
  nda::matrix<ComplexType, nda::F_layout> Mf(nn, P + 1);
  for (long j = 0; j < nn; ++j)
    for (int q = 0; q <= P; ++q) Mf(j, q) = op.M(j, q);
  nda::matrix<ComplexType, nda::F_layout> Bf(nn, nn);
  Bf() = 0.0;
  for (long j = 0; j < nn; ++j) Bf(j, j) = 1.0;
  nda::vector<double> s(std::min<long>(nn, P + 1));
  int rank = 0;
  int info = nda::lapack::gelss(Mf, Bf, s, 0.0, rank);
  utils::check(info == 0, "linearized_qp.cpp::make_fit_operator: gelss failed with info = {}", info);
  op.cond = s(0) / s(s.shape(0) - 1);
  op.W = nda::array<ComplexType, 2>(P + 1, nn);
  for (int q = 0; q <= P; ++q)
    for (long j = 0; j < nn; ++j) op.W(q, j) = Bf(q, j);
  return op;
}

nda::array<ComplexType, 3>
low_freq_coefficients(nda::array_const_view<ComplexType, 3> Sigma_wab,
                      fit_operator_t const& op, fit_diagnostics_t& diagnostics) {
  long nn = Sigma_wab.shape(0), n = Sigma_wab.shape(1);
  utils::check(nn == op.n_idx.shape(0) and Sigma_wab.shape(2) == n,
               "linearized_qp.cpp::low_freq_coefficients: Sigma_wab shape ({}, {}, {}) does not match "
               "the fit window of {} nodes", nn, n, Sigma_wab.shape(2), op.n_idx.shape(0));
  int P = op.fit_order;

  // Hermiticity of the data on a symmetric window: S(-iw) == S(iw)^dag
  diagnostics.herm_data = 0.0;
  if (nn % 2 == 0 and op.n_idx(0) == -op.n_idx(nn - 1)) {
    for (long j = 0; j < nn / 2; ++j) {
      long jm = j, jp = nn - 1 - j;   // n_idx(jm) == -n_idx(jp)
      for (long a = 0; a < n; ++a)
        for (long b = 0; b < n; ++b)
          diagnostics.herm_data = std::max(diagnostics.herm_data,
                                    std::abs(Sigma_wab(jp, a, b) - std::conj(Sigma_wab(jm, b, a))));
    }
  }

  // one least-squares solve for all n^2 elements: M (nn, P+1) c (P+1, n^2) = S (nn, n^2).
  // Solved per call by gelss (backward stable: residual ~ eps regardless of cond(M)) rather than
  // by applying the precomputed pseudo-inverse op.W, whose residual grows like eps * cond(M) and
  // would trip the exactly-determined residual gate long before the fit itself degrades. The
  // operator is the same linear map either way, so the fit still commutes with any linear
  // transformation of Sigma. rcond mirrors numpy.linalg.lstsq(rcond=None): eps * max(nn, P+1).
  nda::array<ComplexType, 2> S2(nn, n * n);
  for (long j = 0; j < nn; ++j)
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b) S2(j, a * n + b) = Sigma_wab(j, a, b);
  nda::matrix<ComplexType, nda::F_layout> Mf(nn, P + 1), Bf(nn, n * n);
  for (long j = 0; j < nn; ++j) {
    for (int q = 0; q <= P; ++q) Mf(j, q) = op.M(j, q);
    for (long c = 0; c < n * n; ++c) Bf(j, c) = S2(j, c);
  }
  nda::vector<double> sv(std::min<long>(nn, P + 1));
  int rank = 0;
  double rcond = std::numeric_limits<double>::epsilon() * double(std::max<long>(nn, P + 1));
  int info = nda::lapack::gelss(Mf, Bf, sv, rcond, rank);
  utils::check(info == 0, "linearized_qp.cpp::low_freq_coefficients: gelss failed with info = {}", info);
  nda::array<ComplexType, 2> c2(P + 1, n * n);
  for (int q = 0; q <= P; ++q)
    for (long c = 0; c < n * n; ++c) c2(q, c) = Bf(q, c);

  // residual of the fit, relative to the data scale
  nda::array<ComplexType, 2> R(nn, n * n);
  nda::blas::gemm(op.M, c2, R);
  R -= S2;
  double smax = 0.0, rmax = 0.0;
  for (auto const& v : S2) smax = std::max(smax, std::abs(v));
  for (auto const& v : R)  rmax = std::max(rmax, std::abs(v));
  diagnostics.resid = (smax > 0.0) ? rmax / smax : 0.0;

  // unscale and Hermitize
  nda::array<ComplexType, 3> coeffs(P + 1, n, n);
  double f = 1.0;
  for (int q = 0; q <= P; ++q) {
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b) coeffs(q, a, b) = c2(q, a * n + b) / f;
    f *= op.omega_scale;
    double anti = hermitize(coeffs(q, nda::ellipsis{}));
    if (q == 0) diagnostics.anti_herm_A = anti;
    if (q == 1) diagnostics.anti_herm_B = anti;
  }
  return coeffs;
}

qp_matrix_t linearized_qp_matrix(nda::array_const_view<ComplexType, 2> K_in,
                                 nda::array_const_view<ComplexType, 2> B_in) {
  long n = K_in.shape(0);
  utils::check(K_in.shape(1) == n and B_in.shape(0) == n and B_in.shape(1) == n,
               "linearized_qp.cpp::linearized_qp_matrix: K and B must be square and of equal size");
  nda::array<ComplexType, 2> K(K_in), B(B_in);
  hermitize(K()); hermitize(B());

  nda::array<ComplexType, 2> ImB(n, n);
  ImB = -B;
  for (long a = 0; a < n; ++a) ImB(a, a) += 1.0;
  auto [w, U] = nda::linalg::eigenelements(ImB);          // ascending eigenvalues, columns
  qp_matrix_t r;
  r.min_eig = w(0);
  if (r.min_eig <= 0.0) { r.valid = false; return r; }   // caller aborts or reports (s,k)

  // Z = U diag(1/w) U^dag,  Z^1/2 = U diag(1/sqrt w) U^dag
  nda::array<ComplexType, 2> Uw(n, n), Uwh(n, n);
  for (long l = 0; l < n; ++l) {
    Uw (nda::range::all, l) = U(nda::range::all, l) * (1.0 / w(l));
    Uwh(nda::range::all, l) = U(nda::range::all, l) * (1.0 / std::sqrt(w(l)));
  }
  nda::array<ComplexType, 2> Zhalf(n, n);            // construction intermediate, not kept
  r.Z = nda::array<ComplexType, 2>(n, n);
  nda::blas::gemm(Uw,  nda::dagger(U), r.Z);
  nda::blas::gemm(Uwh, nda::dagger(U), Zhalf);
  hermitize(r.Z()); hermitize(Zhalf());

  // H_QP = Z^1/2 K Z^1/2
  nda::array<ComplexType, 2> tmp(n, n);
  r.Hqp = nda::array<ComplexType, 2>(n, n);
  nda::blas::gemm(K, Zhalf, tmp);
  nda::blas::gemm(Zhalf, tmp, r.Hqp);
  hermitize(r.Hqp());
  auto [E, V] = nda::linalg::eigenelements(r.Hqp);
  r.E = E; r.V = V;

  // Pole weights: the diagonal of Z in the H_QP eigenbasis, <v_l|Z|v_l>.
  nda::array<ComplexType, 2> ZV(n, n), Zqp_mat(n, n);
  nda::blas::gemm(r.Z, r.V, ZV);
  nda::blas::gemm(nda::dagger(r.V), ZV, Zqp_mat);
  r.Zqp = nda::array<double, 1>(n);
  for (long l = 0; l < n; ++l) r.Zqp(l) = Zqp_mat(l, l).real();
  double trZ = 0.0;
  for (long a = 0; a < n; ++a) trZ += r.Z(a, a).real();
  r.sumrule = nda::sum(r.Zqp) - trZ;
  return r;
}

point_result_t try_solve_point(nda::array_const_view<ComplexType, 2> F_ab,
                               nda::array_const_view<ComplexType, 3> Sigma_tab,
                               double mu, imag_axes_ft::IAFT const& ft,
                               fit_operator_t const& op, fit_params_t const& p, int& status) {
  long nt = Sigma_tab.shape(0), n = Sigma_tab.shape(1);
  utils::check(nt == ft.nt_f(), "linearized_qp.cpp::try_solve_point: Sigma_tab has {} tau points, IAFT "
                                "has {}", nt, ft.nt_f());
  utils::check(F_ab.shape(0) == n and F_ab.shape(1) == n and Sigma_tab.shape(2) == n,
               "linearized_qp.cpp::try_solve_point: F_ab and Sigma_tab dimensions disagree");
  long nw = ft.nw_f(), nn = op.n_idx.shape(0);
  status = 0;

  // tau -> iw on the IAFT sampling mesh, then select the window rows
  nda::array<ComplexType, 2> St(nt, n * n), Sw(nw, n * n), Snode(nn, n * n);
  for (long it = 0; it < nt; ++it)
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b) St(it, a * n + b) = Sigma_tab(it, a, b);
  ft.tau_to_w(St, Sw, imag_axes_ft::fermion);
  for (long j = 0; j < nn; ++j) Snode(j, nda::range::all) = Sw(op.w_sel(j), nda::range::all);
  nda::array<ComplexType, 3> Sigma_wab(nn, n, n);
  for (long j = 0; j < nn; ++j)
    for (long a = 0; a < n; ++a)
      for (long b = 0; b < n; ++b) Sigma_wab(j, a, b) = Snode(j, a * n + b);

  point_result_t r;
  auto coeffs = low_freq_coefficients(Sigma_wab, op, r.diagnostics);
  r.A = coeffs(0, nda::ellipsis{});
  r.B = coeffs(1, nda::ellipsis{});
  if (op.exact and r.diagnostics.resid > p.fit_resid_tol) { status = 1; return r; }

  nda::array<ComplexType, 2> K(n, n);
  K = F_ab + r.A;
  for (long a = 0; a < n; ++a) K(a, a) -= mu;
  r.qp = linearized_qp_matrix(K, r.B);
  if (not r.qp.valid) { status = 2; return r; }
  // One energy convention everywhere: the linearization is about z = mu (K carries -mu inside
  // the Z^1/2 sandwich, which is not a constant shift), but the quasiparticle energies and
  // H_QP are returned absolute, like every other CoQui energy. mu stays in result_t for a
  // caller who wants them relative to the Fermi level.
  r.qp.E += mu;
  for (long a = 0; a < n; ++a) r.qp.Hqp(a, a) += mu;
  return r;
}

point_result_t solve_point(nda::array_const_view<ComplexType, 2> F_ab,
                           nda::array_const_view<ComplexType, 3> Sigma_tab,
                           double mu, imag_axes_ft::IAFT const& ft,
                           fit_operator_t const& op, fit_params_t const& p) {
  int status = 0;
  auto r = try_solve_point(F_ab, Sigma_tab, mu, ft, op, p, status);
  utils::check(status != 1,
               "linearized_qp.cpp::solve_point: exactly-determined fit (n_fit={}, fit_order={}) no "
               "longer interpolates: relative residual {:.2e} > {:.0e}, cond(design)={:.2e}. Lower "
               "n_fit.", p.n_fit, op.fit_order, r.diagnostics.resid, p.fit_resid_tol, op.cond);
  utils::check(status != 2,
               "linearized_qp.cpp::solve_point: 1 - B is not positive definite (min eigenvalue {:.3e}); "
               "for a causal self-energy B <= 0, so this is a failed fit or a sign error. Not regularized.",
               r.qp.min_eig);
  return r;
}

// The one loop behind both public overloads. comm == nullptr means serial: every point on
// this process and no reduction, which also keeps the array path free of any MPI requirement
// (the Python binding may be called in a process that never initialized MPI).
static result_t solve_impl(boost::mpi3::communicator* comm,
                           nda::array_const_view<ComplexType, 4> F_skab,
                           nda::array_const_view<ComplexType, 5> Sigma_tskab,
                           double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                           on_failure_e on_failure) {
  long ns = F_skab.shape(0), nk = F_skab.shape(1), n = F_skab.shape(2);
  long rank = comm ? comm->rank() : 0, size = comm ? comm->size() : 1;
  utils::check(F_skab.shape(3) == n,
               "linearized_qp.cpp::linearized_qp_solve: F_skab must be square in its last two "
               "indices, got ({}, {}).", n, F_skab.shape(3));
  utils::check(Sigma_tskab.shape(0) == ft.nt_f() and Sigma_tskab.shape(1) == ns and
               Sigma_tskab.shape(2) == nk and Sigma_tskab.shape(3) == n and Sigma_tskab.shape(4) == n,
               "linearized_qp.cpp::linearized_qp_solve: Sigma_tskab shape is inconsistent with "
               "F_skab and the IAFT tau mesh ({} points).", ft.nt_f());
  auto op = make_fit_operator(ft, p);

  result_t res;
  res.mu = mu; res.n_fit = p.n_fit; res.fit_order = op.fit_order; res.cond = op.cond;
  for (auto* X : {&res.Z_skab, &res.Hqp_skab, &res.V_skab})
    *X = nda::array<ComplexType, 4>::zeros({ns, nk, n, n});
  res.E_ska   = nda::array<double, 3>::zeros({ns, nk, n});
  res.Zqp_ska = nda::array<double, 3>::zeros({ns, nk, n});
  for (auto* X : {&res.min_eig_sk, &res.resid_sk, &res.anti_herm_A_sk, &res.anti_herm_B_sk,
                  &res.herm_data_sk, &res.sumrule_sk})
    *X = nda::array<double, 2>::zeros({ns, nk});
  res.status_sk   = nda::array<long, 2>::zeros({ns, nk});

  for (long sk = rank; sk < ns * nk; sk += size) {
    long is = sk / nk, ik = sk % nk;
    // the (t, is, ik, ...) slice of a 5-D array is not contiguous once ns*nk > 1
    auto Sigma_tab = nda::make_regular(Sigma_tskab(nda::range::all, is, ik, nda::ellipsis{}));
    int status = 0;
    auto r = try_solve_point(F_skab(is, ik, nda::ellipsis{}), Sigma_tab, mu, ft, op, p, status);
    if (on_failure == on_failure_e::abort) {
      utils::check(status != 1,
                   "linearized_qp.cpp::linearized_qp_solve: at (s={}, k={}) the exactly-determined "
                   "fit (n_fit={}, fit_order={}) no longer interpolates the data: relative residual "
                   "{:.2e} > {:.0e} (cond {:.2e}). Lower n_fit.", is, ik, p.n_fit, op.fit_order,
                   r.diagnostics.resid, p.fit_resid_tol, op.cond);
      utils::check(status != 2,
                   "linearized_qp.cpp::linearized_qp_solve: at (s={}, k={}) 1 - B is not positive "
                   "definite (min eigenvalue {:.3e}); Sigma(iw) is not causal there or the fit "
                   "failed.", is, ik, r.qp.min_eig);
    }
    res.status_sk(is, ik) = status;
    res.resid_sk(is, ik)       = r.diagnostics.resid;
    res.anti_herm_A_sk(is, ik) = r.diagnostics.anti_herm_A;
    res.anti_herm_B_sk(is, ik) = r.diagnostics.anti_herm_B;
    res.herm_data_sk(is, ik)   = r.diagnostics.herm_data;
    if (status == 1) continue;                    // qp was not computed
    res.min_eig_sk(is, ik) = r.qp.min_eig;
    if (status == 2) continue;                    // only min_eig is meaningful
    res.Z_skab(is, ik, nda::ellipsis{})     = r.qp.Z;
    res.Hqp_skab(is, ik, nda::ellipsis{})   = r.qp.Hqp;
    res.V_skab(is, ik, nda::ellipsis{})     = r.qp.V;
    res.E_ska(is, ik, nda::range::all)      = r.qp.E;
    res.Zqp_ska(is, ik, nda::range::all)    = r.qp.Zqp;
    res.sumrule_sk(is, ik)                  = r.qp.sumrule;
  }
  if (comm) {
    for (auto* X : {&res.Z_skab, &res.Hqp_skab, &res.V_skab})
      comm->all_reduce_in_place_n(X->data(), X->size(), std::plus<>{});
    for (auto* X : {&res.E_ska, &res.Zqp_ska})
      comm->all_reduce_in_place_n(X->data(), X->size(), std::plus<>{});
    for (auto* X : {&res.min_eig_sk, &res.resid_sk, &res.anti_herm_A_sk, &res.anti_herm_B_sk,
                    &res.herm_data_sk, &res.sumrule_sk})
      comm->all_reduce_in_place_n(X->data(), X->size(), std::plus<>{});
    comm->all_reduce_in_place_n(res.status_sk.data(), res.status_sk.size(), std::plus<>{});
  }
  return res;
}

result_t linearized_qp_solve(boost::mpi3::communicator& comm,
                             nda::array_const_view<ComplexType, 4> F_skab,
                             nda::array_const_view<ComplexType, 5> Sigma_tskab,
                             double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                             on_failure_e on_failure) {
  return solve_impl(std::addressof(comm), F_skab, Sigma_tskab, mu, ft, p, on_failure);  // mpi3 overloads operator&
}

result_t linearized_qp_solve(nda::array_const_view<ComplexType, 4> F_skab,
                             nda::array_const_view<ComplexType, 5> Sigma_tskab,
                             double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                             on_failure_e on_failure) {
  return solve_impl(nullptr, F_skab, Sigma_tskab, mu, ft, p, on_failure);
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

static ladder_result_t ladder_impl(boost::mpi3::communicator* comm,
                                   nda::array_const_view<ComplexType, 4> F_skab,
                                   nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                   double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                                   int n_fit_max, on_failure_e on_failure) {
  long ns = F_skab.shape(0), nk = F_skab.shape(1), n = F_skab.shape(2);
  ladder_result_t lad;
  lad.n_fit_mesh_max = max_n_fit_on_mesh(ft, p.symmetric_window);
  lad.mesh_limited = lad.n_fit_mesh_max < n_fit_max;
  int top = std::min(n_fit_max, lad.n_fit_mesh_max);
  lad.err_fit_sk = nda::array<double, 2>::zeros({ns, nk});
  lad.dHqp_sk    = nda::array<double, 2>::zeros({ns, nk});

  std::vector<long> nf_hist; std::vector<double> resid_hist;
  std::vector<nda::array<double, 3>> zqp_hist;
  for (int nf = 2; nf <= top; ++nf) {
    fit_params_t q = p;
    q.n_fit = nf; q.fit_order = -1;                     // exactly determined: the rung interpolates
    q.fit_resid_tol = std::numeric_limits<double>::infinity();   // the gate is applied here
    auto cand = solve_impl(comm, F_skab, Sigma_tskab, mu, ft, q, on_failure);
    double resid_max = nda::max_element(cand.resid_sk);
    if (resid_max > p.fit_resid_tol) {
      lad.stopped_n_fit = nf; lad.stopped_resid = resid_max; lad.stopped_cond = cand.cond;
      break;
    }
    if (lad.n_accepted > 0) {
      // truncation-error estimate: ||Z(N) - Z(N-1)||_2 per point, and the change of H_QP
      nda::array<ComplexType, 2> dZ(n, n);
      for (long is = 0; is < ns; ++is)
        for (long ik = 0; ik < nk; ++ik) {
          dZ = cand.Z_skab(is, ik, nda::ellipsis{}) - lad.last.Z_skab(is, ik, nda::ellipsis{});
          hermitize(dZ());
          auto ev = nda::linalg::eigenvalues(dZ);
          lad.err_fit_sk(is, ik) = std::max(std::abs(ev(0)), std::abs(ev(n - 1)));
          double dh = 0.0;
          for (long a = 0; a < n; ++a)
            for (long b = 0; b < n; ++b)
              dh = std::max(dh, std::abs(cand.Hqp_skab(is, ik, a, b) - lad.last.Hqp_skab(is, ik, a, b)));
          lad.dHqp_sk(is, ik) = dh;
        }
    }
    nf_hist.push_back(nf); resid_hist.push_back(resid_max); zqp_hist.push_back(cand.Zqp_ska);
    lad.last = std::move(cand);
    ++lad.n_accepted;
  }
  lad.n_fit_history = nda::array<long, 1>(lad.n_accepted);
  lad.resid_history = nda::array<double, 1>(lad.n_accepted);
  lad.Zqp_history   = nda::array<double, 4>(lad.n_accepted, ns, nk, n);
  for (int i = 0; i < lad.n_accepted; ++i) {
    lad.n_fit_history(i) = nf_hist[i]; lad.resid_history(i) = resid_hist[i];
    lad.Zqp_history(i, nda::ellipsis{}) = zqp_hist[i];
  }
  return lad;
}

ladder_result_t linearized_qp_ladder(boost::mpi3::communicator& comm,
                                     nda::array_const_view<ComplexType, 4> F_skab,
                                     nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                     double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                                     int n_fit_max, on_failure_e on_failure) {
  return ladder_impl(std::addressof(comm), F_skab, Sigma_tskab, mu, ft, p, n_fit_max, on_failure);
}

ladder_result_t linearized_qp_ladder(nda::array_const_view<ComplexType, 4> F_skab,
                                     nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                     double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                                     int n_fit_max, on_failure_e on_failure) {
  return ladder_impl(nullptr, F_skab, Sigma_tskab, mu, ft, p, n_fit_max, on_failure);
}

} // namespace methods::lqp
