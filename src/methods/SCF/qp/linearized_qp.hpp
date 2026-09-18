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

#ifndef COQUI_LINEARIZED_QP_HPP
#define COQUI_LINEARIZED_QP_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "mpi3/communicator.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

/**
 * Linearized quasiparticle approximation for a dynamic self-energy.
 *
 *   Sigma(iw) ~ A + iw B,   K = F + A - mu,   Z = (1 - B)^-1,
 *   H_QP = Z^1/2 K Z^1/2,   H_QP v = eps v,   Z_qp = <v|Z|v>.
 *
 * Basis-agnostic (any orthonormal basis), per (spin, k). Derivation, fitting rules,
 * traps and measurements:
 *   implementation_notes/quasiparticle_Z_at_arbitrary_k/qp_Z_at_arbitrary_k_note.tex
 *   implementation_notes/quasiparticle_approx/quasiparticle_Z_LQSGW_note.tex
 * Reference implementation: src/python/post_proc/linearized_qp.py
 *
 * This is a different approximation from qp_type == "linearized" (qp_solvers.hpp), which
 * Taylor-expands the *diagonal* quasiparticle equation around the KS energy on the real
 * axis. Here the full matrix Sigma(iw) is expanded around w = 0 on the Matsubara axis.
 *
 * Units and conventions throughout: Hartree, beta in Ha^-1, CoQui Matsubara index n
 * (odd for fermions, w_n = n pi / beta), and quasiparticle energies measured from mu.
 *
 * Typical use at fixed (spin, k), with F including H0:
 *   auto op = make_fit_operator(ft, p);              // once per (ft, p)
 *   auto r  = solve_point(F_ab, Sigma_tab, mu, ft, op, p);
 * or over full arrays in one call: linearized_qp_solve(comm, F_skab, Sigma_tskab, mu, ft, p).
 */
namespace methods::lqp {

/**
 * Controls of the low-frequency fit of Sigma(iw) and of the pole-weight extraction.
 * Defaults are the ones validated in the notes above.
 */
struct fit_params_t {
  // Number of lowest positive fermionic Matsubara frequencies in the fit window. With
  // symmetric_window the window is n = +/-1, +/-3, ..., +/-(2 n_fit - 1), i.e. 2 n_fit nodes.
  int    n_fit            = 6;
  // Highest power of (i w) in the polynomial; -1 means "#nodes - 1", the exactly-determined
  // (interpolating) fit, which is the recommended setting.
  int    fit_order        = -1;
  // Use +/-w pairs. This makes the Hermiticity of the coefficients exact by construction and
  // decouples the even/odd powers; turning it off is for diagnostics only.
  bool   symmetric_window = true;
  // Residual gate of an exactly-determined fit: the polynomial must reproduce the data to this
  // relative accuracy, otherwise the design matrix has lost conditioning (too many nodes) and
  // the coefficients are unreliable.
  double fit_resid_tol    = 1e-8;
};

// Tolerance [Ha] for treating eigenvalues of H_QP as degenerate. Inside a degenerate block Z is
// diagonalized (blockwise projection) so the pole weights do not depend on the arbitrary
// eigenvector choice within the block. Deliberately a fixed constant rather than an input:
// looser reassigns the weights of distinct close bands, tighter is a no-op (a symmetry multiplet
// carries a single irrep, so Schur's lemma already makes Z proportional to the identity on it).
inline constexpr double degeneracy_tol = 1e-8;

/**
 * Everything about the fit that depends only on (IAFT, fit_params_t) and not on the data:
 * build it once with make_fit_operator() and reuse it for every (spin, k).
 */
struct fit_operator_t {
  nda::array<long, 1>        n_idx;      // CoQui Matsubara indices of the window (nnodes)
  nda::array<long, 1>        w_sel;      // (nnodes): row of each window node in IAFT::wn_mesh_f()
  nda::array<ComplexType, 2> M;          // (nnodes, P+1) scaled design matrix (i x_j)^p
  nda::array<ComplexType, 2> W;          // (P+1, nnodes) pseudo-inverse of M
  double omega_scale = 0.0;              // n_max * pi / beta; A^(p) = c_p / omega_scale^p
  double cond        = 0.0;              // condition number of M
  bool   exact       = false;            // P + 1 == nnodes
  int    fit_order   = 0;                // P, the highest power actually used
};

/** Per-(spin,k) diagnostics of the fit; all dimensionless, all "smaller is better". */
struct fit_diagnostics_t {
  double resid       = 0.0;   // max |M c - S| / max |S| over all elements
  double anti_herm_A = 0.0;   // |anti-Hermitian part of A| / |A|, before Hermitization
  double anti_herm_B = 0.0;   // same for B
  double herm_data   = 0.0;   // max |S(iw) - S(-iw)^dag| on the window (0 if not symmetric)
};

/** The linearized quasiparticle problem at one (spin, k), solved. */
struct qp_matrix_t {
  nda::array<double, 1>      E;      // eigenvalues of H_QP, measured from mu [Ha]
  nda::array<ComplexType, 2> V;      // eigenvectors of H_QP (columns)
  nda::array<ComplexType, 2> Z;      // (1 - B)^-1
  nda::array<ComplexType, 2> Zhalf;  // Z^1/2
  nda::array<ComplexType, 2> Hqp;    // Z^1/2 K Z^1/2, Hermitized
  nda::array<double, 1>      Zqp;    // <v_l|Z|v_l>, block-resolved at degeneracies
  double min_eig  = 0.0;             // smallest eigenvalue of 1 - B (> 0 for causal Sigma)
  double sumrule  = 0.0;             // sum(Zqp) - Tr Z; a free consistency check, should be ~0
  int    ndeg_max = 1;               // largest degenerate block encountered
  bool   valid    = true;            // false: 1 - B not positive definite; only min_eig is set
};

/** Result of the fit and the solve at one (spin, k). */
struct point_result_t {
  qp_matrix_t qp;
  nda::array<ComplexType, 2> A, B;   // Hermitized zero-frequency value and slope of Sigma(iw)
  fit_diagnostics_t diagnostics;
};

/** Result over all (spin, k) from linearized_qp_solve(); arrays are (ns, nk, ...). */
struct result_t {
  nda::array<ComplexType, 4> K_skab, B_skab, Z_skab, Zhalf_skab, Hqp_skab, V_skab;
  nda::array<double, 3>      E_ska, Zqp_ska;   // E measured from mu; Zqp = <v|Z|v>
  nda::array<double, 2>      min_eig_sk, resid_sk;
  double mu = 0.0;                             // the mu that was subtracted [Ha]
  int    n_fit = 0, fit_order = 0;             // window actually used
  double cond = 0.0;                           // condition number of the design matrix
};

/**
 * Matsubara indices of the fit window in CoQui's convention (odd n, w_n = n pi / beta).
 * @param n_fit           - [INPUT] number of lowest positive fermionic frequencies (>= 1)
 * @param symmetric_window - [INPUT] if true return the 2*n_fit indices
 *                           -(2 n_fit - 1), ..., -1, 1, ..., 2 n_fit - 1 (ascending);
 *                           if false only the n_fit positive ones
 * @return the window indices, shape (2*n_fit) or (n_fit)
 */
nda::array<long, 1> fit_window(int n_fit, bool symmetric_window);

/**
 * Data-independent part of the fit: the window, the scaled design matrix and its pseudo-inverse,
 * and the position of each window node on the IAFT fermionic sampling mesh. No Matsubara
 * interpolation is involved: both the IR and the DLR meshes contain the lowest odd n as a
 * contiguous run, so the window is a row selection of a tau -> iw transform.
 * @param ft - [INPUT] imaginary-axis transform driver; supplies beta and wn_mesh_f()
 * @param p  - [INPUT] fit controls; n_fit, fit_order and symmetric_window are used here
 * @return the fit operator, reusable for every (spin, k) with this (ft, p)
 * @note Aborts (utils::check) if a window node is not on ft's fermionic sampling mesh, naming
 *       the largest admissible n_fit, and if the fit is under-determined (nnodes < fit_order + 1).
 */
fit_operator_t make_fit_operator(imag_axes_ft::IAFT const& ft, fit_params_t const& p);

/**
 * Step 1 at one (spin, k): fit Sigma(iw) on the window to a polynomial in (i w) and return its
 * coefficients. The same linear operator acts on every matrix element, so the fit commutes with
 * any linear transformation of Sigma (Wannier interpolation, basis rotation). Each coefficient
 * is Hermitized after unscaling.
 * @param Sigma_wab - [INPUT] self-energy on the window, (nnodes, n, n), ordered as op.n_idx
 * @param op        - [INPUT] fit operator from make_fit_operator()
 * @param diagnostics - [OUTPUT] residual, anti-Hermiticity of A and B, data Hermiticity
 * @return coefficients A^(p), shape (fit_order + 1, n, n), with A^(p) in Ha^{1-p}; A^(0) is
 *         Sigma(0) and A^(1) is dSigma/d(i w) at w = 0
 */
nda::array<ComplexType, 3>
low_freq_coefficients(nda::array_const_view<ComplexType, 3> Sigma_wab,
                      fit_operator_t const& op, fit_diagnostics_t& diagnostics);

/**
 * Step 2 at one (spin, k): from the static matrix K and the slope B, build Z = (1 - B)^-1, the
 * quasiparticle Hamiltonian H_QP = Z^1/2 K Z^1/2, its eigenbasis, and the pole weights
 * <v|Z|v> (blockwise-projected at degeneracies).
 * @param K       - [INPUT] static one-body matrix F + Sigma(0) - mu, (n, n); Hermitized on entry
 * @param B       - [INPUT] slope dSigma/d(i w) at w = 0, (n, n); Hermitized on entry
 * @param deg_tol - [INPUT] degeneracy tolerance [Ha] for the blockwise projection of Z;
 *                  leave it at the default unless you are studying the blocking itself
 * @return the solved problem; on success E, V, Z, Zhalf, Hqp, Zqp, sumrule and ndeg_max are set
 * @note If 1 - B is not positive definite the result carries valid == false and only min_eig;
 *       the kernel does not abort, because utils::check terminates the process and a unit test
 *       could not observe it. Callers decide how to fail (solve_point aborts, the Python
 *       backend reports NaN). For a causal Sigma the slope is negative semidefinite, so
 *       min_eig >= 1 and Z has spectrum in (0, 1]: a violation is a failed fit or a sign error,
 *       never physics, and must not be regularized away.
 */
qp_matrix_t linearized_qp_matrix(nda::array_const_view<ComplexType, 2> K,
                                 nda::array_const_view<ComplexType, 2> B,
                                 double deg_tol = degeneracy_tol);

/**
 * Steps 1 + 2 at one (spin, k) starting from Sigma(tau): transform to the Matsubara axis, select
 * the window, fit, and solve the linearized problem. Reports failures through a status code
 * instead of aborting, for callers that want to keep going (the Python backend).
 * @param F_ab      - [INPUT] static one-body matrix, (n, n), **including H0** (the checkpoint's
 *                    F_skij does not contain it; adding it is the caller's responsibility)
 * @param Sigma_tab - [INPUT] dynamic self-energy on ft's fermionic tau mesh, (nt, n, n);
 *                    Any layout: the routine packs the data into its own buffer, so a
 *                    non-contiguous (t, is, ik, ...) slice of a 5-D array may be passed directly
 * @param mu        - [INPUT] chemical potential [Ha]; subtracted from the diagonal of K
 * @param ft        - [INPUT] imaginary-axis transform driver
 * @param op        - [INPUT] fit operator from make_fit_operator(), consistent with ft and p
 * @param p         - [INPUT] fit controls; fit_resid_tol is the gate applied here
 * @param status    - [OUTPUT] 0 = ok; 1 = exactly-determined fit failed the residual gate
 *                    (A, B and diagnostics are filled, qp is not computed); 2 = 1 - B is not
 *                    positive definite (qp.valid == false, only qp.min_eig set)
 * @return the fit coefficients A, B, the diagnostics, and (status == 0) the solved problem
 */
point_result_t try_solve_point(nda::array_const_view<ComplexType, 2> F_ab,
                               nda::array_const_view<ComplexType, 3> Sigma_tab,
                               double mu, imag_axes_ft::IAFT const& ft,
                               fit_operator_t const& op, fit_params_t const& p, int& status);

/**
 * try_solve_point() with the failure modes turned into aborts; this is what the C++ consumers use.
 * Parameters are those of try_solve_point() without `status`.
 * @param F_ab      - [INPUT] static one-body matrix including H0, (n, n)
 * @param Sigma_tab - [INPUT] Sigma(tau) on ft's fermionic mesh, (nt, n, n); any layout
 * @param mu        - [INPUT] chemical potential [Ha]
 * @param ft        - [INPUT] imaginary-axis transform driver
 * @param op        - [INPUT] fit operator from make_fit_operator()
 * @param p         - [INPUT] fit controls
 * @return the solved problem at this (spin, k)
 * @note Aborts (utils::check) on a failed residual gate, naming the residual and the condition
 *       number, and on a non-positive-definite 1 - B, naming its smallest eigenvalue.
 */
point_result_t solve_point(nda::array_const_view<ComplexType, 2> F_ab,
                           nda::array_const_view<ComplexType, 3> Sigma_tab,
                           double mu, imag_axes_ft::IAFT const& ft,
                           fit_operator_t const& op, fit_params_t const& p);

/**
 * Convenience driver: solve_point() for every (spin, k) of full arrays, distributed round-robin
 * over comm and all-reduced, so every rank returns the complete result.
 * @param comm        - [INPUT] communicator the (spin, k) work is spread over
 * @param F_skab      - [INPUT] static one-body matrices, (ns, nk, n, n), **including H0**
 * @param Sigma_tskab - [INPUT] dynamic self-energy, (nt, ns, nk, n, n), on ft's fermionic tau mesh
 * @param mu          - [INPUT] chemical potential [Ha]
 * @param ft          - [INPUT] imaginary-axis transform driver
 * @param p           - [INPUT] fit controls; the fit operator is built internally
 * @return K, B, Z, Z^1/2, H_QP, eigenvectors, energies (from mu), pole weights, and the
 *         per-(spin,k) diagnostics min_eig_sk and resid_sk
 * @note Aborts through solve_point() if any (spin, k) fails the residual gate or the causality
 *       check, so a returned result is always usable.
 */
result_t linearized_qp_solve(boost::mpi3::communicator& comm,
                             nda::array_const_view<ComplexType, 4> F_skab,
                             nda::array_const_view<ComplexType, 5> Sigma_tskab,
                             double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p);

} // namespace methods::lqp

#endif // COQUI_LINEARIZED_QP_HPP
