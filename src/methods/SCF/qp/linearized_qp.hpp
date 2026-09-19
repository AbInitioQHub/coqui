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
#include "methods/SCF/qp/qp_params_t.h"

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
 * Independent NumPy reference (validation only, not part of the package):
 *   implementation_notes/quasiparticle_Z_at_arbitrary_k/linearized_qp_numpy.py
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
 *
 * The result types form one hierarchy, from a single (spin, k) up to full arrays:
 *   fit_operator_t     what depends only on (IAFT, fit_params_t): window, design matrix
 *   fit_diagnostics_t  how well the fit at one (spin, k) went (step 1)
 *   qp_matrix_t        the solved matrix problem at one (spin, k)      (step 2)
 *   point_result_t     = A, B, fit_diagnostics_t, qp_matrix_t at one (spin, k)
 *   result_t           the same fields for every (spin, k), as (ns, nk, ...) arrays, plus a
 *                      per-point status; produced by linearized_qp_solve()
 * Every consumer -- the LQSGW loop, the checkpoint post-processing and the array-in/array-out
 * post-processing behind the Python API -- goes through these.
 */
namespace methods::lqp {

/**
 * Everything about the fit that depends only on (IAFT, fit_params_t) and not on the data:
 * Built once by make_fit_operator() and reused for every (spin, k).
 */
struct fit_operator_t {
  // CoQui Matsubara indices of the window, ascending: +-1, +-3, ..., +-(2 n_fit - 1) when
  // symmetric, the positive half otherwise. Length nnodes.
  nda::array<long, 1>        n_idx;
  // For each window node, the row of IAFT::wn_mesh_f() (i.e. of tau_to_w's output) holding it.
  // The kernel selects rows rather than interpolating, so every node must be on the mesh.
  nda::array<long, 1>        w_sel;
  // Scaled design matrix, (nnodes, P+1): M_jp = (i x_j)^p with x_j = n_j / n_max in [-1, 1].
  nda::array<ComplexType, 2> M;
  // Pseudo-inverse of M, (P+1, nnodes). Kept for diagnostics; the solve itself uses gelss
  // on M per call, which is backward stable where applying W would lose eps * cond.
  nda::array<ComplexType, 2> W;
  // n_max * pi / beta [Ha]. Unscaling: the physical coefficient is A^(p) = c_p / omega_scale^p.
  double omega_scale = 0.0;
  // Condition number of M. Grows with the window; the residual gate is what catches its loss.
  double cond        = 0.0;
  // True when P + 1 == nnodes: the polynomial interpolates the data exactly, and the residual
  // gate (fit_params_t::fit_resid_tol) applies.
  bool   exact       = false;
  // P, the highest power of (i w) actually used (fit_params_t::fit_order resolved from -1).
  int    fit_order   = 0;
};

/**
 * How well the fit at one (spin, k) went (step 1). All dimensionless, all "smaller is better";
 * none of them is a physical quantity.
 */
struct fit_diagnostics_t {
  // Relative residual of the polynomial fit, max |M c - S| / max |S| over all matrix elements.
  // For an exact fit this should be at round-off; larger means the design matrix has lost
  // conditioning and the coefficients are unreliable (the residual gate acts on this).
  double resid       = 0.0;
  // |anti-Hermitian part of A| / |A| before Hermitization; with a symmetric window this is
  // round-off by construction, so anything larger points at the data (see herm_data).
  double anti_herm_A = 0.0;
  // The same for the slope B.
  double anti_herm_B = 0.0;
  // max |S(iw) - S(-iw)^dag| over the window: how far the input itself is from the
  // Hermiticity a fermionic self-energy must have. 0 when the window is not symmetric.
  double herm_data   = 0.0;
};

/**
 * The linearized quasiparticle problem at one (spin, k), solved (step 2): the output of
 * linearized_qp_matrix(), which knows only K and B, not where they came from.
 *
 * Deliberately minimal. Three matrices are kept because each is needed downstream and none
 * is cheap to recover from the others' *consumers*:
 *   Hqp  is the product the self-consistent loop and the checkpoint post-processing use;
 *   Z    is what the ladder's error estimate is built on (||Z(N) - Z(N-1)||_2 between rungs)
 *        and what the per-level sensitivity needs through V^dag Z V;
 *   V    is the eigenbasis both of those and the pole weights refer to.
 * Not kept, because they are derivable in one line from what is: K = Z^-1/2 Hqp Z^-1/2,
 * B = 1 - Z^-1, and Z^1/2 (a construction intermediate). Zqp alone would not do: it is a
 * projection of Z and carries no information about the basis or the ladder step.
 */
struct qp_matrix_t {
  // Eigenvalues of H_QP, ascending, measured from mu [Ha]: the quasiparticle energies.
  nda::array<double, 1>      E;
  // Eigenvectors of H_QP as columns, in the basis of K and B.
  nda::array<ComplexType, 2> V;
  // Z = (1 - B)^-1, the renormalization matrix.
  nda::array<ComplexType, 2> Z;
  // H_QP = Z^1/2 K Z^1/2, Hermitized.
  nda::array<ComplexType, 2> Hqp;
  // Pole weights <v_l|Z|v_l>: the diagonal of Z in the H_QP eigenbasis, one per level.
  // Inside an exactly degenerate group of levels the individual values depend on which
  // orthonormal basis of the group the eigensolver returned; only their sum is invariant. For a
  // symmetry multiplet carrying a single irrep Z is proportional to 1 on the group (Schur), so
  // the ambiguity is void there. Resolving a genuinely ambiguous group would need irrep labels;
  // no tolerance-based blocking is attempted -- it cannot tell an accidental near-degeneracy
  // from a multiplet, and would trade one ambiguity for another (decision of 2026-09-19).
  nda::array<double, 1>      Zqp;
  // Smallest eigenvalue of 1 - B. For a causal Sigma the slope is negative semidefinite, so
  // this is >= 1; a value <= 0 makes the problem ill-defined (see valid).
  double min_eig  = 0.0;
  // sum(Zqp) - Tr Z. Exact identity (V is unitary), so a free consistency check: ~0.
  double sumrule  = 0.0;
  // False when 1 - B is not positive definite: then only min_eig is meaningful and the other
  // members are left empty. The kernel reports this rather than aborting so a caller (or a
  // unit test) can decide what to do.
  bool   valid    = true;
};

/**
 * Steps 1 + 2 at one (spin, k): what try_solve_point() / solve_point() return.
 */
struct point_result_t {
  // Hermitized zero-frequency value A = Sigma(0) and slope B = dSigma/d(iw)|_0 of the fit
  // (step 1), both (n, n) [Ha and dimensionless]. Always filled, even when qp is not.
  nda::array<ComplexType, 2> A, B;
  // Quality of the fit at this point.
  fit_diagnostics_t diagnostics;
  // The solved matrix problem (step 2). Not computed when the fit fails the residual gate
  // (try_solve_point status 1); carries valid == false when 1 - B is indefinite (status 2).
  qp_matrix_t qp;
};

/**
 * What a full-array solve returns: qp_matrix_t for every (spin, k), assembled into
 * (ns, nk, ...) arrays, plus the per-point fit diagnostics and a per-point status. This is
 * the single result type of every consumer, self-consistent or one-shot, C++ or Python;
 * produced by linearized_qp_solve().
 *
 * Three (ns, nk, n, n) matrices per point, for the reasons given at qp_matrix_t. The fit's
 * own outputs K = F + A - mu and B are not stored: the static-versus-dynamic analyses that
 * want them rebuild them from Z and Hqp (K = Z^-1/2 Hqp Z^-1/2, B = 1 - Z^-1); this struct is
 * also what Python receives, wrapped as post_proc_module.ResultT with these member names.
 *
 * Entries of a point whose status is nonzero are zero (the diagnostics are still filled for
 * status 1, plus min_eig for status 2). With on_failure_e::abort the driver never returns
 * such a point, so status_sk is then identically zero.
 */
struct result_t {
  // Z = (1 - B)^-1, H_QP = Z^1/2 K Z^1/2 and the H_QP eigenvectors per point, (ns, nk, n, n).
  nda::array<ComplexType, 4> Z_skab, Hqp_skab, V_skab;
  // Quasiparticle energies measured from mu [Ha] and pole weights <v|Z|v>, (ns, nk, n).
  nda::array<double, 3>      E_ska, Zqp_ska;
  // Per-point diagnostics, (ns, nk): smallest eigenvalue of 1 - B, the four members of
  // fit_diagnostics_t, and the sum rule sum(Zqp) - Tr Z (an exact identity, so ~0).
  nda::array<double, 2>      min_eig_sk, resid_sk, anti_herm_A_sk, anti_herm_B_sk, herm_data_sk,
                             sumrule_sk;
  // Per-point outcome, (ns, nk): 0 ok, 1 the exactly-determined fit failed the residual gate,
  // 2 = 1 - B not positive definite. Identically zero when the driver aborts on failure.
  nda::array<long, 2>        status_sk;
  // The chemical potential that was subtracted from K = F + A - mu [Ha]; E_ska is measured from it.
  double mu = 0.0;
  // The window actually used: number of positive nodes and the polynomial order.
  int    n_fit = 0, fit_order = 0;
  // Condition number of the design matrix (one number: the fit operator is shared by all points).
  double cond = 0.0;
};

/** What linearized_qp_solve() does when a point fails the residual gate or the causality check. */
enum class on_failure_e {
  abort,   // stop the program naming the (spin, k) and the reason: a result must be trustworthy
           // everywhere to be used at all (the LQSGW loop, checkpoint post-processing)
  report   // leave the point zero, record why in result_t::status_sk and carry on: for an
           // analysis over many k-points where one bad point must not hide the rest
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
 * <v|Z|v> (see qp_matrix_t::Zqp for what they mean at exact degeneracies).
 * @param K       - [INPUT] static one-body matrix F + Sigma(0) - mu, (n, n); Hermitized on entry
 * @param B       - [INPUT] slope dSigma/d(i w) at w = 0, (n, n); Hermitized on entry
 * @return the solved problem; on success E, V, Z, Hqp, Zqp and sumrule are set
 * @note If 1 - B is not positive definite the result carries valid == false and only min_eig;
 *       the kernel does not abort, because utils::check terminates the process and a unit test
 *       could not observe it. Callers decide how to fail (solve_point aborts, try_solve_point
 *       reports status 2). For a causal Sigma the slope is negative semidefinite, so
 *       min_eig >= 1 and Z has spectrum in (0, 1]: a violation is a failed fit or a sign error,
 *       never physics, and must not be regularized away.
 */
qp_matrix_t linearized_qp_matrix(nda::array_const_view<ComplexType, 2> K,
                                 nda::array_const_view<ComplexType, 2> B);

/**
 * Steps 1 + 2 at one (spin, k) starting from Sigma(tau): transform to the Matsubara axis, select
 * the window, fit, and solve the linearized problem. Reports failures through a status code
 * instead of aborting, for callers that want to keep going (on_failure_e::report).
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
 * The full-array driver: try_solve_point() for every (spin, k), distributed round-robin over
 * comm and all-reduced, so every rank returns the complete result_t. This is the one loop
 * behind every consumer; the overload below runs the same loop serially.
 * @param comm        - [INPUT] communicator the (spin, k) work is spread over
 * @param F_skab      - [INPUT] static one-body matrices, (ns, nk, n, n), **including H0**
 * @param Sigma_tskab - [INPUT] dynamic self-energy, (nt, ns, nk, n, n), on ft's fermionic tau mesh
 * @param mu          - [INPUT] chemical potential [Ha]
 * @param ft          - [INPUT] imaginary-axis transform driver
 * @param p           - [INPUT] fit controls; the fit operator is built internally
 * @param on_failure  - [INPUT] abort (default) or report a point that fails the residual gate or
 *                      the causality check; see on_failure_e
 * @return Z, H_QP, V, E and the pole weights per point plus the diagnostics (result_t); with
 *         on_failure_e::abort a returned result is usable at every point, with
 *         on_failure_e::report consult status_sk first
 * @note Aborts regardless of on_failure on inconsistent shapes and on a window that is not on
 *       ft's sampling mesh (make_fit_operator); those are programming errors, not outcomes.
 */
result_t linearized_qp_solve(boost::mpi3::communicator& comm,
                             nda::array_const_view<ComplexType, 4> F_skab,
                             nda::array_const_view<ComplexType, 5> Sigma_tskab,
                             double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                             on_failure_e on_failure = on_failure_e::abort);

/**
 * The same, serial: every (spin, k) on the calling process, no communicator and no reduction.
 * Same loop as above; this is the form for a caller that must not depend on MPI being
 * initialized, such as the Python binding. Parameters as above without comm.
 */
result_t linearized_qp_solve(nda::array_const_view<ComplexType, 4> F_skab,
                             nda::array_const_view<ComplexType, 5> Sigma_tskab,
                             double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                             on_failure_e on_failure = on_failure_e::abort);

/**
 * Largest n_fit whose window +-1, ..., +-(2 n_fit - 1) lies entirely on ft's fermionic sampling
 * mesh. The kernel selects rows of the tau -> iw transform rather than interpolating, so this is
 * a hard ceiling: make_fit_operator() aborts one step beyond it. Both IR and DLR meshes carry the
 * lowest odd frequencies as a contiguous run (12 for DLR, 13 for IR at beta=100, wmax=40, "high").
 * @param ft               - [INPUT] imaginary-axis transform driver
 * @param symmetric_window - [INPUT] whether the negative nodes must be on the mesh as well
 * @return the ceiling (>= 1 for any sensible mesh)
 */
int max_n_fit_on_mesh(imag_axes_ft::IAFT const& ft, bool symmetric_window);

/**
 * Result of linearized_qp_ladder(): the last accepted rung plus what the climb measured.
 * Only `last` is a full solution; everything else is bookkeeping of the convergence rule.
 */
struct ladder_result_t {
  // The last rung the gate accepted: a complete result_t at n_fit = n_fit_history.back().
  result_t last;
  // Truncation-error estimate per (ns, nk): the spectral norm of Z(N) - Z(N-1) between the last
  // two accepted rungs. dZ is Hermitian, so this is the supremum of |<v|dZ|v>| over any v -- the
  // largest error any band's pole weight can carry at that point -- and a unitary invariant, so
  // it needs no rotation into the QP eigenbasis. Zero when only one rung was accepted.
  nda::array<double, 2> err_fit_sk;
  // max |H_QP(N) - H_QP(N-1)| per (ns, nk) over the last two accepted rungs [Ha]: the residual
  // error of the quasiparticle Hamiltonian itself. Zero when only one rung was accepted.
  nda::array<double, 2> dHqp_sk;
  // n_fit and the max fit residual of every accepted rung, ascending (length n_accepted).
  nda::array<long, 1>   n_fit_history;
  nda::array<double, 1> resid_history;
  // Pole weights of every accepted rung, (n_accepted, ns, nk, n): how Zqp converged.
  nda::array<double, 4> Zqp_history;
  // Number of accepted rungs. 0 means even n_fit = 2 failed the gate: the input is not usable and
  // `last` is empty. A caller must check this before touching `last`.
  int    n_accepted = 0;
  // True when the climb ended because the next window would leave the sampling mesh, not because
  // the residual gate fired (n_fit_max was higher than the mesh allows).
  bool   mesh_limited = false;
  // max_n_fit_on_mesh(ft, p.symmetric_window), for reporting.
  int    n_fit_mesh_max = 0;
  // The first rung the gate rejected, with its residual and condition number; 0 when none was.
  int    stopped_n_fit = 0;
  double stopped_resid = 0.0, stopped_cond = 0.0;
};

/**
 * The convergence rule of the fit, from the notes: climb the exactly-determined ladder
 * n_fit = 2, 3, ..., each rung interpolating its 2 n_fit symmetric nodes, and stop at the first
 * rung whose relative residual exceeds p.fit_resid_tol (the polynomial no longer interpolates: the
 * design matrix has lost conditioning) or at the end of the sampling mesh. The last accepted rung
 * is the answer, and the change of Z over the last step is its truncation-error estimate. Every
 * rung is one linearized_qp_solve(); p.n_fit and p.fit_order are overridden rung by rung, the
 * other fit controls are honoured.
 * @param comm        - [INPUT] communicator each rung is distributed over
 * @param F_skab      - [INPUT] static one-body matrices, (ns, nk, n, n), **including H0**
 * @param Sigma_tskab - [INPUT] dynamic self-energy, (nt, ns, nk, n, n), on ft's fermionic tau mesh
 * @param mu          - [INPUT] chemical potential [Ha]
 * @param ft          - [INPUT] imaginary-axis transform driver
 * @param p           - [INPUT] fit controls; fit_resid_tol is the gate, symmetric_window is used
 * @param n_fit_max   - [INPUT] highest rung to try; capped at max_n_fit_on_mesh()
 * @param on_failure  - [INPUT] what a rung does with a point whose 1 - B is not positive definite
 * @return see ladder_result_t; check n_accepted > 0 before using `last`
 */
ladder_result_t linearized_qp_ladder(boost::mpi3::communicator& comm,
                                     nda::array_const_view<ComplexType, 4> F_skab,
                                     nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                     double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                                     int n_fit_max, on_failure_e on_failure = on_failure_e::abort);

/** The same, serial (no communicator, no reduction), for callers that must not depend on MPI. */
ladder_result_t linearized_qp_ladder(nda::array_const_view<ComplexType, 4> F_skab,
                                     nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                     double mu, imag_axes_ft::IAFT const& ft, fit_params_t const& p,
                                     int n_fit_max, on_failure_e on_failure = on_failure_e::abort);

} // namespace methods::lqp

#endif // COQUI_LINEARIZED_QP_HPP
