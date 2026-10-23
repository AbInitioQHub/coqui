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
#include "utilities/mpi_context.h"
#include "numerics/shared_array/nda.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/qp/qp_context_t.h"

/**
 * Linearized quasiparticle approximation for a dynamic self-energy.
 *
 *   Sigma(iw) ~ A + iw B,   K = F + A - mu,   Z = (1 - B)^-1,
 *   H_QP = Z^1/2 K Z^1/2 + mu,   H_QP v = eps v,   Z_qp = <v|Z|v>.
 *
 * The module applies to any orthonormal Bloch basis.
 *
 * This is a different approximation from qp_eqn.solver == "linearized" in qp_solvers.hpp, which
 * Taylor-expands the *diagonal* quasiparticle equation around the KS energy on the real
 * axis. Here the full matrix Sigma(iw) is expanded around w = 0 on the Matsubara axis.
 *
 * Chemical potential is added to the resulting quasiparticle energies.
 *
 * Typical use at fixed (spin, k), with F including H0:
 *   auto op = make_fit_operator(ft, params);              // once per (ft, params)
 *   auto results = linearized_qp_kpoint(F_ab, Sigma_tab, mu, ft, op, params);
 * or over full arrays in one call: linearized_qp_solve(mpi, F_skab, Sigma_tskab, mu, ft, params).
 *
 * The result types form one hierarchy, from a single (spin, k) up to full arrays:
 *   fit_operator_t     what depends only on (IAFT, params_t): window rows, design matrix
 *   result_kpoint_t    the fit (step 1), the matrix problem (step 2) and the status at one
 *                      k-point; produced by linearized_qp_kpoint()
 *   shared_result_t    the same members for every (spin, k), as (ns, nk, ...) arrays; the
 *                      (ns, nk, n, n) matrices in memory shared within each node, the smaller
 *                      arrays held by every rank; produced by linearized_qp_solve()
 *   result_t           an owned copy of shared_result_t (to_owned()), wrapped for Python
 */
namespace methods::lqp {

/**
 * Fit operator which is data-independent and can be reused for every (spin, k) 
 * with the same (IAFT, params_t).
 */
struct fit_operator_t {
  // P, the highest power of (i w) actually used (params_t::fit_order resolved from -1).
  int    fit_order   = 0;
  // Position of the fit nodes n_j in the IAFT fermionic mesh.
  nda::array<long, 1> w_sel;
  // Scaled design matrix, (nnodes, P+1): M_jp = (i x_j)^p with x_j = n_j / n_max in [-1, 1].
  nda::array<ComplexType, 2> M;
  // n_max * pi / beta [Ha]. Unscaling: the physical coefficient is A^(p) = c_p / omega_scale^p.
  double omega_scale = 0.0;
  // Condition number of M. Depends only on the window and the order (not on the IAFT or the
  // data), and grows with the window. Quoted in the abort message when the residual gate fires.
  double cond        = 0.0;
};

/**
 * The linearized quasiparticle problem at one (spin, k): how well the fit went (step 1), the
 * solved matrix problem (step 2) and the outcome. 
 * result_t holds the same members for every (spin, k), each with the (s, k) index suffix
 */
struct result_kpoint_t {
  // --- step 1, the fit; all dimensionless, all "smaller is better". Always filled.
  // Relative residual of the polynomial fit, max |M c - S| / max |S| over all matrix elements.
  // For an exact fit this should be at round-off; larger means the design matrix has lost
  // conditioning and the coefficients are unreliable (the residual gate acts on this).
  double resid       = 0.0;
  // |anti-Hermitian part of A| / |A| before Hermitization; with a symmetric window this is
  // round-off by construction, so anything larger points at the input (see anti_herm_Sigma).
  double anti_herm_A     = 0.0;
  // The same for the slope B.
  double anti_herm_B     = 0.0;
  // The same for the input: max |S(iw) - S(-iw)^dag| / max |S| over the +-w pairs of the window,
  // how far Sigma itself is from the Hermiticity a fermionic self-energy must have. 0 when the
  // window is not symmetric.
  double anti_herm_Sigma = 0.0;

  // --- step 2, the matrix problem (solve_linearized_dyson()). Empty unless status == 0, except
  // min_eig_Zinv, which is also set for status 2.
  // Eigenvalues of H_QP, ascending [Ha]: the quasiparticle energies. From
  // solve_linearized_dyson() they are those of Z^1/2 K Z^1/2 (relative to the mu inside K);
  // linearized_qp_kpoint() shifts them and Hqp by +mu, so every result a consumer sees is
  // absolute.
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
  // from a multiplet, and would trade one ambiguity for another.
  nda::array<double, 1>      Zqp;
  // Smallest eigenvalue of Z^-1 = 1 - B (not of H_QP): the causality diagnostic, >= 1 for a
  // causal Sigma (B negative semidefinite); <= 0 gives status 2.
  double min_eig_Zinv  = 0.0;
  // sum(Zqp) - Tr Z. Exact identity (V is unitary), so a free consistency check: ~0.
  double sumrule  = 0.0;

  // --- outcome: 0 ok; 1 the exactly-determined fit failed the residual gate (step 2 not run);
  // 2 = 1 - B is not positive definite. The kernel reports rather than aborts, so a caller
  // (or a unit test) can decide what to do.
  long   status   = 0;
};

/**
 * k-dependent result of the linearized quasiparticle problem: 
 * 
 * Each array is owned by every rank of the node. This is used for the Python interface and
 * for small problems where the arrays fit in memory.
 * 
 * For large problems, use shared_result_t, in which the (ns, nk, n, n) matrices are allocated
 * in shared memory.
 * 
 * Three (ns, nk, n, n) matrices per point. The fit's own outputs K = F + A - mu and B are not
 * stored: the static-versus-dynamic analyses that want them rebuild them from Z, Hqp and mu
 * (K = Z^-1/2 (Hqp - mu) Z^-1/2, since Hqp is absolute; B = 1 - Z^-1); this struct is also what
 * Python receives, wrapped as post_proc_module.ResultT with these member names.
 *
 * Entries of a point whose status is nonzero are zero (the diagnostics are still filled for
 * status 1, plus min_eig_Zinv for status 2). With on_failure_e::abort the driver never returns
 * such a point, so status_sk is then identically zero.
 */
struct result_t {
  // Z = (1 - B)^-1, H_QP = Z^1/2 K Z^1/2 and the H_QP eigenvectors per point, (ns, nk, n, n).
  nda::array<ComplexType, 4> Z_skab, Hqp_skab, V_skab;
  // Absolute quasiparticle energies [Ha] and pole weights <v|Z|v>, (ns, nk, n).
  nda::array<double, 3>      E_ska, Zqp_ska;
  // Per-point diagnostics, (ns, nk): smallest eigenvalue of Z^-1 = 1 - B, the four fit diagnostics
  // of result_kpoint_t, and the sum rule sum(Zqp) - Tr Z (an exact identity, so ~0).
  nda::array<double, 2>      min_eig_Zinv_sk, resid_sk, anti_herm_A_sk, anti_herm_B_sk,
                             anti_herm_Sigma_sk, sumrule_sk;
  // Per-point outcome, (ns, nk): 0 ok, 1 the exactly-determined fit failed the residual gate,
  // 2 = 1 - B not positive definite. Identically zero when the driver aborts on failure.
  nda::array<long, 2>        status_sk;
  // The chemical potential the linearization was made about (K = F + A - mu) [Ha]; subtract it
  // from E_ska for energies relative to the Fermi level.
  double mu = 0.0;
  // The window actually used: number of positive nodes and the polynomial order.
  int    n_fit = 0, fit_order = 0;
};

/**
 * k-dependent result_t with its matrices in node-shared memory: the same members with the same
 * names. The three (ns, nk, n, n) matrices are math::shm::shared_arrays, one copy per node; the
 * smaller (ns, nk, n) and (ns, nk) arrays are ordinary arrays held by every rank.
 *
 * to_owned() copies it into a result_t.
 *
 * The shared arrays refer to the communicators of the mpi_context_t they were built with (by pointer,
 * as every shared_array does), so that context must outlive the result. 
 */
struct shared_result_t {
  template<typename T, int R> using shm_t = math::shm::shared_array<nda::array_view<T, R>>;
  shm_t<ComplexType, 4> Z_skab, Hqp_skab, V_skab;
  nda::array<double, 3> E_ska, Zqp_ska;
  nda::array<double, 2> min_eig_Zinv_sk, resid_sk, anti_herm_A_sk, anti_herm_B_sk,
                        anti_herm_Sigma_sk, sumrule_sk;
  nda::array<long, 2>   status_sk;
  double mu = 0.0;
  int    n_fit = 0, fit_order = 0;

  /**
   * Constructor: allocates the three matrices in mpi's node-shared memory (collective over
   * mpi.comm) and the other arrays on every rank, all zero-initialized. A shared_array has no
   * default state, so the members cannot be built one by one after construction.
   * @param mpi  - [INPUT] MPI context the arrays are shared within; must outlive the result
   * @param ns   - [INPUT] number of spins
   * @param nk   - [INPUT] number of k-points
   * @param norb - [INPUT] number of orbitals
   */
  shared_result_t(utils::mpi_context_t<boost::mpi3::communicator>& mpi, 
                  long ns, long nk, long norb);
};

/** Owned copy of a node-shared result, for callers that hand it to Python (small arrays). */
result_t to_owned(shared_result_t const& shared);

/** What linearized_qp_solve() does when a point fails the residual gate or the causality check. */
enum class on_failure_e {
  abort,   // stop the program naming the (spin, k) and the reason: a result must be trustworthy
           // everywhere to be used at all (the LQSGW loop, checkpoint post-processing)
  report   // leave the point zero, record why in result_t::status_sk and carry on: for an
           // analysis over many k-points where one bad point must not hide the rest
};

/**
 * Matsubara indices of the fit window.
 * @param n_fit            - [INPUT] number of lowest positive fermionic frequencies (>= 1)
 * @param symmetric_window - [INPUT] if true return the 2*n_fit indices
 *                           -(2*n_fit - 1), ..., -1, 1, ..., 2*n_fit - 1 (ascending);
 *                           if false only the n_fit positive ones
 * @return the window indices, shape (2*n_fit) or (n_fit)
 */
nda::array<long, 1> fit_window(int n_fit, bool symmetric_window);

/**
 * Make fitting operator. 
 * @param ft     - [INPUT] imaginary-axis transform driver
 * @param params - [INPUT] fit control parameters
 * @return the fit operator, reusable for every (spin, k) with this (ft, params)
 * @note Aborts (utils::check) if a window node is not on ft's fermionic sampling mesh, naming
 *       the largest admissible n_fit, and if the fit is under-determined (nnodes < fit_order + 1).
 */
fit_operator_t make_fit_operator(imag_axes_ft::IAFT const& ft, params_t const& params);

/**
 * What step 1 returns: the two low-frequency coefficients of Sigma(iw) the linearization uses,
 * and the diagnostics of the fit that produced them (copied into result_kpoint_t by
 * linearized_qp_kpoint()).
 */
struct low_freq_fit_t {
  // A = Sigma(w=0) [Ha] and B = dSigma/d(iw) at w = 0 (dimensionless), (n, n), Hermitized.
  nda::array<ComplexType, 2> A, B;
  // The fit diagnostics of result_kpoint_t: relative residual, anti-Hermiticity of A and B
  // before Hermitization, and of the data on the +-w pairs (0 without symmetric_window).
  double resid = 0.0, anti_herm_A = 0.0, anti_herm_B = 0.0, anti_herm_Sigma = 0.0;
};

/**
 * Step 1 of linearized QP approx. for a given (spin, k): 
 * fit Sigma(iw) matrix on the window to a polynomial in (iw) and return its two lowest
 * coefficients, A = Sigma(0) and B = dSigma/d(iw) at w = 0. The higher orders are fitted (they
 * absorb the curvature of Sigma on the window) but not returned.
 * 
 * Each coefficient is Hermitized after unscaling.
 * @param Sigma_wab - [INPUT] self-energy on the fitting window, (nnodes, n, n)
 * @param op        - [INPUT] fit operator from make_fit_operator(), consistent with params
 * @param params    - [INPUT] fit controls; with symmetric_window the data Hermiticity
 *                    S(-iw) = S(iw)^dag is checked on the +-w pairs
 * @return A, B and the fit diagnostics (low_freq_fit_t)
 */
low_freq_fit_t low_freq_coefficients(nda::array_const_view<ComplexType, 3> Sigma_wab,
                                     fit_operator_t const& op, params_t const& params);

/**
 * Step 2 of linearized QP approx. for a given (spin, k): 
 * From the static matrix K and the slope B, 
 *   - build Z = (1 - B)^-1, 
 *   - the quasiparticle Hamiltonian H_QP = Z^1/2 K Z^1/2, 
 *   - its eigenbasis, and 
 *   - the pole weights <v|Z|v> 
 * 
 * @param K_ab    - [INPUT] static one-body matrix F + Sigma(w=0) - mu, (n, n), where F includes H0.
 * @param B_ab    - [INPUT] slope dSigma/d(i w) at w = 0, (n, n); Hermitized on entry
 * @param results - [OUTPUT] QP results
 * @note If 1 - B is not positive definite the result carries status == 2 and only min_eig_Zinv
 *       is set; the other members are zero.
 */
void solve_linearized_dyson(nda::array_const_view<ComplexType, 2> K_ab,
                            nda::array_const_view<ComplexType, 2> B_ab,
                            result_kpoint_t& results);

/**
 * Steps 1 + 2 at one k-point (one spin, one k) starting from Sigma(tau): transform to the
 * Matsubara axis, select the window, fit, and solve the linearized Dyson equation.
 * @param F_ab       - [INPUT] static one-body matrix, (n, n), **including H0** (the checkpoint's
 *                     F_skij does not contain it; adding it is the caller's responsibility)
 * @param Sigma_tab  - [INPUT] dynamic self-energy on ft's fermionic tau mesh, (nt, n, n);
 *                     must be contiguous (a (t, is, ik, ...) slice of a 5-D array is not: copy
 *                     it, e.g. with nda::make_regular). The IAFT transform aborts otherwise.
 * @param mu         - [INPUT] chemical potential [Ha]; subtracted from the diagonal of K
 * @param ft         - [INPUT] imaginary-axis transform driver
 * @param op         - [INPUT] fit operator from make_fit_operator(), consistent with ft and params
 * @param params     - [INPUT] fit controls; fit_resid_tol is the gate applied here
 * @param on_failure - [INPUT] abort (default) or report a failed k-point; see on_failure_e
 * @return the fit members, the status (0 ok; 1 the exactly-determined fit failed the residual
 *         gate, step 2 not run; 2 = 1 - B not positive definite, only min_eig_Zinv set) and, for
 *         status 0, the solved problem. With on_failure_e::abort the status is always 0.
 * @note With on_failure_e::abort, aborts (utils::check) on a failed residual gate, naming the
 *       residual and the condition number, and on a non-positive-definite 1 - B, naming its
 *       smallest eigenvalue.
 */
result_kpoint_t linearized_qp_kpoint(nda::array_const_view<ComplexType, 2> F_ab,
                                     nda::array_const_view<ComplexType, 3> Sigma_tab,
                                     double mu, imag_axes_ft::IAFT const& ft,
                                     fit_operator_t const& op, params_t const& params,
                                     on_failure_e on_failure = on_failure_e::abort);

/**
 * The full-array driver: linearized_qp_kpoint() for every (spin, k), distributed round-robin over
 * mpi.comm; each rank writes its points into node-shared arrays, which are then summed over
 * nodes, so every rank sees the complete result.
 * @param mpi         - [INPUT] MPI context: the (spin, k) points are distributed over mpi.comm,
 *                      the result lives in memory shared within each node
 * @param F_skab      - [INPUT] static one-body matrices, (ns, nk, n, n), **including H0**
 * @param Sigma_tskab - [INPUT] dynamic self-energy, (nt, ns, nk, n, n), on ft's fermionic tau mesh
 * @param mu          - [INPUT] chemical potential [Ha]
 * @param ft          - [INPUT] imaginary-axis transform driver
 * @param params      - [INPUT] fit controls; the fit operator is built internally
 * @param on_failure  - [INPUT] abort (default) or report a point that fails the residual gate or
 *                      the causality check; see on_failure_e
 * @return Z, H_QP, V, E and the pole weights per point plus the diagnostics, node-shared
 *         (shared_result_t; every rank sees the complete result through .local()); with
 *         on_failure_e::abort a returned result is usable at every point, with
 *         on_failure_e::report consult status_sk first
 * @note Aborts regardless of on_failure on inconsistent shapes and on a window that is not on
 *       ft's sampling mesh (make_fit_operator); those are programming errors, not outcomes.
 */
shared_result_t linearized_qp_solve(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                    nda::array_const_view<ComplexType, 4> F_skab,
                                    nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                    double mu, imag_axes_ft::IAFT const& ft,
                                    params_t const& params,
                                    on_failure_e on_failure = on_failure_e::abort);

/**
 * Return the largest n_fit for a given IAFT kernel whose fermionic sampling mesh spans 
 * the window +-1, ..., +-(2 n_fit - 1) (symmetric_window) or 1, ..., 2 n_fit - 1 (not symmetric).
 * @param ft               - [INPUT] imaginary-axis transform driver
 * @param symmetric_window - [INPUT] whether the negative nodes must be on the mesh as well
 * @return the largest n_fit 
 */
int max_n_fit_on_mesh(imag_axes_ft::IAFT const& ft, bool symmetric_window);

/**
 * Result of linearized_qp_ladder(): the last accepted fit plus what the climb measured.
 * Only `last` is a full solution; everything else is bookkeeping of the convergence rule.
 */
struct ladder_result_t {
  // The last fit the gate accepted: a complete result_t at n_fit = n_accepted + 1 (fits are tried
  // at n_fit = 2, 3, ... and the climb stops at the first rejection).
  result_t last;
  // Truncation-error estimate per (ns, nk): the spectral norm of Z(N) - Z(N-1) between the last
  // two accepted fits. dZ is Hermitian, so this is the supremum of |<v|dZ|v>| over any v -- the
  // largest error any band's pole weight can carry at that point -- and a unitary invariant, so
  // it needs no rotation into the QP eigenbasis. Zero when only one fit was accepted.
  nda::array<double, 2> err_fit_sk;
  // max |H_QP(N) - H_QP(N-1)| per (ns, nk) over the last two accepted fits [Ha]: the residual
  // error of the quasiparticle Hamiltonian itself. Zero when only one fit was accepted.
  nda::array<double, 2> dHqp_sk;
  // Max fit residual over all (s, k) of every accepted fit, at n_fit = 2, ..., n_accepted + 1.
  nda::array<double, 1> resid_history;
  // Number of accepted fits, >= 1: the ladder aborts when even n_fit = 2 is rejected.
  int    n_accepted = 0;
  // Highest n_fit the climb may reach: min(n_fit_max, max_n_fit_on_mesh(ft, symmetric_window)).
  // The mesh set it when n_fit_cap < n_fit_max (the caller's argument).
  int    n_fit_cap = 0;
  // Max fit residual of the fit the gate rejected, at n_fit = n_accepted + 2 (> fit_resid_tol);
  // 0 when the gate never fired. Then the climb either reached n_fit_cap or stopped at a fit
  // with 1 - B not positive definite at some point (n_accepted + 1 < n_fit_cap).
  double stopped_resid = 0.0;
};

/**
 * Result of linearized_qp_ladder(): ladder_result_t with `last` in node-shared memory. Every
 * member has the name and meaning of its ladder_result_t counterpart (documented there); only
 * the types differ as noted below. to_owned() copies it into a ladder_result_t.
 */
struct shared_ladder_result_t {
  // A shared_result_t, whose three (ns, nk, n, n) matrices are node-shared.
  shared_result_t last;
  // The rest: ordinary arrays and scalars, held identically by every rank.
  nda::array<double, 2> err_fit_sk, dHqp_sk;
  nda::array<double, 1> resid_history;
  int    n_accepted = 0;
  int    n_fit_cap = 0;
  double stopped_resid = 0.0;
};

/** Owned copy of a node-shared ladder result, for callers that hand it to Python. */
ladder_result_t to_owned(shared_ladder_result_t const& shared);

/**
 * Perform a linearized QP approx. with systematically increasing n_fit = 2, 3, ... , and 
 * stop when 
 * - the relative residual exceeds fit_resid_tol, i.e. the design matrix become 
 *   ill-conditioned, or
 * - 1 - B is not positive definite at any (s, k) point, or
 * - reach the largest n_fit within the given fermionic frequency mesh.
 * A fit rejected by either of the first two is not kept, so every accepted fit is valid at every
 * point and err_fit_sk / dHqp_sk compare two valid solutions.
 * @param mpi         - [INPUT] MPI context each fit is distributed over
 * @param F_skab      - [INPUT] static one-body matrices, (ns, nk, n, n), **including H0**
 * @param Sigma_tskab - [INPUT] dynamic self-energy, (nt, ns, nk, n, n), on ft's fermionic tau mesh
 * @param mu          - [INPUT] chemical potential [Ha]
 * @param ft          - [INPUT] imaginary-axis transform driver
 * @param n_fit_max        - [INPUT] highest n_fit to try; capped at max_n_fit_on_mesh()
 * @param fit_resid_tol    - [INPUT] the gate: the largest relative fit residual over all (s, k)
 *                           a fit may have to be accepted
 * @param symmetric_window - [INPUT] fit on +-w pairs (see params_t); every fit is exactly
 *                           determined, so this is the only fit control besides the gate
 * @return node-shared; see ladder_result_t for the members
 * @note Aborts (utils::check, on every rank) when no fit is accepted -- even n_fit = 2 fails the
 *       residual gate or has 1 - B not positive definite at some point: the input is not
 *       usable -- and when n_fit_max < 2 or the sampling mesh admits fewer than two n_fit.
 */
shared_ladder_result_t linearized_qp_ladder(utils::mpi_context_t<boost::mpi3::communicator>& mpi,
                                            nda::array_const_view<ComplexType, 4> F_skab,
                                            nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                            double mu, imag_axes_ft::IAFT const& ft,
                                            int n_fit_max, double fit_resid_tol,
                                            bool symmetric_window);

} // namespace methods::lqp

#endif // COQUI_LINEARIZED_QP_HPP
