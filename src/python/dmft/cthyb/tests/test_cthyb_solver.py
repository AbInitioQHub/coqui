"""
CT-HYB backend of the impurity-solver interface (coqui.dmft.cthyb.CthybSolver), checked against the
exactly solvable Hubbard–Holstein dimer.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_cthyb", reason="triqs_cthyb is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from triqs.gfs import MeshDLRImFreq
from coqui.dmft.cthyb import CthybSolver
from coqui.dmft.solver_api import ImpurityProblem, make_solver
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem

# Monte Carlo budget (n_cycles is the total over ranks; ~29 s per production solve on one rank,
# expansion order ~9, sign 1, Lang-Firsov kernel error 1.4e-7 Ha at dyn_n_l 100 on 10001 points) and
# the tolerances. Measured once on one rank over the seeds 11, 12, 13 at this budget (spread = max -
# min of the three, bias = |three-seed mean - exact|): density spread 1.7e-3, bias 8e-4; |G(iw0)|
# spread 6e-4, bias 6e-5; Sigma_dyn(iw0) spread 1.2e-3, bias 2e-4; Sigma_infty spread 4e-5, bias
# 2e-5; W spread < 0.1 %, bias 1.8 % at iv0 and 0.02 % at iv1; Pi spread 0.3 %, bias 11 % at iv0 and
# 0.1 % at iv1. The Pi(iv0) bias is the plateau subtraction of Q(tau): on this finite system the
# connected correlator has not decayed by beta/2, and the constant enters chi(iv0) alone, amplified by
# 1/(1 - U chi) in the Dyson step (as for CT-INT); every other quantity is at the 1e-3 level. The
# tolerances cover three spreads plus the bias; the occupations-only solve runs at the search budget
# (5 % of the cycles, resolve_mu_search_params), so its density is compared at sqrt(20) times the
# production spread, rounded up.
_SOLVER = {"n_cycles": 200000, "n_warmup_cycles": 2000, "length_cycle": 50, "random_seed": 11,
           "n_tau_bosonic": 10001, "dyn_n_l": 100}
_TOL = {"density": 0.006, "density_search": 0.03, "g0_rel": 0.005, "sigma0": 0.005, "sigma_infty": 2e-4,
        "w_rel": 0.03, "pi_rel": 0.15}


@pytest.fixture(scope="module")
def dimer():
  params = DimerParams()
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ref = DimerED(params).reference(iaft)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1),))
  return params, iaft, ref, problem


def _i0(wn):
  """Position of the lowest positive frequency."""
  return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def test_cthyb_solver_solve(dimer):
  """CthybSolver.solve reproduces the dimer's exact solution within the Monte Carlo tolerances.

  Setup: the dimer (beta 20, U 2, one retarded pole), degenerate_blk [[0, 1]], the budget _SOLVER;
  CthybSolver instances for comm_size = the MPI size.

  Cases (2 compares with 1):
    1. Production solve, h_int "density": validate(problem); density, |G(iw0)|, Sigma_dyn(iw0),
       Sigma_infty, W(iv0), W(iv1) within _TOL, Pi within _TOL["pi_rel"] (set by the plateau bias at
       iv0, see the header); the two spin
       blocks equal; the resolved n_iw follows the DLR mesh and the fit window,
       max(1.4 x largest DLR index, index of fit_max_w + 1, largest DLR index + 1) (reported in
       diagnostics["n_iw"]); the Q_tau(0) equal-time check below 1e-2 (diagnostics["q0_mismatch"]);
       the Lang-Firsov kernel error below 1e-5 Ha (1.5e-7 at dyn_n_l >= 20 on the dimer's single
       boson, 7e-3 at 10: the kernel the Monte Carlo saw is converged, which Sigma(iw0) alone
       cannot tell on this problem);
       the fork's own kernel K_n equals the Python mirror's Legendre coefficients to 1e-8
       (diagnostics["kernel_mirror_mismatch"]: the kernel error reported describes the kernel the
       Monte Carlo used); diagnostics carry average_sign, average_order, auto_corr_time,
       kernel_error, n_stochastic_dyn_vertex_types, max_cond; solver_data is {G_tau, Sigma_iw, density_matrix}
       (no Q_tau unless store_q_tau).
    2. Occupations-only solve (the search budget, 5 % of the cycles): no Sigma, Pi None; density
       within _TOL["density_search"] of case 1's.
    3. h_int "kanamori" (one orbital: the same density-density operator): density, Sigma_dyn(iw0),
       Pi(iv0) and W(iv0) within _TOL of ED, as in case 1. n_stochastic_dyn_vertex_types is
       reported, not asserted (the fork decides per operator whether the Lang-Firsov route
       applies).
    4. Construction: h_int "slater" raises ValueError naming h_int; reuse_config raises naming it;
       mu_search = {restart_n_warmup_cycles: 10} raises naming it; the former keys lang_firsov and
       n_l raise naming them (the Lang-Firsov route is always used, G_l is never measured); make_solver({"type": "cthyb", ...})
       returns a CthybSolver with the cycle controls resolved.
    5. A two-orbital Kanamori h_int (U 2, U' 1.5, J 0.25: spin-flip and pair-hopping terms that
       are not density-density) on a two-orbital copy of the dimer's bath, at a tiny budget: the
       solve completes with the density correlator measured directly (the densities do not commute
       with h_loc, so the kink estimator cannot give it), Pi_iw_data and W_iw_data are finite
       (2, 2) arrays per bosonic point, the result validates; with n_iw = 100 below the DLR range
       the resolved n_iw is raised to the largest DLR index + 1 (the only case where that branch is
       taken: case 1's 1025 covers it).
  """
  params, iaft, ref, problem = dimer
  solver = CthybSolver(dict(_SOLVER), comm_size=mpi.size)
  result = solver.solve(problem, occupations_only=False, chkpt=None)
  result.validate(problem)
  assert abs(result.density - ref["occupations"].sum()) < _TOL["density"]
  i0 = _i0(ref["wn_f"])
  for spin, name in enumerate(("up_0", "down_0")):
    g = result.G_iw_data[spin][i0, 0, 0]
    assert abs(abs(g) - abs(ref["G_iw"][i0, spin])) / abs(ref["G_iw"][i0, spin]) < _TOL["g0_rel"]
    assert abs(result.Sigma_iw_data[spin][i0, 0, 0] - ref["Sigma_dyn_iw"][i0, spin]) < _TOL["sigma0"]
    assert abs(result.Sigma_infty[spin][0, 0] - ref["Sigma_infty"][spin]) < _TOL["sigma_infty"]
  assert np.allclose(result.Sigma_iw_data[0], result.Sigma_iw_data[1])
  for key, exact, tol in (("Pi_iw_data", ref["Pi_iw"], _TOL["pi_rel"]), ("W_iw_data", ref["W_iw"], _TOL["w_rel"])):
    got = getattr(result, key)[0][:, 0, 0]
    assert abs(got[0] - exact[0]) / abs(exact[0]) < tol, key
    assert abs(got[1] - exact[1]) / abs(exact[1]) < tol, key
  mesh_f = MeshDLRImFreq(beta=params.beta, statistic="Fermion", w_max=20.0, eps=1e-10, symmetrize=True)
  dlr_max = int(np.abs([w.index for w in mesh_f]).max())
  n_iw_fit = int(0.5 * (10.0 * params.beta / np.pi - 1.0)) + 1
  assert result.diagnostics["n_iw"] == max(int(1.4 * dlr_max), n_iw_fit, dlr_max + 1)
  assert result.diagnostics["q0_mismatch"] < 1e-2
  assert result.diagnostics["kernel_error"] < 1e-5
  assert result.diagnostics["kernel_mirror_mismatch"] < 1e-8
  assert {"average_sign", "average_order", "auto_corr_time", "kernel_error", "n_stochastic_dyn_vertex_types",
          "max_cond"} <= set(result.diagnostics)
  assert set(result.solver_data) == {"G_tau", "Sigma_iw", "density_matrix"}

  density = solver.solve(problem, occupations_only=True, chkpt=None)
  assert density.occupations_only and density.Pi_iw_data is None
  assert abs(density.density - result.density) < _TOL["density_search"]

  kanamori = CthybSolver(dict(_SOLVER, h_int="kanamori"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  assert abs(kanamori.density - ref["occupations"].sum()) < _TOL["density"]
  assert abs(kanamori.Sigma_iw_data[0][i0, 0, 0] - ref["Sigma_dyn_iw"][i0, 0]) < _TOL["sigma0"]
  for key, exact, tol in (("Pi_iw_data", ref["Pi_iw"], _TOL["pi_rel"]), ("W_iw_data", ref["W_iw"], _TOL["w_rel"])):
    got = getattr(kanamori, key)[0][:, 0, 0]
    assert abs(got[0] - exact[0]) / abs(exact[0]) < tol, key

  with pytest.raises(ValueError, match="h_int"):
    CthybSolver(dict(_SOLVER, h_int="slater"), comm_size=1)
  with pytest.raises(ValueError, match="reuse_config"):
    CthybSolver(dict(_SOLVER, reuse_config=True), comm_size=1)
  with pytest.raises(ValueError, match="restart_n_warmup_cycles"):
    CthybSolver(dict(_SOLVER, mu_search={"restart_n_warmup_cycles": 10}), comm_size=1)
  with pytest.raises(ValueError, match="lang_firsov"):
    CthybSolver(dict(_SOLVER, lang_firsov=False), comm_size=1)
  with pytest.raises(ValueError, match="n_l"):
    CthybSolver(dict(_SOLVER, n_l=30), comm_size=1)
  made = make_solver({"type": "cthyb", "n_cycles": 10, "n_warmup_cycles": 1, "length_cycle": 1}, comm_size=1)
  assert isinstance(made, CthybSolver) and made.production_params["n_cycles"] == 10

  nw_f, nw_b = len(ref["wn_f"]), len(ref["wn_b"])
  v = np.zeros((2, 2, 2, 2), dtype=complex)
  U, Up, J = 2.0, 1.5, 0.25
  for i in range(2):
    v[i, i, i, i] = U
    for j in range(2):
      if i != j:
        v[i, i, j, j] = Up                 # density-density, opposite orbitals
        v[i, j, j, i] = J                  # exchange
        v[i, j, i, j] = J                  # pair hopping
  u = np.zeros((nw_b, 2, 2, 2, 2), dtype=complex)
  for i in range(2):
    for j in range(2):
      u[:, i, i, j, j] = problem.u_weiss_wabcd[:, 0, 0, 0, 0]
  delta = np.zeros((nw_f, 2, 2, 2), dtype=complex)
  for i in range(2):
    delta[:, :, i, i] = problem.delta_wsab[:, :, 0, 0]
  two = ImpurityProblem(
    gf_struct=[("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)], beta=params.beta, iaft=iaft,
    dlr_wmax=20.0, dlr_eps=1e-10, h0_sab=np.array([np.eye(2) * params.eps] * 2, dtype=complex),
    delta_wsab=delta, v_abcd=v, u_weiss_wabcd=u, target_density=2.0,
    occupations_sab=np.array([np.eye(2) * 0.5] * 2), degenerate_blk=None)
  small = {"n_cycles": 400, "n_warmup_cycles": 50, "length_cycle": 20, "random_seed": 3, "h_int": "kanamori",
           "n_tau_bosonic": 2001, "dyn_n_l": 50, "n_tau": 2001, "n_iw": 100}
  kan2 = CthybSolver(small, comm_size=mpi.size).solve(two, occupations_only=False, chkpt=None)
  kan2.validate(two)
  assert kan2.Pi_iw_data[0].shape == (nw_b, 2, 2) and np.isfinite(kan2.Pi_iw_data[0]).all()
  assert kan2.W_iw_data[0].shape == (nw_b, 2, 2) and np.isfinite(kan2.W_iw_data[0]).all()
  assert dlr_max + 1 > 100 and kan2.diagnostics["n_iw"] == dlr_max + 1
