"""
CT-INT backend of the impurity-solver interface (coqui.dmft.ctint.CtintSolver), checked against
the exactly solvable Hubbard–Holstein dimer.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_ctint", reason="triqs_ctint is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.ctint import CtintSolver
from coqui.dmft.solver_api import ImpurityProblem, make_solver
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem

# Monte Carlo budget (n_cycles is the total over ranks; the chiAB measurement dominates the cost, so
# length_cycle is large and the chi grid short: ~27 s per production solve on one rank, expansion
# order ~25, sign 1) and the tolerances. Measured once on one rank over the seeds 11, 12, 13 at this
# budget (spread = max - min of the three, bias = |three-seed mean - exact|): density spread 2.9e-3,
# bias 4.3e-3; |G(iw0)| spread 7e-4, bias 6e-5; Sigma_dyn(iw0) spread 4e-3, bias 2e-4; Sigma_infty
# spread 6e-5, bias 9e-5; W spread 1.8 %, bias 4.3 %; Pi spread 9 %, bias 25 % at iv0 and 14 % at
# iv1. The Pi bias is the plateau subtraction of chiAB(tau): on this finite system the connected
# correlator has not decayed by beta/2, so chi(iv0) is ~8 % low, and the Dyson step Pi = chi/(U chi - 1)
# amplifies that by 1/(1 - U chi) ~ 2.8 (the density route, chi - beta n n, was 25 % off the other
# way: the measured densities' noise times beta). The tolerances cover three spreads plus the bias;
# Pi's is therefore loose and pins the sign and magnitude only, while D0 = D/2 and the sign of u are
# pinned by Sigma_dyn, Sigma_infty and W.
_SOLVER = {"n_cycles": 20000, "n_warmup_cycles": 500, "length_cycle": 200, "n_tau_chi2": 201,
           "random_seed": 11}
_TOL = {"density": 0.015, "g0_rel": 0.005, "sigma0": 0.015, "sigma_infty": 5e-4, "w_rel": 0.10,
        "pi_rel": 0.6}


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


def test_ctint_solver_solve(dimer):
  """CtintSolver.solve reproduces the dimer's exact solution within the Monte Carlo tolerances.

  Setup: the dimer (beta 20, U 2, one retarded pole), degenerate_blk [[0, 1]], the budget _SOLVER;
  one CtintSolver constructed for comm_size = the MPI size.

  Cases (2 depends on the result of 1 for the comparison only):
    1. Production solve: result shapes pass validate(problem); density, |G(iw0)|, Sigma_dyn(iw0),
       Sigma_infty, W(iv0) and W(iv1) agree with ED within _TOL, Pi(iv0) and Pi(iv1) within the
       loose _TOL["pi_rel"] (see the header). This pins D0 = D/2 (a factor 2 shows in Sigma_dyn,
       Sigma_infty and W through the retarded channel), the sign of u (the alpha direction of the
       D0 vertices: a wrong one gives a sign far below 1) and the Hartree-Fock Sigma_infty with
       the polaron shift. The two spin blocks are equal
       (averaged over the degenerate group). Diagnostics carry average_sign, average_order and
       max_cond; solver_data the four library objects.
    2. Occupations-only solve: no Sigma, Pi None; density within _TOL["density"] of case 1's;
       diagnostics carry average_sign.
    3. A gf_struct with blocks of unequal size is rejected by solve with ValueError naming
       "equal size", before any solver is built (the problem is the dimer's arrays padded to
       three orbitals per spin in blocks of size 1 and 2; only the block check runs).
    4. Construction: h_int "full" raises ValueError naming "density"; a benchmark-runner key this
       version does not implement (alpha_mode) raises ValueError naming it, also when it sits
       in the mu_search table (restart_n_warmup_cycles, a CT-SEG cycle control the search set
       would otherwise forward to triqs_ctint and fail on at the first search solve);
       make_solver({"type": "ctint", ...}) returns a CtintSolver with the cycle controls resolved.
  """
  params, iaft, ref, problem = dimer
  solver = CtintSolver(dict(_SOLVER), comm_size=mpi.size)
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
  assert {"average_sign", "average_order", "max_cond"} <= set(result.diagnostics)
  assert abs(result.diagnostics["average_sign"]) > 0.5
  assert set(result.solver_data) == {"M_tau", "G_iw", "density", "chiAB_tau"}

  density = solver.solve(problem, occupations_only=True, chkpt=None)
  assert density.occupations_only and density.Pi_iw_data is None
  assert abs(density.density - result.density) < _TOL["density"]
  assert "average_sign" in density.diagnostics

  nw_f, nw_b = len(ref["wn_f"]), len(ref["wn_b"])
  unequal = ImpurityProblem(
    gf_struct=[("up_0", 1), ("up_1", 2), ("down_0", 1), ("down_1", 2)], beta=params.beta, iaft=iaft,
    dlr_wmax=20.0, dlr_eps=1e-10, h0_sab=np.zeros((2, 3, 3), dtype=complex),
    delta_wsab=np.zeros((nw_f, 2, 3, 3), dtype=complex), v_abcd=np.zeros((3, 3, 3, 3), dtype=complex),
    u_weiss_wabcd=np.zeros((nw_b, 3, 3, 3, 3), dtype=complex), target_density=1.0,
    occupations_sab=np.zeros((2, 3, 3)), degenerate_blk=None)
  with pytest.raises(ValueError, match="equal size"):
    solver.solve(unequal, occupations_only=False, chkpt=None)

  with pytest.raises(ValueError, match="density"):
    CtintSolver(dict(_SOLVER, h_int="full"), comm_size=1)
  with pytest.raises(ValueError, match="alpha_mode"):
    CtintSolver(dict(_SOLVER, alpha_mode="auto"), comm_size=1)
  with pytest.raises(ValueError, match="restart_n_warmup_cycles"):
    CtintSolver(dict(_SOLVER, mu_search={"restart_n_warmup_cycles": 100}), comm_size=1)
  made = make_solver({"type": "ctint", "n_cycles": 10, "n_warmup_cycles": 1, "length_cycle": 1}, comm_size=1)
  assert isinstance(made, CtintSolver) and made.production_params["n_cycles"] == 10
