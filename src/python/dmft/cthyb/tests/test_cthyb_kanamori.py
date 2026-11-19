"""
CT-HYB backend with a non-density-density interaction (coqui.dmft.cthyb.CthybSolver, h_int
"kanamori" and "full"), checked against the exactly solvable two-orbital Kanamori–Holstein impurity.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_cthyb", reason="triqs_cthyb is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.cthyb import CthybSolver
from coqui.dmft.tests.kanamori_holstein_dimer import (KanamoriParams, KanamoriED, make_iaft, make_problem,
                                                      hartree_fock_sigma_infty, kanamori_tensor)

# Monte Carlo budget (n_cycles is the total over ranks; ~13 s per production solve on 4 ranks,
# expansion order ~40, sign 1) and the tolerances. Measured once on 4 ranks over the seeds 11, 12, 13
# at this budget (spread = max - min of the three, bias = |three-seed mean - exact|): density spread
# 1.8e-2, bias 9e-3; |G(iw0)| spread 3e-4, bias 1e-3 relative; Sigma_dyn(iw0) spread 1.5e-3, bias
# 3e-3; Sigma_infty spread 2.2e-3, bias 1.5e-3. The tolerances cover the spread plus the bias with
# room. The hybridization is deliberately strong (V = 1): at V = 0.5 the hybridization expansion
# samples G(tau) of this Kanamori atom so poorly that G(beta-) disagrees with the solver's own
# density matrix by 3e-2 and drifts between seeds, while the density matrix stays exact. The
# density-density operator on the same problem (case 3) sits 2.2e-2 off in Sigma_dyn(iw0), 1.6e-2 in
# the density and 1.4e-3 in |G(iw0)| by ED; only Sigma_dyn(iw0) is clear of the Monte Carlo spread,
# so it is the quantity that makes case 1 a test of the Kanamori operator and not only of the
# density channel. W and Pi come from the directly measured density correlator (a Kanamori h_loc
# does not commute with the densities, so the kink estimator is unavailable), which is far noisier
# than the kink estimator at the same budget: W within 3-7 % and Pi within 10-34 % of ED over the
# three seeds at iv0 and iv1, against 0.4 % and 1.6 % for the density-density operator on the same
# problem; the W and Pi tolerances pin the sign and magnitude only.
_SOLVER = {"n_cycles": 200000, "n_warmup_cycles": 5000, "length_cycle": 50, "random_seed": 11,
           "n_tau_bosonic": 10001, "dyn_n_l": 100}
_TOL = {"density": 0.03, "g0_rel": 0.005, "sigma0": 0.008, "sigma_infty": 6e-3, "w_rel": 0.10, "pi_rel": 0.40}


@pytest.fixture(scope="module")
def kanamori():
  params = KanamoriParams()
  iaft = make_iaft(params, wmax=20.0, eps=1e-10)
  ref = KanamoriED(params).reference(iaft)
  problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10, degenerate_blk=((0, 1, 2, 3),))
  return params, iaft, ref, problem


def _i0(wn):
  """Position of the lowest positive frequency."""
  return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def _compare(result, ref, i0, tol):
  """The agreement of one production result with the ED reference, every colour and orbital pair."""
  assert abs(result.density - ref["occupations"].sum()) < tol["density"]
  for c in range(4):
    g = result.G_iw_data[c][i0, 0, 0]
    assert abs(abs(g) - abs(ref["G_iw"][i0, c])) / abs(ref["G_iw"][i0, c]) < tol["g0_rel"]
    assert abs(result.Sigma_iw_data[c][i0, 0, 0] - ref["Sigma_dyn_iw"][i0, c]) < tol["sigma0"]
    assert abs(result.Sigma_infty[c][0, 0] - ref["Sigma_infty"][c]) < tol["sigma_infty"]
  for key, exact, rel in (("W_iw_data", ref["W_iw"], tol["w_rel"]), ("Pi_iw_data", ref["Pi_iw"], tol["pi_rel"])):
    got = getattr(result, key)[0]
    for index in (0, 1):
      assert np.abs(got[index] - exact[index]).max() / np.abs(exact[index]).max() < rel, (key, index)


def test_cthyb_kanamori(kanamori):
  """CthybSolver.solve with a Kanamori interaction, given as the full tensor, reproduces the exact
  two-orbital solution; the density-density operator on the same tensor does not.

  Setup: the two-orbital Kanamori–Holstein impurity (beta 20, U 2, U' 0.6, J 0.7, V 1, ~2.7
  electrons, one retarded pole), degenerate_blk [[0, 1, 2, 3]], the budget _SOLVER; CthybSolver
  instances for comm_size = the MPI size.

  Cases (2 and 3 reuse the reference of 1; 4 uses the result of 1):
    1. h_int "kanamori": validate(problem); density, |G(iw0)|, Sigma_dyn(iw0) and Sigma_infty of
       every colour within _TOL of ED (see the header); the four colours equal (degenerate_blk);
       the orbital (2, 2) matrices W and Pi at iv0 and iv1 within _TOL of the ED's density-density
       Dyson step (the density correlator is measured directly, since the densities do not commute
       with a Kanamori h_loc; Pi carries the plateau bias of the connected correlator at iv0).
    2. h_int "full" with the same tensor: the same checks. The tensor holds only Kanamori elements,
       so this pins that the Slater path builds the same operator as the Kanamori reduction, which
       problem_to_triqs confirms exactly (the two operators are identical).
    3. h_int "density" on the same problem: Sigma_dyn(iw0) misses the Kanamori ED by more than the
       tolerance (the negative control: the spin-flip and pair-hopping terms change the dynamics
       beyond the Monte Carlo spread; the density and Sigma_infty move by less than the spread and
       are not compared).
    4. Sigma_infty of case 1 equals the Hartree-Fock contraction of the full tensor with the
       solver's own occupations (hartree_fock_sigma_infty, an independent Wick implementation),
       averaged over the degenerate group as the backend averages its Sigma_infty, plus the polaron
       shift u(0) N, to 1e-10: the package's sigma_infty_hf handles the spin-flip and pair-hopping
       monomials, which a density-density test cannot see.
  """
  params, iaft, ref, problem = kanamori
  i0 = _i0(ref["wn_f"])
  result = CthybSolver(dict(_SOLVER, h_int="kanamori"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  result.validate(problem)
  _compare(result, ref, i0, _TOL)
  for c in range(1, 4):
    assert np.allclose(result.Sigma_iw_data[c], result.Sigma_iw_data[0])

  full = CthybSolver(dict(_SOLVER, h_int="full"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  full.validate(problem)
  _compare(full, ref, i0, _TOL)
  from coqui.dmft.triqs_utils import problem_to_triqs
  op_full = problem_to_triqs(problem, h_int_kind="full", screen_j=False, real_hamiltonian=True).h_int
  op_kan = problem_to_triqs(problem, h_int_kind="kanamori", screen_j=False, real_hamiltonian=True).h_int
  assert len(list(op_full - op_kan)) == 0

  density = CthybSolver(dict(_SOLVER, h_int="density"), comm_size=mpi.size).solve(
    problem, occupations_only=False, chkpt=None)
  assert abs(density.Sigma_iw_data[0][i0, 0, 0] - ref["Sigma_dyn_iw"][i0, 0]) > _TOL["sigma0"]

  rho = np.diag(np.concatenate([result.orbital_occupations[name] for name, _ in problem.gf_struct]))
  u0 = -2.0 * params.g**2 / params.omega0
  hf_c = np.diag(hartree_fock_sigma_infty(kanamori_tensor(params.U, params.Uprime, params.J), rho))
  expected = hf_c.mean() + u0 * result.density                 # one degenerate group of all four colours
  got = np.array([result.Sigma_infty[c][0, 0].real for c in range(4)])
  assert np.abs(got - expected).max() < 1e-10
