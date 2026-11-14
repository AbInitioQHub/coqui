"""
CT-SEG backend of the impurity-solver interface (coqui.dmft.ctseg.CtsegSolver), checked against
the exactly solvable Hubbard–Holstein dimer.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import sys
import types

import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_ctseg", reason="triqs_ctseg is required for this backend")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.ctseg import CtsegSolver
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem

# Monte Carlo budget of the tests (n_cycles is the total over ranks) and the tolerances. Measured
# once on one rank over the seeds 11, 12, 13 at this budget (spread = max - min of the three,
# bias = |three-seed mean - exact|): density spread 9e-4, bias 6e-4; |G(iw0)| spread 3e-4, bias
# 1e-4; Sigma_dyn(iw0) spread 8e-4, bias 1e-3; Sigma_infty spread 2e-5, bias 1e-5; Pi(iv0)
# spread 5e-4, bias 8e-3 (0.5 % relative, the largest systematic: the DLR fit of the measured
# nn correlator); W spread 8e-4, bias 1e-3. The tolerances cover three spreads plus the bias.
_SOLVER = {"n_cycles": 200000, "n_warmup_cycles": 2000, "length_cycle": 50, "random_seed": 11,
           "perform_tail_fit": False, "analytic_hf": True}
_TOL = {"density": 0.005, "g0_rel": 0.005, "sigma0": 0.005, "sigma_infty": 0.002, "boson_rel": 0.02}


@pytest.fixture(scope="module")
def dimer():
    params = DimerParams()
    iaft = make_iaft(params)
    ref = DimerED(params).reference(iaft)
    problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10)
    return params, iaft, ref, problem


def _i0(wn):
    """Position of the lowest positive frequency."""
    return int(np.argmin(np.where(wn > 0, wn, np.inf)))


def test_ctseg_solver_solve(dimer):
    """CtsegSolver.solve reproduces the dimer's exact solution within the Monte Carlo tolerances.

    Setup: the dimer (beta 20, U 2, one retarded pole), degenerate_blk [[0, 1]], the budget
    _SOLVER; one CtsegSolver constructed for comm_size = the MPI size.

    Cases (2 depends on the result of 1 for the comparison only):
      1. Production solve: result shapes pass validate(problem); density, |G(iw0)|,
         Sigma_dyn(iw0), Sigma_infty, Pi(iv0), Pi(iv1), W(iv0) and W(iv1) agree with ED within
         _TOL (chi itself is not returned; Pi and W are its two Dyson images). Sigma_infty is
         the analytic [U + D(0)] n of the measured densities, so it tests the "D diagonal kept"
         rule against the exact polaron shift. The two spin blocks are equal (symmetrized).
      2. Density-only solve: no Sigma; occupations within _TOL["density"] of case 1's;
         diagnostics carry average_sign.
    """
    params, iaft, ref, problem = dimer
    solver = CtsegSolver(dict(_SOLVER), comm_size=mpi.size)
    result = solver.solve(problem)
    result.validate(problem)
    assert abs(result.density - ref["occupations"].sum()) < _TOL["density"]
    i0 = _i0(ref["wn_f"])
    for spin, name in enumerate(("up_0", "down_0")):
        g = result.G_iw_data[spin][i0, 0, 0]
        assert abs(abs(g) - abs(ref["G_iw"][i0, spin])) / abs(ref["G_iw"][i0, spin]) < _TOL["g0_rel"]
        assert abs(result.Sigma_iw_data[spin][i0, 0, 0] - ref["Sigma_dyn_iw"][i0, spin]) < _TOL["sigma0"]
        assert abs(result.Sigma_infty[spin][0, 0] - ref["Sigma_infty"][spin]) < _TOL["sigma_infty"]
    assert np.allclose(result.Sigma_iw_data[0], result.Sigma_iw_data[1])
    for key, exact in (("Pi_iw_data", ref["Pi_iw"]), ("W_iw_data", ref["W_iw"])):
        got = getattr(result, key)[0][:, 0, 0]
        assert abs(got[0] - exact[0]) / abs(exact[0]) < _TOL["boson_rel"], key
        assert abs(got[1] - exact[1]) / abs(exact[1]) < _TOL["boson_rel"], key
    assert "average_sign" in result.diagnostics
    assert set(result.solver_data) == {"G_iw", "Sigma_iw", "Pi_iw", "W_iw", "Chi_iw"}

    density = solver.solve(problem, occupations_only=True)
    assert density.occupations_only and density.Pi_iw_data is None
    assert abs(density.density - result.density) < _TOL["density"]
    assert "average_sign" in density.diagnostics


def test_ctseg_solver_config_reuse(dimer, tmp_path):
    """With reuse_config, the production solve saves its configuration and the next iteration's
    solves start from it.

    Setup: a shared checkpoint path; a CtsegSolver with reuse_config and
    restart_n_warmup_cycles 100. Skipped without a CT-SEG build providing ConfigDataT.

    Cases (in order):
      1. Iteration 1, save True: diagnostics restarted_from_config False, config_source_iter -1
         (nothing stored yet).
      2. Iteration 2, save False, occupations-only: restarted_from_config True, config_source_iter 1.
    """
    triqs_ctseg = pytest.importorskip("triqs_ctseg")
    if not hasattr(triqs_ctseg, "ConfigDataT"):
        pytest.skip("needs a TRIQS/ctseg build with configuration reuse (ConfigDataT)")
    _, _, _, problem = dimer
    h5 = mpi.bcast(str(tmp_path)) + "/reuse.h5"
    solver = CtsegSolver(dict(_SOLVER, n_cycles=20000, reuse_config=True, restart_n_warmup_cycles=100),
                         comm_size=mpi.size)
    first = solver.solve(problem, chkpt={'h5': h5, 'impurity': 0, 'iteration': 1, 'save': True})
    assert first.diagnostics["restarted_from_config"] is False
    assert first.diagnostics["config_source_iter"] == -1
    second = solver.solve(problem, occupations_only=True,
                          chkpt={'h5': h5, 'impurity': 0, 'iteration': 2, 'save': False})
    assert second.diagnostics["restarted_from_config"] is True
    assert second.diagnostics["config_source_iter"] == 1


def test_ctseg_solver_config_reuse_params(monkeypatch):
    """CtsegSolver validates the configuration-reuse keys of its solver block at construction.

    Setup: solver blocks with reuse_config on; no solve runs.

    Cases:
      1. restart_n_warmup_cycles missing, zero, negative or fractional: ValueError naming the key.
      2. An integral float (1e4) is cast to int, and the chemical-potential search set inherits
         it when it sets none of its own; warm-up counts are per rank, so they are not divided
         by comm_size (2 here).
      3. A CT-SEG build without configuration reuse (triqs_ctseg replaced by an empty module, no
         ConfigDataT): ImportError naming ConfigDataT, before any solve.
    """
    base = {"n_cycles": 1000, "n_warmup_cycles": 100, "length_cycle": 50}
    for bad in (None, 0, -3, 0.5):
        params = dict(base, reuse_config=True)
        if bad is not None:
            params['restart_n_warmup_cycles'] = bad
        with pytest.raises(ValueError, match="restart_n_warmup_cycles"):
            CtsegSolver(params, comm_size=1)
    solver = CtsegSolver(dict(base, reuse_config=True, restart_n_warmup_cycles=1e4), comm_size=2)
    assert solver.production_params['restart_n_warmup_cycles'] == 10000
    assert isinstance(solver.production_params['restart_n_warmup_cycles'], int)
    assert solver.mu_search_params['restart_n_warmup_cycles'] == 10000
    monkeypatch.setitem(sys.modules, 'triqs_ctseg', types.ModuleType('triqs_ctseg'))
    with pytest.raises(ImportError, match="ConfigDataT"):
        CtsegSolver(dict(base, reuse_config=True, restart_n_warmup_cycles=10), comm_size=1)
