"""
Per-impurity step of the GW+EDMFT driver (coqui.dmft.scf_driver._solve_impurity and
_degenerate_blocks, _make_solvers) on the Hubbard–Holstein dimer with CT-SEG.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("triqs_ctseg", reason="triqs_ctseg is required for this test")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from triqs.utility import mpi
from coqui.dmft.ctseg import CtsegSolver
from coqui.dmft.scf_driver import _solve_impurity, _make_solvers, _degenerate_blocks
from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem

_SOLVER = {"n_cycles": 100000, "n_warmup_cycles": 2000, "length_cycle": 50, "random_seed": 21,
           "perform_tail_fit": False, "analytic_hf": True}
_DENSITY_TOL = 0.02 + 3 * 1.4e-3     # search tolerance + 3 sigma of the search's density estimate


def test_solve_impurity_chemical_potential():
    """_solve_impurity finds the impurity chemical potential that reproduces the target density
    through occupations-only solves, then runs the production solve at that shift.

    Setup: the dimer at its default level eps; the target density is the ED density of the
    dimer at eps - 0.3 (so the exact answer is mu_imp = 0.3); search tolerance 0.02; CT-SEG at
    the budget _SOLVER with the search at 20 % of it.

    Cases:
      1. problem.mu_shift on return reproduces the target: the ED density at eps - mu_shift is within
         _DENSITY_TOL of the target (checked through ED, not through Monte Carlo). The search
         stops on a Monte Carlo density estimate inside the 0.02 band, so the exact density at
         the returned shift can sit up to the estimate's noise outside it: _DENSITY_TOL is the
         search tolerance plus three times the density spread of a 20 000-cycle solve
         (1.4e-3 scaled from the 9e-4 three-seed spread measured at 200 000 cycles).
      2. The production result is at the shifted level: its density is within _DENSITY_TOL of
         the target.
      3. Without chemical_potential: mu_shift stays 0 and the density is the unshifted ED density
         within _DENSITY_TOL.
    """
    params = DimerParams()
    iaft = make_iaft(params)
    problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10)
    shifted_ed = DimerED(DimerParams(eps=params.eps - 0.3))
    target = float(shifted_ed.occupations().sum())
    problem.target_density = target
    imp_params = {"chemical_potential": {"tolerance": 0.02}}
    solver = CtsegSolver(dict(_SOLVER, mu_search={"n_cycles": 20000}), comm_size=mpi.size)
    result = _solve_impurity(solver=solver, problem=problem, imp_params=imp_params, chkpt=None)
    found = float(DimerED(DimerParams(eps=params.eps - problem.mu_shift)).occupations().sum())
    assert abs(found - target) < _DENSITY_TOL
    assert abs(result.density - target) < _DENSITY_TOL
    solver = CtsegSolver(dict(_SOLVER), comm_size=mpi.size)
    problem.mu_shift = 0.0
    result = _solve_impurity(solver=solver, problem=problem, imp_params={}, chkpt=None)
    assert problem.mu_shift == 0.0
    assert abs(result.density - DimerED(params).occupations().sum()) < _DENSITY_TOL


def test_degenerate_blocks_and_make_solvers():
    """_degenerate_blocks decides the degenerate groups of an impurity and _make_solvers builds
    one backend per impurity block with the capability checks.

    Setup: the dimer's hybridization in an Input dict, as the loop lays it out.

    Cases:
      1. _degenerate_blocks returns the explicit groups of imp_params, the inferred groups of
         the dimer's hybridization (both orbitals degenerate: [[0, 1]] from ModEST on the same
         real-forced Delta) when only degenerate_blk_thresh is set, and None when neither is
         set; it modifies no argument.
      2. _make_solvers on two blocks (ctseg with and without a mu_search table) returns two
         CtsegSolver instances; the second has occupations_only None.
      3. _make_solvers with screen_type "gw_edmft" and a block whose solver declares no
         polarization raises ValueError (a dummy class registered under "nopi" for the test).
    """
    params = DimerParams()
    iaft = make_iaft(params)
    problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10)
    Input = {"delta_iw": problem.delta_wsab, "gf_struct": problem.gf_struct, "degenerate_blk": [[0, 1]]}
    solver = CtsegSolver(dict(_SOLVER), comm_size=mpi.size)
    imp_params = {"degenerate_blk": [[1, 0]], "degenerate_blk_thresh": 1e-3}
    assert _degenerate_blocks(Input=Input, imp_params=imp_params, iaft=iaft, dlr_wmax=20.0,
                              dlr_eps=1e-10, solver=solver) == [[1, 0]]
    imp_params = {"degenerate_blk_thresh": 1e-3}
    inferred = _degenerate_blocks(Input=Input, imp_params=imp_params, iaft=iaft, dlr_wmax=20.0,
                                  dlr_eps=1e-10, solver=solver)
    assert [sorted(map(int, g)) for g in inferred] == [[0, 1]]
    assert imp_params == {"degenerate_blk_thresh": 1e-3} and Input["degenerate_blk"] == [[0, 1]]
    assert _degenerate_blocks(Input=Input, imp_params={}, iaft=iaft, dlr_wmax=20.0, dlr_eps=1e-10,
                              solver=solver) is None
    blocks = [{"solver": dict(_SOLVER, type="ctseg", mu_search={"n_cycles": 40 * mpi.size}),
               "chemical_potential": {"tolerance": 0.1}},
              {"solver": dict(_SOLVER, type="ctseg")}]
    solvers = _make_solvers(impurity_params_list=blocks, comm_size=mpi.size, screen_type="gw_edmft")
    assert [type(s) for s in solvers] == [CtsegSolver, CtsegSolver]
    assert solvers[0].mu_search_params["n_cycles"] == 40
    assert solvers[1].mu_search_params["n_cycles"] == max(1, int(int(_SOLVER["n_cycles"] * 0.05) / mpi.size))
    from coqui.dmft.solver_api import ImpuritySolver, SolverCapabilities, register_solver

    class _NoPi(ImpuritySolver):
        name = "nopi"
        capabilities = SolverCapabilities(occupations_only_solve=False, general_h_int=False,
                                          offdiag_delta=False, complex_hamiltonian=False,
                                          retarded_interaction="density", polarization="never")

        def __init__(self, solver_params, *, comm_size):
            pass

    register_solver("nopi", _NoPi)
    with pytest.raises(ValueError, match="polarization"):
        _make_solvers(impurity_params_list=[{"solver": {"type": "nopi"}}], comm_size=1, screen_type="gw_edmft")
