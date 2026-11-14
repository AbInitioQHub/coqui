"""
Impurity-solver contract of the DMFT package (coqui.dmft.solver_api). No TRIQS solver needed.
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from coqui.utils.imag_axes_ft import IAFT
from coqui.dmft import solver_api
from coqui.dmft.solver_api import (ImpurityProblem, ImpurityResult, SolverCapabilities,
                                   ImpuritySolver, register_solver, make_solver,
                                   qmc_cycle_params, resolve_mu_search_params, check_solver_capabilities_for_gw_edmft,
                                   check_solver_capabilities_for_problem)

_GF_STRUCT = [("up_0", 1), ("down_0", 1)]
_CAPS = SolverCapabilities(occupations_only_solve=True, general_h_int=False, offdiag_delta=False,
                           complex_hamiltonian=False, retarded_interaction="density",
                           polarization="always")


@pytest.fixture(scope="module")
def iaft():
    return IAFT(beta=20.0, wmax=20.0, prec="medium", verbose=False, basis="dlr")


def _problem(iaft, **overrides):
    nw_f = iaft.nw_f
    nw_b_pos = len(iaft.wn_mesh("b", positive_only=True))
    fields = dict(
        gf_struct=_GF_STRUCT, beta=iaft.beta, iaft=iaft, dlr_wmax=20.0, dlr_eps=1e-10,
        h0_sab=np.full((2, 1, 1), -0.5, dtype=complex),
        delta_wsab=np.zeros((nw_f, 2, 1, 1), dtype=complex),
        v_abcd=np.full((1, 1, 1, 1), 2.0, dtype=complex),
        u_weiss_wabcd=np.zeros((nw_b_pos, 1, 1, 1, 1), dtype=complex),
        target_density=1.0, occupations_sab=np.full((2, 1, 1), 0.5), degenerate_blk=[[0, 1]])
    fields.update(overrides)
    return ImpurityProblem(**fields)


class _Dummy(ImpuritySolver):
    """Backend stand-in: records its construction arguments, solves nothing."""
    name = "dummy"
    capabilities = _CAPS

    def __init__(self, solver_params, *, comm_size):
        self.solver_params, self.comm_size = dict(solver_params), comm_size
        self.h_int_kind = solver_params.get("h_int", "density")


def test_impurity_problem(iaft):
    """ImpurityProblem validates its arrays against gf_struct and the IAFT meshes.

    Setup: a one-orbital problem on a DLR IAFT (beta 20, wmax 20).

    Cases:
      1. Valid problem: norb is 1; dataclasses.replace(problem, mu_shift=0.3) is how the
         chemical-potential search shifts it: a shallow copy, the original untouched.
      2. delta_wsab with the wrong mesh length raises ValueError naming delta_wsab.
      3. h0_sab with the wrong orbital count raises ValueError naming h0_sab.
      4. gf_struct whose spins have different orbital counts raises ValueError.
    """
    problem = _problem(iaft)
    assert problem.norb == 1
    shifted = replace(problem, mu_shift=0.3)
    assert shifted.mu_shift == 0.3 and problem.mu_shift == 0.0
    assert shifted.h0_sab is problem.h0_sab          # a shallow copy: arrays are shared
    with pytest.raises(ValueError, match="delta_wsab"):
        _problem(iaft, delta_wsab=np.zeros((3, 2, 1, 1), dtype=complex))
    with pytest.raises(ValueError, match="h0_sab"):
        _problem(iaft, h0_sab=np.zeros((2, 2, 2), dtype=complex))
    with pytest.raises(ValueError, match="orbital"):
        _problem(iaft, gf_struct=[("up_0", 1), ("down_0", 2)])


def test_impurity_result(iaft):
    """ImpurityResult validates shapes against the problem and flattens to the Res dict.

    Setup: the problem of test_impurity_problem; arrays of the right shapes.

    Cases:
      1. Production result: solver_agnostic_results() has the five arrays, occupations, density and the
         diagnostics, and validate(problem) passes; occupations_only is False.
      2. Density-only result (no Sigma): occupations_only is True; solver_agnostic_results() carries no Sigma key.
      3. Pi_iw_data without W_iw_data raises ValueError.
      4. A Sigma_iw_data block with the wrong mesh length fails validate(problem) with ValueError.
    """
    problem = _problem(iaft)
    nw_f = iaft.nw_f
    nw_b_pos = len(iaft.wn_mesh("b", positive_only=True))
    occ = {"up_0": np.array([0.5]), "down_0": np.array([0.5])}
    full = ImpurityResult(
        orbital_occupations=occ, density=1.0,
        Sigma_infty=[np.zeros((1, 1)), np.zeros((1, 1))],
        Sigma_iw_data=[np.zeros((nw_f, 1, 1), complex)] * 2,
        G_iw_data=[np.zeros((nw_f, 1, 1), complex)] * 2,
        Pi_iw_data=[np.zeros((nw_b_pos, 1, 1))], W_iw_data=[np.zeros((nw_b_pos, 1, 1))],
        diagnostics={"average_sign": 1.0})
    full.validate(problem)
    flat = full.solver_agnostic_results()
    assert set(flat) == {"orbital_occupations", "density", "Sigma_infty", "Sigma_iw_data",
                         "G_iw_data", "Pi_iw_data", "W_iw_data", "average_sign"}
    assert full.occupations_only is False
    dens = ImpurityResult(orbital_occupations=occ, density=1.0)
    assert dens.occupations_only is True and "Sigma_iw_data" not in dens.solver_agnostic_results()
    with pytest.raises(ValueError, match="W_iw_data"):
        ImpurityResult(orbital_occupations=occ, density=1.0, Pi_iw_data=[np.zeros((nw_b_pos, 1, 1))])
    bad = ImpurityResult(orbital_occupations=occ, density=1.0,
                         Sigma_infty=full.Sigma_infty, G_iw_data=full.G_iw_data,
                         Sigma_iw_data=[np.zeros((3, 1, 1), complex)] * 2)
    with pytest.raises(ValueError, match="Sigma_iw_data"):
        bad.validate(problem)


def test_qmc_cycle_params():
    """qmc_cycle_params splits n_cycles over the ranks; resolve_mu_search_params derives the
    parameter set of the chemical-potential search as the production set with different cycle
    controls, and nothing else.

    Setup: a solver block with n_cycles 1000 (total), comm_size 4, and two keys the search never
    touches (random_seed, perform_tail_fit).

    Cases:
      1. Production: n_cycles 250 per rank; warm-up and cycle length untouched and cast to int;
         a block without a cycle control raises ValueError naming it.
      2. Empty mu_search table (search requested, nothing set): n_cycles is 5 % of the total
         per rank (12), verbosity 0, suppress_solver_output True; every other key equals the
         production value (the two untouched keys included); the production block is not modified.
      3. Overrides are applied, including a 0 warm-up (no truthiness test), and a key outside the
         cycle controls and the solver-output keys ("random_seed", "n_cycle") raises ValueError
         naming it.
      4. A budget below one cycle per rank (n_cycles 10 on 4 ranks: 2 production, 0 search) is
         raised to one cycle per rank, never zero.
    """
    block = {"n_cycles": 1000, "n_warmup_cycles": 20.0, "length_cycle": 5, "random_seed": 7,
             "perform_tail_fit": True}
    production = qmc_cycle_params(solver_params=block, comm_size=4)
    assert production["n_cycles"] == 250 and production["n_warmup_cycles"] == 20
    assert isinstance(production["n_warmup_cycles"], int) and production["random_seed"] == 7
    with pytest.raises(ValueError, match="length_cycle"):
        qmc_cycle_params(solver_params={"n_cycles": 1000, "n_warmup_cycles": 20}, comm_size=4)
    search = resolve_mu_search_params(production_params=block, mu_search={}, comm_size=4)
    assert search["n_cycles"] == 12 and search["verbosity"] == 0
    assert search["suppress_solver_output"] is True
    assert {k: v for k, v in search.items() if k not in ("n_cycles", "verbosity", "suppress_solver_output")} \
        == {k: v for k, v in production.items() if k != "n_cycles"}
    assert block["n_cycles"] == 1000 and "verbosity" not in block
    search = resolve_mu_search_params(
        production_params=block, comm_size=4,
        mu_search={"n_cycles": 400, "length_cycle": 9, "n_warmup_cycles": 0,
                   "solver_output_file": "mu.log", "suppress_solver_output": False})
    assert search["n_cycles"] == 100 and search["length_cycle"] == 9 and search["n_warmup_cycles"] == 0
    assert search["solver_output_file"] == "mu.log" and search["suppress_solver_output"] is False
    for bad in ("random_seed", "n_cycle"):
        with pytest.raises(ValueError, match=bad):
            resolve_mu_search_params(production_params=block, mu_search={bad: 1}, comm_size=4)
    small = dict(block, n_cycles=10)
    assert qmc_cycle_params(solver_params=small, comm_size=4)["n_cycles"] == 2
    assert resolve_mu_search_params(production_params=small, mu_search={}, comm_size=4)["n_cycles"] == 1


def test_make_solver(monkeypatch):
    """make_solver constructs the backend registered under the solver block's type.

    Setup: _Dummy registered under "dummy"; two entries added to the shipped-backend table
    (the lazily imported form), one whose module does not exist and one whose class is not a
    solver.

    Cases:
      1. Registered name: a _Dummy with the block minus "type" (its mu_search table included)
         and comm_size passed through.
      2. An unknown name raises ValueError listing the registered names; "cthyb" and "ctint"
         are such names until their backends exist (no misleading missing-package error).
      3. A shipped entry whose module does not exist raises ImportError naming the module and
         the package it needs; one whose class is not an ImpuritySolver raises TypeError.
         registered_solvers() lists both tables.
      4. A type that is not a string (a class) raises TypeError; a block without "type" raises
         KeyError; register_solver rejects a class that is not an ImpuritySolver.
      5. A block with h_int "full" on a backend without general_h_int raises ValueError naming
         the capability (the backend's configuration against its own capabilities, checked once
         at construction); with the capability granted it constructs.
    """
    register_solver("dummy", _Dummy)
    solver = make_solver({"type": "dummy", "n_cycles": 10, "mu_search": {"n_cycles": 2}}, comm_size=3)
    assert isinstance(solver, _Dummy)
    assert solver.solver_params == {"n_cycles": 10, "mu_search": {"n_cycles": 2}} and solver.comm_size == 3
    with pytest.raises(ValueError, match="dummy"):
        make_solver({"type": "no_such_solver"}, comm_size=1)
    for name in ("cthyb", "ctint"):
        with pytest.raises(ValueError, match="unknown solver type"):
            make_solver({"type": name}, comm_size=1)
    monkeypatch.setitem(solver_api._SHIPPED_SOLVERS, "broken", ("no_such_package.solver", "Solver", "no_such_package"))
    with pytest.raises(ImportError, match="no_such_package"):
        make_solver({"type": "broken"}, comm_size=1)
    monkeypatch.setitem(solver_api._SHIPPED_SOLVERS, "notasolver", ("coqui.dmft.solver_api", "SolverCapabilities", "coqui"))
    with pytest.raises(TypeError, match="ImpuritySolver"):
        make_solver({"type": "notasolver"}, comm_size=1)
    assert {"ctseg", "dummy", "broken", "notasolver"} <= set(solver_api.registered_solvers())
    with pytest.raises(TypeError, match="registered name"):
        make_solver({"type": _Dummy, "n_cycles": 10}, comm_size=1)
    with pytest.raises(KeyError, match="type"):
        make_solver({"n_cycles": 1}, comm_size=1)
    with pytest.raises(TypeError, match="ImpuritySolver"):
        register_solver("capabilities", SolverCapabilities)
    with pytest.raises(ValueError, match="general_h_int"):
        make_solver({"type": "dummy", "h_int": "full"}, comm_size=1)
    monkeypatch.setattr(_Dummy, "capabilities", SolverCapabilities(**{**_CAPS.__dict__, "general_h_int": True}))
    assert make_solver({"type": "dummy", "h_int": "full"}, comm_size=1).h_int_kind == "full"


def test_check_solver_capabilities_for_problem(iaft):
    """check_solver_capabilities_for_problem rejects a problem the backend cannot take, before the
    solve.

    Setup: _Dummy instances with modified capabilities; the one-orbital problem of _problem.

    Cases:
      1. A size-2 block on a backend without offdiag_delta: ValueError naming the capability.
      2. A nonzero u_weiss_wabcd on a backend with retarded_interaction "none": ValueError
         naming it; the same problem with a zero u_weiss_wabcd passes.
    """
    problem = _problem(iaft)
    solver = _Dummy({}, comm_size=1)
    solver.capabilities = SolverCapabilities(**{**_CAPS.__dict__, "offdiag_delta": False})
    big = _problem(iaft, gf_struct=[("up_0", 2), ("down_0", 2)],
                   h0_sab=np.zeros((2, 2, 2), dtype=complex),
                   delta_wsab=np.zeros((len(problem.delta_wsab), 2, 2, 2), dtype=complex),
                   v_abcd=np.zeros((2, 2, 2, 2), dtype=complex),
                   u_weiss_wabcd=np.zeros((len(problem.u_weiss_wabcd), 2, 2, 2, 2), dtype=complex),
                   occupations_sab=np.full((2, 2, 2), 0.5))
    with pytest.raises(ValueError, match="offdiag_delta"):
        check_solver_capabilities_for_problem(solver=solver, problem=big)
    solver.capabilities = SolverCapabilities(**{**_CAPS.__dict__, "retarded_interaction": "none"})
    problem.u_weiss_wabcd[0] = 0.1
    with pytest.raises(ValueError, match="retarded_interaction"):
        check_solver_capabilities_for_problem(solver=solver, problem=problem)
    problem.u_weiss_wabcd[:] = 0.0
    check_solver_capabilities_for_problem(solver=solver, problem=problem)


def test_check_solver_capabilities_for_gw_edmft():
    """check_solver_capabilities_for_gw_edmft aborts before any solve on a capability the workflow needs.

    Setup: _Dummy instances with modified capabilities.

    Cases:
      1. chemical_potential requested, occupations_only_solve False: ValueError naming the
         capability; so does a solver.mu_search table on such a backend, even without a search.
      2. screen_type "gw_edmft", polarization "never": ValueError; with "rpa" it passes.
      3. polarization "density_h_int_only" with h_int_kind "full": ValueError; with "density"
         it passes.
    """
    solver = _Dummy({}, comm_size=1)
    solver.capabilities = SolverCapabilities(**{**_CAPS.__dict__, "occupations_only_solve": False})
    with pytest.raises(ValueError, match="occupations_only_solve"):
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={"chemical_potential": {}}, screen_type="rpa")
    with pytest.raises(ValueError, match="mu_search"):
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={"solver": {"type": "dummy", "mu_search": {}}},
                               screen_type="rpa")
    solver.capabilities = SolverCapabilities(**{**_CAPS.__dict__, "polarization": "never"})
    with pytest.raises(ValueError, match="polarization"):
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={}, screen_type="gw_edmft")
    check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={}, screen_type="rpa")
    solver.capabilities = SolverCapabilities(**{**_CAPS.__dict__, "polarization": "density_h_int_only"})
    solver.h_int_kind = "full"
    with pytest.raises(ValueError, match="polarization"):
        check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={}, screen_type="gw_edmft")
    solver.h_int_kind = "density"
    check_solver_capabilities_for_gw_edmft(solver=solver, imp_params={}, screen_type="gw_edmft")


