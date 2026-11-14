"""
CoQuí-array to TRIQS conversion of the impurity problem (coqui.dmft.triqs_utils.problem_to_triqs).
Serial: python -m pytest <this file> -v     (from outside the repository)
MPI:    mpirun -n 2 python -m pytest <this file> -v
"""
import numpy as np
import pytest

pytest.importorskip("triqs", reason="triqs is required for dmft tests")
pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

import triqs_modest as modest
from coqui.utils.imag_axes_ft import IAFT
from coqui.dmft.solver_api import ImpurityProblem
from coqui.dmft.triqs_utils import (problem_to_triqs, dlr_meshes, to_block_gf, to_block2_gf,
                                    h_int_density_density, delta_block_gf,
                                    symmetrize_h0_op, symmetrize_h_int_op, symmetrize_blk2_gf,
                                    h0_operator_to_array, h0_operator, analyze_degenerate_blocks,
                                    _between_blocks_max, _outside_density_density)
from coqui.dmft import dynamic_4idx_u_to_dd_basis

_GF_STRUCT = [("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)]

@pytest.fixture(scope="module")
def iaft():
    return IAFT(beta=10.0, wmax=10.0, prec="medium", verbose=False, basis="dlr")


def _two_orbital_problem(iaft, *, degenerate_blk=None, mu_shift=0.0, offdiag=0.0, exchange=0.0,
                         imag=0.0):
    """Two orbitals, levels -0.4 and -0.6, Lorentzian Delta with weight 0.3 (+offdiag between the
    orbitals), U 2.0 / U' 1.5 (+exchange V[0,1,1,0]), one retarded pole, optional imaginary part."""
    wn_f = iaft.wn_mesh("f")
    iw = 1j * np.pi * wn_f / iaft.beta
    norb = 2
    delta = np.zeros((len(wn_f), 2, norb, norb), dtype=complex)
    for spin in range(2):
        for orb in range(norb):
            delta[:, spin, orb, orb] = 0.3 / (iw - 0.2) + 1j * imag
        delta[:, spin, 0, 1] = delta[:, spin, 1, 0] = offdiag / (iw - 0.2)
    h0 = np.zeros((2, norb, norb), dtype=complex)
    h0[:, 0, 0], h0[:, 1, 1] = -0.4 + 1j * imag, -0.6
    v = np.zeros((norb,) * 4, dtype=complex)
    for i in range(norb):
        for j in range(norb):
            v[i, i, j, j] = 2.0 if i == j else 1.5
    v[0, 1, 1, 0] = v[1, 0, 0, 1] = exchange
    wn_b = iaft.wn_mesh("b", positive_only=True)
    nu = np.pi * wn_b / iaft.beta
    u = np.zeros((len(wn_b),) + (norb,) * 4, dtype=complex)
    for i in range(norb):
        for j in range(norb):
            u[:, i, i, j, j] = -0.8 / (1.0 + nu**2)
    return ImpurityProblem(gf_struct=_GF_STRUCT, beta=iaft.beta, iaft=iaft, dlr_wmax=10.0,
                           dlr_eps=1e-10, h0_sab=h0, delta_wsab=delta, v_abcd=v, u_weiss_wabcd=u,
                           target_density=2.0, occupations_sab=np.zeros((2, norb, norb)),
                           degenerate_blk=degenerate_blk, mu_shift=mu_shift)


def _legacy(problem):
    """The pre-interface driver path, operation by operation: real h0 operator, Delta made real
    on the tau axis, density-density h_int, density-density U as a Block2Gf, the four symmetrize
    calls, then the mu shift."""
    mesh_f, mesh_b = dlr_meshes(beta=problem.beta, dlr_wmax=problem.dlr_wmax, dlr_eps=problem.dlr_eps)
    h0 = h0_operator(problem.h0_sab, problem.gf_struct, force_real=True)
    delta_tau = problem.iaft.w_to_tau(problem.delta_wsab, 'f')
    delta_tau.imag = 0.0
    delta = to_block_gf(problem.iaft.tau_to_w(delta_tau, 'f'), problem.iaft, problem.gf_struct, mesh_f)
    h_int = h_int_density_density(problem.v_abcd, problem.gf_struct, force_real=True)
    u_iw_s1s2 = dynamic_4idx_u_to_dd_basis(problem.u_weiss_wabcd, nspin=2, screen_j=False)
    u = to_block2_gf(u_iw_s1s2.real, problem.iaft, problem.gf_struct, mesh_b)
    if problem.degenerate_blk is not None:
        delta = modest.symmetrize(delta, problem.degenerate_blk)
        h0 = symmetrize_h0_op(h0, problem.degenerate_blk, problem.gf_struct)
        h_int = symmetrize_h_int_op(h_int, problem.degenerate_blk, problem.gf_struct)
        u = symmetrize_blk2_gf(u, problem.degenerate_blk, problem.gf_struct)
    if problem.mu_shift != 0.0:
        h0_sab = h0_operator_to_array(h0, problem.gf_struct)
        h0 = h0_operator(np.array([h - np.eye(h.shape[0]) * problem.mu_shift for h in h0_sab]),
                         problem.gf_struct, force_real=True)
    return h0, delta, h_int, u


def _assert_same_inputs(got, legacy):
    h0, delta, h_int, u = legacy
    assert (got.h0_op - h0).is_zero()
    assert (got.h_int - h_int).is_zero()
    for name, gf in delta:
        assert np.array_equal(got.delta_iw[name].data, gf.data)
    for name1, name2 in u.indices:
        assert np.array_equal(got.u_weiss_iw[name1, name2].data, u[name1, name2].data)


def test_problem_to_triqs(iaft):
    """problem_to_triqs reproduces today's conversion exactly for the CT-SEG knobs (density-density,
    real); the helpers that measure the reductions for its log report the right magnitudes.

    Setup: a two-orbital problem with four size-1 blocks on a DLR IAFT (beta 10).

    Cases:
      1. No degenerate blocks, no shift: the four TRIQS objects equal today's path element by
         element (regression pin of the refactor: exact equality, the same operations run).
      2. degenerate_blk [[0, 1], [2, 3]] (arrays) and mu_shift 0.3: same, the h0 blocks are
         shifted by -0.3.
      3. The reduction measures (_between_blocks_max, _outside_density_density): an exchange
         element is kept (0 outside density-density + exchange); an element outside the
         density-density and exchange channels is measured; an off-diagonal Delta 0.1 is
         measured as 0.1 / |iw0 - 0.2| between blocks; an h0 element 0.02 between two blocks
         is measured and dropped by the conversion (real downfolded problems carry such
         elements at the 1e-5 level, which the block cutting has always dropped).
      4. degenerate_blk given as lists of ints (TOML form) works like arrays.
      5. An unknown h_int_kind raises ValueError naming h_int (whether a backend can take a
         problem is check_solver_capabilities_for_problem's job, tested with solver_api).
    """
    problem = _two_orbital_problem(iaft)
    got = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
    _assert_same_inputs(got, _legacy(problem))

    problem = _two_orbital_problem(iaft, degenerate_blk=[np.array([0, 1]), np.array([2, 3])], mu_shift=0.3)
    got = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
    _assert_same_inputs(got, _legacy(problem))
    assert np.allclose([blk[0, 0] for blk in got.h0_blocks], -0.5 - 0.3)   # averaged, then shifted

    problem = _two_orbital_problem(iaft, offdiag=0.1, exchange=0.25, imag=0.05)
    problem.h0_sab[:, 0, 1] = problem.h0_sab[:, 1, 0] = 0.02   # between two size-1 blocks: dropped
    assert _between_blocks_max(problem.h0_sab, problem.gf_struct) == pytest.approx(0.02)
    got = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
    assert all(blk.shape == (1, 1) for blk in got.h0_blocks)   # the element has nowhere to go
    assert _outside_density_density(problem.v_abcd, keep_exchange=True) == 0.0   # V[0,1,1,0]: exchange, kept
    problem.v_abcd[0, 1, 0, 1] = 0.3                         # neither density-density nor exchange
    assert _outside_density_density(problem.v_abcd, keep_exchange=True) == pytest.approx(0.3)
    wn_pos = iaft.wn_mesh("f")[iaft.wn_mesh("f") > 0]
    assert _between_blocks_max(problem.delta_wsab, problem.gf_struct) == pytest.approx(
        0.1 / abs(1j * np.pi * wn_pos.min() / iaft.beta - 0.2), rel=1e-6)

    problem = _two_orbital_problem(iaft, degenerate_blk=[[0, 1], [2, 3]])
    got_lists = problem_to_triqs(problem, h_int_kind="density", screen_j=False, real_hamiltonian=True)
    problem = _two_orbital_problem(iaft, degenerate_blk=[np.array([0, 1]), np.array([2, 3])])
    _assert_same_inputs(got_lists, _legacy(problem))

    with pytest.raises(ValueError, match="h_int"):
        problem_to_triqs(problem, h_int_kind="slater", screen_j=False, real_hamiltonian=True)


def test_analyze_degenerate_blocks(iaft):
    """analyze_degenerate_blocks infers the degenerate groups from Delta as the driver did.

    Setup: the two-orbital problem; the two orbitals have the same Delta. The expected result
    is ModEST's own answer on the same (real-forced, DLR-sampled) Delta BlockGf, threshold 1e-3.
    """
    problem = _two_orbital_problem(iaft)
    mesh_f, mesh_b = dlr_meshes(beta=iaft.beta, dlr_wmax=10.0, dlr_eps=1e-10)
    delta_gf = delta_block_gf(problem.delta_wsab, iaft, _GF_STRUCT, mesh_f, force_real=True)
    expected = modest.analyze_degenerate_blocks(delta_gf, threshold=1e-3)
    got = analyze_degenerate_blocks(delta_iw=problem.delta_wsab, iaft=iaft, gf_struct=_GF_STRUCT,
                                    dlr_wmax=10.0, dlr_eps=1e-10, threshold=1e-3, force_real=True)
    assert [list(map(int, g)) for g in got] == [list(map(int, g)) for g in expected]
