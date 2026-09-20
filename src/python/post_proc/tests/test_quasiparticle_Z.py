"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================

Unit tests for the linearized quasiparticle approximation and band-resolved
pole weights (post_proc.linearized_qp: linearized_qp_from_arrays,
linearized_qp_from_checkpoint), which run on the CoQui C++ kernel and return its
result structs (LinearizedQPResult = methods::lqp::result_t, LinearizedQPLadder =
methods::lqp::ladder_result_t) with the C++ member names.

The independent NumPy implementation and the tests that compare the two live with
the note, in implementation_notes/quasiparticle_Z_at_arbitrary_k/.

Analytic reference: Sigma(z) = sum_p V_p / (z - e_p) with Hermitian V_p has
    A = Sigma(0)          = -sum_p V_p / e_p
    B = dSigma/d(iw)|_0   = -sum_p V_p / e_p^2
and for a single scalar pole Z = 1 / (1 + g^2/e^2).  Hermitian but complex
off-diagonal residues are used deliberately: a real-symmetric Sigma would hide
the error of fitting Re and Im separately.
"""

from mpi4py import MPI
import os
import tempfile
import warnings
import numpy as np
import pytest
from h5 import HDFArchive

from coqui import IAFT
from coqui.post_proc import (
    linearized_qp_from_arrays,
    linearized_qp_from_checkpoint,
    LinearizedQPResult,
    LinearizedQPLadder,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _iaft(beta=100.0, wmax=40.0):
    return IAFT(beta=beta, wmax=wmax, prec="high", basis="dlr", verbose=False)


def _iw(iaft):
    return 1j * iaft.wn_mesh("fermion") * np.pi / iaft.beta


def _single_pole(iaft, eps, g):
    return g**2 / (_iw(iaft) - eps)


def _Z_exact(eps, g):
    return 1.0 / (1.0 + g**2 / eps**2)


def _hermitian_poles(n, npole, seed):
    rs = np.random.RandomState(seed)
    e = rs.uniform(4.0, 12.0, npole) * rs.choice([-1.0, 1.0], npole)
    V = []
    for _ in range(npole):
        M = rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n))
        V.append(M + M.conj().T)
    return e, V


def _sigma_w(iaft, e, V):
    iw = _iw(iaft)
    S = np.zeros((len(iw),) + V[0].shape, dtype=np.complex128)
    for ep, Vp in zip(e, V):
        S += Vp[None] / (iw[:, None, None] - ep)
    return S


def _exact_AB(e, V):
    return (-sum(Vp / ep for ep, Vp in zip(e, V)),
            -sum(Vp / ep**2 for ep, Vp in zip(e, V)))


def _random_hermitian(n, seed, scale=1.0):
    rs = np.random.RandomState(seed)
    M = rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n))
    return scale * (M + M.conj().T)


def _random_unitary(n, seed):
    rs = np.random.RandomState(seed)
    q, r = np.linalg.qr(rs.normal(size=(n, n)) + 1j * rs.normal(size=(n, n)))
    return q * (np.diag(r) / np.abs(np.diag(r)))


def _causal_B(n, seed, scale=0.3):
    M = _random_hermitian(n, seed)
    return -scale * (M @ M.conj().T) / n


def _K(res):
    """Static matrix K = F + A - mu = Z^-1/2 (H_QP - mu) Z^-1/2 per (s, k), from the stored Z, H_QP
    and mu (H_QP is absolute; the linearization sits inside the sandwich at z = mu)."""
    w, U = np.linalg.eigh(res.Z_skab)
    Zmh = np.einsum("skal,skl,skbl->skab", U, 1.0 / np.sqrt(w), U.conj())
    Hrel = res.Hqp_skab - res.mu * np.eye(res.Hqp_skab.shape[-1])
    return Zmh @ Hrel @ Zmh


def _B(res):
    """Zero-frequency slope B = I - Z^-1 per (s, k)."""
    return np.eye(res.Z_skab.shape[-1]) - np.linalg.inv(res.Z_skab)


def _Z_scalar(coeffs):
    """Z from the slope of a stack of 1x1 'matrices', shape coeffs[1][..., 0, 0]."""
    return 1.0 / (1.0 - coeffs[1][..., 0, 0].real)


def _poles_near_window(iaft, seed=7):
    """A pole at e = 0.6 lies inside the fit window for n_fit >= 5, so the Taylor
    coefficients grow and the exactly-determined fit hits the conditioning wall
    (residual 1e-12 at n_fit=15, 1e-8 at 20, 4e-6 at 30). Far poles never do."""
    rs = np.random.RandomState(seed)
    V = []
    for _ in range(3):
        M = rs.normal(size=(3, 3)) + 1j * rs.normal(size=(3, 3))
        V.append(M @ M.conj().T)                 # positive semidefinite: a causal Sigma
    return _sigma_w(iaft, [0.6, -7.0, 9.0], V)



def test_scalar_pole_gives_analytic_Z_and_energies():
    """A single pole with a scalar residue, Sigma(iw) = g I / (iw - e), has A = -(g/e) I and
    B = -(g/e^2) I, so the whole construction collapses to scalars: Z = (1 + g/e^2)^-1 on every
    level and H_QP = K / (1 + g/e^2). Checks the kernel against the closed form rather than
    against another implementation."""
    iaft = _iaft(); n = 4
    e, g = 7.0, 3.0
    V = [g * np.eye(n, dtype=np.complex128)]
    St = _sigma_tau(iaft, [e], V)
    F = _random_hermitian(n, 17); mu = 0.3
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], mu, iaft,
                                    converge=False, n_fit=8)
    z = 1.0 / (1.0 + g / e**2)
    np.testing.assert_allclose(res.Zqp_ska[0, 0], np.full(n, z), atol=1e-10)
    eigF = np.linalg.eigvalsh(F)
    # absolute energies: E = Z (F + A - mu) + mu
    np.testing.assert_allclose(np.sort(res.E_ska[0, 0]), np.sort((eigF - g / e - mu) * z + mu), atol=1e-9)
    np.testing.assert_allclose(res.Z_skab[0, 0], z * np.eye(n), atol=1e-10)

# ---------------------------------------------------------------------------
# linearized_qp_from_arrays: basis covariance, projection (non-)commutation, ladder
# ---------------------------------------------------------------------------

def _sigma_tau(iaft, e, V):
    return iaft.w_to_tau(_sigma_w(iaft, e, V), "fermion")


def test_from_arrays_single_fit_returns_a_consistent_result():
    iaft = _iaft(); n = 5
    e, Vp = _hermitian_poles(n, 3, 21); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 22); mu = 0.1
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], mu, iaft, converge=False, n_fit=8)
    assert isinstance(res, LinearizedQPResult) and not isinstance(res, LinearizedQPLadder)
    assert res.E_ska.shape == (1, 1, n) and res.Z_skab.shape == (1, 1, n, n)
    assert res.n_fit == 8 and res.fit_order == 15 and res.status_sk[0, 0] == 0
    # Zqp <= 1 is not asserted here: _hermitian_poles draws Hermitian residues that are not
    # positive semidefinite, so this Sigma is not causal and B is not negative semidefinite.
    assert np.all(res.Zqp_ska > 0.0)
    # pole weights are the diagonal of Z in the H_QP eigenbasis, so they sum to Tr Z
    assert abs(res.Zqp_ska[0, 0].sum() - np.trace(res.Z_skab[0, 0]).real) < 1e-10
    assert abs(res.sumrule_sk[0, 0]) < 1e-10
    # H_QP is stored with its eigendecomposition
    Hqp = res.V_skab[0, 0] @ np.diag(res.E_ska[0, 0]) @ res.V_skab[0, 0].conj().T
    np.testing.assert_allclose(Hqp, res.Hqp_skab[0, 0], atol=1e-11)


def test_from_arrays_is_basis_covariant():
    iaft = _iaft(); n = 5
    e, Vp = _hermitian_poles(n, 3, 23); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 24); U = _random_unitary(n, 25)
    r0 = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft, converge=False, n_fit=6)
    r1 = linearized_qp_from_arrays((U.conj().T @ F @ U)[None, None],
                                   np.einsum("ab,tbc,cd->tad", U.conj().T, St, U)[:, None, None],
                                   0.0, iaft, converge=False, n_fit=6)
    np.testing.assert_allclose(r1.E_ska, r0.E_ska, atol=1e-11)
    np.testing.assert_allclose(r1.Zqp_ska, r0.Zqp_ska, atol=1e-11)
    np.testing.assert_allclose(r1.Z_skab[0, 0], U.conj().T @ r0.Z_skab[0, 0] @ U, atol=1e-11)


def test_projection_commutes_with_K_and_B_but_not_with_Z():
    """K and B come from a linear fit, so V K V^dag is exact; Z = (I-B)^-1 and the
    QP energies are nonlinear and are NOT the projections of the full ones."""
    iaft = _iaft(); n, n_sub = 6, 3
    e, Vp = _hermitian_poles(n, 3, 26); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 27); Vw = _random_unitary(n, 28)[:n_sub]
    proj = lambda X: Vw @ X @ Vw.conj().T
    full = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft, converge=False, n_fit=8)
    St_W = np.array([proj(St[t]) for t in range(St.shape[0])])
    sub = linearized_qp_from_arrays(proj(F)[None, None], St_W[:, None, None], 0.0, iaft, converge=False, n_fit=8)
    np.testing.assert_allclose(_K(sub)[0, 0], proj(_K(full)[0, 0]), atol=1e-11)
    np.testing.assert_allclose(_B(sub)[0, 0], proj(_B(full)[0, 0]), atol=1e-11)
    assert np.abs(sub.Z_skab[0, 0] - proj(full.Z_skab[0, 0])).max() > 1e-3
    assert min(np.abs(full.E_ska[0, 0][:, None] - sub.E_ska[0, 0][None, :]).min(axis=0)) > 1e-4


def test_ladder_converges_and_reports_uncertainties():
    iaft = _iaft(); n = 4
    e, Vp = _hermitian_poles(n, 3, 31); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 32)
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft, converge=True, n_fit_max=10)
    assert isinstance(res, LinearizedQPLadder) and res.n_accepted >= 3
    assert isinstance(res.last, LinearizedQPResult) and res.last.n_fit == res.n_fit_history[-1]
    assert res.Zqp_history.shape == (res.n_accepted,) + res.last.Zqp_ska.shape
    np.testing.assert_allclose(res.Zqp_history[-1], res.last.Zqp_ska)
    # the exact-Z comparison needs the analytic A, B and lives with the NumPy reference
    # (implementation_notes/quasiparticle_Z_at_arbitrary_k/test_linearized_qp_numpy.py)
    # err_fit is one number per (s,k): a unitary invariant of the last change of Z
    sk = res.last.Zqp_ska.shape[:2]
    assert res.err_fit_sk.shape == sk and np.all(res.err_fit_sk >= 0) and res.err_fit_sk.max() < 1e-4
    assert res.dHqp_sk.shape == sk and np.all(res.dHqp_sk >= 0)
    assert np.abs(res.last.sumrule_sk).max() < 1e-12


def test_ladder_is_capped_by_the_sampling_mesh():
    """The kernel selects Sigma(i omega) off the sampling mesh rather than interpolating onto
    arbitrary Matsubara indices, so the ladder cannot climb past the contiguous run of low odd
    frequencies (12 for DLR here). It must stop there and say so, not walk off the mesh."""
    iaft = _iaft(); n = 3
    St = iaft.w_to_tau(_poles_near_window(iaft, seed=33), "fermion")
    res = linearized_qp_from_arrays(_random_hermitian(n, 34)[None, None], St[:, None, None], 0.0,
                                    iaft, converge=True, n_fit_max=60, fit_resid_tol=1e-9)
    nf_mesh = res.n_fit_mesh_max
    assert nf_mesh == 12                                  # DLR, beta=100, wmax=40, prec="high"
    assert res.mesh_limited and res.stopped_n_fit == 0    # ended at the mesh, not at the gate
    assert res.last.n_fit == nf_mesh and res.n_accepted >= 2


def test_single_fit_beyond_the_mesh_raises():
    iaft = _iaft(); n = 3
    e, Vp = _hermitian_poles(n, 2, 3); St = _sigma_tau(iaft, e, Vp)
    with pytest.raises(ValueError, match="out of reach"):
        linearized_qp_from_arrays(_random_hermitian(n, 4)[None, None], St[:, None, None], 0.0,
                                  iaft, converge=False, n_fit=40)


def test_exact_degeneracy_with_scalar_Z_is_unambiguous():
    """Two exactly degenerate H_QP levels with a scalar residue: Z is proportional to I, so the
    per-level <v|Z|v> do not depend on which basis of the degenerate pair the eigensolver
    returned (Schur). No blocking is attempted by the kernel; this pins that the common
    symmetric case needs none."""
    iaft = _iaft(); n = 4
    e, g = 7.0, 3.0
    St = _sigma_tau(iaft, [e], [g * np.eye(n, dtype=np.complex128)])
    U = _random_unitary(n, 41)
    F = U @ np.diag([-1.0, 0.5, 0.5, 2.0]) @ U.conj().T                # exact degeneracy
    F = 0.5 * (F + F.conj().T)
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft,
                                    converge=False, n_fit=8)
    z = 1.0 / (1.0 + g / e**2)
    np.testing.assert_allclose(res.Zqp_ska[0, 0], np.full(n, z), atol=1e-10)
    assert abs(res.sumrule_sk[0, 0]) < 1e-12


# ---------------------------------------------------------------------------
# linearized_qp_from_checkpoint round trip (source="mesh")
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_mesh():
    """Diagonal single-pole self-energies in a synthetic KS-basis checkpoint: with
    B diagonal, Z_qp equals the analytic scalar values and diag(Z) equals Z_qp."""
    iaft = _iaft()
    nw, nt = len(iaft.wn_mesh("fermion")), len(iaft.tau_mesh("fermion"))
    params = [(10.0, 3.0), (12.0, 5.0)]
    ns, nk, norb = 1, 1, len(params)
    Sigma_w = np.zeros((nw, ns, nk, norb, norb), dtype=np.complex128)
    for a, (eps, g) in enumerate(params):
        Sigma_w[:, 0, 0, a, a] = _single_pole(iaft, eps, g)
    Sigma_t = iaft.w_to_tau(Sigma_w, "fermion"); assert Sigma_t.shape[0] == nt
    H0 = np.zeros((ns, nk, norb, norb), dtype=np.complex128); H0[0, 0] = np.diag([0.3, 0.7])
    Fst = np.zeros_like(H0); Fst[0, 0] = np.diag([-0.1, 0.05])
    Z_expected = np.array([_Z_exact(eps, g) for eps, g in params])

    with tempfile.TemporaryDirectory() as tmpdir:
        chkpt = os.path.join(tmpdir, "synthetic.mbpt.h5")
        with HDFArchive(chkpt, "w") as ar:
            iaft.save(ar)
            ar["system"] = {"H0_skij": H0}
            ar["scf"] = {"final_iter": 0, "iter0": {"Sigma_tskij": Sigma_t, "F_skij": Fst, "mu": 0.2}}
        res = linearized_qp_from_checkpoint(chkpt, source="mesh", converge=False, n_fit=6, fit_order=5)
    assert isinstance(res, LinearizedQPResult) and res.Zqp_ska.shape == (ns, nk, norb)
    assert res.mu == 0.2 and res.n_fit == 6 and res.fit_order == 5
    np.testing.assert_allclose(np.sort(res.Zqp_ska[0, 0]), np.sort(Z_expected), rtol=1e-5)
    np.testing.assert_allclose(np.diag(res.Z_skab[0, 0]).real, Z_expected, rtol=1e-5)
    # K includes H0: static levels are H0 + F + A - mu
    A_ex = np.diag([-g**2 / eps for eps, g in params])
    np.testing.assert_allclose(_K(res)[0, 0], H0[0, 0] + Fst[0, 0] + A_ex - 0.2 * np.eye(norb), atol=1e-5)


def test_checkpoint_bad_source_raises():
    with pytest.raises((ValueError, KeyError, FileNotFoundError, RuntimeError, OSError)):
        linearized_qp_from_checkpoint("/nonexistent.h5", source="mesh")


# ---------------------------------------------------------------------------
# failure reporting
# ---------------------------------------------------------------------------

def test_residual_gate_is_reported_in_status():
    """An exactly-determined fit whose residual exceeds fit_resid_tol is not a solution: the kernel
    leaves the point zero with status_sk == 1 and the Python layer warns. A zero tolerance
    trips the gate on any data (the residual is ~1e-15, never exactly 0)."""
    iaft = _iaft(); n = 3
    e, Vp = _hermitian_poles(n, 2, 51); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 52)
    with pytest.warns(RuntimeWarning, match="residual gate"):
        res = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft,
                                        converge=False, n_fit=4, fit_resid_tol=0.0)
    assert res.status_sk[0, 0] == 1 and (res.Zqp_ska == 0.0).all()
    assert 0.0 < res.resid_sk[0, 0] < 1e-12         # the diagnostics of the fit are still filled
    ok = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft,
                                   converge=False, n_fit=4)
    assert ok.status_sk[0, 0] == 0 and ok.resid_sk[0, 0] == res.resid_sk[0, 0]


def test_non_positive_definite_is_reported_in_status():
    # a non-causal (positive) slope makes I - B indefinite: status_sk == 2, zero entries, a warning
    iaft = _iaft(); n = 2
    iw = _iw(iaft)
    S = np.zeros((len(iw), n, n), dtype=np.complex128)
    S[:, 0, 0] = -27.0 / (iw - 4.0)   # weight -27 at e = 4: B = -V/e^2 = +27/16 > 1 -> 1 - B < 0
    St = iaft.w_to_tau(S, "fermion")
    F = np.zeros((n, n), dtype=np.complex128)
    with pytest.warns(RuntimeWarning, match="not positive definite"):
        b = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft,
                                      converge=False, n_fit=4)
    assert b.status_sk[0, 0] == 2 and np.count_nonzero(b.status_sk == 2) == 1
    assert b.min_eig_sk[0, 0] < 0.0
    assert (b.Zqp_ska == 0.0).all() and (b.Z_skab == 0.0).all() and (b.E_ska == 0.0).all()


