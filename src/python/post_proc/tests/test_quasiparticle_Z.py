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
pole weights (post_proc.linearized_qp: matsubara_low_freq_coefficients,
linearized_qp_from_arrays, linearized_qp_from_checkpoint).

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
    matsubara_low_freq_coefficients,
    linearized_qp_from_arrays,
    linearized_qp_from_checkpoint,
    LinearizedQPResult,
)
from coqui.post_proc.linearized_qp import _linearized_qp_matrix, _project_blockwise


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


def _Z_scalar(coeffs):
    """Z from the slope of a stack of 1x1 'matrices', shape coeffs[1][..., 0, 0]."""
    return 1.0 / (1.0 - coeffs[1][..., 0, 0].real)


# ---------------------------------------------------------------------------
# the fit kernel on scalar / diagonal input (ported from the earlier scalar API)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("eps, g", [(10.0, 3.0), (-8.0, 2.0), (12.0, 5.0)])
def test_single_pole_high_order_is_exact(eps, g):
    iaft = _iaft()
    S = _single_pole(iaft, eps, g)[:, None, None]
    c, d = matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=5)
    assert float(_Z_scalar(c)) == pytest.approx(_Z_exact(eps, g), rel=1e-6)
    assert d["exact"] is False and d["resid"] < 1e-6


def test_fit_order_convergence():
    iaft = _iaft(beta=50.0, wmax=30.0)
    S = _single_pole(iaft, 8.0, 2.0)[:, None, None]
    Zx = _Z_exact(8.0, 2.0)
    err1 = abs(float(_Z_scalar(matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=1)[0])) - Zx)
    err3 = abs(float(_Z_scalar(matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=3)[0])) - Zx)
    assert err3 < err1 and err3 < 1e-4


def test_zero_self_energy_gives_unit_Z():
    iaft = _iaft(beta=50.0, wmax=30.0)
    S = np.zeros((len(iaft.wn_mesh("fermion")), 1, 1), dtype=np.complex128)
    assert float(_Z_scalar(matsubara_low_freq_coefficients(S, iaft)[0])) == pytest.approx(1.0, abs=1e-12)


def test_trailing_axes_are_independent():
    iaft = _iaft()
    params = [(10.0, 3.0), (-8.0, 2.0), (12.0, 5.0)]
    S = np.stack([_single_pole(iaft, e, g) for e, g in params], axis=-1)[..., None, None]
    Z = _Z_scalar(matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=5)[0])
    np.testing.assert_allclose(Z, [_Z_exact(e, g) for e, g in params], rtol=1e-6)
    for i, (e, g) in enumerate(params):
        Zi = _Z_scalar(matsubara_low_freq_coefficients(S[:, i], iaft, n_fit=6, fit_order=5)[0])
        assert float(Zi) == pytest.approx(Z[i], rel=1e-12)


@pytest.mark.parametrize("kwargs", [{"fit_order": 0}, {"n_fit": 1, "fit_order": 5}])
def test_invalid_fit_parameters_raise(kwargs):
    iaft = _iaft(beta=50.0, wmax=30.0)
    S = _single_pole(iaft, 8.0, 2.0)[:, None, None]
    with pytest.raises(ValueError):
        matsubara_low_freq_coefficients(S, iaft, **kwargs)


# ---------------------------------------------------------------------------
# the fit kernel on matrices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matrix_pole_A_and_B(seed):
    iaft = _iaft()
    e, V = _hermitian_poles(4, 3, seed)
    c, _ = matsubara_low_freq_coefficients(_sigma_w(iaft, e, V), iaft, n_fit=8, fit_order=5)
    A_ex, B_ex = _exact_AB(e, V)
    assert np.abs(c[0] - A_ex).max() < 1e-6 * np.abs(A_ex).max()
    assert np.abs(c[1] - B_ex).max() < 1e-5 * np.abs(B_ex).max()


def test_exact_A_is_hermitian_and_complex_offdiagonal():
    e, V = _hermitian_poles(4, 3, 0)
    A_ex, _ = _exact_AB(e, V)
    assert np.abs(A_ex - A_ex.conj().T).max() < 1e-12
    assert np.abs((A_ex - np.diag(np.diag(A_ex))).imag).max() > 0.1


def test_symmetric_window_makes_coefficients_hermitian():
    iaft = _iaft()
    e, V = _hermitian_poles(4, 3, 3)
    S = _sigma_w(iaft, e, V)
    _, d_sym = matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=3, symmetric_window=True)
    _, d_pos = matsubara_low_freq_coefficients(S, iaft, n_fit=6, fit_order=3, symmetric_window=False)
    assert d_sym["anti_herm"][0] < 1e-12 * d_sym["scale"][0]
    assert d_pos["anti_herm"][0] > 1e3 * d_sym["anti_herm"][0]


def test_re_im_split_would_be_wrong_off_diagonal():
    iaft = _iaft()
    e, V = _hermitian_poles(3, 3, 5)
    S = _sigma_w(iaft, e, V)
    A_ex, _ = _exact_AB(e, V)
    n_pos = np.arange(1, 13, 2)
    Slow = iaft.w_interpolate(S, n_pos, "fermion")
    w = n_pos * np.pi / iaft.beta
    design = w[:, None] ** np.array([0, 2, 4])[None, :]
    c, *_ = np.linalg.lstsq(design, Slow.real.reshape(len(w), -1), rcond=None)
    A_wrong = c[0].reshape(A_ex.shape).astype(complex)
    assert np.abs(np.diag(A_wrong) - np.diag(A_ex)).max() < 1e-5 * np.abs(A_ex).max()
    off = (A_wrong - A_ex) - np.diag(np.diag(A_wrong - A_ex))
    assert np.abs(off).max() > 0.05 * np.abs(A_ex).max()


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


def test_exact_interpolation_flags_loss_of_conditioning():
    iaft = _iaft()
    S = _poles_near_window(iaft)
    with pytest.warns(RuntimeWarning):
        matsubara_low_freq_coefficients(S, iaft, n_fit=30, exact_resid_tol=1e-8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                    # far below the wall: no warning
        matsubara_low_freq_coefficients(S, iaft, n_fit=10, exact_resid_tol=1e-8)


# ---------------------------------------------------------------------------
# the linearized QP problem at one k (private, but the algebra must be right)
# ---------------------------------------------------------------------------

def test_generalized_eigenproblem_is_solved():
    n = 6
    K = _random_hermitian(n, 1); B = _causal_B(n, 2)
    r = _linearized_qp_matrix(K, B, 1e-8)
    eye = np.eye(n)
    w, U = np.linalg.eigh(eye - B); Zhalf = (U * (1 / np.sqrt(w))) @ U.conj().T
    for l in range(n):
        psi = Zhalf @ r["V"][:, l]
        assert np.linalg.norm(K @ psi - r["E"][l] * (eye - B) @ psi) < 1e-11 * np.linalg.norm(K)
    np.testing.assert_allclose(r["V"].conj().T @ r["V"], eye, atol=1e-12)


def test_sum_rule_and_range():
    n = 7
    r = _linearized_qp_matrix(_random_hermitian(n, 5), _causal_B(n, 6), 1e-8)
    assert abs(r["sumrule"]) < 1e-12
    assert np.all(r["Zqp"] > 0) and np.all(r["Zqp"] <= 1.0 + 1e-12)


def test_scalar_B_makes_Z_uniform_and_scales_energies():
    n = 5
    K = _random_hermitian(n, 7); b = -0.4
    r = _linearized_qp_matrix(K, b * np.eye(n), 1e-8)
    np.testing.assert_allclose(r["Zqp"], 1.0 / (1.0 - b), atol=1e-12)
    np.testing.assert_allclose(r["E"], np.linalg.eigvalsh(K) / (1.0 - b), atol=1e-12)


def test_not_positive_definite_raises():
    with pytest.raises(ValueError):
        _linearized_qp_matrix(_random_hermitian(4, 12), np.diag([-0.3, -0.2, 1.5, -0.1]).astype(complex), 1e-8)


def test_gauge_invariance_nondegenerate():
    n = 5
    K = _random_hermitian(n, 13); B = _causal_B(n, 14); U = _random_unitary(n, 15)
    r0 = _linearized_qp_matrix(K, B, 1e-8)
    r1 = _linearized_qp_matrix(U.conj().T @ K @ U, U.conj().T @ B @ U, 1e-8)
    np.testing.assert_allclose(r1["E"], r0["E"], atol=1e-11)
    np.testing.assert_allclose(r1["Zqp"], r0["Zqp"], atol=1e-11)


def test_gauge_invariance_with_degenerate_Hqp_block():
    """Build backwards so H_QP has an exact triplet while Z is not proportional to
    the identity on it: raw diagonals are basis dependent, block values are not."""
    n = 5
    Zmat = _random_hermitian(n, 16, 0.05) + np.eye(n)
    w, U = np.linalg.eigh(Zmat); assert w.min() > 0
    Zinv = (U * (1 / w)) @ U.conj().T; Zmhalf = (U * (1 / np.sqrt(w))) @ U.conj().T
    Hqp = np.diag([0.0, 0.0, 0.0, 1.0, 2.0]).astype(complex)
    K = Zmhalf @ Hqp @ Zmhalf; B = np.eye(n) - Zinv
    r0 = _linearized_qp_matrix(K, B, 1e-8)
    assert np.all(r0["count"][:3] == 3) and r0["spread"][:3].max() > 1e-3
    G = _random_unitary(n, 17)
    r1 = _linearized_qp_matrix(G.conj().T @ K @ G, G.conj().T @ B @ G, 1e-8)
    np.testing.assert_allclose(np.sort(r1["Zqp"]), np.sort(r0["Zqp"]), atol=1e-10)
    assert np.abs(np.sort(np.diag(r1["Zqp_mat"]).real[:3]) - np.sort(np.diag(r0["Zqp_mat"]).real[:3])).max() > 1e-6


def test_project_blockwise_reduces_to_diagonal_when_nondegenerate():
    n = 4
    c = _random_unitary(n, 18); M = _random_hermitian(n, 19)
    d, spread, count, Mqp = _project_blockwise(np.arange(n, dtype=float), c, M, 1e-8)
    np.testing.assert_allclose(d, np.diag(Mqp).real, atol=1e-12)
    assert np.all(count == 1) and np.all(spread == 0.0)


# ---------------------------------------------------------------------------
# linearized_qp_from_arrays: basis covariance, projection (non-)commutation, ladder
# ---------------------------------------------------------------------------

def _sigma_tau(iaft, e, V):
    return iaft.w_to_tau(_sigma_w(iaft, e, V), "fermion")


def test_from_arrays_single_fit_matches_matrix_solver():
    iaft = _iaft(); n = 5
    e, Vp = _hermitian_poles(n, 3, 21); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 22); mu = 0.1
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], mu, iaft, converge=False, n_fit=8)
    assert isinstance(res, LinearizedQPResult)
    c, _ = matsubara_low_freq_coefficients(iaft.tau_to_w(St, "fermion"), iaft, n_fit=8)
    r = _linearized_qp_matrix(F + c[0] - mu * np.eye(n), c[1], 1e-8)
    np.testing.assert_allclose(res.E_qp[0, 0], r["E"], atol=1e-12)
    np.testing.assert_allclose(res.Z_qp[0, 0], r["Zqp"], atol=1e-12)
    np.testing.assert_allclose(res.Z[0, 0], r["Zmat"], atol=1e-12)
    assert res.err_fit is None and res.err_basis is not None


def test_from_arrays_is_basis_covariant():
    iaft = _iaft(); n = 5
    e, Vp = _hermitian_poles(n, 3, 23); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 24); U = _random_unitary(n, 25)
    r0 = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft, converge=False, n_fit=6)
    r1 = linearized_qp_from_arrays((U.conj().T @ F @ U)[None, None],
                                   np.einsum("ab,tbc,cd->tad", U.conj().T, St, U)[:, None, None],
                                   0.0, iaft, converge=False, n_fit=6)
    np.testing.assert_allclose(r1.E_qp, r0.E_qp, atol=1e-11)
    np.testing.assert_allclose(r1.Z_qp, r0.Z_qp, atol=1e-11)
    np.testing.assert_allclose(r1.Z[0, 0], U.conj().T @ r0.Z[0, 0] @ U, atol=1e-11)


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
    np.testing.assert_allclose(sub.K[0, 0], proj(full.K[0, 0]), atol=1e-11)
    np.testing.assert_allclose(sub.B[0, 0], proj(full.B[0, 0]), atol=1e-11)
    assert np.abs(sub.Z[0, 0] - proj(full.Z[0, 0])).max() > 1e-3
    assert min(np.abs(full.E_qp[0, 0][:, None] - sub.E_qp[0, 0][None, :]).min(axis=0)) > 1e-4


def test_ladder_converges_and_reports_uncertainties():
    iaft = _iaft(); n = 4
    e, Vp = _hermitian_poles(n, 3, 31); St = _sigma_tau(iaft, e, Vp)
    F = _random_hermitian(n, 32)
    res = linearized_qp_from_arrays(F[None, None], St[:, None, None], 0.0, iaft, converge=True, n_fit_max=10)
    assert res.diagnostics["converged"] and len(res.ladder) >= 3
    # rungs approach the analytic Z = <v|(I-B)^-1|v> built from the exact B
    A_ex, B_ex = _exact_AB(e, Vp)
    r_ex = _linearized_qp_matrix(F + A_ex, B_ex, 1e-8)
    assert np.abs(np.sort(res.Z_qp[0, 0]) - np.sort(r_ex["Zqp"])).max() < 1e-5
    assert res.err_fit.shape == res.Z_qp.shape and np.all(res.err_fit >= 0) and res.err_fit.max() < 1e-4
    assert res.err_basis.shape == res.Z_qp.shape and np.all(np.isfinite(res.err_basis))
    assert abs(res.diagnostics["sumrule_violation"]) < 1e-12


def test_ladder_stops_at_conditioning_wall():
    iaft = _iaft(); n = 3
    St = iaft.w_to_tau(_poles_near_window(iaft, seed=33), "fermion")
    res = linearized_qp_from_arrays(_random_hermitian(n, 34)[None, None], St[:, None, None], 0.0, iaft,
                                    converge=True, n_fit_max=60, exact_resid_tol=1e-9)
    stopped = res.diagnostics["ladder_stopped"]
    assert stopped is not None and 15 <= stopped[0] <= 25          # the wall, not n_fit_max
    assert res.diagnostics["n_fit"] == stopped[0] - 1
    assert res.diagnostics["fit_resid"] <= 1e-9 and res.diagnostics["converged"]


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
    assert res.basis == "KS" and res.Z_qp.shape == (ns, nk, norb)
    np.testing.assert_allclose(np.sort(res.Z_qp[0, 0]), np.sort(Z_expected), rtol=1e-5)
    np.testing.assert_allclose(np.diag(res.Z[0, 0]).real, Z_expected, rtol=1e-5)
    # K includes H0: static levels are H0 + F + A - mu
    A_ex = np.diag([-g**2 / eps for eps, g in params])
    np.testing.assert_allclose(res.K[0, 0], H0[0, 0] + Fst[0, 0] + A_ex - 0.2 * np.eye(norb), atol=1e-5)


def test_checkpoint_bad_source_raises():
    with pytest.raises((ValueError, KeyError, FileNotFoundError, RuntimeError, OSError)):
        linearized_qp_from_checkpoint("/nonexistent.h5", source="mesh")
