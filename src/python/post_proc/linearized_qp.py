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

Linearized quasiparticle approximation for a dynamic self-energy, and the
band-resolved quasiparticle pole weights Z^k_lambda at arbitrary k-points.

Public API
----------
matsubara_low_freq_coefficients   low-frequency expansion A + (i w) B + ... of a
                                  matrix-valued Sigma(i w) (complex fit, +/- nodes)
linearized_qp_from_arrays         K = F + A - mu, Z = (I - B)^-1, H_QP = Z^1/2 K Z^1/2,
                                  pole weights <v|Z|v>; any basis, any k-set;
                                  optional convergence ladder with uncertainties
linearized_qp_from_checkpoint     the same from a CoQui checkpoint (KS basis on the
                                  IBZ mesh, or MLWF basis on an interpolated k-path)
LinearizedQPResult                the result container

Derivation, conventions, traps and the measurements behind every default:
implementation_notes/quasiparticle_Z_at_arbitrary_k/qp_Z_at_arbitrary_k_note.tex
"""

import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from coqui import IAFT

__all__ = [
    "matsubara_low_freq_coefficients",
    "linearized_qp_from_arrays",
    "linearized_qp_from_checkpoint",
    "LinearizedQPResult",
]

_HA = 27.211386245988


def matsubara_low_freq_coefficients(Sigma_w, iaft, *, n_fit=3, fit_order=None,
                                    symmetric_window=True, exact_resid_tol=1e-8):
    r"""Expand a matrix-valued fermionic self-energy about zero frequency.

    Fits, by complex least squares on the lowest Matsubara frequencies,

        Sigma_ab(i w) = sum_p A^(p)_ab (i w)^p,   p = 0 .. fit_order,

    and returns the coefficient matrices. ``A^(0)`` is the omega -> 0 value and
    ``A^(1)`` the omega -> 0 slope (``D`` of the linearized QP problem).

    The fit is on the COMPLEX matrix, not on Re/Im separately: splitting Re
    Sigma into even and Im Sigma into odd powers is valid only for diagonal
    elements. Every A^(p) is Hermitian in exact arithmetic. With
    ``symmetric_window=True`` the nodes come in +/- pairs, which makes the fitted
    coefficients Hermitian by construction, decouples even and odd orders, and
    makes the composite tau -> (A, B) map real so that the fit commutes with
    CoQui's zeroing of the R-space imaginary part in the Wannier interpolation.

    Parameters
    ----------
    Sigma_w : np.ndarray, shape (nw, ..., n, n)
        Self-energy on the IAFT fermionic Matsubara sampling mesh; the last two
        axes are the orbital indices, anything in between is carried along.
    iaft : IAFT
        Defines the Matsubara mesh and beta.
    n_fit : int
        Number of lowest positive fermionic frequencies. With
        ``symmetric_window`` the fit uses ``2*n_fit`` nodes.
    fit_order : int or None
        Highest power of (i w). ``None`` (default) = number of nodes minus one,
        i.e. exact polynomial interpolation of the node set -- a Richardson-type
        extrapolation to omega = 0 that converges as ``n_fit`` grows. A smaller
        value is an over-determined least-squares fit whose truncation bias does
        NOT vanish with a wider window (8% on GW, 72% on GW+EDMFT LaNiO3 for
        ``n_fit=6, fit_order=3``). Must be >= 1.
    symmetric_window : bool
        Use +/- frequency pairs (default) rather than positive frequencies only.
    exact_resid_tol : float
        For an exactly-determined fit, the largest relative residual consistent
        with true interpolation; above it the design matrix has lost
        conditioning and a RuntimeWarning is issued.

    Returns
    -------
    coeffs : np.ndarray, shape (fit_order+1, ..., n, n)
        Hermitized coefficient matrices A^(p), in units of ``Sigma_w`` / Ha^p.
    diag : dict
        ``anti_herm`` (per order, before Hermitization), ``scale``, ``resid``,
        ``cond``, ``exact``, ``herm_data`` (|Sigma(iw) - Sigma(-iw)^dag| on the
        window), ``n_idx`` (CoQui Matsubara indices used).
    """
    if not isinstance(iaft, IAFT):
        raise ValueError("matsubara_low_freq_coefficients: iaft must be an IAFT instance.")
    Sigma_w = np.asarray(Sigma_w, dtype=np.complex128)
    if Sigma_w.ndim < 3 or Sigma_w.shape[-1] != Sigma_w.shape[-2]:
        raise ValueError(f"matsubara_low_freq_coefficients: expected shape "
                         f"(nw, ..., n, n), got {Sigma_w.shape}.")
    # CoQui index notation: i w_n = i n pi / beta with n odd for fermions.
    n_pos = np.arange(1, 2 * n_fit, 2)
    n_idx = np.concatenate([-n_pos[::-1], n_pos]) if symmetric_window else n_pos
    n_nodes = len(n_idx)
    if fit_order is None:
        fit_order = n_nodes - 1
    if fit_order < 1:
        raise ValueError(f"matsubara_low_freq_coefficients: fit_order must be >= 1 "
                         f"(the slope is A^(1)), got {fit_order}.")
    n_params = fit_order + 1
    if n_nodes < n_params:
        raise ValueError(f"matsubara_low_freq_coefficients: {n_nodes} fit nodes cannot "
                         f"determine {n_params} parameters; raise n_fit or lower fit_order.")

    S = iaft.w_interpolate(Sigma_w, n_idx, "fermion")            # (nnodes, ..., n, n)
    herm_data = np.nan
    if symmetric_window:
        lo = S[:n_fit][::-1]
        hi = S[n_fit:]
        herm_data = float(np.abs(hi - np.swapaxes(lo, -1, -2).conj()).max())

    # design matrix in x = n / n_max so entries are O(1); unscale afterwards
    n_max = float(np.abs(n_idx).max())
    design = (1j * (n_idx / n_max))[:, None] ** np.arange(n_params)[None, :]
    cond = float(np.linalg.cond(design))
    flat = S.reshape(n_nodes, -1)
    coeffs, *_ = np.linalg.lstsq(design, flat, rcond=None)
    resid = float(np.abs(design @ coeffs - flat).max() / max(np.abs(flat).max(), 1e-300))
    exact = (n_params == n_nodes)
    if exact and resid > exact_resid_tol:
        warnings.warn(
            f"matsubara_low_freq_coefficients: exactly-determined fit (n_fit={n_fit}, "
            f"fit_order={fit_order}) no longer interpolates: relative residual "
            f"{resid:.2e} > {exact_resid_tol:.0e}, cond(design)={cond:.2e}. The "
            f"extracted coefficients are unreliable; lower n_fit.", RuntimeWarning,
            stacklevel=2)
    omega_scale = n_max * np.pi / iaft.beta
    coeffs = coeffs / (omega_scale ** np.arange(n_params))[:, None]
    coeffs = coeffs.reshape((n_params,) + Sigma_w.shape[1:])

    anti = np.empty(n_params)
    scale = np.empty(n_params)
    for p in range(n_params):
        Ap = coeffs[p]
        ApH = np.swapaxes(Ap, -1, -2).conj()
        anti[p] = float(np.abs(0.5 * (Ap - ApH)).max())
        scale[p] = float(np.abs(Ap).max())
        coeffs[p] = 0.5 * (Ap + ApH)
    return coeffs, {"anti_herm": anti, "scale": scale, "resid": resid, "cond": cond,
                    "exact": exact, "herm_data": herm_data, "n_idx": n_idx}


# ---- private helpers --------------------------------------------------------

def _degenerate_blocks(eps, tol):
    """(start, stop) ranges of consecutive eigenvalues within ``tol``; eps ascending."""
    n = len(eps)
    i = 0
    while i < n:
        j = i + 1
        while j < n and (eps[j] - eps[j - 1]) <= tol:
            j += 1
        yield i, j
        i = j


def _project_blockwise(eps, C, M, deg_tol):
    """Diagonal of Hermitian M in the eigenbasis C (columns), resolving degeneracies.

    Inside a degenerate subspace of ``eps`` the eigenvectors are an arbitrary
    basis, so individual diagonal elements are undefined; the eigenvalues of M
    restricted to the block are returned instead (ascending). Reduces to the
    plain diagonal for a non-degenerate level.
    """
    Mqp = C.conj().T @ M @ C
    Mqp = 0.5 * (Mqp + Mqp.conj().T)
    d = np.empty(len(eps))
    spread = np.zeros(len(eps))
    ndeg = np.ones(len(eps), dtype=int)
    for i, j in _degenerate_blocks(eps, deg_tol):
        if j - i == 1:
            d[i] = Mqp[i, i].real
        else:
            ev = np.linalg.eigvalsh(Mqp[i:j, i:j])
            d[i:j] = ev
            spread[i:j] = ev.max() - ev.min()
            ndeg[i:j] = j - i
    return d, spread, ndeg, Mqp


def _basis_sensitivity(eps, Mqp, deg_warn, h_err):
    """How far each <v_l|Z|v_l> can move through eigenvector error of size h_err in H_QP.

    For neighbours m within ``deg_warn``: theta = min(1, h_err/|e_l - e_m|) and
    dZ_l += 2 theta |Z_lm| + theta^2 |Z_mm - Z_ll|. At exact degeneracy this
    bounds the full rotational ambiguity; for well-split levels it decays as
    1/gap, so distinct bands that merely lie close are not flagged.
    """
    n = len(eps)
    sens = np.zeros(n)
    for l in range(n):
        for m in range(n):
            if m == l:
                continue
            gap = abs(eps[l] - eps[m])
            if gap > deg_warn:
                continue
            theta = 1.0 if gap <= h_err else h_err / gap
            sens[l] += (2.0 * theta * abs(Mqp[l, m])
                        + theta ** 2 * abs(Mqp[m, m].real - Mqp[l, l].real))
    return sens


def _linearized_qp_matrix(K, B, deg_tol):
    """The linearized QP problem at ONE (spin, k): Z = (I-B)^-1, H_QP = Z^1/2 K Z^1/2,
    its eigenbasis, and the block-resolved pole weights <v_l|Z|v_l>.

    Raises ValueError if I - B is not positive definite. For a causal self-energy
    B is negative semidefinite, so a violation is a failed fit or a sign error.
    """
    n = K.shape[0]
    eye = np.eye(n)
    K = 0.5 * (K + K.conj().T)
    B = 0.5 * (B + B.conj().T)
    w, U = np.linalg.eigh(eye - B)
    min_eig = float(w.min())
    if min_eig <= 0.0:
        raise ValueError(f"_linearized_qp_matrix: I - B is not positive definite "
                         f"(min eigenvalue {min_eig:.3e}); the linearization is invalid.")
    Zmat = (U * (1.0 / w)) @ U.conj().T
    Zhalf = (U * (1.0 / np.sqrt(w))) @ U.conj().T
    Hqp = Zhalf @ K @ Zhalf
    Hqp = 0.5 * (Hqp + Hqp.conj().T)
    E, V = np.linalg.eigh(Hqp)
    Zqp, spread, count, Zqp_mat = _project_blockwise(E, V, Zmat, deg_tol)
    return {"E": E, "V": V, "Zmat": Zmat, "Hqp": Hqp, "Zqp": Zqp, "Zqp_mat": Zqp_mat,
            "spread": spread, "count": count, "min_eig": min_eig,
            "sumrule": float(Zqp.sum() - np.trace(Zmat).real)}


# ---- result -------------------------------------------------------------------

@dataclass
class LinearizedQPResult:
    """Solution of the linearized quasiparticle problem for a dynamic self-energy.

    All arrays are per (spin, k) in the INPUT basis unless stated:
    ``K`` = F + A - mu, ``B`` = zero-frequency slope, ``Z`` = (I - B)^-1;
    ``E_qp``, ``V`` = eigenvalues/eigenvectors of Z^1/2 K Z^1/2 (H_QP is
    V diag(E_qp) V^dag); ``Z_qp`` = <v_l|Z|v_l>, block-resolved at degeneracies.
    ``err_fit``/``err_basis`` are per-level uncertainties from the convergence
    ladder (extrapolation; eigenvector sensitivity) and are None otherwise.
    """
    K: np.ndarray
    B: np.ndarray
    Z: np.ndarray
    E_qp: np.ndarray
    V: np.ndarray
    Z_qp: np.ndarray
    mu: float
    basis: str
    diagnostics: dict
    kpts: Optional[np.ndarray] = None
    labels: str = ""
    label_idx: Optional[np.ndarray] = None
    err_fit: Optional[np.ndarray] = None
    err_basis: Optional[np.ndarray] = None
    ladder: Optional[list] = None

    def near_fermi(self, window_eV=1.5):
        """Boolean mask of levels with |E_qp| below ``window_eV``."""
        return np.abs(self.E_qp) * _HA < window_eV


def _solve_rung(F_sk, Sigma_wsk, mu, iaft, *, n_fit, fit_order, symmetric_window,
                exact_resid_tol, deg_tol, basis_label):
    """One rung of the ladder on Sigma already transformed to the Matsubara mesh."""
    ns, nk, n = F_sk.shape[0], F_sk.shape[1], F_sk.shape[2]
    coeffs, fd = matsubara_low_freq_coefficients(
        Sigma_wsk, iaft, n_fit=n_fit, fit_order=fit_order,
        symmetric_window=symmetric_window, exact_resid_tol=exact_resid_tol)
    A, B = coeffs[0], coeffs[1]
    eye = np.eye(n)
    K = F_sk + A - mu * eye
    K = 0.5 * (K + np.swapaxes(K, -1, -2).conj())

    shp = (ns, nk, n)
    E = np.full(shp, np.nan); Zqp = np.full(shp, np.nan)
    spread = np.zeros(shp); count = np.ones(shp, dtype=int)
    V = np.zeros((ns, nk, n, n), dtype=np.complex128)
    Z = np.zeros_like(V); Hqp = np.zeros_like(V); Zqp_mat = np.zeros_like(V)
    min_eig = np.full((ns, nk), np.nan); sumrule = np.full((ns, nk), np.nan)
    bad = []
    for s in range(ns):
        for k in range(nk):
            try:
                r = _linearized_qp_matrix(K[s, k], B[s, k], deg_tol)
            except ValueError as ex:
                bad.append((s, k, str(ex))); continue
            E[s, k], V[s, k], Z[s, k], Hqp[s, k] = r["E"], r["V"], r["Zmat"], r["Hqp"]
            Zqp[s, k], Zqp_mat[s, k] = r["Zqp"], r["Zqp_mat"]
            spread[s, k], count[s, k] = r["spread"], r["count"]
            min_eig[s, k], sumrule[s, k] = r["min_eig"], r["sumrule"]
    return {"K": K, "B": B, "Z": Z, "E": E, "V": V, "Hqp": Hqp, "Zqp": Zqp,
            "Zqp_mat": Zqp_mat, "spread": spread, "count": count, "min_eig": min_eig,
            "sumrule": sumrule, "bad": bad, "fit": fd, "n_fit": n_fit,
            "fit_order": coeffs.shape[0] - 1}


def linearized_qp_from_arrays(F_sk, Sigma_tsk, mu, iaft, *, converge=True, n_fit=None,
                              n_fit_max=20, fit_order=None, symmetric_window=True,
                              exact_resid_tol=1e-8, deg_tol=1e-8, deg_warn=1e-3,
                              h_err=None, basis_label="", verbose=False):
    r"""Linearized quasiparticle approximation for a dynamic self-energy given as
    arrays in ANY orthonormal basis, on any set of k-points.

    Expands Sigma_dyn about zero frequency (A = value, B = slope), forms
    K = F + A - mu and Z = (I - B)^-1, solves the linearized Dyson poles
    K psi = eps (I - B) psi as the Hermitian problem Z^1/2 K Z^1/2, and returns
    the pole weights <v_l|Z|v_l> together with K, B, Z. Everything is done in the
    basis of the input, so with Wannier-interpolated arrays this is the
    low-energy (window) approximation, and with KS-basis arrays it is exact
    within the KS space.

    Parameters
    ----------
    F_sk : (ns, nk, n, n) complex
        Static one-body matrix INCLUDING H_0 (the checkpoint's ``F_skij`` lacks
        it; ``linearized_qp_from_checkpoint`` adds ``system/H0_skij``).
    Sigma_tsk : (nt, ns, nk, n, n) complex
        Dynamic self-energy on the IAFT fermionic tau mesh, same basis as F_sk.
    mu : float
        Chemical potential [Ha].
    iaft : IAFT
    converge : bool
        Run the exactly-determined ladder ``n_fit = 2 .. n_fit_max`` (each rung
        interpolates 2*n_fit symmetric nodes) and stop at the first rung whose
        residual exceeds ``exact_resid_tol``; return the last valid rung with
        per-level uncertainties. ``False``: a single fit with ``n_fit``
        (default 3) and ``fit_order``.
    n_fit, n_fit_max, fit_order, symmetric_window, exact_resid_tol
        Fit controls; see :func:`matsubara_low_freq_coefficients`.
    deg_tol : float
        Tolerance [Ha] for treating H_QP levels as degenerate (blockwise
        projection). Keep tight: symmetry multiplets need no blocking (Schur),
        and blocking distinct close bands reassigns their Z.
    deg_warn, h_err : float
        Window [Ha] and assumed H_QP error [Ha] for ``err_basis``. ``h_err=None``
        uses the measured change of H_QP between the last two rungs (1e-5 Ha
        for a single fit).
    basis_label : str
        Free text recorded in the result.

    Returns
    -------
    LinearizedQPResult
    """
    F_sk = np.asarray(F_sk, dtype=np.complex128)
    Sigma_tsk = np.asarray(Sigma_tsk, dtype=np.complex128)
    if Sigma_tsk.shape[1:] != F_sk.shape:
        raise ValueError(f"linearized_qp_from_arrays: Sigma_tsk {Sigma_tsk.shape} and "
                         f"F_sk {F_sk.shape} are inconsistent.")
    leak = iaft.check_leakage(Sigma_tsk, "fermion", name=f"Sigma(tau) [{basis_label}]")
    Sigma_wsk = iaft.tau_to_w(Sigma_tsk, "fermion")
    common = dict(fit_order=fit_order, symmetric_window=symmetric_window,
                  exact_resid_tol=exact_resid_tol, deg_tol=deg_tol,
                  basis_label=basis_label)

    ladder = []
    stopped = None
    if converge:
        for nf in range(2, n_fit_max + 1):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                r = _solve_rung(F_sk, Sigma_wsk, mu, iaft, n_fit=nf, **{**common, "fit_order": None})
            if r["fit"]["resid"] > exact_resid_tol:
                stopped = (nf, r["fit"]["resid"], r["fit"]["cond"])
                break
            ladder.append(r)
        if not ladder:
            raise RuntimeError("linearized_qp_from_arrays: even n_fit=2 fails the "
                               "interpolation check; the input is not usable.")
    else:
        ladder.append(_solve_rung(F_sk, Sigma_wsk, mu, iaft,
                                  n_fit=3 if n_fit is None else n_fit, **common))
    r = ladder[-1]

    ns, nk, n = F_sk.shape[0], F_sk.shape[1], F_sk.shape[2]
    err_fit = err_basis = None
    if converge and len(ladder) > 1:
        rp = ladder[-2]
        dZ = r["Z"] - rp["Z"]
        # extrapolation: change of the Z MATRIX projected on the final eigenvectors
        # (label- and rotation-free); eigenvector error is accounted for separately
        err_fit = np.abs(np.einsum("skal,skab,skbl->skl", r["V"].conj(), dZ, r["V"]))
        dH = np.abs(r["Hqp"] - rp["Hqp"]).max(axis=(-1, -2))          # per (s,k)
    else:
        dH = None
    err_basis = np.zeros((ns, nk, n))
    for s in range(ns):
        for k in range(nk):
            if np.isnan(r["E"][s, k]).any():
                err_basis[s, k] = np.nan; continue
            he = h_err if h_err is not None else (dH[s, k] if dH is not None else 1e-5)
            err_basis[s, k] = _basis_sensitivity(r["E"][s, k], r["Zqp_mat"][s, k], deg_warn, he)
    if not converge:
        err_basis = err_basis if h_err is not None else err_basis   # single fit: assumed h_err
    if r["bad"]:
        warnings.warn(f"linearized_qp_from_arrays: I - B is not positive definite at "
                      f"{len(r['bad'])} (s,k) point(s); results there are NaN. First: "
                      f"s={r['bad'][0][0]} k={r['bad'][0][1]}: {r['bad'][0][2]}",
                      RuntimeWarning, stacklevel=2)

    fd = r["fit"]
    Zq = r["Zqp"]
    diagnostics = {
        "n_fit": r["n_fit"], "fit_order": r["fit_order"], "exact_fit": fd["exact"],
        "fit_resid": fd["resid"], "cond": fd["cond"], "leakage": leak,
        "data_herm_violation": fd["herm_data"],
        "anti_herm_A_rel": float(fd["anti_herm"][0] / max(fd["scale"][0], 1e-300)),
        "anti_herm_B_rel": float(fd["anti_herm"][1] / max(fd["scale"][1], 1e-300)),
        "min_eig_ImB": float(np.nanmin(r["min_eig"])), "n_not_posdef": len(r["bad"]),
        "sumrule_violation": float(np.nanmax(np.abs(r["sumrule"]))),
        "Z_min": float(np.nanmin(Zq)) if np.isfinite(Zq).any() else np.nan,
        "Z_max": float(np.nanmax(Zq)) if np.isfinite(Zq).any() else np.nan,
        "n_Z_out_of_range": int(np.count_nonzero((Zq <= 0.0) | (Zq > 1.0))),
        "deg_count_max": int(r["count"].max()), "deg_spread_max": float(r["spread"].max()),
        "n_flagged_basis": int(np.count_nonzero(err_basis > 1e-3)),
        "ladder_stopped": stopped, "converged": converge and len(ladder) > 1,
    }
    if verbose:
        _report_ladder(ladder, stopped, err_fit, err_basis, r, iaft, basis_label)

    return LinearizedQPResult(
        K=r["K"], B=r["B"], Z=r["Z"], E_qp=r["E"], V=r["V"], Z_qp=Zq, mu=mu,
        basis=basis_label, diagnostics=diagnostics, err_fit=err_fit, err_basis=err_basis,
        ladder=[(x["n_fit"], x["fit_order"], x["Zqp"]) for x in ladder] if converge else None)


def _report_ladder(ladder, stopped, err_fit, err_basis, r, iaft, basis_label):
    print("=" * 76)
    print(f"Linearized QP approximation  [{basis_label}]")
    print("=" * 76)
    if len(ladder) > 1:
        print("  n_fit  order  w_max[eV]   min Z_qp   mean Z_qp  max Z_qp   resid")
        for x in ladder:
            wmax = (2 * x["n_fit"] - 1) * np.pi / iaft.beta * _HA
            print(f"  {x['n_fit']:4d}  {x['fit_order']:5d}  {wmax:8.3f}   {np.nanmin(x['Zqp']):.6f}"
                  f"   {np.nanmean(x['Zqp']):.6f}   {np.nanmax(x['Zqp']):.6f}   {x['fit']['resid']:.1e}")
        if stopped is not None:
            print(f"  [STOP] n_fit={stopped[0]} rejected: residual {stopped[1]:.2e} "
                  f"(cond {stopped[2]:.2e}); the square system no longer interpolates.")
        print(f"  converged with n_fit={r['n_fit']} (order {r['fit_order']})")
        print(f"  err_fit   max {np.nanmax(err_fit):.2e}   (extrapolation, matrix increment on the final eigenvectors)")
    print(f"  err_basis max {np.nanmax(err_basis):.2e}   ({int(np.count_nonzero(err_basis > 1e-3))} level(s) > 1e-3: near-degenerate eigenvectors)")
    print(f"  min eig(I-B) {np.nanmin(r['min_eig']):.4f}   sum rule {np.nanmax(np.abs(r['sumrule'])):.1e}   "
          f"fit residual {r['fit']['resid']:.1e}   Z_qp in [{np.nanmin(r['Zqp']):.4f}, {np.nanmax(r['Zqp']):.4f}]")
    print("=" * 76)


def _h5_walk(ar, path):
    """Open a nested HDF5 path with TRIQS h5 (its `in` tests direct children only)."""
    g = ar
    for part in path.strip("/").split("/"):
        if part not in g:
            raise KeyError(f"linearized_qp_from_checkpoint: '{path}' not found ('{part}' missing).")
        g = g[part]
    return g


def linearized_qp_from_checkpoint(coqui_h5, h5_grp="scf", iteration=-1, *, source="mesh",
                                  **kwargs):
    r"""Linearized quasiparticle approximation from a CoQui MBPT checkpoint.

    Parameters
    ----------
    coqui_h5 : str
        Checkpoint file.
    h5_grp : str
        ``"scf"`` (GW / MBPT) or ``"embed"`` (GW+EDMFT lattice quantities).
    iteration : int
        ``-1`` = ``{h5_grp}/final_iter``.
    source : {"mesh", "kpath"}
        ``"mesh"``: KS basis on the IBZ mesh from ``system/H0_skij`` +
        ``{grp}/iter{N}/F_skij`` and ``Sigma_tskij`` (exact within the KS
        space; the LQSGW route). ``"kpath"``: MLWF basis at the interpolated
        k-points from ``{grp}/iter{N}/wannier_inter/{F_skab, Sigma_tskab}``,
        as written by :func:`spectral_interpolation` (the low-energy window
        approximation; k-points and labels are read back, not respecified).
    **kwargs
        Forwarded to :func:`linearized_qp_from_arrays`.

    Returns
    -------
    LinearizedQPResult
    """
    from h5 import HDFArchive

    with HDFArchive(coqui_h5, "r") as ar:
        if h5_grp not in ar:
            raise KeyError(f"linearized_qp_from_checkpoint: group '{h5_grp}' not found in {coqui_h5}.")
        it = ar[h5_grp]["final_iter"] if iteration == -1 else iteration
        g = _h5_walk(ar, f"{h5_grp}/iter{it}")
        mu = float(g["mu"])
        kpts = None; labels = ""; label_idx = None
        if source == "mesh":
            for ds in ("F_skij", "Sigma_tskij"):
                if ds not in g:
                    raise KeyError(f"linearized_qp_from_checkpoint: '{h5_grp}/iter{it}/{ds}' missing.")
            H0 = np.asarray(_h5_walk(ar, "system/H0_skij"), dtype=np.complex128)
            F = H0 + np.asarray(g["F_skij"], dtype=np.complex128)
            Sigma = np.asarray(g["Sigma_tskij"], dtype=np.complex128)
            sy = ar["system"]
            if "kpoints_crys" in sy and "kp_to_ibz" in sy:
                kc = np.asarray(sy["kpoints_crys"]); k2i = np.asarray(sy["kp_to_ibz"])
                kpts = np.array([kc[np.flatnonzero(k2i == i)[0]] for i in range(F.shape[1])])
            basis = "KS"
        elif source == "kpath":
            if "wannier_inter" not in g:
                raise KeyError(f"linearized_qp_from_checkpoint: '{h5_grp}/iter{it}/wannier_inter' "
                               f"missing -- run coqui.post_proc.spectral_interpolation first.")
            w = g["wannier_inter"]
            for ds in ("F_skab", "Sigma_tskab"):
                if ds not in w:
                    raise KeyError(f"linearized_qp_from_checkpoint: 'wannier_inter/{ds}' missing.")
            F = np.asarray(w["F_skab"], dtype=np.complex128)          # includes H0
            Sigma = np.asarray(w["Sigma_tskab"], dtype=np.complex128)
            kpts = np.asarray(w["kpts"]) if "kpts" in w else None
            labels = w["kpt_labels"] if "kpt_labels" in w else ""
            label_idx = np.asarray(w["kpt_label_idx"]) if "kpt_label_idx" in w else None
            basis = "Wannier (wannier_inter)"
        else:
            raise ValueError(f"linearized_qp_from_checkpoint: source must be 'mesh' or 'kpath', got {source!r}.")

    iaft = IAFT.from_coqui_chkpt(coqui_h5, verbose=False)
    kwargs.setdefault("basis_label", basis)
    res = linearized_qp_from_arrays(F, Sigma, mu, iaft, **kwargs)
    res.kpts, res.labels, res.label_idx = kpts, labels, label_idx
    return res
