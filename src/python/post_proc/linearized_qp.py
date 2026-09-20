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
linearized_qp_from_arrays         K = F + A - mu, Z = (I - B)^-1, H_QP = Z^1/2 K Z^1/2,
                                  pole weights <v|Z|v>; any basis, any k-set; by default the
                                  exactly-determined ladder with its truncation-error estimate
linearized_qp_from_checkpoint     the same from a CoQui checkpoint (KS basis on the
                                  IBZ mesh, or MLWF basis on an interpolated k-path)
LinearizedQPResult                methods::lqp::result_t, the kernel's own result of one fit
LinearizedQPLadder                methods::lqp::ladder_result_t, the ladder's result; its
                                  ``last`` is a LinearizedQPResult

The two result classes ARE the C++ structs (post_proc_module.ResultT / LadderResultT, wrapped
by clair): every public member of the struct is an attribute with its C++ name and array
layout, and the class docstring is the struct's comment. Arrays are views of the C++ storage
and the instances take no new attributes. The members, per (spin, k) in the INPUT basis:

  LinearizedQPResult   Z_skab, Hqp_skab, V_skab   (ns, nk, n, n)  Z = (I-B)^-1, H_QP, its eigenvectors
                       E_ska, Zqp_ska             (ns, nk, n)     H_QP eigenvalues, absolute [Ha]; <v|Z|v>
                       min_eig_sk, resid_sk, anti_herm_A_sk, anti_herm_B_sk, herm_data_sk,
                       sumrule_sk                 (ns, nk)        diagnostics
                       status_sk                  (ns, nk)        0 ok, 1 residual gate, 2 I-B not > 0
                       mu, n_fit, fit_order, cond scalars
  LinearizedQPLadder   last, err_fit_sk, dHqp_sk, n_fit_history, resid_history, Zqp_history,
                       n_accepted, mesh_limited, n_fit_mesh_max, stopped_n_fit/resid/cond

A point with nonzero ``status_sk`` has zero entries; this module warns about such points and
leaves them as the kernel reports them. Energies are absolute, like every CoQui energy; the
linearization is about z = mu and ``res.mu`` is stored, so ``E_ska - mu`` is the Fermi-relative
spectrum. Derived matrices (K = Z^-1/2 (H_QP - mu) Z^-1/2, B = I - Z^-1) are one line of NumPy
and are not stored; the note's ``linearized_qp_numpy.py`` has them.

Everything numerical -- the fit, the matrix problem, the ladder and its error estimate -- is
the CoQui C++ kernel (methods::lqp through pproc_t::linearized_qp{,_ladder}). This module is
the checkpoint plumbing around one call into it. The independent NumPy implementation the
kernel is validated against, and the research diagnostics built on it (err_basis), live with
the note: implementation_notes/quasiparticle_Z_at_arbitrary_k/linearized_qp_numpy.py.

Derivation, conventions, traps and the measurements behind every default:
implementation_notes/quasiparticle_Z_at_arbitrary_k/qp_Z_at_arbitrary_k_note.tex
"""

import warnings

import numpy as np
from coqui import IAFT
from coqui._lib.post_proc_module import (
    ResultT as LinearizedQPResult,
    LadderResultT as LinearizedQPLadder,
    linearized_qp_solve as _solve,
    linearized_qp_ladder as _ladder,
)

__all__ = [
    "linearized_qp_from_arrays",
    "linearized_qp_from_checkpoint",
    "LinearizedQPResult",
    "LinearizedQPLadder",
]

_HA = 27.211386245988


def linearized_qp_from_arrays(F_sk, Sigma_tsk, mu, iaft, *, converge=True, n_fit=None,
                              n_fit_max=20, fit_order=None, symmetric_window=True,
                              fit_resid_tol=1e-8, verbose=False):
    r"""Linearized quasiparticle approximation for a dynamic self-energy given as
    arrays in ANY orthonormal basis, on any set of k-points.

    Expands Sigma_dyn about zero frequency (A = value, B = slope), forms
    K = F + A - mu and Z = (I - B)^-1, solves the linearized Dyson poles
    K psi = eps (I - B) psi as the Hermitian problem Z^1/2 K Z^1/2, and returns
    the pole weights <v_l|Z|v_l> together with Z. Everything is done in the
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
        ``True`` (default): the exactly-determined ladder ``n_fit = 2 .. n_fit_max``
        (each rung interpolates 2*n_fit symmetric nodes), stopping at the first rung
        whose residual exceeds ``fit_resid_tol`` or at the end of the sampling mesh;
        returns a :class:`LinearizedQPLadder` whose ``last`` is the last accepted rung
        and whose ``err_fit_sk`` / ``dHqp_sk`` are the truncation-error estimates.
        ``False``: one fit with ``n_fit`` (default 3) and ``fit_order``; returns a
        :class:`LinearizedQPResult`. The residual gate ``fit_resid_tol`` applies to an
        exactly-determined fit in both modes (``status_sk == 1`` where it fails).
    n_fit, n_fit_max, fit_order, symmetric_window, fit_resid_tol
        Controls of the polynomial fit of ``Sigma(i omega)`` -- the same ones the
        ``lqp`` input block of ``run_lqsgw`` and ``band_interpolation`` carries.
    Notes
    -----
    The kernel selects Matsubara rows off the IAFT sampling mesh rather than
    interpolating, so ``n_fit`` cannot exceed the mesh's contiguous run of low odd
    frequencies (12 for DLR, 13 for IR at beta=100, wmax=40, prec="high"). The ladder
    stops there and reports ``mesh_limited``; a single fit beyond it raises.

    A point whose fit fails the residual gate or whose ``I - B`` is not positive definite
    is reported in ``status_sk`` (1 / 2) with zero entries, and a RuntimeWarning names it.

    Returns
    -------
    LinearizedQPLadder (``converge=True``) or LinearizedQPResult (``converge=False``)
    """
    F_sk = np.ascontiguousarray(F_sk, dtype=np.complex128)
    Sigma_tsk = np.ascontiguousarray(Sigma_tsk, dtype=np.complex128)
    if Sigma_tsk.shape[1:] != F_sk.shape:
        raise ValueError(f"linearized_qp_from_arrays: Sigma_tsk {Sigma_tsk.shape} and "
                         f"F_sk {F_sk.shape} are inconsistent.")
    iaft.check_leakage(Sigma_tsk, "fermion", name="Sigma(tau)")
    ia = (float(iaft.beta), float(iaft.wmax), str(iaft.basis), str(iaft.prec))

    if converge:
        out = _ladder(F_sk, Sigma_tsk, float(mu), *ia, int(n_fit_max), float(fit_resid_tol),
                      bool(symmetric_window))
        if out.n_accepted == 0:
            raise RuntimeError("linearized_qp_from_arrays: even n_fit=2 fails the interpolation "
                               f"check (residual {out.stopped_resid:.2e} > {fit_resid_tol:.0e}); "
                               "the input is not usable.")
        res = out.last
    else:
        n_fit_used = 3 if n_fit is None else int(n_fit)
        nf_mesh = _max_n_fit_on_mesh(iaft, symmetric_window)
        if n_fit_used > nf_mesh:
            raise ValueError(f"linearized_qp_from_arrays: n_fit={n_fit_used} is out of reach -- the "
                             f"fit window must lie on the IAFT sampling mesh, which carries the "
                             f"lowest {nf_mesh} odd frequencies for this IAFT{ia}.")
        out = res = _solve(F_sk, Sigma_tsk, float(mu), *ia, n_fit_used,
                           -1 if fit_order is None else int(fit_order),
                           bool(symmetric_window), float(fit_resid_tol))

    _warn_failed_points(res)
    if verbose:
        _report(out, iaft)
    return out


def linearized_qp_from_checkpoint(coqui_h5, h5_grp="scf", iteration=-1, *, source="mesh",
                                  **kwargs):
    r"""Linearized quasiparticle approximation from a CoQui checkpoint.

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
        approximation).
    **kwargs
        Forwarded to :func:`linearized_qp_from_arrays`.

    Returns
    -------
    LinearizedQPLadder (default) or LinearizedQPResult (``converge=False``), as
    :func:`linearized_qp_from_arrays` returns it. Axis 1 of every array is the k-point
    axis of the source: the IBZ mesh (``system/kpoints_crys`` through ``kp_to_ibz``) or
    the interpolated path (``wannier_inter/{kpts, kpt_labels, kpt_label_idx}``); read those
    from the checkpoint when plotting.
    """
    from h5 import HDFArchive

    with HDFArchive(coqui_h5, "r") as ar:
        if h5_grp not in ar:
            raise KeyError(f"linearized_qp_from_checkpoint: group '{h5_grp}' not found in {coqui_h5}.")
        it = ar[h5_grp]["final_iter"] if iteration == -1 else iteration
        g = _h5_walk(ar, f"{h5_grp}/iter{it}")
        mu = float(g["mu"])
        if source == "mesh":
            for ds in ("F_skij", "Sigma_tskij"):
                if ds not in g:
                    raise KeyError(f"linearized_qp_from_checkpoint: '{h5_grp}/iter{it}/{ds}' missing.")
            H0 = np.asarray(_h5_walk(ar, "system/H0_skij"), dtype=np.complex128)
            F = H0 + np.asarray(g["F_skij"], dtype=np.complex128)
            Sigma = np.asarray(g["Sigma_tskij"], dtype=np.complex128)
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
        else:
            raise ValueError(f"linearized_qp_from_checkpoint: source must be 'mesh' or 'kpath', got {source!r}.")

    iaft = IAFT.from_coqui_chkpt(coqui_h5, verbose=False)
    return linearized_qp_from_arrays(F, Sigma, mu, iaft, **kwargs)


def _max_n_fit_on_mesh(iaft, symmetric_window):
    """Largest n_fit whose window +-1, ..., +-(2 n_fit - 1) lies on the IAFT sampling mesh.

    The kernel selects rows off that mesh rather than interpolating and aborts one step beyond
    it, so a single fit is checked here first. Read from the Python IAFT directly: its mesh is
    the C++ one (pinned by utils/imag_axes_ft/tests/test_iaft.py), so no round trip is needed;
    the ladder path gets the same number back from the kernel.
    """
    wn = {int(x) for x in np.asarray(iaft.wn_mesh("fermion"))}
    nf = 0
    while True:
        n = 2 * (nf + 1) - 1
        if n not in wn or (symmetric_window and -n not in wn):
            return nf
        nf += 1


def _warn_failed_points(res):
    status = np.asarray(res.status_sk)
    for code, what in ((1, "the exactly-determined fit failed the residual gate"),
                       (2, "I - B is not positive definite")):
        pts = np.argwhere(status == code)
        if len(pts) == 0:
            continue
        s, k = (int(x) for x in pts[0])
        detail = (f"min eigenvalue {res.min_eig_sk[s, k]:.3e}" if code == 2 else
                  f"residual {res.resid_sk[s, k]:.2e}, cond {res.cond:.2e}")
        warnings.warn(f"linearized_qp_from_arrays: {what} at {len(pts)} (s,k) point(s); their "
                      f"entries are zero and status_sk == {code}. First: s={s} k={k} ({detail}).",
                      RuntimeWarning, stacklevel=3)


def _report(out, iaft):
    lad = out if isinstance(out, LinearizedQPLadder) else None
    res = lad.last if lad is not None else out
    print("=" * 76)
    print("Linearized QP approximation")
    print("=" * 76)
    if lad is not None:
        print("  n_fit  order  w_max[eV]   min Zqp    mean Zqp   max Zqp")
        for i, nf in enumerate(lad.n_fit_history):
            zq = lad.Zqp_history[i]
            wmax = (2 * nf - 1) * np.pi / iaft.beta * _HA
            print(f"  {nf:4d}  {2 * nf - 1:5d}  {wmax:8.3f}   {zq.min():.6f}   {zq.mean():.6f}"
                  f"   {zq.max():.6f}")
        if lad.stopped_n_fit > 0:
            print(f"  [STOP] n_fit={lad.stopped_n_fit} rejected: residual {lad.stopped_resid:.2e} "
                  f"(cond {lad.stopped_cond:.2e}); the square system no longer interpolates.")
        if lad.mesh_limited:
            print(f"  ladder capped at n_fit = {lad.n_fit_mesh_max} by the IAFT sampling mesh")
        print(f"  last accepted rung n_fit={res.n_fit} (order {res.fit_order})")
        if lad.n_accepted >= 2:
            print(f"  err_fit   max {lad.err_fit_sk.max():.2e}   (spectral norm of the last change "
                  f"of Z)   max |dH_QP| {lad.dHqp_sk.max():.2e} Ha")
    print(f"  min eig(I-B) {res.min_eig_sk.min():.4f}   sum rule {np.abs(res.sumrule_sk).max():.1e}   "
          f"fit residual {res.resid_sk.max():.1e}   Zqp in [{res.Zqp_ska.min():.4f}, "
          f"{res.Zqp_ska.max():.4f}]   failed points {np.count_nonzero(res.status_sk)}")
    print("=" * 76)


def _h5_walk(ar, path):
    """Open a nested HDF5 path with TRIQS h5 (its `in` tests direct children only)."""
    g = ar
    for part in path.strip("/").split("/"):
        if part not in g:
            raise KeyError(f"linearized_qp_from_checkpoint: '{path}' not found ('{part}' missing).")
        g = g[part]
    return g
