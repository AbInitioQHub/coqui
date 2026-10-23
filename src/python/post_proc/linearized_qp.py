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
"""

"""
Linearized quasiparticle approximation of a dynamic self-energy: Sigma(iw) ~ A + iw B fitted at
low Matsubara frequencies, Z = (I - B)^-1, H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu and the pole weights
<v|Z|v>.

linearized_qp_from_arrays       arrays in any orthonormal basis, on any set of k-points;
                                returns LinearizedQPLadder (the converged ladder, default) or
                                LinearizedQPResult (one fit)
linearized_qp_from_checkpoint   a Dyson solution in a CoQui checkpoint; written back to it

LinearizedQPResult and LinearizedQPLadder are the C++ structs methods::lqp::result_t and
ladder_result_t, wrapped by clair: C++ member names and layouts, array attributes are views of
the C++ storage, and the class docstrings are the structs' comments. Energies are absolute
(``E_ska - mu`` is the Fermi-relative spectrum). All numerics are the C++ kernel (methods::lqp),
which aborts where the linearization fails.
"""


import numpy as np
from coqui._lib.iaft_module import IAFT as _IAFTCpp, string_to_basis_enum as _basis_enum
from coqui._lib.post_proc_module import (
    ResultT as LinearizedQPResult,
    LadderResultT as LinearizedQPLadder,
    linearized_qp_solve as _solve,
    linearized_qp_ladder as _ladder,
    linearized_qp_checkpoint as _checkpoint,
)

__all__ = [
    "linearized_qp_from_arrays",
    "linearized_qp_from_checkpoint",
    "LinearizedQPResult",
    "LinearizedQPLadder",
]


def linearized_qp_from_arrays(mpi, F_sk, Sigma_tsk, mu, iaft, *, converge=True, n_fit=6,
                              n_fit_max=20, fit_order=-1, symmetric_window=True,
                              fit_resid_tol=1e-8):
    r"""Linearized quasiparticle approximation of a dynamic self-energy given as arrays, in any
    orthonormal basis and on any set of k-points.

    Fits Sigma(iw) ~ A + iw B at the lowest Matsubara frequencies and returns, per (spin, k),
    Z = (I - B)^-1, the quasiparticle Hamiltonian H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu with its
    eigenvalues and eigenvectors, and the pole weights <v|Z|v>. Everything stays in the basis of
    the input: KS-basis arrays give the result within the KS space, Wannier-interpolated arrays
    its low-energy (window) approximation.

    Meant for small arrays: the inputs and the returned result live on every rank in Python. For
    checkpoint data use :func:`linearized_qp_from_checkpoint`, which keeps Sigma in node-shared
    memory.

    Parameters
    ----------
    mpi : MpiHandler
        MPI environment (``coqui.MpiHandler()``), required also for a single process. The (s, k)
        points are distributed over its ranks and every rank returns the complete result.
    F_sk : (ns, nk, n, n) complex
        Static one-body matrix INCLUDING H_0 (the checkpoint's ``F_skij`` lacks
        it; ``linearized_qp_from_checkpoint`` adds ``system/H0_skij``).
    Sigma_tsk : (nt, ns, nk, n, n) complex
        Dynamic self-energy on the IAFT fermionic tau mesh, same basis as F_sk.
    mu : float
        Chemical potential [Ha].
    iaft : IAFT
        The imaginary-axis transform ``Sigma_tsk`` is sampled with (its fermionic tau mesh).
        ``basis="dlr"``, or ``basis="ir"`` with ``prec`` "high", "medium" or "low"; an IR IAFT
        with a custom ``eps`` raises NotImplementedError.
    converge : bool
        ``True`` (default): the exactly-determined ladder ``n_fit = 2 .. n_fit_max``
        (each fit interpolates 2*n_fit symmetric nodes), stopping at the first fit
        whose residual exceeds ``fit_resid_tol`` or at the end of the sampling mesh;
        returns a :class:`LinearizedQPLadder` whose ``last`` is the last accepted fit
        and whose ``err_fit_sk`` / ``dHqp_sk`` are the truncation-error estimates.
        ``False``: one fit with ``n_fit`` and ``fit_order``; returns a
        :class:`LinearizedQPResult`. The residual gate ``fit_resid_tol`` applies to an
        exactly-determined fit in both modes.
    n_fit, fit_order
        The single fit (``converge=False``): ``n_fit`` lowest positive fermionic Matsubara
        frequencies (default 6) and the highest power of ``i omega`` (default -1, the
        exactly-determined fit). Not used by the ladder, which sets both itself.
    n_fit_max, symmetric_window, fit_resid_tol
        The ladder's highest ``n_fit`` (default 20, capped by the sampling mesh), +-omega
        pairs (default True) and the residual gate (default 1e-8).

    The defaults of ``n_fit``, ``fit_order``, ``symmetric_window`` and ``fit_resid_tol`` are
    those of the ``lqp`` input block of ``run_lqsgw`` and ``band_interpolation``.

    Notes
    -----
    The C++ kernel logs the fit at output level 1 (:func:`coqui.set_verbosity`): a table of
    ``n_fit``, fit order, highest node, fit residual and the range of the pole weights -- one
    row per accepted fit for the ladder, one row for a single fit -- then a line with
    ``min eig(1 - B)``, the sum rule and, for the ladder, the truncation-error estimates.

    The fit window lies on the IAFT sampling mesh, so ``n_fit`` cannot exceed the mesh's
    contiguous run of low odd Matsubara frequencies (12 for DLR, 13 for IR at beta=100,
    wmax=40, prec="high"). The ladder stops there (``n_fit_cap < n_fit_max``,
    ``stopped_resid == 0``). 

    The program aborts where the linearization fails: a single fit that fail the 
    residual gate or has ``I - B`` not positive definite, or a ladder that accept not fit. 
    It also aborts on a single fit whose window does not lie on the sampling mesh.

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
    ft = _cpp_iaft(iaft)

    if converge:
        return _ladder(mpi, F_sk, Sigma_tsk, float(mu), ft, int(n_fit_max), float(fit_resid_tol),
                       bool(symmetric_window))
    return _solve(mpi, F_sk, Sigma_tsk, float(mu), ft, int(n_fit), int(fit_order),
                  bool(symmetric_window), float(fit_resid_tol))


def linearized_qp_from_checkpoint(mpi, coqui_h5, grp_name="scf", iteration=-1, *, source="mesh",
                                  converge=True, n_fit=6, n_fit_max=20, fit_order=-1,
                                  symmetric_window=True, fit_resid_tol=1e-8):
    r"""Linearized quasiparticle approximation to a Dyson-SCF solution stored in a CoQui checkpoint.  

    Parameters
    ----------
    mpi : MpiHandler
        MPI environment (``coqui.MpiHandler()``), required also for a single process. The (s, k)
        points are distributed over its ranks.
    coqui_h5 : str
        Checkpoint file; read, then appended to.
    grp_name : str
        ``"scf"`` (GW / MBPT) or ``"embed"`` (GW+EDMFT lattice quantities).
    iteration : int
        ``-1`` = ``{grp_name}/final_iter``.
    source : {"mesh", "interpolated"}
        ``"mesh"``: the Dyson solution on the k-points the SCF calculation ran on (the IBZ of its
        Monkhorst-Pack grid), in the KS basis: ``{grp}/iter{N}/{F_skij, Sigma_tskij}`` with
        ``system/H0_skij`` added to F. ``"interpolated"``: the Wannier-interpolated Dyson solution
        ``{grp}/iter{N}/wannier_inter/{F_skab, Sigma_tskab}`` written by
        :func:`spectral_interpolation`, in the Wannier basis on its k-points (the low-energy
        window approximation).
    converge, n_fit, n_fit_max, fit_order, symmetric_window, fit_resid_tol
        As in :func:`linearized_qp_from_arrays`: ``converge=True`` (default) climbs the
        exactly-determined ladder up to ``n_fit_max``, ``converge=False`` runs one fit with
        ``n_fit`` (default 6) and ``fit_order`` (default -1).

    Written datasets
    ----------------
    ``source="mesh"``: ``{grp}/iter{N}/qp_approx/{E_ska, Heff_skij, Zqp_ska, mu, scheme}`` and
    ``qp_approx/lqp/{Z_skab, V_skab, min_eig_Zinv_sk, resid_sk, anti_herm_A_sk, anti_herm_B_sk,
    anti_herm_Sigma_sk, sumrule_sk, status_sk, n_fit, fit_order, symmetric_window,
    fit_resid_tol}`` -- the layout
    :func:`band_interpolation` writes with ``qp_approx="lqp"``. ``source="interpolated"``: the same
    layout under ``{grp}/iter{N}/wannier_inter/``, i.e. ``wannier_inter/qp_approx/`` and
    ``wannier_inter/qp_approx/lqp/``, with ``Heff_skab``. With ``converge=True`` also
    ``qp_approx/lqp/ladder/{err_fit_sk, dHqp_sk, resid_history, n_accepted, n_fit_cap,
    stopped_resid}``, holding the climb whose last accepted fit is the result; a later single
    fit removes it. Members mean what they mean in :class:`LinearizedQPResult` /
    :class:`LinearizedQPLadder`. Energies are absolute; ``mu`` is stored beside them. Axis 1 is
    the k-point axis of the source: the IBZ mesh (``system/kpoints_crys`` through ``kp_to_ibz``)
    or the interpolated path (``wannier_inter/{kpts, kpt_labels, kpt_label_idx}``).

    Notes
    -----
    The program aborts (C++, every MPI rank) with a message naming the problem on bad input --
    an unknown ``source``, a missing file, group, iteration or input dataset (``"interpolated"`` needs
    :func:`spectral_interpolation` first), a single fit whose window does not lie on the
    sampling mesh -- and where the linearization fails: a single fit at a point that fails the
    residual gate or has ``1 - B`` not positive definite, or a ladder that accepts no fit. A
    written result therefore has no failed point (``status_sk`` is zero everywhere).
    """
    _checkpoint(mpi, coqui_h5, grp_name, int(iteration), source, int(n_fit), int(fit_order),
                bool(symmetric_window), float(fit_resid_tol), int(n_fit_max) if converge else 0)


def _cpp_iaft(iaft):
    """The C++ IAFT (iaft_module.IAFT) behind a Python IAFT, as the kernel bindings take it.

    TODO: the Python IAFT of basis "ir" is backed by sparse_ir and holds no C++ IAFT, so one is
    built here from its parameters; its sampling mesh equals sparse_ir's (pinned by
    utils/imag_axes_ft/tests/test_iaft.py). Remove this once the Python IAFT carries a C++ IAFT
    for every basis; the bindings and linearized_qp_from_arrays stay as they are.
    """
    if iaft.basis == "dlr":
        # the Python IAFT wraps a C++ IAFT: hand it over
        return iaft._impl._iaft_cpp

    # basis "ir": the Python IAFT is sparse_ir and holds no C++ IAFT, so build the C++ IR basis
    # from the same parameters. The C++ IR backend has tabulated bases only -- lambda = beta*wmax
    # rounded up to 1e2, ..., 1e6, and prec "high" / "medium" / "low" -- which is how the Python
    # IAFT rounds for a preset prec, so both have the same sampling mesh. A custom eps (exact
    # lambda, arbitrary accuracy) has no C++ counterpart.
    if iaft.prec == "custom":
        raise NotImplementedError("linearized_qp_from_arrays: an IR IAFT with a custom eps is not "
                                  "supported; use prec='high', 'medium' or 'low', or basis='dlr'.")
    return _IAFTCpp(float(iaft.beta), float(iaft.wmax), _basis_enum("ir"), str(iaft.prec), False)
