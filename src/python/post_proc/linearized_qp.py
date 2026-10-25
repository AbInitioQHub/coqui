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
Linearized quasiparticle approximation (LQP) of a dynamic self-energy: Sigma(iw) ~ A + iw B fitted
at low Matsubara frequencies, Z = (1 - B)^-1, H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu, and the pole
weights <v|Z|v>.

lqp_from_dyson_scf               a Dyson-SCF solution in a checkpoint, on its k-points
lqp_from_spectral_interpolation  the Wannier-interpolated solution of spectral_interpolation
lqp_from_arrays                  arrays in memory, in any basis and on any k-points
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
    "lqp_from_dyson_scf",
    "lqp_from_spectral_interpolation",
    "lqp_from_arrays",
    "LinearizedQPResult",
    "LinearizedQPLadder",
]


def lqp_from_dyson_scf(mpi, coqui_h5, grp_name="scf", iteration=-1, *, converge=True, n_fit=6,
                       n_fit_max=20, fit_order=-1, symmetric_window=True, fit_resid_tol=1e-8):
    r"""Linearized quasiparticle approximation to a Dyson-SCF solution in a CoQui checkpoint, on
    the k-points of the SCF calculation.

    Fits ``Sigma(iw) ~ A + iw B`` to ``{grp}/iter{N}/Sigma_tskij`` at the lowest Matsubara
    frequencies and writes the result back to the checkpoint.

    Parameters
    ----------
    mpi : MpiHandler
        MPI environment (``coqui.MpiHandler()``).
    coqui_h5 : str
        CoQui checkpoint file.
    grp_name : str
        ``"scf"`` (GW / MBPT) or ``"embed"`` (GW+EDMFT lattice).
    iteration : int
        Iteration to process; ``-1`` (default) is the last one.
    converge : bool
        ``True`` (default): widen the fit window until the fit stops being reliable and keep the
        last reliable fit. ``False``: one fit with ``n_fit`` and ``fit_order``.
    n_fit, fit_order
        The single fit: number of lowest positive Matsubara frequencies (default 6) and highest
        power of ``iw`` (default -1, i.e. ``2*n_fit - 1``).
    n_fit_max, symmetric_window, fit_resid_tol
        Widest window of ``converge=True`` (default 20), fit on ``+-iw`` pairs (default True), and
        the largest accepted fit residual (default 1e-8).

    Written datasets
    ----------------
    In ``{grp}/iter{N}/qp_approx/``, with ``Z = (1 - B)^-1``:

    ``Heff_skij``
        The quasiparticle Hamiltonian ``H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu`` in the KS basis.
    ``E_ska``
        The quasiparticle energies, eigenvalues of ``H_QP`` (absolute, not relative to ``mu``).
    ``Zqp_ska``
        The pole weights ``<v|Z|v>`` of the quasiparticle states, ``v`` the eigenvectors of
        ``H_QP``.
    """
    _checkpoint(mpi, coqui_h5, grp_name, int(iteration), "mesh", int(n_fit), int(fit_order),
                bool(symmetric_window), float(fit_resid_tol), int(n_fit_max) if converge else 0)


def lqp_from_spectral_interpolation(mpi, coqui_h5, grp_name="scf", iteration=-1, *,
                                    converge=True, n_fit=6, n_fit_max=20, fit_order=-1,
                                    symmetric_window=True, fit_resid_tol=1e-8):
    r"""Linearized quasiparticle approximation to the Wannier-interpolated Dyson-SCF solution
    written by :func:`spectral_interpolation`, on its k-points.

    Reads the Wannier-interpolated ``F`` and ``Sigma(tau)`` that :func:`spectral_interpolation`
    has written to ``{grp}/iter{N}/wannier_inter/`` (run it first), linearizes the self-energy
    within the Wannier subspace, and writes the result back to the checkpoint. The quasiparticle
    approximation thus acts on the already interpolated solution, whereas
    :func:`band_interpolation` with ``qp_approx="lqp"`` applies it on the SCF k-points and then
    interpolates ``H_QP``; the two agree when the Wannier subspace equals the KS orbital space
    and otherwise differ by a basis-set error.

    Parameters
    ----------
    As in :func:`lqp_from_dyson_scf`.

    Written datasets
    ----------------
    In ``{grp}/iter{N}/wannier_inter/qp_approx/``, on the k-points of the interpolation, with
    ``Z = (1 - B)^-1``:

    ``Heff_skab``
        The quasiparticle Hamiltonian ``H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu`` in the Wannier
        basis.
    ``E_ska``
        The quasiparticle energies, eigenvalues of ``H_QP`` (absolute, not relative to ``mu``);
        :func:`band_plot_from_spectral_interpolation` plots them.
    ``Zqp_ska``
        The pole weights ``<v|Z|v>`` of the quasiparticle states, ``v`` the eigenvectors of
        ``H_QP``.
    """
    _checkpoint(mpi, coqui_h5, grp_name, int(iteration), "interpolated", int(n_fit),
                int(fit_order), bool(symmetric_window), float(fit_resid_tol),
                int(n_fit_max) if converge else 0)


def lqp_from_arrays(mpi, F_sk, Sigma_tsk, mu, iaft, *, converge=True, n_fit=6, n_fit_max=20,
                    fit_order=-1, symmetric_window=True, fit_resid_tol=1e-8):
    r"""Linearized quasiparticle approximation to a dynamic self-energy given as arrays, in any
    orthonormal basis and on any set of k-points.

    Meant for small arrays, which every rank holds; for checkpoint data use
    :func:`lqp_from_dyson_scf` or :func:`lqp_from_spectral_interpolation`.

    Parameters
    ----------
    mpi : MpiHandler
        MPI environment (``coqui.MpiHandler()``).
    F_sk : (ns, nk, n, n) complex
        Static one-body matrix, including ``H0``.
    Sigma_tsk : (nt, ns, nk, n, n) complex
        Dynamic self-energy on the fermionic tau mesh of ``iaft``, in the basis of ``F_sk``.
    mu : float
        Chemical potential [Ha].
    iaft : IAFT
        The transform ``Sigma_tsk`` is sampled with: ``basis="dlr"``, or ``basis="ir"`` with
        ``prec`` "high", "medium" or "low".
    converge, n_fit, n_fit_max, fit_order, symmetric_window, fit_resid_tol
        As in :func:`lqp_from_dyson_scf`.

    Returns
    -------
    LinearizedQPLadder (``converge=True``, the kept fit in ``last``) or LinearizedQPResult
    (``converge=False``), with ``Z = (1 - B)^-1``:

    ``Hqp_skab``
        The quasiparticle Hamiltonian ``H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu`` in the basis of
        ``F_sk``.
    ``E_ska``
        The quasiparticle energies, eigenvalues of ``H_QP`` (absolute, not relative to ``mu``).
    ``Zqp_ska``
        The pole weights ``<v|Z|v>`` of the quasiparticle states, ``v`` the eigenvectors of
        ``H_QP``.
    ``Z_skab``
        The renormalization matrix ``Z``.
    """
    F_sk = np.ascontiguousarray(F_sk, dtype=np.complex128)
    Sigma_tsk = np.ascontiguousarray(Sigma_tsk, dtype=np.complex128)
    if Sigma_tsk.shape[1:] != F_sk.shape:
        raise ValueError(f"lqp_from_arrays: Sigma_tsk {Sigma_tsk.shape} and "
                         f"F_sk {F_sk.shape} are inconsistent.")
    iaft.check_leakage(Sigma_tsk, "fermion", name="Sigma(tau)")
    ft = _cpp_iaft(iaft)

    if converge:
        return _ladder(mpi, F_sk, Sigma_tsk, float(mu), ft, int(n_fit_max), float(fit_resid_tol),
                       bool(symmetric_window))
    return _solve(mpi, F_sk, Sigma_tsk, float(mu), ft, int(n_fit), int(fit_order),
                  bool(symmetric_window), float(fit_resid_tol))


def _cpp_iaft(iaft):
    """The C++ IAFT (iaft_module.IAFT) behind a Python IAFT, as the kernel bindings take it.

    TODO: the Python IAFT of basis "ir" is backed by sparse_ir and holds no C++ IAFT, so one is
    built here from its parameters; its sampling mesh equals sparse_ir's (pinned by
    utils/imag_axes_ft/tests/test_iaft.py). Remove this once the Python IAFT carries a C++ IAFT
    for every basis; the bindings and lqp_from_arrays stay as they are.
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
        raise NotImplementedError("lqp_from_arrays: an IR IAFT with a custom eps is not "
                                  "supported; use prec='high', 'medium' or 'low', or basis='dlr'.")
    return _IAFTCpp(float(iaft.beta), float(iaft.wmax), _basis_enum("ir"), str(iaft.prec), False)
