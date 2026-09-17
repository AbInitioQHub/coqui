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

import json
import numpy as np
from coqui import IAFT

from coqui._lib import post_proc_module as pproc_mod


def band_interpolation(mf, params):
    """
    Interpolate quasiparticle energies along a k-path using Wannier functions.

    This function has two modes depending on whether a CoQuí MBPT checkpoint
    file already exists:

    - **Checkpoint mode** — if ``outdir/prefix.mbpt.h5`` already exists, quasiparticle
      energies are read from the specified iteration and interpolated.
    - **Mf-only mode** — if no checkpoint exists, the function first creates one,
      writes the mean-field (DFT) solution into ``iter0``, and then performs the
      interpolation on that iteration. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Interpolation options. Supported keys:

        - ``prefix`` *(str, required)* — prefix of the checkpoint file
          (``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, optional, default ``"./"``)* — directory for the
          checkpoint file.
        - ``wannier_file`` *(str, required)* — HDF5 file containing Wannier
          projection matrices and k-points, produced by ``mlwf_h5_from_wannier90_output``.
        - ``kpath`` *(str, required)* — high-symmetry k-path as a multi-line string
          with one ``label kx ky kz`` entry per line (in fractional coordinates).
          Consecutive entries define segments; the path connects them in order.
        - ``bands_num_npoints`` *(int, optional, default ``100``)* — number of
          k-points per segment along the path.
        - ``iteration`` *(int, optional, default ``-1``)* — MBPT iteration to read
          from. ``-1`` selects the latest (in Mf-only mode this is always ``iter0``).
        - ``grp_name`` *(str, optional, default ``"scf"``)* — top-level HDF5 group
          in the checkpoint (``"scf"`` for standard MBPT, ``"embed"`` for embedding).
        - ``translate_home_cell`` *(bool, optional, default ``False``)* — if ``True``,
          translates Wannier centres to the home unit cell.

        The following keys are only used in **Mf-only mode** (no existing checkpoint):

        - ``beta`` *(float, optional, default ``1000.0``, units: 1/Hartree)* —
          inverse temperature, used to determine the Fermi level. 

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5`` under
        ``{grp_name}/iter{N}/qp_approx/wannier_inter/``.

    Examples
    --------
    ::

        from coqui.post_proc import band_interpolation

        band_interpolation(mf, {
            "outdir": "./", "prefix": "svo.evgw",
            "wannier_file": "svo.mlwf.h5",
            "bands_num_npoints": 100,
            "kpath": (
                "G 0.00 0.00 0.00 "
                "X 0.00 0.50 0.00 "
                "M 0.50 0.50 0.00 "
                "G 0.00 0.00 0.00"
            ),
        })
    """
    pproc_mod.band_interpolation(mf, json.dumps(params))


def spectral_interpolation(mf, params):
    """
    Interpolate the spectral function A(k,ω) along a k-path using Wannier functions.

    Reads the Matsubara Green's function from an MBPT checkpoint, projects onto a
    Wannier basis, interpolates onto a dense k-path, and applies analytic continuation
    to produce A(k,ω) for spectral function plots.

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Interpolation and analytic continuation options. Supported keys:

        - ``prefix`` *(str, required)* — prefix of the MBPT checkpoint file
          (``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, required)* — directory containing the MBPT checkpoint.
        - ``wannier_file`` *(str, required)* — HDF5 file containing Wannier
          projection matrices and k-points.
        - ``kpath`` *(str, required)* — high-symmetry k-path in the same format as 
          ``band_interpolation``. 
        - ``bands_num_npoints`` *(int, optional, default ``100``)* — k-points per
          segment along the path.
        - ``iteration`` *(int, optional, default ``-1``)* — MBPT iteration to use.
          ``-1`` selects the latest.
        - ``grp_name`` *(str, optional, default ``"scf"``)* — top-level HDF5 group
          in the checkpoint.
        - ``translate_home_cell`` *(bool, optional, default ``False``)* — translate
          Wannier centres to the home unit cell.
        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm. Only ``"pade"`` is currently implemented. 
        - ``w_min`` *(float, optional, default ``-0.367``, units: Hartree (~-10 eV))* — 
          lower bound of the real frequency output grid (~-10 eV).
        - ``w_max`` *(float, optional, default ``0.367``, units: Hartree (~+10 eV))* — 
          upper bound of the real frequency output grid (~+10 eV).
        - ``Nw`` *(int, optional)* — number of real frequency points. Defaults to a
          resolution of ~0.05 eV across the ``[w_min, w_max]`` window.
        - ``eta`` *(float, optional, default ``π/β``, units: Hartree)* — broadening
          for the analytic continuation.
        - ``Nfit`` *(int, optional)* — number of Matsubara points for the Padé fit.
          Defaults to half the number of available fermionic frequencies.

        Quasiparticle energy options (computed on the IBZ k-mesh, then Wannier-interpolated):

        - ``qp_type`` *(str, optional, default ``"sc"``)* — QP equation solver.
          ``"sc"`` (bisection), ``"sc_newton"`` (secant), or ``"linearized"``.
        - ``qp_eta`` *(float, optional, default ``0.0001``, units: Hartree)* — broadening
          on the real axis used when evaluating the self-energy in the QP equation.
        - ``qp_Nfit`` *(int, optional, default ``18``)* — number of Matsubara points for
          the Padé fit of the self-energy used in the QP equation.
        - ``qp_tol`` *(float, optional, default ``1e-8``)* — convergence tolerance for
          the QP root-finding algorithm.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5`` under:

        - ``{grp_name}/iter{N}/ac/`` — spectral function A(k,ω) (unchanged).
        - ``{grp_name}/iter{N}/qp_approx/E_ska`` — QP energies on the IBZ k-mesh.
        - ``{grp_name}/iter{N}/qp_approx/wannier_inter/E_ska`` — QP band structure
          interpolated along the k-path.

    Examples
    --------
    ::

        from coqui.post_proc import spectral_interpolation

        spectral_interpolation(mf, {
            "outdir": "./", "prefix": "svo.evgw",
            "wannier_file": "svo.mlwf.h5",
            "bands_num_npoints": 100,
            "kpath": (
                "G 0.00 0.00 0.00 "
                "X 0.00 0.50 0.00 "
                "M 0.50 0.50 0.00 "
                "G 0.00 0.00 0.00"
            ),
            "w_min": -0.5, "w_max": 0.5,
        })
    """
    pproc_mod.spectral_interpolation(mf, json.dumps(params))


def local_dos(mf, params):
    """
    Compute the orbital-resolved local density of states.

    Reads the Matsubara Green's function from an MBPT checkpoint, optionally 
    projects onto a Wannier basis, and applies analytic continuation to produce
    the orbital/site-resolved DOS on the real frequency axis.

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Supported keys:

        - ``prefix`` *(str, required)* — prefix of the MBPT checkpoint file
          (``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, optional, default ``"./"``)* — directory containing 
          the MBPT checkpoint.
        - ``wannier_file`` *(str, optional, default ``""``)* — HDF5 file with 
          Wannier projection matrices. If empty, the DOS is computed in the Bloch basis.
        - ``iteration`` *(int, optional, default ``-1``)* — MBPT iteration to use.
          ``-1`` selects the latest.
        - ``grp_name`` *(str, optional, default ``"scf"``)* — top-level HDF5 group
          in the checkpoint.
        - ``translate_home_cell`` *(bool, optional, default ``False``)* — translate
          Wannier centres to the home unit cell (only relevant when ``wannier_file``
          is provided).
        - ``ac_alg`` *(str, optional, default ``"pade"``)* — analytic continuation
          algorithm.
        - ``w_min`` *(float, optional, default ``-0.367``, units: Hartree)* — lower
          bound of the real frequency output grid (~-10 eV).
        - ``w_max`` *(float, optional, default ``0.367``, units: Hartree)* — upper
          bound of the real frequency output grid (~+10 eV).
        - ``Nw`` *(int, optional)* — number of real frequency points. Defaults to a
          resolution of ~0.05 eV across the ``[w_min, w_max]`` window.
        - ``eta`` *(float, optional, default ``π/β``, units: Hartree)* — broadening
          for the analytic continuation.
        - ``Nfit`` *(int, optional)* — number of Matsubara points for the Padé fit.
          Defaults to half the number of available fermionic frequencies.

    Returns
    -------
    None
        Results are written to ``outdir/prefix.mbpt.h5`` under
        ``{grp_name}/iter{N}/ac/``.
    """
    pproc_mod.local_dos(mf, json.dumps(params))


def unfold_bz(mf, params):
    """
    Unfold IBZ-sampled MBPT quantities to the full Brillouin zone.

    Expands quasiparticle energies, Green's functions, and self-energies from the
    irreducible wedge to all k-points in the full BZ using the lattice symmetry
    operations stored in the mean-field object.

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Supported keys:

        - ``prefix`` *(str, required)* — prefix of the MBPT checkpoint file
          (``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, optional, default ``"./"``)*  — directory containing the
          MBPT checkpoint file.

    Returns
    -------
    None
        Full-BZ quantities are written back to ``outdir/prefix.mbpt.h5`` under 
        the same HDF5 group with the `_unfold` suffix.
    """
    pproc_mod.unfold_bz(mf, json.dumps(params))


def dump_vxc(mf, params):
    """
    Extract and save exchange-correlation potential matrix elements from the DFT solution.

    Computes ⟨ψ_{sk}|V_xc|ψ_{sk'}⟩ in the Kohn-Sham orbital basis and writes the
    result to an HDF5 file. Useful for constructing the quasiparticle correction
    Σ - V_xc in post-processing workflows.

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Supported keys:

        - ``prefix`` *(str, required)* — prefix for the output HDF5 file
          (writes to ``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, optional, default ``"./"``)*  — output directory.

    Returns
    -------
    None
        The matrix ``Vxc_skij`` (shape: ``[nspin, nkpts_ibz, nbnd, nbnd]``) is
        written to ``outdir/prefix.mbpt.h5`` under ``system/``.
    """
    pproc_mod.dump_vxc(mf, json.dumps(params))


def dump_hartree(mf, params):
    """
    Compute and save Hartree potential matrix elements.

    This function has two modes depending on whether a CoQuí MBPT checkpoint
    file already exists:

    - **Checkpoint mode** — if ``outdir/prefix.mbpt.h5`` already exists, the density
      matrix is read from the iteration specified by ``scf_iter`` and the Hartree
      potential is computed from it.
    - **Mf-only mode** — if no checkpoint exists, the function creates one, writes the
      mean-field (DFT) solution into ``iter0``, and computes the Hartree potential from
      the DFT density matrix. 

    Parameters
    ----------
    mf : Mf
        Mean-field object for the target system, obtained from ``make_mf``.
    params : dict
        Supported keys:

        - ``prefix`` *(str, required)* — prefix of the checkpoint file
          (``outdir/prefix.mbpt.h5``).
        - ``outdir`` *(str, optional, default ``"./"``)*  — directory for the
          checkpoint file.
        - ``scf_iter`` *(int, optional, default ``-1``)* — iteration whose density
          matrix is used. ``-1`` selects the latest (in Mf-only mode this is always
          ``iter0``).

        The following keys are only used in **Mf-only mode** (no existing checkpoint):

        - ``beta`` *(float, optional, default ``1000.0``, units: 1/Hartree)* —
          inverse temperature, used during density matrix consturction. 

    Returns
    -------
    None
        The matrix ``VH_skij`` (shape: ``[nspin, nkpts_ibz, nbnd, nbnd]``) is
        written to ``outdir/prefix.mbpt.h5`` under ``scf/iter{N}/``.
    """
    pproc_mod.dump_hartree(mf, json.dumps(params))


def pade(g_iw, iaft, stats, wmin, wmax, Nw, *, 
         Nfit=-1, eta=None, ph_sym=False, causal_projection=False):
    """
    Perform Padé analytic continuation from the Matsubara to the real frequency axis.

    Parameters
    ----------
    g_iw : np.ndarray
        Input data on the Matsubara frequency axis. Can have arbitrary shape;
        the first dimension is treated as the frequency axis.
    iaft : IAFT
        Imaginary-axis Fourier transform object. The Matsubara frequency mesh
        and inverse temperature are derived from it.
    stats : str
        Particle statistics: ``"fermion"`` or ``"boson"``.
    wmin : float
        Lower bound of the real frequency output grid (units: Hartree).
    wmax : float
        Upper bound of the real frequency output grid (units: Hartree).
    Nw : int
        Number of real frequency points.
    Nfit : int, optional
        Number of Matsubara points used for the Padé fit. ``-1`` uses all
        available points. Default: ``-1``.
    eta : float, optional
        Broadening (imaginary offset above the real axis, units: Hartree).
        Default: ``π/β``.
    ph_sym : bool, optional
        If ``True``, exploit particle-hole symmetry and use only the
        positive-frequency half of the Matsubara axis. Default: ``False``.
    causal_projection : bool, optional
        If ``True``, apply causal projection to the continued function.
        Default: ``False``.

    Returns
    -------
    g_w : np.ndarray
        Analytically continued data on the real frequency axis, shape
        ``(Nw,) + g_iw.shape[1:]``. Complex array.
    w_mesh : np.ndarray
        Real frequency mesh, shape ``(Nw,)``. Complex array with imaginary part
        equal to ``eta``.
    """
    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT for minipole fitting. ")
    
    if causal_projection:
        g_iw = aaa_adapol_imag(g_iw, iaft, stats, Nfit=Nfit, solver="lstsq", ph_sym=ph_sym)
    
    g_iw = np.asarray(g_iw, dtype=np.complex128)
    iw_mesh = 1j*iaft.wn_mesh(stats=stats, positive_only=ph_sym) * np.pi / iaft.beta
    if g_iw.shape[0] != len(iw_mesh):
        raise ValueError(f"First dimension of g_iw ({g_iw.shape[0]}) must match length of Matsubara mesh ({len(iw_mesh)}) from IAFT.")

    niw = g_iw.shape[0]
    target_shape = g_iw.shape[1:]

    eta = np.pi / iaft.beta if eta is None else float(eta)    

    g_iw_2D = g_iw.reshape(niw, -1)
    g_w_2D, w_mesh = pproc_mod.pade(g_iw_2D, iw_mesh, wmin, wmax, Nw, Nfit, eta, ph_sym)
    g_w = g_w_2D.reshape((Nw,) + target_shape)

    return g_w, w_mesh


def aaa_adapol_on_mesh(g_iw, iaft, stats, w_mesh_out, *, Nfit=40, solver="lstsq", ph_sym=False):
    """
    Perform rational approximation to fit a Green's function using the AAA-based adapol 
    algorithm and evaluate the fitted rational function on an arbitrary output mesh.

    Requires the ``adapol`` package (https://github.com/flatironinstitute/adapol).

    Parameters
    ----------
    g_iw : np.ndarray
        Input data on the Matsubara frequency axis. Supported shapes: 1-D
        ``(niw,)`` for a single scalar, 2-D ``(niw, n)`` for ``n`` independent
        scalars fitted in a loop, or 3-D ``(niw, norb, norb)`` for a full
        matrix fitted jointly.
    iaft : IAFT
        Imaginary-axis Fourier transform object. The Matsubara frequency mesh
        and inverse temperature are derived from it.
    stats : str
        Particle statistics: ``"fermion"`` (or ``"f"``) or ``"boson"`` (or ``"b"``).
    w_mesh_out : array-like of complex
        Output frequency mesh. May be real, imaginary, or mixed complex values.
        Shape ``(Nout,)``.
    Nfit : int, optional
        Number of poles used in the rational approximation. Default: ``40``.
    ph_sym : bool, optional
        If ``True``, use only the positive-frequency half of the Matsubara axis.
        Default: ``False``.

    Returns
    -------
    g_out : np.ndarray
        Fitted data evaluated on ``w_mesh_out``, shape ``(Nout,) + g_iw.shape[1:]``.
        Complex array.
    w_mesh : np.ndarray
        Output frequency mesh as passed in, shape ``(Nout,)``.
    """
    # TODO when g_iw.ndim > 3, we assume (niw, ..., norb, norb) and perform matrix fitting for each "..." index independently.
    try:
        from adapol import anacont as adapol_anacont
    except ImportError:
        raise ImportError("aaa_adapol requires the adapol package (https://github.com/flatironinstitute/adapol). \n"
                          "Ensure that it is installed. ")

    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT.")

    if stats in ['fermion', 'f']:
        statistic = "Fermion"
    elif stats in ['boson', 'b']:
        statistic = "Boson"
    else:
        raise ValueError("Invalid statistic. Use 'fermion' or 'boson'.")

    g_iw = np.asarray(g_iw, dtype=np.complex128)
    target_shape = g_iw.shape[1:]
    if g_iw.ndim > 3:
        raise ValueError("Input g_iw can only have 1, 2, or 3 dimensions.")

    iw_mesh = 1j * iaft.wn_mesh(stats=stats, positive_only=ph_sym) * np.pi / iaft.beta
    if g_iw.shape[0] != len(iw_mesh):
        raise ValueError(f"First dimension of g_iw ({g_iw.shape[0]}) must match length of Matsubara mesh ({len(iw_mesh)}) from IAFT.")

    w_mesh_out = np.asarray(w_mesh_out, dtype=np.complex128)
    Nout = len(w_mesh_out)

    def _fit_scalar(g_scalar):
        gw_func, _, _, _ = adapol_anacont(
            g_scalar[:, None, None], iw_mesh, Np=Nfit, statistics=statistic
        )
        return gw_func(w_mesh_out)[:, 0, 0]

    if g_iw.ndim == 1:
        g_out = _fit_scalar(g_iw)
    elif g_iw.ndim == 2:
        g_out = np.zeros((Nout, g_iw.shape[1]), dtype=complex)
        for i in range(g_iw.shape[1]):
            g_out[:, i] = _fit_scalar(g_iw[:, i])
    else:
        gw_func, _, _, _ = adapol_anacont(g_iw, iw_mesh, Np=Nfit, statistics=statistic)
        g_out = gw_func(w_mesh_out)

    return g_out.reshape((Nout,) + target_shape)


def aaa_adapol(g_iw, iaft, stats, wmin, wmax, Nw, *, Nfit=40, solver="lstsq", eta=None, ph_sym=False):
    """
    Perform analytic continuation using the AAA-based adapol algorithm to evaluate 
    correlation functions on real-frequency axis.

    Requires the ``adapol`` package (https://github.com/flatironinstitute/adapol).

    Parameters
    ----------
    g_iw : np.ndarray
        Input data on the Matsubara frequency axis. Supported shapes: 1-D
        ``(niw,)`` for a single scalar, 2-D ``(niw, n)`` for ``n`` independent
        scalars fitted in a loop, or 3-D ``(niw, norb, norb)`` for a full
        matrix fitted jointly.
    iaft : IAFT
        Imaginary-axis Fourier transform object. The Matsubara frequency mesh
        and inverse temperature are derived from it.
    stats : str
        Particle statistics: ``"fermion"`` (or ``"f"``) or ``"boson"`` (or ``"b"``).
    wmin : float
        Lower bound of the real frequency output grid (units: Hartree).
    wmax : float
        Upper bound of the real frequency output grid (units: Hartree).
    Nw : int
        Number of real frequency points.
    Nfit : int, optional
        Number of poles used in the rational approximation. Default: ``40``.
    eta : float, optional
        Broadening (imaginary offset above the real axis, units: Hartree).
        Default: ``π/β``.
    ph_sym : bool, optional
        If ``True``, use only the positive-frequency half of the Matsubara axis.
        Default: ``False``.

    Returns
    -------
    g_w : np.ndarray
        Analytically continued data on the real frequency axis, shape
        ``(Nw,) + g_iw.shape[1:]``. Complex array.
    w_mesh : np.ndarray
        Real frequency mesh, shape ``(Nw,)``. Complex array with imaginary part
        equal to ``eta``.
    """
    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT.")
    eta = np.pi / iaft.beta if eta is None else float(eta)
    w_mesh = np.linspace(wmin, wmax, Nw) + 1j * eta
    return aaa_adapol_on_mesh(g_iw, iaft, stats, w_mesh, Nfit=Nfit, solver=solver, ph_sym=ph_sym), w_mesh


def aaa_adapol_imag(g_iw, iaft, stats, *, Nfit=40, solver="lstsq", ph_sym=False, iw_mesh_out=None):
    """
    Perform rational approximation to fit a Green's function using the AAA-based adapol
    algorithm and evaluate the fitted rational function on the imaginary frequency axis.

    Requires the ``adapol`` package (https://github.com/flatironinstitute/adapol).

    Parameters
    ----------
    g_iw : np.ndarray
        Input data on the Matsubara frequency axis. Supported shapes: 1-D
        ``(niw,)`` for a single scalar, 2-D ``(niw, n)`` for ``n`` independent
        scalars fitted in a loop, or 3-D ``(niw, norb, norb)`` for a full
        matrix fitted jointly.
    iaft : IAFT
        Imaginary-axis Fourier transform object. The Matsubara frequency mesh
        and inverse temperature are derived from it.
    stats : str
        Particle statistics: ``"fermion"`` (or ``"f"``) or ``"boson"`` (or ``"b"``).
    Nfit : int, optional
        Number of poles used in the rational approximation. Default: ``40``.
    ph_sym : bool, optional
        If ``True``, use only the positive-frequency half of the Matsubara axis
        for fitting. Default: ``False``.
    iw_mesh_out : array-like of complex, optional
        Output imaginary-frequency mesh. If ``None``, uses the same Matsubara
        mesh as the input (derived from ``iaft`` and ``stats``).

    Returns
    -------
    g_iw_out : np.ndarray
        Fitted data evaluated on ``iw_mesh_out``, shape
        ``(Nout,) + g_iw.shape[1:]``. Complex array.
    iw_mesh_out : np.ndarray
        Output imaginary-frequency mesh, shape ``(Nout,)``.
    """
    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT.")
    if iw_mesh_out is None:
        iw_mesh_out = 1j * iaft.wn_mesh(stats=stats, positive_only=ph_sym) * np.pi / iaft.beta
    return aaa_adapol_on_mesh(g_iw, iaft, stats, iw_mesh_out, Nfit=Nfit, solver=solver, ph_sym=ph_sym)


def minipole(g_iw, iaft, stats, wmin, wmax, Nw, *, 
             tol=1e-4, eta=None, ph_sym=False, causal_projection=False):
    """
    Perform analytic continuation using the Minimal Pole algorithm.

    Requires the ``mini_pole`` package 
    (https://github.com/Green-Phys/MiniPole/tree/main).

    Parameters
    ----------
    g_iw : np.ndarray
        Input data on the Matsubara frequency axis. Supported shapes: 1-D
        ``(niw,)`` for a scalar, 2-D ``(niw, norb)`` for orbital-diagonal
        elements, or 3-D ``(niw, norb, norb)`` for a full matrix.
    iaft : IAFT
        Imaginary-axis Fourier transform object. Must be an ``IAFT`` instance
        (array-like input is not supported for this method).
    stats : str
        Particle statistics: ``"fermion"`` (or ``"f"``) or ``"boson"`` (or ``"b"``).
    wmin : float
        Lower bound of the real frequency output grid (units: Hartree).
    wmax : float
        Upper bound of the real frequency output grid (units: Hartree).
    Nw : int
        Number of real frequency points.
    tol : float, optional
        Fitting tolerance for the MiniPole algorithm. Default: ``1e-4``.
    eta : float, optional
        Broadening (imaginary offset above the real axis, units: Hartree).
        Defaults to ``π/β``.
    ph_sym : bool, optional
        If ``True``, exploit particle-hole symmetry when interpolating onto
        the uniform Matsubara grid. Default: ``False``.
    causal_projection : bool, optional
        If ``True``, apply causal projection to the continued function.
        Default: ``False``.

    Returns
    -------
    g_w : np.ndarray
        Analytically continued data on the real frequency axis, shape
        ``(Nw,) + g_iw.shape[1:]``. Complex array.
    w_mesh : np.ndarray
        Real frequency mesh, shape ``(Nw,)``. Complex array with imaginary part
        equal to ``eta``.
    """
    try:
        from mini_pole import MiniPole
    except ImportError:
        raise ImportError("minipole function requires the mini_pole package (https://github.com/Green-Phys/MiniPole/tree/main). \n"
                          "Ensure that it is installed. ")

    if not isinstance(iaft, IAFT):
        raise ValueError("iaft must be an instance of IAFT for minipole fitting. ")
    
    if causal_projection:
        g_iw = aaa_adapol_imag(g_iw, iaft, stats, Nfit=40, solver="lstsq", ph_sym=ph_sym)
    
    g_iw = np.asarray(g_iw, dtype=np.complex128)
    ndim = g_iw.ndim
    if g_iw.ndim == 1 or g_iw.ndim == 2:
        alg_type = "scalar"
        g_iw = g_iw.reshape(g_iw.shape[0], -1)  # Ensure 2D shape for scalar case
    elif g_iw.ndim == 3:
        alg_type = "matrix"
        assert g_iw.shape[1] == g_iw.shape[2], "For matrix case, g_iw must have shape [niw, norb, norb]."
    else:
        raise ValueError("Input g_iw can only have 1, 2, or 3 dimensions.")

    # minipole requires positive uniformlly spaced frequencies  
    nw_max = iaft.wn_mesh(stats=stats)[-1]
    offset = 1 if stats in ['fermion', 'f'] else 0
    iw_mesh_uniform = np.arange(offset, nw_max, 2)
    g_iw_pos = iaft.w_interpolate(g_iw, iw_mesh_uniform, stats=stats, ph_sym=ph_sym)

    eta = np.pi / iaft.beta if eta is None else float(eta)
    w_mesh = np.linspace(wmin, wmax, Nw) + 1j * eta

    #tools for calculating the Green's function from the corresponding poles
    def cal_G_scalar(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[i] / (z - xl[i])
        return G_z

    def cal_G_vector(z, Al, xl):
        G_z = 0.0
        for i in range(xl.size):
            G_z += Al[[i]] / (z.reshape(-1, 1) - xl[i])
        return G_z

    def _minipole_fit_scalar(g_iw, iw_mesh, w_mesh, tol):
        mp = MiniPole(g_iw, iw_mesh*np.pi/iaft.beta, err=tol)
        g_w = cal_G_scalar(w_mesh, mp.pole_weight, mp.pole_location)
        return g_w
    
    def _minipole_fit_matrix(g_iw, iw_mesh, w_mesh, tol):
        mp = MiniPole(g_iw, iw_mesh*np.pi/iaft.beta, err=tol)
        n_orb = g_iw.shape[1]
        g_w = cal_G_vector(w_mesh, mp.pole_weight.reshape(-1, n_orb ** 2), mp.pole_location).reshape(-1, n_orb, n_orb)
        return g_w

    if alg_type == "scalar":
        g_w = np.zeros((Nw, g_iw_pos.shape[1]), dtype=np.complex128)
        for i in range(g_iw_pos.shape[1]):
            g_w[:, i] = _minipole_fit_scalar(g_iw_pos[:, i], iw_mesh_uniform, w_mesh, tol)
    else:
        g_w = _minipole_fit_matrix(g_iw_pos, iw_mesh_uniform, w_mesh, tol)

    if ndim == 1:
        g_w = g_w.reshape(-1)

    return g_w, w_mesh
