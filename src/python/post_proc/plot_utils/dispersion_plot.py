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

import sys
import numpy as np
from scipy.constants import physical_constants
Hartree_eV = physical_constants['Hartree energy in eV'][0]
import matplotlib.pyplot as plt
from h5 import HDFArchive
from coqui import app_log, app_warning


def _has_path(ar, path):
    """Whether ``path`` exists in the archive"""
    grp = ar
    for key in path.split("/"):
        if key not in grp:
            return False
        grp = grp[key]
    return True


def band_plot(ax, coqui_h5, iteration=-1,
              fontsize=16, label="", verbal=True, calc_type="mbpt",
              **kwargs):
    """
    Plot the quasiparticle band structure written by ``band_interpolation`` onto a Matplotlib
    axes.

    Plots ``E_ska - mu`` (eV) along the k-path of ``band_interpolation``: for a Dyson-SCF
    iteration the bands of its ``qp_approx`` setting, for a quasiparticle self-consistent
    iteration or the mean field the iteration's own. For the bands of
    ``lqp_from_spectral_interpolation``, use ``band_plot_from_spectral_interpolation``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot into.
    coqui_h5 : str
        Path to the CoQuí HDF5 checkpoint file (``prefix.mbpt.h5``).
    iteration : int, optional
        SCF iteration to read. ``-1`` (default) selects the last available iteration;
        ``0`` corresponds to the DFT mean-field bands.
    fontsize : int, optional
        Font size for axis labels and tick labels. Default ``16``.
    label : str, optional
        Legend label for the plotted bands. Default ``""``. ``band_plot`` does not draw the
        legend: call ``ax.legend()`` once all curves are on the axes.
    verbal : bool, optional
        If ``True`` (default), prints a summary of what was read (k-points, bands, μ).
    calc_type : str, optional
        ``"mbpt"`` (default) reads the ``scf`` group, ``"dmft"`` the ``embed`` group.
    **kwargs
        Additional keyword arguments forwarded to ``ax.plot`` (e.g. ``color``,
        ``linestyle``, ``linewidth``).

    Returns
    -------
    None

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import coqui.post_proc.plot_utils as plot_utils

        fig, ax = plt.subplots()
        plot_utils.band_plot(ax, "svo.evgw.mbpt.h5", iteration=0,
                             color="tab:blue", linestyle="--", label="PBE")
        plot_utils.band_plot(ax, "svo.evgw.mbpt.h5", iteration=1,
                             color="tab:red", linestyle="-", label="evGW")
        ax.axhline(0, color="black", linewidth=1)
        ax.legend()
        plt.tight_layout()
        plt.savefig("bands.png")
    """
    iter_grp = _iteration_group(coqui_h5, calc_type, iteration)
    with HDFArchive(coqui_h5, 'r') as ar:
        # E_ska, mu and the k-path labels all sit in one group
        if _has_path(ar, f"{iter_grp}/qp_approx/wannier_inter/E_ska"):
            qp_grp = f"{iter_grp}/qp_approx"
        elif _has_path(ar, f"{iter_grp}/wannier_inter/E_ska"):
            qp_grp = iter_grp
        else:
            raise KeyError(f"No interpolated quasiparticle bands of {iter_grp} in {coqui_h5}: "
                           "run band_interpolation first.")
        mu = ar[f"{qp_grp}/mu"]
        E_ska = ar[f"{qp_grp}/wannier_inter/E_ska"]
        label_idx = ar[f"{qp_grp}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"{qp_grp}/wannier_inter/kpt_labels"]

    _draw_bands(ax, E_ska, mu, label_idx, kpt_label_str, coqui_h5, iter_grp, fontsize, label,
                verbal, **kwargs)


def band_plot_from_spectral_interpolation(ax, coqui_h5, iteration=-1,
                                          fontsize=16, label="", verbal=True,
                                          calc_type="mbpt", **kwargs):
    """
    Plot the quasiparticle band structure written by ``lqp_from_spectral_interpolation`` onto a
    Matplotlib axes.

    Plots ``E_ska - mu`` (eV) from ``iter{N}/wannier_inter/qp_approx/``, along the k-path of
    ``spectral_interpolation``. For the bands of ``band_interpolation``, use ``band_plot``.

    Parameters
    ----------
    As in ``band_plot``.

    Returns
    -------
    None

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        from coqui.post_proc import (spectral_interpolation, lqp_from_spectral_interpolation,
                                     spectral_plot, band_plot_from_spectral_interpolation)

        spectral_interpolation(mf, params)
        lqp_from_spectral_interpolation(mpi, "svo.gw.mbpt.h5", grp_name="scf", iteration=1)

        fig, ax = plt.subplots()
        spectral_plot(ax, "svo.gw.mbpt.h5", calc_type="mbpt", iteration=1)
        band_plot_from_spectral_interpolation(ax, "svo.gw.mbpt.h5", iteration=1,
                                              color="tab:red", label="lqp")
        ax.legend()
        plt.savefig("bands.png")
    """
    iter_grp = _iteration_group(coqui_h5, calc_type, iteration)
    with HDFArchive(coqui_h5, 'r') as ar:
        if not _has_path(ar, f"{iter_grp}/wannier_inter/qp_approx/E_ska"):
            raise KeyError(f"No quasiparticle bands of the Wannier-interpolated solution of "
                           f"{iter_grp} in {coqui_h5}: run spectral_interpolation, then "
                           "lqp_from_spectral_interpolation first.")
        mu = ar[f"{iter_grp}/wannier_inter/qp_approx/mu"]
        E_ska = ar[f"{iter_grp}/wannier_inter/qp_approx/E_ska"]
        label_idx = ar[f"{iter_grp}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"{iter_grp}/wannier_inter/kpt_labels"]

    _draw_bands(ax, E_ska, mu, label_idx, kpt_label_str, coqui_h5, iter_grp, fontsize, label,
                verbal, **kwargs)


def _iteration_group(coqui_h5, calc_type, iteration):
    """``{scf|embed}/iter{N}`` for ``calc_type`` and ``iteration`` (``-1`` = the last one)."""
    if calc_type not in ["mbpt", "dmft"]:
        raise ValueError(f"Unknown calc_type = {calc_type}. \n"
                         "Acceptable options are 'mbpt' for many-body perturbation theory "
                         "and 'dmft' for dmft embedding results.")
    h5_grp = "scf" if calc_type == "mbpt" else "embed"
    if iteration == -1:
        with HDFArchive(coqui_h5, 'r') as ar:
            iteration = ar[f"{h5_grp}/final_iter"]
    return f"{h5_grp}/iter{iteration}"


def _draw_bands(ax, E_ska, mu, label_idx, kpt_label_str, coqui_h5, iter_grp, fontsize, label,
                verbal, **kwargs):
    """Plot E_ska - mu [Ha] in eV along the k-path, one line per band, and label the axes."""
    kpt_label = [letter if letter != 'G' else r'$\Gamma$' for letter in kpt_label_str]

    E_ska = (E_ska - mu) * Hartree_eV
    ns, nkpts, nbnd = E_ska.shape
    if verbal:
        app_log(1, "  Plotting QP Band Structure")
        app_log(1, "  --------------------------")
        app_log(1, "  CoQui h5           = {}".format(coqui_h5))
        if iter_grp.endswith("/iter0"):
            app_log(1, "  Iteration          = {} (i.e. DFT bands)".format(iter_grp))
        else:
            app_log(1, "  Iteration          = {}".format(iter_grp))
        app_log(1, "  Number of spins    = {}".format(ns))
        app_log(1, "  Number of k-points = {}".format(nkpts))
        app_log(1, "  Number of bands    = {}".format(nbnd))
        app_log(1, f"  Chemical potential = {mu * Hartree_eV:.3f} (eV)\n")

    for i in range(nbnd):
        ax.plot(np.arange(nkpts), E_ska[0, :, i],
                label=label if i == 0 else None, **kwargs)

    ticks_pos = label_idx - 1
    ax.set_xticks(ticks_pos, kpt_label)
    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel('$\\epsilon - \\mu$ (eV)', fontsize=fontsize)
    ax.set_xlim(0, nkpts - 1)


def spectral_plot(ax, coqui_h5, calc_type, iteration=-1, orb_list=None,
                  fontsize=16, abs_A=True, verbal=True, **kwargs):
    """
    Plot the k-resolved spectral function A(k,ω) from analytic continuation results.

    Reads the Wannier-interpolated Green's function on the real-frequency axis from
    ``scf/iter{N}/ac/G_wskab_inter`` (MBPT) or ``embed/iter{N}/ac/G_wskab_inter``
    (DMFT) in the CoQuí checkpoint, sums diagonal orbital contributions to form
    A(k,ω) = −Im Tr G(k,ω), and displays it as a colormap.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot into.
    coqui_h5 : str
        Path to the CoQuí HDF5 checkpoint file.
    calc_type : str
        Source of the spectral data. ``"mbpt"`` reads from the ``scf`` group;
        ``"dmft"`` reads from the ``embed`` group.
    iteration : int, optional
        Iteration to read. ``-1`` (default) selects the last available.
    orb_list : list of int, optional
        Orbital (band) indices included in the trace. ``None`` (default) uses
        all bands.
    fontsize : int, optional
        Font size for axis labels, tick labels, and colorbar. Default ``16``.
    abs_A : bool, optional
        If ``True``, plot ``|A(k,ω)|`` instead of ``A(k,ω)``. Useful for
        diagnosing negative spectral weight from imperfect analytic continuation.
        Default ``True``.
    verbal : bool, optional
        If ``True`` (default), prints a summary (k-points, frequencies, etc.).
    **kwargs
        Additional keyword arguments forwarded to ``ax.pcolormesh``. Defaults:
        ``cmap="viridis"``, ``vmin=0.0``, ``vmax=30``, ``shading="auto"``.

    Returns
    -------
    None

    Examples
    --------
    ::

        import matplotlib.pyplot as plt
        import coqui.post_proc.plot_utils as plot_utils

        fig, ax = plt.subplots()
        plot_utils.spectral_plot(ax, "svo.gw.mbpt.h5", calc_type="mbpt",
                                 iteration=-1, vmax=25)
        plt.tight_layout()
        plt.savefig("spectral.png")
    """
    if calc_type not in ["mbpt", "dmft"]:
        raise ValueError(f"Unknown calc_type = {calc_type}. \n"
                         "Acceptable options are 'mbpt' for many-body perturbation theory "
                         "and 'dmft' for dmft embedding results.")

    h5_grp = "scf" if calc_type == "mbpt" else "embed"
    with HDFArchive(coqui_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar[f"{h5_grp}/final_iter"]
        G_wska = ar[f"{h5_grp}/iter{iteration}/ac/G_wskab_inter/output"]
        w_mesh = ar[f"{h5_grp}/iter{iteration}/ac/G_wskab_inter/w_mesh"].real
        label_idx = ar[f"{h5_grp}/iter{iteration}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"{h5_grp}/iter{iteration}/wannier_inter/kpt_labels"]

    kpt_label = [letter if letter != 'G' else r'$\Gamma$' for letter in kpt_label_str]

    nw, ns, nkpts, nbnd = G_wska.shape
    if verbal:
        app_log(1, "  Plotting spectral function")
        app_log(1, "  --------------------------")
        app_log(1, "  CoQuí h5                   = {}".format(coqui_h5))
        app_log(1, "  Calculation type           = {}".format(calc_type))
        app_log(1, "  Iteration                  = {}".format(iteration))
        app_log(1, "  Number of real frequencies = {}".format(nw))
        app_log(1, "  Number of spins            = {}".format(ns))
        app_log(1, "  Number of k-points         = {}".format(nkpts))
        if orb_list is None:
            app_log(1, "  Number of bands            = {}".format(nbnd))
        else:
            app_log(1, "  Orbital list               = {}".format(orb_list))
        app_log(1, "  Abs. A(k,w)                = {}".format(abs_A))

    spin_factor = 2.0 if ns == 1 else 1.0
    if orb_list is None:
        orb_list = np.arange(nbnd)

    A_wsk = -1.0 * np.sum(G_wska[:, :, :, orb_list].imag, axis=3)
    A_wk = np.sum(A_wsk, axis=1) * spin_factor

    _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx, fontsize, abs_A, **kwargs)


def _spectral_plot_maxent(ax, coqui_h5, iteration=-1, eta_for_A=0.001,
                         fontsize=16, abs_A=False, verbal=True, **kwargs):
    """
    Plot A(k,ω) for DMFT results using a MaxEnt-continued impurity self-energy.

    Reads the MaxEnt-continued impurity self-energy from
    ``embed/iter{N}/ac/Sigma_imp_wsa`` in the CoQuí checkpoint and uses
    ``py2aimb`` to construct the k-resolved spectral function along the stored
    Wannier k-path. The result is displayed as a false-color plot.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot into.
    coqui_h5 : str
        Path to the CoQuí HDF5 checkpoint file.
    iteration : int, optional
        DMFT iteration to read. ``-1`` (default) selects the last available.
    eta_for_A : float, optional
        Lorentzian broadening in Hartree applied when constructing A(k,ω)
        from the self-energy. Default ``0.001``.
    fontsize : int, optional
        Font size for axis labels, tick labels, and colorbar. Default ``16``.
    abs_A : bool, optional
        If ``True``, plot ``|A(k,ω)|`` instead of ``A(k,ω)``. Default ``False``.
    verbal : bool, optional
        If ``True`` (default), prints a progress summary.
    **kwargs
        Additional keyword arguments forwarded to ``ax.pcolormesh``. Defaults:
        ``cmap="viridis"``, ``vmin=0.0``, ``vmax=30``, ``shading="auto"``.

    Returns
    -------
    None
    """
    import py2aimb.dmft.pproc.spectral as spec

    with HDFArchive(coqui_h5, 'r') as ar:
        if iteration == -1:
            iteration = ar["embed/final_iter"]
        Simp_wsa = ar[f"embed/iter{iteration}/ac/Sigma_imp_wsa/output"]
        w_mesh = ar[f"embed/iter{iteration}/ac/Sigma_imp_wsa/w_mesh"]
        label_idx = ar[f"embed/iter{iteration}/wannier_inter/kpt_label_idx"]
        kpt_label_str = ar[f"embed/iter{iteration}/wannier_inter/kpt_labels"]
        kpts = ar[f"embed/iter{iteration}/wannier_inter/kpts"]

    kpt_label = [letter if letter != 'G' else r'$\Gamma$' for letter in kpt_label_str]

    nw, ns, nbnd = Simp_wsa.shape
    nkpts = kpts.shape[0]
    if verbal:
        app_log(1, "  Plotting spectral function")
        app_log(1, "  --------------------------")
        app_log(1, "  CoQuí h5                   = {}".format(coqui_h5))
        app_log(1, "  Calculation type           = dmft w/ maxent")
        app_log(1, "  Iteration                  = {}".format(iteration))
        app_log(1, "  Number of real frequencies = {}".format(nw))
        app_log(1, "  Number of spins            = {}".format(ns))
        app_log(1, "  Number of k-points         = {}".format(nkpts))
        app_log(1, "  Number of bands            = {}".format(nbnd))
        app_log(1, "  Abs. A(k,w)                = {}".format(abs_A))

    spin_factor = 2.0 if ns == 1 else 1.0
    A_wska = spec.spectral_kpath(coqui_h5, iteration, Simp_wsa, w_mesh, eta=eta_for_A, verbal=verbal)
    A_wsk = np.sum(A_wska, axis=3)
    A_wk = np.sum(A_wsk, axis=1) * spin_factor

    _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx, fontsize, abs_A, **kwargs)


def _spectral_plot(ax, A_wk, w_mesh, kpt_label, label_idx,
                   fontsize=16, abs_A=True, **kwargs):

    kwargs.setdefault('cmap', 'viridis')
    kwargs.setdefault('vmax', 30)
    kwargs.setdefault('vmin', 0.0)
    kwargs.setdefault('shading', 'auto')

    nw, nkpts = A_wk.shape

    neg_value = np.min(A_wk)
    if abs(neg_value) > 1e-3 and not abs_A:
        app_log(f"[WARNING] The spectral has negative values with maximum ~ {neg_value:.3f}. \n"
                f"          Please double check your AC setup. Otherwise, you can set abs_A=True \n"
                f"          to plot |A(k,w)| and compare it w/ A(k,w).")

    cax = ax.pcolormesh(np.arange(nkpts), w_mesh*Hartree_eV,
                        np.abs(A_wk)/Hartree_eV if abs_A else A_wk/Hartree_eV,
                        **kwargs)

    # Add a colorbar to show the scale, and set the size of the colar bar tick labels
    cbar = plt.colorbar(cax, ax=ax)
    cbar.set_label(r"$A(k,\omega)$" if not abs_A else r"$|A(k,\omega)|$", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)

    ax.tick_params(axis='both', which='major', labelsize=fontsize)
    ax.set_ylabel(r'$\epsilon - \mu$ (eV)', fontsize=fontsize)
    ax.set_xlim(0, nkpts - 1)

    # Set custom ticks
    ticks_pos = label_idx - 1
    ax.set_xticks(ticks_pos, kpt_label)
