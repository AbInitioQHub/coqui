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
import numpy as np
from h5 import HDFArchive


def plot_edmft_convergence(dmft_chkpt, *, impurity_index=0, check_w=True,
                           figure_name="edmft"):
  """Plot EDMFT convergence diagnostics for a single impurity.

  Reads the convergence history from the DMFT checkpoint file and produces two
  separate multi-panel figures:

  1. ``{figure_name}_imp{impurity_index}.png`` – *self-consistency convergence*.
     These residuals should decay towards zero as the loop converges:

     - ``|Gloc - Gimp|``  – fermionic self-consistency residual
     - ``|dg_weiss|``     – change in fermionic Weiss field between iterations
     - ``|du_weiss|``     – change in bosonic Weiss field between iterations
     - ``mu_imp``         – impurity chemical potential correction
     - ``|Wloc - Wimp|``  – bosonic self-consistency residual (plotted only when
                            ``check_w=True`` and at least one stored value is valid)

  2. ``{observables_name}_imp{impurity_index}.png`` – *observable convergence*.
     These quantities should approach a constant as the loop converges:

     - ``n_imp``          – total impurity density
     - ``U(w=0)``         – orbital-averaged static local interaction at w=0
     - ``A(w=0)``         – orbital-averaged spectral weight at tau=beta/2
     - ``Sigma_w1``       – orbital-averaged Im[Sigma(iw_1)] (plotted only when
                            stored, i.e. the convergence array has >= 7 entries)

     The observables filename is derived from ``figure_name`` by replacing the
     substring ``"convergence"`` with ``"observables"`` (or, if that substring
     is absent, by appending ``"_observables"``).

  All convergence data are read from
  ``dmft/iter{i}/impurity_{impurity_index}/results/convergence`` (a numpy array
  ``[U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w]`` with an optional
  trailing ``Sigma_w1`` entry), together with the scalar datasets
  ``dmft/iter{i}/impurity_{impurity_index}/results/mu_imp`` and
  ``dmft/iter{i}/impurity_{impurity_index}/results/density``.

  Parameters
  ----------
  dmft_chkpt : str
    Path to the DMFT HDF5 checkpoint file.
  impurity_index : int, optional
    Index of the impurity to plot. Default is 0.
  check_w : bool, optional
    Whether to include the ``|Wloc - Wimp|`` panel in the convergence figure.
    Even when ``True``, the panel is suppressed if all stored ``diff_w`` values
    equal ``-1.0``. Default is ``True``.
  figure_name : str, optional
    Base output filename for the convergence figure.  The string
    ``_imp{impurity_index}`` is always appended before the extension so that
    different impurities do not overwrite each other.  The observables figure
    name is derived from this value.  Default is ``"edmft_convergence"``.

  Returns
  -------
  tuple of str
    ``(convergence_file, observables_file)`` – the filenames the two figures
    were saved to.
  """
  try:
    # Use the object-oriented Figure API instead of pyplot: pyplot keeps a
    # global figure registry (leaks figures unless plt.close() is called) and
    # picks a process-wide backend on import. As an importable library helper
    # we must not touch that global state. See _new_figure() below.
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
  except ImportError as e:
    raise ImportError(
      "plot_edmft_convergence requires matplotlib. "
      "Install it with: pip install matplotlib"
    ) from e

  conv_file = f"{figure_name}_convergence_imp{impurity_index}.png"
  obs_file = f"{figure_name}_observables_imp{impurity_index}.png"

  # ---- Read checkpoint -------------------------------------------------------
  iters             = []
  U_w0_list         = []
  A_w0_list         = []
  diff_g_list       = []
  diff_g_weiss_list = []
  diff_u_weiss_list = []
  diff_w_list       = []
  sigma_w1_list     = []
  mu_imp_list       = []
  density_list      = []

  with HDFArchive(dmft_chkpt, 'r') as ar:
    if "dmft" not in ar.keys():
      raise KeyError(f"No 'dmft' group found in {dmft_chkpt}.")
    dmft_grp = ar["dmft"]

    # Collect all iter{i} groups in order
    iter_keys = sorted(
      [k for k in dmft_grp.keys() if k.startswith("iter")],
      key=lambda k: int(k[4:])
    )
    if not iter_keys:
      raise KeyError(f"No 'iter*' groups found under 'dmft' in {dmft_chkpt}.")

    imp_key = f"impurity_{impurity_index}"
    for ik in iter_keys:
      iter_grp = dmft_grp[ik]
      if imp_key not in iter_grp.keys():
        continue
      res_grp = iter_grp[f"{imp_key}/results"]

      if "convergence" not in res_grp.keys():
        continue

      # [U_w0, A_w0, diff_g, diff_g_weiss, diff_u_weiss, diff_w, (Sigma_w1)]
      conv = np.asarray(res_grp["convergence"])
      iters.append(int(ik[4:]))
      U_w0_list.append(conv[0])
      A_w0_list.append(conv[1])
      diff_g_list.append(conv[2])
      diff_g_weiss_list.append(conv[3])
      diff_u_weiss_list.append(conv[4])
      diff_w_list.append(conv[5])
      # Sigma_w1 is optional (only stored by some solver variants); NaN otherwise
      sigma_w1_list.append(conv[6] if conv.shape[0] >= 7 else np.nan)

      # mu_imp (optional; default 0.0 when absent)
      mu_val = float(res_grp["mu_imp"]) if "mu_imp" in res_grp.keys() else 0.0
      mu_imp_list.append(mu_val)

      # density (optional; -1.0 sentinel when absent)
      density_val = float(res_grp["density"]) if "density" in res_grp.keys() else -1.0
      density_list.append(density_val)

  if not iters:
    raise RuntimeError(
      f"No convergence data found for impurity {impurity_index} in {dmft_chkpt}. "
      "Ensure the DMFT loop has completed at least one iteration."
    )

  iters             = np.array(iters)
  U_w0_arr          = np.array(U_w0_list)
  A_w0_arr          = np.array(A_w0_list)
  diff_g_arr        = np.array(diff_g_list)
  diff_g_weiss_arr  = np.array(diff_g_weiss_list)
  diff_u_weiss_arr  = np.array(diff_u_weiss_list)
  diff_w_arr        = np.array(diff_w_list)
  sigma_w1_arr      = np.array(sigma_w1_list)
  mu_imp_arr        = np.array(mu_imp_list)
  density_arr       = np.array(density_list)

  fontsize, markersize = 18, 10

  def _new_figure(n_panels):
    """Create a standalone Agg-backed figure.

    Uses the object-oriented ``Figure`` API with an explicitly attached Agg
    canvas so that nothing touches matplotlib's global state (active backend,
    pyplot figure registry).  This keeps the helper safe to call from any
    session, interactive or headless, without side effects.
    """
    fig = Figure(figsize=(12, 2.2 * n_panels))
    FigureCanvasAgg(fig)
    axes = fig.subplots(n_panels, 1, sharex=True)
    if n_panels == 1:
      axes = [axes]
    return fig, axes

  def _finalize(fig, axes):
    """Apply shared x-axis ticks/labels and tick sizes, then save."""
    iter_min, iter_max = iters[0], iters[-1]
    axes[-1].set_xticks(np.arange(iter_min, iter_max + 1, max(1, (iter_max - iter_min) // 5)))
    axes[-1].set_xlabel("DMFT iteration", fontsize=fontsize)
    for ax in axes:
      ax.tick_params(axis='both', which='major', labelsize=fontsize - 2)
    fig.tight_layout()

  # ===========================================================================
  # Figure 1: self-consistency convergence (residuals should decay to zero)
  # ===========================================================================
  show_w = check_w and np.any(diff_w_arr > -0.5)   # at least one valid diff_w

  n_conv = 4               # diff_g, diff_g_weiss, diff_u_weiss, mu_imp
  if show_w:
    n_conv += 1

  fig_conv, axes_conv = _new_figure(n_conv)

  panel = 0

  # |Gloc - Gimp|
  ax = axes_conv[panel]; panel += 1
  ax.semilogy(iters, diff_g_arr, 'o-', markersize=markersize, color='C0')
  ax.set_ylabel(r"$|G_{\rm loc} - G_{\rm imp}|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # |dg_weiss|
  ax = axes_conv[panel]; panel += 1
  valid = diff_g_weiss_arr > -0.5
  if np.any(valid):
    ax.semilogy(iters[valid], diff_g_weiss_arr[valid], 'o-', markersize=markersize, color='C1')
  ax.set_ylabel(r"$|\Delta g|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # |du_weiss|
  ax = axes_conv[panel]; panel += 1
  valid = diff_u_weiss_arr > -0.5
  if np.any(valid):
    ax.semilogy(iters[valid], diff_u_weiss_arr[valid], 'o-', markersize=markersize, color='C2')
  ax.set_ylabel(r"$|\Delta u|$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # mu_imp
  ax = axes_conv[panel]; panel += 1
  ax.plot(iters, mu_imp_arr, 'o-', markersize=markersize, color='C3')
  ax.axhline(0.0, color='k', linewidth=0.8, linestyle='--')
  ax.set_ylabel(r"$\mu_{\rm imp}$ (a.u.)", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # |Wloc - Wimp| (optional)
  if show_w:
    ax = axes_conv[panel]; panel += 1
    valid = diff_w_arr > -0.5
    ax.semilogy(iters[valid], diff_w_arr[valid], 'o-', markersize=markersize, color='C4')
    ax.set_ylabel(r"$|W_{\rm loc} - W_{\rm imp}|$", fontsize=fontsize)
    ax.grid(True, alpha=0.3, linestyle='--')

  _finalize(fig_conv, axes_conv)
  fig_conv.savefig(conv_file, dpi=120, bbox_inches='tight')

  # ===========================================================================
  # Figure 2: observable convergence (quantities should approach a constant)
  # ===========================================================================
  show_sigma = np.any(np.isfinite(sigma_w1_arr))

  n_obs = 3                # density, U(w=0), A(w=0)
  if show_sigma:
    n_obs += 1

  fig_obs, axes_obs = _new_figure(n_obs)

  panel = 0

  # impurity total density
  ax = axes_obs[panel]; panel += 1
  valid = density_arr > -0.5
  if np.any(valid):
    ax.plot(iters[valid], density_arr[valid], 'o-', markersize=markersize, color='C5')
  ax.set_ylabel(r"$n_{\rm imp}$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # U(w=0)
  ax = axes_obs[panel]; panel += 1
  ax.plot(iters, U_w0_arr, 'o-', markersize=markersize, color='C6')
  ax.set_ylabel(r"$U(\omega=0)$ (a.u.)", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # A(w=0)
  ax = axes_obs[panel]; panel += 1
  ax.plot(iters, A_w0_arr, 'o-', markersize=markersize, color='C7')
  ax.set_ylabel(r"$A(\omega=0)$", fontsize=fontsize)
  ax.grid(True, alpha=0.3, linestyle='--')

  # Sigma_w1 (optional)
  if show_sigma:
    ax = axes_obs[panel]; panel += 1
    valid = np.isfinite(sigma_w1_arr)
    ax.plot(iters[valid], sigma_w1_arr[valid], 'o-', markersize=markersize, color='C8')
    ax.set_ylabel(r"${\rm Im}\,\Sigma(i\omega_1)$", fontsize=fontsize)
    ax.grid(True, alpha=0.3, linestyle='--')

  _finalize(fig_obs, axes_obs)
  fig_obs.savefig(obs_file, dpi=120, bbox_inches='tight')

  return conv_file, obs_file