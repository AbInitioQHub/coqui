"""
Quasiparticle band structure and band-resolved renormalization along a k-path, from the
linearized quasiparticle approximation of the Wannier-interpolated G0W0 solution:

  Sigma(iw) ~ A + iw B,   Z = (1 - B)^-1,   H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu,

fitted on the lowest Matsubara frequencies, so no analytic continuation is involved. The pole
weight <v|Z|v> of each quasiparticle band is the spectral weight its pole carries.

Run 02a_full_freq_gw.py first: this script post-processes its checkpoint, svo.gw.mbpt.h5, and
uses the Wannier-interpolated self-energy and spectral function written by its
spectral_interpolation call.
"""

from mpi4py import MPI
import os
import numpy as np
from h5 import HDFArchive
import coqui
from coqui.post_proc import linearized_qp_from_checkpoint

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")

coqui_h5 = "svo.gw.mbpt.h5"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)
if not os.path.isfile(coqui_h5):
    raise FileNotFoundError(f"{coqui_h5} not found: run 02a_full_freq_gw.py first.")

# Linearized QP of the Wannier-interpolated solution in scf/iter1/wannier_inter/. The fit window
# is chosen automatically: it grows until the fit stops being reliable, and the last reliable
# fit is kept. Results are written to scf/iter1/wannier_inter/qp_approx/.
linearized_qp_from_checkpoint(mpi, coqui_h5, grp_name="scf", iteration=1, source="interpolated")

if mpi.root():
    with HDFArchive(coqui_h5, "r") as ar:
        Zqp_ska = ar["scf/iter1/wannier_inter/qp_approx/Zqp_ska"]
        label_idx = ar["scf/iter1/wannier_inter/kpt_label_idx"]
        kpt_labels = ar["scf/iter1/wannier_inter/kpt_labels"]
    ns, nkpts, nbnd = Zqp_ska.shape
    kpts = np.arange(nkpts)
    ticks = ([idx - 1 for idx in label_idx],
             [r"$\Gamma$" if label == "G" else label for label in kpt_labels])

    fig, (ax_bands, ax_weights) = plt.subplots(2, 1, figsize=(7, 9), dpi=100,
                                               height_ratios=(3, 2))

    # top: quasiparticle bands on top of the G0W0 spectral function A(k, w)
    coqui.post_proc.spectral_plot(ax_bands, coqui_h5=coqui_h5, iteration=1, calc_type="mbpt")
    coqui.post_proc.band_plot(ax_bands, coqui_h5=coqui_h5, iteration=1, qp_source="interpolated",
                              color="tab:red", label="lqp")
    ax_bands.axhline(y=0, color="black", linestyle="-", linewidth=2.0, alpha=0.5)
    ax_bands.set_ylim(-4, 4)
    ax_bands.legend(loc=4, fontsize=12)

    # bottom: pole weights of the quasiparticle bands, in the order of their energies
    for band in range(nbnd):
        ax_weights.plot(kpts, Zqp_ska[0, :, band], label=f"band {band + 1}")
    ax_weights.set_xlim(0, nkpts)
    ax_weights.set_xticks(*ticks)
    ax_weights.set_ylabel(r"pole weight $\langle v|Z|v \rangle$", fontsize=16)
    ax_weights.tick_params(axis="both", which="major", labelsize=16)
    ax_weights.legend(fontsize=12)

    plt.tight_layout()
    # align the bottom panel with the top one, which shares its width with the colorbar of A(k, w)
    top, bottom = ax_bands.get_position(), ax_weights.get_position()
    ax_weights.set_position([top.x0, bottom.y0, top.width, bottom.height])
    plt.savefig("svo.gw.lqp_renormalization.png", format="png")

mpi.barrier()
