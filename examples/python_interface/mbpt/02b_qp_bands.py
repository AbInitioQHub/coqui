"""
Quasiparticle band structure of a G0W0 solution with the two quasiparticle approximations of
band_interpolation:

  qp_approx = "qp_eqn" : the diagonal quasiparticle equation solved on the real axis, after
                         analytic continuation of Sigma;
  qp_approx = "lqp"    : the linearized quasiparticle approximation on the Matsubara axis,
                         Sigma(iw) ~ A + iw B, H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu with
                         Z = (1 - B)^-1, and no analytic continuation.

Run 02a_full_freq_gw.py first: this script post-processes its checkpoint, svo.gw.mbpt.h5.
"""

from mpi4py import MPI
import os
import coqui

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"
coqui_h5 = "svo.gw.mbpt.h5"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)
if not os.path.isfile(coqui_h5):
    raise FileNotFoundError(f"{coqui_h5} not found: run 02a_full_freq_gw.py first.")

# mean-field object, for the metadata of the system
svo_mf = coqui.make_mf(mpi, params={"prefix": "svo", "outdir": qe_dir}, mf_type="qe")

winter_params = {
    "outdir": "./",
    "prefix": "svo.gw",
    "iteration": 1,
    "wannier_file": wan_h5,
    "kpath": (
      "G 0.00 0.00 0.00 "
      "X 0.00 0.50 0.00 "
      "M 0.50 0.50 0.00 "
      "G 0.00 0.00 0.00 "
    )
}

# Post-process the full-frequency GW solution with two quasiparticle schemes, qp_eqn and lqp.
# Both read the GW solution in scf/iter1 as input and write their quasiparticle solution to the
# same checkpoint, under the same group scf/iter1/qp_approx/, so the second overwrites the first.
# Each band structure is therefore plotted before the other scheme is run.
if mpi.root():
    fig, ax = plt.subplots(1, figsize=(7, 5.5), dpi=100)

# 1. quasiparticle equation (the default)
coqui.post_proc.band_interpolation(svo_mf, {**winter_params, "qp_approx": "qp_eqn",
                                            "qp_eqn": {"solver": "sc", "ac_nfit": 18}})
if mpi.root():
    # plot the Wannier-interpolated quasiparticle bands, E - mu along the k-path, 
    # read from scf/iter1/qp_approx/wannier_inter/
    coqui.post_proc.band_plot(ax, coqui_h5=coqui_h5, iteration=1, color="tab:blue",
                              linestyle="--", label="qp_eqn")
mpi.barrier()

# 2. linearized quasiparticle approximation
coqui.post_proc.band_interpolation(svo_mf, {**winter_params, "qp_approx": "lqp",
                                            "lqp": {"n_fit": 6}})
if mpi.root():
    coqui.post_proc.band_plot(ax, coqui_h5=coqui_h5, iteration=1, color="tab:red",
                              label="lqp")
    ax.axhline(y=0, color='black', linestyle='-', linewidth=2.0, alpha=0.5)
    ax.set_ylim(-4, 4)
    ax.legend(loc=4, fontsize=12)
    plt.tight_layout()
    plt.savefig("svo.gw.qp_bands.png", format="png")
mpi.barrier()
