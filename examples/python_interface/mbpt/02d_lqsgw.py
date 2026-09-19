from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")

import coqui
from coqui.post_proc import band_interpolation
import coqui.post_proc.plot_utils as plot_utils

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# construct MF from a dictionary
mf_params = {
    "prefix": "svo",
    "outdir": qe_dir,
    "nbnd": 40
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# construct thc handler and compute the thc integrals during initialization
eri_params = {
    "ecut": svo_mf.ecutrho()*0.4,
    "thresh": 1e-5,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=eri_params)

# GW
gw_params = {
    "outdir": "./",
    "prefix": "svo.lqsgw",
    # LQSGW is a self-consistent scheme, so let the loop run to conv_thr rather than to a
    # fixed number of iterations. This system converges in ~14 iterations with the default
    # iterative solver (DIIS; see 04_diis.py for its controls).
    "niter": 20,
    "conv_thr": 1e-5,
    "beta": 200,
    "iaft": {
        "prec": "medium"
    },
    # lqsgw linearizes Sigma(iw) on the Matsubara axis: Sigma(iw) ~ A + iw B, then
    # H_QP = Z^1/2 (H0 + V_HF + A - mu) Z^1/2 + mu with Z = (1 - B)^-1. No analytic
    # continuation is involved, so there is no qp_eqn block here.
    "lqp": {
        "n_fit": 6,           # lowest positive fermionic frequencies in the fit window
        "fit_order": -1,      # -1 = exactly-determined (interpolating) fit
        "fit_resid_tol": 1e-8 # residual gate of that fit; aborts if conditioning is lost
    },
}
coqui.run_lqsgw(params=gw_params, h_int=svo_thc)

# Wannier interpolation for LQSGW
winter_params = {
    "outdir": "./",
    "prefix": "svo.lqsgw",
    "iteration": -1,          # -1 = the last iteration of the LQSGW loop
    "wannier_file": wan_h5,
    "bands_num_npoints": 100,
    "kpath": """
      G 0.00 0.00 0.00
      X 0.00 0.50 0.00
      M 0.50 0.50 0.00
      G 0.00 0.00 0.00
    """
}

band_interpolation(svo_mf, winter_params)

# Wannier interpolation for PBE
winter_params["iteration"] = 0
band_interpolation(svo_mf, winter_params)

mpi.barrier()

# Pole weights Z_a^k = <v_a|Z|v_a>, written alongside the QP energies of each iteration.
# They are a diagnostic of the linearization: Z in (0, 1], and the smaller Z, the more
# spectral weight the linearized quasiparticle has left behind in the incoherent part.
if mpi.root():
  from h5 import HDFArchive
  with HDFArchive("svo.lqsgw.mbpt.h5", "r") as ar:
      it = ar["scf"]["final_iter"]
      Z = np.asarray(ar["scf"][f"iter{it}"]["Z_ska"])
  print(f"LQSGW pole weights at iteration {it}: "
        f"Z in [{Z.min():.4f}, {Z.max():.4f}], mean {Z.mean():.4f}")

mpi.barrier()

# Plotting
if mpi.root():
  fig, ax = plt.subplots(1, figsize=(7,5.5), dpi=80)
  plot_utils.band_plot(ax, "svo.lqsgw.mbpt.h5", iteration=0, color='tab:blue', linestyle="--",
                       linewidth=2.0, label='PBE', fontsize=16)
  plot_utils.band_plot(ax, "svo.lqsgw.mbpt.h5", iteration=-1, color='tab:red', linestyle="-",
                       linewidth=2.0, label='lqsgw', fontsize=16)
  ax.axhline(y=0, color = 'black', linestyle = '-', linewidth=2.0, alpha=0.5)
  #ax.set_ylim(-10.884, 10.884)
  ax.legend(loc=1, fontsize=16)
  plt.tight_layout()

  plt.savefig("svo.lqsgw.png", format="png")

mpi.barrier()
