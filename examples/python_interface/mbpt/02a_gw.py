from mpi4py import MPI
import coqui

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# construct MF from a dictionary 
mf_params = {
    "prefix": "svo",
    "outdir": qe_dir
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
    "beta": 200,
    "iaft": {
        "prec": "medium"
    },
    "niter": 1,
    "outdir": "./",
    "prefix": "svo.gw",
}
coqui.run_gw(params=gw_params, h_int = svo_thc)

# Wannier interpolation for G0W0
winter_params = {
    "outdir": "./",
    "prefix": "svo.gw",
    "iteration": 1,
    "wannier_file": wan_h5, 
    "w_min": -0.15,
    "w_max": 0.15,
    "kpath": (
      "G 0.00 0.00 0.00 "
      "X 0.00 0.50 0.00 "
      "M 0.50 0.50 0.00 "
      "G 0.00 0.00 0.00 "
    )
}
coqui.post_proc.spectral_interpolation(svo_mf, winter_params)
# extract QP energies and interpolate to k-path
coqui.post_proc.band_interpolation(svo_mf, winter_params)

if mpi.root():
    fig, ax = plt.subplots(1, figsize=(7, 5.5), dpi=100)

    # Plot spectral function A(k, ω)
    coqui.post_proc.spectral_plot(ax, coqui_h5="svo.gw.mbpt.h5", iteration=1, calc_type="mbpt")
    # Plot QP band structure
    coqui.post_proc.band_plot(ax, coqui_h5="svo.gw.mbpt.h5", iteration=1, color="tab:blue", label="qpGW")
    ax.axhline(y=0, color = 'black', linestyle = '-', linewidth=2.0, alpha=0.5)

    ax.set_ylim(-4, 4)
    ax.legend(loc=4, fontsize=12)
    plt.tight_layout()
    plt.savefig("svo.gw.png", format="png")

mpi.barrier()
