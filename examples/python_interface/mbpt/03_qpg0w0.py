from mpi4py import MPI

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
si_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# construct thc handler and compute the thc integrals during initialization
eri_params = {
    "ecut": svo_mf.ecutwfc()*1.2,
    "thresh": 1e-3,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=eri_params)

# GW
gw_params = {
    "output": "qpg0w0",
    "niter": 1,
    "beta": 200,
    "wmax": 3.0,
    "iaft_prec": "medium",
    "qp_type": "sc",
    "ac_alg": "pade",
    "eta": 1e-6,
    "Nfit": 26
}
coqui.run_qpg0w0(params=gw_params, h_int=svo_thc)

# Wannier interpolation for G0W0
winter_params = {
        "outdir": "./",
        "prefix": "qpg0w0",
        "iteration": 1,
        "wannier_file": wan_h5, 
        "bands_num_npoints": 100, 
        "kpath": """
          W 0.50 0.25 0.75 
          G 0.00 0.00 0.00
          X 0.50 0.00 0.50
          W 0.50 0.25 0.75
          L 0.50 0.50 0.50
          G 0.00 0.00 0.00
        """
    }

band_interpolation(si_mf, winter_params)

# Wannier interpolation for PBE 
winter_params["iteration"] = 0
band_interpolation(si_mf, winter_params)

mpi.barrier()

# Plotting
if mpi.root():
  fig, ax = plt.subplots(1, figsize=(7,5.5), dpi=80)
  plot_utils.band_plot(ax, "qpg0w0.mbpt.h5", iteration=0, color='tab:blue', linestyle="--",  
                       linewidth=2.0, label='PBE', fontsize=16)
  plot_utils.band_plot(ax, "qpg0w0.mbpt.h5", iteration=1, color='tab:red', linestyle="-", 
                       linewidth=2.0, label='G0W0@PBE', fontsize=16)
  ax.axhline(y=0, color = 'black', linestyle = '-', linewidth=2.0, alpha=0.5)
  #ax.set_ylim(-10.884, 10.884)
  ax.legend(loc=1, fontsize=16)
  plt.tight_layout()
  
  plt.savefig("qpg0w0.png", format="png")

mpi.barrier()
