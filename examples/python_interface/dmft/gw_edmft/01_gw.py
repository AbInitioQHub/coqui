import numpy as np
import tomllib

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

beta = 20

mf_params = {"prefix": "svo", "outdir": qe_dir, "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

thc_params = {"thresh": 1e-5, "ecut": 60, "save": "thc.coulomb.h5"}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# self-consistent GW as starting point
gw_params = {
    "restart": False,
    "outdir": "./",
    "prefix": "svo",
    "beta": beta,
    "iaft": {
        "prec": "high"
    },
    "niter": 10,
    "div_treatment": "gygi_metal",
    "iter_alg": {
        "alg": "diis",
        "mixing": 0.6,
        "max_subsp_size": 6,
        "diis_warmup": 3
    }
}
coqui.run_gw(params=gw_params, h_int=thc)
