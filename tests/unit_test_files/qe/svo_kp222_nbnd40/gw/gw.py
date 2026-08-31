import numpy as np
import tomllib

import triqs.utility.mpi as mpi

import coqui
import coqui.dmft as coqui_dmft

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

beta, wmax = 20, 3.0

with open("gw_edmft_params.toml", "rb") as f:
    coqui_params = tomllib.load(f)

mf = coqui.make_mf(coqui_mpi, coqui_params['mean_field']['qe'], mf_type='qe')
thc = coqui.make_thc_coulomb(mf=mf, params=coqui_params['interaction']['thc'])

# self-consistent GW as starting point
gw_params = {
    "restart": False,
    "output": coqui_params["gw"]["output"],
    "beta": beta,
    "wmax": wmax,
    "iaft_prec": "high",
    "niter": 10,
    "div_treatment": "gygi_extrplt",
    "iter_alg": {
        "alg": "diis",
        "mixing": 0.6,
        "max_subsp_size": 6,
        "diis_warmup": 3
    }
}
coqui.run_gw(params=gw_params, h_int=thc)
