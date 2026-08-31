import tomllib

import triqs.utility.mpi as mpi
import triqs_modest as modest

import coqui
import coqui.dmft as coqui_dmft

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=2)

# Define correlated subspace via ModEST
obe = modest.make_one_body_elements_gw(wan_h5)
E1 = modest.make_embedding(obe.C_space)

# Update the rotation matrix
proj_info = coqui_dmft.get_proj_info(obe.P)

# read input parameters 
with open("gw_edmft_params.toml", "rb") as f:
    coqui_params = tomllib.load(f)

mf_params = {"prefix": "svo", "outdir": qe_dir, "nbnd": 40}
mf = coqui.make_mf(coqui_mpi, params=mf_params, mf_type='qe')

thc_params = {"thresh": 1e-5, "ecut": 60, "save": "thc.coulomb.h5"}
thc = coqui.make_thc_coulomb(mf=mf, params=thc_params)

# GW+EDMFT driver
coqui_dmft.run_gw_edmft(thc, E1, proj_info=proj_info, params=coqui_params["gw_edmft"])
