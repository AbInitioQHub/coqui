from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

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


# construct MLWFs via calling Wannier90 in the library mode
w90_params = {
    "prefix": "svo",     # equivalent to wannier90's seedname
}
coqui.wannier90(mf=svo_mf, params=w90_params)

