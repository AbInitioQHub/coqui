from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"
wan_h5 = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/mlwf/svo.mlwf.h5"

coqui_mpi = coqui.MpiHandler()
coqui.set_verbosity(coqui_mpi, output_level=1)

# construct MF from a dictionary
mf_params = {
    "prefix": "svo",
    "outdir": qe_dir
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# construct thc handler and compute the thc integrals during initialization
eri_params = {
    "ecut": svo_mf.ecutwfc()*1.2,
    "thresh": 1e-3,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=eri_params)

crpa_params = {
    "screen_type": "crpa", 
    "prefix": "svo_crpa",
    "wannier_file": wan_h5, 
    "input_type": "mf", 
    "beta": 200,
    "wmax": 2.0,
    "iaft_prec": "medium",
}
Vloc, Wloc = coqui.downfold_local_coulomb(params=crpa_params, h_int=svo_thc)


