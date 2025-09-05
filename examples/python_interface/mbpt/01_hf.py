from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

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
    "ecut": svo_mf.ecutwfc()*1.2,
    "thresh": 1e-3,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=eri_params)

# Hartree-Fock 
hf_params = {
    "beta": 300,
    "wmax": 3.0,
    "iaft_prec": "medium",
    "niter": 8,
    "output": "svo_hf",
    "iter_alg": {
        "alg": "damping",
        "mixing": 0.7
    }
}
coqui.run_hf(params=hf_params, h_int=svo_thc)

