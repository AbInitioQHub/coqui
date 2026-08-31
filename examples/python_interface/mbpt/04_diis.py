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

# GW
gw_params = {
    "beta": 200,
    "iaft": {
        "wmax": 3.0,
        "prec": "medium"
    },
    "niter": 10,
    "outdir": "./",
    "prefix": "svo_gw",
    "iter_alg": {
        "alg": "diis",       # Use DIIS algorithm for self-consistency
        "max_subsp_size": 6, # maximum size of the DIIS subspace
        "diis_warmup": 3,    # number of initial iterations before DIIS is turned on
        "mixing": 0.7,       # mixing parameter before DIIS is turned on
    }
}
coqui.run_gw(params=gw_params, h_int = svo_thc)
