from mpi4py import MPI
import coqui
from coqui.utils.imag_axes_ft import IAFT

from h5 import HDFArchive

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=2)

# construct MF from a dictionary 
mf_params = {
    "prefix": "pyscf", 
    "outdir": "/mnt/home/cyeh/Projects/coqui/tests/unit_test_files/pyscf/si_kp222_krhf/",
}
my_mf = coqui.make_mf(mpi, mf_params, "pyscf")

# construct thc handler and compute the thc integrals during initialization
eri_params = {
    #"save": "thc.eri.h5", 
    "thresh": 1e-6, 
    "init": True
}
my_eri = coqui.make_thc_coulomb(my_mf, params=eri_params)

# GW
beta, wmax, prec = 1000, 1.2, "medium"
gw_params = {
    "beta": beta,
    "wmax": wmax, 
    "iaft_prec": prec, 
    "niter": 1, 
    "restart": False, 
    "output": "gw", 
} 
coqui.run_gw(h_int=my_eri, params=gw_params)


#########

ir = IAFT(beta=beta, wmax=wmax, prec=prec, basis='ir')
with HDFArchive("gw.mbpt.h5", 'r') as ar:
    Gt = ar["scf/iter1/G_tskij"]
    Gt_hf = ar["scf/iter0/G_tskij"]
Gw = ir.tau_to_w(Gt, stats='f')
Dm = -ir.tau_interpolate(Gt, [beta], stats='f')[0]

with HDFArchive(f"gw_Gw_Gt_beta{beta}_wmax{wmax}_{prec}.h5", 'w') as ar:
    ar["Gt"] = Gt
    ar["Gw"] = Gw
    ar["Dm"] = Dm

Gw_hf = ir.tau_to_w(Gt_hf, stats='f')
Dm_hf = -ir.tau_interpolate(Gt_hf, [beta], stats='f')[0]
with HDFArchive(f"hf_Gw_Gt_beta{beta}_wmax{wmax}_{prec}.h5", 'w') as ar:
    ar["Gt"] = Gt_hf
    ar["Gw"] = Gw_hf
    ar["Dm"] = Dm_hf
