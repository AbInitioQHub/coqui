"""
Precompute the THC Coulomb Hamiltonian and reuse many times
"""

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# construct MF from a pre-computed QE results
mf_params = {
    "prefix": "svo",      # prefix of the QE scf/nscf
    "outdir": qe_dir      # outdir of the QE scf/nscf
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# build a THC Coulomb Hamiltonian, and save the integrals to "thc.coulomb.h5"
# TODO rename "save" to "thc_checkpoint"
thc_params = {
    "save": "thc.coulomb.h5",
    "ecut": 40,
    "thresh": 1e-4,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=thc_params)

# Now that the THC integrals are saved in "thc.coulomb.h5", we can reuse it
# many times without recomputing the integrals.
svo_thc_read = coqui.make_thc_coulomb(mf=svo_mf, params=thc_params)
