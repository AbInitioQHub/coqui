"""
Construct a THC Coulomb Hamiltonian from a mean-field object,
returning a ThcCoulomb instance that provides access to the
THC integrals for subsequent many-body solvers.
"""

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# --- Build Mf object from QE results
mf_params = {
    "prefix": "svo",      # prefix of the QE scf/nscf
    "outdir": qe_dir      # outdir of the QE scf/nscf
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# --- Direct construction of THC Coulomb Hamiltonian
thc_params = {
    "ecut": 40,
    "thresh": 1e-4,
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=thc_params)

# --- Deferred construction (init later)
thc_params["init"] = False
svo_thc2 = coqui.make_thc_coulomb(mf=svo_mf, params=thc_params)
assert not svo_thc2.initialized(), "THC integrals should not be initialized yet."

# build explicitly on demand
svo_thc2.init()
assert svo_thc2.initialized(), "THC integrals should be initialized already."

