"""

Example: Constructing THC Coulomb Hamiltonian from QE mean-field object

CoQuí supports three entry points for Coulomb Hamiltonians:
1. THC (Tensor Hypercontraction) integrals built internally (this example, preferred).
2. Cholesky-decomposed integrals built internally (mainly for debugging).
3. GDF integrals generated externally by PySCF.

This examples demonstrates how to construct a THC Coulomb Hamiltonian
using interpolative separable density fitting (ISDF) from a mean-field
object.

The output is a ThcCoulomb instance that provides access to the THC
integrals in the following many-body solvers.


See also
--------
- Cholesky-decomposed Coulomb Hamiltonian: interaction/01_cholesky_coulomb.py
- Least-Squares THC: interactions/01_ls_thc_coulomb.py
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

