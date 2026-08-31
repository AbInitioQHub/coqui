"""
Example: Construct a Cholesky Coulomb Hamiltonian from a mean-field object

This example builds a Cholesky-decomposed Coulomb Hamiltonian and returns a
`CholCoulomb` instance that provides access to the Cholesky integrals for
the following many-body solvers.

Note
--------
- Cholesky integrals in CoQuí are provided mainly for **debugging and validation**.
- They are computationally **much more expensive and less scalable** than the
  THC representation, and are generally not recommended for production calculations.
- Consequently, **not all features available in THC-based electronic structure
  simulations are supported for Cholesky-based Coulomb integrals**.

See also
--------
- Molecules via GDF export (same output format): interaction/20_GDF_coulomb_from_pyscf.py
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

# construct Cholesky Hamiltonian and compute the thc integrals during initialization
chol_params = {
    "path": "chol_coulomb",  # path where the Cholesky integrals are stored
    "tol": 1e-4,             # chol tolerance that controls the number of Cholesky vectors
    "ecut": 40               # plane wave cutoff used for the evaluation of coulomb matrix elements.
}
svo_chol = coqui.make_chol_coulomb(mf=svo_mf, params=chol_params)
