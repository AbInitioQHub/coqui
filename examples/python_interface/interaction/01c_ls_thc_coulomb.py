"""
Example: Build LS-THC Coulomb from Cholesky-decomposed integrals

This example builds a THC Coulomb Hamiltonian via **Least-Squares THC (LS-THC)**
using a pre-computed Cholesky-decomposed Coulomb Hamiltonian.

The resulting `ThcCoulomb` instance can be passed to downstream many-body solvers,
just like the ISDF-based variant.

Notes
-----
- The Cholesky decomposition serves here only as an intermediate representation;
  LS-THC compresses it into a THC form.
- The resulting `ThcCoulomb` object is **indistinguishable in use** from one
  built directly with the ISDF algorithm
  (see `interaction/01_isdf_thc_coulomb.py`).

See also
--------
- Cholesky-decomposed Coulomb Hamiltonian: interaction/01_cholesky_coulomb.py
- ISDF-based THC Coulomb Hamiltonian (preferred): interaction/01_isdf_thc_coulomb.py
- Cholesky Coulomb from GDF using PySCF (molecule): interaction/01_pyscf_gdf_coulomb.py
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
    "tol": 1e-4,             # Cholesky decomposition integrals
    "ecut": 40               # Plane wave cutoff used for the evaluation of coulomb matrix elements.
}
svo_chol = coqui.make_chol_coulomb(mf=svo_mf, params=chol_params)

# construct THC Hamiltonian using LS-THC from the Cholesky Hamiltonian
thc_params = {
    "ecut": 40,
    "thresh": 1e-4,
    "cd_dir": "chol_coulomb" # path where the Cholesky integrals are stored
}
svo_thc = coqui.make_thc_coulomb(mf=svo_mf, params=thc_params)
