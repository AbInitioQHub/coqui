"""
Construct a Mean-Field (Mf) object from QE outputs, which serves as
an entry point for subsequent many-body calculations in CoQuí.

This example:
- Reads SCF/NSCF results from Quantum ESPRESSO.
- Constructs a CoQuí `Mf` object, which holds single-particle states.
- Once initialized, the `Mf` object is agnostic to the underlying DFT code.
"""

from mpi4py import MPI
import coqui

qe_dir = coqui.TEST_INPUT_DIR + "qe/svo_kp222_nbnd40/out"

mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# --- Build Mf object from QE results
mf_params = {
  "prefix": "svo",     # QE SCF/NSCF prefix
  "outdir": qe_dir     # QE output directory
}
svo_mf = coqui.make_mf(mpi, params=mf_params, mf_type="qe")

# Inspect the Mf object
if mpi.root():
  # Print the states of the Mf class
  print(svo_mf)
  print("-"*20)
  # Information of the mean-field input can be extracted directly.
  print(f"mf type = {svo_mf.mf_type()}")
  print(f"nbnd    = {svo_mf.nbnd()}")
  print(f"nkpts   = {svo_mf.nkpts()}")
  print("kpts = ")
  print(svo_mf.kpts())
