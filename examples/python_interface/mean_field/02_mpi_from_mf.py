"""
Extracting the MPI handler from an Mf object.

The Mf object carries its own copy of the MPI handler, which can be
extracted to build other CoQui objects, ensuring consistency in MPI configuration.
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

# The extracted handler can be reused to build another Mf or other
# CoQuí objects that should share the same MPI context.
mpi_from_mf = svo_mf.mpi()

# Check if the mpi context is the same
if mpi_from_mf == mpi:
  if mpi_from_mf.root():
    print("mpi_from_mf is the same as the original mpi.")
else:
  if mpi_from_mf.root():
    print("mpi_from_mf is NOT the same as the original mpi.")
