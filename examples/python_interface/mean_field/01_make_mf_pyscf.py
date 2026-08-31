from mpi4py import MPI
import coqui

pyscf_dir = coqui.TEST_INPUT_DIR + "pyscf/si_kp222_krhf/"

# mpi handler 
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# construct Mf from a pre-computed PySCF results
mf_params = {
  "prefix": "pyscf",      # prefix for pyscf converter
  "outdir": pyscf_dir     # directory for pyscf results
}
si_mf = coqui.make_mf(mpi, params=mf_params, mf_type="pyscf")

# Information of the mean-field input can be extracted directly.
if mpi.root():
  # Print the states of the Mf class
  print(si_mf)
  print("-"*20)
  # Information of the mean-field input can be extracted directly.
  print(f"mf type = {si_mf.mf_type()}")
  print(f"nbnd    = {si_mf.nbnd()}")
  print(f"nkpts   = {si_mf.nkpts()}")
  print("kpts = ")
  print(si_mf.kpts())

