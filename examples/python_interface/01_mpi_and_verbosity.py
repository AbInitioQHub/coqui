"""
01_mpi_and_verbosity.py — Basic MPI usage with CoQuí and verbosity control

This example demonstrates:
  1) Initializing CoQuí's MPI handler, which wraps:
       - Global communicator
       - Intra-node communicator
       - Inter-node communicator
  2) Setting the global logging verbosity level

Notes
-----
- Importing `mpi4py.MPI` initializes MPI.
- Print statements are restricted to the global root to avoid duplicate output.
- `MpiHandler` is non-copyable by design (see the commented example at the end).

      import copy
      copy.copy(coqui_mpi)   # <-- will fail
      copy.deepcopy(coqui_mpi)  # <-- will also fail

"""
from mpi4py import MPI # Initialized MPI environment
import coqui

# Construct CoQuí's MPI handler (wraps global / intranode / internode comms)
coqui_mpi = coqui.MpiHandler()

# Set global logging verbosity (0 = quiet, higher = more verbose, default = 2)
coqui.set_verbosity(coqui_mpi, output_level=1)

if coqui_mpi.root():
    # Report mpi configuration
    print(coqui_mpi)

    # Access ranks
    print("[CoQuí/MPI] Global rank:", coqui_mpi.comm_rank())
    print("[CoQuí/MPI] Intra-node rank:", coqui_mpi.intranode_rank())
    print("[CoQuí/MPI] Inter-node rank:", coqui_mpi.internode_rank())
