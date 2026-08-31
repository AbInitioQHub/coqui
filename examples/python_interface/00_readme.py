"""
The Python interface of CoQui provides a collection of thin wrappers around
core functionalities in the C++ CoQui package.

Usage
-----
To use the CoQuí Python interface, ensure to add CoQuí to the environment variables:

    source {COQUI_INSTALL_DIR}/share/coqui/setup_env.sh

The CoQuí Python interface provides thin wrappers around the core C++ functionalities of CoQuí.

MPI Requirement
---------------
The underlying C++ library requires an MPI environment, even for serial execution.
Therefore, the Python interface must be used within an initialized MPI context.
This can be done, for example, by initializing MPI through `mpi4py` or another MPI-aware
Python framework.

Parallel Execution
------------------
To run a CoQuí Python script in parallel, use:

    mpirun {MPI_FLAGS} python {YOUR_COQUI_PYTHON_SCRIPT}.py

See `01_mpi_and_verbosity.py` for a minimal example using `mpi4py`.

Quick Test
----------
Run this script directly to confirm that the CoQuí Python package can be imported and
that build info (path, version, git branch, git hash) are correct.
"""

import os
import coqui

# CoQui version 
print(f"CoQui version: {coqui.__version__}")

# CoQuí Python module path
print(f"Path for the CoQui Python library: {os.path.dirname(coqui.__file__)}")
# CoQui installation path
print(f"Path for the CoQui installation: {coqui.INSTALL_DIR}")

# CoQui git information
print(f"GIT branch: {coqui.COQUI_GIT_BRANCH}")
print(f"GIT hash: {coqui.COQUI_GIT_HASH}")
