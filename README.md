CoQui: Correlated Quantum Interface
-----------------------------------------------
**Last Updated:** Sept. 2, 2025

CoQuí (**Cor**related **Qu**antum **í**nterface) is a software project 
designed for *ab initio* electronic structure beyond density functional 
theory (DFT). 
It provides scalable implementations of correlated methods based on 
the tensor hypercontraction (THC) representation of Coulomb integrals 
for both molecular and solid-state systems. 

CoQuí can be used in **two interfaces**:

1. **Python interface**  
   Build with `-DCOQUI_PYTHON_SUPPORT=ON`, then import CoQUí as a module 
   to prepare inputs, launch runs, and post-process results.
2. **Input-file interface**  
   Provide an input toml file and run CoQuí from the command line.

## What does CoQui do?
CoQui utilizes distributed linear algebra to enable high-performance
*ab-initio* calculations applicable to:
- Both k-point (periodic) and molecular systems
- Generic single-particle basis sets, such as Kohn-Sham (KS) orbitals,
  Gaussian-type orbitals, and their mixtures.

Currently, `CoQui` interfaces with the following backends 
(see [examples/dft_converter](examples/dft_converter) for input preparation): 
- [Quantum ESPRESSO](https://www.quantum-espresso.org)
- [PySCF](https://pyscf.org)

Below are some key features of `CoQui`. For more detailed examples, 
please visit our [examples](examples/README.md) page.
#### Compressed Representation for Many-Body Hamiltonians
- THC representation for two-electron Coulomb integrals 
  [[ref1](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615),
  [ref2](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]. 
- Cholesky decomposition for two-electron Coulomb integrals. 

#### Many-Body Perturbation Theory
- Hartree-Fock [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615)]
- RPA correlation energy [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615)]
- GW approximation [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]
- Second-order exchange (SOX) diagram [[ref](https://arxiv.org/abs/2404.17744)]
- Self-consistency with quasiparticle approximation 
- Self-consistency with full frequency dependence

#### Downfolding for effective low-energy Hamiltonians
- Maximally localized Wannier functions via Wannier90 interface 
- Constrained RPA to calculate screened interactions [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]
- Local effective low-energy Hamiltonian for further correlated calculations
  [[ref](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085)]

## Getting started with `CoQui` 
### Prerequisites
- C++ compiler that supports at least C++20.
- CMake >= 3.2.0.
- MPI Library: openmpi >= 4. 
- HDF5 >= 1.8.2 for checkpoint file I/O. 
- BLAS Library: OpenBLAS or Intel MKL. 
- LAPACK Library: OpenBLAS or Intel MKL. 
- [SLATE](https://github.com/icl-utk-edu/slate/tree/master) Library for distributed linear algebra.  
- Boost >= 1.68.0
- FFTW >= 3.2

### Installation
`CoQui` uses `CMake` to configure the build process. Follow 
the instructions below step-by-step, and replace the placeholders in 
square brackets (`[]`) with your local settings.

```shell
# Step 1: Clone the git repository of CoQui
git clone https://github.com/AbInitioQHub/coqui.git coqui.src

# Step 2: Create working directory for CMake to build in
mkdir -p coqui.build && cd coqui.build

# Step 3: Configure with CMake
# Replace `[YOUR_INSTALL_PREFIX]` with the directory where you want CoQui installed.
# Replace `[NCORES]` with the number of cores you want to use for the test processes.
# Replace `[SLATE_INSTALL_PATH]` with your SLATE installation path. 
# Add `COQUI_PYTHON_SUPPORT=ON
export slate_ROOT=[SLATE_INSTALL_PATH]
cmake \
        -DCMAKE_INSTALL_PREFIX=[YOUR_INSTALL_PREFIX] \
        -DCTEST_NPROC=[NCORES] \
        -DCOQUI_PYTHON_SUPPORT=ON \ # Optional: enable Python bindings
        ../coqui.src

# Step 4: Build, test and install
# Replace `[NCORES_MAKE] with the number of cores you want to use for the build processes. 
# The ctests will be executed in parallel using `[NCORES]` processors.
make -j[NCORES_MAKE] && ctest && make install

# Verify: the 'coqui' executable should be in [YOUR_INSTALL_PREFIX]/bin
ls -l [YOUR_INSTALL_PREFIX]/bin/coqui

# Step 5: Set CoQui environment 
# You would need to source this in every new shell, or add 
# this line to your ~/.bashrc or ~/.zshrc to make it persistent.
source [YOUR_INSTALL_PREFIX]/share/coqui/coqui_env.sh
```

### Tutorials and Examples
- **Quick start:** See the step-by-step notebooks in the
  [coqui tutorial](https://github.com/AbInitioQHub/coqui-tutorial).
- **Reference inputs:** Browse runnable cases in [examples](examples/README.md).

## Citation
If you use `CoQui` in your research, please consider supporting our developers 
by citing the following papers:

[1] C.-N. Yeh, M. Morales, Low-Scaling Algorithm for the Random Phase
Approximation Using Tensor Hypercontraction with k-point Sampling,
[J. Chem. Theory Comput. 19, 18, 6197–6207 (2023)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.3c00615).

[2] C.-N. Yeh, M. Morales, Low-Scaling Algorithms for GW and Constrained Random Phase
Approximation Using Symmetry-Adapted Interpolative Separable Density Fitting,
[J. Chem. Theory Comput. 20, 8, 3184–3198 (2024)](https://pubs.acs.org/doi/abs/10.1021/acs.jctc.4c00085). 
