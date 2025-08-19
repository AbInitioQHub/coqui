Quantum ESPRESSO Converter
-----------------------------------------------

**Last Updated:** May 17, 2024

This tutorial demonstrates how to prepare mean-field input data for `CoQui`
using `Quantum ESPRESSO (QE)`. We will consider strontium vanadate (SrVO$`_{3}`$)
with cubic symmetry as our example system.
Note that this is not supposed to be a comprehensive tutorial on running DFT
calculations with `QE`. Instead, we will focus on how to convert `QE` results into
a format readable by `CoQui`.


## Input Files
The corresponding input files are located in the `inputs` directory:
- `svo.scf.in` - The input file for a self-consistent field (SCF) calculation in QE.
- `svo.coqui.nscf.in` - The input file for a non-SCF calculation in `QE` to obtain
  Bloch basis functions on an *irreducible* k-mesh with a large number of virtual
  orbitals. These basis functions are essential for constructing a many-body Hamiltonian
  in CoQui.
- `svo.pw2coqui.in` - The input file for the `CoQui` converter.

Optionally, to generate Maximally-Localized Wannier Functions (MLWFs) for further 
downfolding/embedding:
- `svo_t2g.nscf.in` - The input file for a non-SCF calculation in `QE` to obtain Bloch basis
  functions on a *uniform* k-mesh.
- `unfold_wfc.toml` - The `CoQui` input file to overwrite the Bloch basis obtained from
  `svo_t2g.nscf.in`.
- `svo_t2g.win` - The input file for `wannier90.x`.
- `svo_t2g.pw2wan.in` - The input file for `pw2wan90.x`.
- `w90_convert_bloch.py` - Python script for wannier90 converter in `triqs/dft_tools`.  
- `svo_t2g.inp` - The input file for the wannier90 converter in `triqs/dft_tools`.

## Preparing Mean-Field Data in an `CoQui` Readable Format
#### Step 0: Preparing a Working Directory
1. Create a working directory (e.g. `work`), and copy the `inputs` and `pseudo`
   directories into it.

#### Step 1: Ground State Calculation using DFT
1. Inside `work`, create a directory for the SCF calculation: `work/scf`.
2. Runs a self-consistent calculation to determine the ground state properties:
   ```shell
   pw.x < svo.scf.in > svo.scf.out 
   ```
   Note: The number of computed bands (`nbnd`) is internally estimated to be roughly
   half the number of electrons in the unit cell (25 in this case). Once complete,
   the self-consistent charge density and potential will be stored in `work/scf/OUT/svo.save`.

#### Step 2: Computing KS Orbitals as the Bloch Basis for `CoQui`
1. Create a directory named `coqui_nscf` in `work`, and copy the SCF results from
   `work/scf/OUT` into it.
2. Conduct a non-self-consistent calculation using the desired k-mesh and number of bands.
   The chosen `K_POINT` and `nbnd` define the single-particle basis $`\phi^\mathbf{k}_i(\mathbf{r})`$
   for expressing the many-body Hamiltonian in `CoQui`. Unlike mean-field calculations, many-body
   calculations typically require a large number of basis functions to converge.
   ```shell
   pw.x < svo.coqui.nscf.in > svo.coqui.nscf.out 
   ```
   It is crucial to enable space-group symmetries (default in `QE`), which will reduce the
   memory requirements and computational cost in subsequent `CoQui` calculations. Note that
   setting `force_symmorphic = .true.` is necessary since `CoQui` currently does not support
   non-symmorphic symmetries.

#### Step 3: Converting NSCF Outputs into an `CoQui` Readable Format
1. Run `pw2coqui.x` to convert the outputs of the `svo.coqui.nscf.in`
   into a `CoQui` readable format.
   ```shell
   pw2coqui.x < svo.pw2coqui.in
   ```
   