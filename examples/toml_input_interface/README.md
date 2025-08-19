Examples
-----------------------------------------------

**Last Updated:** January 30, 2025

This directory provides examples of inputs demonstrating various uses of CoQui. 

### Input Structure
The input files of CoQui are formatted using [TOML](https://toml.io/en/). Each input file comprises 
multiple sections, each representing a specific task using a `TOML` table. A CoQui 
calculation is defined by chaining different tasks in a specific order to achieve the 
desired result. **The order of the sections is therefore crucial**, as demonstrated in 
the [example](#example-hartree-fock-calculation) below. 

Available sections are 
* [Mean-Field Parser](mean_field/README.md): `[mean_field]`
* [Two-electron Coulomb Integrals](interaction/README.md): `[interaction]`
* [Many-Body Perturbation Theory](mbpt/README.md): `[hf]`, `[gw]`, `[rpa]`, etc.
* [Embedding and Downfolding Routines](embedding/README.md): `[crpa]`, `[downfold_1e]`, `[downfold_2e]`, 
`[mf_downfold]`, etc.
* [Post-Proecssing Routines](pproc): `[band_interpolation]`, `[spectral_interpolation]`, etc.

Overall, each category depends on the one preceding it. The list above is 
organized specifically to reflect this dependency. For a more comprehensive 
understanding, we highly recommend going through the documentation for each section
in the same order. 

Unlike the standard `TOML` format, CoQui permits multiple tables with the same key at the 
*top* level. This feature facilitates the execution of similar tasks with different parameters
(e.g. basis set convergence) in a single run. 

### Example: Hartree-Fock Calculation
To perform a Hartree-Fock (HF) calculation using CoQui, run the following command:
```shell 
mpirun [YOUR_INSTALL_PREFIX]/bin/coqui --filenames hf.toml 
```
The input file `hf.toml` consists sections that defines the mean-field initialization, 
the construction of electron repulsion integrals (ERIs) using tensor hypercontraction (THC), 
and the HF self-consistency loop:
```toml
# hf.toml
# Reads the DFT solution and KS basis from QE
[mean_field.qe]
name     = "my_qe"                  # name tag for this mean_field instance
prefix   = "pwscf"
outdir   = "qe_output_dir/OUT/"

# Compute ERIs in a THC representation 
[interaction.thc]
name        = "my_thc_eri"          # name tag for the current THC-ERI instance
mean_field  = "my_qe"               # name tag for the input mean_field instance 
storage     = "incore"
thresh      = 1e-6
chol_block_size = 8
r_blk       = 20
distr_tol   = 0.4

# Perform HF self-consistency loop
[hf]
interaction = "my_thc_eri"          # name tag for the input ERI instance 
beta      = 2000
lambda    = 1200.0
iaft_prec = "high"
niter     = 12
restart   = false
output    = "hf"
[hf.iter_alg]
alg    = "damping"
mixing = 0.7
```
Without knowing the specifics of each section, the calculation can be
understood as follows:
1. **Initialize a Mean Field Object**: The `[mean_field.qe]` object reads the
   mean-field solution as well as the metadata for the simulated system from the input
   DFT codes (Quantum ESPRESSO in this case).
2. **Construct ERIs in a THC representation**: `[interaction.thc]` takes a `[mean_field.qe]` as input and
   performs the THC decomposition for the ERIs.
3. **Perform the HF Self-Consistency Loop**: The HF self-consistency loop is carried
   out by `[hf]` for the specified many-body Hamiltonian provided by `[interaction.thc]`.

As seen in this example, the dependency among different sections is crucial,
and changing their order may result in quite different outcomes
(an uninitialized error in this case). Despite this slightly more
complex structure of the inputs, the flexibility of combining different sections
in various orders allows CoQui to handle a wider range of tasks.