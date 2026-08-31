Downfolding - Effective Coulomb Interaction
-----------------------------------------------

**Last Updated:** February 07, 2024

Given:
1. A many-body Hamiltonian $H = H_0 + V^{\mathrm{THC}} _{ee}$
2. An input Green's function $G^{\textbf{k}} _{ij}(\tau)$,

the `[downfold_2e]` module evaluates the dynamic screened interactions $\mathcal{U} _{abcd}(\tau)$
in a general four-index format. Currently, CoQui supports two different methodologies to account 
for the screening effects:
- Constrained Random Phase Approximation (cRPA) [[ref]](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.70.195104)
- Extended Dynamic Mean-Field Theory (EDMFT) [[ref]](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.90.086402).

All screening types are integrated through a consistent interface and controlled by the `screen_type` 
parameter. Supported values include: `"bare"`, `"crpa"`, `"crpa_ks"`, `"crpa_vasp"`, and `"edmft"`.
The three crpa options differ in how they handle cases where the projection matrices are not unitary due 
to entanglement. All of them fall back to the standard cRPA in the absence of entanglement.  
- `"crpa"`: The standard cRPA methods which evaluates polarizability within the active space using projection 
  matrices. In the presence of entanglement, this might result in problems if some bands at the fermi are not 
  completely exclusded.  
- `"crpa_ks"`: Evaluate polarizability within the active space using KS bands having strong Wannier 
  orbitals character at each k-point. This is exact if the projection matrices are unitary. 
- `"crpa_vasp"`: Evaluate polarizability within the active space using the regularized projection matrices, 
  as motivated by Merzuk Kaltak's [thesis](https://utheses.univie.ac.at/detail/33771) and VASP implementation. 
  This is exact if the projection matrices are unitary. 

### Input Green's Function $G^{\textbf{k}} _{ij}(\tau)$
The `[downfold_2e]` module accesses the input Green's function $G^{\textbf{k}} _{ij}(\tau)$ through an 
CoQui HDF5 output file.
Depending on the users' intensions, CoQui offers flexibility regarding the theoretical level at which
$G^{\textbf{k}} _{ij}(\tau)$ is obtained. Options include Green's functions derived from:
- a mean-field solution: see [crpa_from_mf.toml](crpa_from_mf.toml)
- MBPT calculations: see [crpa_from_coqui.toml](crpa_from_mbpt.toml) and [crpa_from_coqui_2.toml](crpa_from_mbpt_2.toml)
- DMFT embedding: see [crpa_from_dmft.toml](crpa_from_dmft.toml)

These options are controlled by the following parameters:
1. `input_type`: "mf" or "coqui".
   - Specifies whether the input $G^{\textbf{k}} _{ij}(\tau)$ is a non-interacting Green's function from a
     [mean_field] instance or an interacting Green's function from an `CoQui` MBPT calculation.
2. `input_grp`: [optional] Default: "scf" (the place where MBPT results are stored).
   - The group name within the HDF5 file where the input $G^{\textbf{k}} _{ij}(\tau)$ is stored.
3. `input_iter`: [optional] Default: last iteration in the "scf" group.
   - The iteration number from which the input $G^{\textbf{k}} _{ij}(\tau)$ is retrieved.

### Example: cRPA screened interactions for V $t _{2g}$ orbitals in $\mathrm{SrVO}_3$
Below is an example of the input configuration for calculating the cRPA $\mathcal{U} _{abcd}(\tau)$
for the V $t _{2g}$ orbitals in $\mathrm{SrVO}_3$ using $G^{\textbf{k}} _{ij}(\tau)$ from a $G_0 W_0$
calculation. This example utilizes the DFT input and $t _{2g}$ Wannier orbitals generated through the
interface with `Quantum ESPRESSO` and `Wannier90`, as detailed in this [tutorial](../../dft_converter/qe/README.md).

Reading the metadata for the simulated system through the `[mean_field.qe]` section, the cRPA
calculation process is outlined as follows:
1. **Construct $H = H_0 + V^{\mathrm{THC}} _{ee}$**: `[interaction.thc]` takes
   `[mean_field.qe]` as input and performs the THC decomposition for $V_{ee}$. The resulting
   $V^{\mathrm{THC}} _{ee}$ is stored in `svo_thc_eri.h5`.
2. **One Iteration of Full-Frequency GW**: The full-fequency GW calculation is carried
   out by `[gw]` for the given $H = H_0 + V^{\mathrm{THC}} _{ee}$. The MBPT output is stored in
   `svo_t2g.mbpt.h5`.
3. **Dynamic Screened Interaction from cRPA**: The `[downfold_2e]` modules calculates the cRPA
   $\mathcal{U} _{abcd}(i\Omega_n)$ using $G^{\textbf{k}} _{ij}(\tau)$ from the previous step and stores
   the results in the `downfold_2e` group within the same HDF5 archive.

```toml
# cRPA screened interaction U_abcd(tau) for V t2g based on the G0W0 electronic structure

[mean_field.qe]                          # QE DFT results for SrVO3
name     = "svo_qe"
prefix   = "svo"
outdir   = "./coqui_nscf/OUT/"
filetype = "h5"

[interaction.thc]                        # THC MB Hamiltonian: H = H0 + V^THC_ee
name            = "svo_eri"
mean_field      = "svo_qe"
storage         = "incore"
save            = "svo_thc_eri.h5"
thresh          = 1e-6
chol_block_size = 8
distr_tol       = 0.4

[gw]                                     # one iteration of full-frequency GW, and the resulting G^k_ij will be used in the subsequent cRPA
interaction = "svo_eri"
restart   = false
output    = "svo_t2g"
beta      = 300
lambda    = 8200.0
iaft_prec = "high"
niter     = 1
conv_thr  = 1e-6
div_treatment = "gygi"

[downfold_2e]                            
interaction     = "svo_eri"              # tag for the input [interaction] instance
wannier_file  = "./wannier/svo_t2g.h5"   # input h5 which defines the targeted correlated subspace
outdir        = "./"                     # directory for the output/input coqui h5 archive
prefix        = "svo_t2g"                # prefix for the input/output coqui h5 archive
screen_type   = "crpa"                   # method for the screening effects: "crpa" or "edmft"
input_type    = "coqui"                 # input G^k_ij(tau) type: "mf" or "coqui". 
                                         # if "mf": create/overwirte a new/existing coqui h5 and take G^k_ij(tau) from the input MF
                                         # if "coqui": read G^k_ij(tau) from the existing "outdir/prefix.mbpt.h5"

# optional parameters
force_real    = true                     # enforce U_abcd(tau) to be real or not
permut_symm   = true                     # enforce permutation symmetry to U_abcd(tau) or not
div_treatment = "gygi"                   # divergence treatment for the q-summation in U_abcd(tau)
```

#### Output data structure
The resulting data from the calculation will be stored in the `downfold_2e` group within the same `coqui` HDF5 archive:
```text
group      /downfold_2e
 dataset    /downfold_2e/C_skIai
 dataset    /downfold_2e/final_iter         
 dataset    /downfold_2e/iter1/Gloc_tsIab    # local Green's function from the input G^k_ij(tau)
 dataset    /downfold_2e/iter1/Uloc_type     # screen_type
 dataset    /downfold_2e/iter1/Uloc_wabcd    # U_abcd(i\omega) on the bosonic IR grid points 
 dataset    /downfold_2e/iter1/Vloc_abcd     # bare interaction 
 dataset    /downfold_2e/iter1/Wloc_wabcd    # fully screened interaction on the bosonic IR grid points 
 dataset    /downfold_2e/iter1/permut_symm   # permutation symmetry: "none", "4-fold", "8-fold"
```
If `q_dependent = true`, the q-dependent bare/screened interactions ($\mathcal{U}^{\textbf{q}} _{abcd}(i\Omega_n)$ and
$V^{\textbf{q}} _{abcd}$) will also be stored:
```text
 dataset    /downfold_2e/iter2/U_wqabcd      # q-dependent screened interaction
 dataset    /downfold_2e/iter2/V_qabcd       # q-dependent bare interaction 
```