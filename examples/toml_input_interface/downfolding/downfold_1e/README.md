Downfolding - Effective One-Body Hamiltonian
-----------------------------------------------

**Last Updated:** February 07, 2024

The `[downfold_1e]` module calculates the effective one-body Hamiltonian for a 
user-defined active space, starting from a precomputed electronic structure 
solution of a crystal. At the highest level, CoQui provides two different downfolding 
schemes:

- **Full-frequency Downfolding**: See [mbpt_downfold.toml](mbpt_downfold.toml).   
This scheme produces a frequency-dependent one-body Hamiltonian, incorporating both 
the non-interacting component and the dynamic self-energy.  
- **Quasiparticle Downfolding**:
See [qp_mbpt_downfold.toml](qp_mbpt_downfold.toml) and [mf_downfold.toml](mf_downfold.toml).<br>
This scheme generates a static effective Hamiltonian by applying the quasiparticle 
approximation to the dynamic self-energy. 

### Prerequisites
The `[downfold_1e]` module relies on the following inputs
- **A precomputed electronic structure solution**:  
This includes the mean-field solutions from the supported DFT backends 
(see [mf_downfold.toml](mf_downfold.toml)) as well as Dyson SCF calculations 
from CoQui. 
- **Effective screened interactions $\mathcal{U}$ from the `[downfold_2e]` module:** See [downfold_2e](../downfold_2e/README.md).  
This is necessary for double-counting evaluations. 

### Example: $G_{0}W_{0}$ quasiparticle Hamiltonian for V $t _{2g}$ orbitals in $\mathrm{SrVO}_3$
Below is an example of the input configuration for calculating the tight-binding Hamiltonian from 
quasiparticle $G_{0}W_{0}$ with the corresponding double counting corrections for the V $t _{2g}$ 
orbitals in $\mathrm{SrVO}_3$. 
This example utilizes the DFT input and $t _{2g}$ Wannier orbitals generated through the
interface with `Quantum ESPRESSO` and `Wannier90`, as detailed in this [tutorial](../../dft_converter/qe/README.md).


```toml
[mean_field.qe]
name     = "mf_qe"
prefix   = "pwscf"
outdir   = "qe_output_dir/OUT/"
filetype = "h5"

[interaction.thc]
name        = "eri"
mean_field  = "mf_qe"
storage     = "incore"
thresh      = 1e-6
chol_block_size = 8
distr_tol   = 0.4

[downfold_2e]                            # cRPA@DFT
input_type    = "mf"                     # input G^k_ij(tau) type: "mf" or "coqui".
                                         # if "mf": create/overwirte a new/existing coqui h5 and take G^k_ij(tau) from the input MF
                                         # if "coqui": read G^k_ij(tau) from the existing "outdir/prefix.mbpt.h5"
interaction   = "eri"                    # tag for the input [interaction] instance
wannier_file  = "wan.h5"                 # input h5 which defines the targeted correlated subspace
outdir        = "./"                     # directory for the output/input coqui h5 archive
prefix        = "qp_mbpt_downfold"       # prefix for the input/output coqui h5 archive
screen_type   = "crpa"                   # method for the screening effects: "crpa" or "edmft"
beta          = 2000
lambda        = 1200.0
iaft_prec     = "high"

# general optional parameters
force_real    = true                     # enforce U_abcd(tau) to be real or not
permut_symm   = true                     # enforce permutation symmetry to U_abcd(tau) or not
div_treatment = "gygi"                   # divergence treatment for the q-summation in U_abcd(tau)

[gw]                                     # G0W0@DFT
restart   = true                         # "restart = true" is necessay to instruct CoQui to continue
                                         # working on the same h5 archive. 
                                         # In addition, there is no need to specify "beta", "lambda", and "iaft_prec"
                                         # since coqui would read them from qp_mbpt_downfold.mbpt.h5.
interaction = "eri"
niter     = 1                            # perform only one iteration
output    = "qp_mbpt_downfold"
div_treatment = "gygi"

[downfold_1e]                            # effective one-body Hamiltonian based on G0W0@DFT and cRPA@DFT
mean_field    = "mf_qe"                  # tag for the input [mean_field] instance
outdir        = "./"                     # directory for the output/input coqui h5 archive
prefix        = "qp_mbpt_downfold"       # prefix for the input/output coqui h5 archive
wannier_file  = "wan.h5"                 # input h5 which defines the targeted correlated subspace
dc_type       = "gw"                     # double counting formula.
                                         # Available options are "hartree", "hf", "gw", "gw_dynamic_u"
# optional variable
qp_selfenergy = true                     # quasiparticle embedding scheme or not. default = false
force_real    = true                     # enforce the downfolded Hamiltonian to be real or not
# qp qpprox. parameters (all optional)
ac_alg        = "pade"                   # analytical continuation algorithm. default = "pade"
eta           = 1e-6                     # infinitesimal shift above the real axis
Nfit          = 30                       # Number of poles to fit
off_diag_mode = "qp_energy"              # QP schemes for off-diagonal self-energy: "qp_energy" or "fermi". 
                                         # default: "qp_energy" 
```

#### Output data structure
The resulting data from the calculation will be stored in the `downfold_1e` group within the CoQui HDF5 archive:
```text
# Full-frequency Downfolding
group      /downfold_1e
 dataset    /downfold_1e/C_skIai
 dataset    /downfold_1e/final_iter         
 dataset    /downfold_1e/iter1/Gloc_wsIab      # local Green's function from the input G^k_ij(iw)
 dataset    /downfold_1e/iter1/H0_sIab         # non-interacting Hamiltnoian
 dataset    /downfold_1e/iter1/Sigma_dc_wsIab  # double counting dynamic GW self-energy
 dataset    /downfold_1e/iter1/Sigma_gw_wsIab  # dynamic GW self-energy
 dataset    /downfold_1e/iter1/Vhf_dc_sIab     # double counting static GW self-energy (i.e. HF)
 dataset    /downfold_1e/iter1/Vhf_gw_sIab     # static GW self-energy
 dataset    /downfold_1e/iter1/dc_type         # double counting type
 dataset    /downfold_1e/iter1/delta_wsIab     # hybridization function
 dataset    /downfold_1e/iter1/g_weiss_wsIab   # fermionic Weiss field
```
```text
# Quasiparticle Downfolding
group      /downfold_1e
 dataset    /downfold_1e/C_skIai
 dataset    /downfold_1e/final_iter         
 dataset    /downfold_1e/iter1/Gloc_wsIab      # local Green's function from the input G^k_ij(iw)
 dataset    /downfold_1e/iter1/H0_sIab         # non-interacting Hamiltnoian
 dataset    /downfold_1e/iter1/Vcorr_dc_sIab   # double counting qpGW self-energy
 dataset    /downfold_1e/iter1/Vcorr_gw_sIab   # qpGW self-energy
 dataset    /downfold_1e/iter1/Vhf_dc_sIab     # double counting static GW self-energy (i.e. HF)
 dataset    /downfold_1e/iter1/Vhf_gw_sIab     # static GW self-energy
 dataset    /downfold_1e/iter1/dc_type         # double counting type
 dataset    /downfold_1e/iter1/delta_wsIab     # hybridization function
 dataset    /downfold_1e/iter1/g_weiss_wsIab   # fermionic Weiss field
```