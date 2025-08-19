Downfolding
-----------------------------------------------

**Last Updated:** July 07, 2024

The downfolding procedure in `CoQui` begins with a many-body Hamiltonian in
the THC representation, expressed as $H = H_0 + V^{\mathrm{THC}} _{ee}$ 
(see [interaction](../interaction/README.md)), and an input Green's function 
$G^{\textbf{k}} _{ij}(\tau)$ stored in an `CoQui` checkpoint file (see [mbpt](../mbpt/README.md)). 
Additionally, users are responsible for providing the definitions of the 
correlated orbitals within the targeted low-energy subspace, as `CoQui` does 
not perform orbital localization itself. The information about the correlated 
orbitals is accessed by `CoQui` through an HDF5 file, structured as follows:
```text
group      /
dataset    /dft_input/proj_mat              # Projection matrix for the localized basis 
dataset    /dft_input/kpts                  # k-points in the full Brillouin zone (BZ)
dataset    /dft_misc_input/band_window      # Band window for the construction of the localized basis 
dataset    /dft_input/wan_centres           # Localization centres
```
Step-by-step instructions on how to prepare such an HDF5 input file via the
`Quantum ESPRESSO` and `Wannier90` interface can be found [here](../dft_converter/qe/README.md).

### Low-energy effective Hamiltonian

Given an electronic structure characterized by the input Green's function $G^{\textbf{k}} _{ij}(\tau)$,
`CoQui` can derive a low-energy effective Hamiltonian for the specified correlated subspace. 
This Hamiltonian is represented as the action of a general impurity problem:

$$
\mathcal{S} = \iint_0^{\beta} \mathrm{d} \tau \mathrm{d} \tau' \sum_{ab}
c^{\dagger}_a(\tau) \mathcal{G}^{-1} _{ab}(\tau-\tau') c_b(\tau') + \frac{1}{2}
\int_0^{\beta} \mathrm{d} \tau \mathrm{d} \tau' \sum _{abcd}
c^{\dagger}_a(\tau)c^{\dagger}_c(\tau')\mathcal{U} _{abcd}(\tau - \tau') c_d(\tau')c_b(\tau)
$$

Here, $\tau$ represents the imaginary time, and $\beta$ denotes the inverse temperature
($a.u.^{-1}$). The subscripts $a$, $b$, $c$, and $d$ indicate orbital indices.
The terms $\mathcal{G} _{ab}(\tau)$ and $\mathcal{U} _{abcd}(\tau)$ represent the
fermionic and bosonic Weiss field, corresponding to the one-body and two-body effective
Hamiltonian components of the impurity, respectively.
Constructing a low-energy model is therefore equivalent to evaluating 
$\mathcal{G} _{ab}(\tau)$ and $\mathcal{U} _{abcd}(\tau)$. `CoQui` divides the downfolding 
process into two independent components:
1. `[downfold_2e]`: Evaluates $\mathcal{U} _{abcd}(\tau)$ using cRPA or EDMFT equations.
   A step-by-step tutorial can be found [here](downfold_2e/README.md).
2. `[downfold_1e]`: Evaluates $\mathcal{G} _{ab}(\tau)$, offering various choices
   for double-counting contributions. A step-by-step tutorial can be found [here](downfold_1e/README.md). 
