Examples - interaction
-----------------------------------------------

**Last Updated:** March 23, 2025

Starting with a single-particle basis set $`\phi^{\textbf{k}}_a(\textbf{r})`$
provided by a `[mean_field]` system, the initialization of a many-body calculation
is invoked through the `[interaction]` section. This process constructs the second
quantization electronic Hamiltonian with the general two-electron Coulomb interactions:

$$
\hat{H}_{\mathrm{int}} = \frac{1}{2}\sum _{abcd}V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd} c^{\textbf{k}_1\dagger} _{a} c^{\textbf{k}_3\dagger} _{c} c^{\textbf{k}_4} _{d} c^{\textbf{k}_2} _{b}
$$

$$
V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd} = \int d\textbf{r} \int d\textbf{r}' \phi^{\textbf{k}_1*} _{a} (\textbf{r})\phi^{\textbf{k}_2} _{b}(\textbf{r})\frac{1}{|\textbf{r}-\textbf{r}'|}\phi^{\textbf{k}_3*} _{c}(\textbf{r}')\phi^{\textbf{k}_4} _{d}(\textbf{r}')
$$

In this context, the momentum transferred between the product basis
$`\phi^{\textbf{k}_{1}}_{a}(\textbf{r})\phi^{\textbf{k}_{2}*}_{b}(\textbf{r})`$
(also known as the pair densities) satisfies the conservation of momentum
$`\textbf{k}_1 - \textbf{k}_2 + \textbf{G} = \textbf{k}_3 - \textbf{k}_4`$,
where $`\textbf{G}`$ represents a reciprocal lattice vector of the system.
Despite the enormous size of $V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd}$
(exhibiting cubic scaling with $`N_{k}`$ and quartic scaling with $`N_{\mathrm{orb}}`$),
this tensor is strongly rank deficient, reflecting the over-completeness of the
product basis for two-electron operators.
Compression schemes for $V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd}$
can drastically reduce both the memory requirement and computational cost in subsequent
many-body calculations.

### [Tensor Hypercontraction](thc_eri.toml) (THC)
`CoQui` employs the following tensor hypercontraction (THC) representation in a
generic Bloch basis set, which includes both Kohn-Sham orbitals and Gaussian-type
bases:
```math
V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd} \approx \sum _{\mu\nu}
X^{\textbf{k}_1*} _{\mu a} X^{\textbf{k}_2} _{\mu b}V^{\textbf{q}} _{\mu\nu} 
X^{\textbf{k}_3*} _{\nu c}X^{\textbf{k}_4} _{\nu d}
```
Here, $`\textbf{q} = \textbf{k}_1 - \textbf{k}_2 + \textbf{G} = \textbf{k}_4 - \textbf{k}_3 + \textbf{G}'`$
represents the momentum transfer, and the greek letters denote the auxiliary basis introduced in the THC
decomposition. The size of the THC auxiliary basis ($`N_{\mu}`$) is determined either as an input parameter
or calculated on-the-fly to achieve the desired accuracy (see [thc_eri.toml](thc_eri.toml)).
Practically, $`N_{\mu} = O(N_{\mathrm{orb}})`$ is expected to provide good accuracy due to the low-rank
structure of $V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd}$.
The THC decomposition separates the orbital and momentum indices, which not only reduces memory requirements
but also facilitates low-scaling algorithms for subsequent many-body perturbation calculations.

The computed THC Coulomb integrals are accessed by passing the `[interaction]` to subsequent many-body
calculations (see [mbpt](../mbpt/README.md)). Optionally, the THC integrals can be saved to
an HDF5 file with the following data structure:
```text
group      /
dataset    /Np                       # Size of the THC auxiliary basis (N_mu)
dataset    /collocation_matrix       # Collocation matrix X^k
dataset    /coulomb_matrix           # Coulomb matrix in the THC auxiliary basis V^q
```
The HDF5 checkpoint file for THC Coulomb integrals is particularly useful for reusing the Coulomb integrals
in various calculations and for external developments in electronic structure theories (see
[thc_eri.toml](thc_eri.toml) for more details).

---

### [Cholesky decomposition](chol_eri.toml)
Additionally, `CoQui` offers Cholesky decomposition as another method to factorize
the Coulomb Hamiltonian   
$V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd}$:
```math
V^{\textbf{k}_1\textbf{k}_2\textbf{k}_3\textbf{k}_4} _{abcd} \approx 
\sum_{P} L^{\textbf{k}_2 \textbf{k}_1 *} _{baP} L^{\textbf{k}_3 \textbf{k}_4} _{cdP}
```
It is important to note that computing Cholesky Coulomb integrals and performing subsequent
many-body calculations exhibit much higher complexities, both in terms of memory requirements
and computational demands. The inclusion of Cholesky decomposition primarily serves as a tool
for debugging. For practical, large-scale computations, the THC formalism is recommended due
to its efficiency and scalability.
