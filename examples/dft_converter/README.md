DFT Converter
-----------------------------------------------

**Last Updated:** Sept. 07, 2025

`CoQui` requires the following input data in order to start a many-body 
electronic structure calculation:  
- **Metadata**: Defines the crystal/molecule information sush as atomic
positions and k-points.
- **Single-Particle Basis Set**: Basis functions used to construct the many-body
  Hamiltonian in subsequent calculations, typically including Kohn-Sham orbitals,
  Gaussian bases, or other generic basis functions.

Every mean-field software package generates results in a completely different
data structure, requiring a customized converter for each. Currently, `CoQui` 
supports two mean-field backends: 
* [Quantum ESPRESSO](https://www.quantum-espresso.org): 
  See [QE tutorial](qe/README.md) and [qe_mf.toml](../mean_field/qe_mf.toml). 
* [PySCF](https://pyscf.org): 
  See [PySCF tutorial](pyscf/README.md) and [pyscf_mf.toml](../mean_field/pyscf_mf.toml).  

Alternatively, input data from any supported mean-field backend can be converted 
into a consistent `CoQui` format (`[mean_field.bdft]`). This is useful when you want 
to create a new `[mean_field]` object by modifying an existing one. 
For more information, see [bdft_mf.toml](../mean_field/bdft_mf.toml). 
