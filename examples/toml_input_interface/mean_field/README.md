Examples - mean_field
-----------------------------------------------

**Last Updated:** May 09, 2024

An `CoQui` calculation always begins with a declaration of `[mean_field]`, 
which is responsible for accessing the input data of the simulated system 
from supported mean-field codes using the corresponding converter. 
This data includes:     
- **Metadata**: Defines the simulated system. 
- **Mean-Field Solution**: Typically from DFT or HF.
- **Single-Particle Basis Set**: Basis functions used to construct the many-body 
Hamiltonian in subsequent calculations, typically including Kohn-Sham orbitals, 
Gaussian bases, or other generic basis functions. 

Every mean-field software package generates results in a completely different 
data structure, requiring a customized reader for each. Currently, `CoQui` 
provides three different mean-field readers: 
* `[mean_field.qe]`: For input data from [Quantum ESPRESSO](https://www.quantum-espresso.org). 
See [qe_mf.toml](qe_mf.toml). 
* `[mean_field.pyscf]`: For input data from [PySCF](https://pyscf.org). 
See [pyscf_mf.toml](pyscf_mf.toml)
* `[mean_field.bdft]`: For input data stored in the `CoQui` internal data structure.

Note that input data from any supported mean-field backend can be converted into the 
consistent `[mean_field.bdft]` format. This is useful when you want to create a new 
`[mean_field]` object by modifying an existing `[mean_field]`. For more information, 
see [bdft_mf.toml](bdft_mf.toml).

### Preparing `Quantum Espresso` inputs 
Provide instructions or examples for setting up `Quantum Espresso` inputs here. 

### Preparing `PySCF` inputs
Provide instructions or examples for setting up `PySCF` inputs here.
