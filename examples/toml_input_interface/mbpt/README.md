Examples - Many-Body Perturbation Theories (MBPTs)
-----------------------------------------------

**Last Updated:** May 28, 2024

A many-body calculation in `CoQui` requires an input `[interaction]`, 
which includes the many-body Hamiltonian expanded in the provided 
single-particle basis.
Additionally, all MBPTs in `CoQui` are finite-temperature calculations 
conducted on the imaginary axis, with the temperature controlled by the 
inverse temperature parameter `beta`. 

## Representation on the Imaginary Axis
MBPTs, including Hartree-Fock, in `CoQui` compute dynamic correlation functions 
like the single-particle Green's function and retarded screened interactions on 
the imaginary axis. The efficiency of MBPT calculations thus strongly relies on 
compact representations for these dynamic quantities to capture information across 
the full frequency regime effectively. 

In `CoQui`, both fermionic and bosonic correlation functions are expanded using an 
intermediate representation (IR) basis with sparse sampling on both the imaginary-time 
and Matsubara frequency axes. The IR basis and sampling points are pre-generated using 
the open-source software package [sparse-ir](https://sparse-ir.readthedocs.io/en/latest/index.html). 

The size of the samplings on the imaginary axis is determined solely by the dimensionless 
parameter $`\Lambda`$ and the user-defined accuracy `iaft_prec`. 
Typically, the value of $`\Lambda`$ must exceed $`\beta\omega_{\mathrm{max}}`$ with
$`\omega_{\mathrm{max}}`$ (a.u.) being the bandwidth of the simulated system.
**It's important for users to ensure that $`\Lambda`$ is sufficiently large for the simulated 
system at a given temperature.**
After $`\Lambda`$ is determined, `CoQui` offers three accuracy levels through the parameter `iaft_prec`:
"high" for 1e-15 accuracy, "mid" for 1e-10 accuracy, and "low" for 1e-6 accuracy. 
Roughly speaking, the size of the sampling points scales as $`O(\log(\Lambda)\log(1/\epsilon))`$ 
where $`\epsilon`$ corresponds to `iaft_prec`. 

```toml
beta      = 2000       # inverse tempeature (a.u.^-1) 
lambda    = 1200.0     # dimensionless parameter: beta * w_max
iaft_prec = "high"     # precision of IR basis: "high": 1e-15, "mid":  1e-10, "low": 1e-6
```
Additionally, a similar compact representation utilizing the [Discrete Lehmann Representation (DLR)](https://github.com/jasonkaye/libdlr)
will soon be supported.

## Self-Consistency 

### Full frequency-dependent self-energy

### Quasiparticle approximation

### Iterative solvers

## Checkpoint file data structure 