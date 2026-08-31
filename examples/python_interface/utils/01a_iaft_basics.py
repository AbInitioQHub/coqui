"""
Example: Imaginary-axis Fourier Transform (IAFT) using Intermediate Representation (IR)

This example demonstrates:
  1) Initialization of the IAFT class with inverse temperature, frequency cutoff,
     and precision parameters
  2) Accessing mesh information for both fermionic and bosonic statistics
  3) Fourier transforms between imaginary-time and Matsubara-frequency domains
  4) Interpolation to arbitrary points on the imaginary-time and frequency axes
  5) Checking for spectral leakage in the intermediate representation (IR)

Overview
--------
The IAFT class provides Fourier transforms on the imaginary axis based on the
intermediate representation (IR) basis and sparse sampling technique. It enables
efficient transformations between imaginary-time and Matsubara-frequency domains
without explicit spectral data.


Dependency
----------
The python IAFT submodule requires the sparse-ir package (with xprec support), version 1.1.7:

  pip install sparse-ir[xprec]==1.1.7

Note: Versions 2.0 and above of sparse-ir have not yet been tested and may not be compatible with this code.
"""

import numpy as np
from coqui.utils.imag_axes_ft import IAFT


# ===================================================
# 1. Initialize IAFT
# ===================================================
beta = 100.0  # Inverse temperature (a.u.)
wmax = 10.0   # Frequency cutoff (a.u.)
prec = 1e-10  # Precision for IR basis 

iaft = IAFT(beta=beta, wmax=wmax, prec=prec, verbose=True, basis='ir')

# ===================================================
# 2. Explore mesh information
# ===================================================
print("\n### Mesh Information ###")

# Fermionic meshes
print("\nFermionic sampling:")
print(f"  Number of tau points (nt_f):  {iaft.nt_f}")
print(f"  Number of freq points (nw_f): {iaft.nw_f}")
print(f"  tau mesh (first 3 points):    {iaft.tau_mesh_f[:3]}")

# Bosonic meshes
print("\nBosonic sampling:")
print(f"  Number of tau points (nt_b):  {iaft.nt_b}")
print(f"  Number of freq points (nw_b): {iaft.nw_b}")
print(f"  tau mesh (first 3 points):    {iaft.tau_mesh_b[:3]}")

# Matsubara frequency indices (CoQuí/IR convention: iwn = n*pi/beta)
wn_f = iaft.wn_mesh('f', phys_notation=False)
wn_b = iaft.wn_mesh('b', phys_notation=False)
print(f"\nMatsubara indices (fermion, first 3): {wn_f[:3]}")
print(f"Matsubara indices (boson, first 3):   {wn_b[:3]}")

# Physical Matsubara frequencies
wn_f_phys = wn_f * np.pi / iaft.beta
wn_b_phys = wn_b * np.pi / iaft.beta
print(f"\nPhysical frequencies (fermion, first 3): {wn_f_phys[:3]}")
print(f"Physical frequencies (boson, first 3):   {wn_b_phys[:3]}")

# ===================================================
# 3. Fourier transforms: tau -> w and w -> tau
# ===================================================
print("\n### Fourier Transforms ###")

# Create a random fermionic object in tau domain
Gt_f = np.random.randn(iaft.nt_f, 2, 2)

# Transform to frequency domain
Gw_f = iaft.tau_to_w(Gt_f, stats='f')
print(f"\nFermionic Green's function shapes:")
print(f"  tau domain: {Gt_f.shape}")
print(f"  frequency domain: {Gw_f.shape}")

# Transform back to tau domain
Gt_f_back = iaft.w_to_tau(Gw_f, stats='f')
print(f"  back to tau domain: {Gt_f_back.shape}")

# Check reconstruction error
reconstruction_error = np.max(np.abs(Gt_f - Gt_f_back))
print(f"  max reconstruction error: {reconstruction_error:.2e}")

# Similar for bosonic Green's function
Gt_b = np.random.randn(iaft.nt_b, 2, 2)
Gw_b = iaft.tau_to_w(Gt_b, stats='b')
print(f"\nBosonic Green's function shapes:")
print(f"  tau domain: {Gt_b.shape}")
print(f"  frequency domain: {Gw_b.shape}")

# ===================================================
# 4. Interpolation to arbitrary points
# ===================================================
print("\n### Interpolation ###")

# Interpolate to arbitrary tau points
tau_interp = np.linspace(0.0, iaft.beta, 50)
Gt_f_interp = iaft.tau_interpolate(Gt_f, tau_interp, stats='f')
print(f"\nInterpolation to {len(tau_interp)} arbitrary tau points:")
print(f"  original shape: {Gt_f.shape}")
print(f"  interpolated shape: {Gt_f_interp.shape}")

# Interpolate to arbitrary Matsubara frequency points (CoQuí/IR convention)
wn_interp = np.array([-5, -3, -1, 1, 3, 5], dtype=int)
Gw_f_interp = iaft.w_interpolate(Gw_f, wn_interp, stats='f', phys_notation=False)
print(f"\nInterpolation to {len(wn_interp)} arbitrary Matsubara points:")
print(f"  original shape: {Gw_f.shape}")
print(f"  interpolated shape: {Gw_f_interp.shape}")
print(f"  interpolated indices: {wn_interp}")

# ===================================================
# 5. Particle-hole symmetry (for bosonic functions)
# ===================================================
print("\n### Particle-Hole Symmetry (Bosonic) ###")

# For bosonic functions with particle-hole symmetry, only positive frequencies
# are stored and used
nw_half = iaft.nw_b // 2 if iaft.nw_b % 2 == 0 else iaft.nw_b // 2 + 1
nt_half = iaft.nt_b // 2 if iaft.nt_b % 2 == 0 else iaft.nt_b // 2 + 1

wn_b_phsym = iaft.wn_mesh(stats='b', positive_only=True)
assert nw_half == len(wn_b_phsym) 

Gt_b_phsym = np.random.randn(nt_half, 2, 2)
Gw_b_phsym = iaft.tau_to_w_phsym(Gt_b_phsym, stats='b')

print(f"\nBosonic function with particle-hole symmetry:")
print(f"  tau domain (half): {Gt_b_phsym.shape}")
print(f"  frequency domain (half): {Gw_b_phsym.shape}")

# Transform back with particle-hole symmetry
Gt_b_phsym_back = iaft.w_to_tau_phsym(Gw_b_phsym, stats='b')
print(f"  back to tau domain (half): {Gt_b_phsym_back.shape}")
