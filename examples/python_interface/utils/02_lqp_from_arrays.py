"""
Example: linearized quasiparticle approximation of a self-energy given as arrays

This example demonstrates:
  1) Sampling a synthetic dynamic self-energy on the imaginary-time mesh of an IAFT
  2) Linearizing it with lqp_from_arrays
  3) Reading the renormalization matrix Z, the quasiparticle bands and the pole weights from
     the result, and checking them against the closed form

Overview
--------
lqp_from_arrays fits Sigma(iw) ~ A + iw B at the lowest Matsubara frequencies and
returns, per (spin, k),

  Z = (1 - B)^-1,   H_QP = Z^1/2 (F + A - mu) Z^1/2 + mu,

the eigenvalues and eigenvectors v of H_QP, and the pole weights <v|Z|v>. It takes F and
Sigma(tau) in any orthonormal basis, on any set of k-points. 
This can be used to linearize a self-energy obtained from any external source. 

Model: two orbitals on a chain, with a k-independent self-energy of one pole per orbital,

  F(k)      = [[eps_1 - 2 t_1 cos k, V], [V, eps_2 - 2 t_2 cos k]],   t_1 t_2 < 0,
  Sigma(iw) = diag(g_a / (iw - e_a)),   a = 1, 2.

Expanding Sigma at low frequency gives the closed form

  A = diag(-g_a / e_a),   B = diag(-g_a / e_a^2),   Z = diag(z_a),   z_a = (1 + g_a / e_a^2)^-1.

Z is the same at every k, but the pole weight <v|Z|v> = sum_a z_a |v_a|^2 of a band follows its
orbital character: with t_1 t_2 < 0 the bands cross, and the weights swap between z_1 and z_2
where they anti-cross.
"""

from mpi4py import MPI
import numpy as np
import coqui
from coqui import IAFT
from coqui.post_proc import lqp_from_arrays

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-deep")

# mpi handler and verbosity
mpi = coqui.MpiHandler()
coqui.set_verbosity(mpi, output_level=1)

# ===================================================
# 1. Model: F(k) and Sigma(tau) on the IAFT tau mesh
# ===================================================
beta = 100.0  # inverse temperature (a.u.)
mu = 0.0      # chemical potential (a.u.)
iaft = IAFT(beta=beta, wmax=10.0, prec="high", basis="dlr", verbose=False)

# F(k): eps_1 = eps_2 = 0.3, t_1 = 0.02, t_2 = -0.01, V = 0.004 (a.u.); A_aa = -0.3 below
# brings the on-site energy back to mu
nkpts = 101
kpts = np.linspace(-np.pi, np.pi, nkpts)
F_sk = np.zeros((1, nkpts, 2, 2), dtype=np.complex128)
F_sk[0, :, 0, 0] = 0.3 - 2 * 0.02 * np.cos(kpts)
F_sk[0, :, 1, 1] = 0.3 + 2 * 0.01 * np.cos(kpts)
F_sk[0, :, 0, 1] = F_sk[0, :, 1, 0] = 0.004

# Sigma(iw): (e_a, g_a) in a.u., both with A_aa = -g_a/e_a = -0.3
poles = [(1.0, 0.3), (0.5, 0.15)]
iw = 1j * np.pi / beta * iaft.wn_mesh("fermion")
Sigma_w = np.zeros((iw.size, 2, 2), dtype=np.complex128)
for orb, (e, g) in enumerate(poles):
    Sigma_w[:, orb, orb] = g / (iw - e)
Sigma_t = iaft.w_to_tau(Sigma_w, "fermion")
Sigma_tsk = np.ascontiguousarray(
    np.broadcast_to(Sigma_t[:, None, None], (iaft.nt_f, 1, nkpts, 2, 2)))

# ===================================================
# 2. Linearized quasiparticle approximation
# ===================================================
# converge=True (default): the fit window grows until the fit stops being reliable, and the
# last reliable fit is kept in ladder.last.
ladder = lqp_from_arrays(mpi=mpi, F_sk=F_sk, Sigma_tsk=Sigma_tsk, mu=mu, iaft=iaft)
res = ladder.last

# ===================================================
# 3. Results, against the closed form
# ===================================================
z = np.array([1.0 / (1.0 + g / e**2) for e, g in poles])
A = np.diag([-g / e for e, g in poles])
identity = np.eye(2)
Hqp_exact = (np.sqrt(z)[:, None] * (F_sk[0] + A - mu * identity) * np.sqrt(z)[None, :]
             + mu * identity)
E_exact = np.linalg.eigvalsh(Hqp_exact)

if mpi.root():
    print(f"\n### Linearized QP: n_fit = {res.n_fit} (fit order {res.fit_order}) ###")
    print(f"  z_a (closed form)              = {z}")
    print(f"  max |Z - diag(z_a)|            = "
          f"{np.max(np.abs(res.Z_skab[0] - np.diag(z))):.1e}")
    print(f"  max |E_QP - E_QP(closed form)| = {np.max(np.abs(res.E_ska[0] - E_exact)):.1e} Ha")
    print(f"  pole weights <v|Z|v> along k   : "
          f"min = {res.Zqp_ska.min():.4f}, max = {res.Zqp_ska.max():.4f}")

    # mean-field and quasiparticle bands; pole weights along k
    fig, (ax_bands, ax_weights) = plt.subplots(2, 1, figsize=(7, 8), dpi=100, sharex=True)
    E_mf = np.linalg.eigvalsh(F_sk[0])
    for band in range(2):
        ax_bands.plot(kpts, E_mf[:, band] - mu, color="tab:blue", linestyle="--",
                      label="F" if band == 0 else None)
        ax_bands.plot(kpts, res.E_ska[0, :, band] - mu, color="tab:red",
                      label="lqp" if band == 0 else None)
        ax_weights.plot(kpts, res.Zqp_ska[0, :, band], label=f"band {band + 1}")
    for orb in range(2):
        ax_weights.axhline(y=z[orb], color="gray", linestyle=":")
    ax_bands.set_ylabel(r"$\epsilon - \mu$ (Ha)")
    ax_bands.legend()
    ax_weights.set_xlabel("k")
    ax_weights.set_ylabel(r"pole weight $\langle v|Z|v \rangle$")
    ax_weights.legend()
    plt.tight_layout()
    plt.savefig("lqp_from_arrays.png", format="png")

mpi.barrier()
