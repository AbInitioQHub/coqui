"""
==========================================================================
CoQuí: Correlated Quantum ínterface

Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==========================================================================
"""
"""
Two-orbital Kanamori–Holstein impurity: the exactly solvable reference for a non-density-density
impurity interaction.

Two impurity orbitals (level eps) with the Kanamori interaction given as the full 4-index tensor
in CoQuí's product-basis convention, ``v_abcd <-> c^+_b c^+_c c_d c_a``:

    v[i, i, i, i] = U        v[i, i, j, j] = U'       (density-density)
    v[i, j, i, j] = J        (spin exchange, c^+_j c^+_i c_j c_i)
    v[i, j, j, i] = J        (pair hopping, c^+_j c^+_j c_i c_i)

    H_int = 1/2 sum_{abcd} sum_{ss'} v_abcd d^+_{bs} d^+_{cs'} d_{ds'} d_{as}

each orbital hybridized with its own bath site (eps_b, hopping V), and one boson mode omega0
coupled to the total impurity charge with strength g, as in the one-orbital dimer:

    H = eps N + H_int + eps_b N_bath + V sum_{as} (d^+_{as} c_{as} + h.c.)
        + omega0 b^+ b + g N (b + b^+),   N = sum_{as} n_{as}

    Delta(i w)      = V^2 / (i w - eps_b)                 every spin-orbital
    u_{aabb}(i nu)  = -2 g^2 omega0 / (omega0^2 + nu^2)   every orbital pair (the boson couples to N)

The static Sigma_infty is the exact omega^0 moment: the Hartree-Fock contractions of the full
tensor with the exact density matrix, plus the polaron shift g <X>. For a diagonal density matrix
it does not depend on the spin-flip and pair-hopping terms; the dynamic self-energy does, which is
what a test of a non-density ``h_int`` has to see.

Everything is a dense matrix in NumPy: 8 fermionic modes (256 states) x (n_boson + 1) boson
states. Energies in Hartree, as in the package. The colour order is that of ``GF_STRUCT``.
"""
from dataclasses import dataclass

import numpy as np

from coqui.utils.imag_axes_ft import IAFT
from coqui.dmft.solver_api import ImpurityProblem

GF_STRUCT = [("up_0", 1), ("up_1", 1), ("down_0", 1), ("down_1", 1)]
NORB = 2
# colour c = index into GF_STRUCT; (orbital, spin) of every colour
COLOR_ORBITAL = [0, 1, 0, 1]
COLOR_SPIN = [0, 0, 1, 1]


@dataclass(frozen=True)
class KanamoriParams:
  beta: float = 20.0
  eps: float = -0.9          # ~2.3 electrons on the impurity with the other defaults (Hund's regime)
  U: float = 2.0
  Uprime: float = 0.6        # U - 2J: rotationally invariant Kanamori
  J: float = 0.7             # large, so that the spin-flip and pair-hopping terms are visible above the Monte Carlo spread
  eps_b: float = 0.3
  V: float = 1.0           # strong enough hybridization for CT-HYB to sample G(tau) well (order ~40)
  omega0: float = 1.0
  g: float = 0.5
  n_boson: int = 20


def kanamori_tensor(U, Uprime, J, norb=NORB):
  """The Kanamori interaction as a product-basis tensor ``v_abcd <-> c^+_b c^+_c c_d c_a``."""
  v = np.zeros((norb, norb, norb, norb))
  for i in range(norb):
    v[i, i, i, i] = U
    for j in range(norb):
      if i != j:
        v[i, i, j, j] = Uprime
        v[i, j, i, j] = J                # spin exchange
        v[i, j, j, i] = J                # pair hopping
  return v


def _jordan_wigner(n_modes):
  """Annihilation operators of n_modes fermionic modes as dense (2^n, 2^n) real matrices."""
  identity = np.eye(2)
  parity = np.diag([1.0, -1.0])
  lower = np.array([[0.0, 1.0], [0.0, 0.0]])      # basis (|0>, |1>): a|1> = |0>
  ops = []
  for mode in range(n_modes):
    factors = [parity] * mode + [lower] + [identity] * (n_modes - mode - 1)
    op = factors[0]
    for factor in factors[1:]:
      op = np.kron(op, factor)
    ops.append(op)
  return ops


def interaction_operator(v_abcd, d_ops):
  """
  ``1/2 sum_{abcd} sum_{ss'} v_abcd d^+_{bs} d^+_{cs'} d_{ds'} d_{as}`` as a dense matrix, from the
  annihilation operators ``d_ops[s][a]`` (spin, orbital).
  """
  norb = v_abcd.shape[0]
  dim = d_ops[0][0].shape[0]
  h_int = np.zeros((dim, dim))
  for a in range(norb):
    for b in range(norb):
      for c in range(norb):
        for d in range(norb):
          coef = v_abcd[a, b, c, d]
          if coef == 0.0:
            continue
          for s in range(2):
            for sp in range(2):
              h_int += 0.5 * coef * (d_ops[s][b].T @ d_ops[sp][c].T @ d_ops[sp][d] @ d_ops[s][a])
  return h_int


def build_hamiltonian(params):
  """The Hamiltonian and the operators the reference needs, as dense matrices."""
  modes = _jordan_wigner(8)
  # colour order of GF_STRUCT for the impurity: (0 up), (1 up), (0 dn), (1 dn); then the bath sites
  d_ops = [[modes[0], modes[1]], [modes[2], modes[3]]]          # [spin][orbital]
  c_ops = [[modes[4], modes[5]], [modes[6], modes[7]]]
  nb = params.n_boson + 1
  boson = np.diag(np.sqrt(np.arange(1, nb)), k=1)
  fermion_identity, boson_identity = np.eye(256), np.eye(nb)

  def fermionic(op):
    return np.kron(op, boson_identity)

  def bosonic(op):
    return np.kron(fermion_identity, op)

  n_imp = [fermionic(d_ops[s][a].T @ d_ops[s][a]) for s in range(2) for a in range(NORB)]
  charge = sum(n_imp)
  n_bath = sum(fermionic(c_ops[s][a].T @ c_ops[s][a]) for s in range(2) for a in range(NORB))
  hopping = sum(fermionic(d_ops[s][a].T @ c_ops[s][a] + c_ops[s][a].T @ d_ops[s][a])
                for s in range(2) for a in range(NORB))
  displacement = bosonic(boson + boson.T)
  v_abcd = kanamori_tensor(params.U, params.Uprime, params.J)
  hamiltonian = (params.eps * charge + fermionic(interaction_operator(v_abcd, d_ops))
                 + params.eps_b * n_bath + params.V * hopping
                 + params.omega0 * bosonic(boson.T @ boson) + params.g * (charge @ displacement))
  colours = [fermionic(d_ops[COLOR_SPIN[c]][COLOR_ORBITAL[c]]) for c in range(4)]
  return hamiltonian, {"d": colours, "X": displacement, "v_abcd": v_abcd}


def hartree_fock_sigma_infty(v_abcd, rho):
  """
  The Hartree-Fock self-energy of ``1/2 sum v_abcd d^+_b d^+_c d_d d_a`` (every spin combination)
  for the colour density matrix ``rho[x, y] = <d^+_x d_y>``: the four Wick contractions of every
  term, as a (4, 4) matrix over the colours of ``GF_STRUCT``. Independent of the package's
  ``sigma_infty_hf``, which works on the TRIQS operator.
  """
  sig = np.zeros((4, 4))
  colour = {(COLOR_ORBITAL[c], COLOR_SPIN[c]): c for c in range(4)}
  for a in range(NORB):
    for b in range(NORB):
      for c in range(NORB):
        for d in range(NORB):
          coef = 0.5 * v_abcd[a, b, c, d]
          if coef == 0.0:
            continue
          for s in range(2):
            for sp in range(2):
              i, j, k, l = colour[b, s], colour[c, sp], colour[d, sp], colour[a, s]
              # coef d^+_i d^+_j d_k d_l: the four contractions, coefficient of d^+_x d_y into sig[x, y]
              sig[i, l] += coef * rho[j, k]
              sig[j, k] += coef * rho[i, l]
              sig[i, k] -= coef * rho[j, l]
              sig[j, l] -= coef * rho[i, k]
  return sig


class KanamoriED:
  """Thermal expectation values and Lehmann functions of the two-orbital impurity."""

  def __init__(self, params):
    self.params = params
    hamiltonian, ops = build_hamiltonian(params)
    energies, vectors = np.linalg.eigh(hamiltonian)
    energies = energies - energies.min()
    self.energies = energies
    self.weights = np.exp(-params.beta * energies)
    self.partition = self.weights.sum()
    self.v_abcd = ops["v_abcd"]
    self.d = [vectors.T @ op @ vectors for op in ops["d"]]       # annihilation operators, eigenbasis
    self.x_mean = float(np.sum(self.weights * np.diag(vectors.T @ ops["X"] @ vectors)) / self.partition)

  def density_matrix(self):
    """``rho[c, c'] = <d^+_c d_c'>`` over the four colours."""
    rho = np.zeros((4, 4))
    for c in range(4):
      for cp in range(4):
        rho[c, cp] = np.sum(self.weights * np.diag(self.d[c].T @ self.d[cp])) / self.partition
    return rho

  def occupations(self):
    return np.diag(self.density_matrix()).copy()

  def green(self, wn_f):
    """G_c(i w_n), w_n = n pi/beta for the odd integers wn_f; shape (nw, 4), colour-diagonal."""
    iw = 1j * np.pi * np.asarray(wn_f) / self.params.beta
    out = np.zeros((len(iw), 4), dtype=complex)
    de = self.energies[None, :] - self.energies[:, None]          # E_n - E_m
    for c in range(4):
      amplitude = self.d[c]                                       # <m| d |n>
      weight = amplitude**2 * (self.weights[:, None] + self.weights[None, :]) / self.partition
      m_idx, n_idx = np.nonzero(np.abs(weight) > 1e-14)
      w_nz, de_nz = weight[m_idx, n_idx], de[m_idx, n_idx]
      for index, z in enumerate(iw):
        out[index, c] = np.sum(w_nz / (z - de_nz))
    return out

  def chi_nn(self, wn_b_pos):
    """
    Connected ``<N_i(tau) N_j(0)>`` of the spin-summed orbital densities ``N_i = n_{i up} + n_{i dn}``
    at ``nu = n pi / beta`` for the even integers ``wn_b_pos >= 0``; shape ``(nw, norb, norb)``,
    real. The disconnected ``beta N_i N_j`` is removed at ``nu = 0``. (The spin-resolved correlator
    summed over spins is the same quantity; the orbital densities keep the Lehmann sums to three
    pairs.)
    """
    beta = self.params.beta
    nu = np.pi * np.asarray(wn_b_pos) / beta
    if nu[0] != 0.0:
      raise ValueError("kanamori_holstein_dimer::chi_nn: the bosonic mesh must start at nu = 0")
    de = self.energies[None, :] - self.energies[:, None]
    dw = (self.weights[None, :] - self.weights[:, None]) / self.partition
    degenerate = np.abs(de) < 1e-12
    static = np.where(degenerate, beta * self.weights[:, None] / self.partition,
                      dw / np.where(degenerate, 1.0, -de))
    densities = [sum(self.d[c].T @ self.d[c] for c in range(4) if COLOR_ORBITAL[c] == i) for i in range(NORB)]
    occ_orb = np.array([np.sum(self.weights * np.diag(n_i)) / self.partition for n_i in densities])
    out = np.zeros((len(nu), NORB, NORB))
    for i in range(NORB):
      for j in range(i, NORB):
        product = densities[i] * densities[j].T
        m_idx, n_idx = np.nonzero(np.abs(product) > 1e-14)
        prod_nz, de_nz, dw_nz, static_nz = product[m_idx, n_idx], de[m_idx, n_idx], dw[m_idx, n_idx], static[m_idx, n_idx]
        out[0, i, j] = np.sum(prod_nz * static_nz)
        out[1:, i, j] = np.real(np.einsum("k,wk->w", prod_nz * dw_nz, 1.0 / (1j * nu[1:, None] - de_nz[None, :])))
        out[:, j, i] = out[:, i, j]
    out[0] -= beta * np.outer(occ_orb, occ_orb)
    return out

  def sigma_infty(self):
    """
    The exact omega^0 moment of the self-energy per colour: the Hartree-Fock contractions of
    ``1/2 sum v_abcd d^+_b d^+_c d_d d_a`` (every spin combination) with the exact density
    matrix, plus the polaron shift ``g <X>``. For the exact eigenstates the density matrix is
    diagonal in the colours, which the general contraction below does not assume.
    """
    sig = hartree_fock_sigma_infty(self.v_abcd, self.density_matrix())
    return np.diag(sig) + self.params.g * self.x_mean

  def reference(self, iaft, n_bose=2):
    """
    Every quantity the solver tests compare with, on the meshes of ``iaft``; the bosonic quantities
    (``chi_dd``, ``u_iw``, ``Pi_iw``, ``W_iw``) on the first ``n_bose`` non-negative bosonic points
    only, since every Lehmann sum of the density correlator costs seconds.
    """
    p = self.params
    wn_f = iaft.wn_mesh('f')
    iw = 1j * np.pi * wn_f / p.beta
    green = self.green(wn_f)
    delta = p.V**2 / (iw - p.eps_b)
    g0 = 1.0 / (iw - p.eps - delta)
    sigma = 1.0 / g0[:, None] - 1.0 / green
    sigma_infty = self.sigma_infty()
    # the density-density channel: chi summed over spins per orbital pair, U_dd = V_dd + u(nu)
    # between opposite spins (v[i, i, j, j]), Pi = chi (U chi - 1)^-1 and W = U - U chi U - V_dd as
    # matrices over the orbitals, the convention of the backends' pi_w_from_chi
    wn_b = iaft.wn_mesh('b', positive_only=True)[:n_bose]
    nu = np.pi * wn_b / p.beta
    chi_dd = self.chi_nn(wn_b)
    u_iw = -2.0 * p.g**2 * p.omega0 / (p.omega0**2 + nu**2)
    v_dd = np.array([[self.v_abcd[i, i, j, j] for j in range(NORB)] for i in range(NORB)])
    pi_iw = np.zeros_like(chi_dd)
    w_iw = np.zeros_like(chi_dd)
    for index in range(len(wn_b)):
      u_full = v_dd + u_iw[index]
      pi_iw[index] = chi_dd[index] @ np.linalg.inv(u_full @ chi_dd[index] - np.eye(NORB))
      w_iw[index] = u_full - u_full @ chi_dd[index] @ u_full - v_dd
    return dict(wn_f=wn_f, wn_b=wn_b, delta_iw=delta, G_iw=green, Sigma_iw=sigma, Sigma_infty=sigma_infty,
                Sigma_dyn_iw=sigma - sigma_infty[None, :], occupations=self.occupations(),
                density_matrix=self.density_matrix(), x_mean=self.x_mean, chi_dd=chi_dd, u_iw=u_iw,
                Pi_iw=pi_iw, W_iw=w_iw)


def make_iaft(params, wmax=20.0, eps=1e-10):
  return IAFT(beta=params.beta, wmax=wmax, prec="custom", eps=eps, verbose=False, basis="dlr")


def make_problem(params, iaft, *, dlr_wmax, dlr_eps, degenerate_blk=((0, 1, 2, 3),)):
  """The two-orbital impurity as an ImpurityProblem (unshifted h0; target density from ED)."""
  ed = KanamoriED(params)
  ref = ed.reference(iaft)
  nw_f = len(ref["wn_f"])
  wn_b = iaft.wn_mesh('b', positive_only=True)
  nu = np.pi * wn_b / params.beta
  u_iw = -2.0 * params.g**2 * params.omega0 / (params.omega0**2 + nu**2)
  delta = np.zeros((nw_f, 2, NORB, NORB), dtype=complex)
  for s in range(2):
    for a in range(NORB):
      delta[:, s, a, a] = ref["delta_iw"]
  u_weiss = np.zeros((len(wn_b), NORB, NORB, NORB, NORB), dtype=complex)
  for a in range(NORB):
    for b in range(NORB):
      u_weiss[:, a, a, b, b] = u_iw
  occ = ref["occupations"]                                        # colour order of GF_STRUCT
  occupations_sab = np.zeros((2, NORB, NORB))
  for c in range(4):
    occupations_sab[COLOR_SPIN[c], COLOR_ORBITAL[c], COLOR_ORBITAL[c]] = occ[c]
  return ImpurityProblem(
    gf_struct=GF_STRUCT, beta=params.beta, iaft=iaft, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps,
    h0_sab=np.array([np.eye(NORB) * params.eps] * 2, dtype=complex), delta_wsab=delta,
    v_abcd=kanamori_tensor(params.U, params.Uprime, params.J).astype(complex),
    u_weiss_wabcd=u_weiss, target_density=float(occ.sum()), occupations_sab=occupations_sab,
    degenerate_blk=[np.array(group) for group in degenerate_blk] if degenerate_blk else None)
