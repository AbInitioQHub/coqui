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
Hubbard–Holstein dimer: the exactly solvable reference of the impurity-solver tests.

One impurity orbital (level eps, static U), one bath site (eps_b, hopping V) and one boson mode
omega0 coupled to the impurity charge with strength g:

    H = eps (n_up + n_dn) + U n_up n_dn + eps_b (n_cup + n_cdn) + V sum_s (d_s^+ c_s + h.c.)
        + omega0 b^+ b + g (n_up + n_dn) (b + b^+)

The boson couples to the charge n_d itself, not to n_d minus a constant: integrating it out gives
exactly the action the solvers receive, with the retarded part's diagonal (the polaron shift)
included:

    Delta(i w) = V^2 / (i w - eps_b)            both spins
    V_abcd     = U                              product basis, norb = 1
    u(i nu)    = -2 g^2 omega0 / (omega0^2 + nu^2)   retarded part of the bosonic Weiss field

Everything is a dense matrix in NumPy (16 fermionic states x (n_boson + 1) boson states).
Energies in Hartree, as in the package.
"""
from dataclasses import dataclass

import numpy as np

from coqui.utils.imag_axes_ft import IAFT
from coqui.dmft.solver_api import ImpurityProblem

GF_STRUCT = [("up_0", 1), ("down_0", 1)]


@dataclass(frozen=True)
class DimerParams:
  beta: float = 20.0
  eps: float = -0.2          # impurity occupation ~1.23 with the other defaults (mid filling)
  U: float = 2.0
  eps_b: float = 0.2
  V: float = 0.5
  omega0: float = 1.0
  g: float = 0.7
  n_boson: int = 20


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


def build_hamiltonian(params):
  """The Hamiltonian and the operators the reference needs, as dense matrices."""
  d_up, d_dn, c_up, c_dn = _jordan_wigner(4)
  nb = params.n_boson + 1
  boson = np.diag(np.sqrt(np.arange(1, nb)), k=1)
  fermion_identity, boson_identity = np.eye(16), np.eye(nb)

  def fermionic(op):
    return np.kron(op, boson_identity)

  def bosonic(op):
    return np.kron(fermion_identity, op)

  n_up, n_dn = fermionic(d_up.T @ d_up), fermionic(d_dn.T @ d_dn)
  n_cup, n_cdn = fermionic(c_up.T @ c_up), fermionic(c_dn.T @ c_dn)
  hopping = fermionic(d_up.T @ c_up + c_up.T @ d_up + d_dn.T @ c_dn + c_dn.T @ d_dn)
  displacement = bosonic(boson + boson.T)
  charge = n_up + n_dn
  hamiltonian = (params.eps * charge + params.U * (n_up @ n_dn)
                 + params.eps_b * (n_cup + n_cdn) + params.V * hopping
                 + params.omega0 * bosonic(boson.T @ boson) + params.g * (charge @ displacement))
  ops = {"d_up": fermionic(d_up), "d_dn": fermionic(d_dn), "n_up": n_up, "n_dn": n_dn,
         "X": displacement}
  return hamiltonian, ops


class DimerED:
  """Thermal expectation values and Lehmann functions of the dimer."""

  def __init__(self, params):
    self.params = params
    hamiltonian, ops = build_hamiltonian(params)
    energies, vectors = np.linalg.eigh(hamiltonian)
    energies = energies - energies.min()
    self.energies = energies
    self.weights = np.exp(-params.beta * energies)
    self.partition = self.weights.sum()
    self.ops = {name: vectors.T @ op @ vectors for name, op in ops.items()}

  def expect(self, name):
    return float(np.sum(self.weights * np.diag(self.ops[name])) / self.partition)

  def occupations(self):
    return np.array([self.expect("n_up"), self.expect("n_dn")])

  def green(self, wn_f):
    """G_sigma(i w_n), w_n = n pi/beta for the odd integers wn_f; shape (nw, 2)."""
    iw = 1j * np.pi * np.asarray(wn_f) / self.params.beta
    out = np.zeros((len(iw), 2), dtype=complex)
    de = self.energies[None, :] - self.energies[:, None]          # E_n - E_m
    for spin, name in enumerate(("d_up", "d_dn")):
      amplitude = self.ops[name]                                  # <m| d |n>
      weight = amplitude**2 * (self.weights[:, None] + self.weights[None, :]) / self.partition
      for index, z in enumerate(iw):
        out[index, spin] = np.sum(weight / (z - de))
    return out

  def chi_nn(self, wn_b_pos):
    """Connected <n_a(tau) n_b(0)> at nu = n pi/beta for the even integers wn_b_pos >= 0;
    shape (nw, 2, 2), real. The disconnected beta n_a n_b is removed at nu = 0."""
    beta = self.params.beta
    nu = np.pi * np.asarray(wn_b_pos) / beta
    if nu[0] != 0.0:
      raise ValueError("hubbard_holstein_dimer::chi_nn: the bosonic mesh must start at nu = 0")
    de = self.energies[None, :] - self.energies[:, None]
    dw = (self.weights[None, :] - self.weights[:, None]) / self.partition
    degenerate = np.abs(de) < 1e-12
    static = np.where(degenerate, beta * self.weights[:, None] / self.partition,
                      dw / np.where(degenerate, 1.0, -de))
    densities = {0: self.ops["n_up"], 1: self.ops["n_dn"]}
    out = np.zeros((len(nu), 2, 2))
    for a in range(2):
      for b in range(2):
        product = densities[a] * densities[b].T                   # <m|n_a|n><n|n_b|m>
        out[0, a, b] = np.sum(product * static)
        for index in range(1, len(nu)):
          out[index, a, b] = np.sum(product * dw / (1j * nu[index] - de)).real
    occ = self.occupations()
    out[0] -= beta * np.outer(occ, occ)
    return out

  def reference(self, iaft):
    """Every quantity the solver tests compare with, on the meshes of ``iaft``."""
    p = self.params
    wn_f = iaft.wn_mesh('f')
    wn_b = iaft.wn_mesh('b', positive_only=True)
    iw = 1j * np.pi * wn_f / p.beta
    green = self.green(wn_f)
    delta = p.V**2 / (iw - p.eps_b)
    g0 = 1.0 / (iw - p.eps - delta)
    sigma = 1.0 / g0[:, None] - 1.0 / green
    occ = self.occupations()
    x_mean = self.expect("X")
    sigma_infty = np.array([p.U * occ[1] + p.g * x_mean, p.U * occ[0] + p.g * x_mean])
    chi_dd = self.chi_nn(wn_b).sum(axis=(1, 2))
    nu = np.pi * wn_b / p.beta
    u_iw = -2.0 * p.g**2 * p.omega0 / (p.omega0**2 + nu**2)
    u_full = p.U + u_iw
    pi_iw = chi_dd / (u_full * chi_dd - 1.0)
    w_iw = u_full - u_full * chi_dd * u_full - p.U
    return dict(wn_f=wn_f, wn_b=wn_b, delta_iw=delta, G_iw=green, Sigma_iw=sigma,
                Sigma_infty=sigma_infty, Sigma_dyn_iw=sigma - sigma_infty[None, :],
                occupations=occ, chi_dd=chi_dd, u_iw=u_iw, Pi_iw=pi_iw, W_iw=w_iw)


def make_iaft(params, wmax=20.0, eps=1e-10):
  return IAFT(beta=params.beta, wmax=wmax, prec="custom", eps=eps, verbose=False, basis="dlr")


def make_problem(params, iaft, *, dlr_wmax, dlr_eps, degenerate_blk=((0, 1),)):
  """The dimer as an ImpurityProblem (unshifted h0; target density from ED)."""
  ed = DimerED(params)
  ref = ed.reference(iaft)
  nw_f, nw_b = len(ref["wn_f"]), len(ref["wn_b"])
  delta = np.zeros((nw_f, 2, 1, 1), dtype=complex)
  delta[:, 0, 0, 0] = delta[:, 1, 0, 0] = ref["delta_iw"]
  occ = ref["occupations"]
  return ImpurityProblem(
    gf_struct=GF_STRUCT, beta=params.beta, iaft=iaft, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps,
    h0_sab=np.full((2, 1, 1), params.eps, dtype=complex), delta_wsab=delta,
    v_abcd=np.full((1, 1, 1, 1), params.U, dtype=complex),
    u_weiss_wabcd=ref["u_iw"].reshape(nw_b, 1, 1, 1, 1).astype(complex),
    target_density=float(occ.sum()),
    occupations_sab=np.array([[[occ[0]]], [[occ[1]]]]),
    degenerate_blk=[np.array(group) for group in degenerate_blk] if degenerate_blk else None)
