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
Post-processing numerics shared by the CT-INT and CT-HYB backends of the impurity-solver
interface: the colour bookkeeping of ``gf_struct``, evaluation of DLR objects at the IAFT points,
the averaging over degenerate blocks, the Hartree-Fock static self-energy, and the density
correlator to Pi/W step. Each backend owns its post-processing sequence; this module holds the
arithmetic they share. TRIQS-only imports: no solver package is needed to import it.

The CT-SEG backend does not use this module: ``ctseg/ctseg_utils.py`` carries its own versions of
the same steps (``seed_for_rank``, the solver-output redirect, ``compute_sigma_infty``,
``post_process_pi``, ModEST's degenerate-block symmetrization), written on TRIQS Green's
functions rather than on colour arrays. The three backends therefore implement the same physics
in two places, and a change to one of these steps (the Hartree-Fock contractions, the Dyson step
of Pi and W, the degenerate averaging, the seed policy) has to be made in both until the CT-SEG
post-processing is moved onto this module.
"""
import os
import sys
from contextlib import contextmanager

import numpy as np
import triqs.utility.mpi as mpi
from triqs.gfs import Gf, BlockGf, MeshImFreq, MeshDLR, make_gf_dlr, make_gf_imtime, fit_gf_dlr
from triqs.operators.util.extractors import extract_U_dict2, dict_to_matrix


@contextmanager
def redirect_solver_output(suppress_output=False, output_file=None):
  """
  Redirect the native stdout/stderr of the TRIQS solve to ``output_file`` (default: discarded)
  while ``suppress_output`` is set; a no-op otherwise. The chemical-potential search sets
  ``suppress_solver_output`` by default (``resolve_mu_search_params``). The same context manager
  as CT-SEG's (``ctseg_solver`` needs ``triqs_ctseg`` to import).
  """
  if not suppress_output:
    yield
    return
  target = open(output_file if output_file else os.devnull, "a", buffering=1)
  saved_stdout_fd = saved_stderr_fd = None
  try:
    try:
      sys.stdout.flush()
      sys.stderr.flush()
    except Exception:
      pass
    saved_stdout_fd, saved_stderr_fd = os.dup(1), os.dup(2)
    os.dup2(target.fileno(), 1)
    os.dup2(target.fileno(), 2)
    yield
  finally:
    try:
      if saved_stdout_fd is not None:
        os.dup2(saved_stdout_fd, 1)
        os.close(saved_stdout_fd)
      if saved_stderr_fd is not None:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)
    finally:
      target.close()


# ---------------------------------------------------------------- colours

def color_table(gf_struct):
  """
  The block, position in the block, spin and orbital of every colour of ``gf_struct``.

  Colours are numbered in ``gf_struct`` order; the spin is 0 for a block name starting with
  ``"up"`` and 1 otherwise; the orbital counts the colours of one spin in order.

  Parameters
  ----------
  gf_struct : list of (str, int)

  Returns
  -------
  block, inner, spin, orb : np.ndarray of int, each of length ``n_color``
  """
  rows, first = [], [0, 0]
  for b, (name, size) in enumerate(gf_struct):
    s = 0 if name.startswith("up") else 1
    rows += [(b, i, s, first[s] + i) for i in range(size)]
    first[s] += size
  table = np.array(rows, dtype=np.int64).reshape(-1, 4)
  return table[:, 0], table[:, 1], table[:, 2], table[:, 3]


def block_offsets(gf_struct):
  """The first colour of every block of ``gf_struct``, as a list of int."""
  return [int(x) for x in np.cumsum([0] + [size for _, size in gf_struct])[:-1]]


def color_groups(gf_struct, degenerate_blk):
  """The colour groups of the block groups ``degenerate_blk`` (``None`` or empty gives ``[]``)."""
  if not degenerate_blk:
    return []
  block, _, _, _ = color_table(gf_struct)
  return [[int(c) for b in group for c in np.flatnonzero(block == int(b))] for group in degenerate_blk]


def seed_for_rank(base, rank=None):
  """
  The random seed of MPI rank ``rank`` (default: this rank) for a user-given base seed:
  ``(base + 928374 * rank) mod (2^31 - 1)``, the policy of the CT-SEG backend (its copy lives in
  ``ctseg_utils``, which needs ``triqs_ctseg`` to import).
  """
  rank = mpi.rank if rank is None else rank
  return (int(base) + 928374 * int(rank)) % (2**31 - 1)


# ---------------------------------------------------------------- TRIQS objects to arrays

def eval_on_points(g, n, stat, beta):
  """
  A Green's function on a DLR Matsubara mesh (or its DLR coefficients) at Matsubara integers.

  Parameters
  ----------
  g : triqs.gfs.Gf or BlockGf
      On ``MeshDLRImFreq`` or ``MeshDLR``; a ``BlockGf`` gives a list in block order.
  n : array of int
      Matsubara integers in CoQuí's convention ``iw = n pi / beta`` (odd for fermions, even for
      bosons), e.g. ``iaft.wn_mesh('f')``.
  stat : {"Fermion", "Boson"}
  beta : float

  Returns
  -------
  np.ndarray, shape ``(len(n),) + target_shape``, complex (or a list of them).
  """
  if isinstance(g, BlockGf):
    return [eval_on_points(gb, n, stat, beta) for _, gb in g]
  gd = g if isinstance(g.mesh, MeshDLR) else make_gf_dlr(g)
  n = np.asarray(n, dtype=np.int64)
  k = (n - 1) // 2 if stat == "Fermion" else n // 2
  uni = MeshImFreq(beta=beta, statistic=stat, n_iw=int(np.abs(k).max()) + 2)
  out = np.empty((len(n),) + tuple(g.target_shape), dtype=complex)
  for i, kk in enumerate(k):
    out[i] = gd(uni(int(kk)))
  return out


def static_color_matrix(h_int, gf_struct):
  """
  The density-density coefficients of ``h_int`` as a colour matrix ``(n_color, n_color)``, real, zero diagonal:
  entry ``[c1, c2]`` multiplies ``n_c1 n_c2`` (TRIQS's ``extract_U_dict2`` / ``dict_to_matrix``,
  the same extraction CT-SEG's post-processing uses).
  """
  return np.real(np.asarray(dict_to_matrix(extract_U_dict2(h_int), gf_struct=gf_struct)))


def color_matrix_at_points(block2gf, gf_struct, n_b, beta):
  """
  A density-density ``Block2Gf`` (block pair -> (d1, d2) matrix) as a real colour matrix
  ``(len(n_b), n_color, n_color)`` at the bosonic Matsubara integers ``n_b``.
  """
  offs = block_offsets(gf_struct)
  n_color = sum(size for _, size in gf_struct)
  out = np.zeros((len(n_b), n_color, n_color))
  for b1, (name1, d1) in enumerate(gf_struct):
    for b2, (name2, d2) in enumerate(gf_struct):
      vals = eval_on_points(block2gf[name1, name2], n_b, "Boson", beta)
      out[:, offs[b1]:offs[b1] + d1, offs[b2]:offs[b2] + d2] = vals.real
  return out


def d0_tau_colors(block2gf, gf_struct, n_tau):
  """
  A density-density ``Block2Gf`` on the bosonic DLR mesh as a real ``D(tau)`` colour matrix
  ``(n_tau, n_color, n_color)`` on ``n_tau`` uniform points of [0, beta].
  """
  offs = block_offsets(gf_struct)
  n_color = sum(size for _, size in gf_struct)
  name0 = gf_struct[0][0]
  mesh = block2gf[name0, name0].mesh
  g = Gf(mesh=mesh, target_shape=(n_color, n_color))
  for b1, (name1, d1) in enumerate(gf_struct):
    for b2, (name2, d2) in enumerate(gf_struct):
      g.data[:, offs[b1]:offs[b1] + d1, offs[b2]:offs[b2] + d2] = block2gf[name1, name2].data
  return np.asarray(make_gf_imtime(make_gf_dlr(g), n_tau).data).real.copy()


def eval_uniform_then_dlr(g, n, stat, beta, w_max, eps):
  """
  A Green's function on a uniform Matsubara mesh at Matsubara integers ``n``: read directly from
  the mesh where the point lies inside it, from a DLR of ``g`` beyond it.

  Sampling a noisy function (a Dyson self-energy) onto DLR nodes and interpolating back amplifies
  the noise between the nodes; the stored values never do. The DLR branch needs ``g`` to decay as
  ``1/iw``: pass the dynamic part of a self-energy, never one that still carries its constant, and
  a function smooth at the highest mesh frequencies (tail-fitted), since the DLR extrapolates
  from them.

  Parameters
  ----------
  g : triqs.gfs.Gf or BlockGf
      On ``MeshImFreq``; a ``BlockGf`` gives a list in block order.
  n : array of int
      Matsubara integers, ``iw = n pi / beta`` (odd for fermions, even for bosons).
  stat : {"Fermion", "Boson"}
  beta : float
  w_max, eps : float
      DLR controls of the extrapolation (``triqs.solver_utils.make_gf_dlr_imfreq``).

  Returns
  -------
  np.ndarray, shape ``(len(n),) + target_shape``, complex (or a list of them).
  """
  if isinstance(g, BlockGf):
    return [eval_uniform_then_dlr(gb, n, stat, beta, w_max, eps) for _, gb in g]
  from triqs.solver_utils import make_gf_dlr_imfreq
  n = np.asarray(n, dtype=np.int64)
  k = (n - 1) // 2 if stat == "Fermion" else n // 2
  k_first = next(iter(g.mesh)).index
  inside = (k >= k_first) & (k < k_first + len(g.mesh))
  out = np.empty((len(n),) + tuple(g.target_shape), dtype=complex)
  for i in np.where(inside)[0]:
    out[i] = g.data[int(k[i]) - k_first]
  if not inside.all():
    out[~inside] = eval_on_points(make_gf_dlr_imfreq(g, w_max=w_max, eps=eps), n[~inside], stat, beta)
  return out


# ---------------------------------------------------------------- degenerate groups

def average_blocks(arrays, groups):
  """Average per-block arrays (one per gf_struct block) over each group of blocks."""
  out = [np.array(a, copy=True) for a in arrays]
  for group in groups or []:
    avg = np.mean([out[int(b)] for b in group], axis=0)
    for b in group:
      out[int(b)] = avg.copy()
  return out


def average_color_matrix(M, groups, spin, orb, block, inner):
  """
  Average the last two axes of a colour matrix within each group of colours, per class of pair.

  The class of a pair is (same spin?, same orbital?, same block?, position in block of each),
  so that only pairs of colours at equivalent positions of equivalent blocks are mixed; pairs
  across groups are untouched. This is the colour-matrix form of ModEST's ``symmetrize`` of the
  density correlator.
  """
  out = np.array(M, copy=True)
  for group in groups or []:
    classes = {}
    for c1 in group:
      for c2 in group:
        key = (bool(spin[c1] == spin[c2]), bool(orb[c1] == orb[c2]), bool(block[c1] == block[c2]),
               int(inner[c1]), int(inner[c2]))
        classes.setdefault(key, []).append((int(c1), int(c2)))
    for pairs in classes.values():
      avg = np.mean([out[..., c1, c2] for c1, c2 in pairs], axis=0)
      for c1, c2 in pairs:
        out[..., c1, c2] = avg
  return out


# ---------------------------------------------------------------- Sigma_infty

def sigma_infty_hf(rho_blocks, h_int, d_nu0, gf_struct):
  """
  The static self-energy per block: the Hartree-Fock decoupling of the solved ``h_int`` with the
  block density matrices, plus the Hartree term of the retarded part at nu = 0 on the diagonal
  (the "D diagonal kept" rule; for size-1 blocks this is ``[U + D(0)] n``).

  Parameters
  ----------
  rho_blocks : list of np.ndarray (d, d)
      ``rho[a, a'] = <c+_a' c_a>`` per block (TRIQS's ``G(tau = 0-)`` convention).
  h_int : triqs.operators.Operator
      Terms ``coef c+_i c+_j c_k c_l``; a term coupling two blocks is dropped (its contraction
      vanishes by the block structure).
  d_nu0 : np.ndarray (n_color, n_color)
      The retarded density-density colour matrix at nu = 0.
  gf_struct : list of (str, int)

  Returns
  -------
  list of np.ndarray (d, d), complex
  """
  index = {name: b for b, (name, _) in enumerate(gf_struct)}
  rho = [np.asarray(r, dtype=complex) for r in rho_blocks]
  sig = [np.zeros((d, d), dtype=complex) for _, d in gf_struct]

  def ev(x, y):                                      # <c+_x c_y>
    return rho[x[0]][y[1], x[1]] if x[0] == y[0] else 0.0

  def add(x, y, value):                              # coefficient of c+_x c_y
    if x[0] == y[0]:
      sig[x[0]][x[1], y[1]] += value

  for monomial, coef in h_int:
    if len(monomial) != 4 or [dag for dag, _ in monomial] != [True, True, False, False]:
      raise ValueError(f"qmc_postproc::sigma_infty_hf: h_int term {monomial} is not of the form c+ c+ c c")
    i, j, k, l = [(index[blk], int(a)) for _, (blk, a) in monomial]
    add(i, l, coef * ev(j, k))
    add(j, k, coef * ev(i, l))
    add(i, k, -coef * ev(j, l))
    add(j, l, -coef * ev(i, k))
  block, inner, _, _ = color_table(gf_struct)
  dens_c = np.concatenate([np.diag(r).real for r in rho])
  hartree = np.asarray(d_nu0, dtype=float) @ dens_c
  for c in range(len(dens_c)):
    sig[block[c]][inner[c], inner[c]] += hartree[c]
  return sig


# ---------------------------------------------------------------- chi -> Pi, W

def connected_by_plateau(Q_tau, beta):
  """
  The connected part of a tau-measured ``<n_a(tau) n_b(0)>`` on a uniform [0, beta] grid (leading
  axis tau): the correlator minus its own plateau, the mean over tau in [beta/4, 3 beta/4].
  Subtracting an independently measured ``n_a n_b`` leaves an O(1e-3) constant that chi(nu = 0)
  multiplies by beta. Returns ``(connected, plateau)``.
  """
  n_tau = Q_tau.shape[0]
  lo, hi = n_tau // 4, n_tau - n_tau // 4
  plateau = np.asarray(Q_tau[lo:hi]).mean(axis=0)
  return np.asarray(Q_tau) - plateau, plateau


def fit_tau_correlator(g_tau, dlr_wmax, dlr_eps):
  """
  The DLR fit of a bosonic correlator measured on a uniform tau grid, limited to what the grid
  resolves: ``w_max = min(dlr_wmax, pi / (4 dtau))``. The full ``dlr_wmax`` admits basis functions
  that decay faster than the grid spacing, whose coefficients explode on noisy data.
  Returns ``(Gf on MeshDLR, w_max used)``.
  """
  n_tau = len(g_tau.mesh)
  dtau = g_tau.mesh.beta / (n_tau - 1)
  w_max = float(min(dlr_wmax, np.pi / (4.0 * dtau)))
  return fit_gf_dlr(g_tau, w_max=w_max, eps=dlr_eps, symmetrize=True), w_max


def pi_w_from_chi(chi_c, U_c, D_c, gf_struct, degenerate_blk):
  """
  The impurity polarization and screened interaction from the connected density correlator.

  In the product basis of orbital pairs, with ``U(nu) = V + D(nu)`` in the density channel (the
  exchange entries of the product-basis ``U`` are built but inert for a density-only ``chi``):
  ``Pi = chi (U chi - 1)^-1`` and ``W = U - U chi U``, back to density-density, the bare static
  ``V`` removed from ``W``. The same Dyson step as CT-SEG's ``post_process_pi``.

  Parameters
  ----------
  chi_c : np.ndarray (n_w, n_color, n_color), real
      The connected ``<n_c n_c'>`` at the stored bosonic points (nu >= 0).
  U_c : np.ndarray (n_color, n_color)
      The static density-density colour matrix of the solved ``h_int`` (``static_color_matrix``).
  D_c : np.ndarray (n_w, n_color, n_color)
      The retarded density-density colour matrix at the same points.
  gf_struct : list of (str, int)
  degenerate_blk : list of index groups or None
      ``chi_c`` is averaged over the groups (class-wise) before the Dyson step.

  Returns
  -------
  dict
      ``Pi_dd``, ``W_dd``, ``chi_dd`` of shape ``(n_w, norb, norb)`` real, and ``max_cond``, the
      largest condition number of ``U chi - 1`` over nu.
  """
  block, inner, spin, orb = color_table(gf_struct)
  n_w, n_color = chi_c.shape[0], chi_c.shape[1]
  norb = int(orb.max()) + 1
  chi = 0.5 * (np.real(chi_c) + np.real(chi_c).transpose(0, 2, 1))
  chi = average_color_matrix(chi, color_groups(gf_struct, degenerate_blk), spin, orb, block, inner)
  chi_dd = np.zeros((n_w, norb, norb))
  for c1 in range(n_color):
    for c2 in range(n_color):
      chi_dd[:, orb[c1], orb[c2]] += chi[:, c1, c2]

  def color(s, m):
    return int(np.flatnonzero((spin == s) & (orb == m))[0])
  up = [color(0, m) for m in range(norb)]
  dn = [color(1, m) for m in range(norb)]
  Uprime = U_c[np.ix_(up, dn)]                      # opposite spins
  Usame = U_c[np.ix_(up, up)]                       # same spin (zero diagonal)
  J = Uprime - Usame
  np.fill_diagonal(J, 0.0)
  Dprime = D_c[:, up, :][:, :, dn]
  Dsame = D_c[:, up, :][:, :, up]
  DJ = Dprime - Dsame
  for i in range(norb):
    DJ[:, i, i] = 0.0

  n_pair = norb * norb

  def ii(i, j):
    return i * norb + j
  chi_pb = np.zeros((n_w, n_pair, n_pair))
  U_pb = np.zeros((n_w, n_pair, n_pair))
  for i in range(norb):
    for j in range(norb):
      chi_pb[:, ii(i, i), ii(j, j)] = chi_dd[:, i, j]
      U_pb[:, ii(i, i), ii(j, j)] = Uprime[i, j] + Dprime[:, i, j]
      if i != j:
        U_pb[:, ii(i, j), ii(i, j)] = J[i, j] + DJ[:, i, j]
        U_pb[:, ii(i, j), ii(j, i)] = J[i, j] + DJ[:, i, j]
  eye = np.eye(n_pair)
  Pi_pb = np.zeros_like(chi_pb)
  W_pb = np.zeros_like(chi_pb)
  max_cond = 0.0
  for w in range(n_w):
    denom = U_pb[w] @ chi_pb[w] - eye
    max_cond = max(max_cond, float(np.linalg.cond(denom)))
    Pi_pb[w] = chi_pb[w] @ np.linalg.pinv(denom)
    W_pb[w] = U_pb[w] - U_pb[w] @ chi_pb[w] @ U_pb[w]
  Pi_dd = np.zeros((n_w, norb, norb))
  W_dd = np.zeros((n_w, norb, norb))
  for i in range(norb):
    for j in range(norb):
      Pi_dd[:, i, j] = Pi_pb[:, ii(i, i), ii(j, j)]
      W_dd[:, i, j] = W_pb[:, ii(i, i), ii(j, j)] - Uprime[i, j]
  return {"Pi_dd": Pi_dd, "W_dd": W_dd, "chi_dd": chi_dd, "max_cond": max_cond}
