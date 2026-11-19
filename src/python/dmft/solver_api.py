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
Impurity-solver interface of the DMFT package.

The GW+EDMFT workflow hands an ``ImpurityProblem`` (CoQuí-native arrays on the crystal IAFT
meshes, unreduced) to an ``ImpuritySolver`` and receives an ``ImpurityResult`` (native arrays
the loop consumes). Each backend declares ``SolverCapabilities``; the shared TRIQS conversion
reduces the problem to what a backend accepts and logs what it drops. Post-processing (Σ, Π, W)
is the backend's own. A backend is selected by name (the ``type`` of the solver block): the
shipped backends are registered here, and an ``ImpuritySolver`` subclass outside the package
is registered with ``register_solver`` before the workflow runs.
"""
import importlib
import inspect
from dataclasses import dataclass, field

import numpy as np

__all__ = ['ImpurityProblem', 'ImpurityResult', 'SolverCapabilities',
           'ImpuritySolver', 'register_solver', 'registered_solvers', 'make_solver',
           'qmc_cycle_params', 'resolve_mu_search_params',
           'check_solver_capabilities_for_gw_edmft', 'check_solver_capabilities_for_problem']
# Static interaction a backend can take:
#   "density"       the density-density part of v_abcd only
#   "kanamori"      the Kanamori elements (density-density, spin flip, pair hopping)
#   "full"          the full v_abcd tensor
_H_INT_KINDS = ("density", "kanamori", "full")

# Retarded interaction a backend can take (the workflow builds the density-density coupling
# only; a retarded spin-spin coupling would be a new kind here and a new workflow path):
#   "none"          static interaction only (u_weiss_wabcd must be zero)
#   "density"       the retarded density-density coupling U(i nu) n_a n_b (CT-SEG's D0, ctint's D0)
_RETARDED_KINDS = ("none", "density")

# Whether a backend returns the impurity polarization (Pi_iw_data / W_iw_data):
#   "always"               for every static interaction it accepts
#   "density_h_int_only"   only with h_int = "density" (cthyb measures the density correlator
#                          through the Lang-Firsov route, which needs densities commuting with h_loc)
#   "never"                it does not measure the density correlator; usable with screen_type = "rpa"
_POLARIZATION_KINDS = ("always", "density_h_int_only", "never")


def _norb_of(gf_struct):
  """Orbitals per spin from ``gf_struct`` (block names ``<spin>_<k>``); both spins must agree."""
  counts = {}
  for name, size in gf_struct:
    spin = name.split('_', 1)[0]
    counts[spin] = counts.get(spin, 0) + int(size)
  if len(set(counts.values())) != 1:
    raise ValueError(f"solver_api::_norb_of: per-spin orbital counts differ: {counts}")
  return next(iter(counts.values()))


def _nw_b_pos(iaft):
  """Length of the non-negative bosonic half mesh, as the ``*_data`` arrays use it."""
  return iaft.nw_b // 2 if iaft.nw_b % 2 == 0 else iaft.nw_b // 2 + 1


@dataclass(frozen=True)
class SolverCapabilities:
  """
  What a backend accepts and returns, declared once on its class.

  The workflow reads ``occupations_only_solve`` and ``polarization`` before any solve; the TRIQS
  conversion reads the other four to decide how far the problem is reduced for the backend.

  Attributes
  ----------
  occupations_only_solve : bool
      The backend can run a cheap solve that returns the occupations only
      (``solve(..., occupations_only=True)``), used by the impurity chemical-potential search.
  general_h_int : bool
      The backend accepts a Kanamori or Slater static interaction (``h_int`` other than
      ``"density"``); otherwise the interaction is reduced to its density-density part.
  offdiag_delta : bool
      The backend accepts ``gf_struct`` blocks of size > 1 with matrix-valued h0 and Delta
      inside a block; otherwise every block must have size 1.
  complex_hamiltonian : bool
      The backend accepts complex h0 and Delta; otherwise both are forced real.
  retarded_interaction : str
      ``"none"`` or ``"density"``: which retarded couplings the backend takes (see
      ``_RETARDED_KINDS``).
  polarization : str
      ``"always"``, ``"density_h_int_only"`` or ``"never"``: when the backend returns the
      impurity polarization (see ``_POLARIZATION_KINDS``).
  """
  occupations_only_solve: bool
  general_h_int: bool
  offdiag_delta: bool
  complex_hamiltonian: bool
  retarded_interaction: str
  polarization: str

  def __post_init__(self):
    if self.retarded_interaction not in _RETARDED_KINDS:
      raise ValueError(f"solver_api::SolverCapabilities: retarded_interaction must be one of "
                       f"{_RETARDED_KINDS}, got {self.retarded_interaction!r}")
    if self.polarization not in _POLARIZATION_KINDS:
      raise ValueError(f"solver_api::SolverCapabilities: polarization must be one of "
                       f"{_POLARIZATION_KINDS}, got {self.polarization!r}")


@dataclass
class ImpurityProblem:
  """
  The impurity problem in CoQuí's conventions (Hartree, product basis), unreduced.

  Array names carry their dimensions as a suffix: ``w`` Matsubara frequency, ``s`` spin,
  ``a``..``d`` Wannier orbitals. Every array is a NumPy array on the sampling meshes of ``iaft``.

  Attributes
  ----------
  gf_struct : list of (str, int)
      Block structure of the impurity, block names ``<spin>_<k>``.
  beta : float
      Inverse temperature.
  iaft : IAFT
      Imaginary-axis transform of the crystal; every array below lives on its sampling meshes.
  dlr_wmax, dlr_eps : float
      DLR controls of the TRIQS meshes a backend builds for this problem.
  h0_sab : ndarray, (2, norb, norb)
      One-body Hamiltonian per spin.
  delta_wsab : ndarray, (nw_f, 2, norb, norb)
      Hybridization on ``iaft.wn_mesh('f')``.
  v_abcd : ndarray, (norb, norb, norb, norb)
      Bare Coulomb interaction.
  u_weiss_wabcd : ndarray, (nw_b_pos, norb, norb, norb, norb)
      Retarded part of the bosonic Weiss field on the non-negative half of ``iaft.wn_mesh('b')``;
      the full static interaction is ``v_abcd + u_weiss_wabcd[0]``.
  target_density : float
      Local density the impurity chemical-potential search aims at.
  occupations_sab : ndarray, (2, norb, norb)
      Density matrix of the local Green's function, stored in ``inputs/`` for backends that need
      occupations before solving (none of the shipped backends reads it).
  degenerate_blk : list of index groups, optional
      Degenerate groups of ``gf_struct`` blocks the backend symmetrizes over.
  mu_shift : float
      Subtracted from ``h0_sab`` by the backend. The chemical-potential search sets it in
      place, so after the search the problem carries the shift the production solve used.
  """
  gf_struct: list
  beta: float
  iaft: object
  dlr_wmax: float
  dlr_eps: float
  h0_sab: np.ndarray
  delta_wsab: np.ndarray
  v_abcd: np.ndarray
  u_weiss_wabcd: np.ndarray
  target_density: float
  occupations_sab: np.ndarray
  degenerate_blk: object = None
  mu_shift: float = 0.0

  def __post_init__(self):
    self.gf_struct = [(str(name), int(size)) for name, size in self.gf_struct]
    norb = _norb_of(self.gf_struct)
    expected = {
      "h0_sab": (2, norb, norb),
      "delta_wsab": (self.iaft.nw_f, 2, norb, norb),
      "v_abcd": (norb,) * 4,
      "u_weiss_wabcd": (_nw_b_pos(self.iaft),) + (norb,) * 4,
      "occupations_sab": (2, norb, norb),
    }
    for name, shape in expected.items():
      got = np.shape(getattr(self, name))
      if got != shape:
        raise ValueError(f"solver_api::ImpurityProblem: {name} has shape {got}, expected {shape}")
    if 0 not in set(int(n) for n in self.iaft.wn_mesh('b')):
      raise ValueError("solver_api::ImpurityProblem: nu = 0 must be on the bosonic mesh")
    if float(self.beta) != float(self.iaft.beta):
      raise ValueError(f"solver_api::ImpurityProblem: beta {self.beta} differs from the IAFT's "
                       f"{self.iaft.beta}")

  @property
  def norb(self):
    return _norb_of(self.gf_struct)


@dataclass
class ImpurityResult:
  """
  Impurity results as NumPy arrays on the meshes of the problem's ``iaft``.

  Backend-native objects (e.g. TRIQS Green's functions, operators) go into
  ``solver_data``. 

  Attributes
  ----------
  orbital_occupations : dict of str -> ndarray, (size,)
      Occupation of every orbital, per ``gf_struct`` block.
  density : float
      Total impurity density.
  Sigma_infty : list of ndarray, (size, size) per block, or None
      Static part of the self-energy.
  Sigma_iw_data : list of ndarray, (nw_f, size, size) per block, or None
      Dynamic self-energy on ``iaft.wn_mesh('f')``, without ``Sigma_infty``.
  G_iw_data : list of ndarray, (nw_f, size, size) per block, or None
      Impurity Green's function on ``iaft.wn_mesh('f')``.
  Pi_iw_data : list of one ndarray, (nw_b_pos, norb, norb), or None
      Density-density impurity polarization, spin-summed, on the non-negative bosonic half
      mesh. Optional: a backend that measured no density correlator leaves it None, and the
      workflow then requires ``screen_type = "rpa"``.
  W_iw_data : list of one ndarray, (nw_b_pos, norb, norb), or None
      Impurity screened interaction without the bare interaction; given together with
      ``Pi_iw_data``.
  diagnostics : dict
      Small h5-writable values (average sign, perturbation order, restart flags, ...), written
      flat into the checkpoint's ``results/`` next to the arrays.
  solver_data : dict
      Backend-native objects of this solve, written to ``impurity_{i}/solver_results/``.
  """
  orbital_occupations: dict
  density: float
  Sigma_infty: object = None
  Sigma_iw_data: object = None
  G_iw_data: object = None
  Pi_iw_data: object = None
  W_iw_data: object = None
  diagnostics: dict = field(default_factory=dict)
  solver_data: dict = field(default_factory=dict)

  def __post_init__(self):
    if (self.Pi_iw_data is None) != (self.W_iw_data is None):
      raise ValueError("solver_api::ImpurityResult: Pi_iw_data and W_iw_data must be given "
                       "together (both or neither)")
    fermionic = [self.Sigma_infty is None, self.Sigma_iw_data is None, self.G_iw_data is None]
    if any(fermionic) and not all(fermionic):
      raise ValueError("solver_api::ImpurityResult: Sigma_infty, Sigma_iw_data and G_iw_data "
                       "must be given together (all or none)")

  @property
  def occupations_only(self):
    """
    True for the result of a occupations-only solve: occupations and density, no Sigma, G, Pi, W.
    """
    return self.Sigma_iw_data is None

  def validate(self, problem):
    """
    Validate the shapes of the result's arrays against the problem's structure. 
    """
    names = [name for name, _ in problem.gf_struct]
    if set(self.orbital_occupations) != set(names):
      raise ValueError(f"solver_api::ImpurityResult: orbital_occupations keys "
                       f"{sorted(self.orbital_occupations)} differ from the blocks {names}")
    for name, size in problem.gf_struct:
      if np.shape(self.orbital_occupations[name]) != (size,):
        raise ValueError(f"solver_api::ImpurityResult: orbital_occupations[{name!r}] has shape "
                         f"{np.shape(self.orbital_occupations[name])}, expected ({size},)")
    if self.occupations_only:
      return
    nw_f = problem.iaft.nw_f
    for key, lead in (("Sigma_infty", ()), ("Sigma_iw_data", (nw_f,)), ("G_iw_data", (nw_f,))):
      blocks = getattr(self, key)
      if len(blocks) != len(problem.gf_struct):
        raise ValueError(f"solver_api::ImpurityResult: {key} has {len(blocks)} blocks, "
                         f"expected {len(problem.gf_struct)}")
      for blk, (_, size) in zip(blocks, problem.gf_struct):
        if np.shape(blk) != lead + (size, size):
          raise ValueError(f"solver_api::ImpurityResult: {key} block has shape {np.shape(blk)}, "
                           f"expected {lead + (size, size)}")
    if self.Pi_iw_data is not None:
      norb = problem.norb
      shape = (_nw_b_pos(problem.iaft), norb, norb)
      for key in ("Pi_iw_data", "W_iw_data"):
        blocks = getattr(self, key)
        if len(blocks) != 1 or np.shape(blocks[0]) != shape:
          raise ValueError(f"solver_api::ImpurityResult: {key} must be one array of shape "
                           f"{shape}")

  def solver_agnostic_results(self):
    """
    The backend-independent results the GW+EDMFT loop needs to proceed, as one flat dict:
    occupations, density, the self-energy, Green's function, polarization and screened interaction
    (``Sigma_infty``, ``Sigma_iw_data``, ``G_iw_data``, ``Pi_iw_data``, ``W_iw_data``; ``None``
    where the backend returned nothing, so that the absence is explicit), and the diagnostics.
    ``solver_data`` is not included.

    Raises
    ------
    ValueError
        For an occupations-only result (the chemical-potential search reads ``density`` directly).
    """
    if self.occupations_only:
      raise ValueError("solver_api::ImpurityResult.solver_agnostic_results: an occupations-only "
                       "result has no solver-agnostic arrays to merge")
    out = {"orbital_occupations": self.orbital_occupations, "density": float(self.density)}
    for key in ("Sigma_infty", "Sigma_iw_data", "G_iw_data", "Pi_iw_data", "W_iw_data"):
      out[key] = getattr(self, key)
    out.update(self.diagnostics)
    return out


class ImpuritySolver:
  """
  Base class of an impurity-solver backend.

  A backend is a subclass that sets the class attributes below, validates and normalizes its
  own parameters in ``__init__``, and implements ``solve``. 

  Attributes
  ----------
  name : str
      Registry name of the backend (``"ctseg"``).
  capabilities : SolverCapabilities
      What the backend accepts and returns.
  h_int_kind : str
      The static interaction the backend was configured for (``"density"``, ``"kanamori"``
      or ``"full"``), set in ``__init__``; read by ``check_solver_capabilities_for_gw_edmft``.
  """
  name = None
  capabilities = None
  h_int_kind = "density"

  def __init__(self, solver_params, *, comm_size):
    """
    Construct the backend from its solver block; a subclass validates and normalizes here.

    Parameters
    ----------
    solver_params : dict
        The ``solver`` block of the impurity without its ``type``: the backend's own keys plus
        whatever it forwards to the underlying solver. Unknown keys are the backend's to
        reject or forward. Its optional ``mu_search`` table holds the parameters of the
        occupations-only solves of the chemical-potential search: the production parameters
        with different QMC cycle controls (a QMC backend derives them with
        ``resolve_mu_search_params``); a backend without the
        ``occupations_only_solve`` capability rejects the table.
    comm_size : int
        Number of MPI ranks, for backends that divide a Monte Carlo budget over ranks
        (``qmc_cycle_params``).

    Raises
    ------
    ValueError, TypeError, ImportError
        An invalid parameter, or a missing dependency of the backend. Raised from data every
        rank holds, before any collective call, so that the failure is uniform over the ranks.
    """
    raise NotImplementedError("solver_api::ImpuritySolver: a backend must implement __init__")

  def solve(self, problem, *, occupations_only=False, chkpt=None):
    """
    Solve one impurity problem. Collective over the MPI ranks.

    Parameters
    ----------
    problem : ImpurityProblem
        The problem, unreduced; the backend reduces it to what it accepts (for the TRIQS
        backends through ``triqs_utils.problem_to_triqs``) and applies ``problem.mu_shift``.
    occupations_only : bool
        Run the cheap solve of the chemical-potential search: the result carries the
        occupations, the density and the diagnostics only. Requires the
        ``occupations_only_solve`` capability.
    chkpt : dict or None
        Where the backend may read and write its own state for this solve (CT-SEG: the
        Monte Carlo configurations under ``impurity_{i}/``): ``h5`` (the DMFT checkpoint
        file), ``impurity`` (its index), ``iteration`` (the DMFT iteration being solved) and
        ``save`` (whether the backend may write). ``None`` when there is no checkpoint. A
        backend never writes into the checkpoint's ``results/``.

    Returns
    -------
    ImpurityResult
        Arrays on the meshes of ``problem.iaft``; ``Pi_iw_data`` and ``W_iw_data`` are
        ``None`` when the backend did not measure the density correlator.
    """
    raise NotImplementedError("solver_api::ImpuritySolver: a backend must implement solve")

  def describe(self):
    """
    Describe the configured backend for the log.

    Returns
    -------
    str
        One block of text, wrapped at about 90 columns, that the workflow logs once per
        impurity: the backend, its capabilities and the resolved parameters (defaults
        included). The base implementation gives the name and the capabilities only.
    """
    return f"{self.name}: {self.capabilities}"


def qmc_cycle_params(solver_params, comm_size):
  """
  Normalize the Monte Carlo cycle controls of one parameter set of a QMC backend.

  Parameters
  ----------
  solver_params : dict
      The parameters of one kind of solve. ``n_cycles`` (the total over MPI ranks),
      ``n_warmup_cycles`` and ``length_cycle`` (both per rank) are required; every other key
      is kept as is.
  comm_size : int
      Number of MPI ranks; every rank runs ``n_cycles / comm_size`` cycles, at least one.

  Returns
  -------
  dict
      ``solver_params`` with the per-rank ``n_cycles`` and integer warm-up and cycle length.

  Raises
  ------
  ValueError
      A required cycle control is missing.
  """
  params = dict(solver_params)
  for key in ("n_cycles", "n_warmup_cycles", "length_cycle"):
    if key not in params:
      raise ValueError(f"solver_api::qmc_cycle_params: the solver block needs '{key}'")
  params["n_cycles"] = max(1, int(int(params["n_cycles"]) / comm_size))
  params["n_warmup_cycles"] = int(params["n_warmup_cycles"])
  params["length_cycle"] = int(params["length_cycle"])
  return params


def resolve_mu_search_params(production_params, mu_search, comm_size):
  """
  Resolve the parameters of the occupations-only solves of the chemical-potential search.

  The search solves run with the production parameters, except for the QMC cycle controls. 

  Parameters
  ----------
  production_params : dict
      The production parameters as the solver block gives them (``n_cycles`` the total over
      ranks), without the ``mu_search`` table.
  mu_search : dict
      The ``mu_search`` table of the solver block. It may set the cycle controls ``n_cycles``
      (total over ranks, default 5 % of the production total), ``n_warmup_cycles``,
      ``length_cycle`` and ``restart_n_warmup_cycles``, and the solver-output keys
      ``suppress_solver_output`` (default True) and ``solver_output_file``. Empty when the
      user set nothing.
  comm_size : int
      Number of MPI ranks, as for ``qmc_cycle_params``.

  Returns
  -------
  dict
      ``production_params`` with the search defaults (``n_cycles`` at 5 %, ``verbosity = 0``,
      the solver output suppressed) and then ``mu_search`` merged on top, normalized by
      ``qmc_cycle_params``. Every key not in ``mu_search`` keeps its production value.

  Raises
  ------
  ValueError
      A ``mu_search`` key that is neither a cycle control nor a solver-output key, or a
      missing production cycle control.
  """
  # what the table may set: the Monte Carlo cycle controls, and where the solver's own console
  # output goes; every other parameter is inherited from the production set
  qmc_cycle_keys = {"n_cycles", "n_warmup_cycles", "length_cycle", "restart_n_warmup_cycles"}
  solver_output_keys = {"suppress_solver_output", "solver_output_file"}
  unknown = set(mu_search) - qmc_cycle_keys - solver_output_keys
  if unknown:
    raise ValueError(f"solver_api::resolve_mu_search_params: unknown solver.mu_search "
                     f"key(s) {sorted(unknown)}; allowed: {sorted(qmc_cycle_keys)} and "
                     f"{sorted(solver_output_keys)}; every other parameter is inherited from "
                     f"the production set")
  if "n_cycles" not in production_params:
    raise ValueError("solver_api::resolve_mu_search_params: the solver block needs "
                     "'n_cycles'")
  params = {**production_params,
            "n_cycles": int(int(production_params["n_cycles"]) * 0.05),
            "verbosity": 0,
            "suppress_solver_output": True,
            **mu_search}
  return qmc_cycle_params(params, comm_size=comm_size)


# Backends by name, as the ``type`` of a solver block selects them. ``register_solver`` adds
# classes here; the shipped backends come from ``_SHIPPED_SOLVERS`` below.
_REGISTERED_SOLVERS = {}

# Shipped backends: name -> (module, class name, TRIQS package it needs). They are imported only
# when selected, so that ``coqui.dmft`` loads without their TRIQS packages.  
_SHIPPED_SOLVERS = {
  "ctseg": ("coqui.dmft.ctseg", "CtsegSolver", "triqs_ctseg"),
  "ctint": ("coqui.dmft.ctint", "CtintSolver", "triqs_ctint"),
  "cthyb": ("coqui.dmft.cthyb", "CthybSolver", "triqs_cthyb"),
}


def registered_solvers():
  """The names ``solver.type`` accepts: the shipped backends and those added with ``register_solver``."""
  return sorted(set(_SHIPPED_SOLVERS) | set(_REGISTERED_SOLVERS))


def register_solver(name, solver_class):
  """
  Register an impurity-solver backend under the name that ``solver.type`` selects it by.

  Parameters
  ----------
  name : str
      The registry name, e.g. ``"mine"`` for ``type = "mine"``.
  solver_class : ImpuritySolver subclass
      The backend class.

  Raises
  ------
  TypeError
      ``solver_class`` is not an ``ImpuritySolver`` subclass.
  """
  if not (inspect.isclass(solver_class) and issubclass(solver_class, ImpuritySolver)):
    raise TypeError(f"solver_api::register_solver: {solver_class!r} is not an ImpuritySolver "
                    f"subclass")
  _REGISTERED_SOLVERS[name] = solver_class


def _import_shipped_solver(name):
  """
  The class of a shipped backend, imported on first use.

  Parameters
  ----------
  name : str
      A key of ``_SHIPPED_SOLVERS``.

  Returns
  -------
  type
      The backend class (checked against ``ImpuritySolver`` by the caller).

  Raises
  ------
  ImportError
      The backend's module cannot be imported; the message names the TRIQS package it needs.
  """
  module_name, class_name, package = _SHIPPED_SOLVERS[name]
  try:
    module = importlib.import_module(module_name)
  except ImportError as error:
    raise ImportError(f"solver_api::make_solver: cannot import the backend {name!r} "
                      f"({module_name}): {error}. It requires the Python package "
                      f"'{package}'.") from error
  return getattr(module, class_name)


def make_solver(solver_block, *, comm_size):
  """
  Construct the impurity solver described by a ``solver`` block.

  Parameters
  ----------
  solver_block : dict
      The ``solver`` table of one impurity. ``type`` is the registered name of the backend
      (``"ctseg"``, ``"ctint"``, ``"cthyb"``, or a name added with ``register_solver``). Every other key, the
      ``mu_search`` table included, is passed to the backend's constructor unchanged.
  comm_size : int
      Number of MPI ranks, for the backends that divide their Monte Carlo budget over ranks.

  Returns
  -------
  ImpuritySolver

  Raises
  ------
  KeyError
      No ``type`` in the block.
  ValueError
      A name that is not registered (the message lists the registered names), or a backend
      configured for a static interaction its capabilities do not cover (``h_int`` other than
      ``"density"`` without ``general_h_int``).
  ImportError
      A shipped backend whose TRIQS package cannot be imported (the message names it).
  TypeError
      A ``type`` that is not a string, or a shipped entry that is not an ``ImpuritySolver``
      subclass.
  """
  solver_params = dict(solver_block)
  if 'type' not in solver_params:
    raise KeyError("solver_api::make_solver: the solver block needs a 'type' "
                   f"(registered: {registered_solvers()})")
  kind = solver_params.pop('type')
  if not isinstance(kind, str):
    raise TypeError(f"solver_api::make_solver: 'type' must be the registered name of a backend, "
                    f"got {kind!r}")
  if kind in _REGISTERED_SOLVERS:
    cls = _REGISTERED_SOLVERS[kind]
  elif kind in _SHIPPED_SOLVERS:
    cls = _import_shipped_solver(kind)
  else:
    raise ValueError(f"solver_api::make_solver: unknown solver type {kind!r}; registered: "
                     f"{registered_solvers()} (register_solver adds a backend)")
  if not (inspect.isclass(cls) and issubclass(cls, ImpuritySolver)):
    raise TypeError(f"solver_api::make_solver: {kind!r} resolves to {cls!r}, not an "
                    f"ImpuritySolver subclass")
  solver = cls(solver_params, comm_size=comm_size)
  # the backend takes "density" always, the other static interactions with general_h_int
  h_int_ok = solver.h_int_kind == "density" or solver.capabilities.general_h_int
  if not h_int_ok:
    raise ValueError(f"solver_api::make_solver: h_int = {solver.h_int_kind!r} needs a backend "
                     f"with the general_h_int capability, which {kind!r} does not have")
  return solver

def check_solver_capabilities_for_gw_edmft(solver, imp_params, screen_type):
  """
  Check the solver's capabilities against the GW+EDMFT setup, and raise a ValueError
  if the backend cannot provide what the workflow requires.

  Parameters
  ----------
  solver : ImpuritySolver
      Instance of the impurity solver backend.
  imp_params : dict
      Impurity parameters: a ``chemical_potential`` entry requests the chemical-potential
      search, a ``mu_search`` table in its ``solver`` block configures the search's solves.
  screen_type : str
      Screening type in the GW+EDMFT workflow; anything but ``"rpa"`` needs the impurity
      polarization.

  Raises
  ------
  ValueError
      The search is requested or configured but the backend has no occupations-only solve, or
      the polarization is needed but the backend cannot return it for its ``h_int_kind``.
  """
  caps = solver.capabilities
  if not caps.occupations_only_solve:
    if imp_params.get('chemical_potential') is not None:
      raise ValueError(f"solver_api::check_solver_capabilities_for_gw_edmft: the chemical-potential search needs "
                       f"an occupations-only solve, which backend {solver.name!r} does not "
                       f"provide (capability occupations_only_solve)")
    if 'mu_search' in imp_params.get('solver', {}):
      raise ValueError(f"solver_api::check_solver_capabilities_for_gw_edmft: solver.mu_search configures the "
                       f"occupations-only solves of the chemical-potential search, which backend "
                       f"{solver.name!r} does not provide (capability occupations_only_solve)")
  polarization_unavailable = caps.polarization == 'never' or (
    caps.polarization == 'density_h_int_only' and solver.h_int_kind != 'density')
  if screen_type != 'rpa' and polarization_unavailable:
    raise ValueError(f"solver_api::check_solver_capabilities_for_gw_edmft: screen_type {screen_type!r} needs the "
                     f"impurity polarization, which backend {solver.name!r} cannot produce "
                     f"(capability polarization = {caps.polarization!r}, h_int = "
                     f"{solver.h_int_kind!r}); use screen_type = \"rpa\" or another backend")


def check_solver_capabilities_for_problem(solver, problem):
  """
  Check the solver's capabilities against the ``ImpurityProblem``. 

  Parameters
  ----------
  solver : ImpuritySolver
      Instance of the impurity solver backend.
  problem : ImpurityProblem
      The problem about to be solved.

  Raises
  ------
  ValueError
      A ``gf_struct`` block of size > 1 on a backend without ``offdiag_delta``, or a retarded
      interaction on a backend with ``retarded_interaction = "none"``.
  """
  if not solver.capabilities.offdiag_delta and any(size != 1 for _, size in problem.gf_struct):
    raise ValueError(f"solver_api::check_solver_capabilities_for_problem: blocks of size > 1 "
                     f"({problem.gf_struct}) need a backend with the offdiag_delta capability, "
                     f"which {solver.name!r} does not have")
  if solver.capabilities.retarded_interaction == "none" and np.abs(problem.u_weiss_wabcd).max() > 0.0:
    raise ValueError(f"solver_api::check_solver_capabilities_for_problem: the problem carries a "
                     f"retarded interaction but backend {solver.name!r} declares "
                     f"retarded_interaction = \"none\"")
