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
Utility functions for TRIQS interface to EDMFT.

This module provides helper routines that facilitate the conversion and
manipulation of data between CoQuí and the TRIQS ecosystem in the context
of EDMFT, including:

- Constructing TRIQS Green’s functions (Gf, BlockGf, Block2Gf) from raw data
  such as Weiss fields, hybridization functions, and interaction tensors.
- Handling mappings between orbital subspaces, solver structures, and block
  Green’s functions.
- Extracting and reducing interaction tensors (e.g. to Hubbard–Kanamori form).
- Enforcing density–density or real-Hamiltonian approximations as required by
  TRIQS solvers (e.g. CT-SEG).
- Miscellaneous utilities for mesh handling, and TRIQS container construction.

Dependencies
------------
This module relies on the TRIQS ecosystem, in particular:
- `triqs.gfs` for Green’s function containers and operations
- `triqs.operators` for operator algebra
- `triqs.utility.mpi` for parallelism
"""
import numpy as np
from itertools import product

from triqs.gfs import (
    inverse, iOmega_n, Gf, make_gf_dlr, BlockGf, Block2Gf, 
    MeshImFreq, MeshDLRImFreq, 
)
from triqs.operators import c_dag, c, Operator, util
from triqs.operators.util.extractors import block_matrix_from_op, extract_U_dict2, dict_to_matrix
import coqui.dmft as coqui_dmft


def get_c_to_solver_mapping(solver_struct):
    # Ex: 
    # C space = [ ('up', nbnd), ('down', nbnd) ]
    # solver  = [ ('up_0', a), ('up_1', b), ('down_0', a), ('down_1', b) ]
    #             where a + b = nbnd
    c_to_solver = {}
    o1_up, o1_dn = 0, 0
    for blk_name, blk_dim in solver_struct:
        if blk_name[:2] == "up":
            for i in range(blk_dim):
                c_to_solver[("up", i+o1_up)] = (blk_name, i)
            o1_up += blk_dim
        else:
            for i in range(blk_dim):
                c_to_solver[("down", i+o1_dn)] = (blk_name, i)
            o1_dn += blk_dim

    return c_to_solver


def get_spin_and_orbital_index(block_name, gf_struct_list, imp_index):
    """
    Given a block name (e.g., 'up_3') and a gf_struct (list of (name, size)),
    return:
        spin : int (0 for up, 1 for down)
        orbital_range : tuple(start, stop)  # indices in flattened orbital basis
    """
    # Parse spin and block index from the name
    try:
        prefix, idx_str = block_name.split('_')
    except ValueError:
        raise ValueError(f"Invalid block name '{block_name}'; must look like 'up_0', 'down_3', etc.")

    spin_prefix = prefix.lower()
    if spin_prefix == "up" or spin_prefix == "ud":
        spin = 0
    elif spin_prefix == "down":
        spin = 1
    else:
        raise ValueError(f"Unrecognized spin prefix '{prefix}' in '{block_name}'.")

    # 1) Compute same-spin offset from all previous gf_structs
    offset = 0
    for k in range(imp_index):
        for name, size in gf_struct_list[k]:
            if name.lower().startswith(spin_prefix):
                offset += int(size)

    # 2) Find local (start, stop) within the target gf_struct, counting only same-spin blocks
    gf_struct = gf_struct_list[imp_index]
    start_local = 0
    found = False
    for name, size in gf_struct:
        if name.startswith(spin_prefix + '_'):
            if name == block_name:
                found = True
                stop_local = start_local + size
                break
            start_local += size

    if not found:
        raise KeyError(f"Block name '{block_name}' not found in gf_struct.")

    # 3) Apply offset
    start = offset + start_local
    stop  = offset + stop_local

    return spin, (start, stop)


def v_pb_to_triqs_notation(v_pb_ijkl):
    # v_pb_ijkl <-> c^dag_j c^dag_k c_l c_i
    # v_triqs_jkil  <-> c^dag_j c^dag_k c_l c_i
    v_triqs_jkil = v_pb_ijkl.transpose(1, 2, 0, 3).copy()
    return v_triqs_jkil


def h_int_density_density(v_abcd_pb, gf_struct, force_real=True):
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    V, Vp = util.reduce_4index_to_2index(
        v_pb_to_triqs_notation(v_abcd_pb.real if force_real else v_abcd_pb)
    )
    h_int = util.h_int_density(['up', 'down'], V.shape[0], U=V, Uprime=Vp, 
                               map_operator_structure=c_to_solver)
    return h_int


def h_int_slater(v_abcd_pb, gf_struct, force_real=True):
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    return util.h_int_slater(['up', 'down'], v_abcd_pb.shape[-1],
                              v_pb_to_triqs_notation(v_abcd_pb.real if force_real else v_abcd_pb),
                              map_operator_structure=c_to_solver)
    

def h0_operator(h0_sab, gf_struct, *, force_real=True):
    h0_blk_mat = arr_to_blk_arr(h0_sab, gf_struct)
    return blk_h0_to_h0_operator(h0_blk_mat, gf_struct, force_real)


def blk_h0_to_h0_operator(blk_h0, gf_struct, force_real=True):
    h_loc0_mat = {block : blk_h0[ibl].real if force_real else blk_h0[ibl] for ibl, (block, _) in enumerate(gf_struct) }
    c_dag_vec  = {block : np.matrix([[c_dag(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    c_vec      = {block : np.matrix([[c(block, o) for o in range(bl_size)]]) for block, bl_size in gf_struct }
    return sum(c_dag_vec[block]*h_loc0_mat[block]*c_vec[block].T for block, bl_size in gf_struct)[0,0]


def h0_operator_to_array(h0_op, gf_struct):
    return blk_arr_to_arr(block_matrix_from_op(h0_op, gf_struct), gf_struct)


def to_block_gf(giw_ir, iaft, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Fermion', "Only mesh_iw.statistic == Fermion is supported."
    assert len(giw_ir.shape) == 4, "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[2] == giw_ir.shape[3], "giw_ir needs to have dimensions (nw, nspins, nbnd, nbnd)"
    assert giw_ir.shape[0] == iaft.nw_f, "giw_ir.shape[0] != iaft.nw_f"
    
    # giw_ir = (nw, nspin, nbnd, nbnd)
    nbnd = giw_ir.shape[-1]

    blk_gf = BlockGf(mesh=mesh_iw, gf_struct=gf_struct)
    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])

    # E.g. block structure = [ ("up_0", 1), ("down_0", 1), ("up_1", 2), ("down_1", 2) ]
    offset = [0, 0]
    for blk_name, blk_dim in gf_struct:
        s = 0 if blk_name[:2] == "up" else 1
        blk_gf[blk_name].data[:] = iaft.w_interpolate(
            giw_ir[:, s, offset[s]:offset[s]+blk_dim, offset[s]:offset[s]+blk_dim],
            mesh_iw_idx,
            stats="f",
            phys_notation=True
        )
        offset[s] += blk_dim
        assert offset[s] <= nbnd, f"Spin {s} block exceeds band range"

    return blk_gf


def to_block2_gf(Diw_ir, iaft, gf_struct, mesh_iw):

    assert mesh_iw.statistic == 'Boson', "Only mesh_iw.statistic == Boson is supported."
    assert len(Diw_ir.shape) == 5, \
        "Diw_ir needs to have dimensions (nw, nspin, nspin, nbnd, nbnd)"
    assert Diw_ir.shape[1] == Diw_ir.shape[2], \
        "Diw_ir needs to have dimensions (nw, nspin, nspin, nbnd, nbnd)"
    assert Diw_ir.shape[-1] == Diw_ir.shape[-2], \
        "Diw_ir needs to have dimensions (nw, nspin, nspin, nbnd, nbnd)"
    nw_half = iaft.nw_b//2 if iaft.nw_b%2==0 else iaft.nw_b//2 + 1
    assert Diw_ir.shape[0] == nw_half, "Diw_ir.shape[0] != nw_b_half"

    # Diw_ir = (nw, nspin, nspin, nbnd, nbnd)
    nbnd = Diw_ir.shape[-1]

    mesh_iw_idx = np.array([iwn.index for iwn in mesh_iw])
    Diw_data = iaft.w_interpolate_phsym(Diw_ir, mesh_iw_idx, stats="b", phys_notation=True)

    gf_array = []
    o1 = [0, 0]
    for name1, dim1 in gf_struct:
        s1 = 0 if name1[:2] == "up" else 1

        gf_list = []
        o2 = [0, 0]
        for name2, dim2 in gf_struct:
            s2 = 0 if name2[:2] == "up" else 1

            gf = Gf(mesh = mesh_iw, target_shape = (dim1, dim2))
            gf.data[:] = Diw_data[:, s1, s2, o1[s1]:o1[s1]+dim1, o2[s2]:o2[s2]+dim2]

            o2[s2] += dim2
            assert o2[s2] <= nbnd, f"Spin {s2} block exceeds band range"

            gf_list.append(gf)

        o1[s1] += dim1
        assert o1[s1] <= nbnd, f"Spin {s1} block exceeds band range"

        gf_array.append(gf_list)

    names = [name for name, _ in gf_struct]
    return Block2Gf(names, names, gf_array)


def to_triqs_containers(h0, delta_iw, Vimp, u_weiss_iw, iaft, 
                        gf_struct, triqs_iw_mesh, 
                        density_hamiltonian, real_hamiltonian=True,
                        screen_j_in_u_dd=False):
    """
    Convert raw CoQui outputs (NumPy arrays) into TRIQS containers 
    (e.g. `triqs.operators.many_body_operator`, `BlockGf`, and `Block2Gf`).

    This function provides a bridge between CoQui’s raw many-body data and 
    TRIQS’ Green’s function representation. It constructs the one-particle and 
    two-particle objects required for TRIQS-based impurity solvers or DMFT-like 
    workflows.

    Parameters
    ----------
    h0 : np.ndarray
        One-particle Hamiltonian matrix in the impurity subspace.
    delta_iw : np.ndarray
        Hybridization function Δ(iωₙ) on the Matsubara frequency axis, in raw array form.
    Vimp : np.ndarray
        Bare Coulomb interaction tensor in the impurity basis.
    u_weiss_iw : np.ndarray
        Dynamical screened interaction U(iωₙ) from cRPA or EDMFT preprocessing, 
        given as raw arrays on the Matsubara axis.
    iaft : object
        Imaginary-time/frequency transform kernel (e.g. IR/IAFT object) used for 
        Fourier transforms between τ and iωₙ.
    gf_struct : dict
        Block structure of Green’s functions, mapping orbital/spin indices to block labels.
    triqs_iw_mesh : dict
        Dictionary of TRIQS mesh objects:
        - `"fermion"` : Matsubara mesh for fermionic objects (Δ, G).
        - `"boson"`   : Matsubara mesh for bosonic objects (U).
    density_hamiltonian : bool
        Whether to use the density-density approximation for the Coulomb interaction. 
        This will also enforce the non-interacting Hamiltonian to density operator only. 
        Currently must be `True`. Non-density-density interactions are not implemented.
    real_hamiltonian : bool, optional
        If `True`, enforce the bare Hamiltonian (h0, Vimp) to be real-valued. 
        Default is `True`.
    screen_j_in_u_dd : bool, optional
        If `True`, modify the same spin density-density dynamic screening with screening J.

    Returns
    -------
    h0 : triqs.operators.many_body_operator
        One-particle Hamiltonian in TRIQS operator form.
    delta_iw : triqs.gfs.BlockGf
        Hybridization function Δ(iωₙ) as a TRIQS block Green’s function.
    h_int : triqs.operators.many_body_operator
        Local interaction Hamiltonian in density-density approximation.
    u_weiss_iw : triqs.gfs.Block2Gf
        Dynamical screened interaction U(iωₙ) as a TRIQS two-particle block Green’s function.

    Notes
    -----
    - This conversion assumes the density-density approximation for Coulomb 
      interactions; a full four-index mapping is not yet implemented.
    """

    assert density_hamiltonian, (
        "Convertion to non-density-density Hamiltonian is not implemented yet."
    )

    h0 = h0_operator(h0, gf_struct, force_real=real_hamiltonian)
    h_int = h_int_density_density(Vimp, gf_struct, force_real=real_hamiltonian)

    if "fermion" not in triqs_iw_mesh:
        triqs_iw_mesh['fermion'] = MeshDLRImFreq(
            beta=iaft.beta, statistic='Fermion',
            w_max=triqs_iw_mesh['dlr_wmax'], eps=triqs_iw_mesh['dlr_eps'], symmetrize=True
        )
    if "boson" not in triqs_iw_mesh:
        triqs_iw_mesh['boson'] = MeshDLRImFreq(
            beta=iaft.beta, statistic='Boson',
            w_max=triqs_iw_mesh['dlr_wmax'], eps=triqs_iw_mesh['dlr_eps'], symmetrize=True
        )


    if real_hamiltonian:
        # FT to tau space and enforce to real values
        delta_tau = iaft.w_to_tau(delta_iw, 'f')
        delta_tau.imag = 0.0
        delta_iw_gf = to_block_gf(iaft.tau_to_w(delta_tau, 'f'), iaft,
                                  gf_struct, triqs_iw_mesh["fermion"])
    else:
        delta_iw_gf = to_block_gf(delta_iw, iaft,
                                  gf_struct, triqs_iw_mesh["fermion"])

    # u_iw_s1s2 has dimension of (niw, nspin, nspin, nbnd, nbnd)
    u_iw_s1s2 = coqui_dmft.dynamic_4idx_u_to_dd_basis(
        u_weiss_iw, nspin=2, screen_j=screen_j_in_u_dd
    )

    u_weiss_iw_gf = to_block2_gf(
        u_iw_s1s2.real if real_hamiltonian else u_iw_s1s2,
        iaft, gf_struct, triqs_iw_mesh["boson"]
    )

    return h0, delta_iw_gf, h_int, u_weiss_iw_gf


def gf_dlr_from_ir(giw_ir, iaft, mesh_dlr_iw):
    
    stats = 'f' if mesh_dlr_iw.statistic == 'Fermion' else 'b'
    target_shape = giw_ir.shape[1:]

    if stats == 'b':
        giw_ir_pos = giw_ir.copy()
        giw_ir = np.zeros([giw_ir_pos.shape[0] * 2 - 1] + target_shape, dtype=complex)
        giw_ir[: giw_ir_pos.shape[0]] = giw_ir_pos[::-1]
        giw_ir[giw_ir_pos.shape[0] :] = giw_ir_pos[1:]
    
    gf_dlr_iw = Gf(mesh=mesh_dlr_iw, target_shape=target_shape)

    # prepare idx array for spare ir
    if stats == 'f':
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])
    else:
        mesh_dlr_iw_idx = np.array([iwn.index for iwn in mesh_dlr_iw])

    gf_dlr_iw.data[:] = iaft.w_interpolate(giw_ir, mesh_dlr_iw_idx, stats=stats, phys_notation=True)

    return make_gf_dlr(gf_dlr_iw)


def gf_dlr_to_ir(gf_dlr, iaft):
    stats = 'b' if gf_dlr.mesh.statistic == 'Boson' else 'f'
    ir_idx = iaft.wn_mesh(stats=stats, phys_notation=True)
    nw = len(ir_idx)
    iw_mesh_uniform = MeshImFreq(
        beta=gf_dlr.mesh.beta,
        statistic=gf_dlr.mesh.statistic,
        n_iw=ir_idx[-1]+10
    )
    def fill_gf_ir(gf_input):
        gf_ir_out = np.zeros((nw,) + gf_input.data[:].shape[1:], dtype=complex)
        # we don't symmetrize gf here. Call `make_hermitian` externally if needed. 
        for i, idx in enumerate(ir_idx):
            gf_ir_out[i] = gf_input(iw_mesh_uniform(idx))

        return gf_ir_out

    if isinstance(gf_dlr, BlockGf):

        gf_iw_data = []
        for blk_name, gf in gf_dlr:
            gf_iw_data.append(fill_gf_ir(gf))

        return gf_iw_data

    elif isinstance(gf_dlr, Gf):

        return fill_gf_ir(gf_dlr)

    else:
        raise ValueError(f"gf_dlr_to_ir: Invalid type of gf_dlr")


def gf_dlr_to_ir_phsym(gf_dlr, iaft):
    assert gf_dlr.mesh.statistic == "Boson", (
        "gf_dlr_to_ir_phsym: Gf statistics must be Boson"
    )
    ir_idx_b = iaft.wn_mesh(stats='b', phys_notation=True, positive_only=True)
    nw_b_pos = len(ir_idx_b)
    iw_mesh_uniform_b = MeshImFreq(
        beta=gf_dlr.mesh.beta,
        statistic=gf_dlr.mesh.statistic,
        n_iw=ir_idx_b[-1]+10
    )
    gf_iw_data = np.zeros((nw_b_pos,) + gf_dlr.data[:].shape[1:], dtype=complex)
    for i, idx in enumerate(ir_idx_b):
        gf_iw_data[i] = gf_dlr(iw_mesh_uniform_b(idx))

    return gf_iw_data


def blk_gf_to_blk_arr(block_gf):
    blk_array = []
    for blk_name, gf in block_gf:
        blk_array.append(gf.data[:])
    return blk_array


def blk_gf_to_arr(block_gf, gf_struct):
    blk_array = blk_gf_to_blk_arr(block_gf)
    return blk_arr_to_arr(blk_array, gf_struct)


def blk_gf_to_gf(blk_gf, gf_struct):
    arr = blk_gf_to_arr(blk_gf, gf_struct)
    target_shape = arr.shape[1:]
    gf = Gf(mesh = blk_gf.mesh, target_shape=target_shape)
    gf.data[:] = arr
    return gf


def blk_arr_to_arr(blk_array, gf_struct):
    """
    Convert a list of spin/orbital block arrays into a single dense array.

    Parameters
    ----------
    blk_array : list of ndarray
        List of block data arrays. Each block corresponds to one entry in `gf_struct`,
        and all blocks must have the same leading dimensions (e.g. frequency, k-points).
        The last two dimensions of each block are the orbital indices for that block.
    gf_struct : list of (str, int)
        Structure definition for the blocks. Each element is a tuple `(block_name, dim)`,
        where `block_name` is a string like `"up_0"`, `"down_1"`, and `dim` is the orbital
        dimension of that block.

    Returns
    -------
    array : ndarray
        Combined dense array with shape
        ``arr_shape[:-2] + (nspin, n_orb, n_orb)``,
        where `arr_shape` is the shape of the first element of `blk_array`.
        The extra dimension `nspin` is the number of spin blocks (e.g. 2 for "up"/"down"),
        and `n_orb` is the total number of orbitals per spin.
        Each block from `blk_array` is placed into the correct slice of the full array
        according to `gf_struct`.

    Notes
    -----
    - Spin blocks are identified by the prefix of `block_name` before the underscore
      (e.g. `"up"` in `"up_0"`).
    - Orbital sub-blocks are placed contiguously along the orbital axes, with offsets
      determined by the cumulative dimensions of the blocks.
    - Assumes each spin has the same total orbital count; raises if not.
    """

    if blk_array is None:
        return None

    assert len(blk_array) == len(gf_struct),  (
        f"Inconistent number of blocks between block_array ({len(blk_array)}) "
        f"and gf_struct ({len(gf_struct)})."
    )

    spin_blk = []
    for name, _ in gf_struct:
        spin = name.split('_', 1)[0]
        if spin not in spin_blk:
            spin_blk.append(spin)
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}
    nspin = len(spin_blk)

    # Compute per-spin orbital totals and validate they match
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    counts = set(per_spin_counts.values())
    assert len(counts) == 1, (
        "Per-spin orbital totals must match across spins; got "
        f"{per_spin_counts}"
    )
    n_orb = counts.pop()

    # Infer leading shape/dtype from first block
    arr_shape = blk_array[0].shape
    leading_shape = arr_shape[:-2]

    # Allocate dense array
    array = np.zeros(leading_shape+(nspin, n_orb, n_orb), dtype=blk_array[0].dtype)
    
    # Fill with block data
    offsets = [0] * nspin
    for (blk_name, dim), blk_data in zip(gf_struct, blk_array):
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        assert blk_data.shape[:-2] == leading_shape, (
            f"Leading dims mismatch for block {blk_name}: "
            f"{blk_data.shape[:-2]} vs {leading_shape}"
        )

        start, stop = offsets[s], offsets[s]+dim
        array[..., s, start:stop, start:stop] = blk_data
        offsets[s] += dim

    return array


def arr_to_blk_arr(array, gf_struct):
    """
    Convert a dense array into a list of spin/orbital block arrays per `gf_struct`.

    Parameters
    ----------
    array : ndarray
        Dense data with shape: leading_shape + (nspin, n_orb, n_orb).
    gf_struct : list[tuple[str, int]]
        List of (block_name, dim), e.g. [("up_0", d0), ("up_1", d1), ("down_0", d2), ...].
        The spin label is the prefix before the first underscore.

    Returns
    -------
    blk_array : list of ndarray
        Blocks in the same order as `gf_struct`. Each has shape: leading_shape + (dim, dim).

    Raises
    ------
    AssertionError
        If shapes are inconsistent with `gf_struct` or per-spin orbital totals
        do not match the dense array’s orbital size.
    """
    assert array.ndim >= 3, "array must have at least 3 dimensions (…, nspin, n_orb, n_orb)"
    leading_shape = array.shape[:-3]
    nspin, n_orb_0, n_orb_1 = array.shape[-3:]
    assert n_orb_0 == n_orb_1, f"Last two dims must be square; got {(n_orb_0, n_orb_1)}"

    # Preserve spin order as first seen in gf_struct
    spin_blk = []
    for name, _ in gf_struct:
        s = name.split('_', 1)[0]
        if s not in spin_blk:
            spin_blk.append(s)
    assert len(spin_blk) == nspin, (
        f"Spin count mismatch: array has {nspin}, gf_struct implies {len(spin_blk)}"
    )
    spin_to_idx = {s: i for i, s in enumerate(spin_blk)}

    # Check per-spin total orbitals vs array
    per_spin_counts = {s: 0 for s in spin_blk}
    for name, dim in gf_struct:
        s = name.split('_', 1)[0]
        per_spin_counts[s] += dim
    totals = set(per_spin_counts.values())
    assert len(totals) == 1 and totals.pop() == n_orb_0, (
        "Per-spin orbital totals must match the dense array's orbital size; "
        f"got per-spin {per_spin_counts}, array n_orb={n_orb_0}"
    )

    # Extract blocks
    offsets = [0] * nspin
    blk_array = []
    for blk_name, dim in gf_struct:
        s_name = blk_name.split('_', 1)[0]
        s = spin_to_idx[s_name]

        start, stop = offsets[s], offsets[s]+dim
        assert stop <= n_orb_0, (
            f"Block {blk_name} (dim={dim}) exceeds spin-orbital range "
            f"[{start}:{stop}) with n_orb={n_orb_0}"
        )

        blk_array.append(array[..., s, start:stop, start:stop].copy())
        offsets[s] += dim

    for s_idx, off in enumerate(offsets):
        assert off == n_orb_0, (
            f"Unused orbital slots remain for spin index {s_idx}: "
            f"filled {off} / {n_orb_0}"
        )

    return blk_array


# deprecated - it is implemented in modest now
def extract_ij(sigma_infty_embed, embedding):
    gf_struct_list = [embedding.imp_block_shape[imp] for imp in range(embedding.n_impurities)]
    sigma_infty_list = []
    for imp_idx in range(len(gf_struct_list)):
        imp_out = []
        for blk_name, blk_dim in gf_struct_list[imp_idx]:
            spin, orb_range = get_spin_and_orbital_index(blk_name, gf_struct_list, imp_idx)
            imp_out.append(sigma_infty_embed[spin][orb_range[0]:orb_range[1], orb_range[0]:orb_range[1]])
        sigma_infty_list.append(imp_out)

    return sigma_infty_list


# deprecated - it is implemented in modest now
def extract_wij(sigma_iw_embed, embedding):
    gf_struct_list = [embedding.imp_block_shape[imp] for imp in range(embedding.n_impurities)]
    sigma_iw_list = []
    for imp_idx in range(len(gf_struct_list)):
        imp_out = []
        for blk_name, blk_dim in gf_struct_list[imp_idx]:
            spin, orb_range = get_spin_and_orbital_index(blk_name, gf_struct_list, imp_idx)
            imp_out.append(sigma_iw_embed[spin][:, orb_range[0]:orb_range[1], orb_range[0]:orb_range[1]])
        sigma_iw_list.append(imp_out)

    return sigma_iw_list


def imp_results_to_raw_data(g_iw, sigma_iw, w_iw, pi_iw, iaft=None):
    if isinstance(sigma_iw.mesh, MeshDLRImFreq) and isinstance(pi_iw.mesh, MeshDLRImFreq):
        return _dlr_imp_results_to_raw_data(g_iw, sigma_iw, w_iw, pi_iw, iaft)
    elif isinstance(sigma_iw.mesh, MeshImFreq) and isinstance(pi_iw.mesh, MeshImFreq):
        raise ValueError("imp_results_to_raw_data for uniform grid is disable for now")
        #return _full_mesh_imp_results_to_raw_data(g_iw, sigma_iw, w_iw, pi_iw, iaft)
    else:
        raise ValueError("Incompatible mesh types for sigma_iw and pi_iw.")


def _dlr_imp_results_to_raw_data(g_iw, sigma_iw, w_iw, pi_iw, iaft=None):
    """
    Convert impurity self-energy and polarization Green's functions from TRIQS objects
    to raw NumPy array representations, optionally projected onto an intermediate
    representation (IR) Matsubara frequency mesh.

    Parameters
    ----------
    g_iw     : BlockGf on MeshDLRImFreq mesh
    sigma_iw : BlockGf on MeshDLRImFreq mesh
    w_iw     : Gf on MeshDLRImFreq mesh
    pi_iw    : Gf on MeshDLRImFreq mesh
    iaft : optional
        IR kernel to get the IR Matsubara frequency meshes.
        If provided, results will be interpolated on these IR grids.
        If None, the data on the original full Matsubara mesh are returned.

    Returns
    -------
    dict
        A dictionary containing:
        - ``"G_iw_data"``: list of ndarray
            Block array representation of the impurity Green's function.
        - ``"Sigma_iw_data"``: list of ndarray
            Block array representation of the impurity self-energy.
        - ``"Pi_iw_data"``: list of ndarray
            Block array representation of the impurity polarizability.
        - ``"W_iw_data"``: list of ndarray
            Block array representation of the impurity screened interaction.
    """
    if iaft is None:
        g_iw_data = coqui_dmft.blk_gf_to_blk_arr(g_iw)
        sigma_iw_data = coqui_dmft.blk_gf_to_blk_arr(sigma_iw)
        pi_iw_data = [pi_iw.data[:]]
        w_iw_data = [w_iw.data[:]]
        return {"G_iw_data": g_iw_data, "Sigma_iw_data": sigma_iw_data,
                "W_iw_data": w_iw_data, "Pi_iw_data": pi_iw_data}

    # converter Sigma and Pi to IR Matsubara mesh
    g_iw_data     = coqui_dmft.make_hermitian(
        coqui_dmft.gf_dlr_to_ir(make_gf_dlr(g_iw), iaft)
    )
    sigma_iw_data = coqui_dmft.make_hermitian(
        coqui_dmft.gf_dlr_to_ir(make_gf_dlr(sigma_iw), iaft)
    )
    pi_iw_data = [coqui_dmft.gf_dlr_to_ir_phsym(make_gf_dlr(pi_iw), iaft)]
    w_iw_data = [coqui_dmft.gf_dlr_to_ir_phsym(make_gf_dlr(w_iw), iaft)]
    return {"G_iw_data": g_iw_data, "Sigma_iw_data": sigma_iw_data,
            "W_iw_data": w_iw_data, "Pi_iw_data": pi_iw_data}


def _full_mesh_imp_results_to_raw_data(g_iw, sigma_iw, w_iw, pi_iw, iaft=None):
    """
    Convert impurity self-energy and polarization Green's functions from TRIQS objects
    to raw NumPy array representations, optionally projected onto an intermediate
    representation (IR) Matsubara frequency mesh.

    Parameters
    ----------
    sigma_iw : BlockGf on MeshImFreq mesh
    pi_iw : Gf on MeshImFreq mesh
    iaft : optional
        IR kernel object providing methods to obtain the IR Matsubara frequency meshes.
        If provided, both `sigma_iw` and `pi_iw` will be interpolated on these IR grids.
        If None, the data on the original full Matsubara mesh are returned.

    Returns
    -------
    dict
        A dictionary containing:
        - ``"Sigma_iw_data"`` : list of ndarray
            Block array representation of the impurity self-energy.
        - ``"Pi_iw_data"`` : list of ndarray
            Block array representation of the impurity polarizability.
    """
    # solver_res.Sigma_iw (TRIQS BlockGf) -> solver_res.Sigma_iw_data (block array)
    sigma_iw_data = coqui_dmft.blk_gf_to_blk_arr(sigma_iw)
    # solver_res.Pi_iw (TRIQS Gf) -> solver_res.Pi_iw_data (block array)
    pi_iw_data = [pi_iw.data[:]]

    if iaft is None:
        return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}

    # converter Sigma and Pi to IR Matsubara mesh
    ir_idx_f = iaft.wn_mesh(stats='f', phys_notation=True)
    nw_f = len(ir_idx_f)
    nw_f_half = nw_f // 2
    for i, sigma_dyn in enumerate(sigma_iw_data):
        sigma_dyn_ir = np.zeros((nw_f,) + sigma_dyn.shape[1:], dtype=complex)
        for idx in range(nw_f_half):
            iw_pos = nw_f_half + idx
            iw_neg = nw_f_half - idx - 1
            data_idx = sigma_iw.mesh.to_data_index(ir_idx_f[iw_pos])
            sigma_dyn_ir[iw_pos] = sigma_dyn[data_idx]
            sigma_dyn_ir[iw_neg] = sigma_dyn[data_idx].conj()
        sigma_iw_data[i] = sigma_dyn_ir

    # interpolate solver_res.Pi_iw to ir grid
    ir_idx_b = iaft.wn_mesh(stats='b', phys_notation=True, positive_only=True)
    nw_b_pos = len(ir_idx_b)
    pi_iw_ir = np.zeros((nw_b_pos,) + pi_iw_data[0].shape[1:], dtype=complex)
    for idx in range(nw_b_pos):
        data_idx = pi_iw.mesh.to_data_index(ir_idx_b[idx])
        pi_iw_ir[idx] = pi_iw_data[0][data_idx]
    pi_iw_data[0] = pi_iw_ir

    return {"Sigma_iw_data": sigma_iw_data, "Pi_iw_data": pi_iw_data}


def symmetrize_blk_mat(blk_mat, deg_blk):
    blk_mat_sym = blk_mat.copy()
    for i, blks in enumerate(deg_blk):
        blk_avg = np.sum(blk_mat[blks]) / len(blks)
        for b in blks:
            blk_mat_sym[b] = blk_avg
    return blk_mat_sym


def symmetrize_h0_op(h0_op, deg_blk, gf_struct):
    h0_blk_mat = block_matrix_from_op(h0_op, gf_struct)
    h0_blk_sym = symmetrize_blk_mat(h0_blk_mat, deg_blk)
    return blk_h0_to_h0_operator(h0_blk_sym, gf_struct)


def symmetrize_h_int_op(h_int_op, deg_blk, gf_struct):
    # extract the density-density Coulomb matrix U_dd
    u_dd = dict_to_matrix(extract_U_dict2(h_int_op), gf_struct=gf_struct)
    n_orb = u_dd.shape[0]//2

    for i, blks in enumerate(deg_blk):

        u_diag_buffer, u_offdiag_buffer, up_diag_buffer, up_offdiag_buffer = 0.0, 0.0, 0.0, 0.0
        diag_count, offdiag_count = 0, 0
        diag_count_prime, offdiag_count_prime = 0, 0

        # different treatment for same-spin and opposite-spin in case they are different
        for b1, b2 in product(blks, repeat=2):
            blk_name1, blk_name2 = gf_struct[b1][0], gf_struct[b2][0]
            spin1, orb_blk1 = blk_name1.split('_')
            spin2, orb_blk2 = blk_name2.split('_')

            # same spin
            if spin1 == spin2:
                if orb_blk1 == orb_blk2:
                    u_diag_buffer += u_dd[b1, b2]
                    diag_count += 1
                else:
                    u_offdiag_buffer += u_dd[b1, b2]
                    offdiag_count += 1
            else:
                if orb_blk1 == orb_blk2:
                    up_diag_buffer += u_dd[b1, b2]
                    diag_count_prime += 1
                else:
                    up_offdiag_buffer += u_dd[b1, b2]
                    offdiag_count_prime += 1

        for b1, b2 in product(blks, repeat=2):
            blk_name1, blk_name2 = gf_struct[b1][0], gf_struct[b2][0]
            spin1, orb_blk1 = blk_name1.split('_')
            spin2, orb_blk2 = blk_name2.split('_')

            if spin1 == spin2:
                if orb_blk1 == orb_blk2 and diag_count > 0:
                    u_dd[b1, b2] = u_diag_buffer / diag_count
                elif orb_blk1 != orb_blk2 and offdiag_count > 0:
                    u_dd[b1, b2] = u_offdiag_buffer / offdiag_count
            else:
                if orb_blk1 == orb_blk2 and diag_count_prime > 0:
                    u_dd[b1, b2] = up_diag_buffer / diag_count_prime
                elif orb_blk1 != orb_blk2 and offdiag_count_prime > 0:
                    u_dd[b1, b2] = up_offdiag_buffer / offdiag_count_prime
    
    c_to_solver = get_c_to_solver_mapping(gf_struct)
    U_same_spin = u_dd[:n_orb, :n_orb]
    U_opposite_spin = u_dd[:n_orb, n_orb:]
    h_int = util.h_int_density(['up', 'down'], n_orb, U=U_same_spin, Uprime=U_opposite_spin, 
                               map_operator_structure=c_to_solver)
    return h_int


def symmetrize_blk2_gf(u_iw_blk_gf2, deg_blk, gf_struct):
    u_iw_sym = u_iw_blk_gf2.copy()
    # 'up_0', 'up_1', 'dn_0', 'dn_1'
    for i, blks in enumerate(deg_blk):

        # same-spin, same-orbital blocks
        u_diag_buffer = u_iw_blk_gf2[gf_struct[blks[0]][0], gf_struct[blks[0]][0]].copy()
        u_diag_buffer << 0.0
        # opposite-spin, same-orbital blocks
        up_diag_buffer = u_diag_buffer.copy()
        # same-spin, off-diagonal orbital blocks
        u_offdiag_buffer = u_diag_buffer.copy()
        # opposite-spin, off-diagonal orbital blocks
        up_offdiag_buffer = u_diag_buffer.copy()
        diag_count, offdiag_count = 0, 0
        diag_count_prime, offdiag_count_prime = 0, 0

        # different treatment for same-spin and opposite-spin in case they are different
        for b1, b2 in product(blks, repeat=2):
            blk_name1, blk_name2 = gf_struct[b1][0], gf_struct[b2][0]
            spin1, orb_blk1 = blk_name1.split('_')
            spin2, orb_blk2 = blk_name2.split('_')

            # same spin
            if spin1 == spin2:
                if orb_blk1 == orb_blk2:
                    u_diag_buffer += u_iw_blk_gf2[blk_name1, blk_name2]
                    diag_count += 1
                else:
                    u_offdiag_buffer += u_iw_blk_gf2[blk_name1, blk_name2]
                    offdiag_count += 1
            else:
                if orb_blk1 == orb_blk2:
                    up_diag_buffer += u_iw_blk_gf2[blk_name1, blk_name2]
                    diag_count_prime += 1
                else:
                    up_offdiag_buffer += u_iw_blk_gf2[blk_name1, blk_name2]
                    offdiag_count_prime += 1

        u_diag_buffer /= diag_count
        up_diag_buffer /= diag_count_prime
        u_offdiag_buffer /= offdiag_count
        up_offdiag_buffer /= offdiag_count_prime

        for b1, b2 in product(blks, repeat=2):
            blk_name1, blk_name2 = gf_struct[b1][0], gf_struct[b2][0]
            spin1, orb_blk1 = blk_name1.split('_')
            spin2, orb_blk2 = blk_name2.split('_')

            if spin1 == spin2:
                if orb_blk1 == orb_blk2:
                    u_iw_sym[blk_name1, blk_name2] << u_diag_buffer
                else:
                    u_iw_sym[blk_name1, blk_name2] << u_offdiag_buffer
            else:
                if orb_blk1 == orb_blk2:
                    u_iw_sym[blk_name1, blk_name2] << up_diag_buffer
                else:
                    u_iw_sym[blk_name1, blk_name2] << up_offdiag_buffer

    return u_iw_sym
