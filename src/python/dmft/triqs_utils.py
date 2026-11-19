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
from dataclasses import dataclass
from itertools import product

from triqs.gfs import (
    inverse, iOmega_n, Gf, make_gf_dlr, BlockGf, Block2Gf, 
    MeshImFreq, MeshDLRImFreq, 
)
from triqs.operators import c_dag, c, Operator, util
from triqs.operators.util.extractors import block_matrix_from_op, extract_U_dict2, dict_to_matrix
import triqs_modest as modest
import coqui.dmft as coqui_dmft
from coqui import app_log
from coqui.dmft.qmc_postproc import eval_on_points


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


def delta_block_gf(delta_wsab, iaft, gf_struct, mesh_f, *, force_real=True):
    """
    Convert the hybridization function Delta(iw) to a TRIQS ``BlockGf``.

    Parameters
    ----------
    delta_wsab : np.ndarray
        Delta(iw) of shape ``(niw, nspin, norb, norb)`` on the fermionic sampling frequencies
        of ``iaft``.
    iaft : IAFT
    gf_struct : list of (str, int)
        The block structure, one ``(name, size)`` per block, e.g. ``[("up_0", 1), ("down_0", 1)]``.
    mesh_f : triqs.gf.MeshDLRImFreq
        The fermionic Matsubara mesh of the result.
    force_real : bool, optional
        Drop the imaginary part on the tau axis. Default ``True``.

    Returns
    -------
    triqs.gf.BlockGf
        Delta(iw) on ``mesh_f``, one block per entry of ``gf_struct``.
    """
    if force_real:
        delta_tau = iaft.w_to_tau(delta_wsab, 'f')
        delta_tau.imag = 0.0
        delta_wsab = iaft.tau_to_w(delta_tau, 'f')
    return to_block_gf(delta_wsab, iaft, gf_struct, mesh_f)


def h_int_operator(v_abcd, gf_struct, *, kind="density", force_real=True):
    """
    Build the static interaction operator of an impurity from 4-index Coulomb tensor.

    Currently assume no spin-dependent. 

    Parameters
    ----------
    v_abcd : np.ndarray
        The static Coulomb tensor of shape ``(norb, norb, norb, norb)`` in the product basis
        (``v_abcd <-> c^dag_b c^dag_c c_d c_a``).
    gf_struct : list of (str, int)
        The block structure, one ``(name, size)`` per block; the operator is written in the
        block names of ``gf_struct``.
    kind : types of Coulomb interactions, optional 
        - ``"density"``: the density-density only;
        - ``"kanamori"``: Hubbard-Kanamori form; 
        - ``"full"``: the general 4-index Coulomb Hamiltonian;
        Default ``"density"``.
    force_real : bool, optional
        Drop the imaginary part of the tensor. Default ``True``.

    Returns
    -------
    triqs.operators.Operator
        The interaction operator.

    Raises
    ------
    ValueError
        For an unknown ``kind``.
    """
    if kind == "density":
        return h_int_density_density(v_abcd, gf_struct, force_real=force_real)
    if kind == "kanamori":
        return h_int_slater(_kanamori_tensor(v_abcd), gf_struct, force_real=force_real)
    if kind == "full":
        return h_int_slater(v_abcd, gf_struct, force_real=force_real)
    raise ValueError(f"triqs_utils::h_int_operator: kind must be density, kanamori or full, "
                     f"got {kind!r}")


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
    """
    A DLR Green's function (``Gf`` or ``BlockGf`` on a DLR mesh) at the points of the IAFT mesh of its
    statistics, as an array ``(nw,) + target_shape`` (a list of them for a ``BlockGf``). The gf is
    not symmetrized here; call ``make_hermitian`` externally if needed.
    """
    if not isinstance(gf_dlr, (Gf, BlockGf)):
        raise ValueError(f"gf_dlr_to_ir: Invalid type of gf_dlr")
    stats = 'b' if gf_dlr.mesh.statistic == 'Boson' else 'f'
    return eval_on_points(gf_dlr, iaft.wn_mesh(stats=stats), gf_dlr.mesh.statistic, gf_dlr.mesh.beta)


def gf_dlr_to_ir_phsym(gf_dlr, iaft):
    """
    A bosonic DLR ``Gf`` at the non-negative half of the IAFT bosonic mesh, as an array
    ``(nw_b_pos,) + target_shape``.
    """
    assert gf_dlr.mesh.statistic == "Boson", (
        "gf_dlr_to_ir_phsym: Gf statistics must be Boson"
    )
    return eval_on_points(gf_dlr, iaft.wn_mesh(stats='b', positive_only=True), "Boson", gf_dlr.mesh.beta)


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


@dataclass
class TriqsInputs:
  """
  The impurity problem as the TRIQS backends take it, built by ``problem_to_triqs``.

  Attributes
  ----------
  mesh_f : triqs.gf.MeshDLRImFreq
      The symmetrized fermionic DLR Matsubara mesh of ``delta_iw``.
  mesh_b : triqs.gf.MeshDLRImFreq
      The symmetrized bosonic DLR Matsubara mesh of ``u_weiss_iw``.
  h0_op : triqs.operators.Operator
      The one-body part, ``sum c^dag h0 c`` over the blocks of ``gf_struct``, with ``mu_shift``
      subtracted from its diagonal.
  h0_blocks : list of ndarray
      The same one-body part as one matrix per ``gf_struct`` block, in the order of ``gf_struct``.
  delta_iw : triqs.gf.BlockGf
      The hybridization function on ``mesh_f``, one block per ``gf_struct`` block.
  h_int : triqs.operators.Operator
      The static interaction, ``h_int_density`` or a Slater operator depending on ``h_int_kind``.
  u_weiss_iw : triqs.gf.Block2Gf
      The retarded density-density interaction ``U_ij(i nu)`` on ``mesh_b``, indexed by pairs of
      ``gf_struct`` blocks.
  """
  mesh_f: object
  mesh_b: object
  h0_op: object
  h0_blocks: list
  delta_iw: object
  h_int: object
  u_weiss_iw: object


def dlr_meshes(beta, dlr_wmax, dlr_eps):
  """Symmetrized fermionic and bosonic DLR Matsubara meshes of the impurity problem."""
  mesh_f = MeshDLRImFreq(beta=beta, statistic='Fermion', w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)
  mesh_b = MeshDLRImFreq(beta=beta, statistic='Boson', w_max=dlr_wmax, eps=dlr_eps, symmetrize=True)
  return mesh_f, mesh_b


def _between_blocks_max(array_sab, gf_struct):
  """
  The largest element of a one-body array that couples two different ``gf_struct`` blocks.

  Parameters
  ----------
  array_sab : np.ndarray
      A one-body array of shape ``(..., nspin, norb, norb)``, e.g. ``h0_sab`` or ``delta_wsab``.
  gf_struct : list of (str, int)

  Returns
  -------
  float
      Largest element being dropped. 
  """
  norb = array_sab.shape[-1]
  block_of = {0: np.zeros(norb, dtype=int), 1: np.zeros(norb, dtype=int)}
  offset = [0, 0]
  for index, (name, size) in enumerate(gf_struct):
    spin = 0 if name[:2] == "up" else 1
    block_of[spin][offset[spin]:offset[spin] + size] = index
    offset[spin] += size
  largest = 0.0
  for spin in (0, 1):
    mask = block_of[spin][:, None] != block_of[spin][None, :]
    if mask.any():
      largest = max(largest, float(np.abs(array_sab[..., spin, :, :][..., mask]).max()))
  return largest

def _outside_density_density(A_abcd, *, keep_exchange):
  """
  The largest element of a four-index interaction tensor outside its density-density part.

  Parameters
  ----------
  A_abcd : np.ndarray
      An interaction tensor of shape ``(..., norb, norb, norb, norb)`` in the product basis
      (``v_abcd`` or ``u_weiss_wabcd``).
  keep_exchange : bool
      Count the exchange elements ``A[a, b, b, a]`` as kept, as the density-density ``h_int``
      and the ``screen_j`` treatment of the retarded interaction do.

  Returns
  -------
  float
      ``max |A_abcd[..., a, b, c, d]|`` over the elements with ``a != b`` or ``c != d``,
      excluding the exchange elements when ``keep_exchange`` is set; 0 when nothing is dropped.
  """
  norb = A_abcd.shape[-1]
  a, b, c, d = np.indices((norb,) * 4)
  keep = (a == b) & (c == d)
  if keep_exchange:
    keep |= (a == d) & (b == c)
  dropped = A_abcd[..., ~keep]
  return float(np.abs(dropped).max()) if dropped.size else 0.0


def _kanamori_tensor(v_abcd):
  """
  The Kanamori part of a Coulomb tensor: every other element set to zero.

  Parameters
  ----------
  v_abcd : np.ndarray
      The static Coulomb tensor of shape ``(norb, norb, norb, norb)`` in the product basis.

  Returns
  -------
  np.ndarray
      A copy holding only the density-density ``v[i, i, j, j]``, exchange ``v[i, j, j, i]`` and
      pair-hopping ``v[i, j, i, j]`` elements, for the Kanamori form of ``h_int_operator``.
  """
  norb = v_abcd.shape[-1]
  out = np.zeros_like(v_abcd)
  for i in range(norb):
    for j in range(norb):
      out[i, i, j, j] = v_abcd[i, i, j, j]
      out[i, j, j, i] = v_abcd[i, j, j, i]
      out[i, j, i, j] = v_abcd[i, j, i, j]
  return out


def _log_dropped_elements(problem, *, h_int_kind, screen_j, real_hamiltonian):
  """
  Log the largest element, in Ha, that each reduction of ``problem_to_triqs`` drops.

  A warning is added when the one-body part couples orbitals of different ``gf_struct`` blocks
  above ``between_blocks_warn`` (downfolded problems carry such elements at the 1e-5 level as
  numerical noise; the block cutting drops them either way). Nothing is returned: the log is
  the only report.

  Parameters
  ----------
  problem : ImpurityProblem
      The problem about to be converted.
  h_int_kind : {"density", "kanamori", "full"}
      The form of the static interaction; the elements of ``v_abcd`` outside density-density
      and exchange are reported for ``"density"`` only, the other forms drop nothing.
  screen_j : bool
      Whether the retarded exchange is kept (then it is not counted as dropped).
  real_hamiltonian : bool
      Whether the imaginary parts of h0 and Delta(tau) are dropped (then their size is reported).
  """
  gf_struct = problem.gf_struct
  h0_between_blocks = _between_blocks_max(problem.h0_sab, gf_struct)
  delta_between_blocks = _between_blocks_max(problem.delta_wsab, gf_struct)
  u_outside_dd = _outside_density_density(problem.u_weiss_wabcd, keep_exchange=screen_j)
  lines = [f"  dropped h0 between blocks        : {h0_between_blocks:.2e} Ha",
           f"  dropped Delta between blocks     : {delta_between_blocks:.2e} Ha",
           f"  dropped u outside density-density: {u_outside_dd:.2e} Ha"
           + (" (exchange kept, screen_j)" if screen_j else "")]
  if h_int_kind == "density":
    v_outside_dd = _outside_density_density(problem.v_abcd, keep_exchange=True)
    lines.append(f"  dropped V outside density-density + exchange: {v_outside_dd:.2e} Ha")
  if real_hamiltonian:
    imag_h0 = float(np.abs(problem.h0_sab.imag).max())
    imag_delta_tau = float(np.abs(problem.iaft.w_to_tau(problem.delta_wsab, 'f').imag).max())
    lines.append(f"  dropped imaginary parts          : h0 {imag_h0:.2e}, "
                 f"Delta(tau) {imag_delta_tau:.2e} Ha")
  app_log(1, "Impurity problem reduced for the solver (h_int = " + h_int_kind + "):\n"
             + "\n".join(lines) + "\n")
  between = max(h0_between_blocks, delta_between_blocks)
  # an h0/Delta element between two blocks above this is physics, not downfolding noise (~1e-5 Ha)
  between_blocks_warn = 1e-3
  if between > between_blocks_warn:
    app_log(1, f" [WARNING] the impurity problem couples orbitals of different gf_struct blocks "
               f"({between:.2e} Ha, far above\n  the numerical noise of a downfolding); the block "
               f"structure drops that coupling. Use blocks that contain the coupled\n  orbitals "
               f"with a backend that accepts them (offdiag_delta).\n")


def problem_to_triqs(problem, *, h_int_kind="density", screen_j=False, real_hamiltonian=True):
  """
  Build the TRIQS containers of an ``ImpurityProblem``.

  The most general impurity Hamiltonian the function can return is

      H = sum_{blocks} c^dag h0 c  +  h_int[v_abcd]  +  sum_{ij} U_ij(i nu) n_i n_j

  with the hybridization Delta(i omega) alongside. In physical terms:

  - One-body part (h0 and Delta):
    - block-diagonal in ``gf_struct`` and spin-diagonal (two spin blocks ``up`` and ``down``),
      matrix-valued inside a block;
    - elements between two ``gf_struct`` blocks are dropped;
    - the imaginary parts are dropped when ``real_hamiltonian`` is set;
    - ``mu_shift`` is subtracted from the diagonal of h0.
  - Static Coulomb interaction, its form is set by ``h_int_kind``:
    - ``"density"``: density-density-only interactions
    - ``"kanamori"``: Hubbard-Kanamori interactions
    - ``"full"``: general four-index interactions
    - currently, assume the same interaction for both spins.
    - the imaginary parts are dropped when ``real_hamiltonian`` is set; 
  - Retarded interaction:
    - currently, only the density-density part is supported. (extension will be addressed in the future)
    - the exchange part is folded into the same-spin channel when ``screen_j`` is set;

  When ``problem.degenerate_blk`` is given, h0, Delta and the retarded interaction are
  symmetrized over the degenerate groups of blocks. The static interaction is symmetrized only
  for ``h_int_kind = "density"``: the symmetrization of a Kanamori or full operator is not
  implemented yet.

  Each part is built by its own builder (``h0_operator``, ``delta_block_gf``,
  ``h_int_operator``, ``dynamic_4idx_u_to_dd_basis`` + ``to_block2_gf``); this function owns the
  sequence: the drop report, the meshes, the ``mu_shift``, the builders, the symmetrization.

  Parameters
  ----------
  problem : ImpurityProblem
  h_int_kind : {"density", "kanamori", "full"}, optional
      The form of the static interaction, see above. Default ``"density"``.
  screen_j : bool, optional
      Keep the retarded exchange in the same-spin channel of ``u_weiss_iw`` (as in
      ``dynamic_4idx_u_to_dd_basis``). Default ``False``.
  real_hamiltonian : bool, optional
      Drop the imaginary parts of h0 and Delta(tau) and the interactions. 
      Default ``True``.

  Returns
  -------
  TriqsInputs
      The converted problem in TRIQS containers. See ``TriqsInputs`` for the data format. 

  Raises
  ------
  ValueError
      For an unknown ``h_int_kind``.
  """
  gf_struct = problem.gf_struct
  if h_int_kind not in ("density", "kanamori", "full"):
    raise ValueError(f"triqs_utils::problem_to_triqs: h_int must be density, kanamori or full, "
                     f"got {h_int_kind!r}")
  _log_dropped_elements(problem, h_int_kind=h_int_kind, screen_j=screen_j,
                        real_hamiltonian=real_hamiltonian)

  # imaginary mesh for the impurity problem in TRIQS
  mesh_f, mesh_b = dlr_meshes(beta=problem.beta, dlr_wmax=problem.dlr_wmax, dlr_eps=problem.dlr_eps)
  
  # non-interacting Hamiltonian
  h0_sab = problem.h0_sab - problem.mu_shift * np.eye(problem.h0_sab.shape[-1])
  h0_op = h0_operator(h0_sab, gf_struct, force_real=real_hamiltonian)
  # hybridization function
  delta_gf = delta_block_gf(problem.delta_wsab, problem.iaft, gf_struct, mesh_f,
                            force_real=real_hamiltonian)
  # static Coulomb interaction
  h_int = h_int_operator(problem.v_abcd, gf_struct, kind=h_int_kind, 
                         force_real=real_hamiltonian)
  # dynamic Coulomb interaction (currently only density-density terms)
  u_iw_s1s2 = coqui_dmft.dynamic_4idx_u_to_dd_basis(problem.u_weiss_wabcd, nspin=2, screen_j=screen_j)
  u_gf = to_block2_gf(u_iw_s1s2.real if real_hamiltonian else u_iw_s1s2, problem.iaft, gf_struct, mesh_b)
  
  # symmetrize over degenerate blocks if given
  if problem.degenerate_blk is not None:
    delta_gf = modest.symmetrize(delta_gf, problem.degenerate_blk)
    h0_op = symmetrize_h0_op(h0_op, problem.degenerate_blk, gf_struct)
    if h_int_kind == "density":
      h_int = symmetrize_h_int_op(h_int, problem.degenerate_blk, gf_struct)
    else:
      app_log(1, f" [WARNING] triqs_utils::problem_to_triqs: the symmetrization of h_int over the "
                 f"degenerate blocks is implemented\n  for h_int = \"density\" only; the "
                 f"{h_int_kind} interaction is left unsymmetrized while h0, Delta and u_weiss "
                 f"are symmetrized.\n")
    u_gf = symmetrize_blk2_gf(u_gf, problem.degenerate_blk, gf_struct)

  # the matrices of the final (shifted, symmetrized) h0 operator
  h0_blocks = [np.asarray(blk, dtype=complex) for blk in block_matrix_from_op(h0_op, gf_struct)]
  return TriqsInputs(mesh_f=mesh_f, mesh_b=mesh_b, h0_op=h0_op, h0_blocks=h0_blocks,
                     delta_iw=delta_gf, h_int=h_int, u_weiss_iw=u_gf)


def analyze_degenerate_blocks(delta_iw, iaft, gf_struct, dlr_wmax, dlr_eps, threshold, *,
                              force_real=True):
  """
  Degenerate groups of ``gf_struct`` blocks inferred from Delta(iw) with ModEST's rule, on the
  same (real-forced, DLR-sampled) Delta the driver passed to ModEST before this interface.
  """
  mesh_f, _ = dlr_meshes(beta=iaft.beta, dlr_wmax=dlr_wmax, dlr_eps=dlr_eps)
  delta_gf = delta_block_gf(delta_iw, iaft, gf_struct, mesh_f, force_real=force_real)
  return modest.analyze_degenerate_blocks(delta_gf, threshold=threshold)
