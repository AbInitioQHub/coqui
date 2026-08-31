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

import numpy as np
import h5._h5py as h5
import sparse_ir

'''
The script that precomputed IR grid and sparse sampling using the sparse_ir python library 
'''

def gen_dimensionless_ir_bases(outdir, prefix, lmbda, eps=None):
    '''
    Generate dimensionless IR grid and sparse sampling, and store in a hdf5 file (lmbda.h5)
    :param outdir: output path
    :param lmbda: lmbda = beta * wmax
    :return:
    '''
    beta = 100
    wmax = lmbda/beta
    if eps is None:
        bases = sparse_ir.FiniteTempBasisSet(beta=beta, wmax=wmax)
    else:
        bases = sparse_ir.FiniteTempBasisSet(beta=beta, wmax=wmax, eps=eps)
    tau_mesh_f = bases.smpl_tau_f.sampling_points
    tau_mesh_b = bases.smpl_tau_b.sampling_points
    wn_mesh_f = bases.smpl_wn_f.sampling_points
    wn_mesh_b = bases.smpl_wn_b.sampling_points
    nt_f, nw_f = tau_mesh_f.shape[0], wn_mesh_f.shape[0]
    nt_b, nw_b = tau_mesh_b.shape[0], wn_mesh_b.shape[0]
    print("### Lambda = {} ###".format(lmbda))
    print("accuracy = {}".format(bases.accuracy))
    print("nt_f, nw_f, nt_b, nw_b = {}, {}, {}, {}".format(nt_f, nw_f, nt_b, nw_b))

    # transformation matrices in the dimensionless format
    Ttl_ff = bases.basis_f.u(tau_mesh_f).T * np.sqrt(beta/2.0)
    Twl_ff = bases.basis_f.uhat(wn_mesh_f).T * np.sqrt(1.0/beta)
    Ttl_bb = bases.basis_b.u(tau_mesh_b).T * np.sqrt(beta/2.0)
    Twl_bb = bases.basis_b.uhat(wn_mesh_b).T * np.sqrt(1.0/beta)

    Tlt_ff = np.linalg.pinv(Ttl_ff)
    Tlt_bb = np.linalg.pinv(Ttl_bb)

    # Ttw_ff = Ttl_ff * [Twl_ff]^{-1}
    Ttw_ff = np.dot(Ttl_ff, np.linalg.pinv(Twl_ff))
    Twt_ff = np.dot(Twl_ff, np.linalg.pinv(Ttl_ff))
    Ttw_bb = np.dot(Ttl_bb, np.linalg.pinv(Twl_bb))
    Twt_bb = np.dot(Twl_bb, np.linalg.pinv(Ttl_bb))

    # Gt_f = Ttl_ff * Gl_f -> Gl_f = (Ttl_ff)^{-1} * Gt_f
    # Gt_b = Ttl_bf * Gl_f = Ttl_bf * (Ttl_ff)^{-1} * Gt_f
    #      = Ttt_bf * Gt_f
    # -> Ttt_bf = Ttl_bf * (Ttl_ff)^{-1}
    Ttl_bf = bases.basis_f.u(tau_mesh_b).T * np.sqrt(beta/2.0)
    Ttt_bf = np.dot(Ttl_bf, np.linalg.pinv(Ttl_ff))
    # Ttt_fb = Ttl_fb * (Ttl_bb)^{-1}
    Ttl_fb = bases.basis_b.u(tau_mesh_f).T * np.sqrt(beta/2.0)
    Ttt_fb = np.dot(Ttl_fb, np.linalg.pinv(Ttl_bb))

    # Tbetat_ff = T_betal_ff * (Ttl_ff)^{-1}
    T_betal_ff = bases.basis_f.u(beta).T * np.sqrt(beta/2.0)
    T_betat_ff = np.dot(T_betal_ff, np.linalg.pinv(Ttl_ff))

    # tau = 0^+
    T_zerol_ff = bases.basis_f.u(0).T * np.sqrt(beta/2.0)
    T_zerot_ff = np.dot(T_zerol_ff, np.linalg.pinv(Ttl_ff))

    Ttt_bf = Ttt_bf.astype(complex)
    T_betat_ff = T_betat_ff.astype(complex)
    Tlt_ff = Tlt_ff.astype(complex)

    Ttt_fb = Ttt_fb.astype(complex)
    Tlt_bb = Tlt_bb.astype(complex)

    tau_mesh_f = tau_mesh_f * (2.0/beta) - 1.0
    tau_mesh_b = tau_mesh_b * (2.0/beta) - 1.0

    if eps is None:
        filename = outdir + "/" + prefix + ".h5"
    else:
        filename = outdir + "/" + prefix + "." + str(eps) + ".h5"
    f = h5.File(filename, 'w')
    g = h5.Group(f)
    h5.h5_write(g, "version", sparse_ir.__version__)

    f_grp = g.create_group("fermion")
    h5.h5_write(f_grp, "nt", np.int32(nt_f))
    h5.h5_write(f_grp, "nw", np.int32(nw_f))
    h5.h5_write(f_grp, "tau_mesh", tau_mesh_f)
    h5.h5_write(f_grp, "wn_mesh", wn_mesh_f)
    h5.h5_write(f_grp, "Ttw", Ttw_ff)
    h5.h5_write(f_grp, "Twt", Twt_ff)
    h5.h5_write(f_grp, "Ttt_bf", Ttt_bf)
    h5.h5_write(f_grp, "T_betat", T_betat_ff)
    h5.h5_write(f_grp, "T_zerot", T_zerot_ff)
    h5.h5_write(f_grp, "Tct", Tlt_ff)

    b_grp = g.create_group("boson")
    h5.h5_write(b_grp, "nt", np.int32(nt_b))
    h5.h5_write(b_grp, "nw", np.int32(nw_b))
    h5.h5_write(b_grp, "tau_mesh", tau_mesh_b)
    h5.h5_write(b_grp, "wn_mesh", wn_mesh_b)
    h5.h5_write(b_grp, "Ttw", Ttw_bb)
    h5.h5_write(b_grp, "Twt", Twt_bb)
    h5.h5_write(b_grp, "Ttt_fb", Ttt_fb)
    h5.h5_write(b_grp, "Tct", Tlt_bb)

    del f

lamb_dict = {"1e2": 100, "1e3": 1000, "1e4": 10000, "1e5": 100000, "1e6": 1000000}
for fname in lamb_dict:
    print(fname)
    gen_dimensionless_ir_bases("./", fname, lamb_dict[fname], 1e-6)
    gen_dimensionless_ir_bases("./", fname, lamb_dict[fname], 1e-10)
    gen_dimensionless_ir_bases("./", fname, lamb_dict[fname], 1e-15)
