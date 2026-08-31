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

import sys
import h5py
from coqui._lib.utils_module import app_log

class IterSCF(object):
    @staticmethod
    def damping_impl(h5_grp, mixing, X, dataset):
        app_log(1, "IterSCF::damping: alpha * X + (1-alpha) * X_prev")
        app_log(1, "  - X              = {}".format(dataset))
        app_log(1, "  - X_prev from h5 = {}".format(h5_grp.name + "/" + dataset))
        app_log(1, "  - alpha          = {}\n".format(mixing))
        sys.stdout.flush()

        X_prev = h5_grp[dataset][()].view(complex)[...,0]
        X[:] = mixing * X + (1 - mixing) * X_prev


    @staticmethod
    def damping(coqui_h5, current_iter, mixing, data_dict):
        app_log(1, f"\nApplying mixing for impurity self-energy with mixing factor = {mixing}")
        app_log(1, "We will mix the double counting terms if there is no previous impurity solution.")
        with h5py.File(coqui_h5, 'r') as f:
            df_grp = f['downfold_1e']
            found_imp_solutions = False
            if f"iter{current_iter-1}" in df_grp:
                prev_iter_grp = df_grp[f"iter{current_iter - 1}"]
                if data_dict.keys() <= prev_iter_grp.keys():
                    found_imp_solutions = True
                    for key, data in data_dict.items():
                        IterSCF.damping_impl(prev_iter_grp, mixing, data, key)

            if not found_imp_solutions:
                iter_grp = df_grp[f"iter{current_iter}"]
                for key, data in data_dict.items():
                    # mix with dc contribution instead
                    dc_key = key.replace("imp", "dc")
                    IterSCF.damping_impl(iter_grp, mixing, data, dc_key)

    @staticmethod
    def diis(current_res, current_vec, *,
             diis_subspace, diis_restart, current_iter,
             extrplt=True, diis_chkpt="diis.h5"):
        import os
        from collections import deque
        from .diis import diis_solve

        app_log(1, "\nIterSCF::diis")
        app_log(1, "----------------")
        app_log(1, "  - checkpoint h5        = {}".format(diis_chkpt))
        app_log(1, "  - vector list          = {}".format(list(current_vec.keys())))
        app_log(1, "  - max subspace size    = {}".format(diis_subspace))
        app_log(1, "  - current iteration    = {}".format(current_iter))
        app_log(1, "  - extrapolation        = {}".format(extrplt))
        sys.stdout.flush()

        # initialize vector, residual spaces
        res_list = deque(maxlen=diis_subspace)
        vec_list = deque(maxlen=diis_subspace)

        if diis_restart and not os.path.exists(diis_chkpt):
            diis_restart = False

        if not diis_restart:
            # create diis h5 for intermediate data
            with h5py.File(diis_chkpt, 'w') as f:
                f["current_iter"] = current_iter
                f["diis_subspace"] = diis_subspace

                res_list.append(current_res)
                res_grp = f.create_group("residuals")
                res_grp["space_size"] = len(res_list)
                res_grp["res0"] = res_list[0]

                vec_list.append(current_vec)
                vec_grp = f.create_group("vectors")
                vec_grp["space_size"] = len(vec_list)
                for key, data in vec_list[0].items():
                    vec_grp[f"vec0/{key}"] = data

        else:
            with h5py.File(diis_chkpt, 'a') as f:
                # check diis parameters are consistent with those in diis.h5
                assert diis_subspace == f["diis_subspace"][()]

                # recover diis status from the previous calculation
                res_grp = f["residuals"]
                res_size = res_grp["space_size"][()]
                for i in range(res_size):
                    res = res_grp[f"res{i}"][()]
                    res_list.append(res)
                # current iteration
                res_list.append(current_res)

                vec_grp = f["vectors"]
                vec_size = vec_grp["space_size"][()]
                for i in range(vec_size):
                    vec = {}
                    for key in current_vec.keys():
                        vec[key] = vec_grp[f"vec{i}/{key}"][()]
                    vec_list.append(vec)
                # current iteration
                vec_list.append(current_vec)

                # update diis.h5
                f["current_iter"][()] = current_iter
                res_grp["space_size"][()] = len(res_list)
                for i in range(len(res_list)):
                    if f"res{i}" in res_grp:
                        res_grp[f"res{i}"][()] = res_list[i]
                    else:
                        res_grp[f"res{i}"] = res_list[i]

                vec_grp["space_size"][()] = len(vec_list)
                for i in range(len(vec_list)):
                    for key, data in vec_list[i].items():
                        if f"vec{i}/{key}" in vec_grp:
                            vec_grp[f"vec{i}/{key}"][()] = data
                        else:
                            vec_grp[f"vec{i}/{key}"] = data

            app_log(1, "  - restart diis          = {}".format(diis_restart))
            app_log(1, "  - current residual size = {}".format(len(res_list)))
            app_log(1, "  - current vector size   = {}".format(len(vec_list)))
            sys.stdout.flush()

        if extrplt:
            app_log(1, "\nSolving DIIS equation for extrapolation coefficients:")
            coeffs = diis_solve(res_list)
            app_log(1, f"Coefficients = {coeffs}\n")
            # linear extrapolation
            for key in current_vec.keys():
                current_vec[key] *= coeffs[-1]
                for i, c in enumerate(coeffs[:-1]):
                    current_vec[key] += c * vec_list[i][key]
        else:
            app_log(1, f"\nDIIS starting iteration not yet achieved - "
                       f"expanding vector and residual spaces without DIIS extrapolation.\n")


