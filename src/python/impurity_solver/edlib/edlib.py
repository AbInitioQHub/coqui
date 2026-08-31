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
import os
import numpy as np
import h5py
import subprocess

from py2aimb.imag_axes_ft import IAFT
from py2aimb.dmft.iterative_scf.iter_scf import IterSCF
from py2aimb.dmft import extract_t_and_delta, bath_fit, estimate_zero_moment

""" Utilities to interface with EDLib (https://github.com/Q-solvers/EDLib/tree/master) """


class EDLib(object):
    @staticmethod
    def solve(aimbes, ed_prefix: str, ed_cmd: str,
              nbath: int, wmax: float = None,
              NEV: int = 10, NCV: int = 60,
              nlanc: int = 100, boltzmann_cutoff: float = 1e-12,
              truncate_decimal: int = 12, occ_sector=None,
              iter_solver: str = "mixing",
              it_1e: int = None, it_2e: int = None,
              exact_hf_shift: bool = False,
              **kwargs):
        """
        Solve impurity problem for a given AIMBES system.
        :param aimbes: AIMBES reader instance
        :param ed_prefix: str
            Prefix for input files: prefix.edlib.input.h5 and prefix.param
        :param nbath: int
            Number of bath orbitals per impurity orbital
        :param wmax: float
            Frequency window for the impurity problem to determine the sizes of IR grid.
            The IR grid from the AIMBES chkpt file is suited for the bulk system, which
            is too large for the impurity. The abusive usage of too large IR grid would
            result in numerical instability.
            The default is None (use the IR grid from the AIMBES chkpt file)
        :param NEV: int
            Number of eigenvalues to find for each particle sector in ED
        :param NCV: int
            Number of Lanczos vectors to be generated in ED
        :param nlanc: int
            Number of Lanczos iterations in ED
        :param boltzmann_cutoff: float: float
            Cutoff for Boltzmann factor in ED
        :param truncate_decimal: int
            Truncated decimal for impurity Hamiltonian parsed to ED.
            This is useful to preserve degeneracy in the presence of numerical noise within a DMFT loop.
        :param occ_sector: numpy.ndarray(dim=2)
            Occupation sectors for EDLib. Default = None
        :param mixing: float
            Mixing parameter for impurity self-energy. Default = 1.0, i.e. no mixing
        :param it_1e: int
            Iteration for downfold_1e. Default = None, i.e. read from aimbes checkpoint file.
        :param it_2e: int
            Iteration for downfold_2e. Default = None, i.e. read from aimbes checkpoint file.
        :param exact_hf_shift: bool. Default = False.
            Compute the static impurity self-energy as the Vhf[\rho_imp], \rho_imp is the impurity density.
        """
        print("Preparing input files for EDLib (https://github.com/Q-solvers/EDLib/tree/master): ")
        print("    - file prefix = {}".format(ed_prefix))
        print("    - nbath / impurity orbital = {}".format(nbath))
        print("    - number of eigenvalues = {}".format(NEV))
        print("    - NCV = {}".format(NCV))
        print("    - boltzmann_cutoff = {}".format(boltzmann_cutoff))
        print("    - truncated decimal = {}\n".format(truncate_decimal))
        sys.stdout.flush()

        g_weiss_wsIab, U_abcd = aimbes.impurity_weiss_field(static_U=True, force_real=True, it_1e=it_1e, it_2e=it_2e)

        if wmax is None:
            print("Uses the IR basis read from the AIMBES chkpt file.")
            iaft_imp = aimbes.iaft
            two_iaft = False

            g_weiss_imp = g_weiss_wsIab
        else:
            print("Constructs a new IR basis with wmax = {}, beta={}".format(wmax, aimbes.iaft.beta))
            iaft_imp = IAFT(aimbes.iaft.beta, wmax*aimbes.iaft.beta, aimbes.iaft.prec, basis='ir')
            two_iaft = True

            g_weiss_imp = aimbes.iaft.w_interpolate(g_weiss_wsIab, iaft_imp.wn_mesh('f'), 'f')
            iaft_imp.check_leakage(g_weiss_imp, 'f', "g_weiss_interpolate", w_input=True)

        # write inputs for EDlib
        t_sIab, delta_wsIab = extract_t_and_delta(g_weiss_imp, iaft_imp)
        bath_energy, bath_hyb, delta_fit = bath_fit(delta_wsIab[:, :, 0],
                                                    1j*iaft_imp.wn_mesh('f')*np.pi/iaft_imp.beta, nbath)
        # force real
        bath_energy.imag = 0.0
        bath_hyb.imag = 0.0
        if truncate_decimal is not None:
            bath_energy = bath_energy.round(decimals=truncate_decimal)
            bath_hyb = bath_hyb.round(decimals=truncate_decimal)

        iwn_mesh = iaft_imp.wn_mesh('f', phys_notation=True)
        niw_pos = max(abs(iwn_mesh[0]), iwn_mesh[-1]) + 1
        EDLib.write_ed_input(ed_prefix, bath_energy, bath_hyb,
                             t_sIab[:, 0], U_abcd,
                             iaft_imp.beta, nlanc, boltzmann_cutoff, niw_pos,
                             NEV, NCV, truncate_decimal=None,
                             occ_sector=occ_sector)
        with h5py.File(ed_prefix+".edlib.input.h5", 'a') as f:
            f['delta_fit/data'] = delta_fit
            f['delta_fit/iw_mesh'] = iaft_imp.wn_mesh('f') * np.pi / iaft_imp.beta
            f['t_sIab'] = t_sIab

        # call EDLib through subprocess
        print("EDLib command: ")
        print(ed_cmd + " {}.param\n".format(ed_prefix))
        sys.stdout.flush()

        cmd = ed_cmd.split() + [ed_prefix+".param"]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            raise ValueError("EDLib command fail with returncode = {}".format(result.returncode))
        sys.stdout.flush()

        # update iaft_imp if not two_iaft
        if not two_iaft:
            iaft_imp = None

        EDLib.post_processing(aimbes, ed_prefix, t_sIab, U_abcd,
                              bath_energy, bath_hyb, it_1e, iaft_imp=iaft_imp,
                              iter_solver=iter_solver, exact_hf_shift=exact_hf_shift,
                              **kwargs)


    @staticmethod
    def post_processing(aimbes, ed_prefix, t_sIab, U_chem,
                        bath_energy, bath_hyb, it_1e=None, iaft_imp=None,
                        exact_hf_shift=True, iter_solver="mixing", **kwargs):
        """

        :param aimbes:
        :param ed_prefix:
        :param t_sIab:
        :param bath_energy:
        :param bath_hyb:
        :param it_1e:
        :param iaft_imp:
        :param mixing:
        :param exact_hf_shift:
        :return:
        """
        import scipy.linalg as LA
        from py2aimb.dmft import hybridization

        ns, nImps, nImpOrbs = t_sIab.shape[:3]
        iw_mesh = aimbes.iaft.wn_mesh('f', phys_notation=True)

        # read impurity Green's function
        with h5py.File(ed_prefix + ".edlib.sim.h5", "r") as edout:
            if nImpOrbs == 1:
                Gimp_wsab_uniform = edout["results/G_omega/data"][()].view(complex)[..., 0]
            else:
                Gimp_wsab_uniform = edout["results/G_ij_omega/data"][()].view(complex)[..., 0]
            Gimp_wsab_uniform = np.einsum("wIs->wsI", Gimp_wsab_uniform)
            Gimp_wsab_uniform = Gimp_wsab_uniform.reshape(Gimp_wsab_uniform.shape[:2] + (nImpOrbs, nImpOrbs))

        if iaft_imp is not None:
            iw_mesh_tmp = iaft_imp.wn_mesh('f', phys_notation=True)
            Gimp_wsIab_tmp = np.zeros((iaft_imp.nw_f, ns, 1, nImpOrbs, nImpOrbs), dtype=complex)
            for i, n in enumerate(iw_mesh_tmp):
                for s in range(ns):
                    Gimp_wsIab_tmp[i, s, 0] = Gimp_wsab_uniform[n, s] if n >= 0 else Gimp_wsab_uniform[abs(n) - 1, s].conj()
            Gimp_tsIab_tmp = iaft_imp.w_to_tau(Gimp_wsIab_tmp, stats='f')
            iaft_imp.check_leakage(Gimp_tsIab_tmp, stats='f', name="Impurity Green's function")

            # interpolate to aimbes.iaft IR grid
            Gimp_wsIab = iaft_imp.w_interpolate(Gimp_wsIab_tmp, iw_mesh, stats='f', phys_notation=True)
            Dm_sIab = -1 * iaft_imp.tau_interpolate(Gimp_tsIab_tmp, [iaft_imp.beta], stats='f')[0]

            del Gimp_wsab_uniform

            # compute impurity self-energy
            iw_mesh_tmp = (2*iw_mesh_tmp+1)*np.pi/iaft_imp.beta
            Simp_wsIab_tmp = np.zeros(Gimp_wsIab_tmp.shape, dtype=complex)
            delta_fit_wsab = hybridization(iw_mesh_tmp*1.j, bath_energy, bath_hyb)
            iaft_imp.check_leakage(delta_fit_wsab, stats='f', name="Fitted hybridization", w_input=True)
            for i, wn in enumerate(iw_mesh_tmp):
                for s in range(ns):
                    Gimp_inv = LA.inv(Gimp_wsIab_tmp[i, s, 0])
                    G0_inv = (1j*wn)*np.eye(nImpOrbs) - t_sIab[s, 0] - delta_fit_wsab[i, s]
                    # this will BREAK when Gloc(t) is not real, e.g. SOC.
                    Simp_wsIab_tmp[i, s, 0] = G0_inv - Gimp_inv

            if exact_hf_shift:
                Simp_static = EDLib.hf_shift(Dm_sIab, U_chem)
            else:
                Simp_static = estimate_zero_moment(Simp_wsIab_tmp, iw_mesh_tmp, "impurity self-energy")
            Simp_wsIab_tmp = Simp_wsIab_tmp - Simp_static
            iaft_imp.check_leakage(Simp_wsIab_tmp, stats='f', name="Impurity self-energy", w_input=True)
            Simp_wsIab = iaft_imp.w_interpolate(Simp_wsIab_tmp, iw_mesh, stats='f', phys_notation=True)
            aimbes.iaft.check_leakage(Simp_wsIab, stats='f', name="Impurity self-energy (interpolated)",
                                      w_input=True)

        else:
            Gimp_wsIab = np.zeros((aimbes.iaft.nw_f, ns, 1, nImpOrbs, nImpOrbs), dtype=complex)
            for i, n in enumerate(iw_mesh):
                for s in range(ns):
                    Gimp_wsIab[i,s,0] = Gimp_wsab_uniform[n,s] if n >= 0 else Gimp_wsab_uniform[abs(n)-1,s].conj()
            del Gimp_wsab_uniform

            # compute impurity self-energy
            iw_mesh = (2*iw_mesh+1)*np.pi/aimbes.iaft.beta
            Simp_wsIab = np.zeros(Gimp_wsIab.shape, dtype=complex)
            delta_fit_wsab = hybridization(iw_mesh*1.j, bath_energy, bath_hyb)
            aimbes.iaft.check_leakage(delta_fit_wsab, stats='f', name="Fitted hybridization", w_input=True)
            for i, wn in enumerate(iw_mesh):
                for s in range(ns):
                    Gimp_inv = LA.inv(Gimp_wsIab[i, s, 0] * wn) * wn
                    # this will BREAK when Gloc(t) is not real, e.g. SOC.
                    # t_sIab already contains chemical potential
                    Simp_wsIab[i,s,0] = 1j*wn*np.eye(nImpOrbs) - t_sIab[s, 0] - delta_fit_wsab[i, s] - Gimp_inv

            Gimp_tsIab = aimbes.iaft.w_to_tau(Gimp_wsIab, stats='f')
            aimbes.iaft.check_leakage(Gimp_tsIab, stats='f', name="Impurity Green's function")
            Dm_sIab = -1 * aimbes.iaft.tau_interpolate(Gimp_tsIab, [aimbes.iaft.beta], stats='f')[0]
            if exact_hf_shift:
                Simp_static = EDLib.hf_shift(Dm_sIab, U_chem)
            else:
                Simp_static = estimate_zero_moment(Simp_wsIab, iw_mesh, "impurity self-energy")
            Simp_wsIab = Simp_wsIab - Simp_static
            aimbes.iaft.check_leakage(Simp_wsIab, stats='f', name="Impurity self-energy", w_input=True)
            del Gimp_tsIab

        it_1e_ = aimbes.df_1e_iter if it_1e is None else it_1e

        # iterative solver
        if iter_solver == "mixing":
            mixing = kwargs["mixing"] if "mixing" in kwargs else 0.7
            IterSCF.damping(aimbes.aimbes_h5, it_1e_, mixing,
                            {"Vhf_imp_sIab": Simp_static, "Sigma_imp_wsIab": Simp_wsIab})
        elif iter_solver == "ddiis" or iter_solver == "cdiis":
            diis_start = kwargs["diis_start"]

            # current residual
            if os.path.exists("diis.h5"):
                with h5py.File("diis.h5", 'r') as f:
                    vec_size = f["vectors/space_size"][()]
                    vec_grp = f[f"vectors/vec{vec_size-1}"]
                    res_Vhf = Simp_static - vec_grp["Vhf_imp_sIab"][()]
                    res_Sigma = Simp_wsIab - vec_grp["Sigma_imp_wsIab"][()]
            else:
                with h5py.File(aimbes.aimbes_h5, 'r') as f:
                    iter_grp = f[f"downfold_1e/iter{it_1e_}"]
                    res_Vhf = Simp_static - iter_grp["Vhf_dc_sIab"][()].view(complex)[..., 0]
                    res_Sigma = Simp_wsIab - iter_grp["Sigma_dc_wsIab"][()].view(complex)[..., 0]
            current_res = np.concatenate([res_Vhf.flatten(), res_Sigma.flatten()])

            if it_1e_ >= diis_start:
                # growing vector/residual spaces and then diis extrapolation
                IterSCF.diis(current_res, {"Vhf_imp_sIab": Simp_static, "Sigma_imp_wsIab": Simp_wsIab},
                             diis_subspace=kwargs["diis_subspace"], diis_restart=kwargs["diis_restart"],
                             current_iter=it_1e_, extrplt=True, diis_chkpt="diis.h5")
            else:
                # growing vector/residual spaces only
                IterSCF.diis(current_res, {"Vhf_imp_sIab": Simp_static, "Sigma_imp_wsIab": Simp_wsIab},
                             diis_subspace=kwargs["diis_subspace"], diis_restart=kwargs["diis_restart"],
                             current_iter=it_1e_, extrplt=False, diis_chkpt="diis.h5")
                # simple mixing instead
                mixing = kwargs["mixing"] if "mixing" in kwargs else 0.7
                IterSCF.damping(aimbes.aimbes_h5, it_1e_, mixing,
                                {"Vhf_imp_sIab": Simp_static, "Sigma_imp_wsIab": Simp_wsIab})

        print("Static impurity self-energy:")
        print("Spin 0")
        print(Simp_static[0, 0].real)
        if Simp_static.shape[0] == 2:
            print("Spin 1")
            print("{}\n".format(Simp_static[1, 0].real))
        else:
            print("")

        print("Correlated density from Gimp:")
        print("Spin 0")
        print(Dm_sIab[0, 0].real)
        if Dm_sIab.shape[0] == 2:
            print("Spin 1")
            print("{}\n".format(Dm_sIab[1, 0].real))
        else:
            print("")
        sys.stdout.flush()

        print("Writing impurity solutions to downfold_1e/iter{}".format(it_1e_))
        sys.stdout.flush()

        with h5py.File(aimbes.aimbes_h5, 'a') as f:
            if "G_imp_wsIab" in f['downfold_1e/iter{}'.format(it_1e_)]:
                f['downfold_1e/iter{}/G_imp_wsIab'.format(it_1e_)][...] = (
                    Gimp_wsIab.view(float).reshape(Gimp_wsIab.shape[0], ns, nImps, nImpOrbs, nImpOrbs, 2))
            else:
                f['downfold_1e/iter{}/G_imp_wsIab'.format(it_1e_)] = (
                    Gimp_wsIab.view(float).reshape(Gimp_wsIab.shape[0], ns, nImps, nImpOrbs, nImpOrbs, 2))
            f['downfold_1e/iter{}/G_imp_wsIab'.format(it_1e_)].attrs["__complex__"] = np.int8(1)

            if "Sigma_imp_wsIab" in f['downfold_1e/iter{}'.format(it_1e_)]:
                f['downfold_1e/iter{}/Sigma_imp_wsIab'.format(it_1e_)][...] = (
                    Simp_wsIab.view(float).reshape(Simp_wsIab.shape[0], ns, nImps, nImpOrbs, nImpOrbs, 2))
            else:
                f['downfold_1e/iter{}/Sigma_imp_wsIab'.format(it_1e_)] = (
                    Simp_wsIab.view(float).reshape(Simp_wsIab.shape[0], ns, nImps, nImpOrbs, nImpOrbs, 2))
            f['downfold_1e/iter{}/Sigma_imp_wsIab'.format(it_1e_)].attrs["__complex__"] = np.int8(1)

            if "Vhf_imp_sIab" in f['downfold_1e/iter{}'.format(it_1e_)]:
                f['downfold_1e/iter{}/Vhf_imp_sIab'.format(it_1e_)][...] = (
                    Simp_static.view(float).reshape(ns, nImps, nImpOrbs, nImpOrbs, 2))
            else:
                f['downfold_1e/iter{}/Vhf_imp_sIab'.format(it_1e_)] = (
                    Simp_static.view(float).reshape(ns, nImps, nImpOrbs, nImpOrbs, 2))
            f['downfold_1e/iter{}/Vhf_imp_sIab'.format(it_1e_)].attrs["__complex__"] = np.int8(1)

    @staticmethod
    def hf_shift(Dm_sIab, U_chem):
        print("Compute HF shift for the impurity self-energy via HF equations.\n")
        sys.stdout.flush()
        F_sIab = np.zeros(Dm_sIab.shape, dtype=complex)
        nspin, nImp = Dm_sIab.shape[:2]
        # Coulomb term
        spin_factor = 2 if nspin == 1 else 2
        for s in range(nspin):
            for I in range(nImp):
                F_sIab[0, I] += spin_factor * np.einsum('dc,abcd->ab',
                                                       Dm_sIab[s, I], U_chem)
        if nspin == 2:
            F_sIab[1] = F_sIab[0]

        # Exchange term
        for s in range(nspin):
            for I in range(nImp):
                F_sIab[s, I] -= np.einsum('cd,acdb->ab', Dm_sIab[s, I], U_chem)

        return F_sIab

    @staticmethod
    def generate_parameter_file(param_file, NSITES, NSPINS, INPUT_FILE, OUTPUT_FILE, section_headers=None):
        """
        Generates a parameter file with the specified values
        :param param_file: str
            Name of the parameter file.
        :param NSITES: int
            Number of sites.
        :param NSPINS: int
            Number of spins.
        :param INPUT_FILE: str
            Input h5 file name.
        :param OUTPUT_FILE: str
            Output h5 file name.
        :param section_headers: dict, optional
            Additional sections and their parameters
        """
        with open(param_file, 'w') as file:
            # Write the basic parameters
            file.write(f'NSITES={NSITES}\n')
            file.write(f'NSPINS={NSPINS}\n')
            file.write(f'INPUT_FILE={INPUT_FILE}\n')
            file.write(f'OUTPUT_FILE={OUTPUT_FILE}\n')
            file.write('\n')

            # Write sections if any
            if section_headers:
                for section, params in section_headers.items():
                    file.write(f'[{section}]\n')
                    for key, value in params.items():
                        file.write(f'{key}={value}\n')
                    file.write('\n')

    @staticmethod
    def write_ed_input(prefix, bath_energy, bath_hyb,
                       t, U,
                       beta, nlanc, boltzmann_cutoff, niw,
                       NEV, NCV, truncate_decimal=None, occ_sector=None):
        """
        Write input files for EDLib (https://github.com/Q-solvers/EDLib/tree/master)

        WARNING: The imaginary part of the impurity Hamiltonian is neglected!

        :param prefix: str
            Prefix for input files: prefix.edlib.input.h5 and prefix.param
        :param bath_energy: numpy.ndarray(dim=2)
            Energies for the bath orbitals: (nspin, nbath)
        :param bath_hyb: numpy.ndarray(dim=3)
            Coupling between the bath and the impurity orbitals: (nspin, nbath, nImpOrbs)
        :param t: numpy.ndarray(dim=3)
            One-body Hamiltonian for the impurity orbitals: (nspin, nImpOrbs, nImpOrbs)
        :param U: numpy.ndarray(dim=4)
            Coulomb interactions for the impurity orbitals: (nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs)
        :param beta: float
        :param boltzmann_cutoff: float
        :param niw: int
        :param NEV: int
        :param NCV: int
        :param truncate_decimal: int
        """
        ns, nbath, nImpOrbs = bath_hyb.shape

        ed_param = prefix + ".param"
        ed_input = prefix+".edlib.input.h5"
        ed_output= prefix+".edlib.sim.h5"
        restrict_sector = True if occ_sector is not None else False

        # EDLib parameter file
        param_sec = {
            'storage': {
                'MAX_DIM': '143325',             # Maximum dimension of the Hamiltonian matrix.
                'MAX_SIZE': '80000000',          # Maximum size of the matrix arrays. Must be between MAX_DIM and MAX_DIM^2.
                'EIGENVALUES_ONLY': '0'          # Compute only eigenvalues.
            },
            'arpack': {
                'NEV': NEV,
                'NCV': NCV,
                'SECTOR': str(restrict_sector).upper()     # Read symmetry sectors from file
            },
            'lanc': {
                'NLANC': nlanc,
                'BOLTZMANN_CUTOFF': boltzmann_cutoff,
                'NOMEGA': niw,
                'BETA': beta
            },
            'siam': {
                'NORBITALS': nImpOrbs
            }
        }
        EDLib.generate_parameter_file(ed_param, nbath+nImpOrbs, 2, ed_input, ed_output,
                                      section_headers=param_sec)

        # EDLib input, including the impurity Hamiltonian.
        edlib_h5 = h5py.File(ed_input, "w")

        if ns == 2:
            t_imp = np.einsum("sij->ijs", t).real
            bath_hyb_ = np.einsum("sbi->ibs", bath_hyb).real
            bath_energy_ = np.einsum("sb->bs", bath_energy).real
        else:
            # duplicate for spin degeneracy
            t_imp = np.array([t[0], t[0]])
            bath_hyb_ = np.array([bath_hyb[0], bath_hyb[0]])
            bath_energy_ = np.array([bath_energy[0], bath_energy[0]])
            t_imp = np.einsum("sij->ijs", t_imp).real
            bath_hyb_ = np.einsum("sbi->ibs", bath_hyb_).real
            bath_energy_ = np.einsum("sb->bs", bath_energy_).real

        if truncate_decimal is not None:
            t_imp = t_imp.round(decimals=truncate_decimal)
            bath_hyb_ = bath_hyb_.round(decimals=truncate_decimal)
            bath_energy_ = bath_energy_.round(decimals=truncate_decimal)

        bath = edlib_h5.create_group("Bath")
        bath["Epsk/values"] = bath_energy_
        for i in range(nImpOrbs):
            Vk_g = bath.create_group("Vk_"+str(i))
            Vk_g.create_dataset("values", data=bath_hyb_[i], dtype=float)
            H0_g = edlib_h5.create_group("H0_"+str(i))
            H0_g.create_dataset("values", data=t_imp[i], dtype=float)

        # ERIs in physicists' notation
        U_ = np.einsum("ijkl->ikjl", U).real
        if truncate_decimal is not None:
            U_ = U_.round(decimals=truncate_decimal)
        UU = np.zeros((2, 2) + U_.shape)
        UU[0, 0] = U_
        UU[0, 1] = U_
        UU[1, 0] = U_
        UU[1, 1] = U_
        int_g = edlib_h5.create_group("interaction")
        int_g.create_dataset("values", shape=(2, 2, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs,), data=UU, dtype=float)

        if occ_sector is not None:
            occ_bath_up = len(np.where(bath_energy_[:,0]<0)[0])
            occ_bath_down = len(np.where(bath_energy_[:,1]<0)[0])
            occ_sector[:,0] += occ_bath_up
            occ_sector[:,1] += occ_bath_down
            print("Number of occupied bath orbital: spin 0:  {}; spin 1: {}".format(occ_bath_up, occ_bath_down))
            sys.stdout.flush()
            sec_g = edlib_h5.create_group("sectors")
            sec_g.create_dataset("values", data=occ_sector)

        edlib_h5.create_dataset("mu", shape=(), data=0.0, dtype=float)
        if nImpOrbs > 1:
            orbitals = []
            for i in range(nImpOrbs):
                for j in range(nImpOrbs):
                    if i == j:
                        continue
                    orbitals.append([i, j])
            edlib_h5["GreensFunction_orbitals/values"] = np.array(orbitals)


if __name__ == "__main__":
    from py2aimb.aimbes import AIMBES
    from py2aimb.dmft import solver

    edlib_bin = "/mnt/home/cyeh/Projects/EDLib/build/build_rome/examples/anderson-example"
    ed_command = "mpirun --map-by socket:pe=1 {}".format(edlib_bin)

    aimb_h5 = "qpgw_ed.mbpt.h5"

    my_aimb = AIMBES(aimb_h5)

    occ_sector = np.array([[1,0],[0,1],])
    solver.EDLib.solve(my_aimb, ed_prefix="svo", ed_cmd=ed_command,
                       nbath=3, wmax=10,
                       NEV=10, NCV=60,
                       nlanc=100, boltzmann_cutoff=1e-12, truncate_decimal=6,
                       occ_sector=occ_sector, mixing=0.7)

