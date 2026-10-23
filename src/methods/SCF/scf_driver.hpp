/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2026 Simons Foundation & The CoQuí developer team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 * 
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * ==========================================================================
 */


#ifndef COQUI_SCF_DRIVER_HPP
#define COQUI_SCF_DRIVER_HPP

#include "mpi3/communicator.hpp"
#include "mpi3/core.hpp"
#include "utilities/Timer.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"

#include "utilities/mpi_context.h"
#include "methods/SCF/qp/qp_context_t.h"
#include "methods/SCF/scf_common.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/mb_state/mb_state.hpp"

namespace methods {
/**
 * Generic self-consistent loop for full-frequency Green's function methods. 
 * The template parameters allow for different types of self-energy solvers (e.g. HF, GW, GF2) 
 * and different types of Coulomb Hamiltonian representations (e.g. THC, Cholesky).
 */
template<typename dyson_type, typename eri_t, typename corr_solver_t>
auto scf_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
              solvers::mb_solver_t<corr_solver_t> mb_solver, iter_scf::iter_scf_t *iter_solver = nullptr,
              int niter = 1, bool restart = false, double conv_tol = 1e-9, bool const_mu = false,
              std::string input_grp = "scf", int input_iter = -1, bool eval_thermodynamics = false)
              -> std::tuple<double, double>;

/**
 * Self-consistent loop on a static quasiparticle Hamiltonian Heff. Each iteration builds G from
 * the current Heff, evaluates the self-energy, reduces it to a new Heff, mixes it with the
 * previous ones (iter_solver), re-diagonalizes, updates mu and the density, and writes
 * scf/iter{N} (MBState::dump_qpscf).
 *
 * qp_params selects the scheme along two independent axes:
 *   qp_scf_mode = "qpscf", qp_approx = "qp_eqn": qpGW
 *   qp_scf_mode = "evscf", qp_approx = "qp_eqn": evGW
 *   qp_scf_mode = "qpscf", qp_approx = "lqp"   : LQSGW, the linearized Sigma
 *   qp_scf_mode = "evscf", qp_approx = "lqp"   : not implemented, rejected
 * With mb_solver.corr == nullptr (HF) no self-energy is added and the axes are irrelevant.
 *
 * The template parameters allow for different types of self-energy solvers (e.g. HF, GW, GF2)
 * and different types of Coulomb Hamiltonian representations (e.g. THC, Cholesky).
 * @param mb_state    - [INPUT/OUTPUT] many-body state; Heff, MO, E and Dm are (re)allocated here
 * @param mb_eri      - [INPUT] ERI bundle (Hartree-Fock and correlated parts)
 * @param FT          - [INPUT] imaginary-axis transform driver
 * @param qp_params   - [INPUT] the two axes and each family's controls (qp_context_t), plus the
 *                      chemical-potential update and keep_scr_coulomb_fixed
 * @param mb_solver   - [INPUT] HF solver and, beyond HF, the correlated and screening solvers
 * @param iter_solver - [INPUT] mixing of Heff between iterations (damping or DIIS)
 * @param niter       - [INPUT] maximum number of iterations
 * @param restart     - [INPUT] continue from the last iteration in the checkpoint
 * @param conv_tol    - [INPUT] stop when the largest change of Heff falls below it
 * @return the total energy of the last iteration [Ha]
 */
template<typename eri_t, typename corr_solver_t>
double qp_scf_loop(MBState &mb_state, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                   qp_context_t &qp_params, solvers::mb_solver_t<corr_solver_t> mb_solver,
                   iter_scf::iter_scf_t *iter_solver = nullptr, int niter = 1,
                   bool restart = false, double conv_tol = 1e-8);

/**
 * RPA energy functional loop.
 */
template<typename eri_t, typename dyson_type>
double rpa_loop(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri, const imag_axes_ft::IAFT& FT,
                solvers::mb_solver_t<solvers::gw_t> mb_solver);
} // namespace methods

#endif // COQUI_SCF_DRIVER_HPP
