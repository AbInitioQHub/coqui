/**
 * ==========================================================================
 * CoQuí: Correlated Quantum ínterface
 *
 * Copyright (c) 2022-2025 Simons Foundation & The CoQuí developer team
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


#ifndef COQUI_SCF_COMMON_HPP
#define COQUI_SCF_COMMON_HPP

#include "configuration.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"

#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "methods/SCF/qp_params_t.h"
#include "methods/SCF/mb_solver_t.h"
#include "methods/mb_state/mb_state.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"
#include "numerics/shared_array/nda.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace methods {
  // TODO Put everything in "scf" namespace to isolate using directives
template<nda::Array Array_base_t>
using sArray_t = math::shm::shared_array<Array_base_t>;
using Array_view_3D_t = nda::array_view<ComplexType, 3>;
using Array_view_4D_t = nda::array_view<ComplexType, 4>;
using Array_view_5D_t = nda::array_view<ComplexType, 5>;

template<math::shm::SharedArray sArr_t>
auto distributed_tau_to_w(mpi3::communicator& comm,
                          const sArr_t& X_tau_shm,
                          const imag_axes_ft::IAFT& FT,
                          std::array<long, 5> w_grid, std::array<long, 5> w_bsize={0})
  requires( ::nda::get_rank<std::decay_t<sArr_t>> == 5 ) {
  decltype(nda::range::all) all;
  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;
  auto [nts, ns, nkpts, nbnd, _] = X_tau_shm.shape();
  auto nw = FT.nw_f();

  if (nda::sum(X_tau_shm.local()) != ComplexType(0.0))
    FT.check_leakage(X_tau_shm, imag_axes_ft::fermion, "self-energy");

  int np = comm.size();
  long nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.2);
  np /= nkpools;
  long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
  long np_j = np / np_i;

  auto dX_wskij = make_distributed_array<Array_5D_t>(comm, {1, 1, nkpools, np_i, np_j},
                                                     {nw, ns, nkpts, nbnd, nbnd});
  auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dX_wskij.local_shape();
  auto k_rng = dX_wskij.local_range(2);
  auto i_rng = dX_wskij.local_range(3);
  auto j_rng = dX_wskij.local_range(4);

  {
    nda::array<ComplexType, 5> X_t_sub(nts, ns, nk_loc, ni_loc, nj_loc);
    X_t_sub = X_tau_shm.local()(all, all, k_rng, i_rng, j_rng);

    auto X_w_loc = dX_wskij.local();
    FT.tau_to_w(X_t_sub, X_w_loc, imag_axes_ft::fermion);
  }

  auto dX_wskij_out = make_distributed_array<Array_5D_t>(
      comm, w_grid, {nw, ns, nkpts, nbnd, nbnd}, w_bsize);
  math::nda::redistribute(dX_wskij, dX_wskij_out);
  return dX_wskij_out;
}

template<nda::MemoryArray Array_t>
void hermitize(Array_t &&A_ij, std::string name="") {
  constexpr int rank = ::nda::get_rank<Array_t>;
  using value_type = typename std::decay_t<Array_t>::value_type;
  if (name != "")
    app_log(2, "Explicitly make {} Hermitian.", name);
  utils::check(A_ij.shape(rank-1)==A_ij.shape(rank-2),
               "hermitize: Incorrect dimensions of {} for the last axes = ({}, {})",
               name, A_ij.shape(rank-2), A_ij.shape(rank-1));
  long nbnd = A_ij.shape(rank-1);
  long dim0 = std::accumulate(A_ij.shape().begin(), A_ij.shape().end()-2, 1, std::multiplies<>{});
  auto A_Iij = nda::reshape(A_ij, std::array<long,3>{dim0,nbnd,nbnd});
  nda::array<value_type, 2> buffer(nbnd, nbnd);
  for (size_t I=0; I<dim0; ++I) {
    buffer = ( A_Iij(I,nda::ellipsis{}) + nda::conj(nda::transpose(A_Iij(I,nda::ellipsis{}))) ) * 0.5;
    A_Iij(I,nda::ellipsis{}) = buffer;
  }
}

auto get_mf_MOs(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, hamilt::pseudopot &psp)
  -> std::tuple<sArray_t<Array_view_4D_t>, sArray_t<Array_view_3D_t> >;

void update_MOs(sArray_t<Array_view_4D_t> &sMO_skij, sArray_t<Array_view_3D_t> &sE_ski,
                const sArray_t<Array_view_4D_t> &sF_skij, const sArray_t<Array_view_4D_t> &sS_skij);

void update_Dm(sArray_t<Array_view_4D_t> &sDm_skij,
               const sArray_t<Array_view_4D_t> &sMO_skij, const sArray_t<Array_view_3D_t> &sE_ski,
               const double mu, const double beta);

void update_G(sArray_t<Array_view_5D_t> &sG_tskij,
              const sArray_t<Array_view_4D_t> &sMO_skia, const sArray_t<Array_view_3D_t> &sE_ska,
              double mu, const imag_axes_ft::IAFT &FT);

template<nda::ArrayOfRank<5> Array_base_t>
void compute_G_from_mf(h5::group iter_grp, imag_axes_ft::IAFT &ft, sArray_t<Array_base_t> &sG_tskij);

template<typename X_t>
double update_mu(double old_mu, const mf::MF &mf, const X_t &sE_ski, double beta, double mu_tol=1e-9);

double compute_Nelec(double mu, const mf::MF &mf, const sArray_t<Array_view_3D_t> &sE_ski, double beta);

/**
 * Given a dynamic self-energy, solve the quasi-particle equation and update sE_ska in-place
 * @param sE_ska        - [INPUT] initial qp energies
 *                        [OUTPUT] updated qp energies
 * @param sSigma_tskij  - [INPUT] dynamic self-energy on imaginary-time axis
 * @param sVhf_skij     - [INPUT] Hartree-Fock Hamiltonian (including H0)
 * @param sMO_skia      - [INPUT] MO coefficients
 * @param mu            - [INPUT] chemical potential
 * @param FT            - [INPUT] Fourier transform driver on imaginary axes
 * @param qp_params    - [INPUT] setups for quasiparticle eqn
 */
void solve_qp_eqn(sArray_t<Array_view_3D_t> &sE_ska,
                  const sArray_t<Array_view_5D_t> &sSigma_tskij,
                  const sArray_t<Array_view_4D_t> &sVhf_skij,
                  const sArray_t<Array_view_4D_t> &sMO_skia,
                  double mu,
                  const imag_axes_ft::IAFT &FT, qp_params_t &qp_params);

/**
 * Given a dynamic self-energy in the primary basis, return the static approximation (Phys. Rev. Lett. 96, 226402 (2006)) of it
 * (still in the primary basis):
 *     Sigma_tskij -> Sigma_tskab -> Vcorr_skab -> Vcorr_skij
 * mode "qp_energy": off_diagonals are approximated as
 *     V_corr_ab = 0.25 * [ V_corr_ab(e_a) + V_corr_ab(e_b) + V_corr_ba(e_b)* + V_corr_ba(e_a)* ]
 * mode "fermi": off-diagonals are approximated by V_corr(w=0)
 * @param sSigma_tskij - [INPUT] dynamic self-energy
 * @param sMO_skia     - [INPUT] MO coefficients
 * @param sE_ska       - [INPUT] quasi-particle energies
 * @param mu           - [INPUT] chemical potential
 * @param FT           - [INPUT] Fourier transform driver on imaginary axes
 * @param qp_params   - [INPUT] setups for quasi-paritcle eqn, including mode "qp_energy" or mode "fermi"
 * @return static approximation to the dynamic self-energy in the primary basis
 */
auto qp_approx(const sArray_t<Array_view_5D_t> &sSigma_tskij,
               const sArray_t<Array_view_4D_t> &sMO_skia,
               const sArray_t<Array_view_3D_t> &sE_ska, double mu,
               const imag_axes_ft::IAFT &FT, qp_params_t &qp_params)
  -> sArray_t<Array_view_4D_t>;

/**
 * Given a correlated solver, compute the dynamic self-energy and
 * solve the quasi-particle Eqn in the presence of correlated potential V_corr(w) in MO basis
 * Specifically, this functions does the following:
 * 1. compute dynamic self-energy from 'corr_solver'
 * 2. compute diagonals of V_corr(iw) in MO basis
 * 3. solve the quasiparticle equations for e_qp with V_corr(w)
 * 4. compute the new Heff using the new e_qp and MO coefficients
 * 5. update e_qp and Heff in-place
 * @tparam eri_t         - THC-ERI
 * @tparam corr_solver_t - correlated solver
 * @param mb_state       - [INPUT/OUTPUT] many-body state whose contents will be updated 
 * @param mu             - [INPUT] chemical potential
 * @param mb_solver      - [INPUT] many-body solver context
 * @param eri            - [INPUT] THC-ERI instance
 * @param FT             - [INPUT] Fourier transform driver on imaginary axes
 * @param qp_params     - [INPUT] setups for quasi-paritcle eqn
 * @param fixed_w        - [INPUT] whether to keep the screened interactions fixed from mb_state. 
 *                         If false, the dynamically screened interaction will be updated at each iteration.
 */
template<typename eri_t, typename corr_solver_t>
void add_evscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params, 
                     bool fixed_w=false);
/**
 * Add correlated potential V_corr with static approximation to the effective quasiparticle Hamiltonian
 * The static approximation follows T. Kotani et. al., Phys. Rev. B 76, 165106 (2007) in which
 * mode "qp_energy" (mode A in Kotani's work): off_diagonals are approximated as
 *     V_corr_ab = 0.25 * [ V_corr_ab(e_a) + V_corr_ab(e_b) + V_corr_ba(e_b)* + V_corr_ba(e_a)* ]
 * mode "fermi" (mode B in Kotani's work): off-diagonals are approximated by V_corr(w=0)
 * @tparam keep_W        - whether to keep the screened interactions from corr_solver_t fixed
 * @tparam eri_t         - THC-ERI
 * @tparam corr_solver_t - correlated solver
 * @param mb_state       - [INPUT/OUTPUT] many-body state whose contents will be updated
 * @param corr_solver    - [INPUT] correlated solver instance
 * @param eri            - [INPUT] THC-ERI instance
 * @param FT             - [INPUT] Fourier transform driver on imaginary axes
 * @param mu             - [INPUT] chemical potential
 * @param qp_params     - [INPUT] setups for quasiparitcle eqn
 * @return new qp energies in share memory sE_ska(ns, nkpts, nbnd)
 */
template<typename eri_t, typename corr_solver_t>
void add_qpscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params);

template<typename function_t>
double qp_eqn_linearized(double Vhf, function_t &Sigma, long I, double mu, double eps_ks, double eta = 0.0);
template<typename function_t>
std::tuple<double,double,bool> qp_eqn_secant(double Vhf, function_t &Sigma, long I, double mu, double eps0, int maxiter = 100, double tol = 1e-6, double eta = 0.0);
template<typename function_t>
std::tuple<double,double> qp_eqn_bisection(double Vhf, function_t &Sigma, long I, double mu, double eps0, double tol = 1e-6, double eta = 0.0);
template<typename function_t>
std::tuple<double,bool> qp_eqn_spectral(double Vhf, function_t &Sigma, long I, double mu, double eps_ks, double tol = 1e-6, double eta = 0.0);


/** Dyson-SCF utilities **/

/**
 * Update Green's function
 */
template<typename dyson_type, typename X_t, typename Xt_t>
void update_G(dyson_type &dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT,
              X_t & Dm, Xt_t &G, const X_t & F, const Xt_t &Sigma, double &mu,
              bool const_mu = false);

/**
 * Update chemical potential (_mu) for current _F and _Sigma
 * Solve f(mu) = nelec(mu) - nelec_target = 0 using bisection method
 */
template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu(double old_mu, dyson_type& dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT,
                 const X_t&F, const Xt_t&G, const Xt_t&Sigma);

/**
 * Compute the total number of electrons as a function of mu and spectra
 * @param mu - [INPUT] chemical potential
 * @param spectra - [INPUT] eigenvalues of F + Sigma(iwn)
 * @return - Nelec
 */
double compute_Nelec(double mu, const nda::array<ComplexType, 4> &spectra,
                     const mf::MF &mf, const imag_axes_ft::IAFT &FT);

/**
 * Iterative solver for the SCF solution of effective quasiparticle Hamiltonian.
 * @param context     - [INPUT] mpi context
 * @param iter_solver - [INPUT] iterative solver
 * @param it          - [INPUT] current iteration
 * @param h5_prefix   - [INPUT] prefix for h5 archive that stores QP results from previous iterations
 * @param sHeff_skij  - [INPUT] quasiparticle Hamiltonian at the current iteration
 * @return - maximum norm of the SCF error
 */
template<typename comm_t, typename X_t>
double solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                     long it, std::string h5_prefix, X_t &sHeff_skij);
/**
 * Iterative solver for the SCF solution of self-energy
 * @param context      - [INPUT] mpi context
 * @param iter_solver  - [INPUT] iterative solver
 * @param it           - [INPUT] current iteration
 * @param h5_prefix    - [INPUT] prefix for h5 archive that stores QP results from previous iterations
 * @param sF_skij      - [INPUT] static self-energy at the current iteration
 * @param sSigma_tskij - [INPUT] dynamic self-energy at the current iteration
 * @param FT           - [INPUT] Fourier transform driver on imaginary axes
 * @param restart      - [INPUT] whether this is a restart SCF
 * @return - maximum norm of the SCF error for F and Sigma
 */
template<typename comm_t, typename X_t, typename Xt_t>
auto solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                     long iteration, std::string h5_prefix, X_t &sF_skij, Xt_t &sSigma_tskij,
                     const imag_axes_ft::IAFT *FT,
                     std::array<std::string,3> dataset={"scf", "F_skij", "Sigma_tskij"})
  -> std::tuple<double, double>;

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto damping_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
                  long iteration, std::string h5_prefix, X_t &sF_skij, Xt_t &sSigma_tskij,
                  std::array<std::string,3> datasets={"scf", "F_skij", "Sigma_tskij"})
-> std::tuple<double, double>;

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto diis_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
               long iteration, std::string h5_prefix, X_t &sF_skij, Xt_t &sSigma_tskij,
               const imag_axes_ft::IAFT *FT,
               std::array<std::string,3> datasets={"scf", "F_skij", "Sigma_tskij"})
-> std::tuple<double, double>;

template<typename comm_t, typename X_t, nda::ArrayOfRank<1> Array1D>
double eval_corr_energy(comm_t& comm, const imag_axes_ft::IAFT &FT,
                        const X_t & G_shm, const X_t & Sigma_shm,
                        Array1D &k_weight);
template<typename X_t, nda::ArrayOfRank<1> Array1D>
auto eval_hf_energy(const X_t &sDm_skij, const X_t &sF_skij, const X_t &sH0_skij,
                    Array1D &k_weight, bool F_has_H0=true)
                    -> std::tuple<double, double>;

template<typename dyson_type>
void write_mf_data(mf::MF &mf, const imag_axes_ft::IAFT &ft, dyson_type &dyson,
                   std::string output);
void write_mf_data(mf::MF &mf, const imag_axes_ft::IAFT &ft, hamilt::pseudopot &,
                   std::string output);

/**
 * Read the Green's function of a given iteration from a AIMBES checkpoint file.
 * @param context  - [INPUT] mpi context
 * @param filename - [INPUT] file name of the check file
 * @return std::tuple(iteration_number, Green's function in shared memory)
 */
template<typename MPI_Context_t>
auto read_greens_function(MPI_Context_t &context, mf::MF *mf,
                          std::string filename, long scf_iter=-1, std::string scf_grp = "scf")
-> sArray_t<Array_view_5D_t>;

}
#endif // COQUI_SCF_COMMON_HPP
