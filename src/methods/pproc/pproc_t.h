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


#ifndef COQUI_PPROC_T_HPP
#define COQUI_PPROC_T_HPP

#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "numerics/ac/AC_t.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/SCF/qp/qp_params_t.h"
#include "methods/SCF/qp/qp_solvers.hpp"
#include "methods/SCF/qp/linearized_qp.hpp"

namespace methods {
  namespace mpi3 = boost::mpi3;

  // TODO This should not be a class! Separate these into free functions.
  // TODO Useful features:
  //      1. band-gap estimator
  /**
   * A proxy for different post-processing steps after a mbpt calculation.
   * The mbpt solution is given by reading the bdft h5 output file: outdir/prefix.mbpt.h5
   */
  
  class pproc_t {
  public:
    pproc_t(utils::mpi_context_t<mpi3::communicator> &context, std::string prefix, std::string outdir):
        _context(context), _scf_output(outdir+"/"+prefix) {

      for (auto& v: {"READ", "WRITE", "AC"}) {
        _Timer.add(v);
      }

    }

    /**
     * Perform analytical continuation
     * @param mf - [INPUT] a mean-field instance for metadata of the system
     * @param ac_context - [INPUT] parameters for ac
     * @param dataset - [INPUT] dataset for ac
     */
    void analyt_cont(mf::MF &mf, analyt_cont::ac_context_t &ac_context, std::string dataset="G_tskij");
    /**
     * Perform Wannier interpolation to the mbpt solutions on the provided k-points
     * @param mf - [INPUT] a mean-field instance for all the metadata of the system
     * @param project_file - [INPUT] a h5 file which stores the projection matrices and the target k-points
     * @param target - [INPUT] type of the mbpt calculation: quasiparticle or dyson
     */
    void wannier_interpolation(mf::MF &mf, ptree const& pt, std::string project_file, std::string target,
                               std::string grp_name="scf", long iter=-1, 
                               bool translate_home_cell=false);
    /**
     * Wannier interpolation plus analytical continuation for spectral functions on the provided k-points.
     * Also computes quasiparticle energies on the IBZ k-mesh from the dynamic self-energy and
     * Wannier-interpolates them along the k-path.
     * (Only for dyson-type calculation)
     * @param mf - [INPUT] a mean-field instance for all the metadata of the system
     * @param project_file - [INPUT] a h5 file which stores the projection matrices and the target k-points
     * @param ac_params - [INPUT] parameters for ac
     * @param qp_params - [INPUT] parameters for quasiparticle equation solver
     */
    void spectral_interpolation(mf::MF &mf, ptree const& pt, std::string project_file,
                                analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);

    void local_density_of_state(mf::MF &mf, std::string project_file, analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);
            
    /**
     * Compute quasiparticle energies on the IBZ k-mesh by solving the QP equation
     * E_a = F_aa + Re[Sigma_aa(E_a - mu)] for each (spin, k, band).
     * Writes E_ska to {grp_name}/iter{N}/qp_approx/E_ska in the checkpoint file.
     */
    void compute_qp_on_ibz_kmesh(mf::MF &mf, const qp_params_t &qp_params, 
                                std::string grp_name="scf", long iter=-1);

    /**
     * Post-processing entry point of the linearized quasiparticle approximation on arrays the
     * caller already holds -- no checkpoint, no MPI distribution. It is the same kernel and the
     * same driver as the LQSGW loop and the checkpoint post-processing (methods::lqp), run
     * serially -- no MPI is required to be initialized -- with on_failure_e::report: a point that fails the residual gate or the causality
     * check is left zero and flagged in result_t::status_sk instead of aborting, because in an
     * analysis over many k-points one bad point must not hide the rest.
     *
     * The Python API (coqui.post_proc.linearized_qp) is built on this; it also uses it on
     * Wannier-window quantities, which have no checkpoint layout of their own.
     *
     * @param F_skab      - [INPUT] static one-body matrix (ns, nk, n, n), **including H0**
     * @param Sigma_tskab - [INPUT] dynamic self-energy (nt, ns, nk, n, n) on ft's fermionic tau mesh
     * @param mu          - [INPUT] chemical potential [Ha]
     * @param ft          - [INPUT] imaginary-axis transform driver Sigma was sampled with
     * @param p           - [INPUT] fit controls
     * @return methods::lqp::result_t for every (s,k); consult status_sk before trusting a point
     */
    static lqp::result_t linearized_qp(nda::array_const_view<ComplexType, 4> F_skab,
                                       nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                       double mu, imag_axes_ft::IAFT const& ft,
                                       lqp::fit_params_t const& p);

    /**
     * Same, with the IAFT built from its defining parameters, for a caller that has none to pass:
     * the Python binding, whose IAFT for basis = "ir" is a sparse_ir object with no C++
     * counterpart. Sigma_tskab must then be sampled on the tau mesh this IAFT generates. For
     * "ir" that means the sparse_ir mesh must equal the C++ tables', which is pinned by the unit
     * test src/python/utils/imag_axes_ft/tests/test_iaft.py (1e-12) rather than checked here.
     * @param beta, wmax, basis, prec - [INPUT] the IAFT constructor arguments
     */
    static lqp::result_t linearized_qp(nda::array_const_view<ComplexType, 4> F_skab,
                                       nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                       double mu, double beta, double wmax,
                                       std::string const& basis, std::string const& prec,
                                       lqp::fit_params_t const& p);

    /**
     * The convergence rule on arrays (lqp::linearized_qp_ladder), serial and reporting, with the
     * IAFT built from its parameters like linearized_qp() above: climb the exactly-determined
     * ladder n_fit = 2, 3, ... until the residual gate fires or the sampling mesh ends, return the
     * last accepted rung with its truncation-error estimate. What the Python API runs by default.
     * @param n_fit_max - [INPUT] highest rung to try (capped at the mesh)
     * @param p         - [INPUT] fit controls; fit_resid_tol is the gate, symmetric_window is used
     */
    static lqp::ladder_result_t linearized_qp_ladder(nda::array_const_view<ComplexType, 4> F_skab,
                                                     nda::array_const_view<ComplexType, 5> Sigma_tskab,
                                                     double mu, double beta, double wmax,
                                                     std::string const& basis, std::string const& prec,
                                                     lqp::fit_params_t const& p, int n_fit_max);

  private:
    /**
     * qp_approx == "lqp" branch of compute_qp_on_ibz_kmesh: matrix linearization of Sigma(iw)
     * around w = 0 in the KS basis:
     *  1. K = F + Sigma(0) - mu,
     *  2. Z = (1 - dSigma/d(iw))^-1
     *  3. H_QP = Z^1/2 K Z^1/2, 
     * solved independently at every (s,k).
     * 
     * Writes to qp_approx/{E_ska, Heff_skij, Z_ska, mu, scheme} and
     * qp_approx/lqp/{min_eig_sk, fit_resid_sk, n_fit, fit_order, cond}.
     *
     * @param FT           - [INPUT] Fourier transform driver on the imaginary axes, as read from
     *                       the checkpoint; supplies the tau and Matsubara meshes and beta
     * @param qp_params    - [INPUT] only the lqp member is used; see methods::lqp::fit_params_t
     * @param sFhf_skij    - [INPUT] static one-body matrix (ns, nk, nb, nb) in the KS basis,
     *                       **including H0** (the caller adds system/H0_skij to F_skij)
     * @param sSigma_tskij - [INPUT] dynamic self-energy (nt, ns, nk, nb, nb) on FT's fermionic
     *                       tau mesh, same basis as sFhf_skij
     * @param mu           - [INPUT] chemical potential [Ha] of the iteration being processed
     * @param filename     - [INPUT] checkpoint the results are appended to
     * @param grp_name     - [INPUT] top-level group, "scf" or "embed"
     * @param iter         - [INPUT] iteration index; the results go to {grp_name}/iter{iter}
     */
    void lqp_on_ibz_kmesh_impl(imag_axes_ft::IAFT const& FT, qp_params_t const& qp_params,
                                 math::shm::shared_array<nda::array_view<ComplexType, 4>> const& sFhf_skij,
                                 math::shm::shared_array<nda::array_view<ComplexType, 5>> const& sSigma_tskij,
                                 double mu, std::string filename, std::string grp_name, long iter);

    /**
     * qp_approx == "qp_eqn" branch (qp_eqn.solver in {"sc", "sc_bisection", "sc_newton", "linearized"}) of
     * compute_qp_on_ibz_kmesh: 
     * 1. diagonalize F, 
     * 2. analytically continue the diagonal Sigma_aa(iw) by Pade, 
     * 3. solve the scalar quasiparticle equation on the real axis band by band 
     *    (at the quasiparticle energy for the "sc*" variants, by first-order expansion 
     *     around eps_KS for "linearized").
     * 
     * Writes qp_approx/{E_ska, Heff_skij, mu, scheme, ac_solver}.
     *
     * @param FT           - [INPUT] Fourier transform driver on the imaginary axes
     * @param qp_params    - [INPUT] only the ac member is used; see qp_eqn_params_t
     * @param sFhf_skij    - [INPUT] static one-body matrix (ns, nk, nb, nb) including H0
     * @param sSigma_tskij - [INPUT] dynamic self-energy (nt, ns, nk, nb, nb) on the tau mesh
     * @param mu           - [INPUT] chemical potential [Ha]
     * @param filename     - [INPUT] checkpoint the results are appended to
     * @param grp_name     - [INPUT] top-level group, "scf" or "embed"
     * @param iter         - [INPUT] iteration index
     */
    void qp_ac_on_ibz_kmesh_impl(imag_axes_ft::IAFT const& FT, qp_params_t const& qp_params,
                                 math::shm::shared_array<nda::array_view<ComplexType, 4>> const& sFhf_skij,
                                 math::shm::shared_array<nda::array_view<ComplexType, 5>> const& sSigma_tskij,
                                 double mu, std::string filename, std::string grp_name, long iter);

    template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    void read_scf_dataset(std::string dataset,
                          memory::darray_t<local_Array_4D_t, communicator_t> &A_tski);

    /* Read full dataset without extracting a diagonal
     * from a hdf5 group.
     * The function is supposed to work for "scf" and "system" groups 
     * and with 5D and 4D respectively.
     */
    template<nda::MemoryArray local_Array_t, typename communicator_t>
    void read_scf_dataset_full(std::string dataset, std::string group,
                                 memory::darray_t<local_Array_t, communicator_t> &A);

    template<nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    void dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                        memory::darray_t<local_Array_4D_t, communicator_t> &dA_out,
                        nda::array<ComplexType, 1> &iw_mesh,
                        memory::darray_t<local_Array_4D_t, communicator_t> &dA_in,
                        std::string dataset, std::string grp_name="scf", int iter=-1);

    template<nda::MemoryArray local_Array_t>
    void dump_ac_output(nda::array<ComplexType, 1> &w_mesh,
                        local_Array_t &A_out,
                        nda::array<ComplexType, 1> &iw_mesh,
                        local_Array_t& A_in,
                        std::string dataset, std::string grp_name="scf", int iter=-1);

    template<nda::ArrayOfRank<5> local_Array_5D_t, nda::ArrayOfRank<4> local_Array_4D_t, typename communicator_t>
    auto evaluate_GS_diag(memory::darray_t<local_Array_5D_t, communicator_t> & dG_tau_skij,
                          memory::darray_t<local_Array_4D_t, communicator_t> & dS_skij)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

  private:
    utils::mpi_context_t<mpi3::communicator> &_context;
    std::string _scf_output;
    utils::TimerManager _Timer;
  };
} // methods


#endif //COQUI_PPROC_T_HPP
