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
#include "methods/SCF/qp/qp_context_t.h"
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
     * Spectral function of a Dyson-SCF solution on the target k-points of the Wannier projection:
     * 1. Wannier-interpolates {grp_name}/iter{N}/{F_skij, Sigma_tskij} and solves Dyson's equation
     *    there (wannier_interpolation(..., "dyson")), writing {grp_name}/iter{N}/wannier_inter/
     *    {F_skab, Sigma_tskab, G_tskab, G_wskab, kpts, kpt_labels, kpt_label_idx, ...};
     * 2. analytically continues G_wskab to the real axis, writing {grp_name}/iter{N}/ac/.
     * No quasiparticle approximation is applied here: the band structure from a quasiparticle
     * Hamiltonian is band_interpolation's, and the linearized quasiparticle approximation of the
     * interpolated solution is linearized_qp_checkpoint with source = "interpolated", which reads this
     * wannier_inter/ output. (Only for a Dyson-type calculation)
     * @param mf                  - [INPUT] a mean-field instance for all the metadata of the system
     * @param pt                  - [INPUT] input parameters, forwarded to wannier_interpolation
     * @param project_file        - [INPUT] a h5 file which stores the projection matrices and the
     *                              target k-points
     * @param ac_params           - [INPUT] parameters for ac
     * @param grp_name            - [INPUT] top-level group, "scf" or "embed"
     * @param iter                - [INPUT] iteration; -1 = {grp_name}/final_iter
     * @param translate_home_cell - [INPUT] forwarded to wannier_interpolation
     */
    void spectral_interpolation(mf::MF &mf, ptree const& pt, std::string project_file,
                                analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);

    void local_density_of_state(mf::MF &mf, std::string project_file, analyt_cont::ac_context_t &ac_params,
                                std::string grp_name="scf", long iter=-1, bool translate_home_cell=false);
            
    /**
     * Quasiparticle approximation as a post-processing step of a Dyson-SCF solution: 
     * 1. reads the iteration {grp_name}/iter{N} of a full-frequency calculation (scf_loop), 
     * i.e. F_skij, Sigma_tskij and mu, 
     * 2. adds system/H0_skij to F and reduces the dynamic Sigma(tau) to a static
     * quasiparticle Hamiltonian 
     * 3. writes it back to {grp_name}/iter{N}/qp_approx/. 
     *  
     * qp_params.qp_approx selects the QP scheme:
     *   "qp_eqn": diagonal in the KS basis -- analytic continuation of Sigma_aa(iw) and the scalar
     *             quasiparticle equation E_a = F_aa + Re[Sigma_aa(E_a - mu)] band by band
     *             (qp_eqn_from_dyson_impl); Heff_skij = U diag(E_qp) U^dagger.
     *   "lqp"   : the matrix linearization Sigma(iw) ~ A + iw B, one fit at qp_params.lqp.n_fit,
     *             aborting on a failed point (linearized_qp_checkpoint with source = "mesh").
     * Each reads its own inputs into node-shared memory (dimensions from the dataset shapes),
     * writes the controls it used to qp_approx/<scheme>/ (see qp_approx_reusable) and removes
     * the other scheme's results.
     *
     * @param qp_params - [INPUT] qp_approx and the controls of that family; the qp_scf_mode,
     *                    chemical-potential and keep_scr_coulomb_fixed members are not used
     * @param grp_name  - [INPUT] top-level group, "scf" or "embed"
     * @param iter      - [INPUT] iteration; -1 = {grp_name}/final_iter
     */
    void qp_approx_from_dyson(const qp_context_t &qp_params, std::string grp_name="scf",
                              long iter=-1);

    /**
     * Linearized quasiparticle approximation on a Dyson solution stored in a CoQui checkpoint. 
     *
     * The input/output is determined based on the source argument: 
     * 1. source = "mesh": the Dyson-SCF solution on the k-points of the SCF calculation, in
     *    the primary basis.
     *    reads  {grp}/iter{N}/{F_skij, Sigma_tskij, mu} and system/H0_skij (added to F)
     *    writes {grp}/iter{N}/qp_approx/{E_ska, Heff_skij, Zqp_ska, mu, scheme} and
     *           {grp}/iter{N}/qp_approx/lqp/{diagnostics, controls[, ladder/]}
     * 2. source = "interpolated": the Wannier-interpolated Dyson solution on the interpolated
     *    k-points, in the Wannier basis, as written by spectral_interpolation.
     *    reads  {grp}/iter{N}/wannier_inter/{F_skab, Sigma_tskab} (H0 included) and {grp}/iter{N}/mu
     *    writes {grp}/iter{N}/wannier_inter/qp_approx/{E_ska, Heff_skab, Zqp_ska, mu, scheme} and
     *           {grp}/iter{N}/wannier_inter/qp_approx/lqp/{diagnostics, controls[, ladder/]}
     *
     * Both cases write the same layout under the group they read from. The lqp/ group is
     * recreated on every run, so nothing of an earlier run survives in it.
     *
     * @param mpi        - [INPUT] MPI context
     * @param filename   - [INPUT] checkpoint file (read, then appended to)
     * @param source     - [INPUT] "mesh" (use case 1) or "interpolated" (use case 2)
     * @param grp_name   - [INPUT] top-level group, "scf" or "embed"
     * @param iter       - [INPUT] iteration; -1 = {grp}/final_iter
     * @param params     - [INPUT] fit controls; a single fit uses all of them, the ladder only
     *                     fit_resid_tol and symmetric_window
     * @param n_fit_max  - [INPUT] 0: one fit with params; > 0: the exactly-determined ladder up to
     *                     this n_fit
     * @note Aborts on a missing input dataset; a single fit aborts on a point that fails the
     *       residual gate or has 1 - B not positive definite (on_failure_e::abort), and the
     *       ladder, which rejects such a fit and stops, aborts when no fit is accepted or
     *       n_fit_max < 2 (see lqp::linearized_qp_ladder). A written result therefore has no
     *       failed point.
     */
    static void linearized_qp_checkpoint(utils::mpi_context_t<mpi3::communicator>& mpi,
                                         std::string const& filename, std::string const& source,
                                         std::string const& grp_name, long iter,
                                         lqp::params_t const& params, int n_fit_max);

    /**
     * Whether {grp}/iter{N}/qp_approx/ holds a quasiparticle Hamiltonian band_interpolation may
     * reuse for the approximation it was asked for, instead of recomputing it: Heff_skij
     * present, written by the same qp_approx scheme, and every control stored in
     * qp_approx/<scheme>/ equal to the request -- qp_eqn {solver, ac_alg, ac_nfit, eta, tol};
     * lqp {n_fit, fit_order (resolved), symmetric_window, fit_resid_tol}. A ladder result
     * qualifies like a single fit: it stores the setup of its last accepted fit, which is a
     * single fit with that setup. No failed point needs checking: linearized_qp_checkpoint
     * aborts rather than write one. A checkpoint from an earlier CoQui version, which stores no controls, is never
     * reused. When a qp_approx/Heff_skij exists but is not reused, the reason is logged.
     * @param iter_grp  - [INPUT] the open {grp}/iter{N} group
     * @param requested - [INPUT] the quasiparticle approximation band_interpolation was given
     */
    static bool qp_approx_reusable(h5::group iter_grp, qp_context_t const& requested);

  private:
    /**
     * qp_approx == "qp_eqn" branch (qp_eqn.solver in {"sc", "sc_bisection", "sc_newton", "linearized"}) of
     * qp_approx_from_dyson, the counterpart of linearized_qp_checkpoint for "lqp":
     * 0. read {grp_name}/iter{N}/{F_skij, Sigma_tskij, mu} and system/H0_skij into node-shared
     *    memory (dimensions from the dataset shapes, the IAFT the checkpoint's),
     * 1. diagonalize F + H0,
     * 2. analytically continue the diagonal Sigma_aa(iw) (qp_eqn.ac_alg),
     * 3. solve the scalar quasiparticle equation on the real axis band by band 
     *    (at the quasiparticle energy for the "sc*" variants, by first-order expansion 
     *     around eps_KS for "linearized").
     * 
     * Writes qp_approx/{E_ska, Heff_skij, mu, scheme} and the controls
     * qp_approx/qp_eqn/{solver, ac_alg, ac_nfit, eta, tol}; removes the Zqp_ska and lqp/ of an
     * earlier lqp run, and the legacy ac_solver.
     *
     * @param filename - [INPUT] checkpoint file (read, then appended to)
     * @param grp_name - [INPUT] top-level group, "scf" or "embed"
     * @param iter     - [INPUT] iteration; -1 = {grp_name}/final_iter
     * @param params   - [INPUT] qp_eqn controls; see methods::qp_eqn::params_t
     */
    void qp_eqn_from_dyson_impl(std::string const& filename, std::string const& grp_name,
                                long iter, qp_eqn::params_t const& params);

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
