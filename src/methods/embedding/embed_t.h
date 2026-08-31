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


#ifndef COQUI_EMBED_T_H
#define COQUI_EMBED_T_H

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/distributed_array/nda.hpp"

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "methods/SCF/mb_solver_t.h"
#include "methods/mb_state/mb_state.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "numerics/iter_scf/iter_scf_t.hpp"
#include "numerics/ac/AC_t.hpp"
#include "utilities/mpi_context.h"
#include "methods/embedding/projector_t.h"
#include "methods/tools/chkpt_utils.h"
#include "methods/SCF/scf_common.hpp"

namespace methods {

  namespace mpi3 = boost::mpi3;

  /**
   * This class is responsible for downfolding and upfolding between correlated local subspaces and the environment.
   * It consists two main functions:
   * a) embedding.upfold(O_crsytal, O_loc) upfolds O_loc and add it to O_crystal
   * b) embedding.downfold(O_crystal, O_loc) downfold O_crystal and store it to O_loc
   */
  class embed_t {
  public:
    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    using mpi_context_t = utils::mpi_context_t<>;

  public:
    // w/o projector version. projector should be provided by MBState
    // This should be the only allowed constructor once the migration to MBState is complete.
    embed_t(mf::MF &MF): _context(MF.mpi()), _MF(std::addressof(MF)), _Timer() {}

    embed_t(mf::MF &MF, std::string C_file, bool translate_home_cell=false):
    _context(MF.mpi()), _MF(std::addressof(MF)),
    _proj(std::in_place, MF, C_file, translate_home_cell), _Timer() {}


    embed_t(mf::MF &MF,
            const nda::array<ComplexType, 5> &C_ksIai,
            const nda::array<int, 3> &band_window,
            const nda::array<RealType, 2> &kpts_crys,
            bool translate_home_cell=false):
        _context(MF.mpi()), _MF(std::addressof(MF)),
        _proj(std::in_place, MF, C_ksIai, band_window, kpts_crys, translate_home_cell),
        _Timer() {}

    void dmft_embed(MBState &mb_state, simple_dyson &dyson,
                    iter_scf::iter_scf_t *iter_solver=nullptr,
                    bool qp_approx_mbpt=false, bool corr_only=false);

    auto downfold_gloc(MBState& mb_state, bool force_real, std::string g_grp, long g_tier)
    -> nda::array<ComplexType, 5>;

    /**
     * @param context       - [INPUT] mpi communicator
     * @param outdir        - [INPUT] director for h5 checkpoint file
     * @param prefix        - [INPUT] prefix for h5 checkpoint file
     * @param scf_type      - [INPUT] SCF solution type: many_boby or quasipartical
     * @param qp_selfenergy - [INPUT] whether apply QP approximation to self-energy or not
     *                                (only used when scf_type == many_body)
     * @param dc_type       - [INPUT] dmft: phi-functional; edmft: psi-functional
     *                                (right now, it is only used when scf_type == quasiparticle)
     * @param qp_params    - [INPUT] quasiparticle approximation parameters
     * @param format_type   - [OPTION] Type of output: "default", "interaction_static"
     */
    void downfolding(MBState &mb_state, ptree const& pt,
                     qp_params_t *qp_params = nullptr, std::string format_type="default");

    template<THC_ERI thc_t>
    void hf_downfolding(std::string outdir, std::string prefix,
                        thc_t& eri, imag_axes_ft::IAFT &ft,
                        bool force_real, std::string hf_div_treatment="gygi");


    void add_Vhf_correction(MBState &mb_state);
    void add_Sigma_dyn_correction(MBState &mb_state, bool subtract_dc=true);
    template<nda::ArrayOfRank<4> Array_base_t>
    void add_Vcorr_correction(sArray_t<Array_base_t> &sVcorr_skij, MBState &mb_state);

    inline void print_dmft_embed_timers() {
      app_log(2, "\n  EMBED timers");
      app_log(2, "  ------------");
      app_log(2, "    Total:                    {0:.3f} sec", _Timer.elapsed("EMBED_TOTAL"));
      app_log(2, "    Allocation:               {0:.3f} sec", _Timer.elapsed("EMBED_ALLOC"));
      app_log(2, "    Upfold:                   {0:.3f} sec", _Timer.elapsed("EMBED_UPFOLD"));
      app_log(2, "    Find chemical potential:  {0:.3f} sec", _Timer.elapsed("EMBED_FIND_MU"));
      app_log(2, "    Dyson equation:           {0:.3f} sec", _Timer.elapsed("EMBED_DYSON"));
      app_log(2, "    Iterative solver:         {0:.3f} sec", _Timer.elapsed("EMBED_ITERATIVE"));
      app_log(2, "    Read:                     {0:.3f} sec", _Timer.elapsed("EMBED_READ"));
      app_log(2, "    Write:                    {0:.3f} sec\n", _Timer.elapsed("EMBED_WRITE"));
    }

    inline void print_downfold_mb_timers() {
      app_log(2, "\n  DOWNFOLD_1E timers");
      app_log(2, "  ------------------");
      app_log(2, "    Total:                    {0:.3f} sec", _Timer.elapsed("DF_TOTAL"));
      app_log(2, "    Allocation:               {0:.3f} sec", _Timer.elapsed("DF_ALLOC"));
      app_log(2, "    Double counting:          {0:.3f} sec", _Timer.elapsed("DF_DC"));
      app_log(2, "    Downfold:                 {0:.3f} sec", _Timer.elapsed("DF_DOWNFOLD"));
      app_log(2, "    Upfold:                   {0:.3f} sec", _Timer.elapsed("DF_UPFOLD"));
      app_log(2, "    Find chemical potential:  {0:.3f} sec", _Timer.elapsed("DF_FIND_MU"));
      app_log(2, "    Dyson equation:           {0:.3f} sec", _Timer.elapsed("DF_DYSON"));
      app_log(2, "    Fermionic Weiss field:    {0:.3f} sec", _Timer.elapsed("DF_G_WEISS"));
      app_log(2, "    Read:                     {0:.3f} sec", _Timer.elapsed("DF_READ"));
      app_log(2, "    Write:                    {0:.3f} sec\n", _Timer.elapsed("DF_WRITE"));
    }

    inline void print_downfold_hf_timers() {
      app_log(2, "\n  DOWNFOLD_1E timers");
      app_log(2, "  ------------------");
      app_log(2, "    Total:                    {0:.3f} sec", _Timer.elapsed("DF_TOTAL"));
      app_log(2, "    Allocation:               {0:.3f} sec", _Timer.elapsed("DF_ALLOC"));
      app_log(2, "    Double counting:          {0:.3f} sec", _Timer.elapsed("DF_DC"));
      app_log(2, "    Downfold:                 {0:.3f} sec", _Timer.elapsed("DF_DOWNFOLD"));
      app_log(2, "    Read:                     {0:.3f} sec", _Timer.elapsed("DF_READ"));
      app_log(2, "    Write:                    {0:.3f} sec\n", _Timer.elapsed("DF_WRITE"));
    }

  private:
    /*** dmft_embed implementation details ***/
    void dmft_embed_logic(long gw_iter, long weiss_f_iter, long embed_iter, std::string filename);
    void dmft_embed_impl(MBState &mb_state, simple_dyson &dyson,
                         iter_scf::iter_scf_t *iter_solver=nullptr,
                         bool corr_only=false);
    void dmft_embed_qp_impl(MBState &mb_state, simple_dyson &dyson,
                            iter_scf::iter_scf_t *iter_solver=nullptr);

    /*** downfold_1e implementation details ***/
    void downfold_hf_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                           std::string filename);
    auto gw_edmft_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                        std::string filename, bool update_dc)
    -> std::tuple<long, std::string>;
    long downfold_1e_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter,
                           std::string filename, bool update_dc);

    /**
     * Compute a downfolded 1e Hamiltonian using a many-body solution from a checkpoint h5 file.
     * @param context  - [INPUT] MPI communicator
     * @param filename - [INPUT] checkpoint h5 file
     * @param dc_type  - [INPUT] double counting type
     */
    void downfold_mb_solution_impl(
        MBState &mb_state, bool update_dc, std::string dc_type,
        bool force_real, std::string g_k_grp = "", long g_k_iter = -1,
        std::array<double, 2> mixing = {1.0, 1.0},
        std::string g_weiss_type="dmft");

    /**
     * Compute a downfolded 1e Hamiltonian using a many-body solution from a checkpoint h5 file.
     * Quasiparticle approximation is applied to the dynamic part of the self-energy, including GW and DC terms.
     * @param context  - [INPUT] MPI communicator
     * @param filename - [INPUT] checkpoint h5 file
     * @param dc_type  - [INPUT] double counting type
     */
    void downfold_mb_solution_qp_impl(MBState &mb_state, qp_params_t &qp_params,
                                      bool update_dc, std::string dc_type,
                                      bool force_real, std::string format_type = "default");
    /**
     * Compute a downfolded 1e Hamiltonian from a mean-field solution.
     * @param context  - [INPUT] MPI communicator
     * @param filename - [INPUT] checkpoint h5 file
     */
    template<THC_ERI thc_t>
    void downfold_hf_impl(std::string prefix,
                          thc_t& eri, imag_axes_ft::IAFT &ft,
                          bool force_real, std::string hf_div_treatment="gygi");

    auto double_counting_hf_bare(h5::group &gh5_dc, h5::group &gh5_V, 
                                 long dc_iter, std::string dc_src_grp,
                                 long weiss_b_iter, bool force_real, std::string format_type)
    -> nda::array<ComplexType, 4>;
    auto double_counting_hf_bare(std::string prefix, 
                                 long dc_iter, std::string dc_src_grp,
                                 long weiss_b_iter, bool force_real, std::string format_type)
    -> nda::array<ComplexType, 4>;

    auto double_counting(const nda::array<ComplexType, 5> &Gloc_tsIab,
                         h5::group &gh5,
                         std::string dc_type,
                         imag_axes_ft::IAFT &ft)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>;

    auto read_double_counting_qp(std::string filename, long weiss_f_iter, imag_axes_ft::IAFT &ft)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>;

    auto double_counting_qp(h5::group &gh5, h5::group &gh5_V, std::string prefix,
                            std::string dc_type, long dc_iter, std::string dc_src_grp,
                            long weiss_b_iter, imag_axes_ft::IAFT &ft,
                            double mu, sArray_t<Array_view_4D_t> &sMO_skia, sArray_t<Array_view_3D_t> &sE_ska,
                            qp_params_t &qp_params, bool force_real, std::string format_type)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>;

    auto double_counting_qp(std::string prefix,
                            std::string dc_type, long dc_iter, std::string dc_src_grp,
                            long weiss_b_iter, imag_axes_ft::IAFT &ft,
                            double mu, sArray_t<Array_view_4D_t> &sMO_skia, sArray_t<Array_view_3D_t> &sE_ska,
                            qp_params_t &qp_params, bool force_real, std::string format_type)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>;

    /**
     * Compute fermionic Weiss field:
     *     g(iw)] = [ [G(iw)]^{-1} + Vhf_imp + Sigma_imp(iw) ]^{-1}
     * @param Gloc_wsIab      - [INPUT] Local Green's function
     * @param hf_imp_sIab     - [INPUT] Static (i.e. Hartree-Fock) impurity self-energy
     * @param sigma_imp_wsIab - [INPUT] Dynamic impurity self-energy
     * @return Fermionic Weiss field g_wsIab.
     */
    auto compute_g_weiss(const nda::array<ComplexType, 5> &Gloc_wsIab,
                         const nda::array<ComplexType, 4> &hf_imp_sIab,
                         const nda::array<ComplexType, 5> &sigma_imp_wsIab)
    -> nda::array<ComplexType, 5>;

    /**
     * Compute fermionic Weiss field:
     *     g(iw)] = [ [G(iw)]^{-1} + Vhf_imp + Sigma_imp(iw) ]^{-1}
     * @param Gloc_wsIab      - [INPUT] Local Green's function
     * @param filename        - [INPUT] Checkpoint file where Vhf_imp and Sigma_imp stored
     * @param weiss_f_iter    - [INPUT] Iteration of downfold_1e
     * @return Fermionic Weiss field g_wsIab.
     */
    auto compute_g_weiss(const nda::array<ComplexType, 5> &Gloc_wsIab,
                         h5::group h5_grp, long weiss_f_iter)
    -> nda::array<ComplexType, 5>;

    auto compute_hybridization(const nda::array<ComplexType, 5> &g_wsIab,
                               const nda::array<ComplexType, 4> &t_sIab,
                               double mu, imag_axes_ft::IAFT &ft)
    -> nda::array<ComplexType, 5>;

  private:
    std::shared_ptr<mpi_context_t> _context;
    mf::MF* _MF = nullptr;

    // TODO Remove _proj
    std::optional<projector_t> _proj;
    //long _nImps = -1;
    //long _nOrbs_W = -1;
    //long _nImpOrbs = -1;

    utils::TimerManager _Timer;

  public:
    mf::MF* MF() { return _MF; }
    //auto C_skIai() const& { return _proj.C_skIai(); }
    //auto W_rng() const& { return _proj.W_rng(); }
    //long nImps() const { return _nImps; }
    //long nImpOrbs() const { return _nImpOrbs; }
    //long nOrbs_W() const { return _nOrbs_W; }
    //std::string C_file() const { return _proj.C_file(); }

  };
} // methods


#endif //COQUI_EMBED_T_H
