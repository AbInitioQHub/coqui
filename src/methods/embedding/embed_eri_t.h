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


#ifndef COQUI_EMBED_ERI_T_H
#define COQUI_EMBED_ERI_T_H

#include "configuration.hpp"
#include "utilities/mpi_context.h"
#include "nda/h5.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/sparse/csr_blas.hpp"

#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/embedding/projector_boson_t.h"
#include "methods/mb_state/mb_state.hpp"
#include "methods/embedding/permut_symm.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/GW/gw_t.h"
#include "methods/embedding/dc_utilities.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/tools/chkpt_utils.h"

// TODO CNY
//   1. Move downfold/upfold routines to projector_boson_t

namespace methods {
  namespace mpi3 = boost::mpi3;
  /**
   * Screened interactions based on constrained RPA.
   *
   * Usage:
   *   // Initialize with a given mean-field object, IAFT grid,
   *   // and a file with Wannier transformation matrices.
   *   embed_eri_t embed_2e(&mf, wannier_file);
   *
   *   // Screened interaction from cRPA based on a given output.mbpt.h5 
   *   embed_2e.downfolding(mpi_context, output, thc_eri, "crpa", factorization_type);
   *
   *   // Screened interaction from EDMFT based on a given output.mbpt.h5
   *   embed_2e.downfolding(mpi_context, output, thc_eri, "edmft", factorization_type);
   */
  // TODO
  //    1. make h5 output a class member
  class embed_eri_t {
  public:
    template<int N>
    using shape_t = std::array<long,N>;
    using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;

  public:
    embed_eri_t(mf::MF &MF,
                std::string div = "gygi", std::string bare_div = "gygi",
                std::string output_type = "default"):
    _context(MF.mpi()), _MF(std::addressof(MF)),
    _div_treatment(div), _bare_div_treatment(bare_div), _Timer(),
    _output_type(output_type) {

      if (_MF->nkpts_ibz() == 1 and _div_treatment != "ignore_g0") {
        app_log(2, " embed_eri_t: nkpts_ibz == 1 while div_treatment != ignore. Will take div_treatment = ignore_g0 anyway!");
        _div_treatment = "ignore_g0";
      }

      if (_bare_div_treatment!="ignore_g0" and _bare_div_treatment!="gygi") {
        app_log(2, " embed_eri_t: bare_div_treatment only supports \"ignore_g0\" and \"gygi\". "
                   " coqui will take bare_div_treatment = \"gygi\" instead.");
        _bare_div_treatment = "gygi";
      }
    }

    ~embed_eri_t() = default;

    template<bool return_wt = false, THC_ERI thc_t>
    auto compute_downfolded_coulomb_tensors(
      thc_t &eri, MBState &mb_state, std::string screen_type,
      bool force_permut_symm, bool force_real, imag_axes_ft::IAFT *ft,
      std::string greens_func_source, long greens_func_iteration,
      bool write_to_hdf5 = false, bool q_dependent_output = false)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >;

    template<THC_ERI thc_t>
    void downfolding_edmft(thc_t &eri, MBState &mb_state, ptree const& pt,
                           std::string screen_type);


    template<THC_ERI thc_t>
    void downfolding_crpa(thc_t &eri, MBState &mb_state, ptree const& pt,
                          std::string screen_type,
                          std::string factorization_type = "none", double thresh=1e-6);

    template<bool return_wt = false>
    auto compute_downfolded_coulomb_tensors_impl(
      THC_ERI auto &eri, MBState &mb_state,
      std::string screen_type, std::string permut_symm, const imag_axes_ft::IAFT &ft, 
      std::string greens_func_source, long greens_func_iteration, 
      bool write_to_hdf5, bool q_dependent_output)
    -> std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5> >;

    void downfold_edmft_impl(THC_ERI auto &eri, MBState &mb_state,
                             std::string screen_type, std::string permut_symm,
                             std::array<std::string, 2> g_grp,
                             std::array<long, 2> g_iter,
                             std::array<double, 2> mixing = {1.0, 1.0});

    void downfold_crpa_impl(THC_ERI auto &eri, MBState &mb_state,
                            std::string screen_type,
                            [[maybe_unused]] std::string factorization_type,
                            std::string permut_symm,
                            std::string greens_func_source = "", long greens_func_iteration = -1,
                            bool q_dependent = false,
                            [[maybe_unused]] double thresh = 1e-6);

    void downfold_screen_model_impl(THC_ERI auto &eri, MBState &mb_state,
                                    std::string screen_type,
                                    std::string factorization_type,
                                    std::string permut_symm,
                                    std::string greens_func_source = "", long greens_func_iteration = -1,
                                    double thresh = 1e-6);

    void downfold_bare_impl(std::string output,
                            THC_ERI auto &eri,
                            const projector_boson_t &proj_boson,
                            std::string factorization_type,
                            std::string permut_symm,
                            double thresh = 1e-6);

    template<nda::MemoryArray Array_base_t>
    auto to_product_basis(Array_base_t &A_bacd) {
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;
      if constexpr (rank==4) {

        size_t nbnd = A_bacd.shape(0);
        nda::array<value_type, 4> A_pb_abcd(A_bacd.shape());
        for (size_t a=0; a<nbnd; ++a) {
          for (size_t b=0; b<nbnd; ++b) {
            A_pb_abcd(a,b,nda::ellipsis{}) = A_bacd(b,a,nda::ellipsis{});
          }
        }
        return A_pb_abcd;

      } else if constexpr (rank==5) {

        size_t dim0 = A_bacd.shape(0);
        size_t nbnd = A_bacd.shape(1);
        nda::array<value_type, 5> A_pb_abcd(A_bacd.shape());
        for (size_t i=0; i<dim0; ++i) {
          for (size_t a = 0; a < nbnd; ++a) {
            for (size_t b = 0; b < nbnd; ++b) {
              A_pb_abcd(i,a,b,nda::ellipsis{}) = A_bacd(i,b,a,nda::ellipsis{});
            }
          }
        }
        return A_pb_abcd;

      } else {
        static_assert(rank==4 or rank==5, "to_product_basis: rank != 5 or 4.");
      }
    }

    /**
     * compute bosonic Weiss field from local RPA polarizability
     * @param G_tsIab - [INPUT] Local Green's function within the active space
     * @param W_wabcd - [INPUT] Local screened interactions within the active space
     * @param V_abcd  - [INPUT] Local bare interactions within the active space
     * @return EDMFT bosonic Weiss field U(w, a, b, c, d) within the active space
     */
    auto u_bosonic_weiss_rpa(nda::array<ComplexType, 5> &G_tsIab,
                             nda::array<ComplexType, 5> &W_wabcd,
                             nda::array<ComplexType, 4> &V_abcd,
                             const imag_axes_ft::IAFT &ft,
                             bool density_only=false)
    -> sArray_t<Array_view_5D_t>;

    /**
     * compute bosonic Weiss field from EDMFT polarizability in place
     * @param W_wabcd - [INPUT] Local screened interaction
     * @param V_abcd  - [INPUT] Local bare Coulomb interaction
     * @param sPi_imp_wabcd - [INPUT/OUTPUT] Impurity polarizability/bosonic Weiss field
     */
    void u_bosonic_weiss_edmft_in_place(
        const nda::array<ComplexType, 5> &W_wabcd,
        const nda::array<ComplexType, 4> &V_abcd,
        sArray_t<Array_view_5D_t> &sPi_imp_wabcd);

    /**
     * compute bosonic Weiss field from EDMFT polarizability
     * @param weiss_b_grp - [INPUT] h5 group where the edmft polarizability is stored
     * @param W_wabcd - [INPUT] Local screened interactions within the active space
     * @param V_abcd  - [INPUT] Local bare interactions within the active space
     * @return EDMFT bosonic Weiss field U(w, a, b, c, d) within the active space
     */
    auto u_bosonic_weiss_edmft(h5::group weiss_b_grp,
                               nda::array<ComplexType, 5> &W_wabcd,
                               nda::array<ComplexType, 4> &V_abcd,
                               const imag_axes_ft::IAFT &ft)
    -> sArray_t<Array_view_5D_t>;

    /**
     * Compute the bosonic Weiss field u(i\Omega) in the local product basis
     * @param sW_pb_wabcd    - [INPUT] dynamic local screened interaction in the local product basis
     * @param V_abcd         - [INPUT] static local screened interaction in the local product basis
     * @param Pi_imp_wabcd   - [INPUT] impurity polarizability in the local product basis
     *                       - [OUTPUT] bosonic Weiss field in the local product basis
     */
    void dyson_for_u_weiss_in_place(sArray_t<Array_view_5D_t> &sW_pb_wabcd, sArray_t<Array_view_4D_t> &sV_pb_abcd,
                                    sArray_t<Array_view_5D_t> &sPi_imp_pb_wabcd);

    template<bool return_eps_inv=false, typename G_t, THC_ERI thc_t>
    auto local_eri_impl(const G_t &G, thc_t &thc, const imag_axes_ft::IAFT &ft,
                        std::string screen_type="rpa",
                        const nda::array_view<ComplexType, 5> *pi_imp = nullptr,
                        const nda::array_view<ComplexType, 5> *pi_dc = nullptr);
    template<bool return_eps_inv=false, THC_ERI thc_t>
    auto local_eri_impl(MBState &mb_state, thc_t &thc, const imag_axes_ft::IAFT &ft,
                        std::string screen_type="rpa");
    template<bool return_eps_inv=false, THC_ERI thc_t>
    auto rpa_q_eri_impl(MBState &mb_state, thc_t &thc, const imag_axes_ft::IAFT &ft,
                        std::string screen_type="rpa");
    auto rpa_chol_eri_impl(MBState &mb_state, THC_ERI auto &thc, const imag_axes_ft::IAFT &ft,
                           std::string factorization_type, double thresh, std::string screen_type="rpa");

    template<THC_ERI thc_t>
    auto compute_collation_impurity_basis(thc_t &thc, const projector_boson_t &proj_boson, nda::range u_rng);

    template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
    auto downfold_V(thc_t &thc, const B_t &B_qIPab)
        -> nda::array<ComplexType, 4>;

    template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
    auto downfold_Vq(thc_t &thc, const B_t &B_qIPab)
        -> nda::array<ComplexType, 5>;

    auto downfold_cholesky(THC_ERI auto &thc, const projector_boson_t &proj_boson,
                           math::nda::DistributedArrayOfRank<3> auto &dV_qPQ,
                           ComplexType div_correction_factor, double thresh = 1e-6);

    auto downfold_cholesky_high_memory(THC_ERI auto &thc, const projector_boson_t &proj_boson,
                                       math::nda::DistributedArrayOfRank<3> auto &dV_qPQ,
                                       ComplexType div_correction_factor,
                                       double thresh = 1e-6);

    template<THC_ERI thc_t>
    auto downfold_V_thc(thc_t &thc, double thresh = 1e-6);

    template<THC_ERI thc_t, nda::MemoryArray Array_4D_t, typename communicator_t,
        nda::ArrayOfRank<5> B_t>
    auto downfold_W(thc_t &thc, memory::darray_t<Array_4D_t, communicator_t> &dW_wqPQ,
                    const B_t &B_qIPab, const nda::array<ComplexType, 1> &eps_inv_head)
        -> nda::array<ComplexType, 5>;

    template<THC_ERI thc_t, nda::MemoryArray Array_4D_t, typename communicator_t, nda::ArrayOfRank<5> B_t>
    auto downfold_Wq(thc_t &thc, memory::darray_t<Array_4D_t, communicator_t> &dW_wqPQ,
                     const B_t &B_qIPab)
    -> nda::array<ComplexType, 6>;

    template<THC_ERI thc_t, nda::ArrayOfRank<5> B_t>
    void V_div_correction(nda::array<ComplexType, 4> &V_cdab, const B_t &B_qIPab, thc_t &thc);

    template<THC_ERI thc_t>
    void W_div_correction(thc_t &thc,
                          nda::array<ComplexType, 5> &W_wcdab,
                          const nda::array<ComplexType, 5> &B_qIPab,
                          const nda::array<ComplexType, 1> &eps_inv_head);

    template<nda::MemoryArrayOfRank<4> Array_4D_t>
    auto orbital_average_int(const Array_4D_t &V_abcd) -> std::tuple<ComplexType, ComplexType, ComplexType, ComplexType>;

    auto downfold_2e_logic(long gw_iter, long weiss_f_iter, long weiss_b_iter, long embed_iter)
    -> std::tuple<std::string, long>;
    void check_downfold_2e_logic(std::string greens_func_source, long greens_func_iteration, long gw_iter, long embed_iter);

    inline void print_downfold_timers() {
      app_log(2, "\n  DOWNFOLD_2E timers");
      app_log(2, "  ------------------");
      app_log(2, "    Total:                    {0:.3f} sec", _Timer.elapsed("DF_TOTAL"));
      app_log(2, "    Downfold:                 {0:.3f} sec", _Timer.elapsed("DF_DOWNFOLD"));
      app_log(2, "    Symmetrization:           {0:.3f} sec", _Timer.elapsed("DF_SYMM"));
      app_log(2, "    Read:                     {0:.3f} sec", _Timer.elapsed("DF_READ"));
      app_log(2, "    Write:                    {0:.3f} sec\n", _Timer.elapsed("DF_WRITE"));
    }

  private:
    std::shared_ptr<mpi_context_t> _context;
    mf::MF* _MF = nullptr;

    std::string _div_treatment;
    std::string _bare_div_treatment;
    utils::TimerManager _Timer;

    std::string _output_type = "default";

  public:
    mf::MF* MF() const { return _MF; }
    const std::string& div_treatment() const { return _div_treatment; }

  }; // embed_eri_t
} // methods


#endif //COQUI_EMBED_ERI_T_H
