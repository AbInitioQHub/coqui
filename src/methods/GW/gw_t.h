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


#ifndef COQUI_GW_T_H
#define COQUI_GW_T_H

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "IO/app_loggers.h"

#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/ERI/detail/concepts.hpp"

namespace methods {
  namespace solvers {
    using namespace memory;

    /**
     * Solver for computing GW self-energy matrix.
     * One-body Hamiltonian is provided from mf::MF while Green's function
     * and ERIs are provided at runtime
     *
     * Usage:
     *   gw_t mygw(comm, std::addressof(myMF));
     *   mygw.evaluate(G, Sigma, thc_eri);
     *   mygw.evaluate(G, Sigma, chol_eri);
     */
    class gw_t {
    public:
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<nda::MemoryArray Array_base_t>
      using dsArray_t = math::shm::distributed_shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long,N>;

    public:
      gw_t(const imag_axes_ft::IAFT *ft, std::string div = "gygi",
           std::string output = "coqui");

      ~gw_t() {}

      // external functions of THC-GW/THC-RPA
      /**
       * Evalaute THC-GW self-energy
       * @param G_tskij      - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       */
      void evaluate(MBState &mb_state, THC_ERI auto const& thc, bool verbose=true);

      /**
       * Evalaute THC-GW self-energy
       * @param G_tskij      - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_view_5D_t>
      void evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                    sArray_t<Array_view_5D_t> &sSigma_tskij,
                    THC_ERI auto const& thc, scr_coulomb_t* scr_eri=nullptr,
                    bool verbose=true);

      /**
       * Evaluate THC-RPA correlation energy
       * @param G_tskij  - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc      - [INPUT] THC-ERI object
       * @return - RPA correlation energy
       */
      double rpa_energy(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc);

      /**
       * Evaluate GW self-energy
       * @param G_tskij      - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param dW_qtPQ      - [INPUT] Screened interaction in THC auxiliary basis: (nts, nqpts_ibz, Np, Np)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       * @param alg          - "R": convolution on R space; "k": convolution on k space
       */
      template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
      void eval_Sigma_all(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                          memory::darray_t<Array_4D_t, communicator_t> &dW_qtPQ,
                          sArray_t<Array_5D_t> &sSigma_tskij,
                          THC_ERI auto &thc,
                          std::string alg = "R");


      // external functions of Chol-GW
      void evaluate(MBState &mb_state, Cholesky_ERI auto &chol, bool verbose=true);

      /**
       * Evaluate Chol-GW self-energy
       * @param G_tskij      - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param chol         - [INPUT] Cholesky ERI object
       */
      template<nda::MemoryArray Array_5D_t>
      void evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                    sArray_t<Array_5D_t> &Sigma_tskij,
                    Cholesky_ERI auto &chol, scr_coulomb_t* scr_eri=nullptr,
                    bool verbose=true);

      /**
       * Evaluate Chol-RPA correlation energy
       * @param G_tskij  - [INPUT] Green's function in primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param chol     - [INPUT] Cholesky ERI object
       * @return - RPA correlation energy
       */
      double rpa_energy(const nda::MemoryArrayOfRank<5> auto &G_tskij, Cholesky_ERI auto &chol);

      template<nda::MemoryArray Array_3D_t>
      void evaluate_P0(size_t iq,
                       const nda::MemoryArrayOfRank<5> auto &G_tskij,
                       sArray_t<Array_3D_t> &sP0_tPQ,
                       Cholesky_ERI auto &chol,
                       int batch_size = -1,
                       bool print_mpi = false);

      template<nda::MemoryArray Array_5D_t>
      void evaluate_Sigma(size_t iq,
                          const nda::MemoryArrayOfRank<5> auto &G_tskij,
                          const nda::MemoryArrayOfRank<3> auto &P_tPQ,
                          sArray_t<Array_5D_t> &sSigma_tskij,
                          Cholesky_ERI auto &chol,
                          int batch_size = -1,
                          bool print_mpi = false);

      void print_thc_gw_timers();
 
      void print_thc_rpa_timers(); 

      void print_chol_gw_timers(); 

      void print_rpa_gw_timers(); 

      //void set_MF(mf::MF *MF) { _MF = MF; }

    private:
      /*** THC implementation details ***/
      template<nda::MemoryArray Array_view_5D_t, typename dArray_4D_t>
      void thc_gw_Xqindep(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                          sArray_t<Array_view_5D_t> &sSigma_tskij,
                          THC_ERI auto &thc, dArray_4D_t &dW_qtPQ,
                          const nda::MemoryArrayOfRank<1> auto &eps_inv_head);

      /**
       * transform self-energy from a THC auxiliary basis to a primary basis
       * @param dSigma_tskPQ - [INPUT] self-energy on THC interpolating points:
       *                       (nt_half, ns, nkpts_ibz, Np, Np). coverage of nt_half depends on "minus_t"
       * @param sSigma_tskij - [OUTPUT] self-energy in the primary basis: (nts, ns, nkpts_ibz, nbnd, nbnd)
       * @param thc          - [INPUT] THC-ERI instance
       * @param kp_map       - [INPUT] mapping of k-points based on a given symmetry operation
       * @param minus_t      - [INPUT] false: self-energy at tau=(0,beta/2); true: self-energy at tau=(0,-beta/2)
       */
      template<nda::MemoryArray Array_primary_t, nda::MemoryArray Array_aux_t, typename communicator_t>
      void setup_Sigma_primary(const memory::darray_t<Array_aux_t, communicator_t> &dSigma_tskPQ,
                               sArray_t<Array_primary_t> &sSigma_tskij,
                               THC_ERI auto &thc,
                               nda::ArrayOfRank<1> auto const& kp_map,
                               bool minus_t);

      /**
       * Evaluate GW self-energy by computing the convolution on the R space
       * @tparam Winp_in_Rspace - whether the input W is in the R space already
       * @tparam Wout_in_Rspace - whether the output W should be in the R space
       * @param minus_t         - false: compute self-energy at tau=(0,beta/2); true: compute self-energy at tau=(0,-beta/2)
       */
      template<bool Winp_in_Rspace, bool Wout_in_Rspace,
          nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t,
          typename communicator_t>
      void eval_Sigma_all_Rspace(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                 memory::darray_t<Array_4D_t, communicator_t> &dW_qtPQ,
                                 sArray_t<Array_5D_t> &sSigma_tskij,
                                 THC_ERI auto &thc,
                                 bool minus_t);

      /**
       * Evaluate GW self-energy by computing the convolution on the k space
       * @param minus_t - false: compute self-energy at tau=(0,beta/2); true: compute self-energy at tau=(0,-beta/2)
       */
      template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_4D_t, typename communicator_t>
      void eval_Sigma_all_kspace(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                 const memory::darray_t<Array_4D_t, communicator_t> &dW_qtPQ,
                                 sArray_t<Array_5D_t> &sSigma_tskij,
                                 THC_ERI auto &thc,
                                 bool minus_t);
      // details of eval_Sigma_all_kspace
      template<nda::MemoryArray Array_4D_t, typename communicator_t>
      void eval_Sigma_all_k_impl(long it, const memory::darray_t<Array_4D_t, communicator_t> &dG_skPQ,
                                 const memory::darray_t<Array_4D_t, communicator_t> &dW_qtPQ,
                                 memory::darray_t<Array_4D_t, communicator_t> &dSigma_skPQ,
                                 THC_ERI auto &thc, long isym);

      template<nda::MemoryArray Array_view_5D_t>
      void Sigma_div_correction(sArray_t<Array_view_5D_t> &sSigma_tskij,
                                const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                THC_ERI auto &thc,
                                const nda::array<ComplexType, 1> &eps_inv_head);

      double thc_rpa_Xqindep(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc);

      /**
       * Evaluate THC-RPA correlation energy
       * @param dPi_wqPQ - [INPUT] Polarization function: (nw, nqpts_ibz, Np, Np)
       * @param thc      - [INPUT] THC ERI object
       * @return - RPA correlation energy
       */
      template<nda::MemoryArrayOfRank<4> Array_w_t, typename communicator_t>
      double thc_rpa_energy_all_impl(memory::darray_t<Array_w_t, communicator_t> &dPi_wqPQ,
                                     THC_ERI auto &thc);

      /*** Cholesky implementation details ***/
      template<nda::MemoryArray Array_3D_t>
      ComplexType chol_rpa_energy_impl(const sArray_t<Array_3D_t> &sP0_wPQ);

      template<nda::MemoryArray Array_3D_t>
      void dyson_P(sArray_t<Array_3D_t> &sP0_tPQ,
                   sArray_t<Array_3D_t> &sP0_wPQ);

    private:
      const imag_axes_ft::IAFT* _ft = nullptr;

      std::string _div_treatment = "ignore_g0";

      // current iteration in SCF. Modified externally.
      long _iter = 0;
      std::string _output = "coqui";
      utils::TimerManager _Timer;

    public:
      long& iter() { return _iter; }
      std::string& output() { return _output; }
      std::string& div_treatment() { return _div_treatment; }

    };
  } // solvers
} // methods

#endif //COQUI_GW_T_H
