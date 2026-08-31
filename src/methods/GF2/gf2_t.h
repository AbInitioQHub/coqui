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


#ifndef COQUI_GF2_T_H
#define COQUI_GF2_T_H

#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "mean_field/MF.hpp"
#include "methods/GW/gw_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/ERI/detail/concepts.hpp"

#define W_GATHER_OLD 0 // a switch to the old version for debugging

// TODO Pass mf::MF and mpi_context_t via ERI

namespace methods {
  namespace solvers {
    using namespace memory;

    /**
     * Solver for computing GF2 self-energy matrix.
     * One-body Hamiltonian is provided from mf::MF while Green's function
     * and ERIs are provided at runtime
     *
     * Divergence treatment is ignored for the direct and exchange GF2 terms, 
     * but it can be used for the GW direct term.
     *
     * The interface is similar to GW, but has more options:
     * Direct terms (THC only): 
     *    "gf2" -- GF2 second-order direct term
     *    "gw"  -- THC only. A full GW (GW calculation will run inside GF2)
     * Exchange algorithms: 
     *     "none"  -- exchange will be skipped
     *     "orb"   -- THC only. A scheme with full parallelization over v will be executed. 
     *                It is good for large systems due to smaller memory requirements
     *     "3ind"  -- THC only. The fastest algorithm available, 
     *                good for rather small systems due to demanding memory requirements
     *     "3ind_reallocate" -- THC only. A slower 3ind version with less memory requirements
     *
     * Usage:
     *   gf2_t mygf2(comm, std::addressof(myMF), std::addressof(myft), gygi, "gf2", "orb", "gf2", "bdft");
     *   mygf2.evaluate(G, Sigma, thc_eri);
     *   mygf2.evaluate(G, Sigma, chol_eri);
     */
    class gf2_t {
    public:
      using mpi_context_t = utils::mpi_context_t<mpi3::communicator>;
      template<nda::MemoryArray Array_base_t>
      using sArray_t = math::shm::shared_array<Array_base_t>;
      template<nda::MemoryArray Array_base_t>
      using dsArray_t = math::shm::distributed_shared_array<Array_base_t>;
      template<int N>
      using shape_t = std::array<long,N>;

    public:
      gf2_t(mf::MF *MF, imag_axes_ft::IAFT *ft,
           std::string div = "gygi", 
           std::string direct_type="gf2",
           std::string exchange_alg="orb",
           std::string exchange_type="gf2",
           std::string output = "coqui",
           bool save_C_ = true, bool sosex_save_memory_ = true);

      ~gf2_t() = default;

      /**
       * Evaluate THC-GF2 self-energy
       * @param G_tskij     - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param thc          - [INPUT] thc_reader_t ERI object
       */
      void evaluate(MBState &mb_state, THC_ERI auto &thc);

      /**
       * Evaluate Cholesky-GF2 self-energy
       * @param G_tskij      - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param chol         - [INPUT] chol_reader_t ERI object
       */
      void evaluate(MBState &mb_state, Cholesky_ERI auto &chol);

      // print timers
      void print_chol_gf2_timers();
      void print_thc_gf2_timers();
      void print_thc_sosex_timers(); 

      // accessor functions
      long& iter(); 
      std::string output() const;
      std::string gw_div_treatment() const;
      double& t_thresh(); 

      std::string direct_type() const; 
      std::string exchange_type() const; 
      std::string exchange_alg() const; 
      bool sosex_save_memory() const; 
      bool save_C() const; 

    private:
      void thc_gf2_Xqindep(MBState &mb_state, THC_ERI auto &thc);

      /**
       * Evaluate the THC-GF2 direct term of the self-energy via the thc-gw class.
       * @param G_tskij     - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       */
       void thc_run_direct(MBState &mb_state, THC_ERI auto &thc);

      /**
       * Evaluate THC-GF2 Z Pi Z W-like term for the direct self-energy term
       * @param dPi_tqPQ     - [INPUT] GW's Pi in aux basis: (nts, nkpts, NP, NP)
       * @param thc          - [INPUT] THC ERI object
       */
       template<nda::MemoryArray Array_4D_t, typename communicator_t>
       void eval_UPiU_in_place(memory::darray_t<Array_4D_t, communicator_t> &dPi_tqPQ,
                               THC_ERI auto &thc);

      /**
       * Evaluate THC-GF2 exchange term of self-energy with 3ind and 3ind_reallocate algorithms
       * @param G_tskij     - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       */
       template<nda::MemoryArray Array_5D_t>
       void thc_gf2_scheme1(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                            sArray_t<Array_5D_t> &sSigma_tskij, bool reallocate,
                            THC_ERI auto &thc);

      /**
       * Evaluate THC-GF2 exchange term of self-energy with orb algorithm
       * @param G_tskij     - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param sSigma_tskij - [OUTPUT] Self-energy in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_view_5D_t>
      void thc_gf2_scheme1_orb_4D(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                  sArray_t<Array_view_5D_t> &sSigma_tskij,
                                  THC_ERI auto &thc);

      /**
       * Construct subcommunicators for orb algorithm for parallelization over orbital index
       * @param nvpools      - [INPUT] size of vcomm subcommunicator
       * @param thc          - [INPUT] THC ERI object
       */
      auto prepare_vcomm(long nvpools, THC_ERI auto &thc)
      -> std::array<mpi3::communicator, 2>; 

      /**
       * Allocate Z for THC-GF2 exchange term
       * @param G_tskij     - [INPUT] Green's function in primary basis: (nts, ns, nkpts, nbnd, nbnd)
       * @param dB_qkPR      - [INPUT] B_4D intermediate: (nkpts, nkpts, NP, NP)
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto allocate_Z(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                      memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                      THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 3>, mpi3::communicator>; 

      /**
       * Allocate B_5D intermediate for THC-GF2 exchange term
       * @param dU_qkPqs      - [INPUT] U_5D intermediate: (nkpts, nkpts, NP, nbnd, nbnd)
       * @param thc           - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_5D_t1, typename communicator_t>
      auto allocate_B(const memory::darray_t<Array_5D_t1, communicator_t> &dU_qkPqs,
                      THC_ERI auto &thc)
      -> std::array<memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>, 2>;

      /**
       * Allocate B_4D intermediate for THC-GF2 exchange term
       * @param thc           - [INPUT] THC ERI object
       * @param comm          - [INPUT] B_4D subcommunicator
       * @param print_mpi     - [INPUT] whether to print MPI processors
       */
      auto allocate_B(THC_ERI auto &thc, mpi3::communicator& comm, bool print_mpi)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>; 

      /**
       * Allocate C_5D intermediate for THC-GF2 exchange term
       * @param dU_qkRvw      - [INPUT] U_5D intermediate: (nkpts, nkpts, NP, nbnd, nbnd)
       * @param thc           - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_5D_t1, typename communicator_t>
      auto allocate_C(const memory::darray_t<Array_5D_t1, communicator_t> &dU_qkRvw,
                      THC_ERI auto &thc)
      -> std::array<memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>, 2>; 

      /**
       * Allocate D_4D intermediate for THC-GF2 exchange term
       * @param B_pgrid      - [INPUT] processor grid used for B_4D and C_4D intermediates
       * @param B_bsize      - [INPUT] block sizes used for B_4D and C_4D intermediates
       * @param comm         - [INPUT] B_4D subcommunicator
       * @param thc          - [INPUT] THC ERI object
       */
      template<typename Array>
      auto allocate_D(Array B_pgrid, Array B_bsize, mpi3::communicator& comm,
                      THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>; 

      /**
       * Construct subcommunicators for parallelization over q and k k-indices
       * @param dB_qkPR      - [INPUT] B_4D intermediate: (nkpts, nkpts, NP, NP)
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      mpi3::communicator make_qk_intracom(memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                                          THC_ERI auto &thc);

      /**
       * Estimate ||Z(q)||, where the norm is either "frobenious", "abs", or "max"
       * @param dZ_qPR       - [INPUT] Z: (nkpts, NP, NP)
       * @param norm_def     - [INPUT] norm definition above
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_3D_t1, typename communicator_t>
      auto estimate_Z_norm(memory::darray_t<Array_3D_t1, communicator_t> &dZ_qPR,
                           const std::string& norm_def, THC_ERI auto &thc)
      ->  memory::array<HOST_MEMORY, ComplexType, 1>;

      /**
       * Estimate an upper bound of ||B(q,k)|| without computing B, 
       * where the norm is either "frobenious", "abs", or "max"
       * @param dB_qkPR      - [INPUT] B: (nkpts, nkpts, NP, NP)
       * @param Z_q_norms    - [INPUT] ||Z(q)|| array: (nkpts)
       * @param GRQ          - [INPUT] Green's function in aux basis: (nkpts, NP, NP)
       * @param GRv          - [INPUT] Green's function in aux and prim basis: (nkpts, NP, nbnd)
       * @param is           - [INPUT] alpha or beta spin
       * @param norm_def     - [INPUT] norm definition above
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto estimate_B_norm(memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                           nda::array<ComplexType, 1> &Z_q_norms, nda::array<ComplexType, 3>& GRQ,
                           nda::array<ComplexType, 3>& GQv, size_t is, const std::string& norm_def,
                           THC_ERI auto &thc)
      ->  memory::array<HOST_MEMORY, ComplexType, 2>;

      /**
       * Estimate an upper bound of ||C(q,k)|| without computing C, 
       * where the norm is either "frobenious", "abs", or "max"
       * @param dC_qkPR      - [INPUT] C: (nkpts, nkpts, NP, NP)
       * @param Z_q_norms    - [INPUT] ||Z(q)|| array: (nkpts)
       * @param GSP          - [INPUT] Green's function in aux basis: (nkpts, NP, NP)
       * @param is           - [INPUT] alpha or beta spin
       * @param norm_def     - [INPUT] norm definition above
       * @param thc          - [INPUT] THC ERI object
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto estimate_C_norm(memory::darray_t<Array_4D_t1, communicator_t> &dC_qkPR,
                           nda::array<ComplexType, 1> &Z_q_norms, nda::array<ComplexType, 3>& GSP,
                           size_t is, const std::string& norm_def, THC_ERI auto &thc)
      ->  memory::array<HOST_MEMORY, ComplexType, 2>;

      /**
       * Returns buffers needed for B_4D build
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto allocate_buffers_B(memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                              mpi3::communicator& qk_intra_comm, THC_ERI auto &thc)
      -> std::array<memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 2>, mpi3::communicator>, 3>; 

      /**
       * Performs this contraction at a fixed v:
       * B_PR^{qk} = sum_{Q} Z_{PQ}^q * G_{RQ}^{k_u} G_{Qv}^{k_v}
       */
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_3D_t1,
          nda::MemoryArray Array_2D_t1, typename communicator_t>
      void build_B_4D(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                    memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                    memory::darray_t<Array_3D_t1, communicator_t> &dZ_qPQ,
                    memory::darray_t<Array_2D_t1, communicator_t> &dZ_PQ,
                    memory::darray_t<Array_2D_t1, communicator_t> &dXX_QR,
                    memory::darray_t<Array_2D_t1, communicator_t> &dB_PR,
                    nda::array<ComplexType, 3>& GRQ, nda::array<ComplexType, 3>& GQv,
                    size_t is, size_t it, size_t iv, THC_ERI auto &thc);
      /**
       * Performs this contraction at all v:
       * B_PRv^{qk} = sum_{Q} Z_{PQ}^q * G_{RQ}^{k_u} G_{Qv}^{k_v}
       */
      template<nda::MemoryArray Array_5D_t, typename communicator_t>
      auto build_B(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                   const memory::darray_t<Array_5D_t, communicator_t> &dU_qkPqs,
                   size_t is, size_t it, THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>; 

      template<nda::MemoryArray Array_5D_t1,  nda::MemoryArray Array_5D_t2,
                typename communicator_t>
      void build_B(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                   const memory::darray_t<Array_5D_t1, communicator_t> &dU_qkPqs,
                   memory::darray_t<Array_5D_t2, communicator_t> &dB_qkPRv,
                   size_t is, size_t it, THC_ERI auto &thc);

      /**
       * Evaluate B and redistribute it with reallocation to proc grid (1,1,n1,n2,1) for building D later
       */
      template<nda::MemoryArray Array_5D_t1, typename communicator_t>
      auto build_B_and_redistribute(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        const memory::darray_t<Array_5D_t1, communicator_t> &dU_qkPqs,
                        size_t is, size_t it, THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>;

      /**
       * Evaluate B and redistribute it without reallocation to proc grid (1,1,n1,n2,1) for building D later
       */
      template<nda::MemoryArray Array_6D_t1, nda::MemoryArray Array_5D_t1, typename communicator_t>
      void build_B_and_redistribute(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        const memory::darray_t<Array_6D_t1, communicator_t> &dU_sqkPqs,
                        memory::darray_t<Array_5D_t1, communicator_t> &dB_qkPRv,
                        memory::darray_t<Array_5D_t1, communicator_t> &dB_qkPRv_contr,
                        size_t is, size_t it, THC_ERI auto &thc);

      /**
       * Evaluate C and redistribute it with reallocation to proc grid (1,1,n1,n2,1) for building D later
       */
      template<nda::MemoryArray Array_5D_t1, typename communicator_t>
      auto build_C_and_redistribute(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        const memory::darray_t<Array_5D_t1, communicator_t> &dU_qkRvw,
                        size_t is, size_t it, THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>; 

      /**
       * Evaluate C and redistribute it without reallocation to proc grid (1,1,n1,n2,1) for building D later
       */
      template<nda::MemoryArray Array_6D_t1, nda::MemoryArray Array_5D_t1, typename communicator_t>
      void build_C_and_redistribute(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        const memory::darray_t<Array_6D_t1, communicator_t> &dU_sqkRvw,
                        memory::darray_t<Array_5D_t1, communicator_t> &dC_qkPRv_out,
                        memory::darray_t<Array_5D_t1, communicator_t> &dC_qkPRv,
                        size_t is, size_t it, THC_ERI auto &thc);

      /**
       * Performs this contraction at a fixed v:
       * C_PR^{qk} = sum_{S} Z_{RS}^q * G_{SP}^{k} X_{vS}^{k_v}
       */
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_3D_t1,
             nda::MemoryArray Array_2D_t1, typename communicator_t>
      void build_C_4D(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                      memory::darray_t<Array_4D_t1, communicator_t> &dC_qkPR,
                      memory::darray_t<Array_3D_t1, communicator_t> &dZ_qRS,
                      memory::darray_t<Array_2D_t1, communicator_t> &dZ_PQ,
                      memory::darray_t<Array_2D_t1, communicator_t> &dXX_QR,
                      memory::darray_t<Array_2D_t1, communicator_t> &dB_PR,
                      nda::array<ComplexType, 3>& GSP,
                      size_t is, size_t it, size_t iv, THC_ERI auto &thc);

      /**
       * Returns transformed and partly transformed to aux basis Green's functions 
       * G_{QR}(is, it) and G_{Qv}(is, nt-is-1)
       */
      auto prepare_G(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                     size_t is, size_t it, THC_ERI auto &thc)
       -> std::array< nda::array<ComplexType, 3>, 2>;

      template<nda::MemoryArray Array_5D_t1, typename communicator_t>
      auto build_D(const memory::darray_t<Array_5D_t1, communicator_t> &dB_qkPRv,
                   const memory::darray_t<Array_5D_t1, communicator_t> &dC_qkPRv,
                   size_t is, size_t it, THC_ERI auto &thc)
       -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>;

      template<nda::MemoryArray Array_5D_t1, nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto build_D(const memory::darray_t<Array_5D_t1, communicator_t> &dB_qkPRv,
                   const memory::darray_t<Array_5D_t1, communicator_t> &dC_qkPRv,
                   memory::darray_t<Array_4D_t1, communicator_t> &dD_qkPR,
                   size_t is, size_t it, THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 4>, mpi3::communicator>; 

      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      void build_D_4D(const memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                      const memory::darray_t<Array_4D_t1, communicator_t> &dC_qkPR,
                      memory::darray_t<Array_4D_t1, communicator_t> &dD_qkPR,
                      nda::array<long,2>& B_kPR_bounds,
                      nda::array<ComplexType, 3>& B_kPR,
                      mpi3::communicator& PR_comm,
                      size_t is, size_t it, THC_ERI auto &thc);

      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto get_qkPR_bounds(const memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                           mpi3::communicator& PR_comm)
      -> nda::array<long,2>;

      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      auto get_B_q_origins(const memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                           mpi3::communicator& PR_comm)
      -> nda::array<long,1>;

      template<nda::MemoryArray Array_t, typename communicator_t>
      void print_self_norm(memory::darray_t<Array_t, communicator_t> &dA, std::string name);

      /**
       * Finds processors that have iq k-point index of B_4D in the PR_comm, 
       * containing processors with the same P and R origins
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      std::vector<long> search_sender_proc(size_t iq,
                        const memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR,
                        mpi3::communicator& PR_comm);

      /**
       * Create a processor group with the same P and R origins of B
       */
      template<nda::MemoryArray Array_4D_t1, typename communicator_t>
      mpi3::group create_PR_group(const memory::darray_t<Array_4D_t1, communicator_t> &dB_qkPR);

      /**
       * Non-parallel and parallel variants of Sigma evaluation
       */
      template<nda::MemoryArray Array_view_5D_t, nda::MemoryArray Array_4D_t1, typename communicator_t>
      void evaluate_Sigma(sArray_t<Array_view_5D_t> &sSigma_tskij,
                          const memory::darray_t<Array_4D_t1, communicator_t> &dD_qkPR,
                          size_t is, size_t it, THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t, nda::MemoryArray Array_4D_t1, typename communicator_t>
      void evaluate_Sigma_par(sArray_t<Array_view_5D_t> &sSigma_tskij,
                              const memory::darray_t<Array_4D_t1, communicator_t> &dD_qkPR,
                              nda::array<long,2>& D_kPR_bounds, 
                              nda::array<ComplexType, 3>& D_kPR,
                              mpi3::communicator& PR_comm,
                              size_t is, size_t it, THC_ERI auto &thc);


      template<nda::MemoryArray Array_6D_t1, typename communicator_t>
      void build_U_Pqs_all(THC_ERI auto &thc,
                   memory::darray_t<Array_6D_t1, communicator_t>& dA_sqkPqs,
                   bool bare, bool dynamic_only);

      template<nda::MemoryArray Array_view_2D_t>
      void transform_G(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                       Array_view_2D_t& G_tr,
                       size_t is, size_t ik, size_t it,
                       std::string code,
                       THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t, nda::MemoryArray Array_view_2D_t>
      void transform_Sigma(sArray_t<Array_view_5D_t> &sSigma_tskij,
                           Array_view_2D_t& Sigma_PR,
                           size_t is, size_t ik, size_t it,
                           THC_ERI auto &thc);

      template<nda::MemoryArray Array_5D_t>
      void chol_run_2(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                      sArray_t<Array_5D_t> &sSigma_tskij,
                      Cholesky_ERI auto &chol);

      ////////// SOSEX FUNCTIONS ////////

      template<nda::MemoryArray Array_view_5D_t>
      void thc_sosex_scheme2(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                             sArray_t<Array_view_5D_t> &sSigma_tskij, THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t>
      void thc_sosex_scheme2_4D(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                sArray_t<Array_view_5D_t> &sSigma_tskij, THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t>
      void thc_sosex_scheme2_4D_no_reallocation(
          const nda::MemoryArrayOfRank<5> auto &G_tskij,
          sArray_t<Array_view_5D_t> &sSigma_tskij, THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t>
      void add_2sosex(sArray_t<Array_view_5D_t> &sSigma_tskij, THC_ERI auto &thc);

      template<nda::MemoryArray local_Array_6D_t>
      auto prepare_C_from_saved(memory::darray_t<local_Array_6D_t, mpi3::communicator> &dB_tsqkPR, size_t v)
      -> memory::darray_t<local_Array_6D_t, mpi3::communicator>;

      template<nda::MemoryArray local_Array_6D_t, nda::MemoryArray local_Array_5D_t,
             nda::MemoryArray local_Array_6D_t1>
      auto build_C_sosex(const sArray_t<local_Array_5D_t> &sG_tskijQ,
                         const memory::darray_t<local_Array_6D_t1, mpi3::communicator> &dU_sqkRvw,
                         memory::darray_t<local_Array_6D_t, mpi3::communicator> &dB_tsqkPR,
                         size_t v, THC_ERI auto &thc)
      -> memory::darray_t<local_Array_6D_t, mpi3::communicator>;

      template<nda::MemoryArrayOfRank<5> Array_5D_t>
      auto prepare_G_sosex(const Array_5D_t &G_tskij, THC_ERI auto &thc)
      -> std::array< sArray_t<nda::array_view<typename std::decay_t<Array_5D_t>::value_type, 5>>, 2>;

      template<nda::MemoryArray local_Array_t, typename communicator_t>
      auto tau_to_w_full(memory::darray_t<local_Array_t, communicator_t> &dA_tx,
                         std::array<long, ::nda::get_rank<std::decay_t<local_Array_t>>> w_pgrid_out,
                         std::array<long, ::nda::get_rank<std::decay_t<local_Array_t>>> w_bsize_out,
                         std::string name, bool reset_input)
      -> memory::darray_t<local_Array_t, mpi3::communicator>;

      template<nda::MemoryArray local_Array_t, typename communicator_t>
      void tau_to_w_full(memory::darray_t<local_Array_t, communicator_t> &dA_tx,
                         memory::darray_t<local_Array_t, communicator_t> &dA_wx,
                         std::string name);

      template<nda::MemoryArray local_Array_t, typename communicator_t>
      auto w_to_tau_full(memory::darray_t<local_Array_t, communicator_t> &dA_wx,
                        std::array<long, ::nda::get_rank<std::decay_t<local_Array_t>>> t_pgrid_out,
                        std::array<long, ::nda::get_rank<std::decay_t<local_Array_t>>> t_bsize_out,
                        std::string name,
                        bool reset_input)
      -> memory::darray_t<local_Array_t, mpi3::communicator>;

      template<nda::MemoryArray local_Array_t, typename communicator_t>
      void w_to_tau_full(memory::darray_t<local_Array_t, communicator_t> &dA_wx,
                       memory::darray_t<local_Array_t, communicator_t> &dA_tx,
                        std::string name);

      memory::array<HOST_MEMORY, ComplexType, 2> get_dyn_W(THC_ERI auto &thc, size_t iw, size_t iq);

      memory::array<HOST_MEMORY, ComplexType, 2> get_dyn_W_ranged(THC_ERI auto &thc, size_t iw, size_t iq,
                                                                  nda::range P_rng, nda::range Q_rng);

      auto build_WGG(THC_ERI auto &thc,
        memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 7>, mpi3::communicator>& dA_wsqkQuv) 
       -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 7>, mpi3::communicator>;

      auto build_WGG_4D(THC_ERI auto &thc,
                        memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& dA_wsqkQu, std::optional<nda::array<ComplexType, 4>>& Wloc_wqPQ_opt, bool reset_A)
       -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>; 

      void build_WGG_4D(THC_ERI auto &thc,
                        memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& dA_wsqkQu,
                        memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& dWA_wsqkQu,
                        std::optional<nda::array<ComplexType, 4>>& Wloc_wqPQ_opt, bool reset_A);

      template<nda::MemoryArray Array_7D_t1, typename communicator_t>
      auto build_B_WA(memory::darray_t<Array_7D_t1, communicator_t> &dWA_tsqkPuv, THC_ERI auto &thc)
        -> memory::darray_t<Array_7D_t1, communicator_t>;

      template<nda::MemoryArray Array_6D_t1, typename communicator_t>
      auto build_B_WA_4D(memory::darray_t<Array_6D_t1, communicator_t> &dWA_tsqkPu,
                         THC_ERI auto &thc, bool reset_WA)
        -> memory::darray_t<Array_6D_t1, communicator_t>;

      template<nda::MemoryArray Array_6D_t1, typename communicator_t>
      void build_B_WA_4D(memory::darray_t<Array_6D_t1, communicator_t> &dWA_tsqkPu,
                         memory::darray_t<Array_6D_t1, communicator_t> &dB_tsqkPR,
                         THC_ERI auto &thc, bool reset_WA);


      void allocate_C_for_saving(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc);

      void check_C_for_saving(THC_ERI auto &thc,
           memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 5>, mpi3::communicator>& dC_qkPRv);


      template<nda::MemoryArray Array_7D_t1, typename communicator_t>
      void redistribute_BC_sosex(memory::darray_t<Array_7D_t1, communicator_t> &dB_tsqkPRv,
                                 THC_ERI auto &thc);

      auto allocate_A_sosex_4D(THC_ERI auto &thc)
      -> std::array<memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>,2>;

      auto allocate_B_sosex_4D(THC_ERI auto &thc,
           memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& A_tx,
           memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& D)
      -> std::array<memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>,2>;

      template<nda::MemoryArray Array_7D_t1, typename communicator_t>
      auto allocate_D_sosex(memory::darray_t<Array_7D_t1, communicator_t> &dB_tsqkPRv,
                            THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>;

      auto allocate_D_sosex(THC_ERI auto &thc)
      -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>;

      void allocate_B_for_saving(const nda::MemoryArrayOfRank<5> auto &G_tskij, THC_ERI auto &thc);


      template<nda::MemoryArray Array_view_5D_t>
      auto build_GG_Quv_all_full_t(THC_ERI auto &thc,
                                   const sArray_t<Array_view_5D_t> &sG_tskiQ,
                                   const sArray_t<Array_view_5D_t> &sG_tskQj)
       -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 7>, mpi3::communicator>;

      template< nda::MemoryArray Array_view_5D_t>
      auto build_GG_Qu_all_full_t_4D(THC_ERI auto &thc,
                                     const sArray_t<Array_view_5D_t> &sG_tskiQ,
                                     const sArray_t<Array_view_5D_t> &sG_tskQj, size_t v)
       -> memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>;

      template< nda::MemoryArray Array_view_5D_t>
      void build_GG_Qu_all_full_t_4D(THC_ERI auto &thc,
                memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator>& dA_tsqkQu_v,
                const sArray_t<Array_view_5D_t> &sG_tskiQ, const sArray_t<Array_view_5D_t> &sG_tskQj, size_t v); 

      template<nda::MemoryArray Array_7D_t1, nda::MemoryArray Array_6D_t1, typename communicator_t>
      void build_D_sosex_all_t(const memory::darray_t<Array_7D_t1, communicator_t> &dB_tsqkPRv,
                        const memory::darray_t<Array_7D_t1, communicator_t> &dC_tsqkPRv,
                        memory::darray_t<Array_6D_t1, communicator_t> &dD_tsqkPR,
                        THC_ERI auto &thc);

      template<nda::MemoryArray Array_6D_t1, nda::MemoryArray Array_6D_t2, typename communicator_t>
      void build_D_sosex_all_t_4D(const memory::darray_t<Array_6D_t2, communicator_t> &dB_tsqkPR,
                        const memory::darray_t<Array_6D_t2, communicator_t> &dC_tsqkPR,
                        memory::darray_t<Array_6D_t1, communicator_t> &dD_tsqkPR,
                        THC_ERI auto &thc);

      template<nda::MemoryArray Array_view_5D_t, nda::MemoryArray Array_6D_t1, typename communicator_t>
      void evaluate_Sigma_sosex_all_t(sArray_t<Array_view_5D_t> &sSigma_tskij,
                              const memory::darray_t<Array_6D_t1, communicator_t> &dD_tsqkPR,
                              THC_ERI auto &thc);

      ////////// CHOLESKY CONTRACTION KERNELS ////////

      /**
       *  Performs I_tuvp = sum_w v_tuvw * G_wp
       */
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2, nda::MemoryArray Array_view_2D_t>
      void contract_4_4(const Array_4D_t1& v_tuvw, const Array_view_2D_t& G_wp, Array_4D_t2& I_tuvp);
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2, nda::MemoryArray Array_view_2D_t>
      void contract_4_4_inplace(Array_4D_t1& v_tuvw, const Array_view_2D_t& G_wp, Array_4D_t2& buffer);

      /**
       *  Performs I_tqvp = sum_u v_tuvw * G_uq
       */
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2, nda::MemoryArray Array_view_2D_t>
      void contract_4_2(const Array_4D_t1& v_tuvw, const Array_view_2D_t& G_uq, Array_4D_t2& I_tqvw);
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2,
               nda::MemoryArray Array_4D_t3, nda::MemoryArray Array_view_2D_t>
      void contract_4_2_inplace(Array_4D_t1& v_tuvw, const Array_view_2D_t& G_uq,
                                Array_4D_t2& buffer1, Array_4D_t3& buffer2);

      /**
       *  Performs I_tusw = sum_v v_tuvw * G_vs
       */
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2, nda::MemoryArray Array_view_2D_t>
      void contract_4_3(const Array_4D_t1& v_tuvw, const Array_view_2D_t& G_vs, Array_4D_t2& I_tusw);
      template<nda::MemoryArray Array_4D_t1, nda::MemoryArray Array_4D_t2,
               nda::MemoryArray Array_4D_t3, nda::MemoryArray Array_view_2D_t>
      void contract_4_3_inplace(Array_4D_t1& v_tuvw, const Array_view_2D_t& G_vs,
                                Array_4D_t2& buffer1, Array_4D_t3& buffer2);

      template<nda::MemoryArray Array_3D_t1, nda::MemoryArray Array_3D_t2, nda::MemoryArray Array_4D_t>
      void merge_4indx_basic(const Array_3D_t1& L1_Qpr, const Array_3D_t2& L2_Qqs, Array_4D_t& I_prqs);

      // Reconstruct integrals from Cholesky ERI
      template<nda::MemoryArray Array_4D_t>
      void build_int(Array_4D_t& V_prqs, size_t q1, size_t k_r, size_t k_q, size_t is, Cholesky_ERI auto &chol);


      // Direct term in Cholesky-GF2
      template<nda::MemoryArray Array_5D_t>
      void chol_run_direct(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                           sArray_t<Array_5D_t> &sSigma_tskij, Cholesky_ERI auto &chol);

      // Evaluate Cholesky self-energy
      template<nda::MemoryArray Array_4D_t, nda::MemoryArray Array_2D_t>
      void build_Sigma(const Array_4D_t& I1_rqps, const Array_4D_t& I2_tqps,
                       Array_2D_t &Sigma_tr);

      template<nda::MemoryArray Array_ND_t>
      void set_zero(Array_ND_t& A);

      ///// NORMS ////
      template<nda::MemoryArray Array_t>
      void print_self_norm(const Array_t& A, std::string name);

      template <typename Array_t>
      double frob_norm(Array_t const &a);

      template <typename Array_t>
      double frob_norm2(Array_t const &a);

      template<nda::MemoryArray Array_t, typename communicator_t>
      double frob_norm(memory::darray_t<Array_t, communicator_t> &dA);

      template<nda::MemoryArray Array_t, typename communicator_t>
      double frob_norm2(memory::darray_t<Array_t, communicator_t> &dA);

      template <typename Array_t>
      double abs_norm(Array_t const &a);

      template<nda::MemoryArray Array_t, typename communicator_t>
      double abs_norm(memory::darray_t<Array_t, communicator_t> &dA);

      template <typename Array_t>
      double max_norm(Array_t const &a);

      template<nda::MemoryArray Array_t, typename communicator_t>
      double max_norm(memory::darray_t<Array_t, communicator_t> &dA);

      double norm_X(size_t is, size_t iv, THC_ERI auto &thc);

      double norm_X(size_t is, THC_ERI auto &thc);

      memory::array<HOST_MEMORY, ComplexType, 2> get_static_W(size_t iq, THC_ERI auto &thc, bool dynamic_only);

      // sSigma += sSigma_exc
      template<nda::MemoryArray Array_5D_t, nda::MemoryArray Array_5D_t2>
      void add_exc_to_Sigma(sArray_t<Array_5D_t> &sSigma,
                            sArray_t<Array_5D_t2> &sSigma_exc, ComplexType scale=1.0);

    private:
      std::shared_ptr<mpi_context_t> _context;
      mf::MF *_MF = nullptr;
      imag_axes_ft::IAFT* _ft = nullptr;

      std::string _div_treatment = "ignore_g0";
      double _t_thresh = 0.0; // it prescreening threshold
      std::string _direct_type = "gf2";
      std::string _exchange_type = "gf2";
      std::string _exchange_alg = "orb";
      bool _save_C = true;
      bool _sosex_save_memory = true;

      // current iteration in SCF. Modified externally.
      long _iter = 0;
      std::string _output = "coqui";
      utils::TimerManager _Timer;


      /*
       * Saves all indices needed for _MF->qk_to_k1 functionality  
       */
      void setup_qk_to_k1(nda::array<size_t, 2>& kmap) { 
          size_t nkpts = kmap.shape(0);
          for(size_t q1 = 0; q1 < nkpts; q1++)
          for(size_t k_p = 0; k_p < nkpts; k_p++) {
              size_t k_r = _MF->qk_to_k2(q1, k_p); 
              kmap(q1,k_r) = k_p;
          }
      }

      /*
       * Returns _MF->qk_to_k1 functionality  
       */
      inline size_t qk_to_k1(size_t q1, size_t k, nda::array<size_t, 2>& kmap) {return kmap(q1,k);}

      using Array_6D_t = memory::array<HOST_MEMORY, ComplexType, 6>;
      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator> dU_sqkPqs_int; 

      memory::darray_t<memory::array<HOST_MEMORY, ComplexType, 6>, mpi3::communicator> dW0_sqkPqs_int;
      // w=0 slice of the dynamical part of the screened interaction
      // to get a full W(w=0), one needs to add bare interaction
      std::optional<memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator > > dW0_qPQ_opt;

      // the dynamical part of the screened interaction
      // used only in dynamic_sosex
      std::optional<memory::darray_t<nda::array<ComplexType, 3>, mpi3::communicator > > dW_wq_PQ_opt;
      std::optional<memory::darray_t<nda::array<ComplexType, 4>, mpi3::communicator > > dW_wqPQ_sosex_opt;
      // save C to reuse in sosex
      std::optional<memory::darray_t<nda::array<ComplexType, 7>, mpi3::communicator > > dC_tsqkPRv_opt;

      std::optional< nda::array<ComplexType, 3>> buffer; 

      using Array_5D_memt = memory::array_view<HOST_MEMORY, ComplexType, 5>;
      std::optional< sArray_t<Array_5D_memt >> sSigma_WdynWdyn_opt; // for 2SOSEX 

      bool U_computed = false;

      bool doing_dyn = false;

   };
}
}

#endif
