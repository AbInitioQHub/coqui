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


#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "nda/blas.hpp"
#include "numerics/shared_array/nda.hpp"
#include "numerics/nda_functions.hpp"

#include "utilities/proc_grid_partition.hpp"
#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/GW/gw_t.h"
#include "methods/GW/cholesky_gw.icc"

namespace methods {
  namespace solvers {

    // TODO
    //   - Store P0 and P for tau_b = 0 ~ (nt_b/2 - 1)
    //   - MPI parallelization over (tau, P/a) and also the q-axis
    //   - Check will q-dependent Np lead to a bug?
    //   - Profiling

    void gw_t::evaluate(MBState &mb_state, Cholesky_ERI auto &chol, bool verbose) {
      using namespace math::shm;
      using Array_3D_t = nda::array_view<ComplexType, 3>;
      if (verbose) {
        // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-gw
        app_log(1, "\n"
                   "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┌─┐┬ ┬\n"
                   "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───│ ┬│││\n"
                   "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ └─┘└┴┘\n");

        _ft->metadata_log();
      }
      utils::check(mb_state.mpi == chol.mpi(),
                   "gw_t::cholesky_gw::evaluate: Cholesky_ERI and MBState should have the same MPI context.");
      utils::check(mb_state.sG_tskij.has_value(),
                   "gw_t::cholesky_gw::evaluate: sG_tskij is not initialized in MBState.");
      utils::check(mb_state.sSigma_tskij.has_value(),
                   "gw_t::cholesky_gw::evaluate: sSigma_tskij is not initialized in MBState.");
      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(),
                   "gw_t::cholesky_gw::evaluate: Symmetry is not implemented yet.");
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "chol-gw:: we assume nt_f == nt_b at least for now \n"
                   "(will lift the restriction at some point...)");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "cholesky-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL", "ALLOC", "COMM",
                     "EVALUATE_P0", "DYSON_P","EVALUATE_SIGMA",
                     "IMAG_FT", "ERI_READER"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");

      auto mpi = chol.mpi();
      auto& sG_tskij = mb_state.sG_tskij.value();
      auto& sSigma_tskij = mb_state.sSigma_tskij.value();

      _Timer.start("ALLOC");
      size_t nt_half = (_ft->nt_f()%2==0)? _ft->nt_f()/2 : _ft->nt_f()/2 + 1;
      size_t nw_half = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;

      sSigma_tskij.set_zero();
      sArray_t<Array_3D_t> sP0_tPQ(*mpi, {nt_half, chol.Np(), chol.Np()});
      sArray_t<Array_3D_t> sP0_wPQ(*mpi, {nw_half, chol.Np(), chol.Np()});
      _Timer.stop("ALLOC");
      // TODO these should be input parameters
      int Np_batch = sP0_tPQ.local().shape(1);
      int nbnd_batch = sSigma_tskij.local().shape(3);
      for (size_t iq = 0; iq < chol.MF()->nkpts(); ++iq) {
        _Timer.start("EVALUATE_P0");
        evaluate_P0(iq, sG_tskij.local(), sP0_tPQ, chol, Np_batch, (iq==0)? true : false);
        _Timer.stop("EVALUATE_P0");

        _Timer.start("DYSON_P");
        dyson_P(sP0_tPQ, sP0_wPQ);
        _Timer.stop("DYSON_P");

        _Timer.start("EVALUATE_SIGMA");
        evaluate_Sigma(iq, sG_tskij.local(), sP0_tPQ.local(), sSigma_tskij, chol, nbnd_batch, (iq==0)? true : false);
        _Timer.stop("EVALUATE_SIGMA");
      }
      _Timer.start("COMM");
      sSigma_tskij.win().fence();
      sSigma_tskij.all_reduce();
      sSigma_tskij.win().fence();
      _Timer.stop("COMM");
      _Timer.stop("TOTAL");
      print_chol_gw_timers();
      chol.print_timers();
    }

    template<nda::MemoryArray Array_5D_t>
    void gw_t::evaluate(const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        sArray_t<Array_5D_t> &sSigma_tskij,
                        Cholesky_ERI auto &chol, scr_coulomb_t* scr_eri, bool verbose) {
      using namespace math::shm;
      using Array_3D_t = nda::array_view<ComplexType, 3>;
      if (verbose) {
        // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-gw
        app_log(1, "\n"
                   "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┌─┐┬ ┬\n"
                   "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───│ ┬│││\n"
                   "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ └─┘└┴┘\n");

        _ft->metadata_log();
      }

      utils::check(chol.MF()->nkpts() == chol.MF()->nkpts_ibz(),
                   "gw_t::cholesky_gw::evaluate: Symmetry is not implemented yet.");
      utils::check(_ft->nt_f() == _ft->nt_b(),
                   "chol-gw:: we assume nt_f == nt_b at least for now \n"
                   "(will lift the restriction at some point...)");
      { // Check if tau_mesh is symmetric w.r.t. beta/2
        auto tau_mesh = _ft->tau_mesh();
        long nts = tau_mesh.shape(0);
        for (size_t it = 0; it < nts; ++it) {
          size_t imt = nts - it - 1;
          double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
          utils::check(diff <= 1e-6, "cholesky-gw: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                       tau_mesh(it), tau_mesh(imt));
        }
      }

      for( auto& v: {"TOTAL", "ALLOC", "COMM",
                     "EVALUATE_P0", "DYSON_P","EVALUATE_SIGMA",
                     "IMAG_FT", "ERI_READER"} ) {
        _Timer.add(v);
      }

      (void) scr_eri;  // turn off warning, remove when in use
      //utils::check(scr_eri==nullptr, "chol-gw: scr_eri != nullptr is not implemented. "
      //                               "This means evGW0 is not implemented yet!");

      _Timer.start("TOTAL");

      auto mpi = chol.mpi();

      _Timer.start("ALLOC");
      size_t nt_half = (_ft->nt_f()%2==0)? _ft->nt_f()/2 : _ft->nt_f()/2 + 1;
      size_t nw_half = (_ft->nw_b()%2==0)? _ft->nw_b()/2 : _ft->nw_b()/2 + 1;

      sSigma_tskij.set_zero();
      sArray_t<Array_3D_t> sP0_tPQ(*mpi, {nt_half, chol.Np(), chol.Np()});
      sArray_t<Array_3D_t> sP0_wPQ(*mpi, {nw_half, chol.Np(), chol.Np()});
      _Timer.stop("ALLOC");
      // TODO these should be input parameters
      int Np_batch = sP0_tPQ.local().shape(1);
      int nbnd_batch = sSigma_tskij.local().shape(3);
      for (size_t iq = 0; iq < chol.MF()->nkpts(); ++iq) {
        _Timer.start("EVALUATE_P0");
        evaluate_P0(iq, G_tskij, sP0_tPQ, chol, Np_batch, (iq==0)? true : false);
        _Timer.stop("EVALUATE_P0");

        _Timer.start("DYSON_P");
        dyson_P(sP0_tPQ, sP0_wPQ);
        _Timer.stop("DYSON_P");

        _Timer.start("EVALUATE_SIGMA");
        evaluate_Sigma(iq, G_tskij, sP0_tPQ.local(), sSigma_tskij, chol, nbnd_batch, (iq==0)? true : false);
        _Timer.stop("EVALUATE_SIGMA");
      }
      _Timer.start("COMM");
      sSigma_tskij.win().fence();
      sSigma_tskij.all_reduce();
      sSigma_tskij.win().fence();
      _Timer.stop("COMM");
      _Timer.stop("TOTAL");
      print_chol_gw_timers();
      chol.print_timers();
    }

    template<nda::MemoryArray Array_3D_t>
    void gw_t::evaluate_P0(size_t iq,
                     const nda::MemoryArrayOfRank<5> auto &G_tskij,
                     sArray_t<Array_3D_t> &sP0_tPQ,
                     Cholesky_ERI auto &chol,
                     int batch_size,
                     bool print_mpi) {
      decltype(nda::range::all) all;
      sP0_tPQ.set_zero();
      size_t nt_half  = sP0_tPQ.local().shape(0);
      size_t nts   = G_tskij.shape(0);
      size_t ns    = G_tskij.shape(1);
      size_t nkpts = chol.MF()->nkpts();
      size_t Np    = chol.Np();
      size_t nbnd  = chol.MF()->nbnd();

      if (batch_size < 0) batch_size = Np;
      utils::check(Np % batch_size == 0, "gw_t::evaluate_P0: Np % batch_size != 0");
      size_t n_batch = Np / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sP0_tPQ.communicator(), nt_half, n_batch);
      if (print_mpi) {
        app_log(2, "    - evaluate_P0: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nt_half axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along Np axis = {}", dim1_comm_size);
      }

      _Timer.start("ALLOC");
      nda::array<ComplexType, 3> L_Pab_conj(batch_size, nbnd, nbnd);
      auto L_Pa_b_conj = nda::reshape(L_Pab_conj, shape_t<2>{batch_size*nbnd, nbnd});

      nda::array<ComplexType, 3> X_Pac(batch_size, nbnd, nbnd);
      auto X_Pa_c = nda::reshape(X_Pac, shape_t<2>{batch_size*nbnd, nbnd});
      auto X_P_ac = nda::reshape(X_Pac, shape_t<2>{batch_size, nbnd*nbnd});

      nda::array<ComplexType, 3> X2_acP(nbnd, nbnd, batch_size);
      auto X2_ac_P = nda::reshape(X2_acP, shape_t<2>{nbnd*nbnd, batch_size});
      auto X2_a_cP = nda::reshape(X2_acP, shape_t<2>{nbnd, nbnd*batch_size});

      nda::array<ComplexType, 3> Y_dcP(nbnd, nbnd, batch_size);
      auto Y_d_cP = nda::reshape(Y_dcP, shape_t<2>{nbnd, nbnd*batch_size});
      auto Y_dc_P = nda::reshape(Y_dcP, shape_t<2>{nbnd*nbnd, batch_size});

      nda::matrix<ComplexType> Z_QP(Np, batch_size);
      _Timer.stop("ALLOC");

      double spin_factor = (ns == 1)? -2.0/nkpts : -1.0/nkpts; // FIXME keep 1/nkpts just for debug

      sP0_tPQ.win().fence();
      for (size_t it = dim0_rank; it < nt_half; it += dim0_comm_size) { // MPI
        size_t itt = nts - it - 1;
        for (size_t is = 0; is < ns; ++is) {
          for (size_t ik = 0; ik < nkpts; ++ik) {
            long ikmq = chol.MF()->qk_to_k2(iq, ik); // K(ikmq) = K(ik) - Q(iq) + G
            auto L_Pab = chol.V(iq, is, ik);
            auto Gmt_bc = nda::transpose(G_tskij(itt, is, ikmq, all, all));
            auto Gt_da  = nda::transpose(G_tskij(it, is, ik, all, all));
            for (size_t PP = dim1_rank; PP < n_batch; PP += dim1_comm_size) { // MPI
              nda::range P_range(PP*batch_size, (PP+1)*batch_size);
              // X_Pac = L_Pab_conj * Gmt_bc
              L_Pab_conj = nda::conj(L_Pab(P_range, nda::ellipsis{}));
              nda::blas::gemm(L_Pa_b_conj, Gmt_bc, X_Pa_c);

              // X2_acP = X_Pac
              X2_ac_P = nda::transpose(X_P_ac);

              // Y_dcP = Gt_da * X2_acP
              nda::blas::gemm(Gt_da, X2_a_cP, Y_d_cP);

              // P0_PQ = Y_dcP * L_Qdc (L_Pab)
              auto L_Q_dc = nda::reshape(L_Pab, shape_t<2>{Np, nbnd*nbnd});
              nda::blas::gemm(ComplexType(spin_factor), L_Q_dc, Y_dc_P, ComplexType(0.0), Z_QP);
              sP0_tPQ.local()(it, P_range, all) += nda::transpose(Z_QP);
            }
          }
        }
      }
      _Timer.start("COMM");
      sP0_tPQ.win().fence();
      sP0_tPQ.all_reduce();
      _Timer.stop("COMM");
    }

    // TODO
    //   - optimize memory usage
    //   - parallel over j as well in order to reduce memory usage
    template<nda::MemoryArray Array_G_t>
    void gw_t::evaluate_Sigma(size_t iq,
                        const nda::MemoryArrayOfRank<5> auto &G_tskij,
                        const nda::MemoryArrayOfRank<3> auto &P_tPQ,
                        sArray_t<Array_G_t> &sSigma_tskij,
                        Cholesky_ERI auto &chol,
                        int batch_size,
                        bool print_mpi) {
      decltype(nda::range::all) all;
      size_t nt_half  = P_tPQ.shape(0);
      size_t nts   = G_tskij.shape(0);
      size_t ns    = G_tskij.shape(1);
      size_t nkpts = chol.MF()->nkpts();
      size_t Np    = chol.Np();
      size_t nbnd  = chol.MF()->nbnd();

      if (batch_size < 0) batch_size = nbnd;
      utils::check(nbnd % batch_size == 0, "gw_t::evaluate_sigma: nbnd % batch_size != 0");
      size_t n_batch = nbnd / batch_size;
      auto[dim0_rank, dim0_comm_size, dim1_rank, dim1_comm_size] =
          utils::setup_two_layer_mpi(sSigma_tskij.communicator(), nts, n_batch);
      if (print_mpi) {
        app_log(2, "    - evaluate_Sigma: batch size = {}", batch_size);
        app_log(2, "    - MPI processors along nts axis = {}", dim0_comm_size);
        app_log(2, "    - MPI processors along nbnd axis = {}", dim1_comm_size);
      }

      _Timer.start("ALLOC");

      // MAM: No need for most of these intermediates, fix!!!
      nda::array<ComplexType, 3> L_Pia(Np, batch_size, nbnd);
      auto L_Pi_a = nda::reshape(L_Pia, shape_t<2>{Np*batch_size, nbnd});

      nda::array<ComplexType, 3> X_Pib(Np, batch_size, nbnd);
      auto X_Pi_b = nda::reshape(X_Pib, shape_t<2>{Np*batch_size, nbnd});
      auto X_P_ib = nda::reshape(X_Pib, shape_t<2>{Np, batch_size*nbnd});

      nda::array<ComplexType, 3> Y_Qib(Np, batch_size, nbnd);
      auto Y_Q_ib = nda::reshape(Y_Qib, shape_t<2>{Np, batch_size*nbnd});

      nda::array<ComplexType, 3> Y2_ibQ(batch_size, nbnd, Np);
      auto Y2_ib_Q = nda::reshape(Y2_ibQ, shape_t<2>{batch_size*nbnd, Np});
      auto Y2_i_bQ = nda::reshape(Y2_ibQ, shape_t<2>{batch_size, nbnd*Np});

      nda::array<ComplexType, 3> L_bQj_conj(nbnd, Np, nbnd);
      auto L_b_Qj_conj = nda::reshape(L_bQj_conj, shape_t<2>{nbnd, Np*nbnd});
      auto L_bQ_j_conj = nda::reshape(L_bQj_conj, shape_t<2>{nbnd*Np, nbnd});
      _Timer.stop("ALLOC");

      sSigma_tskij.win().fence();
      for (size_t it = dim0_rank; it < nts; it += dim0_comm_size) {
        size_t it_b = (it < nt_half)? it : nts-it-1;
        auto PtF_QP = nda::transpose(P_tPQ(it_b, nda::ellipsis{}));
        for (size_t is = 0; is < ns; ++is) {
          for (size_t ik = 0; ik < nkpts; ++ik) {
            long ikmq = chol.MF()->qk_to_k2(iq, ik); // K(ikmq) = K(ik) - Q(iq) + G
            auto Gt_ab = G_tskij(it, is, ikmq, all, all);
            auto L_Qj_b = nda::reshape(chol.V(iq, is, ik), shape_t<2>{Np*nbnd, nbnd});
            L_b_Qj_conj = nda::conj(nda::transpose(L_Qj_b));
            for (size_t ii = dim1_rank; ii < n_batch; ii += dim1_comm_size) {
              nda::range i_range(ii*batch_size, (ii+1)*batch_size);
              // X_Pib = L_Pia * Gt_ab
              L_Pia = chol.V(iq, is, ik)(all, i_range, all);
              nda::blas::gemm(L_Pi_a, Gt_ab, X_Pi_b);
              
              // Y_Qib = Pt_PQ * X_Pib
              nda::blas::gemm(PtF_QP, X_P_ib, Y_Q_ib);
              
              // Y2_ibQ = Y_Qib
              Y2_ib_Q = nda::transpose(Y_Q_ib);
              
              // Sigma_ij = Y2_ibQ * L_jbQ_conj
              auto Sigma_ij = sSigma_tskij.local()(it, is ,ik, i_range, all);
              // FIXME remove 1/nkpts
              nda::blas::gemm(ComplexType(-1.0/nkpts), Y2_i_bQ, L_bQ_j_conj, ComplexType(1.0), Sigma_ij);
            } 
          } 
        } 
      } 
      sSigma_tskij.win().fence();
    } 

    // instantiations
    using Arr3 = nda::array<ComplexType, 3>;
    using Arrv3 = nda::array_view<ComplexType, 3>;
    using Arrv3_2 = nda::array_view<ComplexType, 3, nda::C_layout>;
    using Arr = nda::array<ComplexType, 5>;
    using Arrv = nda::array_view<ComplexType, 5>;
    using Arrv2 = nda::array_view<ComplexType, 5, nda::C_layout>;

    template void gw_t::evaluate(const Arr &, sArray_t<Arrv> &, chol_reader_t &, scr_coulomb_t*, bool);
    template void gw_t::evaluate(const Arrv &, sArray_t<Arrv> &, chol_reader_t &, scr_coulomb_t*, bool);
    template void gw_t::evaluate(const Arrv2 &, sArray_t<Arrv> &, chol_reader_t &, scr_coulomb_t*, bool);

    template void gw_t::evaluate(MBState &mb_state, chol_reader_t&, bool);

    template void gw_t::evaluate_P0(size_t, const Arr &, sArray_t<Arrv3> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_P0(size_t, const Arrv &, sArray_t<Arrv3> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_P0(size_t, const Arrv2 &, sArray_t<Arrv3> &, chol_reader_t &, int , bool);

    // all combinations, just in case, might be overkill
    template void gw_t::evaluate_Sigma(size_t, const Arr &, const Arr3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv &, const Arr3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv2 &, const Arr3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);

    template void gw_t::evaluate_Sigma(size_t, const Arr &, const Arrv3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv &, const Arrv3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv2 &, const Arrv3 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);

    template void gw_t::evaluate_Sigma(size_t, const Arr &, const Arrv3_2 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv &, const Arrv3_2 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);
    template void gw_t::evaluate_Sigma(size_t, const Arrv2 &, const Arrv3_2 &, sArray_t<Arrv> &, chol_reader_t &, int , bool);


  } // solvers
} // methods
