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


#include <cmath>

#include "methods/SCF/simple_dyson.h"
#include "nda/h5.hpp"
#include "nda/linalg.hpp"
#include "nda/nda.hpp"

namespace methods {

  template<typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(G_t &_G_shm,
                                 const F_t &_sF_skij, const Sigma_t &_Sigma_shm, double mu) {
    _Timer.start("DYSON");
    using math::nda::make_distributed_array;
    using Array_5D_t = nda::array<ComplexType, 5>;

    // processor grid for Dyson equation
    std::array<long, 5> w_pgrid;
    std::array<long, 5> w_bsize;
    {
      int np = _context->comm.size();
      int nwpools = utils::find_proc_grid_max_npools(np, _nw, 0.4);
      np /= nwpools;
      int nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.4);
      np /= nkpools;
      int np_i = utils::find_proc_grid_min_diff(np, 1, 1);
      int np_j = np / np_i;

      w_pgrid = {nwpools, 1, nkpools, np_i, np_j};
      long ibsize = std::min({1024, _nbnd/np_i, _nbnd/np_j});
      w_bsize = {1, 1, 1, ibsize, ibsize};

      utils::check(nwpools*nkpools*np_i*np_j == _context->comm.size(), "solve_dyson: pgrid mismatches!");

      app_log(2, "Dyson equation for Green's function:");
      app_log(2, "  - processor grid for G/Self-energy: (w, k, i, j) = ({}, {}, {}, {})", nwpools, nkpools, np_i, np_j);
      app_log(2, "  - block size: (w, k, i, j) = ({}, {}, {}, {})", 1, 1, ibsize, ibsize);
    }

    _Timer.start("SIGMA_TAU_TO_W");
    auto dSigma_wskij = distributed_tau_to_w(_context->comm, _Sigma_shm, *_FT, w_pgrid, w_bsize);
    _Timer.stop("SIGMA_TAU_TO_W");
    auto dG_wskij = make_distributed_array<Array_5D_t>(_context->comm, w_pgrid,
                                                       {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd}, w_bsize);
    auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dSigma_wskij.local_shape();
    auto [w_org, s_org, k_org, i_org, j_org] = dSigma_wskij.origin();
    auto i_rng = dSigma_wskij.local_range(3);
    auto j_rng = dSigma_wskij.local_range(4);

    // Setup wk_intra_comm
    int color = w_org*_nkpts_ibz + k_org;
    int key = _context->comm.rank();
    mpi3::communicator wk_intra_comm = _context->comm.split(color, key);
    utils::check(wk_intra_comm.size() == w_pgrid[3]*w_pgrid[4], "wk_intra_comm.size() != pgrid[3]*pgrid[4]");
    auto dX = make_distributed_array<nda::array<ComplexType, 2>>(wk_intra_comm, {w_pgrid[3],w_pgrid[4]},
        {_nbnd,_nbnd}, {w_bsize[3],w_bsize[4]});

    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto Sigma_w_loc = dSigma_wskij.local();
    auto G_w_loc = dG_wskij.local();
    auto X_loc = dX.local();

    // Dyson on w-axis
    _Timer.start("DYSON_LOOP");
    for (long nsk = 0; nsk < nw_loc*ns_loc*nk_loc; ++nsk) {
      long n = nsk / (ns_loc*nk_loc); // nsk = n*ns_loc*nk_loc + s*nk_loc + k
      long s = (nsk / nk_loc) % ns_loc;
      long k = nsk % nk_loc;

      long wn = _FT->wn_mesh()(n+w_org);
      ComplexType omega_mu = _FT->omega(wn) + mu;
      X_loc = omega_mu * S(s+s_org,k+k_org,i_rng,j_rng) - H0(s+s_org,k+k_org,i_rng,j_rng)
              - F(s+s_org,k+k_org,i_rng,j_rng) - Sigma_w_loc(n,s,k,nda::ellipsis{});
      math::nda::slate_ops::inverse(dX);
      G_w_loc(n,s,k,nda::ellipsis{}) = X_loc;
    }
    _Timer.stop("DYSON_LOOP");
    dSigma_wskij.reset();
    _context->comm.barrier();

    // G(w) -> G(tau)
    {
      int np = _context->comm.size();
      long nkpools = utils::find_proc_grid_max_npools(np, _nkpts_ibz, 0.2);
      np /= nkpools;
      long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
      long np_j = np / np_i;

      auto dG_wskij_tmp = make_distributed_array<Array_5D_t>(_context->comm, {1, 1, nkpools, np_i, np_j},
                                                             {_nw, _ns, _nkpts_ibz, _nbnd, _nbnd});
      _Timer.start("REDISTRIBUTE");
      math::nda::redistribute(dG_wskij, dG_wskij_tmp);
      _Timer.stop("REDISTRIBUTE");
      dG_wskij.reset(); 

      auto dG_tskij = make_distributed_array<Array_5D_t>(_context->comm, {1, 1, nkpools, np_i, np_j},
                                                         {_nts, _ns, _nkpts_ibz, _nbnd, _nbnd});
      auto Gt_loc = dG_tskij.local();
      auto Gw_loc = dG_wskij_tmp.local();
      _FT->w_to_tau(Gw_loc, Gt_loc, imag_axes_ft::fermion);
      dG_wskij_tmp.reset();

      _FT->check_leakage(dG_tskij, imag_axes_ft::fermion, "Green's function");

      // Gather to shared memory
      _Timer.start("DYSON_GATHER");
      math::nda::gather_to_shm(dG_tskij, _G_shm);
      _Timer.stop("DYSON_GATHER");
    }
    _context->comm.barrier();
    _Timer.stop("DYSON");
    print_timers();
  }

  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void simple_dyson::solve_dyson(Dm_t &_sDm_skij, G_t &_G_shm, const F_t &_sF_skij,
                                 const Sigma_t &_Sigma_shm, double mu) {
    solve_dyson(_G_shm, _sF_skij, _Sigma_shm, mu);
    if (_context->node_comm.root()) {
      auto Dm = _sDm_skij.local();
      _FT->tau_to_beta(_G_shm.local(), Dm);
      Dm *= -1;
    }
    _context->comm.barrier();
  }


  template<typename X_t, typename Xt_t>
  void simple_dyson::compute_eigenspectra(
    const X_t&_sF_skij, 
    const Xt_t &_Sigma_shm, 
    nda::array<ComplexType, 4> &spectra){

    utils::check(spectra.shape() == std::array<long, 4>{_nw, _ns, _nkpts_ibz, _nbnd},
                 "simple_dyson::compute_eigenspectra: Incorrect dimension for spectra.");
    using math::shm::make_shared_array;
    
    spectra() = 0.0;
    nda::matrix<ComplexType> Sigmaw_ij(_nbnd, _nbnd);
    nda::matrix<ComplexType> FpSigma(_nbnd, _nbnd);
    auto& SFS = Sigmaw_ij;
    auto sS_inv = make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd});
    auto Sigma_tskij = _Sigma_shm.local();
    auto S  = _sS_skij.local();
    auto H0 = _sH0_skij.local();
    auto F  = _sF_skij.local();
    auto S_inv = sS_inv.local();

    int node_rank = _context->node_comm.rank();
    int node_size = _context->node_comm.size();
    sS_inv.win().fence();
    for (size_t sk = node_rank; sk < _ns*_nkpts_ibz; sk+=node_size) {
      size_t is = sk / _nkpts_ibz;
      size_t ik = sk % _nkpts_ibz;
      nda::matrix_const_view<ComplexType> S_ij = S(is, ik, nda::ellipsis{});
      S_inv(is, ik, nda::ellipsis{}) = nda::inverse(S_ij);
    }
    sS_inv.win().fence();

    auto nw_half = _nw/2 + _nw%2;

    // Define M(iw) = S^{-1} * [H0 + F + Sigma(iw)], we have M(-iw) = S^{-1} * M(iw)^{\dagger} * S and thus 
    // eigvals(M(-iw)) = conj(eigvals(M(iw))). Thus we only need to compute eigvals for half of the frequencies. 
    // This relation holds for general non-orthogonal basis with general oeverlap matrix S. 
    for (size_t nsk = _context->comm.rank(); nsk < nw_half*_ns*_nkpts_ibz; nsk+=_context->comm.size()) {

      // nsk = n_neg*ns*nkpts_ibz + is*nkpts_ibz + k
      auto n_neg = nsk / (_ns*_nkpts_ibz); 
      auto n_pos = _nw - n_neg - 1;
      auto is = (nsk / _nkpts_ibz) % _ns;
      auto ik = nsk % _nkpts_ibz;

      auto Sigma_tij = nda::make_regular(Sigma_tskij(nda::range::all, is, ik, nda::ellipsis{}));
      _FT->tau_to_w(Sigma_tij, Sigmaw_ij, imag_axes_ft::fermion, n_pos);

      FpSigma = H0(is, ik, nda::ellipsis{}) + F(is, ik, nda::ellipsis{}) + Sigmaw_ij;

      nda::blas::gemm(ComplexType(1.0), S_inv(is, ik, nda::ellipsis{}), FpSigma, ComplexType(0.0), SFS);

      // Matsubara quantities are not Hermitian!
      spectra(n_pos, is, ik, nda::range::all) = nda::linalg::geigenvalues(SFS);

      // exploit eigvals(M(-iw)) = conj(eigvals(M(iw)))
      if (n_neg != n_pos) {
        spectra(n_neg, is, ik, nda::range::all) = nda::conj(spectra(n_pos, is, ik, nda::range::all));
      }
    }
    _context->comm.all_reduce_in_place_n(spectra.data(), spectra.size(), std::plus<>{});
  }




  /** Instantiation of public template **/
  template void simple_dyson::solve_dyson(sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);
  template void simple_dyson::solve_dyson(sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&,
      const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);

  template void simple_dyson::compute_eigenspectra(const sArray_t<Array_view_4D_t>&,
      const sArray_t<Array_view_5D_t> &, nda::array<ComplexType, 4> &);

} // methods
