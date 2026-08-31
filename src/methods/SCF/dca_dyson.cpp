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


#include "dca_dyson.h"
#include "nda/linalg/eigenelements.hpp"

namespace methods {
dca_dyson::dca_dyson(utils::mpi_context_t<mpi3::communicator> &context, mf::MF *mf,
           imag_axes_ft::IAFT *ft, mf::MF &dca_mf, double mu_tol,
           std::string mu_update_alg) :
  _context(context), _MF(mf), _FT(ft), _PSP(hamilt::make_pseudopot(*_MF)), _nk_tilde(dca_mf.nkpts()/mf->nkpts()), _mu_tol(mu_tol),
  _mu_update_alg(mu_update_alg),
    _sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                            {mf->nspin(), mf->nkpts(), mf->nbnd(), mf->nbnd()})),
    _sS_skij(math::shm::make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                           {mf->nspin(), mf->nkpts(), mf->nbnd(), mf->nbnd()})),
    _sH0_lattice_sKij(math::shm::make_shared_array<Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm, {mf->nspin(), dca_mf.nkpts(), mf->nbnd(), mf->nbnd()})),
    _sS_lattice_sKij(math::shm::make_shared_array<Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm, {mf->nspin(), dca_mf.nkpts(), mf->nbnd(), mf->nbnd()})),
    _Timer() {
  utils::check(_MF->nkpts() == _MF->nkpts_ibz(), 
               "dca_dyson: Symmetry not yet implemented.");
  if (_context.comm.size()% _context.node_comm.size()!=0) {
    APP_ABORT("DCA: number of processors on each node should be the same.");
  }
  if ((dca_mf.nkpts() % mf->nkpts()) != 0 and mf->nspin()!= dca_mf.nspin() and mf->nbnd() != dca_mf.nbnd()) {
    APP_ABORT("DCA: DCA and Cluster mean field are different.");
  }

  // one-body Hamiltonian
  {
    long np = _context.comm.size();
    long np_s = (np % mf->nspin()== 0)? mf->nspin() : 1;
    long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)mf->nkpts(), 1.0);
    long np_i = np / (np_s*np_k);
    std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
    long blk_i = std::min( {(long)1024, mf->nbnd()/np_i});
    std::array<long, 4> bsize = {1, 1, blk_i, 1};
    app_log(3, "One-body Hamiltonian in distributed array: ");
    app_log(3, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
    app_log(3, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 1);
    auto dH0 = hamilt::H0<HOST_MEMORY>(*_MF, _context.comm, _PSP.get(), 
                                       nda::range(_MF->nkpts()), nda::range(_MF->nbnd()), pgrid, bsize);
    // gather at root, then broadcast to all nodes
    auto H0_loc = _sH0_skij.local();
    math::nda::gather(0, dH0, &H0_loc);
    if (_context.node_comm.root())
      _context.internode_comm.broadcast_n(H0_loc.data(), H0_loc.size(), 0);
  }
  long np = _context.comm.size();
  long np_s = (np % mf->nspin()== 0)? mf->nspin() : 1;
  long np_k = utils::find_proc_grid_max_npools(np/np_s, (long)dca_mf.nkpts(), 1.0);
  long np_i = np / (np_s*np_k);
  std::array<long, 4> pgrid = {np_s, np_k, np_i, 1};
  long blk_i = std::min( {(long)1024, mf->nbnd()/np_i});
  std::array<long, 4> bsize = {1, 1, blk_i, 1};
  app_log(3, "One-body full-lattice Hamiltonian in distributed array: ");
  app_log(3, "  - pgrid = ({}, {}, {}, {})", np_s, np_k, np_i, 1);
  app_log(3, "  - bsize = ({}, {}, {}, {})\n", 1, 1, blk_i, 1);
  {
    auto dH0_lat = hamilt::H0<HOST_MEMORY>(dca_mf, _context.comm, _PSP.get(),  
                                           nda::range(dca_mf.nkpts()), nda::range(_MF->nbnd()), pgrid, bsize);
    auto H0_latt_loc = _sH0_lattice_sKij.local();
    math::nda::gather(0, dH0_lat, &H0_latt_loc);
    if (_context.node_comm.root())
      _context.internode_comm.broadcast_n(H0_latt_loc.data(), H0_latt_loc.size(), 0);
  }
  {
    auto dS = hamilt::ovlp<HOST_MEMORY>(dca_mf, _context.comm,
                                        nda::range(dca_mf.nkpts()), nda::range(_MF->nbnd()), pgrid, bsize);
    auto S_loc = _sS_lattice_sKij.local();
    math::nda::gather(0, dS, &S_loc);
    if (_context.node_comm.root())
      _context.internode_comm.broadcast_n(S_loc.data(), S_loc.size(), 0);
    _context.comm.barrier();
  }

  for( auto& v: {"MBPT_SOLVERS",
                  "DYSON", "SIGMA_W", "DYSON_LOOP", "REDISTRIBUTE", "W_TO_TAU", "DYSON_GATHER",
                  "G_BROADCAST", "DM",
                  "FIND_MU", "SCF_LOOP", "ENERGY", "WRITE"} ) {
    _Timer.add(v);
  }

  app_log(1, "*******************************");
  app_log(1, " COQUI DYSON SCF ");
  app_log(1, "*******************************");
  app_log(1, "    - Total number of processors: {}", _context.comm.size());
  app_log(1, "    - Number of nodes: {}", _context.internode_comm.size());
  app_log(1, "    - Number of processors per node: {}\n", _context.node_comm.size());
  _context.comm.barrier();
}

template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
void dca_dyson::solve_dyson(Dm_t&_sDm_skij, G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu) {
  _Timer.start("DYSON");
  using math::nda::make_distributed_array;
  using Array_5D_t = nda::array<ComplexType, 5>;

  // processor grid for Dyson equation
  std::array<long, 5> w_pgrid;
  std::array<long, 5> w_bsize;
  {
    int np = _context.comm.size();
    int nwpools = utils::find_proc_grid_max_npools(np, _FT->nw_f(), 0.4);
    np /= nwpools;
    int nkpools = utils::find_proc_grid_max_npools(np, _MF->nkpts(), 0.4);
    np /= nkpools;
    int np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    int np_j = np / np_i;

    w_pgrid = {nwpools, 1, nkpools, np_i, np_j};
    long ibsize = std::min({1024l, _MF->nbnd()/np_i, _MF->nbnd()/np_j});
    w_bsize = {1, 1, 1, ibsize, ibsize};

    utils::check(nwpools*nkpools*np_i*np_j == _context.comm.size(), "solve_dyson: pgrid mismatches!");

    app_log(2, "Dyson equation for Green's function:");
    app_log(2, "  - processor grid for G/Self-energy: (w, k, i, j) = ({}, {}, {}, {})", nwpools, nkpools, np_i, np_j);
    app_log(2, "  - block size: (w, k, i, j) = ({}, {}, {}, {})", 1, 1, ibsize, ibsize);
  }

  auto dSigma_wskij = distributed_tau_to_w(_context.comm, _Sigma_shm, *_FT, w_pgrid, w_bsize);
  auto dG_wskij = make_distributed_array<Array_5D_t>(_context.comm, w_pgrid,
                                                     {_FT->nw_f(), _MF->nspin(), _MF->nkpts(), _MF->nbnd(), _MF->nbnd()}, w_bsize);
  auto [nw_loc, ns_loc, nk_loc, ni_loc, nj_loc] = dSigma_wskij.local_shape();
  auto [w_org, s_org, k_org, i_org, j_org] = dSigma_wskij.origin();
  auto i_rng = dSigma_wskij.local_range(3);
  auto j_rng = dSigma_wskij.local_range(4);

  // Setup wk_intra_comm
  int color = w_org*_MF->nkpts() + k_org;
  int key = _context.comm.rank();
  mpi3::communicator wk_intra_comm = _context.comm.split(color, key);
  utils::check(wk_intra_comm.size() == w_pgrid[3]*w_pgrid[4], "wk_intra_comm.size() != pgrid[3]*pgrid[4]");
  auto dX = make_distributed_array<nda::array<ComplexType, 2>>(wk_intra_comm, {w_pgrid[3],w_pgrid[4]},
                                                               {_MF->nbnd(), _MF->nbnd()}, {w_bsize[3],w_bsize[4]});
  auto S  = _sS_lattice_sKij.local();
  auto H0 = _sH0_lattice_sKij.local();
  auto F_loc  = _sF_skij.local();
  auto Sigma_w_loc = _Sigma_shm.local();
  auto G_w_loc = _G_shm.local();
  auto X_loc = dX.local();

  // Dyson on w-axis
  _Timer.start("DYSON_LOOP");
  for (long nsk = 0; nsk < nw_loc*ns_loc*nk_loc; ++nsk) {
    long n = nsk / (ns_loc*nk_loc); // nsk = n*ns_loc*nk_loc + s*nk_loc + k
    long s = (nsk / nk_loc) % ns_loc;
    long k = nsk % nk_loc;

    G_w_loc(n,s,k,nda::ellipsis{}) = 0.0;
    long wn = _FT->wn_mesh()(n+w_org);
    ComplexType omega_mu = _FT->omega(wn) + mu;
    for (long k_tilde = 0; k_tilde < _nk_tilde; ++k_tilde) {
      X_loc = omega_mu * S(s+s_org,(k+k_org)*_nk_tilde+k_tilde,i_rng,j_rng) - H0(s+s_org,(k+k_org)*_nk_tilde+k_tilde,i_rng,j_rng)
            - F_loc(s+s_org,k+k_org,i_rng,j_rng) - Sigma_w_loc(n,s,k,nda::ellipsis{});
      math::nda::slate_ops::inverse(dX);
      G_w_loc(n,s,k,nda::ellipsis{}) += X_loc/double(_nk_tilde);
    }
  }
  _Timer.stop("DYSON_LOOP");
  dSigma_wskij.reset();
  _context.comm.barrier();

  // G(w) -> G(tau)
  {
    int np = _context.comm.size();
    long nkpools = utils::find_proc_grid_max_npools(np, _MF->nkpts(), 0.2);
    np /= nkpools;
    long np_i = utils::find_proc_grid_min_diff(np, 1, 1);
    long np_j = np / np_i;

    auto dG_wskij_tmp = make_distributed_array<Array_5D_t>(_context.comm, {1, 1, nkpools, np_i, np_j},
                                                           {_FT->nw_f(), _MF->nspin(), _MF->nkpts(), _MF->nbnd(), _MF->nbnd()});
    _Timer.start("REDISTRIBUTE");
    math::nda::redistribute(dG_wskij, dG_wskij_tmp);
    _Timer.stop("REDISTRIBUTE");
    dG_wskij.reset();

    auto dG_tskij = make_distributed_array<Array_5D_t>(_context.comm, {1, 1, nkpools, np_i, np_j},
                                                       {_FT->nt_f(), _MF->nspin(), _MF->nkpts(), _MF->nbnd(), _MF->nbnd()});
    auto Gt_loc = dG_tskij.local();
    auto Gw_loc = dG_wskij_tmp.local();
    _FT->w_to_tau(Gw_loc, Gt_loc, imag_axes_ft::fermion);
    dG_wskij_tmp.reset();

    _FT->check_leakage(dG_tskij, imag_axes_ft::fermion, "Green's function");

    // Gather to shared memory
    auto G_shm = _G_shm.local();
    _Timer.start("DYSON_GATHER");
    for (int node = 0; node < _context.internode_comm.size(); ++node) {
      int node_root = 0;
      if (_context.internode_comm.rank() == node and
          _context.node_comm.root()) {
        node_root = _context.comm.rank();
      }
      _context.comm.all_reduce_in_place_n(&node_root, 1,std::plus<>{});
      math::nda::gather(node_root, dG_tskij, &G_shm);
    }
    _Timer.stop("DYSON_GATHER");
  }

  _Timer.start("DM");
  if (_context.node_comm.root()) {
    auto Dm_loc = _sDm_skij.local();
    _FT->tau_to_beta(_G_shm.local(), Dm_loc);
    Dm_loc *= -1;
  }
  _Timer.stop("DM");

  _context.comm.barrier();
  _Timer.stop("DYSON");
};

template<typename X_t, typename Xt_t>
void dca_dyson::compute_eigenspectra(double mu, [[maybe_unused]] const X_t&_sF_skij, const Xt_t &_G_shm, [[maybe_unused]] const Xt_t &_Sigma_shm, nda::array<ComplexType, 4> &spectra) {
  utils::check(spectra.shape() == std::array<long, 4>{_FT->nw_f(), _MF->nspin(), _MF->nkpts(), _MF->nbnd()},
               "dca_dyson::compute_eigenspectra: Incorrect dimension for spectra.");
  using math::shm::make_shared_array;
  auto sS_inv = math::shm::make_shared_array<Array_view_4D_t>(
      _context.comm, _context.internode_comm, _context.node_comm,
                                                         {_MF->nspin(), _MF->nkpts(), _MF->nbnd(), _MF->nbnd()});
  // Compute overlap matrix for cluster Green's function
  {
    int node_rank = _context.node_comm.rank();
    int node_size = _context.node_comm.size();
    auto S_inv = sS_inv.local();
    auto S = _sS_skij.local();
    sS_inv.win().fence();
    for (size_t sk = node_rank; sk < _MF->nspin() * _MF->nkpts(); sk += node_size) {
      size_t is = sk / _MF->nkpts();
      size_t ik = sk % _MF->nkpts();
      nda::matrix_const_view<ComplexType> S_ij = S_inv(is, ik, nda::ellipsis{});
      S(is, ik, nda::ellipsis{}) = nda::inverse(S_ij);
    }
    sS_inv.win().fence();
  }
  spectra() = 0.0;
  // TODO shared memory or distributed array!!!
  nda::array<ComplexType, 4> Gw_skij(_MF->nspin(), _MF->nkpts(), _MF->nbnd(), _MF->nbnd());
  nda::matrix<ComplexType> FpSigma(_MF->nbnd(), _MF->nbnd());

  nda::matrix<ComplexType> SFS(_MF->nbnd(), _MF->nbnd());
  auto S = _sS_skij.local();
  auto S_inv = sS_inv.local();

  for (size_t n = _context.comm.rank(); n < _FT->nw_f(); n+= _context.comm.size()) {
    _FT->tau_to_w(_G_shm.local(), Gw_skij, imag_axes_ft::fermion, n);
    long wn = _FT->wn_mesh()(n);
    ComplexType omega_mu = _FT->omega(wn) + mu;
    for (size_t i = 0; i < _MF->nspin()*_MF->nkpts(); ++i) {
      size_t is = i / _MF->nkpts(); // i = is * _nkpts + ik
      size_t ik = i % _MF->nkpts();
      nda::matrix_const_view<ComplexType> G_ij = Gw_skij(is, ik, nda::ellipsis{});
      FpSigma = omega_mu * S(is, ik, nda::range::all, nda::range::all) - nda::inverse(G_ij);
      nda::blas::gemm(ComplexType(1.0), S_inv(is, ik, nda::range::all, nda::range::all), FpSigma,
                      ComplexType(0.0), SFS);

      auto eigvals = spectra(n, is, ik, nda::range::all);
      // Matsubara quantities are not Hermitian!
      eigvals = nda::linalg::geigenvalues(SFS);
    }
  }
  _context.comm.all_reduce_in_place_n(spectra.data(), spectra.size(), std::plus<>{});
}

template void dca_dyson::compute_eigenspectra(double, const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_5D_t> &, const sArray_t<Array_view_5D_t> &, nda::array<ComplexType, 4> &);
template void dca_dyson::solve_dyson(sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t> &, double);
}
