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


#include "nda/linalg.hpp"
#include "nda/linalg/det_and_inverse.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/ac/AC_t.hpp"

#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "methods/tools/chkpt_utils.h"
#include "methods/ERI/mb_eri_context.h"
#include "methods/SCF/mb_solver_t.h"
#include "methods/SCF/scf_common.hpp"

namespace methods {

auto get_mf_MOs(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, hamilt::pseudopot &psp)
  -> std::tuple<sArray_t<Array_view_4D_t>, sArray_t<Array_view_3D_t> > {
  using math::shm::make_shared_array;
  auto sF_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                    {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  hamilt::set_fock(mf, std::addressof(psp), sF_skij, false);
  auto sS_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                    {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  hamilt::set_ovlp(mf, sS_skij);

  auto sMO_skij = make_shared_array<Array_view_4D_t>(context.comm, context.internode_comm, context.node_comm,
                                                     {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()});
  auto sE_ski = make_shared_array<Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                   {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()});
  update_MOs(sMO_skij, sE_ski, sF_skij, sS_skij);

  return std::make_tuple(std::move(sMO_skij), std::move(sE_ski));
}

void update_MOs(sArray_t<Array_view_4D_t> &sMO_skij, sArray_t<Array_view_3D_t> &sE_ski,
                const sArray_t<Array_view_4D_t> &sF_skij, const sArray_t<Array_view_4D_t> &sS_skij) {
  auto [ns, nkpts, nbnd, nbnd2] = sF_skij.shape();
  sMO_skij.win().fence();
  sE_ski.win().fence();
  for (long sk = sF_skij.node_comm()->rank(); sk < ns*nkpts; sk+=sF_skij.node_comm()->size()) {
    long is = sk / nkpts;
    long ik = sk % nkpts;
    auto F         = sF_skij.local()(is, ik, nda::ellipsis{});
    auto S         = sS_skij.local()(is, ik, nda::ellipsis{});

    auto [mo_e, coeffs] = nda::linalg::eigenelements(F, S);
    sMO_skij.local()(is, ik, nda::ellipsis{}) = coeffs;
    for (long i = 0; i < sE_ski.shape()[2]; ++i) sE_ski.local()(is, ik, i) = mo_e(i);
  }
  sMO_skij.win().fence();
  sE_ski.win().fence();
}

void update_Dm(sArray_t<Array_view_4D_t> &sDm_skij,
               const sArray_t<Array_view_4D_t> &sMO_skij, const sArray_t<Array_view_3D_t> &sE_ski,
               const double mu, const double beta) {
  auto FD_occ = nda::map([&](ComplexType e) { return 1.0 / ( 1 + std::exp( (e.real()-mu) * beta ) ); });

  auto [ns, nkpts, nbnd, nbnd2] = sDm_skij.shape();
  nda::array<RealType, 1> occ(nbnd);
  nda::array<ComplexType, 2> fCdag(nbnd, nbnd);

  sDm_skij.win().fence();
  for (size_t sk = sDm_skij.node_comm()->rank(); sk < ns*nkpts; sk += sDm_skij.node_comm()->size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;

    occ = FD_occ(sE_ski.local()(is, ik, nda::range::all));
    fCdag = nda::transpose(nda::conj(sMO_skij.local()(is, ik, nda::ellipsis{})));
    for (size_t i = 0; i < nbnd; ++i) {
      fCdag(i, nda::range::all) *= occ(i);
    }

    auto C = sMO_skij.local()(is, ik, nda::ellipsis{});
    auto Dm = sDm_skij.local()(is, ik, nda::ellipsis{});
    nda::blas::gemm(C, fCdag, Dm);
  }
  sDm_skij.win().fence();
}

void update_G(sArray_t<Array_view_5D_t> &sG_tskij,
              const sArray_t<Array_view_4D_t> &sMO_skia, const sArray_t<Array_view_3D_t> &sE_ska,
              double mu, const imag_axes_ft::IAFT &FT) {
  double beta = FT.beta();
  auto compute_G0 = [&](double e, double t) {
    double x = e-mu;
    if (x > 0) {
      return -std::exp( -x*t ) / (1 + std::exp( -x*beta ));
    } else {
      return -std::exp( x*(beta-t) ) / (1 + std::exp( x*beta ));
    }
  };

  auto [nts, ns, nkpts, nbnd, nbnd2] = sG_tskij.shape();
  auto x_mesh = FT.tau_mesh();
  auto x_to_tau = nda::map([&](double x) { return (x+1) * FT.beta()/2.0; });
  nda::array<double, 1> tau_mesh = x_to_tau(x_mesh);
  nda::array<ComplexType, 2> GCdag_aj(nbnd, nbnd);

  sG_tskij.win().fence();
  for (size_t tsk = sG_tskij.node_comm()->rank(); tsk < nts*ns*nkpts; tsk += sG_tskij.node_comm()->size()) {
    size_t it = tsk / (ns*nkpts); // tsk = it*ns*nkpts + is*nkpts + ik
    size_t is = (tsk / nkpts) % ns;
    size_t ik = tsk % nkpts;

    GCdag_aj = nda::transpose(nda::conj(sMO_skia.local()(is, ik, nda::ellipsis{})));
    for (size_t a = 0; a < nbnd; ++a) {
      GCdag_aj(a, nda::range::all) *= compute_G0(sE_ska.local()(is, ik, a).real(), tau_mesh(it));
    }

    auto C_ia = sMO_skia.local()(is, ik, nda::ellipsis{});
    auto G_ij = sG_tskij.local()(it, is, ik, nda::ellipsis{});
    nda::blas::gemm(C_ia, GCdag_aj, G_ij);
  }
  sG_tskij.win().fence();
  sG_tskij.communicator()->barrier();
}

template<nda::ArrayOfRank<5> Array_base_t>
void compute_G_from_mf(h5::group iter_grp, imag_axes_ft::IAFT &ft,
                       sArray_t<Array_base_t> &sG_tskij) {
  using math::shm::make_shared_array;
  long ns = sG_tskij.shape()[1];
  long nkpts_ibz = sG_tskij.shape()[2];
  long nbnd = sG_tskij.shape()[3];
  // Construct the Green's function for a mean-field solution on-the-fly
  auto sMO_skia = make_shared_array<Array_view_4D_t>(
      *sG_tskij.communicator(), *sG_tskij.internode_comm(), *sG_tskij.node_comm(),
      {ns, nkpts_ibz, nbnd, nbnd});
  auto sE_ska = make_shared_array<Array_view_3D_t>(
      *sG_tskij.communicator(), *sG_tskij.internode_comm(), *sG_tskij.node_comm(),
      {ns, nkpts_ibz, nbnd});
  double mu;

  sMO_skia.win().fence();
  if (sG_tskij.node_comm()->root()) {
    auto MO_loc = sMO_skia.local();
    auto E_loc = sE_ska.local();
    nda::h5_read(iter_grp, "MO_skia", MO_loc);
    nda::h5_read(iter_grp, "E_ska", E_loc);
  }
  sMO_skia.win().fence();
  h5::h5_read(iter_grp, "mu", mu);

  update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
  sG_tskij.communicator()->barrier();
}

void solve_qp_eqn(sArray_t<Array_view_3D_t> &sE_ska,
                  const sArray_t<Array_view_5D_t> &sSigma_tskij,
                  const sArray_t<Array_view_4D_t> &sVhf_skij,
                  const sArray_t<Array_view_4D_t> &sMO_skia,
                  double mu,
                  const imag_axes_ft::IAFT &FT, qp_params_t &qp_params) {
  using math::shm::make_shared_array;
  using math::nda::make_distributed_array;
  using local_Array_4D_t = nda::array<ComplexType, 4>;
  using local_Array_3D_t = nda::array<ComplexType, 3>;

  auto comm = sE_ska.communicator();
  auto [ns, nkpts, nbnd, nbnd2] = sVhf_skij.shape();
  auto nt = FT.nt_f();
  auto nw = FT.nw_f();

  int np = comm->size();
  // nbnd > np -> nkpools = 1, np_a = np
  // np_a = np / nkpools <= nbnd -> nkpools > np/nbnd
  int nkpools = utils::find_min_col(np, nbnd, (np%nbnd==0)? np/nbnd : np/nbnd+1);
  int np_a    = np / nkpools;
  utils::check(np_a <= nbnd, "solve_qp_eqn: np_a({}) > nbnd({})", np_a, nbnd);
  utils::check(nkpools <= nkpts, "solv_qp_eqn: nkpools({}) > nkpts({})", nkpools, nkpts);
  utils::check(comm->size() % nkpools == 0, "solve_qp_eqn: comm.size({}) % nkpools({}) != 0", np, nkpools);

  auto dSigma_wska = make_distributed_array<local_Array_4D_t>(*comm, {1, 1, nkpools, np_a}, {nw, ns, nkpts, nbnd}, {1, 1, 1, 1});
  auto s_rng = dSigma_wska.local_range(1);
  auto k_rng = dSigma_wska.local_range(2);
  auto a_rng = dSigma_wska.local_range(3);
  auto [nw_loc, ns_loc, nk_loc, na_loc] = dSigma_wska.local_shape();


  // ------ basis transform from primary to MO basis ------
  {
    auto dSigma_tska = make_distributed_array<local_Array_4D_t>(
        *comm, {1, 1, nkpools, np_a}, {nt, ns, nkpts, nbnd}, {1, 1, 1, 1});
    auto Sigma_tska_loc = dSigma_tska.local();

    nda::array<ComplexType, 2> SigmaC_ia(nbnd, na_loc);
    nda::array<ComplexType, 2> Sigma_ab(na_loc, na_loc);
    for (size_t it = 0; it < nt; ++it) {
      for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
        for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {

          nda::blas::gemm(sSigma_tskij.local()(it, is, ik, nda::ellipsis{}),
                          sMO_skia.local()(is, ik, nda::range::all, a_rng),
                          SigmaC_ia);
          nda::blas::gemm(ComplexType(1.0),
                          nda::dagger(sMO_skia.local()(is, ik, nda::range::all, a_rng)),
                          SigmaC_ia,
                          ComplexType(0.0),
                          Sigma_ab);
          Sigma_tska_loc(it, is_loc, ik_loc, nda::range::all) = nda::diagonal(Sigma_ab);
        }
      }
    }
    FT.tau_to_w(dSigma_tska.local(), dSigma_wska.local(), imag_axes_ft::fermion);
  }

  // ------ basis transformation from primary to MO basis ------
  auto dVhf_ska = make_distributed_array<local_Array_3D_t>(*comm, {1, nkpools, np_a}, {ns, nkpts, nbnd}, {1, 1, 1});
  auto dE_ska = make_distributed_array<local_Array_3D_t>(*comm, {1, nkpools, np_a}, {ns, nkpts, nbnd}, {1, 1, 1});
  {
    nda::array<ComplexType, 2> VC_ia(nbnd, na_loc);
    nda::array<ComplexType, 2> V_ab(na_loc, na_loc);
    auto Vhf_ska_loc = dVhf_ska.local();
    for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
      for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {

        nda::blas::gemm(sVhf_skij.local()(is, ik, nda::ellipsis{}),
                        sMO_skia.local()(is, ik, nda::range::all, a_rng),
                        VC_ia);
        nda::blas::gemm(ComplexType(1.0),
                        nda::dagger(sMO_skia.local()(is, ik, nda::range::all, a_rng)),
                        VC_ia,
                        ComplexType(0.0),
                        V_ab);
        Vhf_ska_loc(is_loc, ik_loc, nda::range::all) = nda::diagonal(V_ab);
      }
    }

    auto E_loc = dE_ska.local();
    E_loc = sE_ska.local()(s_rng, k_rng, a_rng);
  }

  // ------ Solve quasi-particle equation for Heff ------
  {
    long dim1 = ns_loc * nk_loc * na_loc;
    auto E_loc_1D = nda::reshape(dE_ska.local(), std::array<long, 1>{dim1});
    auto Vhf_loc_1D = nda::reshape(dVhf_ska.local(), std::array<long, 1>{dim1});
    auto Sigma_loc_2D = nda::reshape(dSigma_wska.local(), std::array<long, 2>{nw, dim1});

    // bypass clang openmp error: error: capturing a structured binding is not yet supported in OpenMP
    auto nk_loc_ = nk_loc;
    auto na_loc_ = na_loc;
    auto I_to_ska = [&](size_t I) {
      // I = s_loc*nk_loc*na_loc + k_loc*na_loc + a_loc
      size_t s_loc = I / (nk_loc_*na_loc_);
      size_t k_loc = (I/na_loc_) % nk_loc_;
      size_t a_loc = I % na_loc_;
      return std::make_tuple(s_rng.first()+s_loc, k_rng.first()+k_loc, a_rng.first()+a_loc);
    };

    analyt_cont::AC_t AC(qp_params.ac_alg);
    auto n_to_iw = nda::map([&](int n) { return FT.omega(n); });
    nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));

    app_log(2, "\n* Solving quasiparticle equation for given Sigma(iw): ");
    app_log(2, "  - processor grid for quasi-particle equation: (s, k, a) = ({}, {}, {})", 1, nkpools, np_a);
    app_log(2, "  - quasi-particle equation algorithm:          {}", qp_params.qp_type);
    app_log(2, "  - ac algorithm:                               {}", qp_params.ac_alg);
    app_log(2, "  - eta:                                        {}", qp_params.eta);
    app_log(2, "  - tolerance for quasi-particle equation:      {}\n", qp_params.tol);
    AC.init(iw_mesh, Sigma_loc_2D, qp_params.Nfit);
    if (qp_params.qp_type == "sc" or qp_params.qp_type == "sc_bisection") {
      double res;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), res) =
            qp_eqn_bisection(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.tol, qp_params.eta);
      }
    } else if (qp_params.qp_type == "sc_newton") {
      bool conv;
      double res;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), res, conv) =
            qp_eqn_secant(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), 400, qp_params.tol, qp_params.eta);
        if (!conv) {
          auto [is, ik, ia] = I_to_ska(I);
          app_warning("secant method fails to converge at (s,k,a) = ({},{},{}); residual = {}", is, ik, ia, res);
        }
      }
    } else if (qp_params.qp_type == "linearized") {
      for (size_t I = 0; I < dim1; ++I) {
        E_loc_1D(I) = qp_eqn_linearized(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.eta);
      }
    } else if (qp_params.qp_type == "spectral") {
      bool conv;
      for (size_t I = 0; I < dim1; ++I) {
        std::tie(E_loc_1D(I), conv) = qp_eqn_spectral(Vhf_loc_1D(I).real(), AC, I, mu, E_loc_1D(I).real(), qp_params.tol, qp_params.eta);
        if (!conv) {
          auto [is, ik, ia] = I_to_ska(I);
          app_warning("spectral method fails to converge at (s,k,a) = ({},{},{})", is, ik, ia);
        }
      }
    } else {
      utils::check(false, "solve_qp_eqn: unknown type of qp equation: {}", qp_params.qp_type);
    }
  }
  dSigma_wska.reset();
  dVhf_ska.reset();

  sE_ska.set_zero();
  {
    auto E_ska_loc = dE_ska.local();
    sE_ska.win().fence();
    sE_ska.local()(s_rng, k_rng, a_rng) = E_ska_loc;
    sE_ska.win().fence();
    sE_ska.all_reduce();
  }
  dE_ska.reset();
  comm->barrier();
}

template<typename eri_t, typename corr_solver_t>
void add_evscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params, 
                     bool fixed_w) {
  using math::shm::make_shared_array;

  auto mpi = eri.mpi();

  auto& sHeff_skij = mb_state.sHeff_skij.value();
  auto& sMO_skia = mb_state.sMO_skia.value();
  auto& sE_ska = mb_state.sE_ska.value();
  
  auto [ns, nkpts, nbnd, nbnd2] = sHeff_skij.shape();
  auto nt = FT.nt_f();

  // Evaluate dynamic self-energy and solve the quasiparticle equation.
  mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  {
    update_G(mb_state.sG_tskij.value(), sMO_skia, sE_ska, mu, FT);
    FT.check_leakage(mb_state.sG_tskij.value(), imag_axes_ft::fermion, "Green's function");
    
    // update dyanmically screened interaction in mb_state if necessary.
    if (not fixed_w or not mb_state.dW_qtPQ.has_value()) {
      utils::check(mb_solver.scr_eri!=nullptr, "add_evscf_vcorr: mb_solver.scr_eri == nullptr when update_W is true.");
      mb_solver.scr_eri->update_w(mb_state, eri, mb_solver.corr->iter());
    }
    
    mb_solver.corr->evaluate(mb_state, eri);
    FT.check_leakage(mb_state.sSigma_tskij.value(), imag_axes_ft::fermion, "Self-energy");
    mpi->comm.barrier();

    // deallocate dynamically screened interaction if it is not fixed for the next iteration.
    if (not fixed_w) {
      mb_state.dW_qtPQ.reset();
    }
  }

  // Solve the QP equation to get new QP energies. 
  solve_qp_eqn(sE_ska, mb_state.sSigma_tskij.value(), sHeff_skij, sMO_skia, mu, FT, qp_params);
  mb_state.sG_tskij.reset();
  mb_state.sSigma_tskij.reset();

  /** Update sHeff_skij with the new QP energies while keeping MO coefficients the same */
  // Basis transformation back to the primary basis.
  auto sMOinv_skai = make_shared_array<Array_view_4D_t>(*mpi, {ns, nkpts, nbnd, nbnd});
  sMOinv_skai.win().fence();
  for (size_t sk = mpi->node_comm.rank(); sk < ns*nkpts; sk += mpi->node_comm.size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;
    auto MO = make_matrix_view(sMO_skia.local()(is, ik, nda::ellipsis{}));
    sMOinv_skai.local()(is, ik, nda::ellipsis{}) = nda::inverse(MO);
  }
  sMOinv_skai.win().fence();

  sHeff_skij.win().fence();
  if (mpi->node_comm.rank() < ns*nkpts) {
    nda::array<ComplexType, 2> C_aj(nbnd, nbnd);
    nda::array<ComplexType, 2> Cdag_ia(nbnd, nbnd);
    for (size_t sk = mpi->node_comm.rank(); sk < ns * nkpts; sk += mpi->node_comm.size()) {
      size_t is = sk / nkpts;
      size_t ik = sk % nkpts;
      C_aj = sMOinv_skai.local()(is, ik, nda::ellipsis{});
      Cdag_ia = nda::transpose(nda::conj(C_aj));
      for (size_t a = 0; a < nbnd; ++a) {
        C_aj(a, nda::range::all) *= sE_ska.local()(is, ik, a);
      }
      nda::blas::gemm(Cdag_ia, C_aj, sHeff_skij.local()(is, ik, nda::ellipsis{}));
    }
  }
  sHeff_skij.win().fence();
}

auto qp_approx(const sArray_t<Array_view_5D_t> &sSigma_tskij,
               const sArray_t<Array_view_4D_t> &sMO_skia,
               const sArray_t<Array_view_3D_t> &sE_ska, double mu,
               const imag_axes_ft::IAFT &FT, qp_params_t &qp_params)
               -> sArray_t<Array_view_4D_t> {
  using math::shm::make_shared_array;
  using math::nda::make_distributed_array;
  using local_Array_5D_t = nda::array<ComplexType, 5>;

  auto comm = sSigma_tskij.communicator();
  auto internode_comm = sSigma_tskij.internode_comm();
  auto node_comm = sSigma_tskij.node_comm();
  auto [ns, nkpts, nbnd] = sE_ska.shape();
  auto nt = FT.nt_f();
  auto nw = FT.nw_f();

  int np = comm->size();
  long nkpools = utils::find_proc_grid_max_npools(np, nkpts, 0.2);
  np /= nkpools;
  long np_a = utils::find_proc_grid_min_diff(np, 1, 1);
  long np_b = np / np_a;
  utils::check(nkpools > 0 and nkpools <= nkpts,
               "qp_approx:: nkpools <= 0 or nkpools > nkpts. nkpools = {}", nkpools);
  utils::check(comm->size() % nkpools == 0, "qp_approx:: gcomm.size() % nkpools != 0");
  utils::check(np_a < nbnd and np_b < nbnd, "qp_approx: np_a({}) or np_b({}) > nbnd({})", np_a, np_b, nbnd);

  auto dSigma_wskab = make_distributed_array<local_Array_5D_t>(*comm, {1, 1, nkpools, np_a, np_b},
                                                               {nw, ns, nkpts, nbnd, nbnd}, {1, 1, 1, 1, 1});
  auto s_rng = dSigma_wskab.local_range(1);
  auto k_rng = dSigma_wskab.local_range(2);
  auto a_rng = dSigma_wskab.local_range(3);
  auto b_rng = dSigma_wskab.local_range(4);
  auto [nw_loc, ns_loc, nk_loc, na_loc, nb_loc] = dSigma_wskab.local_shape();

  // ------ basis transform from primary to MO basis ------
  {
    auto dSigma_tskab = make_distributed_array<local_Array_5D_t>(*comm, {1, 1, nkpools, np_a, np_b},
                                                               {nt, ns, nkpts, nbnd, nbnd},
                                                               {1, 1, 1, 1, 1});
    auto Sigma_tskab_loc = dSigma_tskab.local();

    nda::array<ComplexType, 2> C_jb(nbnd, nb_loc);
    nda::array<ComplexType, 2> SigmaC_ib(nbnd, nb_loc);
    nda::array<ComplexType, 2> Cdag_ai(na_loc, nbnd);
    auto Sigma_ab = C_jb(nda::range(na_loc), nda::range::all);
    for (size_t it = 0; it < nt; ++it) {
      for (auto [is_loc, is]: itertools::enumerate(s_rng)) {
        for (auto [ik_loc, ik]: itertools::enumerate(k_rng)) {
          C_jb = sMO_skia.local()(is, ik, nda::range::all, b_rng);
          nda::blas::gemm(sSigma_tskij.local()(it, is, ik, nda::ellipsis{}), C_jb, SigmaC_ib);

          auto C_ia = sMO_skia.local()(is, ik, nda::range::all, a_rng);
          Cdag_ai = nda::transpose(nda::conj(C_ia));
          nda::blas::gemm(Cdag_ai, SigmaC_ib, Sigma_ab);
          Sigma_tskab_loc(it, is_loc, ik_loc, nda::range::all, nda::range::all) = Sigma_ab;
        }
      }
    }
    FT.tau_to_w(dSigma_tskab.local(), dSigma_wskab.local(), imag_axes_ft::fermion);
  }

  // Static approximation for V_QPGW
  long dim1 = ns_loc * nk_loc * na_loc * nb_loc;
  // local I index to global [s, k, a, b]
  // bypass clang openmp error: error: capturing a structured binding is not yet supported in OpenMP
  auto nk_loc_ = nk_loc;
  auto na_loc_ = na_loc;
  auto nb_loc_ = nb_loc;
  auto I_to_skab = [&](size_t I) {
    // I = s_loc*nk_loc*na_loc*nb_loc + k_loc*na_loc*nb_loc + a_loc*nb_loc + b_loc
    size_t s_loc = I / (nk_loc_*na_loc_*nb_loc_);
    size_t k_loc = ( I / (na_loc_*nb_loc_) ) % nk_loc_;
    size_t a_loc = ( I / nb_loc_ ) % na_loc_;
    size_t b_loc = I % nb_loc_;
    return std::make_tuple(s_rng.first()+s_loc, k_rng.first()+k_loc, a_rng.first()+a_loc, b_rng.first()+b_loc);
  };

  analyt_cont::AC_t AC(qp_params.ac_alg);
  auto n_to_iw = nda::map([&](int n) { return FT.omega(n); });
  nda::array<ComplexType, 1> iw_mesh(n_to_iw(FT.wn_mesh()));

  app_log(2, "\n* Applying the static approximation (Phys. Rev. Lett. 93, 126406) to Sigma(w): ");
  app_log(2, "  - processor grid for V_QPGW : (s, k, a, b) = ({}, {}, {})", 1, nkpools, np_a, np_b);
  app_log(2, "  - ac algorithm:               {}", qp_params.ac_alg);
  app_log(2, "  - eta:                        {}", qp_params.eta);
  app_log(2, "  - off-diagonal mode:          {}\n", qp_params.off_diag_mode);
  auto Sigma_loc_2D = nda::reshape(dSigma_wskab.local(), std::array<long, 2>{nw, dim1});
  AC.init(iw_mesh, Sigma_loc_2D, qp_params.Nfit);

  auto sVcorr_skij = make_shared_array<Array_view_4D_t>(*comm, *internode_comm, *node_comm, {ns, nkpts, nbnd, nbnd});
  sVcorr_skij.win().fence();
  for (size_t I = 0; I < dim1; ++I) {
    auto [s, k, a, b] = I_to_skab(I);
    if (qp_params.off_diag_mode == "qp_energy") {
      double eps_a = sE_ska.local()(s, k, a).real() - mu;
      double eps_b = sE_ska.local()(s, k, b).real() - mu;
      sVcorr_skij.local()(s, k, a, b) = 0.5 * ( AC.evaluate(ComplexType(eps_a, qp_params.eta), I)
                                                 + AC.evaluate(ComplexType(eps_b, qp_params.eta), I) );
    } else if (qp_params.off_diag_mode == "fermi") {
      double eps_a = (a == b)? sE_ska.local()(s, k, a).real() - mu : 0.0;
      sVcorr_skij.local()(s, k, a, b) = AC.evaluate(ComplexType(eps_a, qp_params.eta), I);
    } else {
      utils::check(false, "qp_approx: unknown off-diagonal mode: {}", qp_params.off_diag_mode);
    }
  }
  sVcorr_skij.win().fence();
  sVcorr_skij.all_reduce();

  // prepare for inverse transformation from MO to primary basis
  auto sMOinv_skai = make_shared_array<Array_view_4D_t>(*comm, *internode_comm, *node_comm, {ns, nkpts, nbnd, nbnd});
  sMOinv_skai.win().fence();
  for (size_t sk = node_comm->rank(); sk < ns*nkpts; sk += node_comm->size()) {
    size_t is = sk / nkpts;
    size_t ik = sk % nkpts;
    auto MO = make_matrix_view(sMO_skia.local()(is, ik, nda::ellipsis{}));
    sMOinv_skai.local()(is, ik, nda::ellipsis{}) = nda::inverse(MO);
  }
  sMOinv_skai.win().fence();

  // Hermitize V_QPGW and then do basis transformation from MO to primary basis
  sVcorr_skij.win().fence();
  if (node_comm->rank() < ns*nkpts) {
    nda::array<ComplexType, 2> V_QPGW_ab(nbnd, nbnd);
    nda::array<ComplexType, 2> VC_aj(nbnd, nbnd);
    nda::array<ComplexType, 2> Cdag_ia(nbnd, nbnd);
    for (size_t sk = node_comm->rank(); sk < ns*nkpts; sk += node_comm->size()) {
      size_t is = sk / nkpts;
      size_t ik = sk % nkpts;

      // Extract the Hermitian part of V_QPGW since V_QPGW in principle in non-Hermitian
      V_QPGW_ab = 0.5 * ( sVcorr_skij.local()(is, ik, nda::ellipsis{})
                          + nda::transpose(nda::conj(sVcorr_skij.local()(is, ik, nda::ellipsis{}))) );

      nda::blas::gemm(V_QPGW_ab, sMOinv_skai.local()(is,ik,nda::ellipsis{}), VC_aj);
      Cdag_ia = nda::transpose(nda::conj(sMOinv_skai.local()(is,ik,nda::ellipsis{})));
      nda::blas::gemm(Cdag_ia, VC_aj, sVcorr_skij.local()(is, ik, nda::ellipsis{}));
    }
  }
  sVcorr_skij.win().fence();
  return sVcorr_skij;
}

template<typename eri_t, typename corr_solver_t>
void add_qpscf_vcorr(MBState &mb_state,
                     double mu,
                     solvers::mb_solver_t<corr_solver_t> &mb_solver,
                     eri_t &eri,
                     const imag_axes_ft::IAFT &FT,
                     qp_params_t &qp_params) {
  using math::shm::make_shared_array;

  auto& sHeff_skij = mb_state.sHeff_skij.value();
  auto& sMO_skia = mb_state.sMO_skia.value();
  auto& sE_ska = mb_state.sE_ska.value();
  auto mpi = eri.mpi();
  auto [ns, nkpts, nbnd, nbnd2] = sHeff_skij.shape();
  auto nt = FT.nt_f();

  mb_state.sSigma_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  mb_state.sG_tskij.emplace(make_shared_array<Array_view_5D_t>(*mpi, {nt, ns, nkpts, nbnd, nbnd}));
  update_G(mb_state.sG_tskij.value(), sMO_skia, sE_ska, mu, FT);
  FT.check_leakage(mb_state.sG_tskij.value(), imag_axes_ft::fermion, "Green's function");
  
  // compute screen interaction
  utils::check(mb_solver.scr_eri!=nullptr, "add_qpscf_vcorr: mb_solver.scr_eri == nullptr.");
  mb_solver.scr_eri->update_w(mb_state, eri, mb_solver.corr->iter());
  
  // compute dynamic self-energy in the primary basis
  mb_solver.corr->evaluate(mb_state, eri);
  FT.check_leakage(mb_state.sSigma_tskij.value(), imag_axes_ft::fermion, "Self-energy");

  // Add the correlation contribution to the QP Hamiltonian.
  auto sVcorr_skij = qp_approx(mb_state.sSigma_tskij.value(),  sMO_skia, sE_ska, mu, FT, qp_params);
  if (mpi->node_comm.root()) sHeff_skij.local() += sVcorr_skij.local();
  mpi->comm.barrier();
 
  // deallocation
  mb_state.dW_qtPQ.reset();
  mb_state.sG_tskij.reset();
  mb_state.sSigma_tskij.reset();
  mpi->comm.barrier();
}



template<typename function_t>
auto qp_eqn_spectral(double Vhf, function_t &Sigma, long I, double mu, double eps0, double tol, double eta)
  -> std::tuple<double, bool> {
  double w_min = eps0 - mu - 1.0;
  double w_max = eps0 - mu + 1.0;
  //long Nw = (w_max - w_min) / tol1;
  long Nw = 1000;

  nda::array<ComplexType, 1> w_grid(Nw);
  int i = 0;
  std::transform(w_grid.begin(), w_grid.end(), w_grid.begin(),
                 [&](const ComplexType & ) {return w_min + (i++)*(w_max - w_min)/(Nw - 1) + eta*1i;} );

  nda::array<ComplexType, 1> SigmaR_w(Nw);
  Sigma.evaluate(w_grid, SigmaR_w, I);

  // estimates peak position
  double Aw_max = 0.0;
  long nmax = -1;
  for (size_t n = 0; n < Nw; ++n) {
    ComplexType Gw = 1.0 / (w_grid(n) + mu - Vhf - SigmaR_w(n));
    if (std::abs(Gw.imag()) > Aw_max) {
      nmax = n;
      Aw_max = std::abs(Gw.imag());
    }
  }

  // fine tune the peak within a given tol
  auto Aw = [&](ComplexType w) {
    return std::abs((1.0 / ( w + mu - Vhf - Sigma.evaluate(w, I) )).imag());
  };
  ComplexType eps_qp = w_grid(nmax);
  ComplexType eps_qp_dw = eps_qp + tol;
  double Aw_dw = Aw(eps_qp_dw);
  if ( Aw_dw >= Aw_max) {
    while (Aw_dw >= Aw_max) {
      eps_qp = eps_qp_dw;
      Aw_max = Aw_dw;
      eps_qp_dw += tol;
      Aw_dw = Aw(eps_qp_dw);
    }
  } else {
    eps_qp_dw = eps_qp - tol;
    Aw_dw = Aw(eps_qp_dw);
    while (Aw_dw >= Aw_max) {
      eps_qp = eps_qp_dw;
      Aw_max = Aw_dw;
      eps_qp_dw -= tol;
      Aw_dw = Aw(eps_qp_dw);
    }
  }

  if (nmax != -1) {
    return std::make_tuple(eps_qp.real()+mu, true);
  } else {
    return std::make_tuple(eps0, false);
  }
}

template<typename function_t>
double qp_eqn_linearized(double Vhf, function_t &Sigma, long I, double mu, double eps_ks, double eta) {
  double h = 1e-6;
  double Sigma_ks = Sigma.evaluate(ComplexType(eps_ks-mu, eta), I).real();
  double dSigma = (Sigma.evaluate(ComplexType(eps_ks-mu+h, eta), I).real() - Sigma_ks);
  double Z = 1.0 / (1.0 - dSigma/h);

  return Z * ( Vhf + Sigma_ks - (1 - 1.0/Z) * eps_ks );
}

template<typename function_t>
auto qp_eqn_secant(double Vhf, function_t &Sigma, long I, double mu, double w0, int maxiter, double tol, double eta)
  -> std::tuple<double, double, bool> {
  auto qp_res = [&](ComplexType w) {
    return (w - Vhf - Sigma.evaluate(w-mu, I)).real();
  };

  bool conv = false;
  double p=0, p0, p1, q, q0, q1;
  double eps = 1e-4;

  p0 = w0;
  p1 = (p0 >= 0)? w0 * (1 + eps) + eps : w0 * (1 + eps) - eps;
  q0 = qp_res(ComplexType(p0, eta));
  q1 = qp_res(ComplexType(p1, eta));

  if (std::abs(q1) < std::abs(q0)) {
    double tmp = p1;
    p1 = p0;
    p0 = tmp;

    tmp = q1;
    q1 = q0;
    q0 = tmp;
  }

  for (long it = 0; it < maxiter; ++it) {
    if (std::abs(q1) > std::abs(q0)) {
      p = (-q0/q1 * p1 + p0) / (1 - q0/q1);
    } else {
      p = (-q1/q0 * p0 + p1) / (1 - q1/q0);
    }

    if (std::abs(p - p1) < tol) {
      // TODO we should check res as well?
      conv = true;
      break;
    }

    p0 = p1;
    q0 = q1;
    p1 = p;
    q1 = qp_res(ComplexType(p1, eta));
  }

  q = qp_res(ComplexType(p, eta));
  return std::make_tuple(p, q, conv);
}

template<typename function_t>
auto qp_eqn_bisection(double Vhf, function_t &Sigma, long I, double mu, double eps0, double tol, double eta)
  -> std::tuple<double, double> {
  auto qp_res = [&](ComplexType w) {
    return (w - Vhf - Sigma.evaluate(w-mu, I)).real();
  };

  double eps1, eps2, eps_mid;
  double eps = eps0;
  double res = qp_res(ComplexType(eps0, eta));
  double delta = 0.01;

  app_log(6, "I = {0}, Vhf = {1:.12f}, Sigma = {2:.12f}, eps = {3:.12f}, res = {4:.12f}",
          I, Vhf, Sigma.evaluate(ComplexType(eps0,eta), I).real(), eps, res);
  if (std::abs(res) < tol) return std::make_tuple(eps, res);

  if (res >= 0) {
    eps2 = eps0;
    eps1 = eps0 - delta;
    double res1 = qp_res(ComplexType(eps1, eta));
    while (res1 > 0) {
      app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps1, res1);
      eps1 -= delta;
      res1 = qp_res(ComplexType(eps1, eta));
    }
  } else {
    eps1 = eps0;
    eps2 = eps0 + delta;
    double res2 = qp_res(ComplexType(eps2, eta));
    while (res2 < 0) {
      app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps2, res2);
      eps2 += delta;
      res2 = qp_res(ComplexType(eps2, eta));
    }
  }
  eps_mid = (eps1 + eps2) * 0.5;
  res = qp_res(ComplexType(eps_mid, eta));
  while (std::abs(res) > tol) {
    app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps_mid, res);
    if (res >= 0) {
      eps2 = eps_mid;
    } else {
      eps1 = eps_mid;
    }
    eps_mid = (eps1 + eps2) * 0.5;
    res = qp_res(ComplexType(eps_mid, eta));
  }
  app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps_mid, res);
  eps = eps_mid;
  return std::make_tuple(eps, res);
}

double compute_Nelec(double mu, const mf::MF &mf, const sArray_t<Array_view_3D_t> &sE_ski, double beta) {
  auto [ns, nkpts, nbnd] = sE_ski.shape();
  auto FD_occ = nda::map([&](ComplexType e) { return 1.0 / ( 1 + std::exp( (e.real()-mu) * beta ) ); });
  //nda::array<double, 3> nel_ski(ns, nkpts, nbnd);
  //nel_ski = FD_occ(sE_ski.local());
  nda::array<double, 1> nel_i(nbnd);

  auto k_weight = mf.k_weight();
  double nel = 0.0;
  for (size_t s = 0; s < ns; ++s) {
    for (size_t k = 0; k < nkpts; ++k) {
      nel_i = FD_occ(sE_ski.local()(s, k, nda::range::all));
      nel += k_weight(k) * nda::sum(nel_i);
    }
  }
  if (ns == 1 and mf.npol()==1) nel *= 2.0;
  //double nel = (ns == 2)? nda::sum(nel_ski) / nkpts : 2 * nda::sum(nel_ski) / nkpts;
  return nel;
}

template<typename X_t>
double update_mu(double old_mu, const mf::MF &mf, const X_t &sE_ski, double beta, double mu_tol) {
  double nel_target = mf.nelec();
  double nel, mu1, mu2, mu_mid;
  double mu = old_mu;
  double delta = 0.2;
  nel = compute_Nelec(old_mu, mf, sE_ski, beta);
  app_log(2, "Initial chemical potential (mu) = {}, nelec = {}", old_mu, nel);

  if (std::abs(nel - nel_target) < mu_tol) {
    app_log(2, "Chemical potential found (mu) = {} a.u.", mu);
    app_log(2, "Number of electrons per unit cell = {}", nel);
    return mu;
  }

  if (nel >= nel_target) {
    mu2 = old_mu;
    mu1 = old_mu - delta;
    double nel1 = compute_Nelec(mu1, mf, sE_ski, beta);
    while (nel1 > nel_target) {
      mu1 -= delta;
      nel1 = compute_Nelec(mu1, mf, sE_ski, beta);
    }
    app_log(4, "mu = {}, nelec = {}", mu1, nel1);
  } else {
    mu1 = old_mu;
    mu2 = old_mu + delta;
    double nel2 = compute_Nelec(mu2, mf, sE_ski, beta);
    while (nel2 < nel_target) {
      mu2 += delta;
      nel2 = compute_Nelec(mu2, mf, sE_ski, beta);
    }
    app_log(4, "mu = {}, nelec = {}", mu2, nel2);
  }
  mu_mid = (mu1 + mu2) * 0.5;
  nel = compute_Nelec(mu_mid, mf, sE_ski, beta);
  app_log(4, "mu = {}, nelec = {}", mu_mid, nel);

  while (std::abs(nel - nel_target) >= mu_tol) {
    if (nel >= nel_target) {
      mu2 = mu_mid;
    } else {
      mu1 = mu_mid;
    }
    mu_mid = (mu1 + mu2) * 0.5;
    nel = compute_Nelec(mu_mid, mf, sE_ski, beta);
    app_log(4, "mu = {}, nelec = {}", mu_mid, nel);
  }
  mu = mu_mid;
  app_log(2, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(2, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename comm_t, typename X_t>
double solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                       long it, std::string h5_prefix, X_t &sHeff_skij) {
  double conv = 0;
  if (it == 1) {
    // Just check changes w.r.t. mf
    if (context.node_comm.root()) {
      auto H_mf = nda::make_regular(sHeff_skij.local());
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      if (grp.has_subgroup("scf/iter0")) {
        auto iter_grp = grp.open_group("scf/iter0");
        if (iter_grp.has_dataset("Heff_skij")) {
          // checkpoint from a qp scf
          nda::h5_read(iter_grp, "Heff_skij", H_mf);
        } else if (iter_grp.has_dataset("F_skij")) {
          // checkpoint from a dyson scf
          nda::h5_read(iter_grp, "F_skij", H_mf);
          nda::array<ComplexType, 4> H0(H_mf.shape());
          auto sys_grp = grp.open_group("system");
          nda::h5_read(sys_grp, "H0_skij", H0);
          H_mf += H0;
        }
      }
      H_mf -= sHeff_skij.local();
      auto max_iter = max_element(H_mf.data(), H_mf.data()+H_mf.size(),
                                   [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      conv =  std::abs((*max_iter));
    }
    context.node_comm.broadcast_n(&conv, 1, 0);
  } else {
    iter_solver.metadata_log();
    if (context.node_comm.root()) {
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      if (grp.has_subgroup("scf/iter" + std::to_string(it-1))) {
        auto scf_grp = grp.open_group("scf");
        conv = iter_solver.solve(sHeff_skij.local(), "Heff_skij", scf_grp, it);
      }
    }
    context.node_comm.broadcast_n(&conv, 1, 0);
  }
  context.comm.barrier();
  return conv;
}

void write_mf_data(mf::MF &mf, const imag_axes_ft::IAFT &ft,
                   hamilt::pseudopot &psp, std::string output) {
  auto mpi = mf.mpi();
  sArray_t<Array_view_4D_t> sHeff_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sS_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sMO_skia(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_3D_t> sE_ska(math::shm::make_shared_array<Array_view_3D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd()}));
  double mu = 0.0;

  hamilt::set_fock(mf, std::addressof(psp), sHeff_skij, false);
  hamilt::set_H0(mf, std::addressof(psp), sH0_skij);
  hamilt::set_ovlp(mf, sS_skij);
  update_MOs(sMO_skia, sE_ska, sHeff_skij, sS_skij);
  mu = update_mu(mu, mf, sE_ska, ft.beta());
  update_Dm(sDm_skij, sMO_skia, sE_ska, mu, ft.beta());

  chkpt::write_metadata(mpi->comm, mf, ft, sH0_skij, sS_skij, output);
  chkpt::dump_scf(mpi->comm, 0, sDm_skij, sHeff_skij, sMO_skia, sE_ska, mu, output);
}

/** Instantiation of public template **/

template double update_mu(double, const mf::MF&, const sArray_t<Array_view_3D_t>&, double, double);

template void add_evscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, thc_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, bool);
template void add_evscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, chol_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&, bool);

template void add_qpscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, thc_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&);
template void add_qpscf_vcorr(MBState&, double, solvers::mb_solver_t<>&, chol_reader_t&, const imag_axes_ft::IAFT&, qp_params_t&);

template double qp_eqn_linearized(double, analyt_cont::AC_t &, long, double, double, double);
template std::tuple<double,double> qp_eqn_bisection(double, analyt_cont::AC_t &, long, double, double, double, double);
template std::tuple<double,double,bool> qp_eqn_secant(double, analyt_cont::AC_t &, long, double, double, int, double, double);
template std::tuple<double,bool> qp_eqn_spectral(double, analyt_cont::AC_t &, long, double, double, double, double);

template double solve_iterative(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&, long, std::string,
                                sArray_t<Array_view_4D_t>&);
template void compute_G_from_mf(h5::group, imag_axes_ft::IAFT&, sArray_t<nda::array_view<ComplexType, 5>>&);

} // methods
