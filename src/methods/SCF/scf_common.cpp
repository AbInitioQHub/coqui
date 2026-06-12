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


#include "scf_common.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "mean_field/MF.hpp"
#include "utilities/mpi_context.h"
#include "methods/tools/chkpt_utils.h"
#include "simple_dyson.h"

namespace methods {

double compute_Nelec(double mu, const nda::array<ComplexType, 4> &spectra,
                     const mf::MF &mf, const imag_axes_ft::IAFT &FT) {
  auto [_nw, _ns, nkpts, _nbnd] = spectra.shape();
  nda::array<ComplexType, 2> Xw(_nw, _ns);
  nda::array<ComplexType, 2> Xt(FT.nt_f(), _ns);
  nda::array<ComplexType, 1> nelecs(_ns);
  auto k_weight = mf.k_weight();
  double scl = (_ns == 1 and mf.npol() == 1 ? -2.0 : -1.0); 

  for (size_t n = 0; n < _nw; ++n) {
    long wn = FT.wn_mesh()(n);
    ComplexType omega_mu = FT.omega(wn) + mu;
    for (size_t is = 0; is < _ns; ++is) {
      for (size_t ik = 0; ik < nkpts; ++ik) {
        for (size_t ib = 0; ib < _nbnd; ++ib) {
          Xw(n, is) += k_weight(ik) / (omega_mu - spectra(n, is, ik, ib));
        }
      }
    }
  }

  FT.w_to_tau(Xw, Xt, imag_axes_ft::fermion);
  FT.tau_to_beta(Xt, nelecs);

  ComplexType nelec = scl*std::accumulate(nelecs.begin(),nelecs.end(),ComplexType(0.0));
  if (nelec.imag() / mf.nelec() >= 1e3*FT.eps()) {
    app_log(1, "[WARNING] nelec.imag()/nelec_target = {}", nelec.imag() / mf.nelec());
  }

  return nelec.real();
}

template<typename X_t, nda::ArrayOfRank<1> Array1D>
auto eval_hf_energy(const X_t &sDm_skij, const X_t &sF_skij, const X_t &sH0_skij,
                    Array1D &k_weight, bool F_has_H0)
  -> std::tuple<double, double> {
  auto [ns, nkpts, nbnd, nbnd2] = sDm_skij.shape();
  // HF energy = Tr[Dm*H0] + 0.5*Tr[Dm*F] + e_nuc
  nda::matrix<ComplexType> buffer(nbnd, nbnd);
  ComplexType e_1e(0.0, 0.0);
  ComplexType e_hf(0.0, 0.0);
  for (size_t is = 0; is < ns; ++is) {
    for (size_t ik = 0; ik < nkpts; ++ik) {
      nda::matrix_const_view<ComplexType> Dm_ij =
          sDm_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> F_ij =
          sF_skij.local()(is, ik, nda::ellipsis{});
      nda::matrix_const_view<ComplexType> H0_ij =
          sH0_skij.local()(is, ik, nda::ellipsis{});

      buffer = Dm_ij * H0_ij;
      auto diag_H0 = nda::diagonal(buffer);
      e_1e += k_weight(ik) * nda::sum(diag_H0);

      buffer = (F_has_H0)? 0.5 * Dm_ij * (F_ij - H0_ij) : 0.5 * Dm_ij * F_ij;
      auto diag_F = nda::diagonal(buffer);
      e_hf += k_weight(ik) * nda::sum(diag_F);
    }
  }
  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  e_1e *= spin_factor;
  e_hf *= spin_factor;
  // TODO CNY: _MF->e_nuc() is missing
  if (e_1e.imag() / e_1e.real() >= 1e-8) {
    app_log(1, "[WARNING] e_1e.imag()/e_1e.real() = {}, e_1e.imag() = {}, e_1e.real() = {}",
            e_1e.imag()/e_1e.real(), e_1e.imag(), e_1e.real());
  }
  if (e_hf.imag() / e_hf.real() >= 1e-8) {
    app_log(1, "[WARNING] e_hf.imag()/e_hf.real() = {}, e_hf.imag() = {}, e_hf.real() = {}",
            e_hf.imag()/e_hf.real(), e_hf.imag(), e_hf.real());
  }
  return std::make_tuple(e_1e.real(), e_hf.real());
}

template<typename comm_t, typename X_t, nda::ArrayOfRank<1> Array1D>
double eval_corr_energy(comm_t& comm, const imag_axes_ft::IAFT &FT,
                        const X_t & G_shm, const X_t & Sigma_shm,
                        Array1D &k_weight) {
  decltype(nda::range::all) all;
  int nw = FT.nw_f();
  auto [nts, ns, nkpts, nbnd, nbnd2] = G_shm.shape();
  nda::array<ComplexType, 2> SigmaG_ws(nw, ns);
  nda::array<ComplexType, 4> Sigma_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_tski(nts, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> Sigma_wski(nw, ns, nkpts, nbnd);
  nda::array<ComplexType, 4> G_wski(nw, ns, nkpts, nbnd);
  auto SigmaG_ws_1D =
      nda::reshape(SigmaG_ws, std::array<long, 1>{nw * ns});
  auto Sigma_w_3D = nda::reshape(
      Sigma_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});
  auto G_w_3D = nda::reshape(
      G_wski, std::array<long, 3>{nw * ns, nkpts, nbnd});

  int size = comm.size();
  int rank = comm.rank();
  comm.barrier();
  for (size_t i = rank; i < nbnd; i += size) {
    Sigma_tski = Sigma_shm.local()(all, all, all, i, all);
    G_tski = G_shm.local()(all, all, all, all, i);
    FT.tau_to_w(Sigma_tski, Sigma_wski, imag_axes_ft::fermion);
    FT.tau_to_w(G_tski, G_wski, imag_axes_ft::fermion);
    for (size_t ws = 0; ws < nw * ns; ++ws) {
      for (size_t ik = 0; ik < nkpts; ++ik ) {
        SigmaG_ws_1D(ws) += k_weight(ik) * nda::blas::dot(Sigma_w_3D(ws, ik, all), G_w_3D(ws, ik, all));
      }
    }
  }
  comm.all_reduce_in_place_n(SigmaG_ws.data(), SigmaG_ws.size(),
                             std::plus<>{});

  nda::array<ComplexType, 2> SigmaG_ts(nts, ns);
  nda::array<ComplexType, 1> SigmaG_beta_s(ns);
  FT.w_to_tau(SigmaG_ws, SigmaG_ts, imag_axes_ft::fermion);
  FT.tau_to_beta(SigmaG_ts, SigmaG_beta_s);

  // MAM: need to know npol here, scale only when npol==1 and ns==1
  RealType spin_factor = (ns == 2) ? 1.0 : 2.0;
  ComplexType e_corr = (-0.5 * spin_factor) * nda::sum(SigmaG_beta_s);
  if (e_corr.imag() / e_corr.real() >= 1e2*FT.eps()) {
    app_log(1, "[WARNING] e_corr.imag()/e_corr.real() = {}, e_corr.imag() = {}, e_corr.real() = {}",
            e_corr.imag()/e_corr.real(), e_corr.imag(), e_corr.real());
  }
  return e_corr.real();
}

// JHL: This function only works for HF and GW
template<typename dyson_type, typename X_t, typename Xt_t>
void eval_thermodynamic_properties(dyson_type &dyson, const X_t &sF_skij,
                                   const Xt_t &sG_tskij, const Xt_t &sSigma_tskij,
                                   const std::vector<double> &energies, double e_rpa,
                                   double mu, bool F_has_H0) {
  decltype(nda::range::all) all;

  auto MF = dyson.MF();
  auto FT = dyson.FT();

  auto ns = MF->nspin();
  auto nkpts = MF->nkpts();
  auto nbnd = MF->nbnd();
  auto npol = MF->npol();
  auto k_weight = MF->k_weight();

  auto beta = FT->beta();
  auto nt = FT->nt_f();
  auto nw = FT->nw_f();

  RealType spin_factor = 0;
  ComplexType Phi_1(0.0, 0.0);
  ComplexType Phi_2(0.0, 0.0);
  ComplexType Phi(0.0, 0.0);
  ComplexType tr_Sigma_G_1(0.0, 0.0);
  ComplexType tr_Sigma_G_2(0.0, 0.0);
  ComplexType tr_Sigma_G(0.0, 0.0);
  ComplexType tr_ln_G0(0.0, 0.0);
  ComplexType grand_potential(0.0, 0.0);
  auto tr_ln_1_minus_G0_Sigma_t = nda::array<ComplexType, 2>::zeros({nt, 1});
  auto tr_ln_1_minus_G0_Sigma_w = nda::array<ComplexType, 2>::zeros({nw, 1});
  auto tr_ln_1_minus_G0_Sigma_beta = nda::array<ComplexType, 1>::zeros({1});

  Phi_1 = energies[1];
  Phi_2 = e_rpa;
  Phi = Phi_1 + Phi_2;

  tr_Sigma_G_1 = energies[1] * 2;
  tr_Sigma_G_2 = energies[2] * 2;
  tr_Sigma_G = tr_Sigma_G_1 + tr_Sigma_G_2;

  for (size_t is = 0; is < ns; ++is) {
    for (size_t ik = 0; ik < nkpts; ++ik) {
      nda::matrix_const_view<ComplexType> F_ij = sF_skij.local()(is, ik, all, all);
      nda::matrix_const_view<ComplexType> H0_ij = dyson.sH0_skij().local()(is, ik, all, all);
      nda::matrix_const_view<ComplexType> S_ij = dyson.sS_skij().local()(is, ik, all, all);
      nda::matrix<ComplexType> F = F_has_H0 ? nda::make_regular(F_ij) : nda::make_regular(F_ij + H0_ij);
      auto eigenvalues = nda::linalg::eigenvalues(F, nda::make_regular(S_ij));
      ComplexType buffer(0.0, 0.0);
      for (size_t ibnd = 0; ibnd < nbnd; ++ibnd) {
        if (eigenvalues(ibnd) - mu > 0) {
          buffer += std::log(1.0 + std::exp(-beta * (nda::real(eigenvalues(ibnd)) - mu)));
        } else {
          buffer += std::log(1.0 + std::exp(beta * (nda::real(eigenvalues(ibnd)) - mu)));
          buffer -= (eigenvalues(ibnd) - mu) * beta;
        }
      }
      tr_ln_G0 += buffer * k_weight(ik);
    }
  }
  spin_factor = (npol == 1 and ns == 1) ? 2.0 : 1.0;
  tr_ln_G0 *= spin_factor / beta;

  auto I = nda::eye<ComplexType>(nbnd);
  nda::matrix<ComplexType> G0_Sigma_w(nbnd, nbnd);
  nda::matrix<ComplexType> one_minus_G0_Sigma_w(nbnd, nbnd);
  for (size_t n = 0; n < nw; ++n) {
    nda::array<ComplexType, 4> Sigmaw_skij({ns, nkpts, nbnd, nbnd});
    auto wn = FT->wn_mesh()(n);
    ComplexType omega_mu = FT->omega(wn) + mu;
    FT->tau_to_w(sSigma_tskij.local(), Sigmaw_skij, imag_axes_ft::fermion, n);
    for (size_t is = 0; is < ns; ++is) {
      for (size_t ik = 0; ik < nkpts; ++ik) {
        nda::matrix_const_view<ComplexType> F_ij = sF_skij.local()(is, ik, all, all);
        nda::matrix_const_view<ComplexType> H0_ij = dyson.sH0_skij().local()(is, ik, all, all);
        nda::matrix_const_view<ComplexType> S_ij = dyson.sS_skij().local()(is, ik, all, all);
        nda::matrix<ComplexType> F = F_has_H0 ? nda::make_regular(F_ij) : nda::make_regular(F_ij + H0_ij);
        nda::matrix<ComplexType> G0_inv = omega_mu * S_ij - F;
        // calculate G_0 \Sigma by solving G_0^{-1} X = \Sigma
        G0_Sigma_w = Sigmaw_skij(is, ik, all, all);
        // nda tensor branch requies F_layout
        nda::matrix<ComplexType, nda::F_layout> G0_Sigma_w_F_layout = G0_Sigma_w;
        nda::array<int, 1> ipiv(G0_inv.extent(0));
        nda::lapack::getrf(G0_inv, ipiv);
        nda::lapack::getrs(G0_inv, G0_Sigma_w_F_layout, ipiv);
        G0_Sigma_w = G0_Sigma_w_F_layout;
        one_minus_G0_Sigma_w = I - G0_Sigma_w;
        // JHL: Is (1-G_0\Sigma) hermitian?
        nda::matrix<ComplexType> buffer(nbnd, nbnd);
        nda::blas::gemm(1.0, nda::conj(nda::transpose(one_minus_G0_Sigma_w)), one_minus_G0_Sigma_w, 0.0, buffer);
        nda::lapack::getrf(buffer, ipiv);
        for (size_t ibnd = 0; ibnd < nbnd; ++ibnd) {
          tr_ln_1_minus_G0_Sigma_w(n, 0) += nda::log(std::abs(buffer(ibnd, ibnd).real())) * 0.5 * k_weight(ik);
        }
      }
    }
  }
  FT->w_to_tau(tr_ln_1_minus_G0_Sigma_w, tr_ln_1_minus_G0_Sigma_t, imag_axes_ft::fermion);
  FT->tau_to_beta(tr_ln_1_minus_G0_Sigma_t, tr_ln_1_minus_G0_Sigma_beta);
  spin_factor = (npol == 1 and ns == 1) ? 2.0 : 1.0;
  tr_ln_1_minus_G0_Sigma_beta(0) *= -1 * spin_factor;

  grand_potential = Phi - tr_Sigma_G - tr_ln_G0 - tr_ln_1_minus_G0_Sigma_beta(0);

  app_log(1, "\n");
  app_log(1, "Grand potential contributions");
  app_log(1, "--------------------");
  app_log(1, "  Luttinger-Ward:                  {:>20.12f} a.u.", Phi.real());
  app_log(1, "  tr G*Sigma:                      {:>20.12f} a.u.", -tr_Sigma_G.real());
  app_log(1, "  tr ln(-G0):                      {:>20.12f} a.u.", -tr_ln_G0.real());
  app_log(1, "  tr ln(1-G0*Sigma):               {:>20.12f} a.u.", -tr_ln_1_minus_G0_Sigma_beta(0).real());
  app_log(1, "  total grand potential:           {:>20.12f} a.u.", grand_potential.real());
  app_log(1, "\n");

  if (std::abs(tr_ln_1_minus_G0_Sigma_beta(0).imag() / tr_ln_1_minus_G0_Sigma_beta(0).real()) >= 1e-8) {
    app_log(1, "[WARNING] Abs (Tr ln(1-G0*Sigma).imag() / Tr ln(1-G0*Sigma).real()) = {},\n",
            tr_ln_1_minus_G0_Sigma_beta(0).imag() / tr_ln_1_minus_G0_Sigma_beta(0).real());
    app_log(1, "          (Tr ln(1-G0*Sigma)).imag() = {},\n", tr_ln_1_minus_G0_Sigma_beta(0).imag());
    app_log(1, "          (Tr ln(1-G0*Sigma)).real() = {} \n", tr_ln_1_minus_G0_Sigma_beta(0).real());
  }

  nda::array<ComplexType, 4> spectra(FT->nw_f(), MF->nspin(), MF->nkpts_ibz(), MF->nbnd());
  dyson.compute_eigenspectra(sF_skij, sSigma_tskij, spectra);
  double n_electron = compute_Nelec(mu, spectra, *MF, *FT);
  double helmholtz_free_energy = grand_potential.real() + mu * n_electron;
  double entropy = (energies[3] - helmholtz_free_energy) * beta;

  app_log(1, "\n");
  app_log(1, "Electron thermodynamic properties");
  app_log(1, "--------------------");
  app_log(1, "  energy:                          {:>20.12f} a.u.", energies[3]);
  app_log(1, "  grand potential:                 {:>20.12f} a.u.", grand_potential.real());
  app_log(1, "  Helmholtz free energy:           {:>20.12f} a.u.", helmholtz_free_energy);
  app_log(1, "  beta:                            {:>20.12f} a.u.", beta);
  app_log(1, "  entropy:                         {:>20.12f} a.u.", entropy);
  app_log(1, "  chemical potential:              {:>20.12f} a.u.", mu);
  app_log(1, "  number of electrons:             {:>20.12f}", n_electron);
  app_log(1, "\n");

}

template<typename dyson_type, typename X_t, typename Xt_t>
void update_G(dyson_type &dyson, const mf::MF &mf, const imag_axes_ft::IAFT &FT, X_t & Dm, Xt_t &G,
              const X_t & F, const Xt_t &Sigma, double &mu, bool const_mu) {
  app_log(2, "* Solving Green's function:");
  if(!const_mu)
    mu = update_mu(mu, dyson, mf, FT, F, Sigma);
  dyson.solve_dyson(Dm, G, F, Sigma, mu);
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu_bisection(double old_mu, dyson_type& dyson, const mf::MF &mf,
                           const imag_axes_ft::IAFT &FT,
                           const X_t&F, const Xt_t&Sigma) {
  double nel_target = mf.nelec();
  double delta = 0.2;
  nda::array<ComplexType, 4> FpSigma_spectra(FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
  dyson.compute_eigenspectra(F, Sigma, FpSigma_spectra);
  auto eval_f = [&](double mu) {
    return compute_Nelec(mu, FpSigma_spectra, mf, FT) - nel_target;
  };

  double nel_old = compute_Nelec(old_mu, FpSigma_spectra, mf, FT);
  app_log(2, "Initial chemical potential (mu) = {}, nelec = {}", old_mu, nel_old);

  auto [mu, f_mu] = detail::update_mu_bisection_impl(old_mu, dyson.mu_tol(), delta, eval_f);
  double nel = f_mu + nel_target;
  app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(1, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu_midpoint(double old_mu, dyson_type& dyson, const mf::MF &mf,
                          const imag_axes_ft::IAFT &FT, const X_t&F,
                          const Xt_t&Sigma) {
  double nel_target = mf.nelec();
  double tol = dyson.mu_tol();
  double delta = 0.2;

  nda::array<ComplexType, 4> FpSigma_spectra(
      FT.nw_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd());
  dyson.compute_eigenspectra(F, Sigma, FpSigma_spectra);

  auto eval_f = [&](double mu) {
    return compute_Nelec(mu, FpSigma_spectra, mf, FT) - nel_target;
  };

  double f_old = eval_f(old_mu);
  app_log(2, "Initial chemical potential (mu) = {}, nelec - target = {}",
          old_mu, f_old);

  auto [mu, f_mu, mu_left, mu_right] =
      detail::update_mu_midpoint_impl(old_mu, tol, delta, eval_f);
  double nel = f_mu + nel_target;
  app_log(1, "Chemical potential bounds found (mu_left, mu_right) = ({}, {}) a.u.",
          mu_left, mu_right);
  app_log(1, "Chemical potential found (mu) = {} a.u.", mu);
  app_log(1, "Number of electrons per unit cell = {}", nel);
  return mu;
}

template<typename dyson_type, typename X_t, typename Xt_t>
double update_mu(double old_mu, dyson_type& dyson, const mf::MF &mf,
                 const imag_axes_ft::IAFT &FT,
                 const X_t&F, const Xt_t&Sigma) {
  if (dyson.mu_update_alg() == "bisection") {
    return update_mu_bisection(old_mu, dyson, mf, FT, F, Sigma);
  } else if (dyson.mu_update_alg() == "midpoint") {
    return update_mu_midpoint(old_mu, dyson, mf, FT, F, Sigma);
  } else {
    utils::check(
      false, "scf_common.cpp::update_mu: unknown mu update algorithm {}.", dyson.mu_update_alg());
  }
  return old_mu;
}

template<typename X_t, typename Xt_t>
auto diis_init(iter_scf::iter_scf_t& iter_solver,
               long iteration, std::string output,
               X_t &sF_skij, Xt_t &sSigma_tskij, const imag_axes_ft::IAFT *FT) {
  utils::check(iter_solver.iter_alg() == iter_scf::DIIS, "diis_init: iter_solver is not DIIS type.");
  h5::file file(output+".mbpt.h5", 'r');
  h5::group grp(file);
  utils::check(grp.has_subgroup("scf"), "Simulation HDF5 file does not have an scf group");
  auto scf_grp = grp.open_group("scf");
  auto sys_grp = grp.open_group("system");
  nda::array<ComplexType, 4> H0 = sF_skij.local();
  nda::array<ComplexType, 4> S = sF_skij.local();
  nda::h5_read(sys_grp, "H0_skij", H0);
  nda::h5_read(sys_grp, "S_skij", S);
  double mu = 0;
  if (scf_grp.has_subgroup("iter" + std::to_string(iteration-1))) {
    auto mf_grp = scf_grp.open_group("iter" + std::to_string(iteration-1));
    h5::h5_read(mf_grp, "mu", mu);
  }
  iter_solver.initialize(sF_skij.local(), sSigma_tskij.local(), mu, S, H0, FT, output);
}

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto damping_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
                  long iteration, std::string h5_prefix,
                  X_t &sF_skij, Xt_t &sSigma_tskij,
                  std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (iteration == 1) {
    utils::check(false, "damping_impl: it = 1 is not allowed.");
  } else {
    iter_solver.metadata_log();
    if (context.node_comm.root()) {
      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      std::string grp_name = datasets[0]+"/iter"+std::to_string(iteration-1);
      utils::check(grp.has_subgroup(grp_name),
                   "damping_impl: {} does not exist in {}.", grp_name, filename);
      auto scf_grp = grp.open_group(datasets[0]);
      conv_F = iter_solver.solve(sF_skij.local(), datasets[1], scf_grp, iteration);
      conv_Sigma = iter_solver.solve(sSigma_tskij.local(), datasets[2], scf_grp, iteration);
    }
    context.node_comm.broadcast_n(&conv_F, 1, 0);
    context.node_comm.broadcast_n(&conv_Sigma, 1, 0);
  }
  context.comm.barrier();
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename MPI_Context_t, typename X_t, typename Xt_t>
auto diis_impl(MPI_Context_t &context, iter_scf::iter_scf_t& iter_solver,
               long iteration, std::string h5_prefix, X_t &sF_skij, Xt_t &sSigma_tskij,
               const imag_axes_ft::IAFT *FT, std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (iteration == 1) {
    utils::check(false, "diis_impl: iteration = 1 is not allowed.");
  } else {
    iter_solver.metadata_log();
    int internode_proc_holding_extrap = 0;
    // DIIS does not support mpi yet
    if (context.comm.root()) { // A global communicator here is needed for DIIS

      if (not iter_solver.is_initialized()) {
        diis_init(iter_solver, iteration, h5_prefix, sF_skij, sSigma_tskij, FT);
      }

      std::string filename = h5_prefix + ".mbpt.h5";
      h5::file file(filename, 'r');
      h5::group grp(file);
      std::string grp_name = datasets[0]+"/iter"+std::to_string(iteration-1);
      utils::check(grp.has_subgroup(grp_name),
                   "diis_impl: {} does not exist in {}.", grp_name, filename);
      auto scf_grp = grp.open_group(datasets[0]);
      auto residuals = iter_solver.solve(
        sF_skij.local(), datasets[1], sSigma_tskij.local(), datasets[2], scf_grp, iteration);
      conv_F = residuals[0];
      conv_Sigma = residuals[1];
      internode_proc_holding_extrap = context.internode_comm.rank();
    }
    context.comm.broadcast_n(&conv_F, 1, 0);
    context.comm.broadcast_n(&conv_Sigma, 1, 0);
    // internode_proc_holding_extrap should be 0 everywhere, but if not,
    // the broadcast below ensures that all procs get iteration
    context.comm.broadcast_n(&internode_proc_holding_extrap, 1, 0);
    // Send extrapolated F and Sigma to all nodes
    sF_skij.broadcast_to_nodes(internode_proc_holding_extrap);
    sSigma_tskij.broadcast_to_nodes(internode_proc_holding_extrap);
  }
  context.comm.barrier();
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename comm_t, typename X_t, typename Xt_t>
auto solve_iterative(utils::mpi_context_t<comm_t> &context, iter_scf::iter_scf_t& iter_solver,
                     long iteration, std::string h5_prefix,
                     X_t &sF_skij, Xt_t &sSigma_tskij, const imag_axes_ft::IAFT *FT,
                     std::array<std::string,3> datasets)
  -> std::tuple<double, double> {
  double conv_F = 0;
  double conv_Sigma = 0;
  if (iteration == 1) {
    // Just check changes w.r.t. mf
    if (context.node_comm.root()) {
      auto F_mf = nda::make_regular(sF_skij.local());
      h5::file file(h5_prefix+".mbpt.h5", 'r');
      h5::group grp(file);
      if (grp.has_subgroup("scf/iter0")) {
        auto mf_grp = grp.open_group("scf/iter0");
        if (mf_grp.has_dataset("F_skij")) {
          nda::h5_read(mf_grp, "F_skij", F_mf);
        } else if (mf_grp.has_dataset("Heff_skij")) {
          // checkpoint from a qp scf
          nda::h5_read(mf_grp, "Heff_skij", F_mf);
          nda::array<ComplexType, 4> H0(F_mf.shape());
          auto sys_grp = grp.open_group("system");
          nda::h5_read(sys_grp, "H0_skij", H0);
          F_mf -= H0;
        }
      }
      F_mf -= sF_skij.local();
      auto Fmax_iter = max_element(F_mf.data(), F_mf.data()+F_mf.size(),
                                   [](auto a, auto b) { return std::abs(a) < std::abs(b); });
      conv_F =  std::abs((*Fmax_iter));
    }
    context.node_comm.broadcast_n(&conv_F, 1, 0);
    auto Sigma_max_iter = max_element(sSigma_tskij.local().data(), sSigma_tskij.local().data()+sSigma_tskij.local().size(),
                                      [](auto a, auto b) { return std::abs(a) < std::abs(b); });
    conv_Sigma =  std::abs((*Sigma_max_iter));
    if (iter_solver.iter_alg() == iter_scf::DIIS and context.comm.root()) {
      // Initialize DIIS solver at the root process since the solver currently doesn't support mpi
      diis_init(iter_solver, iteration, h5_prefix, sF_skij, sSigma_tskij, FT);
    }
    context.comm.barrier();
  } else {

    if (iter_solver.iter_alg() == iter_scf::damping) {
      std::tie(conv_F, conv_Sigma) = damping_impl(context, iter_solver, iteration, h5_prefix,
                                                  sF_skij, sSigma_tskij, datasets);
    } else if (iter_solver.iter_alg() == iter_scf::DIIS) {
      std::tie(conv_F, conv_Sigma) = diis_impl(context, iter_solver, iteration, h5_prefix,
                                               sF_skij, sSigma_tskij, FT, datasets);
    } else {
      utils::check(false, "scf_common::solve_iterative: unknown type of iterative algorithm.");
    }
  }
  return std::make_tuple(conv_F, conv_Sigma);
}

template<typename dyson_type>
void write_mf_data(mf::MF &mf,
                   const imag_axes_ft::IAFT &ft, dyson_type &dyson,
                   std::string output) {
  auto mpi = mf.mpi();
  sArray_t<Array_view_4D_t> sF_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_4D_t> sDm_skij(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> G_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  sArray_t<Array_view_5D_t> Sigma_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), mf.nspin(), mf.nkpts_ibz(), mf.nbnd(), mf.nbnd()}));
  hamilt::set_fock(mf, dyson.PSP(), sF_skij, true);
  double mu = 0.0;

  // init Green's function. By default, we update mu as well.
  update_G(dyson, mf, ft, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, false);

  chkpt::write_metadata(mpi->comm, mf, ft, dyson.sH0_skij(), dyson.sS_skij(), output);
  chkpt::dump_scf(mpi->comm, 0, sDm_skij, G_shm, sF_skij, Sigma_shm, mu, output);
}

template<typename MPI_Context_t>
auto read_greens_function(MPI_Context_t &context, mf::MF *mf,
                          std::string filename, long scf_iter, std::string scf_grp)
-> sArray_t<Array_view_5D_t> {
  using math::shm::make_shared_array;

  h5::file file(filename, 'r');
  h5::group grp(file);

  nda::array<double, 1> tau_mesh;
  auto iaft_grp = h5::group(file).open_group("imaginary_fourier_transform");
  auto tau_grp = iaft_grp.open_group("tau_mesh");
  nda::h5_read(tau_grp, "fermion", tau_mesh);
  int nts = tau_mesh.shape(0);
  int ns = mf->nspin();
  int nkpts_ibz = mf->nkpts_ibz();
  int nbnd = mf->nbnd();

  auto sG_tskij = make_shared_array<Array_view_5D_t>(context.comm, context.internode_comm,
                                                     context.node_comm, {nts, ns, nkpts_ibz, nbnd, nbnd});

  auto iter_grp = h5::group(file).open_group(scf_grp+"/iter"+std::to_string(scf_iter));
  if (iter_grp.has_dataset("G_tskij")) {
    // it's a Dyson type calculation -> read Green's function
    sG_tskij.win().fence();
    if (context.node_comm.root()) {
      auto Gloc = sG_tskij.local();
      nda::h5_read(iter_grp, "G_tskij", Gloc);
    }
    sG_tskij.win().fence();
  } else {
    // it's a qp type calculation -> construct the Green's function on-the-fly
    auto ft = imag_axes_ft::read_iaft(filename, false);
    auto sMO_skia = make_shared_array<Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd, nbnd});
    auto sE_ska = make_shared_array<Array_view_3D_t>(
        context.comm, context.internode_comm, context.node_comm, {ns, nkpts_ibz, nbnd});
    double mu;

    sMO_skia.win().fence();
    if (context.node_comm.root()) {
      auto MO_loc = sMO_skia.local();
      auto E_loc = sE_ska.local();
      nda::h5_read(iter_grp, "MO_skia", MO_loc);
      nda::h5_read(iter_grp, "E_ska", E_loc);
    }
    sMO_skia.win().fence();
    h5::h5_read(iter_grp, "mu", mu);

    update_G(sG_tskij, sMO_skia, sE_ska, mu, ft);
  }
  context.comm.barrier();
  return sG_tskij;
}


template auto eval_hf_energy(const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
                             nda::array_contiguous_const_view<double, 1>&, bool)
    -> std::tuple<double, double>;

template double eval_corr_energy(mpi3::communicator& comm, const imag_axes_ft::IAFT &,
                                 const sArray_t<Array_view_5D_t> &, const sArray_t<Array_view_5D_t> &,
                                 nda::array_contiguous_const_view<double, 1>&);

template void eval_thermodynamic_properties(simple_dyson&, const sArray_t<Array_view_4D_t>&,
                                            const sArray_t<Array_view_5D_t>&, const sArray_t<Array_view_5D_t>&,
                                            const std::vector<double>&, double, double, bool);

template void update_G(simple_dyson &, const mf::MF &, const imag_axes_ft::IAFT &,
                       sArray_t<Array_view_4D_t> & Dm, sArray_t<Array_view_5D_t> &G,
                       const sArray_t<Array_view_4D_t> & F, const sArray_t<Array_view_5D_t> &Sigma, double&,
                       bool);

template double update_mu(double, simple_dyson&, const mf::MF &, const imag_axes_ft::IAFT &,
                          const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&);

template auto solve_iterative(utils::mpi_context_t<mpi3::communicator>&, iter_scf::iter_scf_t&, long, std::string,
                              sArray_t<Array_view_4D_t>&, sArray_t<Array_view_5D_t>&, const imag_axes_ft::IAFT*,
                              std::array<std::string,3>)
         -> std::tuple<double, double>;

template void write_mf_data(mf::MF&, const imag_axes_ft::IAFT&, simple_dyson&,
                            std::string);
template auto read_greens_function(utils::mpi_context_t<>&, mf::MF*, std::string, long, std::string)
    -> sArray_t<Array_view_5D_t>;
} // methods
