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


#ifndef COQUI_GRADIENT_COMMON_H
#define COQUI_GRADIENT_COMMON_H

#include "gradient_common.h"

#include "IO/app_loggers.h"

namespace methods {

nda::array<ComplexType, 2> eval_grad_1e(std::shared_ptr<mf::MF> mf, const nda::MemoryArrayOfRank<4> auto &D_skij)
{
  auto grad_1e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
    for (int direction = 0; direction < 3; ++direction) {
      grad_1e(iatom, direction) += eval_grad_1e(iatom, direction, mf, D_skij);
    }
  }
  return grad_1e;
}

nda::array<ComplexType, 2> eval_grad_pulay(std::shared_ptr<mf::MF> mf,
                                           const imag_axes_ft::IAFT &FT,
                                           const nda::MemoryArrayOfRank<4> auto &S_skij,
                                           const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                           double mu)
{
  auto DE_skij = eval_DE(mf, FT, S_skij, G_tskij, mu);
  auto grad_pulay = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
    for (int direction = 0; direction < 3; ++direction) {
      grad_pulay(iatom, direction) += eval_grad_pulay(iatom, direction, mf, DE_skij);
    }
  }
  return grad_pulay;
}

nda::array<ComplexType, 2> eval_grad_pulay(std::shared_ptr<mf::MF> mf,
                                           const nda::MemoryArrayOfRank<4> auto &D_skij,
                                           const nda::MemoryArrayOfRank<4> auto &F_skij,
                                           const nda::MemoryArrayOfRank<4> auto &S_skij,
                                           const nda::MemoryArrayOfRank<4> auto &H0_skij,
                                           bool F_has_H0)
{
  auto DE_skij = eval_DE(mf, D_skij, F_skij, S_skij, H0_skij, F_has_H0);
  auto grad_pulay = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
    for (int direction = 0; direction < 3; ++direction) {
      grad_pulay(iatom, direction) += eval_grad_pulay(iatom, direction, mf, DE_skij);
    }
  }
  return grad_pulay;
}

nda::array<ComplexType, 4> eval_DE(std::shared_ptr<mf::MF> mf,
                                   const imag_axes_ft::IAFT &FT,
                                   const nda::MemoryArrayOfRank<4> auto &S_skij,
                                   const nda::MemoryArrayOfRank<5> auto &G_tskij,
                                   double mu)
{
  int nbnd = mf->nbnd();
  int nspin = mf->nspin();
  int nkpts = mf->nkpts();

  auto nt = FT.nt_f();
  auto nw = FT.nw_f();

  auto DE_skij = nda::array<ComplexType, 4>::zeros({nspin, nkpts, nbnd, nbnd});
  auto DE_beta_skij = nda::array<ComplexType, 4>::zeros({nspin, nkpts, nbnd, nbnd});
  auto Gw_skij = nda::array<ComplexType, 4>::zeros({nspin, nkpts, nbnd, nbnd});
  auto omega_mu_G_tskij = nda::array<ComplexType, 5>::zeros({nt, nspin, nkpts, nbnd, nbnd});
  auto omega_mu_G_wskij = nda::array<ComplexType, 5>::zeros({nw, nspin, nkpts, nbnd, nbnd});

  for (int n = 0; n < nw; ++n) {
    auto wn = FT.wn_mesh()(n);
    ComplexType omega_mu = FT.omega(wn) + mu;
    FT.tau_to_w(G_tskij, Gw_skij, imag_axes_ft::fermion, n);
    for (int ispin = 0; ispin < nspin; ++ispin) {
      for (int ik = 0; ik < nkpts; ++ik) {
        nda::matrix<ComplexType> S = nda::make_regular(S_skij(ispin, ik, nda::ellipsis{}));
        nda::matrix<ComplexType> S_inv = nda::inverse(S);
        omega_mu_G_wskij(n, ispin, ik, nda::ellipsis{}) = omega_mu * Gw_skij(ispin, ik, nda::ellipsis{}) - S_inv;
      }
    }
  }
  FT.w_to_tau(omega_mu_G_wskij, omega_mu_G_tskij, imag_axes_ft::fermion);
  FT.tau_to_beta(omega_mu_G_tskij, DE_beta_skij);

  DE_skij = -1 * DE_beta_skij;

  return DE_skij;
}

nda::array<ComplexType, 4> eval_DE(std::shared_ptr<mf::MF> mf,
                                   const nda::MemoryArrayOfRank<4> auto &D_skij,
                                   const nda::MemoryArrayOfRank<4> auto &F_skij,
                                   const nda::MemoryArrayOfRank<4> auto &S_skij,
                                   const nda::MemoryArrayOfRank<4> auto &H0_skij,
                                   bool F_has_H0)
{
  int nbnd = mf->nbnd();
  int nspin = mf->nspin();
  int nkpts = mf->nkpts();

  auto DE_skij = nda::array<ComplexType, 4>::zeros({nspin, nkpts, nbnd, nbnd});
  for (int ispin = 0; ispin < nspin; ++ispin) {
    for (int ikpt = 0; ikpt < nkpts; ++ikpt) {
      auto tmp = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
      auto SC = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
      auto F = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
      auto S = nda::array<ComplexType, 2>::zeros({nbnd, nbnd});
      if (F_has_H0) {
        F = F_skij(ispin, ikpt, nda::ellipsis{});
      } else {
        F = F_skij(ispin, ikpt, nda::ellipsis{}) + H0_skij(ispin, ikpt, nda::ellipsis{});
      }
      S = S_skij(ispin, ikpt, nda::ellipsis{});
      auto [energies, coeffs] = nda::linalg::eigenelements(F, S);
      nda::blas::gemm(1.0, S_skij(ispin, ikpt, nda::ellipsis{}), coeffs, 0.0, SC);
      nda::blas::gemm(1.0, nda::transpose(nda::conj(SC)), D_skij(ispin, ikpt, nda::ellipsis{}), 0.0, tmp);
      nda::blas::gemm(1.0, tmp, SC, 0.0, DE_skij(ispin, ikpt, nda::ellipsis{}));
      for (int ibnd = 0; ibnd < nbnd; ++ibnd) {
        DE_skij(ispin, ikpt, ibnd, ibnd) = energies(ibnd) * DE_skij(ispin, ikpt, ibnd, ibnd);
      }
      nda::blas::gemm(1.0, coeffs, DE_skij(ispin, ikpt, nda::ellipsis{}), 0.0, tmp);
      nda::blas::gemm(1.0, tmp, nda::transpose(nda::conj(coeffs)), 0.0, DE_skij(ispin, ikpt, nda::ellipsis{}));
    }
  }
  return DE_skij;
}

ComplexType eval_grad_1e(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                         const nda::MemoryArrayOfRank<4> auto &D_skij)
{
  int nspin = mf->nspin();
  int nkpts = mf->nkpts();
  int npol = mf->npol();

  nda::array<RealType, 1> k_weight = mf->k_weight();

  ComplexType tmp_grad = 0;
  auto H0_grad = mf->H0_grad();
  RealType spin_factor = (nspin == 1 and npol == 1) ? 2.0 : 1.0;
  for (int ispin = 0; ispin < nspin; ++ispin) {
    for (int ikpt = 0; ikpt < nkpts; ++ikpt) {
      tmp_grad += k_weight(ikpt) * spin_factor *
                  nda::sum(D_skij(ispin, ikpt, nda::ellipsis{}) *
                           H0_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));
    }
  }
  return tmp_grad;
}

ComplexType eval_grad_pulay(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                            const imag_axes_ft::IAFT &FT,
                            const nda::MemoryArrayOfRank<4> auto &S_skij,
                            const nda::MemoryArrayOfRank<5> auto &G_tskij,
                            double mu)
{
  auto DE_skij = eval_DE(mf, FT, S_skij, G_tskij, mu);

  ComplexType tmp_grad = 0;
  tmp_grad = eval_grad_pulay(iatom, direction, mf, DE_skij);
  return tmp_grad;
}


ComplexType eval_grad_pulay(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                            const nda::MemoryArrayOfRank<4> auto &D_skij,
                            const nda::MemoryArrayOfRank<4> auto &F_skij,
                            const nda::MemoryArrayOfRank<4> auto &S_skij,
                            const nda::MemoryArrayOfRank<4> auto &H0_skij,
                            bool F_has_H0)
{
  auto DE_skij = eval_DE(mf, D_skij, F_skij, S_skij, H0_skij, F_has_H0);

  ComplexType tmp_grad = 0;
  tmp_grad = eval_grad_pulay(iatom, direction, mf, DE_skij);
  return tmp_grad;
}

ComplexType eval_grad_pulay(int iatom, int direction, std::shared_ptr<mf::MF> mf,
                            const nda::MemoryArrayOfRank<4> auto &DE_skij)
{
  int nspin = mf->nspin();
  int nkpts = mf->nkpts();
  int npol = mf->npol();

  nda::array<RealType, 1> k_weight = mf->k_weight();

  ComplexType tmp_grad = 0;
  auto S_grad = mf->S_grad();
  RealType spin_factor = (nspin == 1 and npol == 1) ? 2.0 : 1.0;
  for (int ispin = 0; ispin < nspin; ++ispin) {
    for (int ikpt = 0; ikpt < nkpts; ++ikpt) {
      tmp_grad -= k_weight(ikpt) * spin_factor *
                  nda::sum(DE_skij(ispin, ikpt, nda::ellipsis{}) *
                           S_grad(iatom, direction, ispin, ikpt, nda::ellipsis{}));
    }
  }
  return tmp_grad;
}

template<typename data_type>
void print_mbpt_gradient(const nda::array<data_type, 2>& gradient, std::shared_ptr<mf::MF> mf,
                          const std::string &str, bool bohr)
{
  double factor;
  std::string unit;
  if (bohr) {
    factor = 1;
    unit = "(hartree/bohr)";
  } else {
    factor = 0.52917721054482;
    unit = "(hartree/angstrom)";
  }

  app_log(1, "  {}", str);
  app_log(1, "  --------------------------------------------------------------------------------------");
  app_log(1, "   {:<5}{:<5}{:>20}{:>20}{:>20}", "id", "nuc", "X" + unit, "Y" + unit, "Z" + unit);
  app_log(1, "  --------------------------------------------------------------------------------------");
  for (int iatom = 0; iatom < mf->number_of_atoms(); ++iatom) {
    app_log(1, "   {:<5}{:<5}{:>+20.10f}{:>+20.10f}{:>+20.10f}",
      iatom, mf->atomic_id(iatom),
      nda::real(gradient(iatom, 0) * factor),
      nda::real(gradient(iatom, 1) * factor),
      nda::real(gradient(iatom, 2) * factor));
  }
  app_log(1, "\n");
}

template<typename communicator_t, typename data_type>
void write_mbpt_gradient(communicator_t &comm, const nda::array<data_type, 2>& gradient,
                         const std::string &output, long iter)
{
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    std::string iter_grp_name = "iter" + std::to_string(iter);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto scf_grp = (grp.has_subgroup("scf")) ? grp.open_group("scf") : grp.create_group("scf");
    auto iter_grp = (scf_grp.has_subgroup(iter_grp_name)) ?
                    scf_grp.open_group(iter_grp_name) : scf_grp.create_group(iter_grp_name);

    nda::h5_write(iter_grp, "gradient", nda::make_regular(nda::real(gradient)), false);
  }
  comm.barrier();
}

using Arr2D = nda::array<ComplexType, 2>;
using Arr4D = nda::array<ComplexType, 4>;
using Arr5D = nda::array<ComplexType, 5>;
using Arrv4D = nda::array_view<ComplexType, 4>;
using Arrv5D = nda::array_view<ComplexType, 5>;

template Arr2D eval_grad_1e(std::shared_ptr<mf::MF>, const Arr4D&);
template Arr2D eval_grad_1e(std::shared_ptr<mf::MF>, const Arrv4D&);

template Arr2D eval_grad_pulay(std::shared_ptr<mf::MF>, const imag_axes_ft::IAFT&, const Arr4D&, const Arr5D&, double);
template Arr2D eval_grad_pulay(std::shared_ptr<mf::MF>, const imag_axes_ft::IAFT&, const Arrv4D&, const Arrv5D&, double);
template Arr2D eval_grad_pulay(std::shared_ptr<mf::MF>, const Arr4D&, const Arr4D&, const Arr4D&, const Arr4D&, bool);
template Arr2D eval_grad_pulay(std::shared_ptr<mf::MF>, const Arrv4D&, const Arrv4D&, const Arrv4D&, const Arrv4D&, bool);

template Arr4D eval_DE(std::shared_ptr<mf::MF>, const imag_axes_ft::IAFT&, const Arr4D&, const Arr5D&, double);
template Arr4D eval_DE(std::shared_ptr<mf::MF>, const imag_axes_ft::IAFT&, const Arrv4D&, const Arrv5D&, double);

template Arr4D eval_DE(std::shared_ptr<mf::MF>, const Arr4D&, const Arr4D&, const Arr4D&, const Arr4D&, bool);
template Arr4D eval_DE(std::shared_ptr<mf::MF>, const Arrv4D&, const Arrv4D&, const Arrv4D&, const Arrv4D&, bool);

template ComplexType eval_grad_1e(int, int, std::shared_ptr<mf::MF>, const Arr4D&);
template ComplexType eval_grad_1e(int, int, std::shared_ptr<mf::MF>, const Arrv4D&);

template ComplexType eval_grad_pulay(int, int, std::shared_ptr<mf::MF>, const Arr4D&, const Arr4D&, const Arr4D&, const Arr4D&, bool);
template ComplexType eval_grad_pulay(int, int, std::shared_ptr<mf::MF>, const Arrv4D&, const Arrv4D&, const Arrv4D&, const Arrv4D&, bool);
template ComplexType eval_grad_pulay(int, int, std::shared_ptr<mf::MF>, const Arr4D&);
template ComplexType eval_grad_pulay(int, int, std::shared_ptr<mf::MF>, const Arrv4D&);

template void print_mbpt_gradient(const nda::array<RealType, 2>&, std::shared_ptr<mf::MF>,
                                  const std::string&, bool);
template void print_mbpt_gradient(const nda::array<ComplexType, 2> &, std::shared_ptr<mf::MF>,
                                  const std::string&, bool);

template void write_mbpt_gradient(mpi3::communicator&, const nda::array<RealType, 2>&, const std::string&, long);
template void write_mbpt_gradient(mpi3::communicator&, const nda::array<ComplexType, 2>&, const std::string&, long);

} // namespace methods

#endif
