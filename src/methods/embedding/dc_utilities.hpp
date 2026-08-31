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


#ifndef AIMBES_DC_UTILITIES_HPP
#define AIMBES_DC_UTILITIES_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "IO/app_loggers.h"
#include "mean_field/MF.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/SCF/scf_driver.hpp"

#include "mean_field/model_hamiltonian/model_utils.hpp"

namespace methods {

nda::array<ComplexType, 4>
hartree_double_counting(const nda::MemoryArrayOfRank<4> auto &Dm_sIab,
                        const nda::MemoryArrayOfRank<4> auto &V_abcd) {
  auto [ns, nImps, nImpOrbs, nImpOrbs2] = Dm_sIab.shape();
  utils::check(nImps==1, "hartree_double_counting: nImps > 1 is not implemented.");

  nda::array<ComplexType, 4> J_dc_sIab(Dm_sIab.shape());
  double spin_factor = (ns==1)? 2.0 : 1.0;
  for (size_t s=0; s<ns; ++s) {
    for (size_t a=0; a<nImpOrbs; ++a) {
      for (size_t b=0; b<nImpOrbs; ++b) {
        for (size_t c=0; c<nImpOrbs; ++c) {
          for (size_t d=0; d<nImpOrbs; ++d) {
            J_dc_sIab(0,0,a,b) += spin_factor * Dm_sIab(s,0,d,c) * V_abcd(a,b,c,d);
          }
        }
      }
    }
  }
  if (ns==2) J_dc_sIab(1,nda::ellipsis{}) = J_dc_sIab(0,nda::ellipsis{});
  return J_dc_sIab;
}

nda::array<ComplexType, 4>
exchange_double_counting(const nda::MemoryArrayOfRank<4> auto &Dm_sIab,
                         const nda::MemoryArrayOfRank<4> auto &V_abcd) {
  auto [ns, nImps, nImpOrbs, nImpOrbs2] = Dm_sIab.shape();
  utils::check(nImps==1, "exchange_double_counting: nImps > 1 is not implemented.");

  nda::array<ComplexType, 4> K_dc_sIab(Dm_sIab.shape());
  for (size_t s=0; s<ns; ++s) {
    for (size_t a=0; a<nImpOrbs; ++a) {
      for (size_t b=0; b<nImpOrbs; ++b) {
        for (size_t c=0; c<nImpOrbs; ++c) {
          for (size_t d=0; d<nImpOrbs; ++d) {
            K_dc_sIab(s,0,a,b) -= Dm_sIab(s,0,c,d) * V_abcd(a,c,d,b);
          }
        }
      }
    }
  }
  return K_dc_sIab;
}

// V_abcd = sum_n V_3d(n,a,b) * conj(V_3d(n,d,c))
// J(a,b) = sum_n_c_d D(d,c) V(n,a,b) * conj(V(n,d,c)) 
nda::array<ComplexType, 4> hartree_double_counting(const nda::MemoryArrayOfRank<4> auto &Dm_sIab,
                                                   const nda::MemoryArrayOfRank<3> auto &V_nab) {
  auto [ns, nImps, nImpOrbs, nImpOrbs2] = Dm_sIab.shape();
  utils::check(nImps==1, "hartree_double_counting: nImps > 1 is not implemented.");

  nda::array<ComplexType, 4> J_dc_sIab(Dm_sIab.shape());
  double spin_factor = (ns==1)? 2.0 : 1.0;
  long nchol = V_nab.extent(0);
  nda::array<ComplexType, 1> T(nchol, 0.0); 
  for (size_t s=0; s<ns; ++s) {
    for (size_t n=0; n<nchol; ++n) {
      for (size_t c=0; c<nImpOrbs; ++c) {
        for (size_t d=0; d<nImpOrbs; ++d) {
          T(n) += spin_factor * Dm_sIab(s,0,d,c) * conj(V_nab(n,d,c));
        }
      }
    }
  }
    
  for (size_t n=0; n<nchol; ++n) {
    for (size_t a=0; a<nImpOrbs; ++a) {
      for (size_t b=0; b<nImpOrbs; ++b) {
        J_dc_sIab(0,0,a,b) += T(n) * V_nab(n,a,b);
      }
    }
  }
  if (ns==2) J_dc_sIab(1,nda::ellipsis{}) = J_dc_sIab(0,nda::ellipsis{});
  return J_dc_sIab;
}

nda::array<ComplexType, 4> exchange_double_counting(const nda::MemoryArrayOfRank<4> auto &Dm_sIab,
                                                    const nda::MemoryArrayOfRank<3> auto &V_nab) {
  using nda::range;
  decltype(nda::range::all) all;
  auto [ns, nImps, nImpOrbs, nImpOrbs2] = Dm_sIab.shape();
  utils::check(nImps==1, "exchange_double_counting: nImps > 1 is not implemented.");

  nda::array<ComplexType, 4> K_dc_sIab(Dm_sIab.shape());
  long nchol = V_nab.extent(0);
  nda::array<ComplexType, 2> Tcb(nImpOrbs,nImpOrbs);
  Tcb() = ComplexType(0.0);
  K_dc_sIab() = ComplexType(0.0);

  //K_dc_sIab(s,0,a,b) -= Dm_sIab(s,0,c,d) * V_abcd(a,c,d,b);
  //                   -= sum_n Dm_sIab(s,0,c,d) * L(n,a,c) * conj(L(n,b,d)) 
  //T(c,b) += Dm_sIab(s,0,c,d) * conj(V_nab(n,b,d));
  //K(a,b)   -= V_nab(n,a,c) * T(c,b);
// batch dispatch for openmp or gpu
  for (size_t s=0; s<ns; ++s) { 
    for (size_t n=0; n<nchol; ++n) { 
      nda::blas::gemm(ComplexType(1.0),Dm_sIab(s,0,all,all),nda::dagger(V_nab(n,all,all)),
                      ComplexType(0.0),Tcb);
      nda::blas::gemm(ComplexType(-1.0),V_nab(n,all,all),Tcb,ComplexType(1.0),K_dc_sIab(s,0,all,all));
    }
  }
  return K_dc_sIab;
}

//MAM: instead of reimplementing gw_double_counting_dmft for cholesky decomposed U_nab, 
//use choleksy based GW solver, just needs G_tskij in shared memory and a chol_reader object
//which can be created from /Interaction object already 

  template<bool w_out, typename MPI_context_t>
  auto gw_double_counting_dmft_cholesky(
      std::shared_ptr<MPI_context_t> mpi,
      const nda::MemoryArrayOfRank<5> auto &G_tsIab,
      std::string chol_file, imag_axes_ft::IAFT &ft)
  {
    utils::check(G_tsIab.extent(0) == ft.nt_f(), "Error: Shape mismatch.");
    int nb = G_tsIab.extent(3);

    std::string prefix = "__dummy_model__";
    auto mf = std::make_shared<mf::MF>(mf::MF(mf::model::make_dummy_model(mpi,nb,1.0)));
    solvers::hf_t hf;
    solvers::gw_t gw(&ft, "ignore_g0", prefix);
    chol_reader_t chol(mf, "./", chol_file, each_q, single_file);

    sArray_t<Array_view_5D_t> Sigma_shm(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {ft.nt_f(), 1, 1, nb, nb}));

    gw.evaluate(G_tsIab, Sigma_shm, chol);
    nda::array<ComplexType, 5> gw_dc_tsIab(G_tsIab.shape());
    gw_dc_tsIab() = Sigma_shm.local();
    return gw_dc_tsIab;
  }

  template<bool w_out>
  auto gw_double_counting_dmft(utils::Communicator auto &comm,
                               const nda::MemoryArrayOfRank<5> auto &G_tsIab,
                               const nda::MemoryArrayOfRank<4> auto &V_abcd,
                               imag_axes_ft::IAFT &ft)
  -> nda::array<ComplexType, 5> {
  auto [nts, ns, nImps, nImpOrbs, nImpOrbs2] = G_tsIab.shape();
  utils::check(nts==ft.nt_f(), "gw_double_counting_dmft: incorrect dimension of nt = {}. It should be {}",
               nts, ft.nt_f());

  nda::array<ComplexType, 5> gw_dc_tsIab(G_tsIab.shape());

  // polarization
  double spin_factor = (ns==1)? -2.0 : -1.0;
  long nt_half = (nts%2==0)? nts/2 : nts/2 + 1;
  nda::array<ComplexType, 5> Pi_tabcd(nt_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  for (size_t it=0; it<nt_half; ++it) {
    size_t itt = nts-it-1;
    for (size_t a=0; a<nImpOrbs; ++a) {
      for (size_t b=0; b<nImpOrbs; ++b) {
        for (size_t c=0; c<nImpOrbs; ++c) {
          for (size_t d=0; d<nImpOrbs; ++d) {
            for (size_t s=0; s<ns; ++s) {
              Pi_tabcd(it,a,b,c,d) += spin_factor * G_tsIab(it,s,0,b,d) * G_tsIab(itt,s,0,c,a);
            }
          }
        }
      }
    }
  }
  ft.check_leakage_PHsym(Pi_tabcd, imag_axes_ft::boson, std::addressof(comm), "local polarizability");

  // dyson equation for screened interactions
  long nw_half = (ft.nw_b()%2==0)? ft.nw_b()/2 : ft.nw_b()/2 + 1;
  nda::array<ComplexType, 5> Pi_wabcd(nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  ft.tau_to_w_PHsym(Pi_tabcd, Pi_wabcd);

  nda::array<ComplexType, 4> Vpb_bacd(V_abcd.shape());
  for (size_t a=0; a<nImpOrbs; ++a) {
    for (size_t b=0; b<nImpOrbs; ++b) {
      Vpb_bacd(b, a, nda::ellipsis{}) = V_abcd(a, b, nda::ellipsis{});
    }
  }

  auto Pi_3D = nda::reshape(Pi_wabcd, std::array<long, 3>{nw_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  auto W_3D = nda::reshape(Pi_wabcd, std::array<long, 3>{nw_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  auto Vpb_2D = nda::reshape(Vpb_bacd, std::array<long, 2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  nda::matrix<ComplexType> tmp(nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs);
  for (size_t w=0; w<nw_half; ++w) {
    tmp() = 1.0;
    // tmp = I - V*Pi
    nda::blas::gemm(ComplexType(-1.0), Vpb_2D, Pi_3D(w,nda::ellipsis{}), ComplexType(1.0), tmp);
    // tmp = inverse(tmp)
    tmp = nda::inverse(tmp);
    // W = [I - V*Pi]^{-1}*V - V
    nda::blas::gemm(tmp, Vpb_2D, W_3D(w, nda::ellipsis{}));
    W_3D(w, nda::ellipsis{}) -= Vpb_2D;
  }

  nda::array<ComplexType, 5> Wpb_tabcd(nt_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  //auto Wpb_tabcd = Pi_tabcd();
  ft.w_to_tau_PHsym(Pi_wabcd, Wpb_tabcd);
  ft.check_leakage_PHsym(Wpb_tabcd, imag_axes_ft::boson, std::addressof(comm), "local screened interactions");

  // self-energy
  for (size_t it=0; it<nt_half; ++it) {
    size_t imt = nts-it-1;
    for (size_t is=0; is<ns; ++is) {
      for (size_t a=0; a<nImpOrbs; ++a) {
        for (size_t b=0; b<nImpOrbs; ++b) {
          for (size_t c=0; c<nImpOrbs; ++c) {
            for (size_t d=0; d<nImpOrbs; ++d) {
              gw_dc_tsIab(it,is,0,a,b) -= G_tsIab(it,is,0,c,d) * Wpb_tabcd(it,c,a,d,b);
              if (it != imt)
                gw_dc_tsIab(imt,is,0,a,b) -= G_tsIab(imt,is,0,c,d) * Wpb_tabcd(it,c,a,d,b);
            }
          }
        }
      }
    }
  }
  ft.check_leakage(gw_dc_tsIab, imag_axes_ft::fermion, std::addressof(comm), "double counting GW self-energy");

  if constexpr (w_out) {
    nda::array<ComplexType, 5> gw_dc_wsIab(ft.nw_f(), ns, nImps, nImpOrbs, nImpOrbs);
    ft.tau_to_w(gw_dc_tsIab, gw_dc_wsIab, imag_axes_ft::fermion);
    return gw_dc_wsIab;
  } else {
    return gw_dc_tsIab;
  }
}

template<bool w_out>
auto gw_double_counting_dmft(utils::Communicator auto &comm,
                             const nda::MemoryArrayOfRank<5> auto &G_tsIab,
                             const nda::MemoryArrayOfRank<4> auto &V_abcd,
                             const nda::MemoryArrayOfRank<5> auto &U_wabcd,
                             imag_axes_ft::IAFT &ft)
-> nda::array<ComplexType, 5> {
  auto [nts, ns, nImps, nImpOrbs, nImpOrbs2] = G_tsIab.shape();
  utils::check(nts==ft.nt_f(), "gw_double_counting_dmft: incorrect dimension of nt = {}. It should be {}",
               nts, ft.nt_f());

  nda::array<ComplexType, 5> gw_dc_tsIab(G_tsIab.shape());

  // polarization
  double spin_factor = (ns==1)? -2.0 : -1.0;
  long nt_half = (nts%2==0)? nts/2 : nts/2 + 1;
  nda::array<ComplexType, 5> Pi_tabcd(nt_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  for (size_t it=0; it<nt_half; ++it) {
    size_t itt = nts-it-1;
    for (size_t a=0; a<nImpOrbs; ++a) {
      for (size_t b=0; b<nImpOrbs; ++b) {
        for (size_t c=0; c<nImpOrbs; ++c) {
          for (size_t d=0; d<nImpOrbs; ++d) {
            for (size_t s=0; s<ns; ++s) {
              Pi_tabcd(it,a,b,c,d) += spin_factor * G_tsIab(it,s,0,b,d) * G_tsIab(itt,s,0,c,a);
            }
          }
        }
      }
    }
  }
  ft.check_leakage_PHsym(Pi_tabcd, imag_axes_ft::boson, std::addressof(comm), "local polarizability");

  // dyson equation
  long nw_half = (ft.nw_b()%2==0)? ft.nw_b()/2 : ft.nw_b()/2 + 1;
  nda::array<ComplexType, 5> Pi_wabcd(nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  ft.tau_to_w_PHsym(Pi_tabcd, Pi_wabcd);

  nda::array<ComplexType, 4> Vpb_bacd(V_abcd.shape());
  nda::array<ComplexType, 5> Upb_wbacd(U_wabcd.shape());
  for (size_t a=0; a<nImpOrbs; ++a) {
    for (size_t b=0; b<nImpOrbs; ++b) {
      Vpb_bacd(b, a, nda::ellipsis{}) = V_abcd(a, b, nda::ellipsis{});
      Upb_wbacd(nda::range::all, b, a, nda::ellipsis{}) = U_wabcd(nda::range::all, a, b, nda::ellipsis{});
    }
  }

  auto Pi_3D = nda::reshape(Pi_wabcd, std::array<long, 3>{nw_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  auto W_3D = nda::reshape(Pi_wabcd, std::array<long, 3>{nw_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  auto Vpb_2D = nda::reshape(Vpb_bacd, std::array<long, 2>{nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  auto Upb_3D = nda::reshape(Upb_wbacd, std::array<long, 3>{nw_half, nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs});
  nda::matrix<ComplexType> tmp(nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs);
  nda::matrix<ComplexType> VpU(nImpOrbs*nImpOrbs, nImpOrbs*nImpOrbs);
  for (size_t w=0; w<nw_half; ++w) {
    VpU = Vpb_2D + Upb_3D(w,nda::ellipsis{});
    tmp() = 1.0;
    // tmp = I - (V+U)*Pi
    nda::blas::gemm(ComplexType(-1.0), VpU, Pi_3D(w,nda::ellipsis{}), ComplexType(1.0), tmp);
    // tmp = inverse(tmp)
    tmp = nda::inverse(tmp);
    // W = [I - (V+U)*Pi]^{-1}*(V+U) - V
    nda::blas::gemm(tmp, VpU, W_3D(w, nda::ellipsis{}));
    W_3D(w, nda::ellipsis{}) -= Vpb_2D;
  }

  nda::array<ComplexType, 5> Wpb_tabcd(nt_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  ft.w_to_tau_PHsym(Pi_wabcd, Wpb_tabcd);
  ft.check_leakage_PHsym(Wpb_tabcd, imag_axes_ft::boson, std::addressof(comm), "local screened interactions");

  // self-energy
  for (size_t it=0; it<nt_half; ++it) {
    size_t imt = nts-it-1;
    for (size_t is=0; is<ns; ++is) {
      for (size_t a=0; a<nImpOrbs; ++a) {
        for (size_t b=0; b<nImpOrbs; ++b) {
          for (size_t c=0; c<nImpOrbs; ++c) {
            for (size_t d=0; d<nImpOrbs; ++d) {
              gw_dc_tsIab(it,is,0,a,b) -= G_tsIab(it,is,0,c,d) * Wpb_tabcd(it,c,a,d,b);
              if (it != imt)
                gw_dc_tsIab(imt,is,0,a,b) -= G_tsIab(imt,is,0,c,d) * Wpb_tabcd(it,c,a,d,b);
            }
          }
        }
      }
    }
  }
  ft.check_leakage(gw_dc_tsIab, imag_axes_ft::fermion, std::addressof(comm), "double counting GW self-energy");

  if constexpr (w_out) {
    nda::array<ComplexType, 5> gw_dc_wsIab(ft.nw_f(), ns, nImps, nImpOrbs, nImpOrbs);
    ft.tau_to_w(gw_dc_tsIab, gw_dc_wsIab, imag_axes_ft::fermion);
    return gw_dc_wsIab;
  } else {
    return gw_dc_tsIab;
  }
}

template<bool w_out>
auto gw_edmft_double_counting(utils::Communicator auto &comm,
                             const nda::MemoryArrayOfRank<5> auto &G_tsIab,
                             const nda::MemoryArrayOfRank<5> auto &W_wabcd,
                             imag_axes_ft::IAFT &ft)
-> nda::array<ComplexType, 5> {
  auto [nts, ns, nImps, nImpOrbs, nImpOrbs2] = G_tsIab.shape();
  long nt_half = (nts%2==0)? nts/2 : nts/2 + 1;
  utils::check(nts==ft.nt_f(), "gw_edmft_double_counting: incorrect dimension of nt = {}. It should be {}",
               nts, ft.nt_f());

  nda::array<ComplexType, 5> gw_dc_tsIab(G_tsIab.shape());

  nda::array<ComplexType, 5> Wpb_wbacd(W_wabcd.shape());
  for (size_t a=0; a<nImpOrbs; ++a) {
    for (size_t b=0; b<nImpOrbs; ++b) {
      Wpb_wbacd(nda::range::all, b, a, nda::ellipsis{}) = W_wabcd(nda::range::all, a, b, nda::ellipsis{});
    }
  }

  nda::array<ComplexType, 5> Wpb_tbacd(nt_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs);
  ft.w_to_tau_PHsym(Wpb_wbacd, Wpb_tbacd);
  ft.check_leakage_PHsym(Wpb_tbacd, imag_axes_ft::boson, std::addressof(comm), "local screened interactions");

  // self-energy
  for (size_t it=0; it<nt_half; ++it) {
    size_t imt = nts-it-1;
    for (size_t is=0; is<ns; ++is) {
      for (size_t a=0; a<nImpOrbs; ++a) {
        for (size_t b=0; b<nImpOrbs; ++b) {
          for (size_t c=0; c<nImpOrbs; ++c) {
            for (size_t d=0; d<nImpOrbs; ++d) {
              gw_dc_tsIab(it,is,0,a,b) -= G_tsIab(it,is,0,c,d) * Wpb_tbacd(it,c,a,d,b);
              if (it != imt)
                gw_dc_tsIab(imt,is,0,a,b) -= G_tsIab(imt,is,0,c,d) * Wpb_tbacd(it,c,a,d,b);
            }
          }
        }
      }
    }
  }
  ft.check_leakage(gw_dc_tsIab, imag_axes_ft::fermion, std::addressof(comm), "double counting GW self-energy");

  if constexpr (w_out) {
    nda::array<ComplexType, 5> gw_dc_wsIab(ft.nw_f(), ns, nImps, nImpOrbs, nImpOrbs);
    ft.tau_to_w(gw_dc_tsIab, gw_dc_wsIab, imag_axes_ft::fermion);
    return gw_dc_wsIab;
  } else {
    return gw_dc_tsIab;
  }
}

/**
 * Evaluate the double counting polarizability at the RPA level in a product basis.
 * @tparam MPI_Context_t
 * @tparam w_out
 * @param mpi
 * @param G_tsIab
 * @param ft
 * @return
 */
template <bool w_out=true, typename MPI_Context_t>
auto eval_Pi_rpa_dc(MPI_Context_t &mpi,
                    const nda::MemoryArrayOfRank<5> auto &G_tsIab,
                    const imag_axes_ft::IAFT &ft,
                    bool density_only=false)
-> sArray_t<Array_view_5D_t> {
  utils::check(ft.nt_b() == G_tsIab.shape(0),
               "eval_Pi_rpa_dc: Inconsistent nts between IAFT ({}) and G_tsIab ({}).", ft.nt_b(), G_tsIab.shape(0));
  app_log(2, "Evaluates RPA double counting polarizability:\n"
             "  - density-density only: {}\n", density_only);
  long nImpOrbs = G_tsIab.shape(3);
  long ns = G_tsIab.shape(1);
  long nts = ft.nt_b();
  long nts_half = (nts%2==0)? nts/2 : nts/2+1;
  double spin_factor = (ns==1)? -2.0 : -1.0;

  // Pi_abcd(t) = spin_factor*G_bd(t)G_ca(beta-t)
  auto sPi_tabcd = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {nts_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
  auto Pi_tabcd = sPi_tabcd.local();
  int node_rank = mpi.node_comm.rank();
  int node_size = mpi.node_comm.size();
  sPi_tabcd.win().fence();
  if (!density_only) {
    for (size_t it = 0, idx = 0; it < nts_half; ++it) {
      size_t itt = nts - it - 1;
      for (size_t a = 0; a < nImpOrbs; ++a)
      for (size_t b = 0; b < nImpOrbs; ++b)
      for (size_t c = 0; c < nImpOrbs; ++c)
      for  (size_t d = 0; d < nImpOrbs; ++d, ++idx) {
        if (node_rank != idx % node_size) continue;
        for (size_t is = 0; is < ns; ++is)
          Pi_tabcd(it, a, b, c, d) += spin_factor * G_tsIab(it, is, 0, b, d) * G_tsIab(itt, is, 0, c, a);
      }
    }
  } else {
    for (size_t it = 0, idx = 0; it < nts_half; ++it) {
      size_t itt = nts - it - 1;
      for (size_t a = 0; a < nImpOrbs; ++a)
      for (size_t b = 0; b < nImpOrbs; ++b, ++idx) {
          if (node_rank != idx % node_size) continue;
          for (size_t is = 0; is < ns; ++is)
            Pi_tabcd(it, a, a, b, b) += spin_factor * G_tsIab(it, is, 0, a, b) * G_tsIab(itt, is, 0, b, a);
      }
    }
  }
  sPi_tabcd.win().fence();

  long nw_half = (ft.nw_b()%2==0)? ft.nw_b()/2 : ft.nw_b()/2+1;
  auto sPi_wabcd = math::shm::make_shared_array<Array_view_5D_t>(
      mpi, {nw_half, nImpOrbs, nImpOrbs, nImpOrbs, nImpOrbs});
  if (mpi.node_comm.root()) {
    ft.tau_to_w_PHsym(sPi_tabcd.local(), sPi_wabcd.local());
  }
  mpi.node_comm.barrier();

  //auto Pi_tabcd = sPi_tabcd.local();
  if (mpi.comm.root()) {
    auto Pi_wabcd = sPi_wabcd.local();
    h5::file file("pi_rpa_loc_debug.h5", 'w');
    auto grp = h5::group(file);
    nda::h5_write(grp, "Pi_tabcd", Pi_tabcd, false);
    nda::h5_write(grp, "Pi_wabcd", Pi_wabcd, false);
  }
  mpi.comm.barrier();

  if constexpr (w_out) {
    return sPi_wabcd;
  } else {
    return sPi_tabcd;
  }
}


} // methods

#endif
