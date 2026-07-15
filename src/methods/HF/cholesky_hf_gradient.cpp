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


#include "methods/ERI/chol_reader_t.hpp"
#include "methods/HF/hf_gradient_t.h"

namespace methods {

  namespace solvers {

    nda::array<ComplexType, 2> hf_gradient_t::evaluate(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                       Cholesky_ERI auto &&chol)
    {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20chol-hf-gradient
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬ ┬┌─┐┬   ┬ ┬┌─┐  ┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌┌┬┐\n"
                 "║  ║ ║║═╬╗║ ║║  │  ├─┤│ ││───├─┤├┤───│ ┬├┬┘├─┤ │││├┤ │││ │ \n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴ ┴└─┘┴─┘ ┴ ┴└    └─┘┴└─┴ ┴─┴┘┴└─┘┘└┘ ┴ \n");

      utils::check(_mf->nkpts() == _mf->nkpts_ibz(),
                   "hf_t::cholesky_hf_grad::evaluate: Symmetry not yet implemented.");

      for( auto& v: {"TOTAL", "COULOMB", "EXCHANGE", "WRITE"} ) {
        _Timer.add(v);
      }

      _Timer.start("TOTAL");

      app_log(1, "  - nbnd:              {}", _nbnd);
      app_log(1, "  - nbnd_aux:          {}", _nbnd_aux);
      app_log(1, "  - nspin:             {}", _nspin);
      app_log(1, "  - nkpts:             {}", _nkpts);
      app_log(1, "  - npol:              {}", _npol);
      app_log(1, "\n");

      nda::array<ComplexType, 2> tmp_grad_2e = eval_grad_2e(D_skij, chol);

      _Timer.stop("TOTAL");

      print_chol_hf_grad_timers();
      chol.print_timers();

      return tmp_grad_2e;
    }

    nda::array<ComplexType, 2> hf_gradient_t::eval_grad_2e(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                           Cholesky_ERI auto && chol)
    {
      auto tmp_grad_2e = nda::array<ComplexType, 2>::zeros({_natoms, 3});
      tmp_grad_2e = eval_grad_coulomb(D_skij, chol) + eval_grad_exchange(D_skij, chol);
      return tmp_grad_2e;
    }

    nda::array<ComplexType, 2> hf_gradient_t::eval_grad_coulomb(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                                Cholesky_ERI auto && chol)
    {
      auto dm_total = eval_dm_total(D_skij);
      auto tmp_grad_coulomb = nda::array<ComplexType, 2>::zeros({_natoms, 3});
      for (int iatom = 0; iatom < _natoms; ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
          tmp_grad_coulomb(iatom, direction) += eval_grad_coulomb(iatom, direction, dm_total, chol);
        }
      }
      return tmp_grad_coulomb;
    }

    nda::array<ComplexType, 2> hf_gradient_t::eval_grad_exchange(const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                                 Cholesky_ERI auto && chol)
    {
      auto tmp_grad_exchange = nda::array<ComplexType, 2>::zeros({_natoms, 3});
      for (int iatom = 0; iatom < _natoms; ++iatom) {
        for (int direction = 0; direction < 3; ++direction) {
          tmp_grad_exchange(iatom, direction) += eval_grad_exchange(iatom, direction, D_skij, chol);
        }
      }
      return tmp_grad_exchange;
    }

    nda::array<ComplexType, 3> hf_gradient_t::eval_dm_total(const nda::MemoryArrayOfRank<4> auto &D_skij)
    {
      decltype(nda::range::all) all;

      RealType spin_factor = (_nspin == 1 and _npol == 1) ? 2.0 : 1.0;
      auto dm_total = nda::array<ComplexType, 3>::zeros(_nkpts, _nbnd, _nbnd);
      for (int ik = 0; ik < _nkpts; ++ik) {
         for (int is = 0; is < _nspin; ++is) {
          dm_total(ik, all, all) += D_skij(is, ik, all, all) * spin_factor;
        }
      }
      return dm_total;
    }

    nda::array<ComplexType, 6> hf_gradient_t::eval_2bdm(const nda::MemoryArrayOfRank<4> auto &D_skij)
    {
      auto tbdm_sspqrs = nda::array<ComplexType, 6>::zeros({_nspin, _nspin, _nbnd, _nbnd, _nbnd, _nbnd});
      for (int is1 = 0; is1 < _nspin; ++is1) {
        for (int is2 = 0; is2 < _nspin; ++is2) {
          for (int ik = 0; ik < _nkpts; ++ik) {
            for (int p = 0; p < _nbnd; ++p) {
              for (int q = 0; q < _nbnd; ++q) {
                for (int r = 0; r < _nbnd; ++r) {
                  for (int s = 0; s < _nbnd; ++s) {
                    tbdm_sspqrs(is1, is2, p, q, r, s) += D_skij(is1, ik, p, q) * D_skij(is2, ik, r, s) * _k_weight(ik);
                    if (is1 == is2) {
                      tbdm_sspqrs(is1, is2, p, q, r, s) -= D_skij(is1, ik, p, s) * D_skij(is2, ik, r, q) * _k_weight(ik);
                    }
                  }
                }
              }
            }
          }
        }
      }
      return tbdm_sspqrs;
    }

    ComplexType hf_gradient_t::eval_grad_2e(int iatom, int direction,
                                            const nda::MemoryArrayOfRank<4> auto &D_skij,
                                            Cholesky_ERI auto && chol)
    {
      auto D_total_kij = eval_dm_total(D_skij);
      ComplexType tmp_grad(0.0, 0.0);
      tmp_grad += eval_grad_coulomb(iatom, direction, D_total_kij, chol);
      tmp_grad += eval_grad_exchange(iatom, direction, D_skij, chol);
      return tmp_grad;
    }

    ComplexType hf_gradient_t::eval_grad_coulomb(int iatom, int direction,
                                                 const nda::MemoryArrayOfRank<3> auto &D_total_kij,
                                                 Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      _Timer.start("COULOMB");

      ComplexType tmp_grad(0.0, 0.0);

      // coulomb
      //   \sum D_{ji} D_{lk} [ d/dX (ij|kl) ]
      // = \sum D_{ji} D_{lk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D^{*}_{ij} ) (dV_{Qkl} D_{kl} )^{*} +
      //   \sum (dV_{Qij} D^{*}_{ij} ) ( V_{Qkl} D_{kl} )^{*}
      // TO-DO: check and confirm complex conjugate and transpose

      for (int ik = 0; ik < _nkpts; ++ik) {
        auto tmp = nda::array<ComplexType, 2>::zeros({1, 1});
        auto V = chol.V(0, 0, ik);
        auto dV = chol.dV(0, iatom, direction, 0, ik);

        auto dm_total_conj = nda::make_regular(nda::conj(D_total_kij(ik, all, all)));
        auto dm_total_conj_ij_1 = nda::reshape(dm_total_conj, std::array<int, 2>({_nbnd * _nbnd, 1}));
        auto dm_total_kl_1 = nda::reshape(D_total_kij(ik, all, all), std::array<int, 2>({_nbnd * _nbnd, 1}));

        {
          auto V_Pij_P_ij = nda::reshape(V, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(V_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto dV_Qkl_Q_kl = nda::reshape(dV, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(dV_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        {
          auto dV_Pij_P_ij = nda::reshape(dV, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_P_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(dV_Pij_P_ij, dm_total_conj_ij_1, tmp_P_1);
          auto V_Qkl_Q_kl = nda::reshape(V, std::array<int, 2>({_nbnd_aux, _nbnd * _nbnd}));
          auto tmp_Q_1 = nda::array<ComplexType, 2>::zeros({_nbnd_aux, 1});
          nda::blas::gemm(V_Qkl_Q_kl, dm_total_kl_1, tmp_Q_1);
          nda::blas::gemm(1, nda::conj(nda::transpose(tmp_Q_1)), tmp_P_1, 1, tmp);
        }

        tmp_grad += tmp(0, 0) * 0.5 * _k_weight(ik);
      }

      _Timer.stop("COULOMB");

      return tmp_grad;
    }

    ComplexType hf_gradient_t::eval_grad_exchange(int iatom, int direction,
                                                  const nda::MemoryArrayOfRank<4> auto &D_skij,
                                                  Cholesky_ERI auto && chol)
    {
      decltype(nda::range::all) all;

      _Timer.start("EXCHANGE");

      ComplexType tmp_grad(0.0, 0.0);

      // exchange
      //   \sum D_{li} D_{jk} [ d/dX (ij|kl) ]
      // = \sum D_{li} D_{jk} ( V_{Qij} dV^{*}_{Qlk} + dV_{Qij} V^{*}_{Qlk} )
      // = \sum ( V_{Qij} D_{jk} ) (dV_{Qlk} D^{*}_{il} )^{*} +
      //   \sum (dV_{Qij} D_{jk} ) ( V_{Qlk} D^{*}_{il} )^{*}
      // TO-DO: check and confirm conjugate and transpose

      for (int is = 0; is < _nspin; ++is) {
        for (int ik = 0; ik < _nkpts; ++ik) {
          RealType spin_factor = (_nspin == 1 and _npol == 1) ? 1.0 : 0.5;
          auto V = chol.V(0, 0, ik);
          auto dV = chol.dV(0, iatom, direction, 0, ik);
          auto dm_conj = nda::make_regular(nda::conj(D_skij(is, ik, all, all)));

          {
            auto V_Pij_Pi_j = nda::reshape(V, std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
            auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
            nda::blas::gemm(V_Pij_Pi_j, D_skij(is, ik, all, all), tmp_Pik_Pi_k);
            auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({_nbnd_aux, _nbnd, _nbnd});
            for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
              nda::blas::gemm(dm_conj, dV(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
              tmp_Qik_Q_i_k(iQ, all, all) = nda::make_regular(nda::conj(tmp_Qik_Q_i_k(iQ, all, all)));
            }
            auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
            tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * _k_weight(ik);
          }

          {
            auto dV_Pij_Pi_j = nda::reshape(dV, std::array<int, 2>({_nbnd_aux * _nbnd, _nbnd}));
            auto tmp_Pik_Pi_k = nda::array<ComplexType, 2>::zeros({_nbnd_aux * _nbnd, _nbnd});
            nda::blas::gemm(dV_Pij_Pi_j, D_skij(is, ik, all, all), tmp_Pik_Pi_k);
            auto tmp_Qik_Q_i_k = nda::array<ComplexType, 3>::zeros({_nbnd_aux, _nbnd, _nbnd});
            for (int iQ = 0; iQ < _nbnd_aux; ++iQ) {
              nda::blas::gemm(dm_conj, V(iQ, all, all), tmp_Qik_Q_i_k(iQ, all, all));
              tmp_Qik_Q_i_k(iQ, all, all) = nda::make_regular(nda::conj(tmp_Qik_Q_i_k(iQ, all, all)));
            }
            auto tmp_Pik_P_i_k = nda::reshape(tmp_Pik_Pi_k, std::array<int, 3>({_nbnd_aux, _nbnd, _nbnd}));
            tmp_grad -= nda::sum(tmp_Pik_P_i_k * tmp_Qik_Q_i_k) * spin_factor * _k_weight(ik);
          }
        }
      }

      _Timer.stop("EXCHANGE");

      return tmp_grad;
    }

    using Arr2D = nda::array<ComplexType, 2>;
    using Arr3D = nda::array<ComplexType, 3>;
    using Arr4D = nda::array<ComplexType, 4>;
    using Arr5D = nda::array<ComplexType, 5>;
    using Arr6D = nda::array<ComplexType, 6>;
    using Arrv3D = nda::array_view<ComplexType, 3>;
    using Arrv4D = nda::array_view<ComplexType, 4>;

    template Arr2D hf_gradient_t::evaluate(const Arr4D&, chol_reader_t&);
    template Arr2D hf_gradient_t::evaluate(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_gradient_t::eval_grad_2e(const Arr4D&, chol_reader_t&);
    template Arr2D hf_gradient_t::eval_grad_2e(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_gradient_t::eval_grad_coulomb(const Arr4D&, chol_reader_t&);
    template Arr2D hf_gradient_t::eval_grad_coulomb(const Arrv4D&, chol_reader_t&);

    template Arr2D hf_gradient_t::eval_grad_exchange(const Arr4D&, chol_reader_t&);
    template Arr2D hf_gradient_t::eval_grad_exchange(const Arrv4D&, chol_reader_t&);

    template Arr3D hf_gradient_t::eval_dm_total(const Arr4D&);
    template Arr3D hf_gradient_t::eval_dm_total(const Arrv4D&);

    template Arr6D hf_gradient_t::eval_2bdm(const Arr4D&);
    template Arr6D hf_gradient_t::eval_2bdm(const Arrv4D&);

    template ComplexType hf_gradient_t::eval_grad_2e(int, int, const Arr4D&, chol_reader_t&);
    template ComplexType hf_gradient_t::eval_grad_2e(int, int, const Arrv4D&, chol_reader_t&);

    template ComplexType hf_gradient_t::eval_grad_coulomb(int, int, const Arr3D&, chol_reader_t&);
    template ComplexType hf_gradient_t::eval_grad_coulomb(int, int, const Arrv3D&, chol_reader_t&);

    template ComplexType hf_gradient_t::eval_grad_exchange(int, int, const Arr4D&, chol_reader_t&);
    template ComplexType hf_gradient_t::eval_grad_exchange(int, int, const Arrv4D&, chol_reader_t&);

  } // namespace solvers

} // namespace methods
