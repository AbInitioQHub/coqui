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


#include "numerics/nda_functions.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/embedding/projector_boson_t.h"

namespace methods {

  auto projector_boson_t::calc_bosonic_projector(THC_ERI auto &thc) const
  -> sArray_t<Array_view_5D_t> {
    auto mpi = thc.mpi();
    auto C_skIai = _proj_fermi.C_skIai();
    auto W_rng = _proj_fermi.W_rng();
    auto nqpts = _MF->nqpts();
    auto [ns, nkpts, nImps, nImpOrbs, nOrbs_W] = C_skIai.shape();

    nda::array<ComplexType, 5> T_skIPa(ns, nkpts, nImps, thc.Np(), nImpOrbs);
    for (long isk = 0; isk < ns*nkpts; ++isk) {
      long is = isk / nkpts; // isk = is * nkpts + ik
      long ik = isk % nkpts;
      for (long I = 0; I < nImps; ++I) {
        nda::blas::gemm(thc.X(is, 0, ik)(nda::range::all, W_rng[I]), nda::dagger(C_skIai(is, ik, I, nda::ellipsis{})),
                        T_skIPa(is, ik, I, nda::ellipsis{}));
      }
    }

    auto sB_qIPab = math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {nqpts, nImps, thc.Np(), nImpOrbs, nImpOrbs});
    auto B_qIPab = sB_qIPab.local();
    sB_qIPab.win().fence();
    for (size_t iqIP = mpi->comm.rank(); iqIP < nkpts*nImps*thc.Np(); iqIP += mpi->comm.size()) {
      size_t iq = iqIP / (nImps * thc.Np()); // iqIP = iq * nImps * Np + I * Np + p
      size_t I  = (iqIP / thc.Np()) % nImps;
      size_t P  = iqIP % thc.Np();
      for (long isk = 0; isk < ns*nkpts; ++isk) {
        long is = isk / nkpts;
        long ik = isk % nkpts;
        long ikmq = _MF->qk_to_k2(iq, ik);
        nda::blas::gerc(ComplexType(1.0),T_skIPa(is, ikmq, I, P, nda::range::all),
                        T_skIPa(is, ik, I, P, nda::range::all), B_qIPab(iq, I, P, nda::ellipsis{}));
      }
    }
    sB_qIPab.win().fence();
    sB_qIPab.all_reduce();
    if (mpi->node_comm.root())
      B_qIPab() /= nqpts;
    mpi->node_comm.barrier();

    return sB_qIPab;
  }

  auto projector_boson_t::calc_bosonic_projector_symm(THC_ERI auto &thc) const
  -> sArray_t<Array_view_5D_t> {
    auto mpi = thc.mpi();
    auto C_skIai = _proj_fermi.C_skIai();
    auto W_rng = _proj_fermi.W_rng();
    auto [ns, nkpts, nImps, nImpOrbs, nOrbs_W] = C_skIai.shape();
    auto nqpts = _MF->nqpts();
    auto qsymms = _MF->qsymms();

    auto sB_qIPab = math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {nqpts, nImps, thc.Np(), nImpOrbs, nImpOrbs});
    auto B_qIPab = sB_qIPab.local();
    sB_qIPab.win().fence();
    // intermediate objects
    nda::array<ComplexType, 5> T_skIPb(ns, nkpts, nImps, thc.Np(), nImpOrbs);
    nda::array<ComplexType, 2> Cfull_jb(_MF->nbnd(), nImpOrbs);
    nda::array<ComplexType, 2> tmp_ib(_MF->nbnd(), nImpOrbs);

    using math::sparse::T;
    using math::sparse::csrmm;
    sB_qIPab.win().fence();
    for (size_t iq=mpi->comm.rank(); iq<nqpts; iq+=mpi->comm.size()) {
      // symmetry index
      auto sym_it = std::find(qsymms.begin(), qsymms.end(), _MF->qp_symm(iq));
      auto isym = std::distance(qsymms.begin(), sym_it);
      // calculate T_skIPb
      for (size_t isk=0; isk<ns*nkpts; ++isk) {
        size_t is = isk / nkpts;
        size_t ik = isk % nkpts;
        auto ikR = _MF->ks_to_k(isym, ik);
        for (size_t I=0; I<nImps; ++I) {
          if (isym==0) {
            // TskI_Pb = conj(Ck_bj) * X_Pj(k)
            nda::blas::gemm(thc.X(is, 0, ik)(nda::range::all, W_rng[I]),
                            nda::dagger(C_skIai(is, ik, I, nda::ellipsis{})),
                            T_skIPb(is, ik, I, nda::ellipsis{}));
          } else {
            auto [cjg, D_ij] = _MF->symmetry_rotation(isym, ik);
            // D_ij * Cfull_jb = tmp_ib
            Cfull_jb() = 0.0;
            if(not cjg) {
              Cfull_jb(W_rng[I], nda::range::all) = nda::conj(nda::transpose(C_skIai(is, ik, I, nda::ellipsis{})));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
            } else {
              Cfull_jb(W_rng[I], nda::range::all) = nda::transpose(C_skIai(is, ik, I, nda::ellipsis{}));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
              tmp_ib = nda::conj(tmp_ib);
            }
            // X_Pi * tmp_ib = TskI_Pb
            nda::blas::gemm(thc.X(is, 0, ikR), tmp_ib, T_skIPb(is,ik,I,nda::ellipsis{}));
          }
        }
      }

      for (long I = 0; I < nImps; ++I) {
        for (long isk = 0; isk < ns*nkpts; ++isk) {
          long is = isk / nkpts;
          long ik = isk % nkpts;
          long ikmq = _MF->qk_to_k2(iq, ik);

          for (long P = 0; P < thc.Np(); ++P)
            nda::blas::gerc(ComplexType(1.0),T_skIPb(is, ikmq, I, P, nda::range::all),
                            T_skIPb(is, ik, I, P, nda::range::all), B_qIPab(iq, I, P,nda::ellipsis{}));
        }
      }
    }
    sB_qIPab.win().fence();
    sB_qIPab.all_reduce();
    if (mpi->node_comm.root())
      B_qIPab() /= nqpts;
    mpi->node_comm.barrier();

    return sB_qIPab;
  }

} // methods

// instantiation of "public" templates
namespace methods {

  template<nda::Array Array_base_t>
  using sArray_t = math::shm::shared_array<Array_base_t>;
  using Array_view_5D_t = nda::array_view<ComplexType, 5>;

  template sArray_t<Array_view_5D_t> projector_boson_t::calc_bosonic_projector(thc_reader_t &thc) const;
  template sArray_t<Array_view_5D_t> projector_boson_t::calc_bosonic_projector_symm(thc_reader_t &thc) const;

} //