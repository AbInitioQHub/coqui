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


#include <algorithm>
#include <optional>

#include "numerics/nda_functions.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/embedding/projector_boson_t.h"

namespace {

  constexpr long max_product_block_size = 64;
  constexpr size_t max_projector_tile_bytes = size_t{32} * 1024 * 1024;

  // Keep enough independent blocks to occupy the MPI ranks when possible,
  // while bounding both the product-basis dimension and the T tile bytes.
  long product_block_size(long Np, long n_task_groups, long mpi_size,
                          size_t elements_per_product) {
    n_task_groups = std::max(1L, n_task_groups);
    auto blocks_per_group = std::max(1L, (mpi_size + n_task_groups - 1) / n_task_groups);
    auto balanced_block_size = std::max(1L, Np / blocks_per_group);
    auto max_tile_elements = std::max<size_t>(1, max_projector_tile_bytes / sizeof(ComplexType));
    auto memory_block_size = std::max<long>(
        1, static_cast<long>(max_tile_elements / std::max<size_t>(1, elements_per_product)));
    return std::min({max_product_block_size, balanced_block_size, memory_block_size});
  }

} // anonymous namespace

namespace methods {

  auto projector_boson_t::calc_bosonic_projector(THC_ERI auto &thc) const
  -> sArray_t<Array_view_5D_t> {
    auto mpi = thc.mpi();
    auto C_skIai = _proj_fermi.C_skIai();
    auto W_rng = _proj_fermi.W_rng();
    auto nqpts = _MF->nqpts();
    auto [ns, nkpts, nImps, nImpOrbs, nOrbs_W] = C_skIai.shape();
    long NP = thc.Np();

    auto sB_qIPab = math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {nqpts, nImps, NP, nImpOrbs, nImpOrbs});
    auto B_qIPab = sB_qIPab.local();
    sB_qIPab.win().fence();

    // T is independent of q in the no-symmetry path. Distribute disjoint
    // (impurity, product-basis block) pairs across ranks, then reuse each tile
    // for every q. Thus every B(q,I,P,:,:) has exactly one writer before the
    // existing all-reduce, without replicating the full T tensor on each rank.
    size_t elements_per_product = static_cast<size_t>(ns) * nkpts * nImpOrbs;
    auto P_block_size = product_block_size(
        NP, nImps, mpi->comm.size(), elements_per_product);
    auto nP_blocks = (NP + P_block_size - 1) / P_block_size;
    nda::array<ComplexType, 4> T_skPa(ns, nkpts, P_block_size, nImpOrbs);
    for (long iIP = mpi->comm.rank(); iIP < nImps*nP_blocks; iIP += mpi->comm.size()) {
      long I = iIP / nP_blocks;
      long iP_block = iIP % nP_blocks;
      long P0 = iP_block * P_block_size;
      long P1 = std::min(P0 + P_block_size, NP);
      auto P_rng = nda::range(P0, P1);
      auto p_rng = nda::range(0, P1-P0);

      for (long isk = 0; isk < ns*nkpts; ++isk) {
        long is = isk / nkpts;
        long ik = isk % nkpts;
        nda::blas::gemm(thc.X(is, 0, ik)(P_rng, W_rng[I]),
                        nda::dagger(C_skIai(is, ik, I, nda::ellipsis{})),
                        T_skPa(is, ik, p_rng, nda::range::all));
      }

      for (long iq = 0; iq < nqpts; ++iq) {
        for (long isk = 0; isk < ns*nkpts; ++isk) {
          long is = isk / nkpts;
          long ik = isk % nkpts;
          long ikmq = _MF->qk_to_k2(iq, ik);
          for (long p = 0; p < P1-P0; ++p)
            nda::blas::gerc(ComplexType(1.0), T_skPa(is, ikmq, p, nda::range::all),
                            T_skPa(is, ik, p, nda::range::all),
                            B_qIPab(iq, I, P0+p, nda::ellipsis{}));
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

  auto projector_boson_t::calc_bosonic_projector_symm(THC_ERI auto &thc) const
  -> sArray_t<Array_view_5D_t> {
    auto mpi = thc.mpi();
    auto C_skIai = _proj_fermi.C_skIai();
    auto W_rng = _proj_fermi.W_rng();
    auto [ns, nkpts, nImps, nImpOrbs, nOrbs_W] = C_skIai.shape();
    auto nqpts = _MF->nqpts();
    auto qsymms = _MF->qsymms();
    long NP = thc.Np();

    auto sB_qIPab = math::shm::make_shared_array<Array_view_5D_t>(
        *mpi, {nqpts, nImps, NP, nImpOrbs, nImpOrbs});
    auto B_qIPab = sB_qIPab.local();
    sB_qIPab.win().fence();
    // intermediate objects
    size_t elements_per_product = static_cast<size_t>(ns) * nkpts * nImpOrbs;
    auto P_block_size = product_block_size(
        NP, nqpts*nImps, mpi->comm.size(), elements_per_product);
    auto nP_blocks = (NP + P_block_size - 1) / P_block_size;
    nda::array<ComplexType, 4> T_skPb(ns, nkpts, P_block_size, nImpOrbs);
    nda::array<ComplexType, 2> Cfull_jb(_MF->nbnd(), nImpOrbs);
    nda::array<ComplexType, 2> tmp_ib(_MF->nbnd(), nImpOrbs);
    std::optional<nda::array<ComplexType, 4>> Crot_skib;

    using math::sparse::T;
    using math::sparse::csrmm;
    sB_qIPab.win().fence();
    // T is q-dependent. Give every (q,I) at least one worker, adding only the
    // minimum extra workers needed for MPI occupancy. Each worker owns a
    // disjoint subset of P blocks and reuses one cached symmetry rotation.
    long nqI = nqpts*nImps;
    long max_worker_tasks = nqI*nP_blocks;
    long nworker_tasks = std::min(max_worker_tasks, std::max(nqI, long(mpi->comm.size())));
    long workers_per_qI = nworker_tasks / nqI;
    long qI_with_extra_worker = nworker_tasks % nqI;
    for (long qI = 0; qI < nqI; ++qI) {
      long iq = qI / nImps;
      long I = qI % nImps;
      long nworkers = workers_per_qI + (qI < qI_with_extra_worker ? 1 : 0);
      long task0 = qI*workers_per_qI + std::min(qI, qI_with_extra_worker);

      // symmetry index
      auto sym_it = std::find(qsymms.begin(), qsymms.end(), _MF->qp_symm(iq));
      auto isym = std::distance(qsymms.begin(), sym_it);

      for (long iworker = 0; iworker < nworkers; ++iworker) {
        long task = task0 + iworker;
        if (task % mpi->comm.size() != mpi->comm.rank()) continue;

        if (isym != 0) {
          if (not Crot_skib)
            Crot_skib.emplace(ns, nkpts, _MF->nbnd(), nImpOrbs);
          for (long isk = 0; isk < ns*nkpts; ++isk) {
            long is = isk / nkpts;
            long ik = isk % nkpts;
            auto [cjg, D_ij] = _MF->symmetry_rotation(isym, ik);
            Cfull_jb() = 0.0;
            if (not cjg) {
              Cfull_jb(W_rng[I], nda::range::all) =
                  nda::conj(nda::transpose(C_skIai(is, ik, I, nda::ellipsis{})));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
            } else {
              Cfull_jb(W_rng[I], nda::range::all) =
                  nda::transpose(C_skIai(is, ik, I, nda::ellipsis{}));
              csrmm(ComplexType(1.0), *D_ij, Cfull_jb, ComplexType(0.0), tmp_ib);
              tmp_ib = nda::conj(tmp_ib);
            }
            Crot_skib.value()(is, ik, nda::ellipsis{}) = tmp_ib;
          }
        }

        for (long iP_block = iworker; iP_block < nP_blocks; iP_block += nworkers) {
          long P0 = iP_block * P_block_size;
          long P1 = std::min(P0 + P_block_size, NP);
          auto P_rng = nda::range(P0, P1);
          auto p_rng = nda::range(0, P1-P0);

          for (long isk = 0; isk < ns*nkpts; ++isk) {
            long is = isk / nkpts;
            long ik = isk % nkpts;
            auto ikR = _MF->ks_to_k(isym, ik);
            if (isym == 0) {
              nda::blas::gemm(thc.X(is, 0, ik)(P_rng, W_rng[I]),
                              nda::dagger(C_skIai(is, ik, I, nda::ellipsis{})),
                              T_skPb(is, ik, p_rng, nda::range::all));
            } else {
              nda::blas::gemm(thc.X(is, 0, ikR)(P_rng, nda::range::all),
                              Crot_skib.value()(is, ik, nda::ellipsis{}),
                              T_skPb(is, ik, p_rng, nda::range::all));
            }
          }

          for (long isk = 0; isk < ns*nkpts; ++isk) {
            long is = isk / nkpts;
            long ik = isk % nkpts;
            long ikmq = _MF->qk_to_k2(iq, ik);
            for (long p = 0; p < P1-P0; ++p)
              nda::blas::gerc(ComplexType(1.0), T_skPb(is, ikmq, p, nda::range::all),
                              T_skPb(is, ik, p, nda::range::all),
                              B_qIPab(iq, I, P0+p, nda::ellipsis{}));
          }
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
