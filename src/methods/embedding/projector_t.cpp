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


#include "methods/SCF/scf_common.hpp"
#include "numerics/nda_functions.hpp"
#include "methods/embedding/projector_t.h"

namespace methods {

  void projector_t::print_metadata() {
    app_log(1, "  Projector Information");
    app_log(1, "  ---------------------");
    app_log(1, "  Number of impurities                      = {}", _nImps);
    app_log(1, "  Number of local orbitals per impurity     = {}", _nImpOrbs);
    app_log(1, "  Range of primary orbitals for local basis = [{}, {})\n",
            _W_rng[0].first(), _W_rng[0].last());
  }

  template<nda::ArrayOfRank<4> Array_base_t, nda::ArrayOfRank<4> Oloc_t>
  void projector_t::upfold(sArray_t<Array_base_t> &O_skij, const Oloc_t &Oloc_sIab) const {

    utils::check(O_skij.shape()[0] == Oloc_sIab.shape(0) and
                 O_skij.shape()[0] == _C_skIai.shape(0), "embed_t::upfold: ns mismatches.{}, {}, {}",
                 O_skij.shape()[0], Oloc_sIab.shape(0), _C_skIai.shape(0));
    utils::check(Oloc_sIab.shape(1) == _nImps, "embed_t::upfold: nImps mismatches. {}, {}", Oloc_sIab.shape(1), _nImps);
    utils::check(Oloc_sIab.shape(2) == _nImpOrbs, "embed_t::upfold: nImpOrbs mismatches. {}, {}", Oloc_sIab.shape(2), _nImpOrbs);
    utils::check(O_skij.shape()[1]==_MF->nkpts_ibz(), "embed_t::upfold: O_skij.shape[1]({})!=nkpts_ibz({}).",
                 O_skij.shape()[1], _MF->nkpts_ibz());

    O_skij.set_zero();
    auto [ns, nkpts_ibz, nbnd, nbnd_b] = O_skij.shape();

    nda::array<ComplexType, 2> buffer_ib(_nOrbs_W, _nImpOrbs);

    auto O_buffer = sArray_t<Array_base_t>(
        O_skij.communicator(), O_skij.internode_comm(), O_skij.node_comm(),
        {ns, nkpts_ibz, _nOrbs_W, _nOrbs_W});

    auto O_buf_loc = O_buffer.local();
    int rank = O_buffer.communicator()->rank();
    int size = O_buffer.communicator()->size();
    for (long imp = 0; imp < _nImps; ++imp) {
      O_buffer.set_zero();
      O_buffer.win().fence();
      for (long sk = rank; sk < ns * nkpts_ibz; sk += size) {
        long is = sk / nkpts_ibz;
        long ik = sk % nkpts_ibz;

        nda::blas::gemm(ComplexType(1.0),
                        nda::dagger(_C_skIai(is,ik,imp,nda::ellipsis{})),
                        Oloc_sIab(is,imp,nda::ellipsis{}),
                        ComplexType(0.0),
                        buffer_ib);
        nda::blas::gemm(buffer_ib, _C_skIai(is,ik,imp,nda::ellipsis{}), O_buf_loc(is,ik,nda::ellipsis{}));
      }
      O_buffer.win().fence();
      O_buffer.all_reduce();
      if (O_skij.node_comm()->root()) {
        O_skij.local()(nda::range::all, nda::range::all, _W_rng[imp], _W_rng[imp]) += O_buf_loc;
      }
    }
    O_skij.communicator()->barrier();
  }

  template<nda::ArrayOfRank<5> Array_base_t, nda::ArrayOfRank<5> Oloc_t>
  void projector_t::upfold(sArray_t<Array_base_t> &O_tskij, const Oloc_t &Oloc_tsIab) const {

    utils::check(O_tskij.shape()[0] == Oloc_tsIab.shape(0), "embed_t::upfold: nts mismatches. {}, {}",
                 O_tskij.shape()[0], Oloc_tsIab.shape(0));
    utils::check(O_tskij.shape()[1] == Oloc_tsIab.shape(1) and
                 O_tskij.shape()[1] == _C_skIai.shape(0), "embed_t::upfold: ns mismatches.{}, {}, {}",
                 O_tskij.shape()[1], Oloc_tsIab.shape(1), _C_skIai.shape(0));
    utils::check(Oloc_tsIab.shape(2) == _nImps, "embed_t::upfold: nImps mismatches. {}, {}", Oloc_tsIab.shape(2), _nImps);
    utils::check(Oloc_tsIab.shape(3) == _nImpOrbs, "embed_t::upfold: nImpOrbs mismatches. {}, {}", Oloc_tsIab.shape(3), _nImpOrbs);
    utils::check(O_tskij.shape()[2]==_MF->nkpts_ibz(), "embed_t::upfold: O_tskij.shape[2]({})!=nkpts_ibz({}).",
                 O_tskij.shape()[2], _MF->nkpts_ibz());

    O_tskij.set_zero();
    auto [nts, ns, nkpts_ibz, nbnd, nbnd_b] = O_tskij.shape();

    nda::array<ComplexType, 2> buffer_ib(_nOrbs_W, _nImpOrbs);

    auto O_buffer = sArray_t<Array_base_t>(
        O_tskij.communicator(), O_tskij.internode_comm(), O_tskij.node_comm(),
        {nts, ns, nkpts_ibz, _nOrbs_W, _nOrbs_W});

    auto O_buf = O_buffer.local();
    int rank = O_buffer.communicator()->rank();
    int size = O_buffer.communicator()->size();
    for (long imp = 0; imp < _nImps; ++imp) {
      O_buffer.set_zero();
      O_buffer.win().fence();
      for (long tsk = rank; tsk < nts*ns*nkpts_ibz; tsk += size) {
        long it = tsk / (ns*nkpts_ibz); // tsk = it * ns*nkpts_ibz + is * nkpts_ibz + ik
        long is = (tsk / nkpts_ibz) % ns;
        long ik = tsk % nkpts_ibz;

        nda::blas::gemm(ComplexType(1.0),
                        nda::dagger(_C_skIai(is,ik,imp,nda::ellipsis{})),
                        Oloc_tsIab(it,is,imp,nda::ellipsis{}),
                        ComplexType(0.0),
                        buffer_ib);

        nda::blas::gemm(buffer_ib, _C_skIai(is,ik,imp,nda::ellipsis{}), O_buf(it,is,ik,nda::ellipsis{}));
      }
      O_buffer.win().fence();
      O_buffer.all_reduce();

      O_tskij.win().fence();
      if (O_tskij.node_comm()->root()) {
        O_tskij.local()(nda::range::all, nda::range::all, nda::range::all, _W_rng[imp], _W_rng[imp]) += O_buf;
      }
      O_tskij.win().fence();
    }
  }

  template<nda::ArrayOfRank<5> Array_base_t, nda::ArrayOfRank<6> Ac_t>
  void projector_t::upfold(sArray_t<Array_base_t> &O_tskij, const Ac_t &O_tskIab) const {

    utils::check(O_tskij.shape()[0] == O_tskIab.shape(0), "embed_t::upfold: nts mismatches. {}, {}",
                 O_tskij.shape()[0], O_tskIab.shape(0));
    utils::check(O_tskij.shape()[1] == O_tskIab.shape(1) and
                 O_tskij.shape()[1] == _C_skIai.shape(0), "embed_t::upfold: ns mismatches.{}, {}, {}",
                 O_tskij.shape()[1], O_tskIab.shape(1), _C_skIai.shape(0));
    utils::check(O_tskIab.shape(3) == _nImps, "embed_t::upfold: nImps mismatches. {}, {}", O_tskIab.shape(3), _nImps);
    utils::check(O_tskIab.shape(4) == _nImpOrbs, "embed_t::upfold: nImpOrbs mismatches. {}, {}", O_tskIab.shape(4),
                 _nImpOrbs);

    O_tskij.set_zero();
    auto [nts, ns, nkpts, nbnd, nbnd_b] = O_tskij.shape();

    nda::array<ComplexType, 2> buffer_ib(_nOrbs_W, _nImpOrbs);

    auto O_buffer = sArray_t<Array_base_t>(
        O_tskij.communicator(), O_tskij.internode_comm(), O_tskij.node_comm(),
        {nts, ns, nkpts, _nOrbs_W, _nOrbs_W});

    auto O_buf = O_buffer.local();
    int rank = O_buffer.communicator()->rank();
    int size = O_buffer.communicator()->size();
    for (long imp = 0; imp < _nImps; ++imp) {
      O_buffer.set_zero();
      O_buffer.win().fence();
      for (long tsk = rank; tsk < nts * ns * nkpts; tsk += size) {
        long it = tsk / (ns * nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
        long is = (tsk / nkpts) % ns;
        long ik = tsk % nkpts;

        nda::blas::gemm(ComplexType(1.0), nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        O_tskIab(it, is, ik, imp, nda::ellipsis{}),
                        ComplexType(0.0), buffer_ib);

        nda::blas::gemm(buffer_ib, _C_skIai(is,ik,imp,nda::ellipsis{}), O_buf(it,is,ik,nda::ellipsis{}));
      }
      O_buffer.win().fence();
      O_buffer.all_reduce();

      O_tskij.win().fence();
      if (O_tskij.node_comm()->root()) {
        O_tskij.local()(nda::range::all, nda::range::all, nda::range::all, _W_rng[imp], _W_rng[imp]) += O_buf;
      }
      O_tskij.win().fence();
    }
  }

  template<typename comm_t>
  auto projector_t::downfold_k(const nda::MemoryArrayOfRank<3> auto &O_ski, comm_t comm) const
  -> nda::array<ComplexType, 5> {

    utils::check(O_ski.shape()[0] == _C_skIai.shape(0), "embed_t::downfold_k: ns mismatches.{}, {}",
                 O_ski.shape()[0], _C_skIai.shape(0));

    auto [ns, nkpts, nbnd] = O_ski.shape();
    nda::array<ComplexType, 5> O_skIab(ns, nkpts, _nImps, _nImpOrbs, _nImpOrbs);
    O_skIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(_nImpOrbs, _nOrbs_W);
    nda::array<ComplexType, 2> O_ij_W(_nOrbs_W, _nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long sk = rank; sk < ns * nkpts; sk += size) {
      long is = sk / nkpts; // sk = is * nkpts + ik
      long ik = sk % nkpts;
      for (long imp = 0; imp < _nImps; ++imp) {
        O_ij_W = nda::diag(O_ski(is, ik, _W_rng[imp]));
        auto C_ai = _C_skIai(is, ik, imp, nda::ellipsis{});
        nda::blas::gemm(C_ai, O_ij_W, buffer_aj);

        nda::blas::gemm(buffer_aj, nda::dagger(_C_skIai(is,ik,imp,nda::ellipsis{})), O_skIab(is, ik, imp, nda::ellipsis{}));
      }
    }
    comm.all_reduce_in_place_n(O_skIab.data(), O_skIab.size(), std::plus<>{});
    return O_skIab;
  }

  template<nda::ArrayOfRank<3> Array_base_t>
  auto projector_t::downfold_k(const sArray_t<Array_base_t> &O_ski) const -> nda::array<ComplexType, 5> {
    auto O_loc = O_ski.local();
    return downfold_k(O_loc, *O_ski.communicator());
  }

  template<typename comm_t>
  auto projector_t::downfold_k(const nda::MemoryArrayOfRank<4> auto &O_skij, comm_t comm) const
  -> nda::array<ComplexType, 5> {

    utils::check(O_skij.shape()[0] == _C_skIai.shape(0), "embed_t::downfold_k: ns mismatches.{}, {}",
                 O_skij.shape()[0], _C_skIai.shape(0));

    auto [ns, nkpts, nbnd, nbnd_b] = O_skij.shape();

    nda::array<ComplexType, 5> O_skIab(ns, nkpts, _nImps, _nImpOrbs, _nImpOrbs);
    O_skIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(_nImpOrbs, _nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long sk = rank; sk < ns * nkpts; sk += size) {
      long is = sk / nkpts; // sk = is * nkpts + ik
      long ik = sk % nkpts;
      for (long imp = 0; imp < _nImps; ++imp) {
        nda::blas::gemm(_C_skIai(is,ik,imp,nda::ellipsis{}),
                        O_skij(is,ik,_W_rng[imp],_W_rng[imp]), buffer_aj);

        nda::blas::gemm(buffer_aj, nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        O_skIab(is, ik, imp, nda::ellipsis{}));
      }
    }
    for (size_t shift=0; shift<O_skIab.size(); shift+=size_t(1e9)) {
      ComplexType* start = O_skIab.data()+shift;
      size_t count = (shift+size_t(1e9) < O_skIab.size())? size_t(1e9) : O_skIab.size()-shift;
      comm.all_reduce_in_place_n(start, count, std::plus<>{});
    }
    return O_skIab;
  }

  template<nda::ArrayOfRank<4> Array_base_t>
  auto projector_t::downfold_k(const sArray_t<Array_base_t> &O_skij) const -> nda::array<ComplexType, 5> {
    auto O_loc = O_skij.local();
    return downfold_k(O_loc, *O_skij.communicator());
  }

  template<typename comm_t>
  auto projector_t::downfold_k(const nda::MemoryArrayOfRank<5> auto &O_tskij, comm_t comm) const
  -> nda::array<ComplexType, 6> {

    utils::check(O_tskij.shape()[1] == _C_skIai.shape(0), "embed_t::downfold_k: ns mismatches.{}, {}",
                 O_tskij.shape()[1], _C_skIai.shape(0));

    auto [nts, ns, nkpts, nbnd, nbnd_b] = O_tskij.shape();

    nda::array<ComplexType, 6> O_tskIab(nts, ns, nkpts, _nImps, _nImpOrbs, _nImpOrbs);
    O_tskIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(_nImpOrbs, _nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long tsk = rank; tsk < nts*ns*nkpts; tsk += size) {
      long it = tsk / (ns*nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
      long is = (tsk / nkpts) % ns;
      long ik = tsk % nkpts;
      for (long imp = 0; imp < _nImps; ++imp) {
        nda::blas::gemm(_C_skIai(is, ik, imp, nda::ellipsis{}), O_tskij(it,is,ik,_W_rng[imp],_W_rng[imp]), buffer_aj);

        nda::blas::gemm(ComplexType(1.0), buffer_aj, nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        ComplexType(0.0), O_tskIab(it, is, ik, imp, nda::ellipsis{}));
      }
    }
    for (size_t shift=0; shift<O_tskIab.size(); shift+=size_t(1e9)) {
      ComplexType* start = O_tskIab.data()+shift;
      size_t count = (shift+size_t(1e9) < O_tskIab.size())? size_t(1e9) : O_tskIab.size()-shift;
      comm.all_reduce_in_place_n(start, count, std::plus<>{});
    }
    return O_tskIab;
  }

  template<nda::ArrayOfRank<5> Array_base_t>
  auto projector_t::downfold_k(const sArray_t<Array_base_t> &O_tskij) const -> nda::array<ComplexType, 6> {
    auto O_loc = O_tskij.local();
    return downfold_k(O_loc, *O_tskij.communicator());
  }

  template<nda::ArrayOfRank<5> Array_base_t>
  auto projector_t::downfold_k_fbz(const sArray_t<Array_base_t> &O_tskij) const -> nda::array<ComplexType, 6> {

    if (O_tskij.shape()[2] == _MF->nkpts())
      return downfold_k(O_tskij);

    utils::check(O_tskij.shape()[1] == _C_skIai.shape(0), "downfold_k_fbz: ns mismatches.{}, {}",
                 O_tskij.shape()[1], _C_skIai.shape(0));
    utils::check(O_tskij.shape()[2] == _MF->nkpts_ibz(), "downfold_k_fbz: O_tskij.shape[2]({})!=nkpts_ibz({}).",
                 O_tskij.shape()[2], _MF->nkpts_ibz());

    auto [nts, ns, nkpts_ibz, nbnd, nbnd_b] = O_tskij.shape();
    auto nkpts = _MF->nkpts();
    auto kp_trev = _MF->kp_trev();

    nda::array<ComplexType, 6> O_tskIab(nts, ns, nkpts, _nImps, _nImpOrbs, _nImpOrbs);
    O_tskIab() = 0.0;
    nda::array<ComplexType, 2> O_ij_W(_nOrbs_W, _nOrbs_W);
    nda::array<ComplexType, 2> buffer_aj(_nImpOrbs, _nOrbs_W);

    auto O_tskij_loc = O_tskij.local();
    int rank = O_tskij.communicator()->rank();
    int size = O_tskij.communicator()->size();
    for (long tsk = rank; tsk < nts*ns*nkpts; tsk += size) {
      long it = tsk / (ns*nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
      long is = (tsk / nkpts) % ns;
      long ik = tsk % nkpts;
      auto ik_ibz = _MF->kp_to_ibz(ik);

      for (long imp = 0; imp < _nImps; ++imp) {
        if (kp_trev(ik))
          O_ij_W = nda::conj(O_tskij_loc(it, is, ik_ibz, _W_rng[imp], _W_rng[imp]));
        else
          O_ij_W = O_tskij_loc(it, is, ik_ibz, _W_rng[imp], _W_rng[imp]);

        nda::blas::gemm(_C_skIai(is, ik, imp, nda::ellipsis{}), O_ij_W, buffer_aj);

        nda::blas::gemm(ComplexType(1.0), buffer_aj, nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        ComplexType(0.0), O_tskIab(it, is, ik, imp, nda::ellipsis{}));
      }
    }
    for (size_t shift=0; shift<O_tskIab.size(); shift+=size_t(1e9)) {
      ComplexType* start = O_tskIab.data()+shift;
      size_t count = (shift+size_t(1e9) < O_tskIab.size())? size_t(1e9) : O_tskIab.size()-shift;
      O_tskij.communicator()->all_reduce_in_place_n(start, count, std::plus<>{});
    }
    return O_tskIab;
  }

  template<bool force_real, typename comm_t>
  auto projector_t::downfold_loc(const nda::MemoryArrayOfRank<4> auto &O_skij,
                                 comm_t comm, std::string name) const
  -> nda::array<ComplexType, 4> {

    utils::check(O_skij.shape()[0] == _C_skIai.shape(0), "downfold_loc: ns mismatches.{}, {}",
                 O_skij.shape()[0], _C_skIai.shape(0));
    utils::check(O_skij.shape()[1] == _MF->nkpts_ibz(), "downfold_loc: O_skij.shape[1]({})!=nkpts_ibz({}).",
                 O_skij.shape()[1], _MF->nkpts_ibz());

    auto [ns, nkpts_ibz, nbnd, nbnd_b] = O_skij.shape();
    auto nImps = _C_skIai.shape(2);
    auto nImpOrbs = _C_skIai.shape(3);
    auto nOrbs_W = _C_skIai.shape(4);
    auto nkpts = _MF->nkpts();
    auto kp_trev = _MF->kp_trev();

    nda::array<ComplexType, 4> O_sIab(ns, nImps, nImpOrbs, nImpOrbs);
    O_sIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(nImpOrbs, nOrbs_W);
    nda::array<ComplexType, 2> O_ij_W(nOrbs_W, nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long sk = rank; sk < ns*nkpts; sk += size) {
      long is = sk / nkpts; // sk = is * nkpts + ik
      long ik = sk % nkpts;
      auto ik_ibz = _MF->kp_to_ibz(ik);
      for (long imp = 0; imp < nImps; ++imp) {
        if (kp_trev(ik))
          O_ij_W = nda::conj(O_skij(is, ik_ibz, _W_rng[imp], _W_rng[imp]));
        else
          O_ij_W = O_skij(is, ik_ibz, _W_rng[imp], _W_rng[imp]);

        nda::blas::gemm(_C_skIai(is, ik, imp, nda::ellipsis{}), O_ij_W, buffer_aj);


        nda::blas::gemm(ComplexType(1.0), buffer_aj, nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        ComplexType(1.0), O_sIab(is, imp, nda::ellipsis{}));
      }
    }
    comm.all_reduce_in_place_n(O_sIab.data(), O_sIab.size(), std::plus<>{});
    O_sIab() /= nkpts;

    // check the largest imaginary element
    double max_imag = -1;
    nda::for_each(O_sIab.shape(),
                  [&O_sIab, &max_imag](auto ...i) { max_imag = std::max(max_imag, std::abs(O_sIab(i...).imag())); });
    if constexpr (force_real) {

      app_log(2, "Explicit make {} to Hermitian, and set the imaginary part to zero", name);
      app_log(2, "  -> The largest imaginary part = {}. \n", max_imag);
      hermitize_in_tau(O_sIab);
      nda::for_each(O_sIab.shape(),
                    [&O_sIab](auto... i) mutable { O_sIab(i...) = ComplexType(O_sIab(i...).real(), 0.0); });
    } else {
      if (max_imag >= 1e-10) {
        app_log(1, "[WARNING] The largest imaginary part of {} = {} > 1e-10. \n"
                   "Please make sure you know what you are doing. ", name, max_imag);
      }
    }

    return O_sIab;
  }

  template<bool force_real, nda::ArrayOfRank<4> Array_base_t>
  auto projector_t::downfold_loc(const sArray_t<Array_base_t> &O_skij, std::string name) const
  -> nda::array<ComplexType, 4> {
    return downfold_loc<force_real>(O_skij.local(), *O_skij.communicator(), name);
  }

  template<bool force_real, typename comm_t>
  auto projector_t::downfold_loc(const nda::MemoryArrayOfRank<5> auto &O_tskij, comm_t comm,
                                 std::string name) const
  -> nda::array<ComplexType, 5> {

    utils::check(O_tskij.shape()[1] == _C_skIai.shape(0), "downfold_loc: ns mismatches.{}, {}",
                 O_tskij.shape()[1], _C_skIai.shape(0));
    utils::check(O_tskij.shape()[2] == _MF->nkpts_ibz(), "downfold_loc: O_tskij.shape[2]({})!=nkpts_ibz({}).",
                 O_tskij.shape()[2], _MF->nkpts_ibz());

    auto [nts, ns, nkpts_ibz, nbnd, nbnd_b] = O_tskij.shape();
    auto nImps = _C_skIai.shape(2);
    auto nImpOrbs = _C_skIai.shape(3);
    auto nOrbs_W = _C_skIai.shape(4);
    auto nkpts = _MF->nkpts();
    auto kp_trev = _MF->kp_trev();

    nda::array<ComplexType, 5> O_tsIab(nts, ns, nImps, nImpOrbs, nImpOrbs);
    O_tsIab() = 0.0;
    nda::array<ComplexType, 2> buffer_aj(nImpOrbs, nOrbs_W);
    nda::array<ComplexType, 2> O_ij_W(nOrbs_W, nOrbs_W);

    int rank = comm.rank();
    int size = comm.size();
    for (long tsk = rank; tsk < nts*ns*nkpts; tsk += size) {
      long it = tsk / (ns*nkpts); // tsk = it * ns*nkpts + is * nkpts + ik
      long is = (tsk / nkpts) % ns;
      long ik = tsk % nkpts;
      auto ik_ibz = _MF->kp_to_ibz(ik);
      for (long imp = 0; imp < nImps; ++imp) {
        if (kp_trev(ik))
          O_ij_W = nda::conj(O_tskij(it, is, ik_ibz, _W_rng[imp], _W_rng[imp]));
        else
          O_ij_W = O_tskij(it, is, ik_ibz, _W_rng[imp], _W_rng[imp]);

        nda::blas::gemm(_C_skIai(is, ik, imp, nda::ellipsis{}), O_ij_W, buffer_aj);


        nda::blas::gemm(ComplexType(1.0), buffer_aj, nda::dagger(_C_skIai(is, ik, imp, nda::ellipsis{})),
                        ComplexType(1.0), O_tsIab(it, is, imp, nda::ellipsis{}));
      }
    }
    comm.all_reduce_in_place_n(O_tsIab.data(), O_tsIab.size(), std::plus<>{});
    O_tsIab() /= nkpts;

    // check the largest imaginary element
    double max_imag = -1;
    nda::for_each(O_tsIab.shape(),
                  [&O_tsIab, &max_imag](auto ...i) { max_imag = std::max(max_imag, std::abs(O_tsIab(i...).imag())); });
    if constexpr (force_real) {
      app_log(2, "Explicit make {} to Hermitian, and set the imaginary part to zero", name);
      app_log(2, "  -> The largest imaginary part = {}. \n", max_imag);
      hermitize_in_tau(O_tsIab);
      nda::for_each(O_tsIab.shape(),
                    [&O_tsIab](auto... i) mutable { O_tsIab(i...) = ComplexType(O_tsIab(i...).real(), 0.0); });
    } else {
      if (max_imag >= 1e-10) {
        app_log(1, "[WARNING] downfold_loc: the largest imaginary part of {} = {} >= 1e-10. \n"
                   "Please make sure you know what you are doing. ", name, max_imag);
      }
    }

    return O_tsIab;
  }

  template<bool force_real, nda::ArrayOfRank<5> Array_base_t>
  auto projector_t::downfold_loc(const sArray_t<Array_base_t> &O_tskij, std::string name) const
  -> nda::array<ComplexType, 5> {
    return downfold_loc<force_real>(O_tskij.local(), *O_tskij.communicator(), name);
  }


} // methods

// instantiation of "public" templates
namespace methods {

  template void projector_t::upfold(sArray_t<nda::array_view<ComplexType, 4>>&, const nda::array<ComplexType, 4>&) const;
  template void projector_t::upfold(sArray_t<nda::array_view<ComplexType, 5>>&, const nda::array<ComplexType, 5>&) const;
  template void projector_t::upfold(sArray_t<nda::array_view<ComplexType, 5>>&, const nda::array<ComplexType, 6>&) const;

  template nda::array<ComplexType, 6> projector_t::downfold_k_fbz(const sArray_t<nda::array_view<ComplexType, 5>>&) const;

  template nda::array<ComplexType, 6> projector_t::downfold_k(const nda::array<ComplexType, 5>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array<ComplexType, 4>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array<ComplexType, 3>&, mpi3::communicator) const;

  template nda::array<ComplexType, 6> projector_t::downfold_k(const nda::array_view<ComplexType, 5>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array_view<ComplexType, 4>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array_view<ComplexType, 3>&, mpi3::communicator) const;

  template nda::array<ComplexType, 6> projector_t::downfold_k(const nda::array_view<ComplexType, 5, nda::C_layout>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array_view<ComplexType, 4, nda::C_layout>&, mpi3::communicator) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const nda::array_view<ComplexType, 3, nda::C_layout>&, mpi3::communicator) const;

  template nda::array<ComplexType, 6> projector_t::downfold_k(const sArray_t<nda::array_view<ComplexType, 5>>&) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const sArray_t<nda::array_view<ComplexType, 4>>&) const;
  template nda::array<ComplexType, 5> projector_t::downfold_k(const sArray_t<nda::array_view<ComplexType, 3>>&) const;


  template nda::array<ComplexType, 4> projector_t::downfold_loc<true>(const nda::array<ComplexType, 4>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 4> projector_t::downfold_loc<false>(const nda::array<ComplexType, 4>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<true>(const nda::array<ComplexType, 5>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<false>(const nda::array<ComplexType, 5>&, mpi3::communicator, std::string) const;

  template nda::array<ComplexType, 4> projector_t::downfold_loc<true>(const nda::array_view<ComplexType, 4>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 4> projector_t::downfold_loc<false>(const nda::array_view<ComplexType, 4>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<true>(const nda::array_view<ComplexType, 5>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<false>(const nda::array_view<ComplexType, 5>&, mpi3::communicator, std::string) const;

  template nda::array<ComplexType, 4> projector_t::downfold_loc<true>(const nda::array_view<ComplexType, 4, nda::C_layout>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 4> projector_t::downfold_loc<false>(const nda::array_view<ComplexType, 4, nda::C_layout>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<true>(const nda::array_view<ComplexType, 5, nda::C_layout>&, mpi3::communicator, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<false>(const nda::array_view<ComplexType, 5, nda::C_layout>&, mpi3::communicator, std::string) const;

  template nda::array<ComplexType, 5> projector_t::downfold_loc<true>(const sArray_t<nda::array_view<ComplexType, 5>>&, std::string) const;
  template nda::array<ComplexType, 5> projector_t::downfold_loc<false>(const sArray_t<nda::array_view<ComplexType, 5>>&, std::string) const;
  template nda::array<ComplexType, 4> projector_t::downfold_loc<true>(const sArray_t<nda::array_view<ComplexType, 4>>&, std::string) const;
  template nda::array<ComplexType, 4> projector_t::downfold_loc<false>(const sArray_t<nda::array_view<ComplexType, 4>>&, std::string) const;

} //
