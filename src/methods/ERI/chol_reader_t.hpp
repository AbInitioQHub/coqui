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


#ifndef COQUI_CHOLESKY_READER_HPP
#define COQUI_CHOLESKY_READER_HPP

#include <string>

#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"

#include "utilities/Timer.hpp"

#include "methods/ERI/cholesky.h"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/eri_storage_e.hpp"
#include "mean_field/MF.hpp"

namespace methods {
  namespace mpi3 = boost::mpi3;

  enum chol_reading_type_e {
      single_kpair, each_q
  };

  enum chol_writing_type_e {
      multi_file, single_file
  };

  // TODO shared_ptr for MF and mpi_context_t?

  /**
   * Read-only interface for cholesky-type/GDF-type ERIs: V^{K(ik)-Q(iq), K(ik)}
   * The reading pattern is determined at runtime.
   * Still in design stages...
   * Usage:
   *   chol_reader_t chol(mpi, eri_path, std::addressof(MF));
   *   auto V = chol.V(iq, is, ik); // By default, we read V_Qij for a given (iq, ik, is)-pair
   *
   *   // When memory is large enough, one can read V_Qij for all ik with a given iq at once
   *   // by changing the read_type to chol_reading_type::each_q
   *   chol.set_read_type(chol_reading_type::each_q);
   *   auto V = cho.V(iq, is, ik);
   */
  class chol_reader_t {
    template<nda::MemoryArray local_Array_t>
    using dArray_t = math::nda::distributed_array<local_Array_t,mpi3::communicator>;

  public:
    chol_reader_t(std::shared_ptr<mf::MF> MF, ptree const& pt):
      _MF(std::move(MF)), _mpi(_MF->mpi()),
      _eri_dir( io::get_value_with_default<std::string>(pt,"path","./") ),
      _eri_filename( io::get_value_with_default<std::string>(pt,"output","chol_info.h5") ), 
      _storage((_eri_dir=="")? incore : outcore),
      _read_type( io::tolower_copy(io::get_value_with_default<std::string>(pt,"read_type","all")) == "all" ? each_q : single_kpair ), 
      _write_type( io::tolower_copy(io::get_value_with_default<std::string>(pt,"write_type","multi")) == "multi" ? multi_file : single_file ), 
      _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()),
      _nkpts(_MF->nkpts()), _Np(0), _nbnd(_MF->nbnd()), _naux(_MF->nbnd_aux()),
      _tol( io::get_value_with_default<double>(pt,"tol",0.0001) ), 
      _chol_builder_opt(cholesky(_MF.get(), *_mpi, pt) ),
      _Timer() {

      utils::check(_storage != incore, "chol_rader_t: incore version is not implemented yet!");

      for( auto& v: {"BUILD", "READ"}) {
        _Timer.add(v);
      }

      build_and_init();
    }

    // read from existing CD/GDF integrals
    chol_reader_t(std::shared_ptr<mf::MF> MF,
                  std::string eri_dir = "./",
                  std::string eri_filename = "chol_info.h5",
                  chol_reading_type_e read_type = each_q,
                  chol_writing_type_e write_type = multi_file):
        _MF(std::move(MF)), _mpi(_MF->mpi()),
        _eri_dir(eri_dir), 
        _eri_filename(eri_filename), 
        _storage((eri_dir=="")? incore : outcore),
        _read_type(read_type), _write_type(write_type), 
        _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()),
        _nkpts(_MF->nkpts()), _Np(0), _nbnd(_MF->nbnd()), _naux(_MF->nbnd_aux()),
        _tol(-1.0), _Timer() {

      utils::check(_storage != incore, "chol_rader_t: incore version is not implemented yet!");
      if (!std::filesystem::exists(_eri_dir+"/"+_eri_filename))
        utils::check(false, "chol_reader_t: Cholesky ERIs not found!");

      for( auto& v: {"BUILD", "READ"}) {
        _Timer.add(v);
      }

      init();
    }

    // adds meta-data to h5 group
    static void add_meta_data(h5::group& grp, int nc, double tol, int ns,
           int ns_in_basis, int nk, int nb,
           nda::ArrayOfRank<2> auto const& kp,
           nda::ArrayOfRank<2> auto const& qp,
           nda::ArrayOfRank<2> auto const& qk)
    {
      h5::h5_write(grp, "Np", nc);
      h5::h5_write(grp, "tol", tol);
      h5::h5_write(grp, "nspin", ns);
      h5::h5_write(grp, "nspin_in_basis", ns_in_basis);
      h5::h5_write(grp, "nkpts", nk);
      h5::h5_write(grp, "nbnd", nb);
      h5::h5_write(grp, "nbnd_aux", 0);
      nda::h5_write(grp, "kpts", kp, false);
      nda::h5_write(grp, "qpts", qp, false);
      nda::h5_write(grp, "qk_to_kmq", qk, false);
    }

  private:
    void build_and_init() {
      _Timer.start("BUILD");
      if (!_chol_builder_opt)
        utils::check(false, "chol_reader_t: chol_builder is not initialized!");
      long max_Np = -1;
      for (size_t iq = 0; iq < _MF->nqpts(); ++iq) {
        auto dL_Qskij = _chol_builder_opt.value().evaluate<HOST_MEMORY>(iq);
        auto L_loc = dL_Qskij.local();
        L_loc *= std::sqrt(_MF->nkpts());
        std::string filename = _eri_dir + "/" + (_write_type==multi_file       ?
                                                "Vq"+std::to_string(iq)+".h5" :
                                                _eri_filename);
        write_Vq(dL_Qskij, filename, iq);
        max_Np = std::max(max_Np, dL_Qskij.global_shape()[0]);
      }
      _mpi->comm.barrier();
      if (_mpi->comm.root()) write_meta_data(_eri_dir, _eri_filename, (int)max_Np, _MF.get(), _tol);
      _Np = max_Np;
      _chol_builder_opt.reset();
      _mpi->comm.barrier();

      std::string rtype = (_read_type == single_kpair)? "single_kpair" : "each_q";
      app_log(1, "*******************************");
      app_log(1, " Cholesky ERI Reader: ");
      app_log(1, "*******************************");
      app_log(1, "    - Np max  = {}", _Np);
      app_log(1, "    - accuracy = {}", _tol);
      app_log(1, "    - read mode = {}", rtype);
      if (_storage == eri_storage_e::incore) {
        app_log(1, "    - eri storage: {}\n", eriform_enum_to_string(_storage));
      } else {
        app_log(1, "    - eri storage: {}", eriform_enum_to_string(_storage));
        app_log(1, "    - ERI dir = {}", _eri_dir);
        app_log(1, "    - ERI output = {}\n", _eri_filename);
      }
      _Timer.stop("BUILD");
    }

    void init() {
      _Np = read_Np();
      std::string filename = _eri_dir+"/"+_eri_filename;
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction"); 
      h5::h5_read(sgrp, "tol", _tol);

      std::string rtype = (_read_type == single_kpair)? "single_kpair" : "each_q";
      app_log(1, "*******************************");
      app_log(1, " Cholesky ERI Reader: ");
      app_log(1, "*******************************");
      app_log(1, "    - Np max  = {}", _Np);
      app_log(1, "    - accuracy = {}", _tol);
      app_log(1, "    - read mode = {}", rtype);
      app_log(1, "    - eri storage: {}", eriform_enum_to_string(_storage));
      app_log(1, "    - ERI dir = {}", _eri_dir);
      app_log(1, "    - ERI output = {}\n", _eri_filename);
    }

    /**
     * Read V^{K(ik), K(ik)-Q(iq)}
     * @param iq
     * @param ik
     */
    void read_V(size_t iq, size_t is, size_t ik) {
      decltype(nda::range::all) all;
      utils::check(_read_type == single_kpair, "Error: read_V() can only be called in \"single_kpair\" read mode");
      if (!_V_Qij) {
        _V_Qij.emplace(nda::array<ComplexType, 3>(_Np, _nbnd, _nbnd));
      } else {
        _V_Qij.value()() = 0.0;
      }

      //int Np;
      std::string dataset = "Vq" + std::to_string(iq);
      std::string filename = _eri_dir + "/" + (_write_type==multi_file       ?
                                              "Vq"+std::to_string(iq)+".h5" :
                                              _eri_filename);
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction");
      //h5::h5_read(sgrp, "Np", Np);
      auto l = h5::array_interface::get_dataset_info(sgrp, dataset); 
      int Np = l.lengths[0];
      auto Np_range = nda::range(0,Np);
      auto Vqk = _V_Qij.value()(Np_range, nda::ellipsis{});
      nda::h5_read(sgrp, dataset, Vqk, 
        std::tuple{all, std::min(is,size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
    }

    /**
     * Read V^{K(ik), K(ik)-Q(iq)} for all ik
     * @param iq
     */
    void read_Vq(size_t iq, size_t is) {
      utils::check(_read_type == each_q, "Error: read_Vq() can only be called in \"each_q\" read mode");
      if (!_Vq_kQij) {
        _Vq_kQij.emplace(nda::array<ComplexType, 4>(_nkpts, _Np, _nbnd, _nbnd));
      } else {
        _Vq_kQij.value()() = 0.0;
      }
      std::string filename = _eri_dir + "/" + (_write_type==multi_file       ?
                                              "Vq"+std::to_string(iq)+".h5" :
                                              _eri_filename);
      std::string dataset;
      //int Np;

      decltype(nda::range::all) all;
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      h5::group sgrp = grp.open_group("Interaction");
      //h5::h5_read(sgrp, "Np", Np);
      auto l = h5::array_interface::get_dataset_info(sgrp, "Vq" + std::to_string(iq));
      int Np = l.lengths[0];
      auto Np_range = nda::range(0,Np);
      for (size_t ik = 0; ik < _nkpts; ++ik) {
        auto Vqk = _Vq_kQij.value()(ik, Np_range, all, all);
        nda::h5_read(sgrp, "Vq" + std::to_string(iq), Vqk, 
             std::tuple{all, std::min(is,size_t(_ns_in_basis-1)), ik, nda::range(_nbnd), nda::range(_nbnd)});
      }
    }

  public:
    int read_Np(long iq = -1) const {
      int Np;
      if( iq == -1) { // read from meta_data
        std::string filename = _eri_dir+"/"+_eri_filename;
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);
        auto sgrp = grp.open_group("Interaction");
        h5::h5_read(sgrp, "Np", Np);
      } else { // read from dataset
        std::string filename = _eri_dir + "/" + (_write_type==multi_file       ?
                                                "Vq"+std::to_string(iq)+".h5" :
                                                _eri_filename);
        h5::file file = h5::file(filename, 'r');
        h5::group grp(file);
        auto sgrp = grp.open_group("Interaction");
        std::string dataset = "Vq" + std::to_string(iq); 
        auto l = h5::array_interface::get_dataset_info(sgrp, dataset); 
        Np = l.lengths[0];
      }
      return Np;
    }

    /**
     * Return the three index Coulomb tensor V^{k1,k2} at k1 = K(ik), k2 = K(ik)- Q(iq)
     */
    auto V(size_t iq, size_t is, size_t ik) {
      utils::check( is < _ns, "Error in chol_reader_r::read_V: is out of bounds: is:{}",is);
// MAM: should spin also be cached in _V_Qij and _Vq_kQij???
      if (_read_type == single_kpair) {
        if (_Vq_kQij) _Vq_kQij = std::nullopt;
        if (!_V_Qij) _V_Qij.emplace(nda::array<ComplexType, 3>(_Np, _nbnd, _nbnd));
        if (_ik == ik and _iq == iq and _is == std::min(is,size_t(_ns_in_basis-1))) {
          return _V_Qij.value()();
        } else {
          // read a new V_Qij for k1 = K(ik) - Q(iq), k2 = K(ik)
          _Timer.start("READ");
          read_V(iq, is, ik);
          _Timer.stop("READ");
          _is = std::min(is,size_t(_ns_in_basis-1));
          _ik = ik;
          _iq = iq;
          return _V_Qij.value()();
        }
      } else { // _read_type == each_q
        if (_V_Qij) _V_Qij = std::nullopt;
        if (!_Vq_kQij) _Vq_kQij.emplace(nda::array<ComplexType, 4>(_nkpts, _Np, _nbnd, _nbnd));
        if (_iq == iq and _is == std::min(is,size_t(_ns_in_basis-1))) {
          return _Vq_kQij.value()(ik, nda::range::all, nda::range::all, nda::range::all);
        } else {
          // read a new Vq_kQij for Q(iq)
          _Timer.start("READ");
          read_Vq(iq, is);
          _Timer.stop("READ");
          _is = std::min(is,size_t(_ns_in_basis-1));
          _iq = iq;
          return _Vq_kQij.value()(ik, nda::range::all, nda::range::all, nda::range::all);
        }
      }
    }

    auto V_kmq_k(size_t iq, size_t is, size_t ik, bool conj=false) {
      decltype(nda::range::all) all;
      utils::check(is < _ns, "Error in chol_reader_r::read_V: is out of bounds: is:{}",is);
      nda::array<ComplexType, 3> V_kmq_k(_Np, _nbnd, _nbnd);
      auto V_k_kmq = V(iq, is, ik);
      for (size_t Q = 0; Q < _Np; ++Q) {
        if (!conj) {
          V_kmq_k(Q, all, all) = nda::make_regular(nda::transpose(V_k_kmq(Q, all, all)));
        } else {
          auto V_T = nda::transpose(V_k_kmq(Q, all, all));
          V_kmq_k(Q, all, all) = nda::make_regular(nda::conj(V_T));
        }
      }
      return V_kmq_k;
    }

    auto& mpi() const { return _mpi; }
    auto& MF() const { return _MF; }
    int nspin() const { return _ns; }
    int nkpts() const { return _nkpts; }
    int Np() const { return _Np; }
    int nbnd() const { return _nbnd; }
    int nbnd_aux() const { return _naux; }
    chol_reading_type_e& set_read_type() { return _read_type; }
    chol_reading_type_e chol_read_type() const { return _read_type; }
    chol_writing_type_e chol_write_type() const { return _write_type; }

    /**
     * Checks if the output file can be used to initializing an object
     * @param path       - [INPUT] Directory to store ERIs
     * @param output     - [INPUT] ERI h5 file
     * @param nq         - [INPUT] Number of q-points
     * @param write_type - [INPUT] Write type: multi_file or single_file
     * @return - Whether the Cholesky ERIs are initialized already.
     */
    static bool check_init(std::string path, std::string output, int nq,
                           chol_writing_type_e write_type)
    {
      if(!std::filesystem::exists(path+"/"+output)) return false;
      std::string filename = path+"/"+output;
      h5::file file = h5::file(filename, 'r');
      h5::group grp(file);
      if( not grp.has_subgroup("Interaction") ) return false;
      h5::group sgrp = grp.open_group("Interaction");
      if( not( sgrp.has_key("tol") and sgrp.has_key("Np") ) ) return false;
      if(write_type == single_file) {
        for (size_t iq = 0; iq < nq; ++iq) 
          if( not sgrp.has_dataset("Vq"+std::to_string(iq)) ) return false;
      } else {
        for (size_t iq = 0; iq < nq; ++iq) {
          std::string Vq_file = path+"/Vq"+std::to_string(iq)+".h5";
          if(!std::filesystem::exists(Vq_file)) return false;
          h5::file f = h5::file(Vq_file, 'r');
          h5::group g(f);
          if( not g.has_subgroup("Interaction") ) return false;
          h5::group sg = g.open_group("Interaction");
          if( not sg.has_dataset("Vq"+std::to_string(iq)) ) return false; 
        }
      }
      return true;
    }   

    /* a temporary interface between methods::cholesky and methods::chol_reader_t */
    static void write_meta_data(std::string outdir, std::string eri_filename, int Np, mf::MF *mf, double tol) {
      std::string filename = outdir + "/" + eri_filename;
      h5::file file(filename, 'a');
      h5::group grp(file);

      h5::group sgrp = (grp.has_subgroup("Interaction") ? 
                        grp.open_group("Interaction")   :
                        grp.create_group("Interaction",true));

      add_meta_data(sgrp, Np, tol, mf->nspin(), mf->nspin_in_basis(), mf->nkpts(),
                    mf->nbnd(), mf->kpts(), mf->Qpts(), mf->qk_to_k2());           
    }

    /**
     * Write Cholesky ERIs into one h5 file for a given q-point, one dataset for each (q, k) pair.
     * @param Vq_Qskij - [INPUT] Cholesky ERI: L(Nchol, ns, nkpts, nbnd, nbnd)
     * @param filename - [INPUT] name of h5 file where data will be written 
     * @param iq       - [INPUT] q-point index
     */
    template<nda::MemoryArray Array_t>
    static void write_Vq(Array_t &Vq_Qskij, std::string filename, [[maybe_unused]] int iq) {
      static_assert(nda::get_rank<Array_t> == 5, "chol_reader_t::write_Vq: incorrect rank of Vq_Qskij.");
//      int Np = Vq_Qskij.shape(0);
      int ns = Vq_Qskij.shape(1);
      int nkpts = Vq_Qskij.shape(2);

      std::string dataset;
      h5::file file(filename, 'a');
      h5::group grp(file);
      h5::group sgrp = (grp.has_subgroup("Interaction") ?
                        grp.open_group("Interaction")   :
                        grp.create_group("Interaction",true));
//      h5::h5_write(sgrp, "Np", Np);
      for (int isk = 0; isk < ns*nkpts; ++isk) {
        int is = isk / nkpts;
        int ik = isk % nkpts;
        auto Vqsk = Vq_Qskij(nda::range::all, is, ik, nda::range::all, nda::range::all);
        dataset = "Vq" + std::to_string(isk);
        nda::h5_write(sgrp, dataset, Vqsk, false);
      }
    }

    /**
     * Write Cholesky ERIs into one single h5 file and one dataset for a given q-point
     * @param Vq_Qskij - [INPUT] Cholesky ERI in distributed nda array: L(Nchol, ns, nkpts, nbnd, nbnd)
     * @param filename - [INPUT] name of h5 file where data will be written 
     * @param iq       - [INPUT] q-point index
     */
    template<nda::MemoryArray local_Array_t>
    static void write_Vq(const dArray_t<local_Array_t> &Vq_Qskij, std::string filename, int iq) {
      static_assert(nda::get_rank<local_Array_t> == 5, "chol_reader::write_Vq: incorrect rank of Vq_Qskij.");

//      int Np = Vq_Qskij.global_shape()[0];
      std::string dataset = "Vq" + std::to_string(iq);
      app_log(2, "Writing distributed Vq at iq = {} to {}", iq, filename);
      nda::array<ComplexType, 5> Vq_buffer;
      if (Vq_Qskij.communicator()->root()) {
        Vq_buffer = nda::array<ComplexType, 5>(Vq_Qskij.global_shape());
        math::nda::gather(0, Vq_Qskij, &Vq_buffer);

        h5::file file(filename, 'a');
        h5::group grp(file);
        h5::group sgrp = (grp.has_subgroup("Interaction") ?
                          grp.open_group("Interaction")   :
                          grp.create_group("Interaction",true));
//        h5::h5_write(sgrp, "Np", Np);
        nda::h5_write(sgrp, dataset, Vq_buffer, false);
      } else {
        math::nda::gather(0, Vq_Qskij, &Vq_buffer);
      }
      Vq_Qskij.communicator()->barrier();

      // Disable distributed hdf5 write temporarily. It's too slow!
      /*if (Vq_Qskij.communicator()->root()) {
        h5::file file(filename, 'a');
        h5::group grp(file);
        h5::group sgrp = (grp.has_subgroup("Interaction") ?
                          grp.open_group("Interaction")   :
                          grp.create_group("Interaction",true));
        h5::h5_write(sgrp, "Np", Np);
        math::nda::h5_write(sgrp, dataset, Vq_Qskij, false);
      } else {
        h5::group grp;
        h5::group sgrp = (grp.has_subgroup("Interaction") ?
                          grp.open_group("Interaction")   :
                          grp.create_group("Interaction",true));
        math::nda::h5_write(sgrp, dataset, Vq_Qskij, false);
      }*/
    }


    inline void print_timers() {
      app_log(2, "\n***************************************************");
      app_log(2, "                  CHOl-ERI timers ");
      app_log(2, "***************************************************");
      app_log(2, "    BUILD:                {} sec", _Timer.elapsed("BUILD"));
      app_log(2, "    READ:                 {} sec", _Timer.elapsed("READ"));
      app_log(2, "***************************************************\n");
    }

  private:
    std::shared_ptr<mf::MF> _MF;
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi;
    // folder to store eri
    std::string _eri_dir;
    std::string _eri_filename;
    // eri storage type: incore or outcore
    eri_storage_e _storage;
    chol_reading_type_e _read_type;
    chol_writing_type_e _write_type;

    int _ns = 0;
    int _ns_in_basis = 0;
    int _nkpts = 0;
    int _Np = 0;
    int _nbnd = 0;
    int _naux = 0;
    // tolerance of Cholesky-ERI
    double _tol = 0.0;
    std::optional<cholesky> _chol_builder_opt;

    int _is = -1;
    int _ik = -1;
    int _iq = -1;

    std::optional<nda::array<ComplexType, 3> > _V_Qij;
    std::optional<nda::array<ComplexType, 4> > _Vq_kQij;
    mutable utils::TimerManager _Timer;
  };

} // methods

#endif //COQUI_CHOLESKY_READER_HPP
