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


#ifndef COQUI_THC_READER_T_HPP
#define COQUI_THC_READER_T_HPP

#include <string>
#include <optional>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"
#include "mpi3/communicator.hpp"
#include "utilities/mpi_context.h"
#include "nda/nda.hpp"
#include "nda/h5.hpp"
#include "h5/h5.hpp"

#include "numerics/distributed_array/nda.hpp"
#include "numerics/distributed_array/h5.hpp"
#include "numerics/shared_array/nda.hpp"

#include "methods/ERI/detail/concepts.hpp"
#include "methods/ERI/eri_storage_e.hpp"
#include "methods/ERI/thc.h"
#include "methods/ERI/chol_reader_t.hpp"
#include "mean_field/MF.hpp"

namespace methods {

  namespace mpi3 = boost::mpi3;

  /**
   * Read-only interface for thc-ERIs, computed on the fly
   *   a) only supports q-independent interpolating points (i.e. q-independent X) at the moment
   *
   * Usage:
   *   thc_reader_t thc(...);
   *   auto Xk = thc.X(is, ik); // return nda::array_view
   *   auto Yk = thc.Y(is, ik); // return nda::array_view
   *   auto Zq = thc.Z(iq);
   */
  class thc_reader_t {
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using Array_t = memory::array<MEM, ComplexType, R>;
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using Array_view_t = memory::array_view<MEM, ComplexType, R>;
    template<MEMORY_SPACE MEM = HOST_MEMORY, long R = 1>
    using dArray_t = memory::darray_t<memory::array<MEM, ComplexType, R>, mpi3::communicator>;
    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    template<int N>
    using shape_t = std::array<long,N>;
  public:
    thc_reader_t(std::shared_ptr<mf::MF> MF,
                 ptree const& pt, bool get_Sinv_Ivec = false,
                 bool isdf_only = false, bool intialize = true):
      _MF(std::move(MF)), _mpi(_MF->mpi()),
      _MEM_EVAL( io::get_compute_space(pt,"compute") ),
      _storage(string_to_eri_storage_enum(io::tolower_copy(io::get_value_with_default<std::string>(pt,"storage","incore")))), 
      _eri_file( io::get_value_with_default<std::string>(pt,"save","") ), 
      _format( io::get_value_with_default<std::string>(pt,"format","bdft") ),
      _cd_dir( io::get_value_with_default<std::string>(pt,"cd_dir","") ), 
      _X_type("q_indep"),
      _thc_builder_opt(thc(_MF.get(), *_mpi, pt, false)),
      _Np( int(io::get_value_with_default<int>(pt,"nIpts",0)) ), 
      _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()),
      _nqpts(_MF->nqpts()), _nqpts_ibz(_MF->nqpts_ibz()),
      _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()), _nbnd(_MF->nbnd()), 
      _npol(_MF->npol()), _npol_in_basis(_MF->npol_in_basis()),
      _get_Sinv_Ivec(get_Sinv_Ivec),
      x_range( io::get_value_with_default<nda::range>(pt,"X_orbital_range",nda::range(_nbnd)) ), 
      y_range( io::get_value_with_default<nda::range>(pt,"Y_orbital_range",x_range) ), 
      _dZ(math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {_mpi->comm.size(), 1, 1}, {_mpi->comm.size(), 1, 1})),
      _X_shm(math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(*_mpi,{1, 1, 1, 1})),
      _Y_shm{std::nullopt},
      _dSinv_Ivec(std::nullopt),
      _Timer() {
      utils::check(x_range.first() >= 0 and x_range.last() <= _nbnd, 
                   "X orbitals out of range: ({},{}), nbnd:{}",x_range.first(),x_range.last(),_nbnd);
      utils::check(y_range.first() >= 0 and y_range.last() <= _nbnd,
                   "Y orbitals out of range: ({},{}), nbnd:{}",y_range.first(),y_range.last(),_nbnd);
      auto thresh = io::get_value_with_default<double>(pt,"thresh",-1.0);
      // Auto-resolve thresh sentinel: when nIpts is set, use 1e-13 (numerical floor); otherwise use 1e-5
      if (thresh < 0.0) {
        thresh = (_Np > 0) ? 1e-13 : 1e-5;
      }
      utils::check( _Np>0 or thresh>0.0, "Error in thc_reader_t: Must set nIpts and/or thresh");
      if(_storage == eri_storage_e::outcore and _eri_file == "") 
        _eri_file = "./thc.eri.h5";

      if (intialize) {
        if (isdf_only) {
          build_isdf_only(io::get_value_with_default<bool>(pt, "check_accuracy", false),
                          io::get_value_with_default<bool>(pt, "write_zeta_on_fft_mesh", false));
        } else {
          init(true);
        }
      }
    }

    // read existing THC integrals
    thc_reader_t(std::shared_ptr<mf::MF> MF,
                 std::string storage,
                 std::string eri_file = "",
                 bool get_Sinv_Ivec = false, bool intialize = true):
      _MF(MF), _mpi(_MF->mpi()), _MEM_EVAL( DEFAULT_MEMORY_SPACE ),
      _storage(string_to_eri_storage_enum(storage)), 
      _eri_file(eri_file), 
      _format("bdft"),
      _cd_dir(""),
      _X_type("q_indep"), 
      _thc_builder_opt{std::nullopt},
      _Np(read_Np()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()), 
      _nqpts(_MF->nqpts()), _nqpts_ibz(_MF->nqpts_ibz()),
      _ns(_MF->nspin()), _ns_in_basis(_MF->nspin_in_basis()), _nbnd(_MF->nbnd()),
      _npol(_MF->npol()),_npol_in_basis(_MF->npol_in_basis()), _get_Sinv_Ivec(get_Sinv_Ivec),
      x_range(0),  // read later
      y_range(0),  // read later
      _dZ(math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {_mpi->comm.size(), 1, 1}, {_mpi->comm.size(), 1, 1})),
      _X_shm(math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(*_mpi,{1, 1, 1, 1})),
      _Y_shm{std::nullopt},
      _dSinv_Ivec(std::nullopt),
      _Chi_head(_nqpts_ibz, _Np), _Chi_bar_head(_nqpts_ibz, _Np),
      _Timer() {
      if (intialize) init(false);
    };

    ~thc_reader_t() = default;

    thc_reader_t(thc_reader_t const&) = default;
    thc_reader_t(thc_reader_t &&) = default;
    thc_reader_t& operator=(thc_reader_t const&) = default;
    thc_reader_t& operator=(thc_reader_t &&) = default;

    void init(bool build_eri) {

      print_thc_reader_info(build_eri);

      utils::check(_X_type == "q_indep", "thc_reader_t: q-dependent Xk is not implemented yet!");

      for (auto &v: {"BUILD_TOTAL", "BUILD_THC", "BUILD_GATHER", "BUILD_WRITE",
                     "READ_X", "READ_V"})
        _Timer.add(v);

      if (build_eri) {
        utils::check(_thc_builder_opt!=std::nullopt, "thc_builder is not initialized!");
        _thc_builder_opt.value().print_metadata();
        if (_cd_dir == "")
          build();         // build THC-ERI from ISDF
        else
          build_from_CD(); // build THC-ERI from LS-THC
      } else
        read();            // read existing THC-ERI

      _initialized = true;
      app_log(1, "####### End of THC initialization routines #######\n");
    }

  private:

    void print_thc_reader_info(bool build_eri) {
      // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20ThcCoulomb
      app_log(1, "\n"
                 "╔═╗╔═╗╔═╗ ╦ ╦╦  ╔╦╗┬ ┬┌─┐╔═╗┌─┐┬ ┬┬  ┌─┐┌┬┐┌┐ \n"
                 "║  ║ ║║═╬╗║ ║║   ║ ├─┤│  ║  │ ││ ││  │ ││││├┴┐\n"
                 "╚═╝╚═╝╚═╝╚╚═╝╩   ╩ ┴ ┴└─┘╚═╝└─┘└─┘┴─┘└─┘┴ ┴└─┘\n");
      app_log(1, "  Algorithm                       = {}", (_cd_dir=="")? "ISDF" : "LS-THC");
      app_log(1, "  THC integrals access            = {}", eriform_enum_to_string(_storage));
      app_log(1, "  Found precomputed THC integrals = {}", !build_eri);
      if (build_eri) {
        if (_eri_file != "")
          app_log(1, "  --> CoQuí will compute THC integrals and save to: {}", _eri_file);
        else
          app_log(1, "  --> CoQuí will compute THC integrals.");
      } else
        app_log(1, "  --> Reading the precomputed THC integrals from file: {}", _eri_file);
      app_log(1, "");
    }

    void print_thc_summary() {
      app_log(1, "\n  Summary of THC Coulomb Integrals");
      app_log(1, "  --------------------------------");
      app_log(1, "  Number of interpolating points = {}", _Np);
      app_log(1, "  X orbital index range          = [{},{})", x_range.first(),x_range.last());
      app_log(1, "  Y orbital index range          = [{},{})\n", y_range.first(),y_range.last());
    }

    void build() {
      _Timer.start("BUILD_TOTAL");

      _Timer.start("BUILD_THC");
      {
        auto eval = [&]<MEMORY_SPACE MEM>() { 
          auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<MEM>(0, _Np, x_range, y_range);
          _rp = std::move(ri);
          _Np = _rp.size();
          _Timer.stop("BUILD_THC");

          // allocate structures with dynamic _Np
          //_Chi_head = memory::array<HOST_MEMORY, ComplexType, 2>(_nqpts_ibz, _Np);
          //_Chi_bar_head = memory::array<HOST_MEMORY, ComplexType, 2>(_nqpts_ibz, _Np);

          _Timer.start("BUILD_THC");
          auto [_dZ_d, _Chi_head_d, _Chi_bar_head_d, _dSinv_Ivec_d] = _thc_builder_opt.value().evaluate<MEM>(_rp,dXa,dXb,_get_Sinv_Ivec,x_range,y_range);
          utils::check( _get_Sinv_Ivec == _dSinv_Ivec_d.has_value(), "Error: Inconsistent optional return value.");
          _Timer.stop("BUILD_THC");

          // copy to host memory if needed, otherwise just move
          _dZ = std::move(_dZ_d);
          _Chi_head = std::move(_Chi_head_d);
          _Chi_bar_head = std::move(_Chi_bar_head_d());
          _dSinv_Ivec = std::move(_dSinv_Ivec_d);

          // gather dPa to _X_shm
          _Timer.start("BUILD_GATHER");
          gather_X_shm(dXa);
          if(dXb.has_value())
            gather_Y_shm(dXb.value());
          else
            utils::check(x_range == y_range, "thc_reader::build: x_range != y_range with missing dXb value.");
          _Timer.stop("BUILD_GATHER");
        };

        if(_MEM_EVAL == HOST_MEMORY)
          eval.operator()<HOST_MEMORY>();
#if defined(ENABLE_DEVICE)
        else if(_MEM_EVAL == DEVICE_MEMORY)
          eval.operator()<DEVICE_MEMORY>();
        else if(_MEM_EVAL == UNIFIED_MEMORY)
          eval.operator()<UNIFIED_MEMORY>();
#endif
      }

      // scale by nkpts
      auto Z_loc = _dZ.local();
      Z_loc *= _nkpts;

      // save if requested
      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          // MAM: write thc meta-data to into a "metadata" dataset. Useful for external codes/afqmc
          if (_format == "bdft") {
            std::vector<int> arng = {x_range.first(),x_range.last()};
            std::vector<int> brng = {y_range.first(),y_range.last()};
            h5::h5_write(grp, "Np", (int)_Np);
            h5::h5_write(grp, "number_of_bands", (int)_nbnd);
            h5::h5_write(grp, "X_orbital_range", arng);
            h5::h5_write(grp, "Y_orbital_range", brng);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            auto X_0 = _X_shm.local(); 
            nda::h5_write(grp, "collocation_matrix", X_0, false);
            if(_Y_shm.has_value()) {
              auto Y_0 = _Y_shm.value().local(); 
              nda::h5_write(grp, "Y_collocation_matrix", Y_0, false);
            }
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");

      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "  THC-READER::BUILD()");
      app_log(2, "  -------------------");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute thc-eri:             {0:.3f} sec", _Timer.elapsed("BUILD_THC"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write eri:                   {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, " ");

      print_thc_summary();
    }

    void build_from_CD() {
      using math::nda::make_distributed_array;
      utils::check(x_range==y_range, "thc_reader::build_from_CD: x_range!=y_range needs testing. Disabling for now.");
      _Timer.start("BUILD_TOTAL");

      _Timer.start("BUILD_THC");
      auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<HOST_MEMORY>(0, _Np, x_range, y_range);
      _rp = std::move(ri);
      _Np = _rp.size();
      _Timer.stop("BUILD_THC");
     
      utils::check( not _get_Sinv_Ivec, "Finish: SinvIvec not yet written to file. Finish!!!");

      // allocate structures with dynamic _Np
      _Chi_head = nda::array<ComplexType, 2>(_nqpts_ibz, _Np);
      _Chi_bar_head = nda::array<ComplexType, 2>(_nqpts_ibz, _Np);

      _Timer.start("BUILD_THC");
      /*** Read Cholesky ERIs and fit them to THC solver ***/
      auto chol_reader = chol_reader_t(_MF, _cd_dir, "chol_info.h5", single_kpair);
      long nchol_max = chol_reader.Np();
      // MAM: need chol_reader_t with different x/y ranges!

      long np = _mpi->comm.size();
      long nkpools = utils::find_proc_grid_max_npools(np, _nkpts, 0.2);
      np /= nkpools;
      long np_i = utils::find_proc_grid_min_diff(np,1,1);
      long np_j = np / np_i;

      for (size_t q = 0; q < _nqpts_ibz; ++q) {
        // read Cholesky ERIs; nchol might be different at different q-points
        long nchol_q = chol_reader.read_Np(q);
        auto chol_rng = nda::range(0, nchol_q);

        // MAM: might be a problem if I don't propagate change to cholesky code
        auto dbuffer = make_distributed_array<Array_t<HOST_MEMORY,5>>(
            _mpi->comm, {1, 1, nkpools, np_i, np_j}, {nchol_q, _ns_in_basis*_npol_in_basis, _nkpts, x_range.size(), y_range.size()});
        auto s_rng = dbuffer.local_range(1);
        auto k_rng = dbuffer.local_range(2);
        auto i_rng = dbuffer.local_range(3);
        auto j_rng = dbuffer.local_range(4);

        auto buffer_loc = dbuffer.local();
        if(_nkpts != 1) {
          for (auto [is,s] : itertools::enumerate(s_rng)) {
            for (auto [ik, k]: itertools::enumerate(k_rng)) {
              // Lqk_Pij = (nchol_max, nbnd, nbnd)
              auto Lqk_Pij = chol_reader.V(q, s, k);
              buffer_loc(nda::range::all, is, ik, nda::ellipsis{}) = Lqk_Pij(chol_rng, i_rng, j_rng);
            }
          }
        }
        else { // molecular case
          auto sLqk_Pij = math::shm::make_shared_array<nda::array<ComplexType, 3> >(_mpi->comm, 
                             _mpi->internode_comm, _mpi->node_comm,
                             std::array<long int, 3>{nchol_max, x_range.size(), y_range.size()});
          for (auto [is,s] : itertools::enumerate(s_rng)) {
            sLqk_Pij.win().fence();
            if (_mpi->node_comm.root()) {
              sLqk_Pij.local() = chol_reader.V(0, s, 0);
            }
            sLqk_Pij.win().fence();
            buffer_loc(nda::range::all, is, 0, nda::ellipsis{}) = sLqk_Pij.local()(chol_rng, i_rng, j_rng);
          }
        }
        // solve the least-square problem
        // TODO CNY: we need to evaluate _Chi_head and _Chi_bar_head for finite-size corrections
        // TODO CNY: proper warning on the missing _Chi_head and _Chi_bar_head
        auto dZq_uv = _thc_builder_opt.value().evaluate<HOST_MEMORY>(q, _rp, dbuffer);
        auto Zq_loc = dZq_uv.local();
        if (q == 0) {
          auto pgrid = dZq_uv.grid();
          auto block_size = dZq_uv.block_size();
          auto gshape = dZq_uv.global_shape();
          // choose distribution of _dZ based on dZq_uv at q = 0
          _dZ = make_distributed_array<Array_t<HOST_MEMORY,3>>(
              _mpi->comm, {1, pgrid[0], pgrid[1]}, {_nqpts_ibz, gshape[0], gshape[1]},
              {1, block_size[0], block_size[1]});
        }
        auto Z_loc = _dZ.local();
        Z_loc(q, nda::ellipsis{}) = Zq_loc;
      }
      _Chi_head() = 0.0;
      _Chi_bar_head() = 0.0;
      _Timer.stop("BUILD_THC");

      // gather dPa to _X_shm
      _Timer.start("BUILD_GATHER");
      gather_X_shm(dXa);
      if(dXb.has_value()) 
        gather_Y_shm(dXb.value());
      else
        utils::check(x_range == y_range, "thc_reader::build: x_range != y_range with missing dXb value.");
      _Timer.stop("BUILD_GATHER");

      // save if requested
      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          if (_format == "bdft") {
            std::vector<int> arng = {x_range.first(),x_range.last()};
            std::vector<int> brng = {y_range.first(),y_range.last()};
            h5::h5_write(grp, "Np", (int)_Np);
            h5::h5_write(grp, "number_of_bands", (int)_nbnd);
            h5::h5_write(grp, "X_orbital_range", arng);
            h5::h5_write(grp, "Y_orbital_range", brng);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            auto X_0 = _X_shm.local();
            nda::h5_write(grp, "collocation_matrix", X_0, false);
            if(_Y_shm.has_value()) {
              auto Y_0 = _Y_shm.value().local();
              nda::h5_write(grp, "Y_collocation_matrix", Y_0, false);
            }
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, _dZ, _Chi_head, _Chi_bar_head);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");
      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "\n  THC-READER::BUILD_FROM_CD()");
      app_log(2, "  ---------------------------");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute thc-eri:             {0:.3f} sec", _Timer.elapsed("BUILD_THC"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write eri:                   {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, " ");

      print_thc_summary();
    }

    void build_isdf_only(bool check_accuracy=true, bool write_zeta_on_fft_mesh=false) {
      _Timer.start("BUILD_TOTAL");

      _Timer.start("BUILD_ISDF");
      auto [ri,dXa,dXb] = _thc_builder_opt.value().interpolating_points<HOST_MEMORY>(0, _Np);
      _rp = std::move(ri);
      _Np = _rp.size();
      _Timer.stop("BUILD_ISDF");

      app_log(1, "*******************************");
      app_log(1, " ISDF-only builder: ");
      app_log(1, "*******************************");
      app_log(1, "    - Np       = {}", _Np);
      app_log(1, "    - h5 chkpt file = {}", _eri_file);

      _Timer.start("BUILD_ISDF");
      auto dzeta_qur = _thc_builder_opt.value().evaluate_isdf_only<HOST_MEMORY>(_rp,dXa,dXb);
      _Timer.stop("BUILD_ISDF");

      _Timer.start("BUILD_GATHER");
      gather_X_shm(dXa);
      _Timer.stop("BUILD_GATHER");

      _Timer.start("ISDF_CHECK");
      if (check_accuracy) isdf_check(dzeta_qur);
      _Timer.stop("ISDF_CHECK");

      if (_eri_file != "") {
        _Timer.start("BUILD_WRITE");
        if (_mpi->comm.root()) {
          h5::file file(_eri_file, 'w');
          h5::group grp(file);
          if (_format == "bdft") {
            h5::h5_write(grp, "Np", (int)_Np);
            nda::h5_write(grp, "kpts", _MF->kpts(), false);
            nda::h5_write(grp, "qpts", _MF->Qpts(), false);
            h5::h5_write(grp, "nkpts_ibz", _nkpts_ibz);
            h5::h5_write(grp, "nqpts_ibz", _nqpts_ibz);
            nda::h5_write(grp, "collocation_matrix", _X_shm.local(), false);
            _thc_builder_opt.value().save(grp, _format, _rp, dzeta_qur, write_zeta_on_fft_mesh);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        } else {
          h5::group grp;
          if(_format == "bdft" ) {
            _thc_builder_opt.value().save(grp, _format, _rp, dzeta_qur, write_zeta_on_fft_mesh);
          } else {
            APP_ABORT("thc: Unknown file format: {}", _format);
          }
        }
        _mpi->comm.barrier();
        _Timer.stop("BUILD_WRITE");
      }
      if (_storage == eri_storage_e::outcore) _dZ.reset();
      _Timer.stop("BUILD_TOTAL");
      _thc_builder_opt.value().print_timers();
      _thc_builder_opt.reset();
      app_log(2, "***************************************************");
      app_log(2, "                  THC-READER::BUILD_ISDF_ONLY() ");
      app_log(2, "***************************************************");
      app_log(2, "    Build total:                     {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "      - compute ISDF:                {0:.3f} sec", _Timer.elapsed("BUILD_ISDF"));
      app_log(2, "      - gather collocation matrices: {0:.3f} sec", _Timer.elapsed("BUILD_GATHER"));
      app_log(2, "      - ISDF check:                  {0:.3f} sec", _Timer.elapsed("ISDF_CHECK"));
      if (_Timer.elapsed("BUILD_WRITE") > 0)
        app_log(2, "      - write ISDF:                  {0:.3f} sec", _Timer.elapsed("BUILD_WRITE"));
      app_log(2, "***************************************************\n");
    }

    void read() {
      _Timer.start("BUILD_TOTAL");
      // Cache precomputed THC ERIs
      h5::file file(_eri_file, 'r');
      h5::group grp(file);

      utils::check( not _get_Sinv_Ivec, "Finish: SinvIvec not yet written to file. Finish!!!");

      {
        std::vector<int> arng(2);  
        h5::h5_read(grp, "X_orbital_range", arng);
        utils::check(arng.size()==2 and arng[0]>=0 and arng[1]<=_nbnd,
                     "thc_reader::read(): Invalid X_orbital_range."); 
        x_range = nda::range(arng[0],arng[1]);
        h5::h5_read(grp, "Y_orbital_range", arng);
        utils::check(arng.size()==2 and arng[0]>=0 and arng[1]<=_nbnd,
                     "thc_reader::read(): Invalid Y_orbital_range."); 
        y_range = nda::range(arng[0],arng[1]);
      }
      nda::h5_read(grp, "interpolating_points", _rp);
      nda::h5_read(grp, "interpolating_vectors_G0", _Chi_head);
      nda::h5_read(grp, "dual_interpolating_vectors_G0", _Chi_bar_head);
      utils::check(_rp.shape(0) == _Np,
                   "thc_reader_t::build: rp.shape() != Np. Inconsistent dimensions from the precomputed THC-ERI.");

      if(_X_shm.shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, x_range.size()}) {
        _X_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, x_range.size()});
        _mpi->node_comm.barrier(); 
      }
      _X_shm.win().fence();
      if (_mpi->node_comm.root()) {
        auto Xloc = _X_shm.local();
        nda::h5_read(grp, "collocation_matrix", Xloc);
      }
      _X_shm.win().fence();

      if(x_range != y_range) {
        if(not _Y_shm.has_value() or _Y_shm.value().shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, y_range.size()}) {
          _Y_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
              *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, y_range.size()});
          _mpi->node_comm.barrier();
        }
        _Y_shm.value().win().fence();
        if (_Y_shm.value().node_comm()->root()) {
          auto Yloc = _Y_shm.value().local();
          nda::h5_read(grp, "Y_collocation_matrix", Yloc);
        }
        _Y_shm.value().win().fence();
      }

      if (_storage == eri_storage_e::incore) {
        int np = _mpi->comm.size();
        long nqpools = utils::find_proc_grid_max_npools(np, _nqpts_ibz, 0.2);
        utils::check(nqpools > 0 and nqpools <= _nqpts_ibz, "thc_reader_t::build: nqpools <= 0 or nqpools > nqpts");
        utils::check(np % nqpools == 0, "thc_reader_t::build: comm.size() % nqpools != 0");
        int np_PQ = np / nqpools;
        int np_P = utils::find_proc_grid_min_diff(np_PQ, 1, 1);
        int np_Q = np_PQ / np_P;
        _dZ = math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, {nqpools, np_P, np_Q}, {_nqpts_ibz, _Np, _Np});
        math::nda::h5_read(grp, "coulomb_matrix", _dZ);

        _mpi->comm.barrier();
      }
      _Timer.stop("BUILD_TOTAL");

      print_thc_summary();
    }

    int read_Np() {
      int Np;
      h5::file file(_eri_file, 'r');
      h5::group grp(file);
      h5::h5_read(grp, "Np", Np);
      return Np;
    }

  public:
    // The q-independent collocation matrix
    /**
     * Collocation matrix for a given spin 'is', polarization 'ip' and k-point 'ik'
     */
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto X(int is, int ip, int ik) const {
      _Timer.start("READ_X");
      utils::check(is >= 0 and is < _ns, "Error in thc::reader_t::X(is,ip,ik): is out of bounds: is:{}",is);
      utils::check(ip >= 0 and ip < _npol, "Error in thc::reader_t::X(is,ip,ik): is out of bounds: ip:{}",ip);
      int id = is*_npol_in_basis+ip;
      if(_ns_in_basis == 1) id = std::min(ip,_npol_in_basis-1);
      else if(_npol_in_basis == 1) id = is;
      auto Xsk = _X_shm.local()(id, ik, nda::ellipsis{});
      _Timer.stop("READ_X");
      if constexpr (MEM == HOST_MEMORY) {
        return std::as_const(Xsk);  // to make sure it is not modified
      } else {
        return memory::to_memory_space<MEM>(Xsk); 
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto X() const {
      if constexpr (MEM == HOST_MEMORY) {
        auto X_ = _X_shm.local();
        return std::as_const(X_);
      } else {
        return memory::to_memory_space<MEM>(_X_shm.local());
      }
    }

    // The q-independent collocation matrix
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto Y(int is, int ip, int ik) const {
      if(x_range == y_range) {
        return X<MEM>(is,ip,ik);
      } else {
        utils::check(_Y_shm.has_value(), "thc_reader::Y(is,ik): _Y_shm has no value.");
        utils::check(is >= 0 and is < _ns, "Error in thc::reader_t::Y(is,ip,ik): is out of bounds: is:{}",is);
        utils::check(ip >= 0 and ip < _npol, "Error in thc::reader_t::Y(is,ip,ik): is out of bounds: ip:{}",ip);
        _Timer.start("READ_X");
        int id = is*_npol_in_basis+ip;
        if(_ns_in_basis == 1) id = std::min(ip,_npol_in_basis-1);
        else if(_npol_in_basis == 1) id = is;
        auto Ysk = _Y_shm.value().local()(id, ik, nda::ellipsis{});
        _Timer.stop("READ_X");
        if constexpr (MEM == HOST_MEMORY) {
          return std::as_const(Ysk);  // to make sure it is not modified
        } else {
          return memory::to_memory_space<MEM>(Ysk);
        }
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto Y() const {
      if constexpr (MEM == HOST_MEMORY) {
        if(x_range == y_range) {
          auto X_ = _X_shm.local();
          return std::as_const(X_);
        } else {
          utils::check(_Y_shm.has_value(), "thc_reader::Y(): _Y_shm has no value.");
          auto Y_ = _Y_shm.value().local();
          return std::as_const(Y_);
        }
      } else {
        if(x_range == y_range) {
          return memory::to_memory_space<MEM>(_X_shm.local());
        } else {
          utils::check(_Y_shm.has_value(), "thc_reader::Y(): _Y_shm has no value.");
          return memory::to_memory_space<MEM>(_Y_shm.value().local());
        }
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z_same_q(int iq) const {
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Zq(_Np, _Np);
      if (_storage == eri_storage_e::incore) {
        math::nda::gather_sub_matrix(iq, 0, _dZ, &Zq);
        _dZ.communicator()->all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
      } else {
        if (_dZ.communicator()->rank()==0) {
          h5::file file(_eri_file, 'r');
          h5::group grp(file);
          nda::h5_read(grp, "coulomb_matrix", Zq,
                       std::tuple{iq, nda::range::all, nda::range::all});
        }
        _dZ.communicator()->all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Zq;
      } else {
        return memory::to_memory_space<MEM>(Zq);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z(int iq, bool same_q=false) const {
      if (same_q)
        return Z_same_q<MEM>(iq);
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Zq(_Np, _Np);
      if (_storage == eri_storage_e::incore) {
        int iq_at_ip = -1;
        for (int ip = 0; ip < _dZ.communicator()->size(); ++ip) {
          if (ip == _dZ.communicator()->rank()) iq_at_ip = iq;
          _dZ.communicator()->broadcast_n(&iq_at_ip, 1, ip);
          utils::check(iq_at_ip >= 0, "Error: iq_at_ip < 0");
          math::nda::gather_sub_matrix(iq_at_ip, ip, _dZ, &Zq);
        }
      } else {
        h5::file file(_eri_file, 'r');
        h5::group grp(file);
        nda::h5_read(grp, "coulomb_matrix", Zq,
                     std::tuple{iq, nda::range::all, nda::range::all});
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Zq;
      } else {
        return memory::to_memory_space<MEM>(Zq);
      }
    }

    // version that requires less communication
    template<MEMORY_SPACE MEM = HOST_MEMORY>
    memory::array<MEM, ComplexType, 2> Z(long iq, nda::range P_rng, nda::range Q_rng,
                                 long qpool_id, long nqpool,
                                 mpi3::communicator &q_intra_comm) const {
      _Timer.start("READ_V");
      nda::array<ComplexType, 2> Z_PQ(P_rng.size(), Q_rng.size());
      Z_PQ() = 0.0;
      if (_storage == eri_storage_e::incore) {
        for (long iqpool = 0; iqpool < nqpool; ++iqpool) {
          long iqq = (qpool_id==iqpool and q_intra_comm.root())? iq : 0;
          long ip  = (qpool_id==iqpool and q_intra_comm.root())? _mpi->comm.rank() : 0;
          _mpi->comm.all_reduce_in_place_n(&iqq, 1, std::plus<>{});
          _mpi->comm.all_reduce_in_place_n(&ip, 1, std::plus<>{});

          nda::array<ComplexType, 2> Zq;
          if (qpool_id == iqpool) {
            // CNY: improve this! not all processors need the entire matrix
            Zq = nda::array<ComplexType, 2>(_Np, _Np);
            Zq() = 0.0;
            math::nda::gather_sub_matrix(iqq, ip, _dZ, &Zq);
            q_intra_comm.all_reduce_in_place_n(Zq.data(), Zq.size(), std::plus<>{});
            Z_PQ = Zq(P_rng, Q_rng);
          } else {
            math::nda::gather_sub_matrix(iqq, ip, _dZ, &Zq);
          }
        }
      } else {
        h5::file file(_eri_file, 'r');
        h5::group grp(file);
        nda::h5_read(grp, "coulomb_matrix", Z_PQ, std::tuple{iq, P_rng, Q_rng});
      }

      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return Z_PQ;
      } else {
        return memory::to_memory_space<MEM>(Z_PQ);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto dZ(std::array<long, 3> pgrid, std::array<long, 3> bsize = {0, 0, 0}) const {
      _Timer.start("READ_V");
      auto dZ_qPQ = math::nda::make_distributed_array<Array_t<HOST_MEMORY,3>>(_mpi->comm, pgrid, {_nqpts_ibz, _Np, _Np}, bsize);
      if (_storage == eri_storage_e::incore) {
        math::nda::redistribute(_dZ, dZ_qPQ);
      } else {
        auto q_rng = dZ_qPQ.local_range(0);
        auto P_rng = dZ_qPQ.local_range(1);
        auto Q_rng = dZ_qPQ.local_range(2);

        auto Z_loc = dZ_qPQ.local();
        for( auto [iq,q] : itertools::enumerate(q_rng) ) {
          auto Zq = Z(q);
          Z_loc(iq, nda::ellipsis{}) = Zq(P_rng, Q_rng);
        }
      }
      _Timer.stop("READ_V");
      if constexpr (MEM == HOST_MEMORY) {
        return dZ_qPQ;
      } else {
        auto dZ_qPQ_d = math::nda::make_distributed_array<Array_t<MEM,3>>(_mpi->comm, pgrid, {_nqpts_ibz, _Np, _Np}, bsize);  
        dZ_qPQ_d.local() = dZ_qPQ.local();
        return dZ_qPQ_d; 
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto basis_head() const {
      if constexpr (MEM == HOST_MEMORY) {
        auto _C = _Chi_head();
        return std::as_const(_C);
      } else {
        return memory::to_memory_space<MEM>(_Chi_head);
      }
    }

    template<MEMORY_SPACE MEM = HOST_MEMORY>
    auto basis_bar_head() const
    {
      if constexpr (MEM == HOST_MEMORY) {
        auto _C = _Chi_bar_head();
        return std::as_const(_C);
      } else {
        return memory::to_memory_space<MEM>(_Chi_bar_head);
      }
    }

    bool initialized() const { return _initialized; }
    bool thc_builder_is_null() const { return _thc_builder_opt == std::nullopt; }
    int Np() const { return _Np; }
    int nkpts() const { return _nkpts; }
    int nkpts_ibz() const { return _nkpts_ibz; }
    int nqpts() const { return _nqpts; }
    int nqpts_ibz() const { return _nqpts_ibz; }
    int ns() const { return _ns; }
    int ns_in_basis() const { return _ns_in_basis; }
    int npol() const { return _npol; }
    int npol_in_basis() const { return _npol_in_basis; }
    int nbnd() const { return _nbnd; }
    int nbnd_aux() const { return 0; }
    std::string& set_X_type() { return _X_type; }
    const std::string thc_X_type() const { return _X_type; }
    std::string filename() const { return _eri_file; }
    //mpi3::communicator* comm() const { return std::addressof(_mpi->comm); }
    auto& MF() const { return _MF; }
    auto& mpi() const { return _mpi; }
    auto X_orbital_range() const { return x_range; }
    auto Y_orbital_range() const { return y_range; }

    void print_timers() const {
      app_log(2, "\n  THC-READER timers");
      app_log(2, "  -----------------");
      app_log(2, "    BUILD:                {0:.3f} sec", _Timer.elapsed("BUILD_TOTAL"));
      app_log(2, "    READ_X:               {0:.3f} sec", _Timer.elapsed("READ_X"));
      app_log(2, "    READ_V:               {0:.3f} sec\n", _Timer.elapsed("READ_V"));
    }

  private:
    void gather_X_shm(math::nda::DistributedArrayOfRank<4> auto &dXa) {
      using nda::range;
      int norb = x_range.size();
      if(_X_shm.shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb} ) {
        _X_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb});
      }
      auto sX_buffer = math::shm::make_shared_array<nda::array_view<ComplexType, 4>>(
          *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, norb, _Np});
      math::nda::gather_to_shm(dXa, sX_buffer);

      if (_X_shm.node_comm()->root()) {
        for (int is = 0; is < _ns_in_basis*_npol_in_basis; ++is) {
          for (int ik = 0; ik < _nkpts; ++ik) {
            auto Xsk_trans = sX_buffer.local()(is, ik, range::all, range::all); // (norb, Np)
            auto Xsk = _X_shm.local()(is, ik, range::all, range::all); // (Np, norb)
            Xsk = nda::transpose(Xsk_trans);
          }
        }
      }
      _X_shm.communicator()->barrier();
    }

    void gather_Y_shm(math::nda::DistributedArrayOfRank<4> auto &dXb) {
      using nda::range;
      int norb = y_range.size();
      if( not _Y_shm.has_value() or
          _Y_shm.value().shape() != std::array<long,4>{_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb} ) {
        _Y_shm = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, _Np, norb});
      }
      auto sY_buffer = math::shm::make_shared_array<Array_view_t<HOST_MEMORY,4>>(
            *_mpi, {_ns_in_basis*_npol_in_basis, _nkpts, norb, _Np});
      math::nda::gather_to_shm(dXb, sY_buffer);

      if (_Y_shm.value().node_comm()->root()) {
        for (int is = 0; is < _ns_in_basis*_npol_in_basis; ++is) {
          for (int ik = 0; ik < _nkpts; ++ik) {
            auto Ysk_trans = sY_buffer.local()(is, ik, range::all, range::all); // (norb, Np)
            auto Ysk = _Y_shm.value().local()(is, ik, range::all, range::all); // (Np, norb)
            Ysk = nda::transpose(Ysk_trans);
          }
        }
      }
      _Y_shm.value().communicator()->barrier();
    }

    void isdf_check(dArray_t<HOST_MEMORY, 3> const& dzeta_qur) {

      utils::check(_nkpts==1, "isdf_check: currently only supports gamma point. ");

      auto zeta = math::nda::make_distributed_array<Array_t<HOST_MEMORY, 3>>(
          _mpi->comm, {1, 1, _mpi->comm.size()}, dzeta_qur.global_shape());
      math::nda::redistribute(dzeta_qur, zeta);

      auto r_rng = zeta.local_range(2);

      auto sphi_ir = math::shm::make_shared_array<Array_view_t<HOST_MEMORY, 2>>(
          *_mpi, {_nbnd, _MF->nnr()});
      if (_mpi->node_comm.root())
        _MF->get_orbital_set('r', 0, 0, nda::range(_nbnd), sphi_ir.local());
      _mpi->comm.barrier();

      auto zeta_loc = zeta.local();
      auto phi_ir = sphi_ir.local();
      auto phi_ui = _X_shm.local();
      nda::array<ComplexType, 1> rho_fit_r(r_rng.size());
      nda::array<RealType, 1> error_r(r_rng.size());
      double max_diff = -1;
      for (size_t i=0; i<_nbnd; ++i) {
        for (size_t j=0; j<_nbnd; ++j) {
          rho_fit_r() = 0.0;
          for (size_t u=0; u<_Np; ++u)
            rho_fit_r += std::conj(phi_ui(0,0,u,i)) * phi_ui(0,0,u,j) * zeta_loc(0,u,nda::range::all);

          for( auto [ir,r] : itertools::enumerate(r_rng) )
            error_r(ir) = std::abs( std::conj(phi_ir(i, r)) * phi_ir(j, r) - rho_fit_r(ir) );
          max_diff = std::max(max_diff, nda::max_element(error_r));
        }
      }
      double max_diff_global = _mpi->comm.max(max_diff);
      app_log(1, "\nMaximum error of ISDF: {}\n", max_diff_global);
      _mpi->comm.barrier();
    }

  private:
    std::shared_ptr<mf::MF> _MF;
    std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _mpi;
    // where to perform THC evaluation
    MEMORY_SPACE _MEM_EVAL = DEFAULT_MEMORY_SPACE;
    // whether the thc integrals has been initialized
    bool _initialized = false;
    // eri storage type: incore or outcore
    eri_storage_e _storage;
    // file to store eris
    std::string _eri_file;
    // eri format to store
    std::string _format;
    std::string _cd_dir;     // directory for CD eris;
    std::string _X_type;

    std::optional<thc> _thc_builder_opt;

    int _Np;
    int _nkpts;
    int _nkpts_ibz;
    int _nqpts;
    int _nqpts_ibz;
    int _ns;
    int _ns_in_basis;
    int _nbnd;
    int _npol;
    int _npol_in_basis;
    bool _get_Sinv_Ivec;
    nda::range x_range;
    nda::range y_range;

    // add option to keep data on device or on host memory with INCORE
    // keep everything as optionals to keep things simple

    dArray_t<HOST_MEMORY,3> _dZ;
    // Used if storing in HOST_MEMORY
    sArray_t<memory::array_view<HOST_MEMORY, ComplexType, 4>> _X_shm;
    std::optional<sArray_t<memory::array_view<HOST_MEMORY, ComplexType, 4>>> _Y_shm;

    std::optional<dArray_t<HOST_MEMORY,3>> _dSinv_Ivec; 
    memory::array<HOST_MEMORY, ComplexType, 2> _Chi_head;
    memory::array<HOST_MEMORY, ComplexType, 2> _Chi_bar_head;
    memory::array<HOST_MEMORY, long, 1> _rp;

    mutable utils::TimerManager _Timer;
  };

} // methods

#endif //COQUI_THC_READER_T_HPP
