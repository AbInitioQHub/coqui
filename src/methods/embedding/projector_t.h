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


#ifndef COQUI_PROJECTOR_T_H
#define COQUI_PROJECTOR_T_H

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "numerics/shared_array/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "mean_field/MF.hpp"

namespace methods {

  namespace mpi3 = boost::mpi3;

  class projector_t {
  public:
    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;

  public:
    projector_t(mf::MF &MF, std::string C_file, bool translate_home_cell=false,
                bool print_info=true):
    _MF(std::addressof(MF)), _C_file(C_file) {
      read_wannier_basis(_MF, _C_file, _C_skIai, _W_rng, translate_home_cell);
      _nImps = _C_skIai.shape(2);
      _nImpOrbs = _C_skIai.shape(3);
      _nOrbs_W = _C_skIai.shape(4);

      if (print_info) print_metadata();
    }

    projector_t(mf::MF &MF,
                const nda::array<ComplexType, 5> &C_ksIai,
                const nda::array<long, 3> &band_window,
                const nda::array<RealType, 2> &kpts_crys,
                bool translate_home_cell=false,
                bool print_info=true):
    _MF(std::addressof(MF)), _C_file("") {

      utils::check(translate_home_cell == false,
                   "projector_t: translate_home_cell is not implemented for C_skIai and band_window constructor!");

      // 1. Index i doesn't usually span over mf->nbnd()
      // 2. Here, the k index lives on the full MP mesh
      // 3. Since bdft shuffles the order of the k-points in the presence of symmetry,
      //    the k ordering of C_ksIai is likely different from _MF->kpts()
      auto [nkpts, ns, nImps, nImpOrbs, nOrbs_W] = C_ksIai.shape();
      _C_skIai = nda::array<ComplexType, 5>(ns, nkpts, nImps, nImpOrbs, nOrbs_W);
      utils::check(nImps == 1, "read_wannier_basis: Implementations for multiple impurities are not ready yet!");

      // reorder k axis for C: _MF->kpts_crystal()(ik) = kpts_crys( kp_map(ik) )
      nda::array<int, 1> kp_map(nkpts);
      utils::calculate_kp_map(kp_map, _MF->kpts_crystal(), kpts_crys);
      for (long is = 0; is < ns; ++is) {
        for (long ik = 0; ik < nkpts; ++ik) {
          _C_skIai(is, ik, nda::ellipsis{}) = C_ksIai(kp_map(ik), is, nda::ellipsis{});
        }
      }

      _W_rng = std::vector<nda::range>(nImps,nda::range(0));
      for (long I = 0; I < nImps; ++I) {
        // band_window is 1-based, so we need to subtract 1 to convert it to 0-based
        _W_rng[I] = nda::range(band_window(I,0,0)-1, band_window(I,0,1));
      }

      _nImps = _C_skIai.shape(2);
      _nImpOrbs = _C_skIai.shape(3);
      _nOrbs_W = _C_skIai.shape(4);

      if (print_info) print_metadata();
    }

    projector_t(projector_t const&) = default;
    projector_t(projector_t &&) = default;
    projector_t& operator=(projector_t const& other) = default;
    projector_t& operator=(projector_t && other) = default;

    ~projector_t() = default;

    void print_metadata();


    static void read_wannier_basis(mf::MF *mf, std::string C_file,
                                   nda::array<ComplexType, 5> &C_skIai,
                                   std::vector<nda::range> &W_rng,
                                   bool translate_home_cell=false) {

      app_log(2, "\nReading Wannier orbitals from {}. \n", C_file);

      nda::array<ComplexType, 5> C_ksIai;
      nda::array<RealType, 2> kpts_crys;
      nda::array<long, 3> band_window;
      h5::file file(C_file, 'r');
      h5::group grp(file);
      nda::h5_read(grp, "dft_input/proj_mat", C_ksIai);
      nda::h5_read(grp, "dft_input/kpts", kpts_crys);
      nda::h5_read(grp, "dft_misc_input/band_window", band_window);
      band_window() -= 1; // use 0-based notation

      // Apply phase coming from shifting the Wannier centres to the home unit cell R=(0,0,0)
      if (grp.has_dataset("dft_input/wan_centres")) {
        nda::array<RealType, 3> wan_centres;
        nda::h5_read(grp, "dft_input/wan_centres", wan_centres);
        if (translate_home_cell) apply_trans_phase(mf, C_ksIai, kpts_crys, wan_centres);
      } else {
        app_log(2, "    [WARNING] \"dft_input/wan_centres\" does not exist in {}, \n"
                   "              and therefore Wannier centres will not be manually translated to the home unit cell. \n"
                   "              Please make sure the centres coming from Wannier90 are already in the home unit cell.\n", C_file);
      }

      // 1. Index i doesn't usually span over mf->nbnd()
      // 2. Here, the k index lives on the full MP mesh
      // 3. Since bdft shuffles the order of the k-points in the presence of symmetry,
      //    the k ordering of C_ksIai is likely different from _MF->kpts()
      auto [nkpts, ns, nImps, nImpOrbs, nOrbs_W] = C_ksIai.shape();
      nda::array<long, 1> imp_dims(nImps);
      long nImpOrbs_total = 0;
      if (nImps != 1) {
        app_log(1, "╔═══════════════════════════════════════════════════════╗");
        app_log(1, "║ [ WARNING ]                                           ║");
        app_log(1, "║ Multiple impurities were found in the Wannier90 HDF5. ║");
        app_log(1, "║ CoQuí will merge them in sequence and treat them as   ║");
        app_log(1, "║ a single impurity.                                    ║");
        app_log(1, "╚═══════════════════════════════════════════════════════╝\n");
        for (long I=0; I<nImps; ++I) {
          h5::h5_read(grp, "dft_input/shells/"+std::to_string(I)+"/dim", imp_dims(I));
          nImpOrbs_total += imp_dims(I);
        }
      } else {
        imp_dims(0) = nImpOrbs;
        nImpOrbs_total = nImpOrbs;
      }
      C_skIai = nda::array<ComplexType, 5>(ns, nkpts, 1, nImpOrbs_total, nOrbs_W);

      // reorder k axis for C: _MF->kpts_crystal()(ik) = kpts_crys( kp_map(ik) )
      // and combine multiple impurities since CoQui currently only support single impurity
      nda::array<int, 1> kp_map(nkpts);
      utils::calculate_kp_map(kp_map, mf->kpts_crystal(), kpts_crys);
      for (long is = 0; is < ns; ++is) {
        for (long ik = 0; ik < nkpts; ++ik) {
          long offset = 0;
          for (long I = 0; I < nImps; ++I) {
            utils::check(offset+imp_dims(I) <= nImpOrbs_total,
                         "read_wannier_basis: something wrong when combining multiple impurities!");
            C_skIai(is, ik, 0, nda::range(offset, offset+imp_dims(I)), nda::range::all) =
                C_ksIai(kp_map(ik), is, I, nda::range::all, nda::range::all);
            offset += imp_dims(I);
          }
        }
      }

      W_rng = std::vector<nda::range>(1,nda::range(0));
      W_rng[0] = nda::range(band_window(0,0,0), band_window(0,0,1)+1);
    }

    static void apply_trans_phase(mf::MF *mf,
                                  nda::array<ComplexType, 5> &C_ksIai,
                                  const nda::array<RealType, 2> &kpts_crys,
                                  const nda::array<RealType, 3> &wan_centres) {
      auto [nkpts, ns, nImps, nImpOrbs, nOrbs_W] = C_ksIai.shape();

      app_log(2, "Translating each Wannier orbital to the home unit cell individually. \n"
                 "[WARNING] This is incompatible with a previous Wannier90 calculation where \"translate_home_cell=true\" was set!\n");

      // Calculate shifts to home cell
      // Note that wan_centres is in Angstrom
      double ang_to_bohr = 1.8897259885789;
      auto recv = mf->recv();
      double tpi = 2.0 * 3.14159265358979;
      double tpiinv = 1.0 / tpi;
      nda::stack_array<double,3> R_scaled;
      nda::array<RealType, 3> R_trans(wan_centres.shape());
      for (size_t sa=0; sa<ns*nImpOrbs; ++sa) {
        size_t is = sa / nImpOrbs;
        size_t a  = sa % nImpOrbs;
        // R_scaled = A_inv.T * R = 1/(2*pi) * B * R
        nda::blas::gemv(ang_to_bohr * tpiinv, recv, wan_centres(is,a,nda::range::all), 0.0, R_scaled);

        // R_scaled = R_home + R_trans
        R_trans(is,a,0) = std::floor(R_scaled(0));
        R_trans(is,a,1) = std::floor(R_scaled(1));
        R_trans(is,a,2) = std::floor(R_scaled(2));
        app_log(2, "  Centre {0}: ({1:.6f}, {2:.6f}, {3:.6f}) -> ({4:.6f}, {5:.6f}, {6:.6f})\n",
                sa, R_scaled(0), R_scaled(1), R_scaled(2),
                R_scaled(0)-R_trans(is,a,0), R_scaled(1)-R_trans(is,a,1), R_scaled(2)-R_trans(is,a,2));
      }

      // apply phase
      for (size_t ik=0; ik<nkpts; ++ik) {
        for (size_t sa=0; sa<ns*nImpOrbs; ++sa) {
          size_t is = sa / nImpOrbs;
          size_t a  = sa % nImpOrbs;
          double kR = kpts_crys(ik,0)*R_trans(is,a,0) +  kpts_crys(ik,1)*R_trans(is,a,1) +  kpts_crys(ik,2)*R_trans(is,a,2);
          auto Cka = C_ksIai(ik,is,0,a,nda::range::all);
          Cka *= std::exp( ComplexType(0.0, -tpi*kR) );
        }
      }
    }

    template<nda::ArrayOfRank<4> Array_base_t, nda::ArrayOfRank<4> Oloc_t>
    void upfold(sArray_t<Array_base_t> &O_skij, const Oloc_t &Oloc_sIab) const;
    template<nda::ArrayOfRank<5> Array_base_t, nda::ArrayOfRank<5> Oloc_t>
    void upfold(sArray_t<Array_base_t> &O_tskij, const Oloc_t &Oloc_tsIab) const;
    template<nda::ArrayOfRank<5> Array_base_t, nda::ArrayOfRank<6> Ac_t>
    void upfold(sArray_t<Array_base_t> &O_tskij, const Ac_t &O_tskIab) const;

    template<nda::ArrayOfRank<5> Array_base_t>
    auto downfold_k_fbz(const sArray_t<Array_base_t> &O_tskij) const -> nda::array<ComplexType, 6>;

    template<typename comm_t>
    auto downfold_k(const nda::MemoryArrayOfRank<5> auto &O_tskij, comm_t comm) const
    -> nda::array<ComplexType, 6>;
    template<typename comm_t>
    auto downfold_k(const nda::MemoryArrayOfRank<4> auto &O_skij, comm_t comm) const
    -> nda::array<ComplexType, 5>;
    template<typename comm_t>
    auto downfold_k(const nda::MemoryArrayOfRank<3> auto &O_ski, comm_t comm) const
    -> nda::array<ComplexType, 5>;

    template<nda::ArrayOfRank<5> Array_base_t>
    auto downfold_k(const sArray_t<Array_base_t> &O_tskij) const -> nda::array<ComplexType, 6>;
    template<nda::ArrayOfRank<4> Array_base_t>
    auto downfold_k(const sArray_t<Array_base_t> &O_skij) const -> nda::array<ComplexType, 5>;
    template<nda::ArrayOfRank<3> Array_base_t>
    auto downfold_k(const sArray_t<Array_base_t> &O_ski) const -> nda::array<ComplexType, 5>;

    template<bool force_real=false, typename comm_t>
    auto downfold_loc(const nda::MemoryArrayOfRank<4> auto &O_skij, comm_t comm,
                      std::string name = "") const
    -> nda::array<ComplexType, 4>;
    template<bool force_real=false, typename comm_t>
    auto downfold_loc(const nda::MemoryArrayOfRank<5> auto &O_tskij, comm_t comm,
                      std::string name = "") const
    -> nda::array<ComplexType, 5>;

    template<bool force_real=false, nda::ArrayOfRank<4> Array_base_t>
    auto downfold_loc(const sArray_t<Array_base_t> &O_skij, std::string name = "") const
    -> nda::array<ComplexType, 4>;
    template<bool force_real=false, nda::ArrayOfRank<5> Array_base_t>
    auto downfold_loc(const sArray_t<Array_base_t> &O_tskij, std::string name = "") const
    -> nda::array<ComplexType, 5>;

  private:
    mf::MF* _MF = nullptr;

    std::string _C_file;
    nda::array<ComplexType, 5> _C_skIai;
    // orbital range for the W space for each impurity (Assume this is k-independent)
    std::vector<nda::range> _W_rng;

    long _nImps = -1;
    long _nOrbs_W = -1;
    long _nImpOrbs = -1;

  public:
    mf::MF* MF() const { return _MF; }
    auto C_skIai() const { return _C_skIai(); }
    const auto& W_rng() const { return _W_rng; }
    long nImps() const { return _nImps; }
    long nImpOrbs() const { return _nImpOrbs; }
    long nOrbs_W() const { return _nOrbs_W; }
    std::string C_file() const { return _C_file; }

  }; // projector_t

} // methods


#endif //COQUI_PROJECTOR_T_H
