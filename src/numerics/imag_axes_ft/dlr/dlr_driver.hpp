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

#ifndef COQUI_DLR_GENERATOR_HPP
#define COQUI_DLR_GENERATOR_HPP

#ifdef ENABLE_DLR

#include <string>

#include "nda/nda.hpp"
#include "nda/lapack.hpp"

#include "numerics/nda_functions.hpp"
// FIXME hacky solution to updates nda namespace until cppdlr's nda dependency is updated. 
#include "numerics/imag_axes_ft/dlr/nda_linalg_compat.hpp"
// FIXME avoid including full cppdlr header until cppdlr's nda dependency is updated. 
//#include "cppdlr/cppdlr.hpp"
#include "cppdlr/dlr_build.hpp"
#include "cppdlr/dlr_imtime.hpp"
#include "cppdlr/dlr_imfreq.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "../iaft_enum_e.hpp"

namespace imag_axes_ft {
  namespace dlr {

    inline nda::matrix<ComplexType> build_if2it(cppdlr::imfreq_ops const &ifops, cppdlr::imtime_ops const &itops) {
      auto nw = ifops.get_ifnodes().size();
      auto dlr_rank = itops.rank();

      auto if2it = nda::matrix<ComplexType>(dlr_rank, nw);

      // Solve cf2if^T C^T = cf2it^T to obtain C = cf2it * cf2if^{-1}
      
      // Copy cf2it into output matrix (cast to complex)
      if2it(nda::range::all, nda::range(dlr_rank)) = itops.get_cf2it();

      if (nw == dlr_rank) {
        // Square system---use getrs
        auto lu  = nda::matrix<ComplexType>(ifops.get_if2cf_lu());
        auto piv = nda::vector<int>(ifops.get_if2cf_piv());
        nda::lapack::getrs(nda::transpose(lu), nda::transpose(if2it), piv);
      } else {
        // Non-square system---use least squares solver
        auto s       = nda::vector<double>(dlr_rank); // Not needed
        double rcond = 0;                             // Not needed
        int rank     = 0;                             // Not needed
        auto cf2if = nda::matrix<ComplexType>(ifops.get_cf2if());
        // (cf2if)^T * X = (cf2it)^T -> X = (cf2if^-1)^T * cf2it^T 
        //                                = (if2cf^T * cf2it^T)
        nda::lapack::gelss(nda::transpose(cf2if), nda::transpose(if2it), s, rcond, rank);
      }

      return if2it;
    }

    struct DLR {
    public:
      DLR() = default;

      DLR(double beta_, double wmax_, std::string prec_ = "high", bool print_meta_log = false):
          beta(beta_), wmax(wmax_), prec(prec_) {
        eps = eps_from_prec(prec);
        init(print_meta_log);
      }

      DLR(double beta_, double wmax_, double eps_, bool print_meta_log = false):
          beta(beta_), wmax(wmax_), prec("custom"), eps(eps_) {
        utils::check(eps > 0.0,
          "imag_axes_ft::dlr::DLR: eps = {} must be positive.", eps);
        init(print_meta_log);
      }

      static double eps_from_prec(std::string const &prec) {
        if (prec == "high") {
          return 1e-13;
        } else if (prec == "medium") {
          return 1e-10;
        } else if (prec == "low") {
          return 1e-06;
        }

        utils::check(false,
          "imag_axes_ft::dlr::DLR: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"",
          prec);
        return 0.0;
      }

      DLR(const DLR &other) = default;
      DLR(DLR &&other) = default;

      DLR &operator=(const DLR &other) = default;
      DLR &operator=(DLR &&other) = default;

      ~DLR() = default;

      void metadata_log() const {
        app_log(1, "  Mesh details on the imaginary axis");
        app_log(1, "  ----------------------------------");
        app_log(1, "  Discrete Lehmann Representation");
        app_log(1, "  Beta                   = {} a.u.", beta);
        app_log(1, "  Frequency cutoff       = {} a.u.", wmax);
        app_log(1, "  Lambda                 = {}", lambda);
        if (prec == "custom") {
          app_log(1, "  Precision              = {}", eps);
        } else {
          app_log(1, "  Precision              = {} (i.e. eps = {})", prec, eps);
        }
        app_log(1, "  nt_f, nt_b, nw_f, nw_b = {}, {}, {}, {}\n", nt_f, nt_b, nw_f, nw_b);
      }


    auto construct_tau_interpolate_matrix(
      const nda::MemoryArrayOfRank<1> auto &tau_mesh_out,
      bool ph_sym=false) const
    -> nda::array<ComplexType, 2> {
      auto nt_out = tau_mesh_out.size();
      auto nt_in = _it_ops.rank();
      auto eye_nt_in = nda::eye<ComplexType>(nt_in);

      auto rhs_f = nda::matrix<ComplexType, nda::F_layout>(nt_in, nt_out);
      for (int it_out = 0; it_out < nt_out; ++it_out) {
        auto tau = (tau_mesh_out(it_out) + 1.0) * 0.5; // convert from [-1, 1] to [0, 1]
        auto cf2eval = _it_ops.coefs2eval(eye_nt_in, cppdlr::abs2rel(tau));
        rhs_f(nda::range::all, it_out) = cf2eval;
      }

      auto cf2it_t_lu = nda::matrix<ComplexType>(nda::transpose(_it_ops.get_cf2it()));

      nda::array<int, 1> ipiv(nt_in);
      auto info = nda::lapack::getrf(cf2it_t_lu, ipiv);
      utils::check(info == 0,
                   "dlr_driver.hpp::DLR: LU factorization failed for cf2it^T (getrf info = {}).",
                   info);
      info = nda::lapack::getrs(cf2it_t_lu, rhs_f, ipiv);
      utils::check(info == 0,
                   "dlr_driver.hpp::DLR: Linear solve failed for cf2it^T x = cf2eval (getrs info = {}).",
                   info);

      nda::array<ComplexType, 2> T_tt;
      if (ph_sym) {
        // only keep the first half of the tau points for particle-hole symmetry.
        auto nt_half = nt_in / 2 + nt_in % 2;
        T_tt = nda::array<ComplexType, 2>(nt_out, nt_half);
        for (int it_out = 0; it_out < T_tt.extent(0); ++it_out) {
          for (int it_in = 0; it_in < nt_half; ++it_in) {
            int imt = nt_in - it_in - 1;
            T_tt(it_out, it_in) = (it_in == imt)? rhs_f(it_in, it_out) : rhs_f(it_in, it_out) + rhs_f(imt, it_out);
          }
        }
      } else {
        T_tt = nda::array<ComplexType, 2>(nt_out, nt_in);
        T_tt() = nda::transpose(rhs_f);
      }

      return T_tt;
    }

    auto construct_w_interpolate_matrix(
       const nda::MemoryArrayOfRank<1> auto &wn_mesh_out,
       imag_axes_ft::stats_e stats,
       bool ph_sym=false) const
    -> nda::array<ComplexType, 2> {

      auto const &if_ops = (stats == imag_axes_ft::fermion) ? _if_ops_f : _if_ops_b;
      auto nw_in    = static_cast<int>(if_ops.get_ifnodes().size());
      auto nw_out   = static_cast<int>(wn_mesh_out.size());

      // initialize rhs = cf2if_out
      auto rhs = nda::matrix<ComplexType>(nw_out, nw_in);
      auto eye_ndlr_in = nda::eye<ComplexType>(_it_ops.rank());
      for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
        long n_ir = wn_mesh_out(iw_out);
        long n_dlr = (stats == imag_axes_ft::fermion) ? 
            static_cast<int>((n_ir - 1) / 2) : static_cast<int>(n_ir / 2);
        auto cf2eval = if_ops.coefs2eval(eye_ndlr_in, n_dlr);
        rhs(iw_out, nda::range::all) = cf2eval;
      }

      // Non-square system---use least squares solver
      auto s       = nda::vector<double>(_it_ops.rank()); // Not needed
      double rcond = 0;                             // Not needed
      int rank     = 0;                             // Not needed
      auto cf2if = nda::matrix<ComplexType>(if_ops.get_cf2if());
      // (cf2if)^T * X = (cf2if_out)^T -> X = (cf2if^-1)^T * cf2if_out^T
      nda::lapack::gelss(nda::transpose(cf2if), nda::transpose(rhs), s, rcond, rank);
      
      if (not ph_sym) 
        return rhs;

      utils::check(stats == imag_axes_ft::boson, 
        "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
        "ph_sym=true is only supported for bosons.");
      // Only keep the positive-frequency half of the input (including w=0),
      // matching wn_mesh[nw/2:] ordering in the Python ph-sym path.
      auto nw_half = nw_in / 2 + nw_in % 2;
      auto iw_zero = nw_in / 2;
      nda::array<ComplexType, 2> T_ww(nw_out, nw_half);
      for (int iw_pos = 0; iw_pos < nw_half; ++iw_pos) {
        int ipw = iw_zero + iw_pos;
        int imw = iw_zero - iw_pos;
        if (ipw == imw) {
          T_ww(nda::range::all, iw_pos) = rhs(nda::range::all, ipw);
        } else {
          for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
            T_ww(iw_out, iw_pos) = rhs(iw_out, ipw) + rhs(iw_out, imw);
          }
        }
      }
      return T_ww;
    }

    auto construct_w_interpolate_matrix_old(
      const nda::MemoryArrayOfRank<1> auto &wn_mesh_out,
      imag_axes_ft::stats_e stats,
      bool ph_sym=false) const
    -> nda::array<ComplexType, 2> {

      auto const &if_ops = (stats == imag_axes_ft::fermion) ? _if_ops_f : _if_ops_b;
      auto nw_in    = static_cast<int>(if_ops.get_ifnodes().size());
      auto nw_out   = static_cast<int>(wn_mesh_out.size());

      // Build stable values->coefficients transform on the input iw mesh.
      // This path uses cppdlr's internal solve strategy and is numerically robust.
      auto if2cf = if_ops.vals2coefs(nda::eye<ComplexType>(nw_in));

      // Evaluate interpolated values on the output iw mesh (in DLR notation).
      auto T_ww_full = nda::array<ComplexType, 2>(nw_out, nw_in);
      for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
        long n_ir = wn_mesh_out(iw_out);
        if (stats == imag_axes_ft::fermion) {
          utils::check((n_ir % 2) != 0,
                       "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
                       "fermionic wn_mesh_out must be odd in IR notation, got {}.", n_ir);
        } else {
          utils::check((n_ir % 2) == 0,
                       "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
                       "bosonic wn_mesh_out must be even in IR notation, got {}.", n_ir);
        }

        // Convert from CoQuí IR notation (n_ir) to cppdlr notation (n_dlr):
        // fermion: n_ir = 2*n_dlr + 1; boson: n_ir = 2*n_dlr
        auto n_dlr = (stats == imag_axes_ft::fermion) ? static_cast<int>((n_ir - 1) / 2)
                                            : static_cast<int>(n_ir / 2);
        T_ww_full(iw_out, nda::range::all) = if_ops.coefs2eval(if2cf, n_dlr);
      }

      nda::array<ComplexType, 2> T_ww;
      if (ph_sym) {
        utils::check(stats == imag_axes_ft::boson,
                     "dlr_driver.hpp::DLR::construct_w_interpolate_matrix: "
                     "ph_sym=true is only supported for bosons.");
        // Only keep the positive-frequency half of the input (including w=0),
        // matching wn_mesh[nw/2:] ordering in the Python ph-sym path.
        auto nw_half = nw_in / 2 + nw_in % 2;
        auto iw_zero = nw_in / 2;
        T_ww = nda::array<ComplexType, 2>(nw_out, nw_half);
        for (int iw_out = 0; iw_out < nw_out; ++iw_out) {
          for (int iw_pos = 0; iw_pos < nw_half; ++iw_pos) {
            int ipw = iw_zero + iw_pos;
            int imw = iw_zero - iw_pos;
            T_ww(iw_out, iw_pos) = (ipw == imw)
              ? T_ww_full(iw_out, ipw)
              : T_ww_full(iw_out, ipw) + T_ww_full(iw_out, imw);
          }
        }
      } else {
        T_ww = T_ww_full;
      }

      return T_ww;
    }

    auto construct_tau_to_zero_and_beta_matrices() 
    -> std::pair<nda::array<ComplexType, 1>, nda::array<ComplexType, 1>> {
      nda::array<double, 1> tau_mesh_0beta = {-1.0, 1.0}; // tau = 0, beta
      auto T_tt = construct_tau_interpolate_matrix(tau_mesh_0beta);
      return {T_tt(0, nda::range::all), T_tt(1, nda::range::all)};
    }

    private:
      void init(bool print_meta_log) {
        lambda = beta * wmax;
        utils::check(lambda > 0.0, "Invalid value of lambda, {}, in imag_axes_ft::dlr::DLR.", lambda);
        utils::check(beta > 0.0, "Invalid value of beta, {}, in imag_axes_ft::dlr::DLR.", beta);
        utils::check(wmax > 0.0, "Invalid value of wmax, {}, in imag_axes_ft::dlr::DLR.", wmax);
        utils::check(eps > 0.0, "Invalid value of eps, {}, in imag_axes_ft::dlr::DLR.", eps);

        // Construct DLR basis by obtaining DLR real frequencies 
        auto dlr_rf = cppdlr::build_dlr_rf(lambda, eps, cppdlr::SYM);
        auto dlr_rank = dlr_rf.size();

        // get DLR imaginary time object
        _it_ops = cppdlr::imtime_ops(lambda, dlr_rf, cppdlr::SYM);
        utils::check(dlr_rank == _it_ops.rank(), 
        "imag_axes_ft::dlr::DLR: size of DLR real frequency basis, {}, does not match the rank of DLR imaginary time object, {}", 
        dlr_rank, _it_ops.rank());
        // dlr imagianry time interpolation nodes. 
        // TODO keep only one set of nt and tau_mesh for both fermions and bosons
        nt_f = _it_ops.rank();
        nt_b = _it_ops.rank();
        tau_mesh_f = nda::array<double, 1>(nt_f);
        tau_mesh_b = nda::array<double, 1>(nt_b);
        // The native DLR tau nodes live in [0, 1], rather than [0, beta] nor [-1, 1]
        // Convert to the IR basis notation which lives in [-1, 1] instead...
        for (int i = 0; i < nt_f; ++i) {
          tau_mesh_f(i) = 2.0 * cppdlr::rel2abs(_it_ops.get_itnodes(i)) - 1.0;
        }
        tau_mesh_b() = tau_mesh_f;

        // get DLR imaginary frequency objects
        _if_ops_f = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Fermion, cppdlr::SYM);
        _if_ops_b = cppdlr::imfreq_ops(lambda, dlr_rf, cppdlr::Boson, cppdlr::SYM);

        nw_f = _if_ops_f.get_ifnodes().size();
        nw_b = _if_ops_b.get_ifnodes().size();
        wn_mesh_f = nda::array<long, 1>(nw_f);
        wn_mesh_b = nda::array<long, 1>(nw_b);
        // Convert to IR basis notation
        for (int i = 0; i < nw_f; ++i) wn_mesh_f(i) = long(2.0 * _if_ops_f.get_ifnodes(i) + 1);
        for (int i = 0; i < nw_b; ++i) wn_mesh_b(i) = long(2.0 * _if_ops_b.get_ifnodes(i));

        // Matsubara frequency to imaginary time
        auto if2it_f = cppdlr::build_if2it(_if_ops_f, _it_ops) / beta;
        auto if2it_b = cppdlr::build_if2it(_if_ops_b, _it_ops) / beta;
        Ttw_ff = nda::matrix<ComplexType>(nt_f, nw_f);
        Ttw_ff() = if2it_f;
        Ttw_bb = nda::matrix<ComplexType>(nt_b, nw_b);
        Ttw_bb() = if2it_b;
        // imaginary time to Matsubara frequency
        auto it2if_f = cppdlr::build_it2if(_it_ops, _if_ops_f) * beta;
        auto it2if_b = cppdlr::build_it2if(_it_ops, _if_ops_b) * beta;
        Twt_ff = nda::matrix<ComplexType>(nw_f, nt_f);
        Twt_ff() = it2if_f;
        Twt_bb = nda::matrix<ComplexType>(nw_b, nt_b);
        Twt_bb() = it2if_b;

        // fermion <-> boson tau interpolation (same DLR tau nodes)
        Ttt_bf = nda::eye<ComplexType>(nt_b);
        Ttt_fb = nda::eye<ComplexType>(nt_f); 

        // tau -> coefficients
        auto it2cf  = _it_ops.vals2coefs(nda::eye<ComplexType>(nt_f));
        Tct_ff = it2cf;
        Tct_bb = it2cf;
        
        std::tie(T_zero_t_ff, T_beta_t_ff) = construct_tau_to_zero_and_beta_matrices();
        if (print_meta_log) metadata_log();
      }

    public:
      double beta = 0.0;
      double wmax = 0.0;
      std::string prec = "high";
      double eps = 0.0;
      double lambda = 0.0;

      cppdlr::imtime_ops _it_ops;
      cppdlr::imfreq_ops _if_ops_f;
      cppdlr::imfreq_ops _if_ops_b;

      int nt_f = 0;
      int nt_b = 0;
      int nw_f = 0;
      int nw_b = 0;

      // Matsubara frequency mesh
      nda::array<long, 1> wn_mesh_f;
      nda::array<long, 1> wn_mesh_b;
      // tau mesh: relative format (same convention as IR backend)
      nda::array<double, 1> tau_mesh_f;
      // TODO tau_mesh_b as an array view of tau_mesh_f since they are the same. 
      nda::array<double, 1> tau_mesh_b;

      // transformation from w_f to t_f
      nda::matrix<ComplexType> Ttw_ff;
      // transformation from t_f to w_f
      nda::matrix<ComplexType> Twt_ff;
      // transformation from w_b to t_b
      nda::matrix<ComplexType> Ttw_bb;
      // transformation from t_b to w_b
      nda::matrix<ComplexType> Twt_bb;
      // interpolation from t_f to t_b
      nda::matrix<ComplexType> Ttt_bf;
      // interpolation from t_b to t_f
      nda::matrix<ComplexType> Ttt_fb;
      // interpolation matrix to t_f = beta^{-}
      nda::array<ComplexType, 1> T_beta_t_ff;
      // interpolation matrix to t_f = 0^{+}
      nda::array<ComplexType, 1> T_zero_t_ff;
      // tau to DLR coefficients
      nda::matrix<ComplexType> Tct_ff;
      nda::matrix<ComplexType> Tct_bb;
    };
  } // dlr
} // imag_axes_ft

#endif // ENABLE_DLR
#endif // COQUI_DLR_GENERATOR_HPP
