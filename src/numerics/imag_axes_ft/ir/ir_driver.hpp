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


#ifndef COQUI_IR_GENERATOR_HPP
#define COQUI_IR_GENERATOR_HPP

#include <unordered_map>
#include <math.h>
#include <filesystem>

#include "nda/nda.hpp"
#include "h5/h5.hpp"
#include "nda/h5.hpp"

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "IO/app_loggers.h"
#include "../iaft_enum_e.hpp"

namespace imag_axes_ft {
  namespace ir {

    struct IR {
    public:
      IR () = default;

      IR(double beta_, double wmax_, std::string prec_="high", bool print_meta_log = false):
      beta(beta_), wmax(wmax_), prec(prec_) {
        // determine _lambda
        lambda = determine_lambda(beta*wmax);
        wmax = lambda / beta;
        utils::check(lambda > 0.0, "Invalid value of lambda, {}, in imag_axes_ft::ir::ir.", lambda);
        utils::check(wmax > 0.0, "Invalid value of wmax, {}, in imag_axes_ft::ir::ir.", wmax);

        eps = prec_to_eps(prec);
        std::string prec_prefix = prec_to_prefix(prec);
        std::string filename = ir_file(lambda, prec_prefix);
        h5::file file(filename, 'r');
        h5::group grp(file);

        h5::group f_grp = grp.open_group("fermion");
        h5::h5_read(f_grp, "nt", nt_f);
        h5::h5_read(f_grp, "nw", nw_f);
        nda::h5_read(f_grp, "tau_mesh", tau_mesh_f);
        nda::h5_read(f_grp, "wn_mesh", wn_mesh_f);
        nda::h5_read(f_grp, "Ttw", Ttw_ff);
        nda::h5_read(f_grp, "Twt", Twt_ff);
        nda::h5_read(f_grp, "Ttt_bf", Ttt_bf);
        nda::h5_read(f_grp, "T_betat", T_beta_t_ff);
        nda::h5_read(f_grp, "T_zerot", T_zero_t_ff);
        nda::h5_read(f_grp, "Tct", Tct_ff);

        h5::group b_grp = grp.open_group("boson");
        h5::h5_read(b_grp, "nt", nt_b);
        h5::h5_read(b_grp, "nw", nw_b);
        nda::h5_read(b_grp, "tau_mesh", tau_mesh_b);
        nda::h5_read(b_grp, "wn_mesh", wn_mesh_b);
        nda::h5_read(b_grp, "Ttw", Ttw_bb);
        nda::h5_read(b_grp, "Twt", Twt_bb);
        nda::h5_read(b_grp, "Ttt_fb", Ttt_fb);
        nda::h5_read(f_grp, "Tct", Tct_bb);

        // Rescale IR bases
        Ttw_ff *= (std::sqrt(2)/beta);
        Twt_ff *= (beta/std::sqrt(2) );
        Ttw_bb *= (std::sqrt(2)/beta);
        Twt_bb *= (beta/std::sqrt(2) );
        Tct_ff *= std::sqrt(2.0/beta);
        Tct_bb *= std::sqrt(2.0/beta);

        if (print_meta_log) metadata_log();
      }

      IR(const IR& other) = default;
      IR(IR&& other) = default;

      IR& operator=(const IR& other) = default;
      IR& operator=(IR&& other) = default;

      ~IR() {}

      void metadata_log() const {
        std::string prec_prefix = prec_to_prefix(prec);

        app_log(1, "  Mesh details on the imaginary axis");
        app_log(1, "  ----------------------------------");
        app_log(1, "  Intermediate Representation");
        app_log(1, "  Beta                   = {} a.u.", beta);
        app_log(1, "  Frequency cutoff       = {} a.u.", wmax);
        app_log(1, "  Lambda                 = {}", lambda);
        app_log(1, "  Precision              = {}", prec_prefix);
        app_log(1, "  nt_f, nt_b, nw_f, nw_b = {}, {}, {}, {}\n", nt_f, nt_b, nw_f, nw_b);
      }

      std::string ir_file(double lmbda, std::string prec_prefix) {
        std::string source_path = INSTALL_DIR;
        std::string filename = source_path + "/data/ir/" + lambda_map[lmbda] + "." + prec_prefix + ".h5";
        if (std::filesystem::exists(filename))
          return filename;

        source_path = PROJECT_SOURCE_DIR;
        filename = source_path + "/src/numerics/imag_axes_ft/ir/data/" + lambda_map[lmbda] + "." + prec_prefix + ".h5";
        return filename;
      }

      auto construct_tau_interpolate_matrix([[maybe_unused]] const nda::MemoryArrayOfRank<1> auto &tau_mesh_out,
                    [[maybe_unused]] bool ph_sym=false) const
      -> nda::array<ComplexType, 2> {
        
        utils::check(false, "imag_axes_ft::ir::construct_tau_interpolate_matrix is not implemented yet.");

        return nda::array<ComplexType, 2>(1, 1);
      }

      auto construct_w_interpolate_matrix([[maybe_unused]] const nda::MemoryArrayOfRank<1> auto &wn_mesh_out,
                    [[maybe_unused]] imag_axes_ft::stats_e stats,
                    [[maybe_unused]] bool ph_sym=false) const
      -> nda::array<ComplexType, 2> {

        utils::check(false, "imag_axes_ft::ir::construct_w_interpolate_matrix is not implemented yet.");

        return nda::array<ComplexType, 2>(1, 1);
      }

    public:
      double beta;
      double wmax;
      std::string prec;
      double eps;
      double lambda;

      int nt_f;
      int nt_b;
      int nw_f;
      int nw_b;

      // Matsubara frequency mesh
      nda::array<long, 1> wn_mesh_f;
      nda::array<long, 1> wn_mesh_b;
      // tau mesh: [-1, 1] -> [0, beta]
      nda::array<double, 1> tau_mesh_f;
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
      // tau to IR coefficients
      nda::matrix<ComplexType> Tct_ff;
      nda::matrix<ComplexType> Tct_bb;

      inline double determine_lambda(double lbda) {
        if (lbda <= 100) {
          return 100;
        } else if (lbda > 100 and lbda <= 1000) {
          return 1000;
        } else if (lbda > 1000 and lbda <= 10000) {
          return 10000;
        } else if (lbda > 10000 and lbda <= 100000) {
          return 100000;
        } else if (lbda > 100000 and lbda <= 1000000) {
          return 1000000;
        } else {
          return -1;
        }
      }

      inline std::string prec_to_prefix(std::string const& prec_) const {
        if (prec_ == "high") {
          return "1e-15";
        } else if (prec_ == "medium") {
          return "1e-10";
        } else if (prec_ == "low") {
          return "1e-06";
        }

        utils::check(false,
          "imag_axes_ft::ir: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"",
          prec_);
        return "";
      }

      inline double prec_to_eps(std::string const& prec_) const {
        if (prec_ == "high") {
          return 1e-15;
        } else if (prec_ == "medium") {
          return 1e-10;
        } else if (prec_ == "low") {
          return 1e-06;
        }

        utils::check(false,
          "imag_axes_ft::ir: prec = {} is not acceptable. Acceptable list = \"high\", \"medium\", \"low\"",
          prec_);
        return 0.0;
      }

    private:
      std::unordered_map<double, std::string> lambda_map = {
          {100, "1e2"},
          {1000, "1e3"},
          {10000, "1e4"},
          {100000, "1e5"},
          {1000000, "1e6"}
      };

    };
  } // IR
} // imag_axes_ft

#endif //COQUI_IR_GENERATOR_HPP
