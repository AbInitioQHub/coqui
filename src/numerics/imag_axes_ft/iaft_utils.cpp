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

#include "numerics/imag_axes_ft/iaft_utils.hpp"

#include <cmath>

#include "h5/h5.hpp"

#include "numerics/imag_axes_ft/IAFT.hpp"

namespace imag_axes_ft {

  IAFT read_iaft(std::string scf_file, bool print_meta_log) {
    double beta;
    double wmax;
    std::string basis;
    std::optional<std::string> prec;
    std::optional<double> eps;

    h5::file file(scf_file, 'r');
    h5::group grp(file);
    auto iaft_grp = grp.open_group("imaginary_fourier_transform");
    if (iaft_grp.has_dataset("basis")) {
      h5::h5_read(iaft_grp, "basis", basis);
    } else if (iaft_grp.has_dataset("source")) {
      h5::h5_read(iaft_grp, "source", basis);
    } else {
      utils::check(false,
        "iaft_utils.cpp::read_iaft: checkpoint is missing required IAFT basis metadata.");
    }
    h5::h5_read(iaft_grp, "beta", beta);

    if (iaft_grp.has_dataset("prec")) {
      std::string prec_value;
      h5::h5_read(iaft_grp, "prec", prec_value);
      prec = prec_value;
    }
    if (iaft_grp.has_dataset("eps")) {
      double eps_value;
      h5::h5_read(iaft_grp, "eps", eps_value);
      if (eps_value >= 0.0) eps = eps_value;
    }

    if (iaft_grp.has_dataset("wmax")) {
      h5::h5_read(iaft_grp, "wmax", wmax);
    } else {
      double lambda;
      h5::h5_read(iaft_grp, "lambda", lambda);
      wmax = lambda / beta;
    }

    auto const basis_enum = string_to_basis_enum(basis);
    if (basis_enum == dlr_basis) {
      utils::check(prec.has_value() || eps.has_value(),
        "iaft_utils.cpp::read_iaft: DLR checkpoint must provide at least one of prec or eps.");

      bool const build_with_eps = eps.has_value() && (!prec.has_value() || *prec == "custom");
      if (build_with_eps) {
        return IAFT(beta, wmax, basis_enum, *eps, print_meta_log);
      }

      utils::check(prec.has_value(),
        "iaft_utils.cpp::read_iaft: DLR checkpoint is missing required IAFT prec metadata.");
      return IAFT(beta, wmax, basis_enum, *prec, print_meta_log);
    }

    utils::check(prec.has_value(),
      "iaft_utils.cpp::read_iaft: checkpoint is missing required IAFT prec metadata.");
    return IAFT(beta, wmax, basis_enum, *prec, print_meta_log);
  }

  namespace test_utils {

    namespace {

      /**
       * Kernel for imaginary time Green's function. 
       * We uses the convention tau = [-1, 1] which maps to the physical imaginary time [0, beta]. 
       */
      double kernel_tau(double t, double om, double beta) {
        return (om >= 0.0) ?
          -std::exp(-(t + 1.0) * 0.5 * beta * om) / (1.0 + std::exp(-beta * om)) :
          -std::exp((-t + 1.0) * 0.5 * beta * om) / (1.0 + std::exp(beta * om));
      }
      
      /**
       * Kernel for imaginary frequency Green's function from the Fourier transform of kernel_tau.
       * We use the convention that n = (2k + 1) for fermions and n = 2k for bosons, where k is an integer. 
       */
      ComplexType kernel_iw(int n, double om, imag_axes_ft::stats_e stat, double beta) {
        auto iw_n = ComplexType(0.0, n * M_PI);
        if (stat == imag_axes_ft::fermion) {
          return beta / (iw_n - beta * om);
        } else {
          if (n == 0 and om == 0.0) {
            return -beta / 2.0;
          } else {
            // Use tanh(beta * om / 2) for numerical stability:
            // (1 - exp(-x)) / (1 + exp(-x)) = tanh(x / 2)
            return (beta / (iw_n - beta * om)) * std::tanh(0.5 * beta * om);
          }
        }
      }

      nda::vector<double> build_coefficients(int i, int j, int npeak) {
        auto c = nda::vector<double>(npeak);
        for (int l = 0; l < npeak; ++l) {
          c(l) = (std::sin(1000.0 * (i + 2 * j + 3 * l + 7)) + 1.0) / 2.0;
        }
        c = c / nda::sum(c);
        return c;
      }

    } // namespace

    nda::matrix<ComplexType> gfun_tau(int norb, double beta, double t, bool ph_sym) {
      int constexpr npeak = 5;

      auto g = nda::matrix<ComplexType>(norb, norb);
      g = ComplexType(0.0, 0.0);

      for (int i = 0; i < norb; ++i) {
        for (int j = i; j < norb; ++j) {
          auto c = build_coefficients(i, j, npeak);

          for (int l = 0; l < npeak; ++l) {
            auto om = std::sin(2000.0 * (3 * i + 2 * j + l + 6));
            g(i, j) += c(l) * kernel_tau(t, om, beta);
            if (ph_sym) {
              g(i, j) += c(l) * kernel_tau(t, -om, beta);
            }
          }
          if (i != j) {
            g(j, i) = g(i, j);
          }
        }
      }

      return g;
    }

    nda::matrix<ComplexType> gfun_iw(int norb, double beta, int n, imag_axes_ft::stats_e stat, bool ph_sym) {
      int constexpr npeak = 5;

      auto g = nda::matrix<ComplexType>(norb, norb);
      g = ComplexType(0.0, 0.0);

      for (int i = 0; i < norb; ++i) {
        for (int j = i; j < norb; ++j) {
          auto c = build_coefficients(i, j, npeak);

          for (int l = 0; l < npeak; ++l) {
            auto om = std::sin(2000.0 * (3 * i + 2 * j + l + 6));
            g(i, j) += c(l) * kernel_iw(n, om, stat, beta);
            if (ph_sym) {
              g(i, j) += c(l) * kernel_iw(n, -om, stat, beta);
            }
          }
          if (i != j) {
            g(j, i) = g(i, j);
          }
        }
      }

      return g;
    }

  } // namespace test_utils
} // namespace imag_axes_ft
