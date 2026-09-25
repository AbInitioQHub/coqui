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
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>

#include "h5/h5.hpp"

#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"

namespace imag_axes_ft {

  namespace {

    constexpr std::string_view grid_mismatch_explanation =
      "The imaginary-axis grid is rebuilt from the checkpoint's metadata (beta, wmax,\n"
      "prec/eps), so it reproduces the original only when the DLR/IR backend builds the same\n"
      "grid between the code that wrote the checkpoint and the code reading it now. A\n"
      "difference means the two disagree, typically because the basis library was updated in\n"
      "between. Rebuild CoQui using the consistent DLR/IR backend or regenerate the checkpoint\n"
      "with the current build.";

    /**
     * Implementation of the two compare_mesh overloads.
     *
     * @param label    - mesh name to use in the report
     * @param scf_file - checkpoint path, for the report
     * @param stored   - mesh read back from the checkpoint
     * @param rebuilt  - mesh of the reconstructed IAFT
     * @param tol      - absolute tolerance on node values; unused for integral value_t
     * @return true when the meshes agree
     */
    template<typename value_t>
    bool compare_mesh_impl(std::string_view label, std::string const& scf_file,
                           nda::array<value_t, 1> const& stored,
                           nda::array<value_t, 1> const& rebuilt,
                           [[maybe_unused]] double tol) {
      if (stored.size() != rebuilt.size()) {
        app_log(1, " {} has {} nodes in '{}',\n"
                   "but rebuilding the grid from the stored metadata gives {} nodes.\n"
                   "{}",
                label, stored.size(), scf_file, rebuilt.size(), grid_mismatch_explanation);
        return false;
      }

      for (long i = 0; i < stored.size(); ++i) {
        bool differs;
        if constexpr (std::is_integral_v<value_t>)
          differs = (stored(i) != rebuilt(i));
        else
          differs = (std::abs(stored(i) - rebuilt(i)) > tol);

        if (differs) {
          app_log(1, " {} in '{}' differs at index {}:\n"
                     "the checkpoint has {}, the rebuilt grid has {}.\n"
                     "{}",
                  label, scf_file, i, stored(i), rebuilt(i), grid_mismatch_explanation);
          return false;
        }
      }
      return true;
    }

    /**
     * Read one mesh from the checkpoint and compare it against the rebuilt one,
     * aborting on disagreement. A mesh that is not present is reported and skipped:
     * absence is never an error.
     *
     * @param parent      - group holding the mesh datasets (tau_mesh or iwn_mesh)
     * @param dataset     - "fermion" or "boson"
     * @param label       - mesh name to use in log and error messages
     * @param rebuilt     - mesh of the reconstructed IAFT
     * @param scf_file    - checkpoint path, for the messages
     * @param tol         - absolute tolerance; only meaningful for floating-point meshes
     */
    template<typename value_t>
    void check_one_mesh(h5::group const& parent, std::string const& dataset,
                        std::string_view label, nda::array<value_t, 1> const& rebuilt,
                        std::string const& scf_file, double tol = 0.0) {
      app_log(4, "  Checking for {} in '{}'.", label, scf_file);
      if (not parent.has_dataset(dataset)) {
        app_log(1, " [WARNING] {} is absent from '{}';\n"
                   "this part of the grid could not be verified.", label, scf_file);
        return;
      }

      nda::array<value_t, 1> stored;
      nda::h5_read(parent, dataset, stored);

      bool agrees;
      if constexpr (std::is_integral_v<value_t>)
        agrees = compare_mesh(label, scf_file, stored, rebuilt);
      else
        agrees = compare_mesh(label, scf_file, stored, rebuilt, tol);

      utils::check(agrees,
        "iaft_utils.cpp::validate_grid_against_checkpoint: \n"
        "{} in '{}' does not match the grid rebuilt from the checkpoint's own \n"
        "metadata; see the report above.",
        label, scf_file);

      app_log(4, "  {} matches.", label);
    }

  } // anonymous namespace

  bool compare_mesh(std::string_view label, std::string const& scf_file,
                    nda::array<double, 1> const& stored,
                    nda::array<double, 1> const& rebuilt,
                    double tol) {
    return compare_mesh_impl(label, scf_file, stored, rebuilt, tol);
  }

  bool compare_mesh(std::string_view label, std::string const& scf_file,
                    nda::array<long, 1> const& stored,
                    nda::array<long, 1> const& rebuilt) {
    return compare_mesh_impl<long>(label, scf_file, stored, rebuilt, 0.0);
  }

  void validate_grid_against_checkpoint(h5::group const& iaft_grp, IAFT const& ft,
                                        std::string const& scf_file) {
    constexpr double tau_tol = 1e-10;

    bool const has_tau = iaft_grp.has_subgroup("tau_mesh");
    bool const has_iwn = iaft_grp.has_subgroup("iwn_mesh");

    if (not has_tau and not has_iwn) {
      app_log(1, " [WARNING] '{}' stores no tau_mesh or iwn_mesh since it was written \n"
                 "by an older version of CoQui, so its imaginary-axis grid could not be \n"
                 "verified against the one rebuilt from its metadata.", scf_file);
      return;
    }

    app_log(2, "Verifying the imaginary-axis grid of '{}' against the rebuilt IAFT.", scf_file);

    if (has_tau) {
      auto tau_grp = iaft_grp.open_group("tau_mesh");
      check_one_mesh<double>(tau_grp, "fermion", "tau_mesh (fermion)",
                             ft.tau_mesh_f(), scf_file, tau_tol);
      check_one_mesh<double>(tau_grp, "boson", "tau_mesh (boson)",
                             ft.tau_mesh_b(), scf_file, tau_tol);
    } else {
      app_log(1, " [WARNING] '{}' stores no tau_mesh;\n"
                 "the imaginary-time grid could not be verified.", scf_file);
    }

    if (has_iwn) {
      auto iwn_grp = iaft_grp.open_group("iwn_mesh");
      check_one_mesh<long>(iwn_grp, "fermion", "iwn_mesh (fermion)",
                           ft.wn_mesh_f(), scf_file);
      check_one_mesh<long>(iwn_grp, "boson", "iwn_mesh (boson)",
                           ft.wn_mesh_b(), scf_file);
    } else {
      app_log(1, " [WARNING] '{}' stores no iwn_mesh; the Matsubara grid could not be verified.", scf_file);
    }

    app_log(2, " Imaginary-axis grid of '{}' verified.", scf_file);
  }

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
    auto build_iaft = [&]() -> IAFT {
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
    };

    // Rebuild the grid from the checkpoint's metadata.
    auto ft = build_iaft();
    // Validate that the rebuilt grid matches the one that the simulated data lives on. 
    validate_grid_against_checkpoint(iaft_grp, ft, scf_file);
    return ft;
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
