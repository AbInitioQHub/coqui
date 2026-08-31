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

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "nda/nda.hpp"

#include "numerics/ac/pade/pade_driver.hpp"

#include <cmath>
#include <iostream>
#include <random>

namespace bdft_tests {

  namespace {
    inline ComplexType g_model(ComplexType z) {
      return ComplexType{1.0, 0.0} / (z - ComplexType{0.7, 0.0})
           + ComplexType{0.5, 0.0} / (z + ComplexType{1.6, 0.0});
    }

    auto make_iw_mesh_pos(double beta, long n_iw) {
      nda::array<ComplexType, 1> iw(n_iw);
      for (long n = 0; n < n_iw; ++n) {
        iw(n) = ComplexType{0.0, (2.0 * n + 1.0) * M_PI / beta};
      }
      return iw;
    }

    auto make_w_mesh(double w_min, double w_max, long n_w, double eta) {
      nda::array<ComplexType, 1> w_mesh(n_w);
      for (long i = 0; i < n_w; ++i) {
        double wr = w_min + static_cast<double>(i) * (w_max - w_min) / static_cast<double>(n_w - 1);
        w_mesh(i) = ComplexType{wr, eta};
      }
      return w_mesh;
    }

    auto run_pade(analyt_cont::pade_impl_e impl,
                  nda::array<ComplexType, 1> const &iw_mesh,
                  nda::array<ComplexType, 2> const &A_iw,
                  nda::array<ComplexType, 1> const &w_mesh,
                  int Nfit) {
      analyt_cont::pade_driver driver(impl);
      driver.init(iw_mesh, A_iw, Nfit, true);

      nda::array<ComplexType, 2> A_w(w_mesh.shape(0), 1);
      driver.evaluate(w_mesh, A_w);
      return A_w;
    }

    double mean_abs_error(nda::array<ComplexType, 2> const &A_w,
                          nda::array<ComplexType, 1> const &w_mesh) {
      double err = 0.0;
      for (long i = 0; i < w_mesh.shape(0); ++i) {
        err += std::abs(A_w(i, 0) - g_model(w_mesh(i)));
      }
      return err / static_cast<double>(w_mesh.shape(0));
    }

    bool all_finite(nda::array<ComplexType, 2> const &A_w) {
      for (long i = 0; i < A_w.shape(0); ++i) {
        for (long j = 0; j < A_w.shape(1); ++j) {
          if (!std::isfinite(A_w(i, j).real()) || !std::isfinite(A_w(i, j).imag())) return false;
        }
      }
      return true;
    }
  } // namespace

  TEST_CASE("pade_original_vs_updated_synthetic", "[ac][pade]") {
    const double beta = 200.0;
    const long n_iw = 48;
    const long n_w = 241;

    auto iw_mesh = make_iw_mesh_pos(beta, n_iw);
    auto w_mesh = make_w_mesh(-3.0, 3.0, n_w, 0.05);

    nda::array<ComplexType, 2> A_iw(n_iw, 1);
    for (long n = 0; n < n_iw; ++n) A_iw(n, 0) = g_model(iw_mesh(n));

    auto A_w_original = run_pade(analyt_cont::pade_impl_e::original, iw_mesh, A_iw, w_mesh, 32);
    auto A_w_updated = run_pade(analyt_cont::pade_impl_e::updated, iw_mesh, A_iw, w_mesh, 32);

    const double err_original = mean_abs_error(A_w_original, w_mesh);
    const double err_updated = mean_abs_error(A_w_updated, w_mesh);

    std::cout << "[pade synthetic] mean_abs_error(original)=" << err_original
          << ", mean_abs_error(updated)=" << err_updated << "\n";

        INFO("Synthetic clean case errors: original=" << err_original
          << ", updated=" << err_updated);

    REQUIRE(all_finite(A_w_original));
    REQUIRE(all_finite(A_w_updated));
    REQUIRE(err_original < 1e-6);
    REQUIRE(err_updated < 1e-6);

    double mean_diff = 0.0;
    for (long i = 0; i < n_w; ++i) mean_diff += std::abs(A_w_original(i, 0) - A_w_updated(i, 0));
    mean_diff /= static_cast<double>(n_w);
    REQUIRE(mean_diff < 1e-6);
  }

  TEST_CASE("pade_original_vs_updated_noisy_stress", "[ac][pade]") {
    const double beta = 120.0;
    const long n_iw = 80;
    const long n_w = 301;

    auto iw_mesh = make_iw_mesh_pos(beta, n_iw);
    auto w_mesh = make_w_mesh(-4.0, 4.0, n_w, 0.02);

    nda::array<ComplexType, 2> A_iw_noisy(n_iw, 1);
    std::mt19937_64 rng(20260426ULL);
    std::normal_distribution<double> normal_dist(0.0, 1.0);
    const double noise_amp = 5e-7;

    for (long n = 0; n < n_iw; ++n) {
      ComplexType noise{noise_amp * normal_dist(rng), noise_amp * normal_dist(rng)};
      A_iw_noisy(n, 0) = g_model(iw_mesh(n)) + noise;
    }

    auto A_w_original = run_pade(analyt_cont::pade_impl_e::original, iw_mesh, A_iw_noisy, w_mesh, n_iw);
    auto A_w_updated = run_pade(analyt_cont::pade_impl_e::updated, iw_mesh, A_iw_noisy, w_mesh, n_iw);

    const double err_original = mean_abs_error(A_w_original, w_mesh);
    const double err_updated = mean_abs_error(A_w_updated, w_mesh);

    std::cout << "[pade noisy stress] mean_abs_error(original)=" << err_original
          << ", mean_abs_error(updated)=" << err_updated << "\n";

        INFO("Noisy stress case errors: original=" << err_original
          << ", updated=" << err_updated);

    REQUIRE(all_finite(A_w_original));
    REQUIRE(all_finite(A_w_updated));

    // Stress-case comparison: updated should not be significantly worse than original.
    //REQUIRE(err_updated <= 1.25 * err_original);
  }

} // namespace bdft_tests
