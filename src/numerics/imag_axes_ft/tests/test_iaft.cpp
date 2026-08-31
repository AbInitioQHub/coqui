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


#undef NDEBUG

#include "catch2/catch.hpp"
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include <cmath>

#include "utilities/test_common.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace bdft_tests {

  using utils::ARRAY_EQUAL;
  template<int N>
  using shape_t = std::array<long,N>;

  TEST_CASE("iaft_ir_read", "[iaft][ir]") {
    double beta = 1000;
    double wmax = 12.0;
    {
      imag_axes_ft::ir::IR myir(beta, wmax);
      imag_axes_ft::IAFT iaft(myir);
    }
    imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::ir_basis, "high");

    REQUIRE(iaft.beta() == beta);
    REQUIRE(iaft.nt_f() == 137);
    REQUIRE(iaft.nw_f() == 138);
    REQUIRE(iaft.nt_b() == 137);
    REQUIRE(iaft.nw_b() == 137);

    REQUIRE(iaft.Ttw_ff().shape() == shape_t<2>{iaft.nt_f(), iaft.nw_f()});
    REQUIRE(iaft.Twt_ff().shape() == shape_t<2>{iaft.nw_f(), iaft.nt_f()});
    REQUIRE(iaft.Ttt_bf().shape() == shape_t<2>{iaft.nt_b(), iaft.nt_f()});
    REQUIRE(iaft.Ttw_bb().shape() == shape_t<2>{iaft.nt_b(), iaft.nw_b()});
    REQUIRE(iaft.Twt_bb().shape() == shape_t<2>{iaft.nw_b(), iaft.nt_b()});
    REQUIRE(iaft.Ttt_fb().shape() == shape_t<2>{iaft.nt_f(), iaft.nt_b()});
    REQUIRE(iaft.T_beta_t_ff().shape() == shape_t<1>{iaft.nt_f()});

    auto eye1 = iaft.Ttw_ff() * iaft.Twt_ff();
    auto eye2 = iaft.Ttt_bf() * iaft.Ttt_fb();
    ARRAY_EQUAL(eye1, nda::eye<ComplexType>(iaft.nt_f()), 1e-10);
    ARRAY_EQUAL(eye2, nda::eye<RealType>(iaft.nt_b()), 1e-10);
  }

  TEST_CASE("iaft_ir_ft", "[iaft][ir]") {
    decltype(nda::range::all) all;

    std::string source_path = std::string(PROJECT_SOURCE_DIR)+"/tests/unit_test_files/pyscf/si_kp222_krhf/";

    auto test_iaft = [&](double beta, double wmax, std::string prec, double tol) {

      imag_axes_ft::IAFT myft(beta, wmax, imag_axes_ft::ir_basis, prec, true);

      std::string filename = source_path+"/gw_Gw_Gt_beta"+std::to_string(int(beta))+"_wmax"+std::format("{:.1f}", wmax)+"_"+prec+".h5";
      h5::file file(filename, 'r');
      h5::group grp(file);

      nda::array<std::complex<double>, 5> G_tskij_ref;
      nda::array<std::complex<double>, 5> G_wskij_ref;
      nda::array<std::complex<double>, 4> Dm_skij_ref;
      nda::h5_read(grp, "Gt", G_tskij_ref);
      nda::h5_read(grp, "Gw", G_wskij_ref);
      nda::h5_read(grp, "Dm", Dm_skij_ref);

      size_t nts   = G_tskij_ref.shape(0);
      size_t nw    = G_wskij_ref.shape(0);
      size_t ns    = G_tskij_ref.shape(1);
      size_t nkpts = G_tskij_ref.shape(2);
      size_t nbnd  = G_tskij_ref.shape(3);
      // Fourier transform between tau and iwn
      {
        nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
        nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
        myft.tau_to_w(G_tskij_ref, G_wskij, imag_axes_ft::fermion);
        myft.w_to_tau(G_wskij_ref, G_tskij, imag_axes_ft::fermion);

        ARRAY_EQUAL(G_tskij, G_tskij_ref, tol);
        ARRAY_EQUAL(G_wskij, G_wskij_ref, tol);
      }
      // tau to a specific w
      {
        nda::array<std::complex<double>, 5> G_wskij(nw, ns, nkpts, nbnd, nbnd);
        nda::array<std::complex<double>, 4> G_skij(ns, nkpts, nbnd, nbnd);
        for (size_t n = 0; n < nw; ++n) {
          nda::array_view<std::complex<double>, 4> Gw_skij({ns, nkpts, nbnd, nbnd},
                                                           G_wskij.data() + n*ns*nkpts*nbnd*nbnd);
          myft.tau_to_w(G_tskij_ref, G_skij, imag_axes_ft::fermion, n);
          Gw_skij = G_skij;
        }
        ARRAY_EQUAL(G_wskij, G_wskij_ref, tol);
      }
      // Partial Fourier transform
      {
        nda::array<std::complex<double>, 5> G_tskij(nts, ns, nkpts, nbnd, nbnd);
        for (size_t n = 0; n < nw; ++n) {
          auto Gw_skij = G_wskij_ref(n, all, all, all, all);
          myft.w_to_tau_partial(Gw_skij, G_tskij, imag_axes_ft::fermion, n);
        }
        ARRAY_EQUAL(G_tskij, G_tskij_ref, tol);
      }
      // tau = beta^{-} via the interpolation at sparse sampling nodes
      {
        nda::array<std::complex<double>, 4> Dm_skij(ns, nkpts, nbnd, nbnd);
        myft.tau_to_beta(G_tskij_ref, Dm_skij);
        Dm_skij *= -1.0;
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);
      }
    };

    double beta = 1000;
    double wmax = 1.2;
    SECTION("prec_high") {
      test_iaft(beta, wmax, "high", 1e-11);
    }
    SECTION("prec_medium") {
      test_iaft(beta, wmax, "medium", 1e-9);
    }
  }

  TEST_CASE("iaft_gfun_tau_w_roundtrip", "[iaft][ir][dlr]") {
    auto test_gfun_roundtrip = [&](imag_axes_ft::basis_e basis, double tol, imag_axes_ft::stats_e stat, bool ph_sym = false) {
      double beta = 1000.0;
      double wmax = 1.0;
      int norb = 2;

      imag_axes_ft::IAFT myft(beta, wmax, basis, "high", true);
      auto wn_mesh = (stat == imag_axes_ft::fermion)? myft.wn_mesh_f() : myft.wn_mesh_b();
      auto tau_mesh = (stat == imag_axes_ft::fermion)? myft.tau_mesh_f() : myft.tau_mesh_b();
      
      auto G_t_ref = imag_axes_ft::test_utils::build_g_tau(tau_mesh, beta, norb, ph_sym);
      auto G_w_ref = imag_axes_ft::test_utils::build_g_iw(wn_mesh, beta, stat, norb, ph_sym);

      // Verify tau -> iw recovers G(iw)
      {
        nda::array<ComplexType, 3> G_w(G_w_ref.shape(0), norb, norb);
        if (ph_sym) {
          myft.tau_to_w_PHsym(G_t_ref, G_w);
        } else {
          myft.tau_to_w(G_t_ref, G_w, stat);
        }
        ARRAY_EQUAL(G_w, G_w_ref, tol);
      }

      // Verify iw -> tau recovers G(t)
      {
        nda::array<ComplexType, 3> G_t(G_t_ref.shape(0), norb, norb);
        if (ph_sym) {
          myft.w_to_tau_PHsym(G_w_ref, G_t);
        } else {
          myft.w_to_tau(G_w_ref, G_t, stat);
        }
        ARRAY_EQUAL(G_t, G_t_ref, tol);
      }

      // Verify tau -> tau = beta^- and 0^+
      if (stat == imag_axes_ft::fermion and !ph_sym) {
        nda::array<ComplexType, 2> Dm_skij(norb, norb);
        myft.tau_to_beta(G_t_ref, Dm_skij);
        auto Dm_skij_ref = imag_axes_ft::test_utils::gfun_tau(norb, beta, 1.0);
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);

        myft.tau_to_zero(G_t_ref, Dm_skij);
        Dm_skij_ref() = imag_axes_ft::test_utils::gfun_tau(norb, beta, -1.0);
        ARRAY_EQUAL(Dm_skij, Dm_skij_ref, tol);
      }
    };

    SECTION("ir_backend") {
      test_gfun_roundtrip(imag_axes_ft::ir_basis, 1e-10, imag_axes_ft::fermion);
      test_gfun_roundtrip(imag_axes_ft::ir_basis, 1e-10, imag_axes_ft::boson);
      test_gfun_roundtrip(imag_axes_ft::ir_basis, 1e-10, imag_axes_ft::boson, true);
    }

#ifdef ENABLE_DLR
    SECTION("dlr_backend") {
      test_gfun_roundtrip(imag_axes_ft::dlr_basis, 1e-10, imag_axes_ft::fermion);
      test_gfun_roundtrip(imag_axes_ft::dlr_basis, 1e-10, imag_axes_ft::boson);
      test_gfun_roundtrip(imag_axes_ft::dlr_basis, 1e-10, imag_axes_ft::boson, true);
    }
#endif
  }

  TEST_CASE("iaft_accuracy_parameter_rules", "[iaft][ir][dlr]") {
    double beta = 100.0;
    double wmax = 1.2;

#ifdef ENABLE_DLR
    SECTION("dlr_accepts_prec") {
      imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::dlr_basis, "medium", false);
      REQUIRE(iaft.prec() == "medium");
      REQUIRE(iaft.eps() == Approx(1e-10));
    }

    SECTION("dlr_accepts_eps") {
      imag_axes_ft::IAFT iaft(beta, wmax, imag_axes_ft::dlr_basis, 1e-12, false);
      REQUIRE(iaft.prec() == "custom");
      REQUIRE(iaft.eps() == Approx(1e-12));
    }

    SECTION("dlr_prefers_prec_when_both_and_prec_not_custom") {
      ptree pt;
      pt.put("beta", beta);
      pt.put("wmax", wmax);
      pt.put("iaft_basis", "dlr");
      pt.put("iaft_prec", "medium");
      pt.put("iaft_eps", 1e-12);

      imag_axes_ft::IAFT iaft(pt, false);
      REQUIRE(iaft.prec() == "medium");
      REQUIRE(iaft.eps() == Approx(1e-10));
    }

    SECTION("dlr_prefers_eps_when_both_and_prec_custom") {
      // in other words, "custom" is redundant when eps is provided, but we allow it for user clarity
      ptree pt;
      pt.put("beta", beta);
      pt.put("wmax", wmax);
      pt.put("iaft_basis", "dlr");
      pt.put("iaft_prec", "custom");
      pt.put("iaft_eps", 1e-12);

      imag_axes_ft::IAFT iaft(pt, false);
      REQUIRE(iaft.prec() == "custom");
      REQUIRE(iaft.eps() == Approx(1e-12));
    }
#endif
  }

  TEST_CASE("iaft_wmax_default_and_override", "[iaft][ir][dlr]") {
    double beta = 200.0;

    SECTION("use default_wmax_when_missing") {
      ptree pt;
      pt.put("beta", beta);
      pt.put("iaft.basis", "ir");
      pt.put("iaft.prec", "high");

      imag_axes_ft::IAFT iaft_default_1(pt, false, 0.5);
      imag_axes_ft::IAFT iaft_default_2(pt, false, 3.0);

      REQUIRE(iaft_default_1.wmax() == Approx(imag_axes_ft::IAFT(beta, 0.5, imag_axes_ft::ir_basis, "high", false).wmax()));
      REQUIRE(iaft_default_2.wmax() == Approx(imag_axes_ft::IAFT(beta, 3.0, imag_axes_ft::ir_basis, "high", false).wmax()));
      REQUIRE(iaft_default_1.wmax() != Approx(iaft_default_2.wmax()));
    }

    SECTION("explicit_wmax_over_default_values") {
      ptree pt;
      pt.put("beta", beta);
      pt.put("iaft.basis", "ir");
      pt.put("iaft.prec", "high");
      pt.put("iaft.wmax", 1.2);

      imag_axes_ft::IAFT iaft_default_1(pt, false, 0.5);
      imag_axes_ft::IAFT iaft_default_2(pt, false, 3.0);

      auto iaft_ref = imag_axes_ft::IAFT(beta, 1.2, imag_axes_ft::ir_basis, "high", false);
      REQUIRE(iaft_default_1.wmax() == Approx(iaft_ref.wmax()));
      REQUIRE(iaft_default_2.wmax() == Approx(iaft_ref.wmax()));
    }

  }

} // bdft_tests
