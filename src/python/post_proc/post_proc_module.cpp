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


#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "methods/pproc/pproc_drivers.hpp"
#include "methods/SCF/qp/linearized_qp.hpp"

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"

namespace coqui_py::post_proc {

  void band_interpolation(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("band_interpolation", mf.get_mf(), parser.get_root());
  }

  void spectral_interpolation(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("spectral_interpolation", mf.get_mf(), parser.get_root());
  }

  void local_dos(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("local_dos", mf.get_mf(), parser.get_root());
  }

  void unfold_bz(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("unfold_bz", mf.get_mf(), parser.get_root());
  }

  void dump_vxc(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("dump_vxc", mf.get_mf(), parser.get_root());
  }

  void dump_hartree(const Mf &mf, const std::string &params) {
    auto parser = InputParser(params);
    methods::post_processing("dump_hartree", mf.get_mf(), parser.get_root());
  }

  auto pade(nda::array<ComplexType, 2> A_iw, nda::array<ComplexType, 1> iw_mesh,
            double w_min, double w_max, long Nw, int Nfit, double eta, bool is_iw_pos_only)
  -> std::tuple<nda::array<ComplexType, 2>, nda::array<ComplexType, 1>> {
    auto w_grid = analyt_cont::AC_t::w_grid(w_min, w_max, Nw, eta);
    return std::make_tuple(
      methods::pade(std::move(A_iw), std::move(iw_mesh), w_min, w_max, Nw, eta, is_iw_pos_only, Nfit), 
      w_grid
    );
  }


  /**
   * pproc_t::linearized_qp on arrays: one fit at a fixed window, returned as the kernel's own
   * methods::lqp::result_t (Python class ResultT). E_ska is absolute (mu is in the result); 
   * status_sk (ns, nk) is 0 ok, 1 the residual gate failed, 2 = 1 - B not positive definite, 
   * and the entries of a failed point are zero. Sigma_tskab must be on the fermionic tau mesh of
   * IAFT(beta, wmax, basis, prec) and F_skab must include H0.
   */
  methods::lqp::result_t
  linearized_qp_solve(nda::array<ComplexType, 4> F_skab, nda::array<ComplexType, 5> Sigma_tskab,
                      double mu, double beta, double wmax, const std::string &basis,
                      const std::string &prec, int n_fit, int fit_order, bool symmetric_window,
                      double fit_resid_tol) {
    methods::lqp::fit_params_t p;
    p.n_fit = n_fit; p.fit_order = fit_order; p.symmetric_window = symmetric_window;
    p.fit_resid_tol = fit_resid_tol;
    return methods::pproc_t::linearized_qp(F_skab, Sigma_tskab, mu, beta, wmax, basis, prec, p);
  }

  /**
   * pproc_t::linearized_qp_ladder on arrays, returned as the kernel's methods::lqp::ladder_result_t
   * (Python class LadderResultT): the last accepted rung in `last` plus the climb's bookkeeping
   * (err_fit_sk, dHqp_sk, histories, n_accepted, mesh_limited, n_fit_mesh_max, stopped_*).
   * n_accepted == 0 means even n_fit = 2 failed the gate and `last` is empty.
   */
  methods::lqp::ladder_result_t
  linearized_qp_ladder(nda::array<ComplexType, 4> F_skab, nda::array<ComplexType, 5> Sigma_tskab,
                       double mu, double beta, double wmax, const std::string &basis,
                       const std::string &prec, int n_fit_max, double fit_resid_tol,
                       bool symmetric_window) {
    methods::lqp::fit_params_t p;
    p.symmetric_window = symmetric_window; p.fit_resid_tol = fit_resid_tol;
    return methods::pproc_t::linearized_qp_ladder(F_skab, Sigma_tskab, mu, beta, wmax, basis, prec,
                                                  p, n_fit_max);
  }

} // coqui_py

#include "post_proc_module.wrap.cxx"
