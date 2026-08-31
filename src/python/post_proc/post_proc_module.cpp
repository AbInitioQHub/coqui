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

} // coqui_py

#include "post_proc_module.wrap.cxx"
