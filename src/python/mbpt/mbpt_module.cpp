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
#include "methods/MBPT_drivers.h"

#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {

  template<typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params, eri_handler &h_int,
            const nda::array<ComplexType, 5> &C_ksIai,
            const nda::array<long, 3> &band_window,
            const nda::array<RealType, 2> &kpts_crys,
            std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root(),
                  C_ksIai, band_window, kpts_crys,
                  std::move(local_polarizabilities));
  }
  template void mbpt(const std::string&, const std::string&, ThcCoulomb&,
                     const nda::array<ComplexType, 5>&,
                     const nda::array<long, 3>&,
                     const nda::array<RealType, 2>&,
                     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);
  template void mbpt(const std::string&, const std::string&, CholCoulomb&,
                     const nda::array<ComplexType, 5>&,
                     const nda::array<long, 3>&,
                     const nda::array<RealType, 2>&,
                     std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);


  // Pure MBPT interface without C_ksIai, band_window, kpts_crys
  template<typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params, eri_handler &h_int) {
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

  template void mbpt(const std::string&, const std::string&, ThcCoulomb&);
  template void mbpt(const std::string&, const std::string&, CholCoulomb&);

  template<typename hf_eri_handler, typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params,
            eri_handler &h_int, hf_eri_handler &h_int_hf) {
    utils::check(h_int.get_mpi() == h_int_hf.get_mpi(),
                 "mbpt: h_int and h_int_hf must be on the same MPI communicator.");
    utils::check(h_int.get_mf() == h_int_hf.get_mf(),
                 "mbpt: h_int and h_int_hf must be on the same mean-field object.");
    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int_hf.get_eri(), h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

#define MBPT_INST(CORR, HF) \
template void mbpt(const std::string&, const std::string&, CORR&, HF&);

// All combinations of thc/chol for 2 eri slots
MBPT_INST(ThcCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb)

#undef MBPT_INST


  template<typename hartree_eri_handler, typename exchange_eri_handler, typename eri_handler>
  void mbpt(const std::string &solver_type, const std::string &mbpt_params,
            eri_handler &h_int, hartree_eri_handler &h_int_hartree, exchange_eri_handler &h_int_exchange) {
    utils::check(h_int.get_mpi() == h_int_hartree.get_mpi() and h_int.get_mpi() == h_int_exchange.get_mpi(),
                 "mbpt: h_int, h_int_hartree and h_int_exchange must be on the same MPI communicator.");
    utils::check(h_int.get_mf() == h_int_hartree.get_mf() and h_int.get_mf() == h_int_exchange.get_mf(),
                 "mbpt: h_int, h_int_hartree and h_int_exchange must be on the same mean-field object.");

    auto parser = InputParser(mbpt_params);
    methods::mb_eri_t mb_eri(h_int_hartree.get_eri(), h_int_exchange.get_eri(), h_int.get_eri());
    methods::mbpt(solver_type, mb_eri, parser.get_root());
  }

#define MBPT_INST(CORR, HARTREE, EXCHANGE) \
template void mbpt(const std::string&, const std::string&, CORR&, HARTREE&, EXCHANGE&);

// All combinations of thc/chol for 3 eri slots
MBPT_INST(ThcCoulomb, ThcCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, ThcCoulomb, CholCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb, ThcCoulomb)
MBPT_INST(ThcCoulomb, CholCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, ThcCoulomb, CholCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb, ThcCoulomb)
MBPT_INST(CholCoulomb, CholCoulomb, CholCoulomb)

#undef MBPT_INST

} // coqui_py

#include "mbpt_module.wrap.cxx"
