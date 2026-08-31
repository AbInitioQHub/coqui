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

#include "python/mean_field/mf_module.hpp"
#include "python/mean_field/mf_module.wrap.hxx"
#include "python/interaction/eri_module.hpp"
#include "python/interaction/eri_module.wrap.hxx"

namespace coqui_py {


  template<typename eri_handler_t>
  void downfold_2e(eri_handler_t &eri, const std::string &df_params,
                   std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    methods::downfolding_2e(eri.get_eri(), parser.get_root(), std::move(local_polarizabilities));
  }




  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb_with_projector_from_h5(eri_handler_t &eri, const std::string &df_params,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_coulomb_with_projector_from_h5(
      eri.get_eri(), parser.get_root(), std::move(local_polarizabilities));
  }

  template<typename eri_handler_t>
  std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb(eri_handler_t &eri, const std::string &df_params,
                const nda::array<ComplexType, 5> &projector_ksIai,
                const nda::array<long, 3> &band_window,
                const nda::array<RealType, 2> &kpts_crys,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_polarizabilities) {
    auto parser = InputParser(df_params);
    return methods::downfold_coulomb(
      eri.get_eri(), parser.get_root(),
      projector_ksIai, band_window, kpts_crys, 
      std::move(local_polarizabilities));
  }




  auto downfold_gloc_with_projector_from_h5(const Mf &mf, const std::string &df_params)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc_with_projector_from_h5(mf.get_mf(), parser.get_root());
  }

  auto downfold_gloc(const Mf &mf, const std::string &df_params,
                     const nda::array<ComplexType, 5> &projector_ksIai,
                     const nda::array<long, 3> &band_window,
                     const nda::array<RealType, 2> &kpts_crys)
  -> nda::array<ComplexType, 5> {
    auto parser = InputParser(df_params);
    return methods::downfold_gloc(mf.get_mf(), parser.get_root(), projector_ksIai, band_window, kpts_crys);
  }

  void downfold_1e(const Mf &mf, const std::string &df_params) {
    auto parser = InputParser(df_params);
    methods::downfolding_1e(mf.get_mf(), parser.get_root());
  }

  void dmft_embed_with_projector_from_h5(const Mf &mf, const std::string &embed_params,
                  std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                  std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed_with_projector_from_h5(mf.get_mf(), parser.get_root(), 
                        local_hf_potentials, local_selfenergies);
  }

  void dmft_embed(const Mf &mf, const std::string &embed_params,
                  const nda::array<ComplexType, 5> &projector_ksIai,
                  const nda::array<long, 3> &band_window,
                  const nda::array<RealType, 2> &kpts_crys,
                  std::optional<std::map<std::string, nda::array<ComplexType, 4> > > local_hf_potentials,
                  std::optional<std::map<std::string, nda::array<ComplexType, 5> > > local_selfenergies) {
    auto parser = InputParser(embed_params);
    methods::dmft_embed(mf.get_mf(), parser.get_root(),
                        projector_ksIai, band_window, kpts_crys,
                        local_hf_potentials, local_selfenergies);
  }


  // public template instantiation
  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb_with_projector_from_h5(
    ThcCoulomb&, const std::string &, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template std::tuple<nda::array<ComplexType, 4>, nda::array<ComplexType, 5>>
  downfold_coulomb(ThcCoulomb&, const std::string &,
                const nda::array<ComplexType, 5> &,
                const nda::array<long, 3> &,
                const nda::array<RealType, 2> &,
                std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

  template void downfold_2e(ThcCoulomb&, const std::string&, std::optional<std::map<std::string, nda::array<ComplexType, 5> > >);

} // coqui_py

#include "embed_module.wrap.cxx"
