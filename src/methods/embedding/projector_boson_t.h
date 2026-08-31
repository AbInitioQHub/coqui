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


#ifndef BEYONDDFT_PROJECTOR_BOSON_T_H
#define BEYONDDFT_PROJECTOR_BOSON_T_H

#include "configuration.hpp"
#include "utilities/mpi_context.h"

#include "numerics/shared_array/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"
#include "methods/embedding/projector_t.h"

namespace methods {

  namespace mpi3 = boost::mpi3;

  class projector_boson_t {
  public:
    template<nda::Array Array_base_t>
    using sArray_t = math::shm::shared_array<Array_base_t>;
    using Array_view_5D_t = nda::array_view<ComplexType, 5>;
    template<typename comm_t>
    using mpi_context_t = utils::mpi_context_t<comm_t>;

  public:
    projector_boson_t(mf::MF &MF, std::string C_file, bool translate_home_cell=false):
        _proj_fermi(MF, C_file, translate_home_cell), _MF(std::addressof(MF)) {}

    projector_boson_t(mf::MF &MF,
                      const nda::array<ComplexType, 5> &C_ksIai,
                      const nda::array<long, 3> &band_window,
                      const nda::array<RealType, 2> &kpts_crys,
                      bool translate_home_cell=false):
        _proj_fermi(MF, C_ksIai, band_window, kpts_crys, translate_home_cell),
        _MF(std::addressof(MF)) {}

    projector_boson_t(projector_boson_t const&) = default;
    projector_boson_t(projector_boson_t &&) = default;
    projector_boson_t& operator=(projector_boson_t const& other) = default;
    projector_boson_t& operator=(projector_boson_t && other) = default;

    ~projector_boson_t() = default;

    auto calc_bosonic_projector(THC_ERI auto &thc) const
    -> sArray_t<Array_view_5D_t>;

    auto calc_bosonic_projector_symm(THC_ERI auto &thc) const
    -> sArray_t<Array_view_5D_t>;

  private:
    projector_t _proj_fermi;
    mf::MF* _MF = nullptr;

  public:
    const auto& proj_fermi() const { return _proj_fermi; }
    mf::MF* MF() { return _MF; }
    auto C_skIai() const { return _proj_fermi.C_skIai(); }
    const auto& W_rng() const { return _proj_fermi.W_rng(); }
    long nImps() const { return _proj_fermi.nImps(); }
    long nImpOrbs() const { return _proj_fermi.nImpOrbs(); }
    long nOrbs_W() const { return _proj_fermi.nOrbs_W(); }
    std::string C_file() const { return _proj_fermi.C_file(); }

  }; // projector_boson_t

} // methods

#endif //BEYONDDFT_PROJECTOR_BOSON_T_H
