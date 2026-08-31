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


#ifndef MF_MODULE_HPP
#define MF_MODULE_HPP

#include "python/utils/mpi_handler.hpp"
#include "python/utils/mpi_handler.wrap.hxx"

#include "IO/ptree/InputParser.hpp"
#include "mean_field/mf_utils.hpp"

namespace coqui_py {

  C2PY_IGNORE
  inline decltype(auto) make_MF(MpiHandler &mpi, std::string mf_params, const std::string &mf_type) {
    auto parser = InputParser(mf_params);
    return std::make_shared<mf::MF>(mf::make_MF(mpi.get_mpi(), parser.get_root(), mf_type));
  }

  /**
   * @brief Mf class
   *
   * The Mf class encapsulates the state of a mean-field inputs to CoQuí.
   * This class is read-only and is responsible for accessing the input data, including
   *
   *   1. Metadata of the simulated system, such as the number of bands, spins, and k-points.
   *   2. Single-particle basis functions whom CoQuí uses to construct a many-body Hamiltonian in CoQuí.
   */
  class Mf {
  public:
    Mf(MpiHandler &mpi, std::string mf_params, const std::string &mf_type):
    _mf(make_MF(mpi, mf_params, mf_type)) {}
    C2PY_IGNORE
    Mf(std::shared_ptr<mf::MF> mf): _mf(std::move(mf)) {}

    ~Mf() = default;
    Mf(Mf const&) = default;
    Mf(Mf &&) = default;
    Mf& operator=(Mf const&) = default;
    Mf& operator=(Mf &&) = default;

    bool operator==(const Mf& other) const {
      return _mf == other._mf;
    }

    auto mf_type() const { return mf::mf_source_enum_to_string(_mf->mf_type()); }
    auto outdir() const  { return _mf->outdir(); }
    auto prefix() const { return _mf->prefix(); }

    auto nelec() const { return _mf->nelec(); }
    auto nbnd() const { return _mf->nbnd(); }
    auto nspin() const { return _mf->nspin(); }
    auto npol() const { return _mf->npol(); }

    /* FFT grid */
    auto ecutrho () const { return _mf->ecutrho(); }
    auto fft_grid() const { return _mf->fft_grid_dim(); }
    auto ecutwfc () const { return _mf->wfc_truncated_grid()->ecut(); }
    auto fft_grid_wfc() const { return _mf->wfc_truncated_grid()->mesh(); }

    /* k-points info*/
    auto kp_grid() const { return _mf->kp_grid(); }
    auto nkpts() const { return _mf->nkpts(); }
    auto kpts() const { return _mf->kpts(); }
    auto k_weights() const { return _mf->k_weight(); }
    auto nkpts_ibz() const { return _mf->nkpts_ibz(); }
    auto kpts_ibz() const { return _mf->kpts_ibz(); }
    auto nqpts() const { return _mf->nqpts(); }
    auto qpts() const { return _mf->Qpts(); }
    auto nqpts_ibz() const { return _mf->nqpts_ibz(); }
    auto qpts_ibz() const { return _mf->Qpts_ibz(); }

    auto nuclear_energy() const { return _mf->nuclear_energy(); }

    // create a new MpiHandler from mf's mpi context
    auto mpi() const { return MpiHandler(_mf->mpi()); }

    C2PY_IGNORE
    auto get_mf() const { return _mf; }

    friend std::ostream& operator<<(std::ostream& out, const Mf& handler) {
      out << "CoQuí mean-field state\n"
          << "----------------------\n"
          << "  Type                : " << handler.mf_type() << '\n'
          << "  Prefix              : " << handler.prefix() << '\n'
          << "  Output dir          : " << handler.outdir() << '\n'
          << "  Number of electrons (nelec): " << handler.nelec() << '\n'
          << "  Bands (nbnd)        : " << handler.nbnd() << '\n'
          << "  Spins (nspin)       : " << handler.nspin() << '\n'
          << "  Polarization (npol) : " << handler.npol() << '\n'
          << "  Monkhorst-Pack grid : (" << handler.kp_grid()(0) << ", "
          << handler.kp_grid()(1) << ", " << handler.kp_grid()(2) << ")\n"
          << "  K-points            : " << handler.nkpts() << " total, "
          << handler.nkpts_ibz() << " in IBZ\n"
          << "  Q-points            : " << handler.nqpts() << " total, "
          << handler.nqpts_ibz() << " in IBZ\n"
          << "  Kinetic energy cutoff (ecutrho): " << handler.ecutrho() << " a.u.\n"
          << "  FFT grid            : (" << handler.fft_grid()(0) << ", "
          << handler.fft_grid()(1) << ", " << handler.fft_grid()(2) << ")";
      return out;
    }

  private:
    std::shared_ptr<mf::MF> _mf;

  }; // Mf

} // coqui_py

#endif
