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


#ifndef COQUI_MODEL_HAMILTONIAN_READONLY_HPP
#define COQUI_MODEL_HAMILTONIAN_READONLY_HPP

#include <map>
#include <string>
#include <memory>

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include <nda/h5.hpp>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/mf_source.hpp"
#include "mean_field/model_hamiltonian/model_system.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "hamiltonian/pseudo/pseudopot_to_h5.hpp"

namespace mf {
namespace model {

/**
 * Basic interface to Model Hamiltonians in COQUI 
 * Provides access to system consistent with other backends. 
 */
class model_readonly {
public:

  static constexpr mf_source_e mf_src = model_source;
  static constexpr mf_source_e get_mf_source() { return mf_src; }
  static constexpr bool has_orbital_set() { return false; }

  // accessor functions
  auto mpi() const { return sys.mpi; }
  long nbnd() const { return sys.nbnd; }
  long nbnd_aux() const { return sys.nbnd_aux; }
  decltype(auto) kpts() { return sys.bz().kpts(); }
  decltype(auto) kpts_crystal() { return sys.bz().kpts_crys(); }
  int nkpts() const { return sys.bz().nkpts; }
  int nkpts_ibz() const { return sys.bz().nkpts_ibz; }
  decltype(auto) kp_trev() { return sys.bz().kp_trev(); }
  const model_system& get_sys() const {return sys; }
  bool has_wfc_grid() const { return false; }
  auto const& bz() const { return sys.bz(); }
  // should not be called
  int fft_grid_size() const { _abort_("fft_grid_size"); return 0; }
  int nnr() const { _abort_("nnr"); return 0; }
  decltype(auto) lattice() const { _abort_("lattice"); return sys.latt(); }
  decltype(auto) recv() const { _abort_("recv"); return sys.recv(); }
  decltype(auto) fft_grid_dim() const {
    _abort_("fft_grid_dim");
    return fft_mesh();
  }
  decltype(auto) wfc_truncated_grid() const {
    _abort_("wfc_truncated_grid");
    return std::addressof(wfc_g);
  }

public:

  model_readonly(std::shared_ptr<utils::mpi_context_t<> > mpi,
                std::string outdir, std::string prefix, 
                long n_ = -1):
    sys(std::move(mpi), outdir, prefix, n_)
  {
    print_output_info();
  }

  model_readonly(model_system const& model_sys): sys(model_sys)
  {
    print_output_info();
  }

  model_readonly(model_readonly const& other):
    sys(other.sys),
    dmat( other.dmat ) {}

  template<utils::Communicator comm_t>
  model_readonly(model_system&& model_sys): sys(std::move(model_sys) )
  {
    print_output_info();
  }

  model_readonly(model_readonly&& other):
      sys(std::move(other.sys) ),
      dmat( std::move(other.dmat) ) 
  {}

  ~model_readonly() = default; 

  model_readonly& operator=(const model_readonly& other) {
    this->sys = other.sys;
    this->dmat = other.dmat;
    return *this;
  }

  model_readonly& operator=(model_readonly&& other) {
    this->sys = std::move(other.sys);
    this->dmat = std::move(other.dmat);
    return *this;
  }

  template<typename... Args> void get_orbital(Args&&...) {  _abort_("get_orbital"); }

  template<typename... Args> void get_orbital_set(Args&&...) {  _abort_("get_orbital_set"); }

  template<typename... Args> decltype(auto) symmetry_rotation(Args&&...) const
  { 
    _abort_("symmetry_rotation");
    return std::make_tuple(false,  std::addressof(dmat.at(0)));
  }

  void set_pseudopot([[maybe_unused]] std::shared_ptr<hamilt::pseudopot> const& psp_) 
  { _abort_("set_pseudopot"); } 
  std::shared_ptr<hamilt::pseudopot> get_pseudopot() 
  {  
    _abort_("get_pseudopot"); 
    return std::shared_ptr<hamilt::pseudopot>{nullptr};
  }

  void close() {}

  private:

  // system info
  model_system sys;

  grids::truncated_g_grid wfc_g;

  nda::stack_array<int, 3> fft_mesh = {0,0,0};

  // matrices that define symmetry relations between wavefunctions at different k-points
  nda::array<int,2> sk_to_n;  
  std::vector< math::sparse::csr_matrix<ComplexType,HOST_MEMORY,int,int> > dmat;

  void print_output_info() const {
    app_log(1, "  COQUI Model Hamiltonian reader");
    app_log(1, "  ------------------------");
    app_log(1, "    - nspin: {}", sys.nspin);
    app_log(1, "    - nspin in basis: {}", sys.nspin_in_basis);
    app_log(1, "    - npol: {}", sys.npol);
    app_log(1, "    - nbnd  = {}", sys.nbnd);
    app_log(1, "    - Monkhorst-Pack mesh = ({},{},{})", sys.bz().kp_grid(0), sys.bz().kp_grid(1), sys.bz().kp_grid(2));
    app_log(1, "    - nkpts = {}", sys.bz().nkpts);
    app_log(1, "    - nkpts_ibz = {}", sys.bz().nkpts_ibz);
    app_log(1, "    - nelec = {}", sys.nelec);
  }

  void _abort_(std::string m) const {
    APP_ABORT("Error in model_hamiltonian: Calling "+m+" is not allowed."); 
  }

};

} // namespace model
} // namespace mf

#endif //COQUI_MODEL_HAMILTONIAN_READONLY_HPP
