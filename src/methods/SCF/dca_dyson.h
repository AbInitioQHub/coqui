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


#ifndef COQUI_DCA_DYSON_H
#define COQUI_DCA_DYSON_H

#include "mpi3/communicator.hpp"
#include "numerics/shared_array/nda.hpp"

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "methods/SCF/simple_dyson.h"
#include "methods/embedding/embed_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"

#include "utilities/mpi_context.h"
#include "methods/SCF/scf_common.hpp"

namespace methods {

namespace mpi3 = boost::mpi3;
class dca_dyson {
public:
  dca_dyson(utils::mpi_context_t<mpi3::communicator> &context, mf::MF *mf,
            imag_axes_ft::IAFT *ft, mf::MF &dca_mf, double mu_tol = 1e-9,
            std::string mu_update_alg = "bisection");

  template<typename X_t, typename Xt_t>
  void compute_eigenspectra(double mu, const X_t&_sF_skij, const Xt_t &_G_shm, const Xt_t &_Sigma_shm, nda::array<ComplexType, 4> &spectra);

  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void solve_dyson(Dm_t&, G_t&, const F_t&, const Sigma_t &, double mu);

  inline void print_timers() {}

private:
  utils::mpi_context_t<mpi3::communicator> &_context;

  mf::MF* _MF = nullptr;
  imag_axes_ft::IAFT* _FT = nullptr;
  std::shared_ptr<hamilt::pseudopot> _PSP;

  size_t _nk_tilde;
  double _mu_tol;
  std::string _mu_update_alg;

  sArray_t<Array_view_4D_t> _sH0_skij;
  sArray_t<Array_view_4D_t> _sS_skij;
  // DCA lattice data
  sArray_t<Array_view_4D_t> _sH0_lattice_sKij;
  sArray_t<Array_view_4D_t> _sS_lattice_sKij;

  utils::TimerManager _Timer;

public:
  auto mu_tol() const { return _mu_tol; }
  std::string mu_update_alg() const { return _mu_update_alg; }
  const auto &sS_skij() const {return _sS_skij;}
  const auto &sH0_skij() const {return _sH0_skij;}

  const mf::MF* MF() const {return _MF;}
  const imag_axes_ft::IAFT* FT(){return _FT;}
  hamilt::pseudopot* PSP() { return _PSP.get(); }
  const hamilt::pseudopot* PSP() const { return _PSP.get(); }
  auto &Timer() { return _Timer; }
};

}
#endif // COQUI_DCA_DYSON_H
