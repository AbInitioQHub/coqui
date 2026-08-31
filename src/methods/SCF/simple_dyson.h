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


#ifndef COQUI_SIMPLE_DYSON_H
#define COQUI_SIMPLE_DYSON_H

#include "utilities/Timer.hpp"
#include "IO/app_loggers.h"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/distributed_orbital_readers.hpp"
#include "hamiltonian/one_body_hamiltonian.hpp"
#include "hamiltonian/pseudo/pseudopot.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_common.hpp"
#include "methods/tools/chkpt_utils.h"

namespace methods {

/*
 * MAM: Consider creating an object to represent overlap matrices. If the basis is orthonormal,
 * nothing is stored, otherwise the overlap matrix is stored in shared memory. 
 * Subsequent routines can poll the object to know if the basis is orthonormal or not,
 * and act accordingly
 */

// CNY: This class is currently responsible for storing "H0" and "ovlp" matrices.
//      Is this the right place?

/**
 * @class simple_dyson
 * @brief Handler for solving the Dyson equation
 *
 * This class is responsible for solving the Dyson equation for a specified physical system.
 * The system, including nbnd, nkpt, H0 etc, is defined upon construction through a mean-field
 * object. The Dyson equation for computing the Green's function 'G_tau' can be solved by calling:
 *
 *     simple_dyson.solve_dyson(G_tau, F, Sigma_tau, mu);
 *
 * where `F` and `Sigma_tau` represent the static and dynamic components of the self-energy,
 * respectively, and `mu` is the chemical potential.
 */
class simple_dyson {
public:
  simple_dyson(mf::MF* MF, imag_axes_ft::IAFT* FT,
               double mu_tol = 1e-9,
               std::string mu_update_alg = "bisection"):
      _MF(MF), _context(MF->mpi()), _FT(FT), _PSP(hamilt::make_pseudopot(*_MF)),
      _mu_tol(mu_tol), _mu_update_alg(mu_update_alg), _nts(_FT->nt_f()), _nw(_FT->nw_f()),
      _ns(_MF->nspin()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()), 
      _nbnd(_MF->nbnd()),
      _sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _sS_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _Timer() {

    hamilt::set_H0(*_MF, _PSP.get(), _sH0_skij);
    hamilt::set_ovlp(*_MF, _sS_skij);
    if (_context->node_comm.root()) {
      hermitize_in_tau(_sH0_skij.local());
      hermitize_in_tau(_sS_skij.local());
    }

    for( auto& v: {"DYSON", "SIGMA_TAU_TO_W", "DYSON_LOOP", "REDISTRIBUTE", "DYSON_GATHER"} ) {
      _Timer.add(v);
    }
    _context->comm.barrier();
  }

  simple_dyson(mf::MF* MF, imag_axes_ft::IAFT* FT,
               std::string H0_S_chkpt,
               double mu_tol = 1e-9,
               std::string mu_update_alg = "bisection"):
      _MF(MF), _context(_MF->mpi()),
      _FT(FT), _mu_tol(mu_tol), _mu_update_alg(mu_update_alg),
      _nts(_FT->nt_f()), _nw(_FT->nw_f()),
      _ns(_MF->nspin()), _nkpts(_MF->nkpts()), _nkpts_ibz(_MF->nkpts_ibz()),
      _nbnd(_MF->nbnd()),
      _sH0_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _sS_skij(math::shm::make_shared_array<Array_view_4D_t>(*_context, {_ns, _nkpts_ibz, _nbnd, _nbnd})),
      _Timer() {

    chkpt::read_H0(_context->node_comm, H0_S_chkpt, _sH0_skij);
    chkpt::read_ovlp(_context->node_comm, H0_S_chkpt, _sS_skij);
    _context->comm.barrier();

    for( auto& v: {"DYSON", "SIGMA_TAU_TO_W", "DYSON_LOOP", "REDISTRIBUTE", "DYSON_GATHER"} ) {
      _Timer.add(v);
    }
    _context->comm.barrier();
  }

  simple_dyson(simple_dyson const&) = default;
  simple_dyson(simple_dyson &&) = default;
  simple_dyson & operator=(const simple_dyson &) = default;
  simple_dyson & operator=(simple_dyson &&) = default;

  ~simple_dyson(){}

  /**
   * Solve Dyson's equation and update _G_tskij_view for current _mu, _F, and _Sigma
   */
  template<typename G_t, typename F_t, typename Sigma_t>
  void solve_dyson(G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu);
  template<typename Dm_t, typename G_t, typename F_t, typename Sigma_t>
  void solve_dyson(Dm_t&_sDm_skij, G_t&_G_shm, const F_t&_sF_skij, const Sigma_t &_Sigma_shm, double mu);

  /**
   * Compute the eigenvalues of _F + _Sigma(iwn)
   * @param spectra - [OUTPUT] eigenvalues, (nw, ns, nkpts_ibz, nbnd)
   */
  template<typename X_t, typename Xt_t>
  void compute_eigenspectra(const X_t&_sF_skij, const Xt_t &_Sigma_shm, nda::array<ComplexType, 4> &spectra);

  inline void print_timers() {
    app_log(2, "\n  DYSON timers");
    app_log(2, "  ------------");
    app_log(2, "    Dyson eqn:                      {0:.3f} sec", _Timer.elapsed("DYSON"));
    app_log(2, "      - Sigma(t)->Sigma(w):         {0:.3f} sec", _Timer.elapsed("SIGMA_TAU_TO_W"));
    app_log(2, "      - Dyson loop:                 {0:.3f} sec", _Timer.elapsed("DYSON_LOOP"));
    app_log(2, "      - Redistribute                {0:.3f} sec", _Timer.elapsed("REDISTRIBUTE"));
    app_log(2, "      - Gather:                     {0:.3f} sec\n", _Timer.elapsed("DYSON_GATHER"));
  }

private:
  mf::MF* _MF = nullptr;
  std::shared_ptr<utils::mpi_context_t<mpi3::communicator>> _context;
  imag_axes_ft::IAFT* _FT = nullptr;
  std::shared_ptr<hamilt::pseudopot> _PSP;
  double _mu_tol = 1e-9;
  std::string _mu_update_alg = "bisection";

  int _nts;
  int _nw;
  int _ns;
  int _nkpts;
  int _nkpts_ibz;
  int _nbnd;

  sArray_t<Array_view_4D_t> _sH0_skij;
  sArray_t<Array_view_4D_t> _sS_skij;

  utils::TimerManager _Timer;

public:
  const mf::MF* MF() const { return _MF; }
  auto& mpi() const { return _context; }
  imag_axes_ft::IAFT* FT() { return _FT; }
  hamilt::pseudopot* PSP() { return _PSP.get(); }
  const hamilt::pseudopot* PSP() const { return _PSP.get(); }
  double mu_tol() const { return _mu_tol; }
  std::string mu_update_alg() const { return _mu_update_alg; }
  auto H0() const& { return _sH0_skij.local(); }
  const auto &sS_skij() const {return _sS_skij;}
  const auto &sH0_skij() const {return _sH0_skij;}
  auto &Timer() { return _Timer; }
};

}


#endif // COQUI_SIMPLE_DYSON_H
