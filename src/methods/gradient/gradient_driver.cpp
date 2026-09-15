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


#include "gradient_driver.h"

#include <format>
#include <iostream>

#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/mb_eri_context.h"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/gradient/gradient_common.h"
#include "methods/HF/hf_gradient_t.h"
#include "methods/SCF/simple_dyson.h"


namespace methods
{

template<typename dyson_type, typename eri_t, typename corr_solver_t>
void evaluate_gradient(MBState &mb_state, dyson_type &dyson, eri_t &mb_eri_t, const imag_axes_ft::IAFT &FT,
                       solvers::mb_solver_t<corr_solver_t> mb_solver, const std::string &solver_type,
                       const std::string &input_grp, int input_iter)
{
  utils::TimerManager Timer;
  auto mpi = mb_eri_t.corr_eri->get().mpi();
  auto mf = mb_eri_t.corr_eri->get().MF();
  // http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20gradient
  app_log(1, "\n"
             "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌─┐┬─┐┌─┐┌┬┐┬┌─┐┌┐┌┌┬┐\n"
             "║  ║ ║║═╬╗║ ║║  │ ┬├┬┘├─┤ │││├┤ │││ │ \n"
             "╚═╝╚═╝╚═╝╚╚═╝╩  └─┘┴└─┴ ┴─┴┘┴└─┘┘└┘ ┴ \n");
  app_log(1, "  Checkpoint HDF5          = {}", mb_state.coqui_prefix + ".mbpt.h5");
  app_log(1, "    - H5 group             = {}", input_grp);
  app_log(1, "    - Iteration            = {}", input_iter);
  app_log(1, "  - Total number processors:         {}", mpi->comm.size());
  app_log(1, "  Number of processors     = {} cores per node, {} nodes\n",
          mpi->comm.size(), mpi->internode_comm.size());

  if (input_iter == -1) {
    h5::file file(mb_state.coqui_prefix + ".mbpt.h5", 'r');
    auto scf_grp = h5::group(file).open_group(input_grp);
    h5::h5_read(scf_grp, "final_iter", input_iter);
  }

  Timer.start("GRAD_TOTAL");

  // Initialize MBState
  mb_state.sF_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sDm_skij.emplace(math::shm::make_shared_array<Array_view_4D_t>(
      *mpi, {mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sG_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  mb_state.sSigma_tskij.emplace(math::shm::make_shared_array<Array_view_5D_t>(
      *mpi, {FT.nt_f(), mf->nspin(), mf->nkpts_ibz(), mf->nbnd(), mf->nbnd()}));
  auto& sF_skij = mb_state.sF_skij.value();
  auto& sDm_skij = mb_state.sDm_skij.value();
  auto& sG_tskij = mb_state.sG_tskij.value();
  auto& sSigma_tskij = mb_state.sSigma_tskij.value();

  double mu = 0.0;
  long init_it = chkpt::read_scf(mpi->node_comm, sF_skij, sSigma_tskij, mu, mb_state.coqui_prefix, input_grp, input_iter);
  sG_tskij = read_greens_function(*mpi, mf.get(), mb_state.coqui_prefix+ ".mbpt.h5", input_iter, input_grp);
  chkpt::read_dm(mpi->node_comm, mb_state.coqui_prefix, input_iter, sDm_skij);

  auto grad_1e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto grad_2e = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto grad_pulay = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto grad_elec = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});
  auto grad_total = nda::array<ComplexType, 2>::zeros({mf->number_of_atoms(), 3});

  Timer.start("PULAY");
  grad_pulay = eval_grad_pulay(mf, FT, dyson.sS_skij().local(), sG_tskij.local(), mu);
  Timer.stop("PULAY");

  Timer.start("ONE_ELECTRON");
  grad_1e = eval_grad_1e(mf, sDm_skij.local());
  Timer.stop("ONE_ELECTRON");

  Timer.start("TWO_ELECTRON");
  utils::check(solver_type == "hf",
               " evaluate_gradient: Invalid solver_type: {}", solver_type);
  if (solver_type == "hf") {
    if constexpr (std::is_same_v<decltype(mb_eri_t.hf_eri), std::optional<std::reference_wrapper<chol_reader_t>>>) {
      if (mb_eri_t.hf_eri != std::nullopt) {
        solvers::hf_gradient_t hf_grad(mf);
        grad_2e = hf_grad.evaluate(sDm_skij.local(), mb_eri_t.hf_eri->get());
      }
    } else if constexpr(std::is_same_v<decltype(mb_eri_t.corr_eri), std::optional<std::reference_wrapper<chol_reader_t>>>) {
      if (mb_eri_t.corr_eri != std::nullopt) {
        solvers::hf_gradient_t hf_grad(mf);
        grad_2e = hf_grad.evaluate(sDm_skij.local(), mb_eri_t.corr_eri->get());
      }
    }
  }

  Timer.stop("TWO_ELECTRON");

  grad_elec = grad_1e + grad_2e + grad_pulay;
  grad_total = grad_elec + mf->nuclear_gradient();
  print_mbpt_gradient(mf->nuclear_gradient(), mf, "GRAD_NUC");
  print_mbpt_gradient(grad_1e, mf, "GRAD_1ELECTRON");
  print_mbpt_gradient(grad_2e, mf, "GRAD_2ELECTRON");
  print_mbpt_gradient(grad_pulay, mf, "GRAD_PULAY");
  print_mbpt_gradient(grad_elec, mf, "GRAD_ELEC");
  print_mbpt_gradient(grad_total, mf, "GRAD_TOTAL");

  Timer.start("WRITE");
  write_mbpt_gradient(mpi->comm, grad_total, mb_state.coqui_prefix, input_iter);
  Timer.stop("WRITE");

  Timer.stop("GRAD_TOTAL");

  app_log(2, "\n  Gradient timers");
  app_log(2, "  ----------------");
  app_log(2, "    Total:                {0:.3f} sec", Timer.elapsed("GRAD_TOTAL"));
  app_log(2, "    One-electron:         {0:.3f} sec", Timer.elapsed("ONE_ELECTRON"));
  app_log(2, "    Two-electron:         {0:.3f} sec", Timer.elapsed("TWO_ELECTRON"));
  app_log(2, "    Pulay:                {0:.3f} sec", Timer.elapsed("PULAY"));
  app_log(2, "    Write:                {0:.3f} sec\n", Timer.elapsed("WRITE"));

  app_log(1, "####### Gradient routines end #######\n");
}

#define MBPT_GRAD_INST(HF, HARTREE, EXCHANGE, CORR)\
template void evaluate_gradient(MBState&, \
                                simple_dyson&, \
                                mb_eri_t<HF, HARTREE, EXCHANGE, CORR>&, \
                                const imag_axes_ft::IAFT&, \
                                solvers::mb_solver_t<solvers::gw_t>, \
                                const std::string&, const std::string&, int);
// All combinations of chol for hf or corr eri slots
  MBPT_GRAD_INST(chol_reader_t, thc_reader_t, thc_reader_t, thc_reader_t)
  MBPT_GRAD_INST(chol_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_GRAD_INST(chol_reader_t, thc_reader_t, chol_reader_t, thc_reader_t)
  MBPT_GRAD_INST(chol_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_GRAD_INST(chol_reader_t, chol_reader_t, thc_reader_t, thc_reader_t)
  MBPT_GRAD_INST(chol_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_GRAD_INST(chol_reader_t, chol_reader_t, chol_reader_t, thc_reader_t)
  MBPT_GRAD_INST(chol_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
  MBPT_GRAD_INST(thc_reader_t, thc_reader_t, thc_reader_t, chol_reader_t)
  MBPT_GRAD_INST(thc_reader_t, thc_reader_t, chol_reader_t, chol_reader_t)
  MBPT_GRAD_INST(thc_reader_t, chol_reader_t, thc_reader_t, chol_reader_t)
  MBPT_GRAD_INST(thc_reader_t, chol_reader_t, chol_reader_t, chol_reader_t)
#undef MBPT_GRAD_INST

} // namespace methods
