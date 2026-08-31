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



#include "configuration.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/mpi_context.h"
#include "utilities/Timer.hpp"
#include "utilities/proc_grid_partition.hpp"

#include "mean_field/MF.hpp"
#include "methods/GW/gw_t.h"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/scr_coulomb/scr_coulomb_t.h"
#include "methods/ERI/chol_reader_t.hpp"
#include "methods/ERI/thc_reader_t.hpp"
#include "methods/ERI/detail/concepts.hpp"

#include "methods/GF2/gf2_t.h"

namespace methods::solvers {

  gf2_t::gf2_t(mf::MF *MF, imag_axes_ft::IAFT *ft,
      std::string div, 
       std::string direct_type,
       std::string exchange_alg,
       std::string exchange_type,
       std::string output,
       bool save_C_, bool sosex_save_memory_):
     _context(MF->mpi()), _MF(MF), _ft(ft), _div_treatment(div),
     _direct_type(direct_type), _exchange_type(exchange_type), _exchange_alg(exchange_alg),
     _save_C(save_C_), _sosex_save_memory(sosex_save_memory_), _output(output), _Timer()
  {
    if (_context->comm.size()%_context->node_comm.size()!=0) {
      APP_ABORT("SCF: number of processors on each node should be the same.");
    }
  }


  // FIXME Only `gf2` is supported in the MBstate interface. All variations need to be fixed.
  void gf2_t::evaluate(MBState &mb_state, THC_ERI auto &thc) {
    //http://patorjk.com/software/taag/#p=display&f=Calvin%20S&t=COQUI%20thc-gftwo
    app_log(1, "\n"
               "╔═╗╔═╗╔═╗ ╦ ╦╦  ┌┬┐┬ ┬┌─┐ ┌─┐┌─┐┌┬┐┬ ┬┌─┐\n"
               "║  ║ ║║═╬╗║ ║║   │ ├─┤│───│ ┬├┤  │ ││││ │\n"
               "╚═╝╚═╝╚═╝╚╚═╝╩   ┴ ┴ ┴└─┘ └─┘└   ┴ └┴┘└─┘\n");
    app_log(1, "  nbnd  = {}\n"
               "  THC auxiliary basis  = {}\n"
               "  nkpts = {}\n"
               "  nkptz_ibz = {}\n"
               "  divergent treatment at q->0 = {}\n\n"
               "  details of the GF2 algorithm: \n"
               "    direct_type = {}\n"
               "    exchange_type = {}\n"
               "    exchange_alg = {}\n"
               "    t_thresh = {}\n",
            _MF->nbnd(), thc.Np(), _MF->nkpts(), _MF->nkpts_ibz(), _div_treatment,
            _direct_type, _exchange_type, _exchange_alg, _t_thresh);
    _ft->metadata_log();

    utils::check(_exchange_type == "gf2" and _direct_type == "gf2",
                 "gf2_t::evaluate: Only gf2 direct and exchange types are supported in the MBState interface.");
    utils::check(mb_state.mpi == thc.mpi(),
                 "gf2_t::evaluate: THC_ERI and MBState should have the same MPI context.");
    utils::check(mb_state.sG_tskij.has_value(),
                 "gf2_t::evaluate: sG_tskij is not initialized in MBState.");
    utils::check(mb_state.sSigma_tskij.has_value(),
                 "gf2_t::evaluate: sSigma_tskij is not initialized in MBState.");
    utils::check(_ft->nt_f() == _ft->nt_b(),
                 "thc-gf2: We assume nt_f == nt_b at least for now. \n"
                 "        And we assume tau sampling for fermions and bosons are the same.");
    { // Check if tau_mesh is symmetric w.r.t. beta/2
      auto tau_mesh = _ft->tau_mesh();
      long nts = tau_mesh.shape(0);
      for (size_t it = 0; it < nts; ++it) {
        size_t imt = nts - it - 1;
        double diff = std::abs(tau_mesh(it)) - std::abs(tau_mesh(imt));
        utils::check(diff <= 1e-6, "thc-gf2: IAFT grid is not compatible with particle-hole symmetry. {}, {}",
                     tau_mesh(it), tau_mesh(imt));
      }
    }

    for( auto& v: {"TOTAL", "TOTAL_DIR", "TOTAL_EXC", "COMM",
                   "PRIM_TO_AUX", "AUX_TO_PRIM", "BUILD_B", "BUILD_B_INIT", "BUILD_B_RED",
                   "EVALUATE_PI", "BUILD_B_BEFORE_RED", "MAKE_DISTR_ARRAY_XX", "MAKE_DISTR_ARRAY_U",
                   "EVALUATE_PIUPI", "MAKE_DISTR_ARRAY1", "MAKE_DISTR_ARRAY2",
                   "EVALUATE_SIGMA_DIR", "EVALUATE_SIGMA_EXC", "EVALUATE_SIGMA_EXC_NP",
                   "BUILD_UPqs", "CONTRACT_UGG", "BUILD_D", "D_REDUCE",
                   "MULTIPLY_B", "B_LOOP", "B_XX_BUILD", "B_X", "B_Z", "BUILD_D_BUF", "BUILD_D_ZERO",
                   "BUILD_D_PR","BUILD_D_PR_GR", "BUILD_D_PR_RED", "NORMS",
                   "MULTIPLY_C", "BUILD_D_BCAST", "BUILD_D_LOOP", "BUILD_D_OUTLOOP",
                   "CONTRACT_UG", "CONTRACT_CB", "SUM_K_D", "SUM_K_SIGMA", "SIGMA_AUX_TO_AO"
    } ) {
      _Timer.add(v);
    }
    if(_exchange_type == "dynamic_sosex" || _exchange_type == "dynamic_2sosex") {
      for( auto& v: {"G_TRANSFORM_SOSEX",  "IMAG_FT_TtoW", "FT_REDISTRIBUTE", "IMAG_FT_WtoT",
                     "B_WA_SOSEX", "EVALUATE_SIGMA_SOSEX", "BUILD_WGG_SOSEX", "REDISTRIBUTE_BC",
                     "BUILD_D_SOSEX", "BUILD_GG_SOSEX", "W_RETRIEVAL_SOSEX", "W_MULT_SOSEX",
                     "GET_C_SOSEX","C_GEMM_SOSEX", "ALLOC_SOSEX", "REDISTRIBUTE_SOSEX", "TOTAL_SOSEX"
      } ) {
        _Timer.add(v);
      }
    }

    _Timer.start("TOTAL");
    thc_gf2_Xqindep(mb_state, thc);
    _Timer.stop("TOTAL");

    print_thc_gf2_timers();
    print_thc_sosex_timers();
    thc.print_timers();
  }

  void gf2_t::evaluate(MBState &mb_state, Cholesky_ERI auto &chol)
  {
    using namespace math::shm;
    app_log(1, " *** Cholesky GF2 solver begin *** ");
    _ft->metadata_log();
    utils::check(_ft->nt_f() == _ft->nt_b(),
                 "chol-gf2:: we assume nt_f == nt_b at least for now \n"
                 "(will lift the restriction at some point...)");
    utils::check(mb_state.mpi == chol.mpi(),
                 "gf2_t::evaluate: Cholesky_ERI and MBState should have the same MPI context.");
    utils::check(mb_state.sG_tskij.has_value(),
                 "gf2_t::evaluate: sG_tskij is not initialized in MBState.");

    for( auto& v: {"TOTAL", "TOTAL_DIR", "TOTAL_EXC", "ALLOC", "COMM",
                   "EVALUATE_P0", "EVALUATE_SIGMA_DIR", "EVALUATE_SIGMA_EXC",
                   "CONTRACT", "BUILD_INT",
                   "ERI_READER"} ) {
      _Timer.add(v);
    }

    _Timer.start("TOTAL");
    auto G_tskij = mb_state.sG_tskij.value().local();
    auto& sSigma_tskij = mb_state.sSigma_tskij.value();

    _Timer.start("TOTAL_DIR");
    if (direct_type() == "gf2") {
      chol_run_direct(G_tskij, sSigma_tskij, chol);
    } else if(direct_type() == "gw") {
      gw_t gw(_ft, _div_treatment, _output);
      _Timer.start("EVALUATE_SIGMA_DIR");
      gw.evaluate(G_tskij, sSigma_tskij, chol);
      _Timer.stop("EVALUATE_SIGMA_DIR");
    } else {
      APP_ABORT("gf2_t: Unknown exchange type\n");
    }
    _Timer.stop("TOTAL_DIR");

    _Timer.start("TOTAL_EXC");
    if (exchange_alg() != "none") {
      _Timer.start("ALLOC");
      sArray_t<nda::array_view<ComplexType, 5>> sSigma_exc(*_context, sSigma_tskij.shape());
      sSigma_exc.set_zero();
      _Timer.stop("ALLOC");

      chol_run_2(G_tskij, sSigma_exc, chol);

      _Timer.start("COMM");
      sSigma_exc.win().fence();
      sSigma_exc.all_reduce();
      sSigma_exc.win().fence();
      _Timer.stop("COMM");

      add_exc_to_Sigma(sSigma_tskij, sSigma_exc); //Sigma += sSigma_exc.local();
    }

    _Timer.stop("TOTAL_EXC");
    _Timer.stop("TOTAL");

    print_chol_gf2_timers();
    chol.print_timers();

    app_log(1, " **** Cholesky GF2 solver end **** \n");
  }

  void gf2_t::print_chol_gf2_timers() {
    app_log(2, "\n  Chol-GF2 timers");
    app_log(2, "  ---------------");
    app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
    app_log(2, "    Total dir diagram:     {0:.3f} sec", _Timer.elapsed("TOTAL_DIR"));
    app_log(2, "    Total exc diagram:     {0:.3f} sec", _Timer.elapsed("TOTAL_EXC"));
    app_log(2, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
    app_log(2, "    Communication:         {0:.3f} sec", _Timer.elapsed("COMM"));
    if(exchange_type() == "gf2") {
     app_log(2, "    Evaluate_P0:           {0:.3f} sec", _Timer.elapsed("EVALUATE_P0"));
    }
    app_log(2, "    Evaluate_Sigma dir:    {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_DIR"));
    app_log(2, "    Evaluate_Sigma exc:    {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_EXC"));
    app_log(2, "    4-index integrals:     {0:.3f} sec", _Timer.elapsed("BUILD_INT"));
    app_log(2, "    int-G contractions:    {0:.3f} sec", _Timer.elapsed("CONTRACT"));
    app_log(2, "    Chol-ERI reader:       {0:.3f} sec\n", _Timer.elapsed("ERI_READER"));
  }

  void gf2_t::print_thc_gf2_timers(){
    app_log(2, "\n  THC-GF2 timers");
    app_log(2, "  --------------");
    app_log(2, "    Total:                      {0:.3f} sec", _Timer.elapsed("TOTAL"));
    app_log(2, "    Total direct term:          {0:.3f} sec", _Timer.elapsed("TOTAL_DIR"));
    app_log(2, "    Evaluate Pi (R):            {0:.3f} sec", _Timer.elapsed("EVALUATE_PI"));
    app_log(2, "    Evaluate Pi U Pi:           {0:.3f} sec", _Timer.elapsed("EVALUATE_PIUPI"));
    app_log(2, "    Evaluate Sigma direct:      {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_DIR"));
    app_log(2, "    Total exchange term:        {0:.3f} sec", _Timer.elapsed("TOTAL_EXC"));
    if(exchange_alg() != "none") {
    app_log(2, "    Build U_Pqs:                {0:.3f} sec", _Timer.elapsed("BUILD_UPqs"));
    app_log(2, "    Build U_Pqs: make distr XX  {0:.3f} sec", _Timer.elapsed("MAKE_DISTR_ARRAY_XX"));
    app_log(2, "    Build U_Pqs: make distr U   {0:.3f} sec", _Timer.elapsed("MAKE_DISTR_ARRAY_U"));
    app_log(2, "    Build B:                    {0:.3f} sec", _Timer.elapsed("BUILD_B"));
    app_log(2, "    Build B: for loop           {0:.3f} sec", _Timer.elapsed("B_LOOP"));
    app_log(2, "    Build B: X                  {0:.3f} sec", _Timer.elapsed("B_X"));
    app_log(2, "    Build B: Z                  {0:.3f} sec", _Timer.elapsed("B_Z"));
    app_log(2, "    Build B: XX                 {0:.3f} sec", _Timer.elapsed("B_XX_BUILD"));
    app_log(2, "    Build B multiply:           {0:.3f} sec", _Timer.elapsed("MULTIPLY_B"));
    app_log(2, "    Build B before redistr:     {0:.3f} sec", _Timer.elapsed("BUILD_B_BEFORE_RED"));
    app_log(2, "    Build B redistr:            {0:.3f} sec", _Timer.elapsed("BUILD_B_RED"));
    app_log(2, "    Build B init                {0:.3f} sec", _Timer.elapsed("BUILD_B_INIT"));
    app_log(2, "    Build B make distr ar1      {0:.3f} sec", _Timer.elapsed("MAKE_DISTR_ARRAY1"));
    app_log(2, "    Build B make distr ar2      {0:.3f} sec", _Timer.elapsed("MAKE_DISTR_ARRAY2"));
    app_log(2, "    G transform:                {0:.3f} sec", _Timer.elapsed("G_TRANSFORM"));
    app_log(2, "    Build C:                    {0:.3f} sec", _Timer.elapsed("BUILD_C"));
    app_log(2, "    Build C multiply:           {0:.3f} sec", _Timer.elapsed("MULTIPLY_C"));
    app_log(2, "    Build D:                    {0:.3f} sec", _Timer.elapsed("BUILD_D"));
    app_log(2, "    Build D outer loop:         {0:.3f} sec", _Timer.elapsed("BUILD_D_OUTLOOP"));
    app_log(2, "    Build D bcast:              {0:.3f} sec", _Timer.elapsed("BUILD_D_BCAST"));
    app_log(2, "    Build D loop:               {0:.3f} sec", _Timer.elapsed("BUILD_D_LOOP"));
    app_log(2, "    Build D buf :               {0:.3f} sec", _Timer.elapsed("BUILD_D_BUF"));
    app_log(2, "    Build D zero :              {0:.3f} sec", _Timer.elapsed("BUILD_D_ZERO"));
    app_log(2, "    Build D PR   :              {0:.3f} sec", _Timer.elapsed("BUILD_D_PR"));
    app_log(2, "    Build D PR GR  :            {0:.3f} sec", _Timer.elapsed("BUILD_D_PR_GR"));
    app_log(2, "    Build D PR RED  :           {0:.3f} sec", _Timer.elapsed("BUILD_D_PR_RED"));
    app_log(2, "    D reduce:                   {0:.3f} sec", _Timer.elapsed("D_REDUCE"));
    app_log(2, "    Norm evaluation:            {0:.3f} sec", _Timer.elapsed("NORMS"));
    app_log(2, "    Evaluate Sigma exc:         {0:.3f} sec", _Timer.elapsed("EVALUATE_SIGMA_EXC"));
    app_log(2, "    Evaluate Sigma exc non-p:   {0:.3f} sec\n", _Timer.elapsed("EVALUATE_SIGMA_EXC_NP"));
    }
  }

  void gf2_t::print_thc_sosex_timers(){
    if(exchange_type() == "dynamic_sosex" || exchange_type() == "dynamic_2sosex" ) {
    app_log(2, "\n  THC-SOSEX timers");
    app_log(2, "  --------------");
    app_log(2, "    Total SOSEX:              {0:.3f} sec", _Timer.elapsed("TOTAL_SOSEX"));
    app_log(2, "    Distributed allocations:  {0:.3f} sec", _Timer.elapsed("ALLOC_SOSEX"));
    app_log(2, "    Redistributions:          {0:.3f} sec", _Timer.elapsed("REDISTRIBUTE_SOSEX"));
    app_log(2, "    G_ij -> G_Qj,G_iQ:        {0:.3f} sec", _Timer.elapsed("G_TRANSFORM_SOSEX"));
    app_log(2, "    outer(G,G):               {0:.3f} sec", _Timer.elapsed("BUILD_GG_SOSEX"));
    app_log(2, "    FT t->w:                  {0:.3f} sec", _Timer.elapsed("IMAG_FT_TtoW"));
    app_log(2, "    FT w->t:                  {0:.3f} sec", _Timer.elapsed("IMAG_FT_WtoT"));
    app_log(2, "    FT REDISTRIBUTE:          {0:.3f} sec", _Timer.elapsed("FT_REDISTRIBUTE"));
    app_log(2, "    W * GG:                   {0:.3f} sec", _Timer.elapsed("BUILD_WGG_SOSEX"));
    app_log(2, "    W retrieval:              {0:.3f} sec", _Timer.elapsed("W_RETRIEVAL_SOSEX"));
    app_log(2, "    W multiplication:         {0:.3f} sec", _Timer.elapsed("W_MULT_SOSEX"));
    app_log(2, "    B_WA:                     {0:.3f} sec", _Timer.elapsed("B_WA_SOSEX"));
    app_log(2, "    C build/access:           {0:.3f} sec", _Timer.elapsed("GET_C_SOSEX"));
    app_log(2, "    C gemm        :           {0:.3f} sec", _Timer.elapsed("C_GEMM_SOSEX"));
    app_log(2, "    BC REDISTRIBUTE:          {0:.3f} sec", _Timer.elapsed("REDISTRIBUTE_BC"));
    app_log(2, "    Build D:                  {0:.3f} sec", _Timer.elapsed("BUILD_D_SOSEX"));
    app_log(2, "    Sigma:                    {0:.3f} sec\n", _Timer.elapsed("EVALUATE_SIGMA_SOSEX"));
    }
  }

  long& gf2_t::iter() { return _iter; }
  std::string gf2_t::output() const { return _output; }
  std::string gf2_t::gw_div_treatment() const { return _div_treatment; }
  double& gf2_t::t_thresh() { return _t_thresh; }

  std::string gf2_t::direct_type() const {return _direct_type; }
  std::string gf2_t::exchange_type() const {return _exchange_type; }
  std::string gf2_t::exchange_alg() const {return _exchange_alg; }
  bool gf2_t::sosex_save_memory() const {return _sosex_save_memory; }
  bool gf2_t::save_C() const {return _save_C; }

} // namespace methods::solvers {

  // non-instantiated templates
  #include "methods/GF2/thc_gf2.icc"
  #include "methods/GF2/cholesky_gf2.icc"
  #include "methods/GF2/thc_sosex.icc"

namespace methods::solvers {

  // Instantiations
  using methods::thc_reader_t;
  using methods::chol_reader_t;
  using math::shm::shared_array;
  using memory::host_array;
  using memory::host_array_view;
  using nda::C_layout;

  template void gf2_t::evaluate(MBState&, thc_reader_t&);
  template void gf2_t::evaluate(MBState&, chol_reader_t&);

}

