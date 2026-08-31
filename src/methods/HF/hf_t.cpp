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



#include "mpi3/communicator.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/shared_array/nda.hpp"

#include "IO/app_loggers.h"
#include "utilities/Timer.hpp"

#include "mean_field/MF.hpp"
#include "methods/ERI/detail/concepts.hpp"

#include "methods/HF/hf_t.h"

namespace methods {
  namespace solvers {

    namespace mpi3 = boost::mpi3;
    /**
     * The Hartree-Fock solver to compute the Fock matrix.
     * One-body Hamiltonian is provided from MF while density matrix
     * and electron repulsion integrals are provided at runtime.
     *
     * Usage:
     *   hf_t myhf(comm, &myMF);
     *   myhf.evaluate(F_thc, Dm, thc_eri); // THC-HF
     *   myhf.evaluate(F_chol, Dm, cholesky_eri); // Cholesky-HF
     *
     */
    hf_t::hf_t(std::string div): _div_treatment(div), _Timer() {
      if (_div_treatment!="ignore_g0" and _div_treatment!="gygi") {
        app_log(2, " hf_t: div_treatment only supports \"ignore_g0\" and \"gygi\". "
                   " coqui will take div_treatment = \"gygi\" instead.");
        _div_treatment = "gygi";
      }
    }

    /**
     * Finite-size correction for K based on "PRB 80, 085114(2009)"
     * @param Dm_skij - [INPUT] density matrix in the primary basis
     * @return finite-correction for K in the primary basis
     */
    template<nda::MemoryArray AF_t>
    void hf_t::HF_K_correction(sArray_t<AF_t> &sF_skij, const nda::MemoryArrayOfRank<4> auto &Dm_skij, 
                         const nda::MemoryArrayOfRank<4> auto &S_skij,
                         double madelung) {

      if (_div_treatment == "ignore_g0") {
        app_log(1, "No finite-size correction to the non-local HF exchange potential.\n");
        return;
      }
      app_log(1, "  Treatment of long-wavelength divergence in the non-local HF exchange potential : {}\n",
              _div_treatment);

      decltype(nda::range::all) all;
      long ns = Dm_skij.extent(0);
      long nkpts_ibz = Dm_skij.extent(1);
      long nbnd  = Dm_skij.extent(2);

      auto sDelta_skij = math::shm::make_shared_array<AF_t>(
          *sF_skij.communicator(), *sF_skij.internode_comm(), *sF_skij.node_comm(),
          {ns, nkpts_ibz, nbnd, nbnd});

      int rank = sDelta_skij.communicator()->rank();
      int size = sDelta_skij.communicator()->size();
      nda::matrix<ComplexType> buffer(nbnd, nbnd);
      sDelta_skij.win().fence();
      for (int i = rank; i < ns*nkpts_ibz; i += size) {
        int is = i / nkpts_ibz;
        int ik = i % nkpts_ibz;
        // Delta_ij = (-1.0) * madelung * Ssk_ia * Dmsk_ab * Ssk_bj
        auto Delta_sk = sDelta_skij.local()(is, ik, all, all);
        auto Dmsk = Dm_skij(is, ik, all, all);
        auto Ssk = S_skij(is, ik, all, all);
        nda::blas::gemm(ComplexType(-1.0*madelung),Ssk, Dmsk, ComplexType(0.0),buffer);
        nda::blas::gemm(ComplexType(1.0),buffer, Ssk, ComplexType(0.0),Delta_sk);
      }
      sDelta_skij.win().fence();
      sDelta_skij.all_reduce();

      if (sF_skij.node_comm()->root())
        sF_skij.local() += sDelta_skij.local();
      sF_skij.communicator()->barrier();
    }

    void hf_t::print_thc_hf_timers() {
      app_log(2, "\n  THC-HF timers");
      app_log(2, "  -------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(2, "    Primary_to_aux:        {0:.3f} sec", _Timer.elapsed("PRIM_TO_AUX"));
      app_log(2, "    Coulomb:               {0:.3f} sec", _Timer.elapsed("COULOMB"));
      app_log(2, "    Exchange:              {0:.3f} sec", _Timer.elapsed("EXCHANGE"));
      app_log(2, "    Aux_to_primary:        {0:.3f} sec\n", _Timer.elapsed("AUX_TO_PRIM"));
    }

    void hf_t::print_chol_hf_timers() {
      app_log(2, "\n  CHOL-HF timers");
      app_log(2, "  --------------");
      app_log(2, "    Total:                 {0:.3f} sec", _Timer.elapsed("TOTAL"));
      app_log(2, "    Allocations:           {0:.3f} sec", _Timer.elapsed("ALLOC"));
      app_log(2, "    Coulomb:               {0:.3f} sec", _Timer.elapsed("COULOMB"));
      app_log(2, "    Exchange:              {0:.3f} sec", _Timer.elapsed("EXCHANGE"));
      app_log(2, "    Cholesky integrals IO: {0:.3f} sec", _Timer.elapsed("CHOL"));
      app_log(2, "    Cholesky J IO:         {0:.3f} sec", _Timer.elapsed("CHOL J"));
      app_log(2, "    Cholesky K IO:         {0:.3f} sec", _Timer.elapsed("CHOL K"));
      app_log(2, "    J BLAS:                {0:.3f} sec", _Timer.elapsed("J BLAS"));
      app_log(2, "    K BLAS:                {0:.3f} sec\n", _Timer.elapsed("K BLAS"));
   }

    // instantiate templates
    using Arr4D = nda::array<ComplexType, 4>;
    using Arrv4D = nda::array_view<ComplexType, 4>; 
    using Arrv4D2 = nda::array_view<ComplexType, 4, nda::C_layout>; 
    template void hf_t::HF_K_correction(sArray_t<Arr4D> &,Arr4D const&, Arr4D const&, double);
    template void hf_t::HF_K_correction(sArray_t<Arr4D> &,Arrv4D2 const&, Arrv4D2 const&, double);
    template void hf_t::HF_K_correction(sArray_t<Arrv4D> &,Arr4D const&, Arr4D const&, double);
    template void hf_t::HF_K_correction(sArray_t<Arrv4D> &,Arr4D const&, Arrv4D const&, double);
    template void hf_t::HF_K_correction(sArray_t<Arrv4D> &,Arrv4D const&, Arrv4D const&, double);

  }
} // methods

