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


#ifndef COQUI_IAFT_UTILS_HPP
#define COQUI_IAFT_UTILS_HPP

#include <string>

#include "configuration.hpp"
#include "nda/nda.hpp"
#include "IAFT.hpp"

namespace imag_axes_ft {

  class IAFT; // forward declaration

  namespace test_utils {

    /**
     * Construct a Green's function at a given imaginary time with a known analytical form for testing IAFT. 
     * The convention of t is [-1, 1] which maps to the physical imaginary time [0, beta]. 
     */
    nda::matrix<ComplexType> gfun_tau(int norb, double beta, double t, bool ph_sym = false);

    /**
     * Construct a Green's function at a given Matsubara frequency with a known analytical form for testing IAFT.
     * The convention for n is that n = (2k + 1) for fermions and n = 2k for bosons, where k is an integer. 
     */
    nda::matrix<ComplexType> gfun_iw(int norb, double beta, int n, imag_axes_ft::stats_e stat, bool ph_sym = false);

    /**
     * Build Green's function on the imaginary time mesh with a known analytical form for testing IAFT.
     * The convention of t is [-1, 1] which maps to the physical imaginary time [0, beta].
     */
    nda::array<ComplexType, 3> build_g_tau(
      const nda::MemoryArrayOfRank<1> auto &tau_mesh, double beta, int norb, bool ph_sym = false) {

      auto nts = tau_mesh.size();
      // only construct the first half of the tau points for particle-hole symmetry. 
      if (ph_sym) nts = (nts % 2 == 0) ? nts / 2 : nts / 2 + 1; 

      auto G_t_ref = nda::array<ComplexType, 3>(nts, norb, norb);
      for (int it = 0; it < nts; ++it) {
        G_t_ref(it, nda::range::all, nda::range::all) = gfun_tau(norb, beta, tau_mesh(it), ph_sym);
      }

      return G_t_ref;
    }

    nda::array<ComplexType, 3> build_g_iw(
      const nda::MemoryArrayOfRank<1> auto &wn_mesh, double beta, imag_axes_ft::stats_e stat, 
      int norb, bool ph_sym = false) {

      auto nw = wn_mesh.size();
      // only construct positive Matsubara frequencies for particle-hole symmetry. 
      int iw_shift = (ph_sym)? nw / 2 : 0;
      if (ph_sym) nw = (nw % 2 == 0) ? nw / 2 : nw / 2 + 1; 

      auto G_w_ref = nda::array<ComplexType, 3>(nw, norb, norb);
      for (int iw = 0; iw < nw; ++iw) {
        G_w_ref(iw, nda::range::all, nda::range::all) =
          gfun_iw(norb, beta, int(wn_mesh(iw+iw_shift)), stat, ph_sym);
      }

      return G_w_ref;
    }

  } // namespace test_utils

  /**
   * Reconstruct IAFT object from the metadata in bdft scf output
   * @return IAFT
   */
  IAFT read_iaft(std::string scf_file, bool print_meta_log = true);

} // imag_axes_ft



#endif //COQUI_IAFT_UTILS_HPP
