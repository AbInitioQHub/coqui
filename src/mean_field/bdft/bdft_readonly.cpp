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


#include "bdft_readonly.hpp"

namespace mf::bdft {
namespace detail {

auto make_wfc(bdft_system const &sys) -> grids::truncated_g_grid {
  // read miller_wfc and construct grid based on those indices
  h5::file file = h5::file(sys.filename, 'r');
  h5::group grp(file);
  h5::group ogrp = grp.open_group("Orbitals");
  int ngm = 0;
  double wfc_ecut = 0.0;
  nda::array<int, 1> wfc_mesh(3);
  h5::h5_read(ogrp, "wfc_ecut", wfc_ecut);
  nda::h5_read(ogrp, "wfc_fft_grid", wfc_mesh);
  h5::h5_read(ogrp, "wfc_ngm", ngm);
  nda::array<int, 2> mill(ngm, 3);
  nda::h5_read(ogrp, "miller_wfc", mill);
  // check that R is compatible with mesh
  if (sys.mpi->comm.root())
    wfc_mesh = utils::generate_consistent_fft_mesh(wfc_mesh, sys.bz().symm_list, 1e-6, "bdft_readonly::make_wfc");
  sys.mpi->comm.broadcast_n(wfc_mesh.data(), 3);
  return grids::truncated_g_grid(mill, wfc_ecut, wfc_mesh, sys.recv, true);
}


auto make_ksymms(bz_symm const &bz) -> nda::array<int, 1> {
  long nktot = bz.nkpts - bz.nkpts_trev_pairs;
  std::vector<int> syms(48, -1);
  std::vector<int> unique;
  unique.reserve(48);
  for (auto is: bz.kp_symm(nda::range(nktot))) {
    if (syms[is] < 0) {
      syms[is] = unique.size();
      unique.emplace_back(is);
    }
  }
  nda::array<int, 1> ksymms(unique.size());
  for (auto [i, is]: itertools::enumerate(unique)) ksymms(i) = is;
  utils::check(ksymms[0] == 0, "Error in make_swfc_maps: missing identity operation.");
  return ksymms;
}


} // detail



} // mf::bdft