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


#ifndef COQUI_PERMUT_SYMM_HPP
#define COQUI_PERMUT_SYMM_HPP

#include "nda/nda.hpp"

namespace methods {

inline std::string determine_permut_symm(bool force_permut_symm, bool force_real) {
  if (force_real and !force_permut_symm) {
    app_log(2, "determine_permut_symm: Found inconsistency between force_real (true) and permut_symm (false). "
               "Will set both force_real and permut_symm to 'true'. ");
    force_permut_symm = true;
  }
  std::string permut_symm = "none";
  if (force_permut_symm) {
    if (force_real) permut_symm = "8-fold";
    else permut_symm = "4-fold";
  }
  return permut_symm;
}

/**
 * Apply 4-fold permutation symmetry:
 * (ij|kl) = (kl|ij) = (ji|lk)* = (lk|ji)*
 */
template<nda::MemoryArray Array_t>
void apply_permut_symm_4fold(Array_t &V_ijkl, std::string name="") {
  constexpr int rank = ::nda::get_rank<Array_t>;
  if (name != "")
    app_log(2, "Applying 4-fold permutation symmetry to {}.", name);

  constexpr int i=0;
  constexpr int j=1;
  constexpr int k=2;
  constexpr int l=3;
  constexpr std::array<int, 4> klij{k,l,i,j};
  constexpr std::array<int, 4> jilk{j,i,l,k};
  constexpr std::array<int, 4> lkji{l,k,j,i};

  if constexpr (rank == 5) {

    for (size_t w=0; w<V_ijkl.shape(0); ++w) {
      auto Vw = V_ijkl(w,nda::ellipsis{});
      auto buffer = nda::make_regular(Vw);

      // ijkl -> klij
      auto V_klij = nda::permuted_indices_view<nda::encode(klij)>(Vw);
      buffer += V_klij;
      // ijkl -> jilk
      auto V_jilk = nda::permuted_indices_view<nda::encode(jilk)>(Vw);
      buffer += nda::conj(V_jilk);
      // ijkl -> lkji
      auto V_lkji = nda::permuted_indices_view<nda::encode(lkji)>(Vw);
      buffer += nda::conj(V_lkji);
      Vw = buffer / 4.0;
    }

  } else if constexpr (rank == 4) {

    auto buffer = nda::make_regular(V_ijkl);

    // ijkl -> klij
    auto V_klij = nda::permuted_indices_view<nda::encode(klij)>(V_ijkl);
    buffer += V_klij;
    // ijkl -> jilk
    auto V_jilk = nda::permuted_indices_view<nda::encode(jilk)>(V_ijkl);
    buffer += nda::conj(V_jilk);
    // ijkl -> lkji
    auto V_lkji = nda::permuted_indices_view<nda::encode(lkji)>(V_ijkl);
    buffer += nda::conj(V_lkji);
    V_ijkl = buffer / 4.0;

  } else {
    static_assert(rank == 4 or rank == 5, "embed_eri_t::force_permut_symm: Rank != 5 or 4");
  }
}

/**
 * Apply 8-fold permutation symmetry:
 * (ij|kl) = (kl|ij) = (ji|lk) = (lk|ji) = (ji|kl) = (lk|ij) = (ij|lk) = (kl|ji)
 */
template<nda::MemoryArray Array_t>
void apply_permut_symm_8fold(Array_t &V_ijkl, std::string name="") {
  constexpr int rank = ::nda::get_rank<Array_t>;

  if (name != "") app_log(2, "Applying 8-fold permutation symmetry to {}.", name);

  // check the largest imaginary element and set all imaginary parts to zero
  double max_imag = -1;
  nda::for_each(V_ijkl.shape(),
                [&V_ijkl, &max_imag](auto... i) mutable {
                  max_imag = std::max(max_imag, std::abs(V_ijkl(i...).imag()));
                  V_ijkl(i...) = ComplexType(V_ijkl(i...).real(),0.0);});
  if (max_imag >= 1e-10)
    app_log(1, "[WARNING] The largest imaginary part of {} = {} > 1e-10. ", name, max_imag);

  constexpr int i=0;
  constexpr int j=1;
  constexpr int k=2;
  constexpr int l=3;
  constexpr std::array<int, 4> klij{k,l,i,j};
  constexpr std::array<int, 4> jilk{j,i,l,k};
  constexpr std::array<int, 4> lkji{l,k,j,i};
  constexpr std::array<int, 4> jikl{j,i,k,l};
  constexpr std::array<int, 4> lkij{l,k,i,j};
  constexpr std::array<int, 4> ijlk{i,j,l,k};
  constexpr std::array<int, 4> klji{k,l,j,i};

  if constexpr (rank == 5) {

    utils::check(V_ijkl.shape(1)==V_ijkl.shape(2) and V_ijkl.shape(2)==V_ijkl.shape(3) and
                 V_ijkl.shape(3)==V_ijkl.shape(4), "Incorrect dimensions of {}", name);

    for (size_t w=0; w<V_ijkl.shape(0); ++w) {
      auto Vw = V_ijkl(w,nda::ellipsis{});
      auto buffer = nda::make_regular(Vw);

      // ijkl = klij
      auto V_klij = nda::permuted_indices_view<nda::encode(klij)>(Vw);
      buffer += V_klij;
      // ijkl = jilk
      auto V_jilk = nda::permuted_indices_view<nda::encode(jilk)>(Vw);
      buffer += V_jilk;
      // ijkl = lkji
      auto V_lkji = nda::permuted_indices_view<nda::encode(lkji)>(Vw);
      buffer += V_lkji;
      // ijkl = jikl
      auto V_jikl = nda::permuted_indices_view<nda::encode(jikl)>(Vw);
      buffer += V_jikl;
      // ijkl = lkij
      auto V_lkij = nda::permuted_indices_view<nda::encode(lkij)>(Vw);
      buffer += V_lkij;
      // ijkl = ijlk
      auto V_ijlk = nda::permuted_indices_view<nda::encode(ijlk)>(Vw);
      buffer += V_ijlk;
      // ijkl = klji
      auto V_klji = nda::permuted_indices_view<nda::encode(klji)>(Vw);
      buffer += V_klji;
      Vw = buffer / 8.0;
    }

  } else if constexpr (rank == 4) {

    auto buffer = nda::make_regular(V_ijkl);

    // ijkl = klij
    auto V_klij = nda::permuted_indices_view<nda::encode(klij)>(V_ijkl);
    buffer += V_klij;
    // ijkl = jilk
    auto V_jilk = nda::permuted_indices_view<nda::encode(jilk)>(V_ijkl);
    buffer += V_jilk;
    // ijkl = lkji
    auto V_lkji = nda::permuted_indices_view<nda::encode(lkji)>(V_ijkl);
    buffer += V_lkji;
    // ijkl = jikl
    auto V_jikl = nda::permuted_indices_view<nda::encode(jikl)>(V_ijkl);
    buffer += V_jikl;
    // ijkl = lkij
    auto V_lkij = nda::permuted_indices_view<nda::encode(lkij)>(V_ijkl);
    buffer += V_lkij;
    // ijkl = ijlk
    auto V_ijlk = nda::permuted_indices_view<nda::encode(ijlk)>(V_ijkl);
    buffer += V_ijlk;
    // ijkl = klji
    auto V_klji = nda::permuted_indices_view<nda::encode(klji)>(V_ijkl);
    buffer += V_klji;
    V_ijkl = buffer / 8.0;

  } else {
    static_assert(rank == 4 or rank == 5, "embed_eri_t::force_permut_symm: Rank != 5 or 4");
  }
}

template<nda::MemoryArray Array_t>
void apply_permut_symm(Array_t &V_ijkl, std::string permut_symm, std::string name="") {
  if (permut_symm == "4-fold")
    apply_permut_symm_4fold(V_ijkl, name);
  else if (permut_symm == "8-fold")
    apply_permut_symm_8fold(V_ijkl, name);
}

} // methods


#endif //COQUI_PERMUT_SYMM_HPP
