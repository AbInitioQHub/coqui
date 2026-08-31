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


#ifndef COQUI_MBPT_CONTEXT_H
#define COQUI_MBPT_CONTEXT_H

#include "thc_reader_t.hpp"
#include "chol_reader_t.hpp"

namespace methods {

template<typename hf_eri_t=thc_reader_t,
    typename hartree_eri_t=thc_reader_t,
    typename exchange_eri_t=thc_reader_t,
    typename corr_eri_t=thc_reader_t>
struct mb_eri_t {
  std::optional<std::reference_wrapper<hf_eri_t>> hf_eri;
  std::optional<std::reference_wrapper<hartree_eri_t>> hartree_eri;
  std::optional<std::reference_wrapper<exchange_eri_t>> exchange_eri;
  std::optional<std::reference_wrapper<corr_eri_t>> corr_eri;

  mb_eri_t() = default;

  // one eri for all
  mb_eri_t(corr_eri_t& corr):
  hf_eri(std::nullopt), 
  hartree_eri(std::nullopt), 
  exchange_eri(std::nullopt), 
  corr_eri(std::ref(corr))
  {}

  // separate eri for hf and post-hf
  mb_eri_t(hf_eri_t& hf, corr_eri_t& corr):
  hf_eri(std::ref(hf)), 
  hartree_eri(std::nullopt), 
  exchange_eri(std::nullopt), 
  corr_eri(std::ref(corr))
  {}

  // separate eri for J, K and post-hf
  mb_eri_t(hartree_eri_t& hartree, exchange_eri_t& ex, corr_eri_t& corr):
  hf_eri(std::nullopt), 
  hartree_eri(std::ref(hartree)),
  exchange_eri(std::ref(ex)), 
  corr_eri(std::ref(corr)) 
  {}

};

} // methods

#endif //COQUI_MBPT_CONTEXT_H
