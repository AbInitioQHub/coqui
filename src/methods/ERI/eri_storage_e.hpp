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


#ifndef METHODS_ERI_ERI_FORMAT_E_HPP
#define METHODS_ERI_ERI_FORMAT_E_HPP

namespace methods
{

/***********************************************************************/
/*                           eri_storage_e                             */ 
/***********************************************************************/
enum eri_storage_e {
  incore, outcore
};

inline std::string eriform_enum_to_string(int storage_enum) {
  switch(storage_enum) {
    case eri_storage_e::incore:
      return "incore";
    case eri_storage_e::outcore:
      return "outcore";
    default:
      return "not recognized...";
  }
}

inline eri_storage_e string_to_eri_storage_enum(std::string eriform) {
  if (eriform == "incore") {
    return eri_storage_e::incore;
  } else if (eriform == "outcore") {
    return eri_storage_e::outcore;
  } else {
    utils::check(false, "Unrecognized storage type: {}. Available options: incore, outcore", eriform);
    return eri_storage_e::incore;
  }
}

}

#endif // METHODS_ERI_ERI_FORMAT_E_HPP
