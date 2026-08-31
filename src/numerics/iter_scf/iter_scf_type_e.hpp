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


#ifndef COQUI_ITER_SCF_TYPE_E_HPP
#define COQUI_ITER_SCF_TYPE_E_HPP

namespace iter_scf {
  enum iter_alg_e {
    damping, DIIS
  };

  inline std::string alg_enum_to_string(int alg_enum) {
    switch(alg_enum) {
      case iter_alg_e::damping:
        return "damping";
      case iter_alg_e::DIIS:
        return "DIIS";
      default:
        return "not recognized...";
    }
  }

  inline iter_alg_e string_to_alg_enum(std::string alg_type) {
    if (alg_type == "damping") {
      return iter_alg_e::damping;
    } else if (alg_type == "DIIS") {
      return iter_alg_e::DIIS;
    } else {
      utils::check(false, "Unrecognized scf algorithm type");
      return iter_alg_e::damping;
    }
  }


} // iter_alg


#endif //COQUI_ITER_SCF_TYPE_E_HPP
