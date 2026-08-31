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


#ifndef COQUI_AC_TYPE_E_HPP
#define COQUI_AC_TYPE_E_HPP

namespace analyt_cont {
  enum ac_type_e {
    pade
  };

  inline std::string ac_enum_to_string(int ac_enum) {
    switch(ac_enum) {
      case ac_type_e::pade:
        return "pade";
      default:
        return "not recognized...";
    }
  }

  inline ac_type_e string_to_ac_enum(std::string ac_type) {
    if (ac_type == "pade") {
      return ac_type_e::pade;
    } else {
      utils::check(false, "Unrecognized ac_type");
      return ac_type_e::pade;
    }
  }

}

#endif //COQUI_AC_TYPE_E_HPP
