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


#ifndef COQUI_IAFT_ENUM_E_HPP
#define COQUI_IAFT_ENUM_E_HPP

#include "utilities/check.hpp"

namespace imag_axes_ft {
  enum stats_e {
    fermion, boson
  };

  inline std::string stats_enum_to_string(int stats_enum) {
    switch(stats_enum) {
      case stats_e::fermion:
        return "fermion";
      case stats_e::boson:
        return "boson";
      default:
        return "not recognized...";
    }
  }

  inline stats_e string_to_stats_enum(std::string stats) {
    if (stats == "fermion") {
      return stats_e::fermion;
    } else if (stats == "boson") {
      return stats_e::boson;
    } else {
      utils::check(false, "Unrecognized stats: {}. Available options: fermion, boson", stats);
      return stats_e::fermion;
    }
  }

  enum basis_e {
    dlr_basis, ir_basis
  };

  inline std::string basis_enum_to_string(int basis_enum) {
    switch (basis_enum) {
      case basis_e::dlr_basis:
        return "dlr";
      case basis_e::ir_basis:
        return "ir";
      default:
        return "not recognized...";
    }
  }

  inline basis_e string_to_basis_enum(std::string iaft_basis) {
    if (iaft_basis == "dlr") {
      return basis_e::dlr_basis;
    } else if (iaft_basis == "ir") {
      return basis_e::ir_basis;
    } else {
      utils::check(false, "Unrecognized IAFT basis: {}. Available options: dlr, ir", iaft_basis);
      return basis_e::ir_basis;
    }
  }

}

#endif //COQUI_IAFT_ENUM_E_HPP
