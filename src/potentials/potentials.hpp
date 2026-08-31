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


#ifndef COQUI_POTENTIALS_POTENTIALS_H
#define COQUI_POTENTIALS_POTENTIALS_H

#include<variant>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "utilities/check.hpp"
#include "IO/ptree/ptree_utilities.hpp"


#include "potentials/coulomb.hpp"

namespace pots
{

namespace detail
{

inline std::variant<coulomb_t> build_from_pt(ptree const& pt, bool print_metadata = true)
{
  auto type = io::get_value_with_default<std::string>(pt,"type","coulomb"); 
  if( type == "coulomb" ) {
    return std::variant<coulomb_t>{coulomb_t{pt, print_metadata}};
  } else {
    APP_ABORT(" Error: Invalid potential type:{}",type);
  }
  return std::variant<coulomb_t>{coulomb_t{}}; 
}

}

class potential_t
{
  using var_t = std::variant<coulomb_t>;
  using this_t = potential_t;

  public:

    potential_t() = delete;

    potential_t(ptree const& pt, bool print_metadata) : var(detail::build_from_pt(pt, print_metadata)) {}

    // construct with coulomb_t 
    explicit potential_t(coulomb_t const& arg) : var(arg) {}
    explicit potential_t(coulomb_t && arg) : var(std::move(arg)) {}

    this_t& operator=(coulomb_t const& arg) { var = arg; return *this; }
    this_t& operator=(coulomb_t && arg) { var = std::move(arg); return *this; }

    ~potential_t() = default;
    potential_t(potential_t const&) = default;
    potential_t(potential_t&&) = default;
    potential_t& operator=(potential_t const&) = default;
    potential_t& operator=(potential_t&&) = default;
    
    void print_meta() const
    { return std::visit( [&](auto&& v) { return v.print_meta(); }, var); }

    //evaluation routines given a precomputed list of G-vectors
    void evaluate(nda::MemoryArrayOfRank<1> auto&& V,
                nda::MemoryArrayOfRank<2> auto const& lattv,
                nda::MemoryArrayOfRank<2> auto const& gv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
    { 
      std::visit( [&](auto&& v) { v.evaluate(V,nda::stack_array<double,3,3>{lattv},
                                             gv,kp,kq); }, var); 
    }

    //evaluation routines in fft grid 
    void evaluate_in_mesh(nda::range g_rng, nda::MemoryArrayOfRank<1> auto&& V,
                nda::MemoryArrayOfRank<1> auto const& mesh,
                nda::MemoryArrayOfRank<2> auto const& lattv,
                nda::MemoryArrayOfRank<2> auto const& recv,
                nda::ArrayOfRank<1> auto const& kp,
                nda::ArrayOfRank<1> auto const& kq)
    {
      std::visit( [&](auto&& v) { v.evaluate_in_mesh(g_rng,V,
          nda::stack_array<long,3>{mesh},nda::stack_array<double,3,3>{lattv},
          nda::stack_array<double,3,3>{recv},kp,kq); }, var);
    }

  private:

    var_t var; 

};

}

#endif
