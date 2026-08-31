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


#ifndef UTILITIES_NUMERICAL_INTEGRATION_HPP
#define UTILITIES_NUMERICAL_INTEGRATION_HPP

#include "configuration.hpp"
#include "nda/nda.hpp"

namespace utils
{

// 1d equally-spaced linear grid
template<typename T>
class linear_grid 
{
  public:
  linear_grid(T r0, T rN, long N) : _r(N), _dr( (rN-r0)/T(N-1l) ) {
    for( auto [i,v] : itertools::enumerate(_r) ) v = r0 + _dr*T(i);
  }

  ~linear_grid() = default;

  linear_grid(linear_grid const&) = default;
  linear_grid(linear_grid &&) = default;
  linear_grid& operator=(linear_grid const&) = default;
  linear_grid& operator=(linear_grid &&) = default;

  auto size() const { return _r.extent(0); }
  auto r(long i) const { return _r(i); }
  auto dr([[maybe_unused]] long i) const { return _dr; }
  auto r_dr(long i) const { return std::make_tuple(_r(i),_dr(i)); }

  private:
    // 1d equally-spaced linear grid
    ::nda::array<T, 1> _r;
    // spacing
    T _dr = T(0);
};

// 1d equally-spaced logarthmic grid
template<typename T>
class log_grid 
{
  public:
  log_grid(T r0, T rN, long N) : _r(N) { 
    T u0 = std::log(r0), uN = std::log(rN);
    du = ( uN - u0 ) / T(N-1l);
    for( auto i : ::nda::range(N) ) _r(i) = std::exp(u0 + du*T(i));
  }

  ~log_grid() = default;

  log_grid(log_grid const&) = default;
  log_grid(log_grid &&) = default;
  log_grid& operator=(log_grid const&) = default;
  log_grid& operator=(log_grid &&) = default;

  auto size() const { return _r.extent(0); }
  auto r(long i) const { return _r(i); }
  auto dr(long i) const { return _r(i)*du; }
  auto r_dr(long i) const { return std::make_tuple(_r(i),_r(i)*du); }

  private:
    // 1d equally-spaced logarithmic grid
    ::nda::array<T, 1> _r;
    T du = T(0);
};

// same functionality as log_grid, but does not store the grid
template<typename T>
class log_grid_f
{
  public:
  log_grid_f(T r0, T rN, long N) : u0(std::log(r0)), du(( std::log(rN) - u0 ) / T(N-1l)), _N(N) {}
  ~log_grid_f() = default;

  log_grid_f(log_grid_f const&) = default;
  log_grid_f(log_grid_f &&) = default;
  log_grid_f& operator=(log_grid_f const&) = default;
  log_grid_f& operator=(log_grid_f &&) = default;

  auto size() const { return _N; }
  auto r(long i) const { return std::exp(u0+du*T(i)); }
  auto dr(long i) const { return r(i)*du; }
  auto r_dr(long i) const { T ri = r(i); return std::make_tuple(ri,ri*du); }

  private:
    T u0 = 0, du = 0; 
    long _N = 0;
};

template<typename T>
class linear_grid_f
{ 
  public:
  linear_grid_f(T r0, T rN, long N) : u0(r0), du((rN - r0 ) / T(N-1l)), _N(N) {}
  ~linear_grid_f() = default;

  linear_grid_f(linear_grid_f const&) = default;
  linear_grid_f(linear_grid_f &&) = default;
  linear_grid_f& operator=(linear_grid_f const&) = default;
  linear_grid_f& operator=(linear_grid_f &&) = default;

  auto size() const { return _N; }
  auto r(long i) const { return u0+du*T(i); }
  auto dr([[maybe_unused]] long i) const { return du; }
  auto r_dr(long i) const { return std::make_tuple(r(i),du); }

  private:
    long _N = 0;
    T u0 = 0, du = 0;
};

template<typename T, typename grid_t, typename func_t>
auto simpson_rule_f(grid_t && r_grid, func_t && g)
{
  auto N = r_grid.size();
  utils::check(N%2==1, "simpson_rule: Expect an odd number of points in the grid:{}",N);
  T F(0);
  auto [r,dr] = r_grid.r_dr(0);
  F = g(r)*dr;
  std::tie(r,dr) = r_grid.r_dr(N-1);
  F += g(r)*dr;
  T fct(4);
  for(long i=1, sg=-1; i<N-1; ++i) { 
    std::tie(r,dr) = r_grid.r_dr(i);
    F += fct*g(r)*dr;
    fct += T(sg)*T(2.0);
    sg *= -1;
  }
  return F/T(3.0);
}

template<typename T>
auto simpson_rule_array(::nda::MemoryArrayOfRank<1> auto const& dr,
                  ::nda::MemoryArrayOfRank<1> auto const& g) 
{
  static_assert(std::is_same_v<T,::nda::get_value_t<decltype(dr)>>,"Value type mismatch");
  static_assert(std::is_same_v<T,::nda::get_value_t<decltype(g)>>,"Value type mismatch");
  auto N = dr.size();
  utils::check(N%2==1, "simpson_rule: Expect an odd number of points in the grid:{}",N);
  T F = g(0)*dr(0) + g(N-1)*dr(N-1);
  T fct(4);
  for(long i=1, sg=-1; i<N-1; ++i) {
    F += fct*g(i)*dr(i);
    fct += T(sg)*T(2.0);
    sg *= -1;
  }
  return F/T(3.0);
}


}

#endif
