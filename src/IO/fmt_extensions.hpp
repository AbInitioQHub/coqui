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


#ifndef UTILITIES_FMT_EXTENSIONS_HPP
#define UTILITIES_FMT_EXTENSIONS_HPP

#include <format>
#include <complex>
#include <vector>
#include "nda/nda.hpp"

#if defined(ENABLE_SPDLOG)

#include <spdlog/fmt/bundled/format.h>

template <> struct fmt::formatter<std::complex<double>> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template <typename FormatContext>
  auto format(const std::complex<double>& p, FormatContext& ctx) -> decltype(ctx.out()) {
    return format_to(
        ctx.out(),
        "({:f}, {:f})", 
        std::real(p), std::imag(p));
  }
};

template <> struct fmt::formatter<std::complex<float>> {
  constexpr auto parse(format_parse_context& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template <typename FormatContext>
  auto format(const std::complex<float>& p, FormatContext& ctx) -> decltype(ctx.out()) {
    return format_to(
        ctx.out(),
        "({:f}, {:f})", 
        std::real(p), std::imag(p));
  }
};

template<typename T> struct fmt::formatter<std::vector<T>>
{

  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template <typename FormatContext>
  auto format(std::vector<T> const& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    *ctx.out()++ = '[';
    bool first = true;
    for(auto const& v : p) {
      if (!first) {
        *ctx.out()++ = ',';
      }
      format_to(ctx.out(), "{}", v);
      first = false;
    }
    *ctx.out()++ = ']';
    return ctx.out();
  }

};

template <nda::Array Arr>
struct fmt::formatter<Arr>
{

  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template <typename FormatContext>
  auto format(Arr const& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    *ctx.out()++ = '[';
    bool first = true;
    for(auto const& v : p) {
      if (!first) {
        *ctx.out()++ = ',';
      }
      format_to(ctx.out(), "{}", v);
      first = false;
    }
    *ctx.out()++ = ']';
    return ctx.out();
  }

};

#else

namespace std
{
template <> struct formatter<std::complex<double>> {
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  } 

  template<typename FormatContext>
  auto format(const std::complex<double>& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    return std::format_to(
        ctx.out(),
        "({:f}, {:f})", 
        std::real(p), std::imag(p));
  }
};

template <> struct formatter<std::complex<float>> {
//  char presentation = 'f';
  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }
  
  template<typename FormatContext>
  auto format(const std::complex<float>& p, FormatContext& ctx) -> decltype(ctx.out()) {
    return std::format_to(
        ctx.out(),
        "({:f}, {:f})", 
        std::real(p), std::imag(p));
  }
};

template<typename T> 
struct formatter<std::vector<T>> 
{

  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template<typename FormatContext>
  auto format(std::vector<T> const& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    *ctx.out()++ = '[';
    bool first = true;
    for(auto const& v : p) {
      if (!first) {
        *ctx.out()++ = ',';
      }
      std::format_to(ctx.out(), "{}", v);
      first = false;
    }
    *ctx.out()++ = ']';
    return ctx.out();
  }

};

template <nda::Array Arr>
struct formatter<Arr> 
{

  template<typename ParseContext>
  constexpr auto parse(ParseContext& ctx) -> decltype(ctx.begin()) {
    auto it = ctx.begin(), end = ctx.end();
    if (it == end || *it == '}') return it;
    if (*it == 'f') ++it;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  template<typename FormatContext>
  auto format(Arr const& p, FormatContext& ctx) const -> decltype(ctx.out()) {
    *ctx.out()++ = '[';
    bool first = true;
    for(auto const& v : p) {
      if (!first) {
        *ctx.out()++ = ',';
      }
      std::format_to(ctx.out(), "{}", v);
      first = false;
    }
    *ctx.out()++ = ']';
    return ctx.out(); 
  }

};

}

#endif

#endif
