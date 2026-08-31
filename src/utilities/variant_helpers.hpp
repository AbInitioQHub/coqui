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


#ifndef UTILITIES_VARIANT_HELPERS_HPP
#define UTILITIES_VARIANT_HELPERS_HPP

#include <variant>
#include "IO/AppAbort.hpp"

namespace utils
{

namespace detail {

  template<typename ... T>                                                 
  struct Overload : T ... { 
    using T::operator() ...;
  };
  template<class... T> Overload(T...) -> Overload<T...>;

  template<typename T>
  auto monostate_abort = [](std::monostate) { 
				APP_ABORT("Error: Reached monostate_abort."); 
				return T{}; 
			};

  inline auto monostate_abort_void = [](std::monostate) { 
				APP_ABORT("Error: Reached monostate_abort."); 
			};

} // detail

// add callable concept
template<typename T>
auto Overload = [](auto&& callable) {
  return detail::Overload {
    callable,
    detail::monostate_abort<T>
  };
};

template<typename T>
auto Overload_2call = [](auto&& callable1, auto&& callable2) {
  return detail::Overload {
    callable1,
    callable2,
    detail::monostate_abort<T>
  };
};

inline auto Overload_void = [](auto&& callable) {
  return detail::Overload {
    callable,
    detail::monostate_abort_void
  };
};

} // utils

#define VOID_VISITOR(F,obj) \
  std::visit(utils::Overload_void( [&](auto&& v) { v.F(); } ), obj); 

#define VOID_VISITOR_WITH_FORWARD_ARG(F,obj,Args,param) \
  std::visit(utils::Overload_void( [&](auto&& v) { v.F(std::forward<Args>(param)...); } ), obj); 

#define VISITOR(F,T,obj) \
  std::visit(utils::Overload<T>( \
                    [&](auto&& v) { return v.F(); } ), obj); 

#define VISITOR_WITH_ARG(F,T,obj,arg) \
  std::visit(utils::Overload<T>( \
                    [&](auto&& v) { return v.F(arg); } ), obj); 

#define VOID_VISITOR_v2(F1,F2,obj) \
  std::visit(utils::Overload_void( [&](auto&& v) { v.F1().F2; } ), obj); 

#define VISITOR_v2(F1,F2,T,obj) \
  std::visit(utils::Overload<T>( \
                    [&](auto&& v) { return (v.F1().F2); } ), obj); 



#endif
