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


#ifndef UTILITIES_ASSERT_HPP
#define UTILITIES_ASSERT_HPP

#include<string>
#include <source_location>
#include "IO/AppAbort.hpp"

namespace utils
{

/**
 * Checks whether the cond is true, and abort otherwise with provided message (args)
 * @tparam Args
 * @param cond - condition to be verified
 * @param args - messages
 */
template<class... Args>
struct check
{
  check(bool cond, const std::string_view format_string, Args&&... args, const std::source_location& loc = std::source_location::current()) 
  { 
    if(not cond) {
      if constexpr (sizeof...(Args) > 0) 
        APP_ABORT(loc, format_string, std::forward<Args>(args)...);
      else 
        APP_ABORT(loc, format_string);
    }
  }
};

template <typename... Args>
check(bool, const std::string_view, Args&&...) -> check<Args...>;


}

#endif
