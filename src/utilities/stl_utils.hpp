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


#ifndef UTILITIES_STL_UTILS_HPP
#define UTILITIES_STL_UTILS_HPP

#include <iostream>
#include <fstream>
#include <string>

#include "utilities/check.hpp"
#include "nda/nda.hpp"

namespace utils
{

// This is needed when value_t doesn't have a default constructor (in the case of nda::range it is deprecated)
// Thanks to Nils for the implementation!
template<long N, typename value_t>
auto array_of_objects(value_t const& val) {
  return []<auto ... Is>(auto v, std::index_sequence<Is...>) {
    return std::array{((void)Is, v)...};
  }(val,std::make_index_sequence<N>());
};

template<long N>
auto default_array_of_ranges() {
  return array_of_objects<N>(::nda::range(0));
};

inline auto read_file_to_string(const std::string& filename) {
  std::ifstream file(filename);
  utils::check(file.is_open(), "Error: Could not open file: {}", filename);
  std::stringstream buffer;
  buffer << file.rdbuf(); // Read the entire file's content into the stringstream buffer
  file.close();
  return buffer.str(); // Convert the stringstream buffer to a std::string
};

// compares 2 strings, ignoring white spaces
inline bool string_equal(std::string const& s1, std::string const& s2)
{
  int N = std::min(s1.length(),s2.length());
  for(int i=0; i<N; i++) 
    if(s1[i] != s2[i]) return false; 
  for(int i=N; i<s1.length(); ++i)
    if(s1[i] != ' ') return false; 
  for(int i=N; i<s2.length(); ++i)
    if(s2[i] != ' ') return false; 
  return true;
}

}

#endif
