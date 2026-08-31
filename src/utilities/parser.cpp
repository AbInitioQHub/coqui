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


#include <string>
#include <iostream>
#include <fstream>
#include <vector>

namespace utils
{

std::vector<std::string> split(std::string const& str, std::string const& delim)
{
  std::vector<std::string> w;
  auto beg = str.find_first_not_of(delim);
  while(beg != std::string::npos) {
    auto end=str.find_first_of(delim, beg+1);
    if(end == std::string::npos) {
      w.emplace_back(str.substr(beg,str.size()-beg)); 
      break;
    }
    w.emplace_back(str.substr(beg,end-beg)); 
    beg = str.find_first_not_of(delim,end+1);
  }  
  return w;
}

}
