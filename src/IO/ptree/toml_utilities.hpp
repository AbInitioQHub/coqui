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


#ifndef COQUI_TOML_UTILITIES_HPP
#define COQUI_TOML_UTILITIES_HPP

#include <sstream>
#include <fstream>
#include <iostream>
#include <vector>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <toml++/toml.hpp>
#include "IO/app_loggers.h"

using boost::property_tree::ptree;

namespace io {
// Function to trim left trailing spaces
inline void trim_left_space(std::string &s)
{
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
    return !std::isspace(ch);
  }));
}

inline bool reset_section(std::string parent_next, std::string child_next,
                          std::string parent, std::string child) {
  // not a nested table
  if (child_next == "") return true;

  // two nested tables with the same parent but different child
  if (parent_next == parent and child_next != child) return false;

  // two nested tables with the same parent and child
  return true;
}

// Function to split the input string into sections based on tables' name
inline std::vector<std::string> split_into_sections(std::stringstream& stream)
{
  std::vector<std::string> sections;
  std::string line;
  std::string parent;
  std::string child;
  std::string section;

  while (std::getline(stream, line)) {
    // trim spaces in the beginning of each line
    trim_left_space(line);
    // Check if the line is a section header
    if (line.starts_with('[')) {
      // extract section key
      size_t key_end = line.find("]");
      size_t dot_pos = line.find(".");

      if (parent.empty() and child.empty()) {

        parent = (dot_pos==std::string::npos)? line.substr(1, key_end-1) : line.substr(1, dot_pos-1);
        child = (dot_pos==std::string::npos)? "" : line.substr(dot_pos+1, key_end-1);

      } else {

        std::string parent_next = (dot_pos==std::string::npos)?
            line.substr(1, key_end-1) : line.substr(1, dot_pos-1);
        std::string child_next = (dot_pos==std::string::npos)?
            "" : line.substr(dot_pos+1, key_end-1);

        if (reset_section(parent_next, child_next, parent, child)) {
          if (!section.empty()) {
            sections.push_back(section);
            section.clear();
            parent = parent_next;
            child = child_next;
          }
        }

      }
    }
    section += line + "\n";
  }
  if (!section.empty()) {
    sections.push_back(section);
  }
  return sections;
}

inline void read_toml(std::basic_istream< typename ptree::key_type::value_type >& s,
                      ptree& main_pt)
{
  std::stringstream buffer;
  buffer << s.rdbuf();

  auto sections = split_into_sections(buffer);

  app_log(1, "\nInput Parameters");
  app_log(1, "----------------\n");
  // Parse each section and add to the main tree
  for (const auto& section : sections) {
    auto toml_data = toml::parse(section);
    std::ostringstream toml_ss;
    toml_ss << toml_data;
    app_log(1, "{}\n", toml_ss.str());

    std::stringstream json_ss;
    json_ss << toml::json_formatter(toml_data);

    ptree sec_pt;
    boost::property_tree::read_json(json_ss, sec_pt);
    for (const auto& it : sec_pt)
      main_pt.add_child(it.first, it.second);
  }
  app_log(1, "-- End of Input Parameters --\n");
}


}
#endif //COQUI_TOML_UTILITIES_HPP
