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


#ifndef IO_INPUTPARSER_HPP
#define IO_INPUTPARSER_HPP
#include <iostream>
#include <sstream>
#include <fstream>
#include <vector>
#include "IO/ptree/ptree_utilities.hpp"
#include "IO/ptree/toml_utilities.hpp"
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include "IO/app_loggers.h"

class InputParser
{
public:
  InputParser() = default; 
  ptree get_root() const {return pt;}
  InputParser(const InputParser& inp) : pt(inp.get_root()) {}
  InputParser(const ptree& pt0) : pt(pt0) {}
  InputParser(const std::string &input) {
    std::string extension = io::get_file_extension(input);
    if (extension=="json" or extension=="xml" or extension=="toml") {
      // valid file extension found --> input is a file name
      std::ifstream fp(input);
      this->parse(fp, extension);
      fp.close();
    } else {
      // no supported file extension --> input is a string in the json format
      std::stringstream ss;
      ss << input;
      this->parse(ss, "json");
    }
  }

  void read(std::string filename)
  { 
    try {
      std::ifstream fp(filename);
      std::string extension = io::get_file_extension(filename);
      parse(fp, extension);
      fp.close();
    } catch (std::exception const& e) {
      throw e; 
    }
  }

  void parse(std::basic_istream< typename ptree::key_type::value_type >& s, std::string extension)
  {
    // call appropriate parser based on file extension
    if (extension == "json")
    { 
      boost::property_tree::read_json(s, pt);
    } else if (extension == "xml") {
      ptree pt0;
      boost::property_tree::read_xml(s, pt0);
      pt = io::convert_xml(pt0);
    } else if (extension == "toml") {
      io::read_toml(s, pt);
    } else {
      std::string msg = "unknown extension " + extension;
      std::cout << msg << std::endl;
      throw std::runtime_error(extension.c_str());
    }
  }

  void parse(std::string s, std::string extension)
  {
    std::stringstream ss;
    ss << s;
    parse(ss, extension);
  }

private:
  ptree pt;
};

#endif
