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


#ifndef IO_PTREE_UTILITIES_HPP 
#define IO_PTREE_UTILITIES_HPP 
#include <iostream>
#include <map>
#include <sstream>
#include <fstream>
#include <vector>
#include "configuration.hpp"
#include <boost/property_tree/ptree.hpp>
#include <boost/optional.hpp>
#include "IO/AppAbort.hpp"
#include "nda/nda.hpp"

using boost::property_tree::ptree;

namespace io
{

inline ptree convert_xml(const ptree& pt0)
{
  ptree pt1;
  for(auto& it : pt0)
  {
    std::string cname = it.first;
    ptree child = it.second;
    if (cname == "<xmlattr>"){ // promote to child
      for(auto& it1 : child)
      {
        pt1.put(it1.first, it1.second.get_value<std::string>());
      }
    } else if (cname == "<xmlcomment>") { // ignore
    } else if (cname == "parameter") { // rename child by attribute "name"
      std::string pname = child.get<std::string>("<xmlattr>.name");
      std::string text = child.get_value<std::string>();
      pt1.put(pname, text);
    } else if (child.size() < 1) {
      std::string text = child.get_value<std::string>();
      pt1.put(cname, text);
    } else { // recurse
      ptree pt2 = convert_xml(child);
      if( auto str = child.get_value_optional<std::string>() )
        pt2.put_value(*str);	
      pt1.add_child(cname, pt2);
    }
  }
  return pt1;
}

/* -------------------------------- utilities ------------------------------- */
inline void str_rep(std::ostream &out, ptree const& pt, int indent=0) 
{
  for(auto& it : pt)
  {
    for (int ii=0; ii<indent; ii++) out << "  ";
    out << it.first << ": " << it.second.get_value<std::string>() << std::endl;
    str_rep(out, it.second, indent+1);
  }
}

inline void tolower(std::string& s)
{
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return std::tolower(c); });
}

inline std::string tolower_copy(std::string const& s_)
{
  std::string s(s_);
  std::transform(s.begin(), s.end(), s.begin(),
    [](unsigned char c){ return std::tolower(c); });
  return s;
}

inline std::string get_file_extension(const std::string &s)
{ // oreilly c-cookbook/0596007612/ch10s14.html
  size_t i = s.rfind('.', s.length());
  if (i == std::string::npos) return "";
  std::string ext = s.substr(i+1, s.length() - i);
  tolower(ext);
  return ext;
}

template <typename dtype>
std::vector<dtype> str2vec(const std::string s)
{
  std::stringstream ss(s);
  dtype val;
  std::vector<dtype> vec;
  while (ss>>val) vec.push_back(val);
  return vec;
}
  
inline std::string to_string(ptree const& pt)
{  
  std::stringstream ss;
  str_rep(ss, pt);
  return ss.str();
}

template<typename T>
inline bool check_exists(ptree const& pt, std::string const path) 
{
  if( boost::optional<T> val = pt.get_optional<T>(path) ) return true;
  return false;
}

inline bool check_child_exists(ptree const& pt, std::string const id)      
{
  if( auto node = pt.get_child_optional(id) ) return true;
  return false;
}

inline ptree find_child(ptree const& pt, const std::string id, const std::string message="")
{
  if(auto node = pt.get_child_optional(id)) { 
    return pt.get_child(id);
  } else {
    APP_ABORT("Error in io::find("+id+"): "+message); 
  }
  return ptree{};
};

template<typename T>
inline T get_value(ptree const& pt, const std::string id, const std::string message="")
{
  if(auto node = pt.get_child_optional(id)) { 
    if(auto v = node->get_value_optional<T>()) { 
      return *v; 
    } else {
      APP_ABORT("Error in io::get_value("+id+") - Can not extract value from node: "+message);
    }
  } else {
    APP_ABORT("Error in io::get_value("+id+") - Missing Node: "+message);
  }
  return T{}; 
};

template<typename T>
inline std::vector<T> get_array(ptree const& pt, const std::string id, const std::string message="")
{
  std::vector<T> arr;
  if(auto node = pt.get_child_optional(id)) {
    // for this to be an array that can be read by this routine, all children must be nameless and 
    // their values must be convertible to T
    for(auto const& it : *node)
    {
      std::string cname = it.first;
      if(cname != "")
        APP_ABORT("Error in io::get_array("+id+") - Found named node ({}), this is not an array: "+message, cname);
      if(auto v = it.second.get_value_optional<T>()) {
        arr.emplace_back(*v);
      } else {
        APP_ABORT("Error in io::get_array("+id+") - Problems converting value: "+message);
      }
    }
  } else {
    APP_ABORT("Error in io::get_array("+id+") - Node not found: "+message);
  }
  return arr; 
};

template<typename T>
inline T get_value_with_default(ptree const& pt, const std::string id, const T def, bool abort = true) 
{
  if(auto node = pt.get_child_optional(id)) {
    if(auto v = node->get_value_optional<T>()) {
      return *v;
    } else {
      if(abort) { 
        APP_ABORT("Error in io::get_value("+id+") - Can not extract value from node");
      } else {
        return def;
      }
    }
  } 
  return def;
};

template<typename T>
inline std::vector<T> get_array_with_default(ptree const& pt, const std::string id, std::vector<T> const def, bool abort = true) 
{
  std::vector<T> arr;
  if(auto node = pt.get_child_optional(id)) {
    // for this to be an array that can be read by this routine, all children must be nameless and 
    // their values must be convertible to T
    for(auto const& it : *node)
    { 
      std::string cname = it.first;
      if(cname != "") {
        if(abort) {
          APP_ABORT("Error in io::get_array_with_default("+id+") - Found named node ({}), this is not an array", cname);
        } else {
          return def;
        }
      }
      if(auto v = it.second.get_value_optional<T>()) {
        arr.emplace_back(*v);
      } else {
        if(abort) {
          APP_ABORT("Error in io::get_array_with_default("+id+") - Problems converting value");
        } else {
          return def;
        }
      }
    }
  } else {
    return def;
  }
  return arr;
};

template<>
inline nda::range get_value<nda::range>(ptree const& pt, const std::string id, const std::string message) 
{
  auto v = get_array<int>(pt,id,message);
  if(v.size() != 2)
    APP_ABORT("Error in io::get_value<range>("+id+") - Expect 2 integers");
  return nda::range(v[0],v[1]);
};

template<>
inline nda::range get_value_with_default<nda::range>(ptree const& pt, const std::string id, const nda::range def, [[maybe_unused]] bool abort)
{ 
  std::vector<int> vdef = {def.first(),def.last()};
  auto v = get_array_with_default<int>(pt,id,vdef);
  if(v.size() != 2) 
    APP_ABORT("Error in io::get_value_with_default<range>("+id+") - Expect 2 integers");  
  return nda::range(v[0],v[1]);
};

inline MEMORY_SPACE get_compute_space(ptree const& pt, std::string tag = "compute") {
  auto compute = io::get_value_with_default<std::string>(pt,tag,"default");
  if(compute == "default")
    return DEFAULT_MEMORY_SPACE;
  else if(compute == "cpu" or compute == "host")
    return HOST_MEMORY;
#if defined(ENABLE_DEVICE)
  else if(compute == "gpu" or compute == "device")
    return DEVICE_MEMORY;
#endif
#if defined(ENABLE_UNIFIED_MEMORY)
  else if(compute == "unified")
    return UNIFIED_MEMORY;
#endif
  else
    APP_ABORT(" Invalid input option: {} = {}",tag,compute);
  return DEFAULT_MEMORY_SPACE;
}

inline ptree make_ptree(std::map<std::string,std::string> &m)
{
  ptree pt;
  for( auto& it : m )
    pt.put(it.first,it.second);
  return pt; 
}

} // io

inline std::ostream& operator<<(std::ostream &out, const ptree &pt)
{
  io::str_rep(out, pt);
  return out;
}

#endif
