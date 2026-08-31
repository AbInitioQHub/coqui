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


#ifndef UTILITIES_APPABORT_HPP
#define UTILITIES_APPABORT_HPP

#include "configuration.hpp"
#include <iostream>
#include <string>
#include <sstream>
#include <mpi.h>
#include <source_location>
#include <cstdint>
#if defined(ENABLE_CPPTRACE)     
#include <cpptrace/cpptrace.hpp>
#endif

extern bool __app_stacktrace__;

#if defined(ENABLE_SPDLOG)

#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"
template<class... Args>
void APP_ABORT(Args&&... args)
{
  //open err_console and output message
  auto l = spdlog::get("err_console");
  if(not l) 
    auto err_logger = spdlog::stdout_color_mt("err_console");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error("        APPLICATION ABORT: Fatal Error.");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error(std::forward<Args>(args)...);
  spdlog::get("err_console")->error("**********************************************");
  // how to make cpptrace interact with spdlog???
  if(__app_stacktrace__) {
    spdlog::get("err_console")->error("**********************************************");
    spdlog::get("err_console")->error("                Stack Trace                   ");
    spdlog::get("err_console")->error("**********************************************");
#if defined(ENABLE_CPPTRACE)     
    cpptrace::generate_trace().print();
#else
    spdlog::get("err_console")->error("  Not available in current compilation. "); 
    spdlog::get("err_console")->error("  Compile with -DENABLE_CPPTRACE=ON to make this feature available.");
#endif
    spdlog::get("err_console")->error("**********************************************");
  }
  spdlog::get("err_console")->flush(); 
  // Abort
  MPI_Abort(MPI_COMM_WORLD, 1);
}

template<class... Args>
void APP_ABORT(const std::source_location& loc = std::source_location::current(), Args&&... args)
{
  auto l = spdlog::get("err_console");
  if(not l)
    auto err_logger = spdlog::stdout_color_mt("err_console");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error("        APPLICATION ABORT: Fatal Error.");
  spdlog::get("err_console")->error("**********************************************");
  spdlog::get("err_console")->error(" file_name:     {}",loc.file_name());
  spdlog::get("err_console")->error(" function_name: {}",loc.function_name());
  spdlog::get("err_console")->error(" line:          {}",loc.line());
  spdlog::get("err_console")->error(" column:        {}",loc.column());

  spdlog::get("err_console")->error(std::forward<Args>(args)...);
  spdlog::get("err_console")->error("**********************************************");
  // how to make cpptrace interact with spdlog???
  if(__app_stacktrace__) {
    spdlog::get("err_console")->error("**********************************************");
    spdlog::get("err_console")->error("                Stack Trace                   ");
    spdlog::get("err_console")->error("**********************************************");
#if defined(ENABLE_CPPTRACE)     
    cpptrace::generate_trace().print();
#else
    spdlog::get("err_console")->error("  Not available in current compilation. ");
    spdlog::get("err_console")->error("  Compile with -DENABLE_CPPTRACE=ON to make this feature available.");
#endif
    spdlog::get("err_console")->error("**********************************************");
  }
  spdlog::get("err_console")->flush();
  // Abort
  MPI_Abort(MPI_COMM_WORLD, 1);
}

#else

#include <format> 
template<class... Args>
void APP_ABORT(const std::string_view format_string, Args&&... args)
{
  //open err_console and output message
  std::cerr<<"**********************************************";
  std::cerr<<"        APPLICATION ABORT: Fatal Error.\n";
  std::cerr<<"**********************************************\n";
  std::cerr<<std::vformat(format_string,std::make_format_args(args...)) <<"\n";
  std::cerr<<"**********************************************\n";
  if(__app_stacktrace__) {
    std::cerr<<"**********************************************\n";
    std::cerr<<"                Stack Trace                   \n";
    std::cerr<<"**********************************************\n";
#if defined(ENABLE_CPPTRACE)     
    cpptrace::generate_trace().print();
#else
    std::cerr<<"  Not available in current compilation. \n"; 
    std::cerr<<"  Compile with -DENABLE_CPPTRACE=ON to make this feature available.\n";
#endif
    std::cerr<<"**********************************************\n";
  }
  std::cerr.flush(); 
  // Abort
  MPI_Abort(MPI_COMM_WORLD, 1);
}

template<class... Args>
void APP_ABORT(const std::source_location& loc, const std::string_view format_string, Args&&... args)
{
  std::cerr<<"**********************************************\n";
  std::cerr<<"        APPLICATION ABORT: Fatal Error.\n";
  std::cerr<<"**********************************************\n";
  std::cerr<<std::format(" file_name:     {}",loc.file_name()) <<"\n";
  std::cerr<<std::format(" function_name: {}",loc.function_name()) <<"\n";
  std::cerr<<std::format(" line:          {}",loc.line()) <<"\n";
  std::cerr<<std::format(" column:        {}",loc.column()) <<"\n";
  std::cerr<<std::vformat(format_string,std::make_format_args(args...)) <<"\n";
  std::cerr<<"**********************************************\n";
  if(__app_stacktrace__) {
    std::cerr<<"**********************************************\n";
    std::cerr<<"                Stack Trace                   \n";
    std::cerr<<"**********************************************\n";
#if defined(ENABLE_CPPTRACE)     
    cpptrace::generate_trace().print();
#else
    std::cerr<<"  Not available in current compilation. \n";
    std::cerr<<"  Compile with -DENABLE_CPPTRACE=ON to make this feature available.\n";
#endif
    std::cerr<<"**********************************************\n";
  }
  std::cerr.flush();
  MPI_Abort(MPI_COMM_WORLD, 1);
}


#endif // ENABLE_SPDLOG

#endif
