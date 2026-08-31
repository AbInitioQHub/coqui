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


#include <c2py/c2py.hpp>
#include "IO/app_loggers.h"
#include "mpi_handler.hpp"
#include "mpi_handler.wrap.hxx"

#include "utilities/test_input_paths.hpp"

namespace coqui_py {

  /**
   * Set verbosity levels for CoQui logging output.
   * @param mpi_handler  - [INPUT] MPI handler to ensure logging is MPI-aware, i.e. only root prints
   * @param output_level - [INPUT] Level of output verbosity (default: 2)
   * @param debug_level  - [INPUT] Level of debug verbosity (default: 0)
   */
  void set_verbosity(MpiHandler &mpi_handler, int output_level = 2, int debug_level = 0) {
    setup_loggers(mpi_handler.root(), output_level, debug_level);
    mpi_handler.get_mpi()->comm.barrier();
  }

  std::tuple<std::string,std::string> utest_filename(std::string src) {
    return utils::utest_filename(src);
  }

  void app_log(int level, const std::string& msg)   { ::app_log(level, "{}", msg); }
  void app_debug(int level, const std::string& msg) { ::app_debug(level, "{}", msg); }
  void app_warning(const std::string& msg)          { ::app_warning("{}", msg); }
  void app_error(const std::string& msg)            { ::app_error("{}", msg); }

} // coqui_py

#include "utils_module.wrap.cxx"
