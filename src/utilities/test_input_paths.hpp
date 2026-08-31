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


#ifndef UTILITIES_TEST_INPUT_PATHS_HPP
#define UTILITIES_TEST_INPUT_PATHS_HPP

#include<string>
#include<tuple>
#include <filesystem>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "mean_field/mf_source.hpp"

extern std::string qe_prefix, qe_outdir;
extern std::string bdft_prefix, bdft_outdir;
extern std::string pyscf_prefix, pyscf_outdir;

namespace utils
{

inline std::tuple<std::string,std::string> utest_filename(mf::mf_source_e src)
{
  if(src == mf::qe_source) {
    if(std::filesystem::exists(qe_outdir+"/"+qe_prefix+".xml"))
      return std::make_tuple(qe_outdir,qe_prefix);
    else
      return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                                "/tests/unit_test_files/qe/lih_kp222_nbnd16/",
                             std::string("pwscf"));
  } else if(src == mf::bdft_source) {
    if(std::filesystem::exists(bdft_outdir+"/"+bdft_prefix+".h5"))
      return std::make_tuple(bdft_outdir,bdft_prefix);
    else
      return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                                "/tests/unit_test_files/bdft/lih_kp222_nbnd16/",
                             std::string("bdft"));
  } else if(src == mf::pyscf_source) {
    if(std::filesystem::exists(pyscf_outdir+"/"+pyscf_prefix+".h5"))
      return std::make_tuple(pyscf_outdir,pyscf_prefix);
    else
      return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                                "/tests/unit_test_files/pyscf/si_kp222_krhf/",
                             std::string("pyscf"));
  } else if(src == mf::model_source ) {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                                "/tests/unit_test_files/model/nb2_chol_gamma/",
                             std::string("model"));
  }
  APP_ABORT("Error in utest_filename: Unknown source type.");
  return std::make_tuple(std::string(""),std::string(""));
}

inline std::tuple<std::string,std::string> utest_filename(std::string src)
{
  if ( src == "model_chol" ) {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                                "/tests/unit_test_files/model/nb2_chol_gamma/",
                             std::string("model"));
  } else if (src == "qe_si211") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/si_kp211_ndnb8/",
                           std::string("pwscf"));
  } else if (src == "qe_si111") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/si_kp111_nbnd8/",
                           std::string("pwscf"));
  } else if (src == "qe_si222_so") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/si_kp222_nbnd8_so/",
                           std::string("pwscf"));
  } else if (src == "qe_lih222") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp222_nbnd16/",
                           std::string("pwscf"));
  } else if (src == "qe_lih222_sym") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp222_nbnd16_sym/",
                           std::string("pwscf"));
  } else if (src == "qe_lih223") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp223_nbnd16/",
                           std::string("pwscf"));
  } else if (src == "qe_lih223_inv") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp223_nbnd16_inv_only/",
                           std::string("pwscf"));
  } else if (src == "qe_lih223_sym") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp223_nbnd16_sym/",
                           std::string("pwscf"));
  } else if (src == "qe_lih222_hf") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/lih_kp222_nbnd16_hf/",
                           std::string("pwscf"));
  } else if (src == "qe_GaAs222_hf") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/GaAs_kp222_hf/",
                           std::string("pwscf"));
  } else if (src == "qe_GaAs222_so_hf") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/GaAs_kp222_so_hf/",
                           std::string("pwscf"));
  } else if (src == "qe_GaAs222_so") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/qe/GaAs_kp222_so/",
                           std::string("pwscf"));
  } else if (src == "qe_svo222_sym") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                           "/tests/unit_test_files/qe/svo_kp222_nbnd40/out/",
                           std::string("svo"));
  } else if (src == "bdft_lih222") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/bdft/lih_kp222_nbnd16/",
                           std::string("bdft"));
  } else if (src == "bdft_lih222_sym") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/bdft/lih_kp222_nbnd16_sym/",
                           std::string("bdft"));
/*
  } else if (src == "bdft_si222") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/bdft/si_kp222_krhf/",
                           std::string("bdft"));
*/
  } else if (src == "pyscf_si222") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/pyscf/si_kp222_krhf/",
                           std::string("pyscf"));
  } else if (src == "pyscf_h2_222") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/pyscf/h2_kp222_krhf/",
                           std::string("pyscf"));
  } else if (src == "pyscf_li_222u") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/pyscf/li_kp222_kuhf/",
                           std::string("pyscf"));
  } else if (src == "pyscf_h2o_mol") {
    return std::make_tuple(std::string(PROJECT_SOURCE_DIR)+
                               "/tests/unit_test_files/pyscf/h2o_mol/",
                           std::string("pyscf"));
  } else {
    utils::check(false, "Unrecognized test system: {}. "
                        "Available options: qe_si211, qe_lih222, bdft_lih222, bdft_lih222_sym, pyscf_si222, pyscf_h2_222, pyscf_li_222u", src);
    return std::make_tuple("", "");
  }
}

} // utils

#endif
