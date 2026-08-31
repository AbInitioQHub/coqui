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


#ifndef MEAN_FIELD_DEFAULT_MF_HPP 
#define MEAN_FIELD_DEFAULT_MF_HPP 

#include<string>
#include<tuple>
#include <filesystem>

#include "configuration.hpp"
#include "utilities/check.hpp"
#include "utilities/type_traits.hpp"
#include "utilities/test_common.hpp"
#include "utilities/concepts.hpp"
#include "mean_field/mf_source.hpp"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "utilities/mpi_context.h"

namespace mf 
{

template<utils::Communicator comm_t>
inline decltype(auto) default_MF(std::shared_ptr<utils::mpi_context_t<comm_t>>& comm, mf::mf_source_e src,
                                 std::string outdir, std::string prefix,
                                 mf_input_file_type_e ftype = xml_input_type)
{
  return mf::make_MF(comm, src, outdir, prefix, ftype);
}

template<utils::Communicator comm_t>
inline decltype(auto) default_MF(std::shared_ptr<utils::mpi_context_t<comm_t>>& comm, mf::mf_source_e src,
                                 mf_input_file_type_e ftype = xml_input_type)
{
  auto [outdir,prefix] = utils::utest_filename(src);
  return mf::make_MF(comm, src, outdir, prefix, ftype);
}

template<utils::Communicator comm_t>
inline decltype(auto) default_MF(std::shared_ptr<utils::mpi_context_t<comm_t>>& comm, std::string src,
                                 mf_input_file_type_e ftype = xml_input_type)
{
  if(src == "model_chol") {

    auto [outdir,prefix] = utils::utest_filename("model_chol");
    return default_MF(comm, mf::model_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_si211") {

    auto [outdir,prefix] = utils::utest_filename("qe_si211");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_si111") {

    auto [outdir,prefix] = utils::utest_filename("qe_si111");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_si222_so") {

    auto [outdir,prefix] = utils::utest_filename("qe_si222_so");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_lih222") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih222");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_lih222_sym") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih222_sym");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_lih223") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih223");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_lih223_inv") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih223_inv");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_lih223_sym") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih223_sym");
    return default_MF(comm, mf::qe_source, outdir, prefix, ftype);

  } else if (src == "qe_lih222_hf") {

    auto [outdir,prefix] = utils::utest_filename("qe_lih222_hf");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_GaAs222_hf") {

    auto [outdir,prefix] = utils::utest_filename("qe_GaAs222_hf");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_GaAs222_so_hf") {

    auto [outdir,prefix] = utils::utest_filename("qe_GaAs222_so_hf");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_GaAs222_so") {

    auto [outdir,prefix] = utils::utest_filename("qe_GaAs222_so");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "qe_svo222_sym") {

    auto [outdir,prefix] = utils::utest_filename("qe_svo222_sym");
    return default_MF(comm, mf::qe_source, outdir, prefix, h5_input_type);

  } else if (src == "bdft_lih222") {

    auto [outdir,prefix] = utils::utest_filename("bdft_lih222");
    return default_MF(comm, mf::bdft_source, outdir, prefix, ftype);

  } else if (src == "bdft_lih222_sym") {

    auto [outdir,prefix] = utils::utest_filename("bdft_lih222_sym");
    return default_MF(comm, mf::bdft_source, outdir, prefix, ftype);

/*
  } else if (src == "bdft_si222") {

    auto [outdir,prefix] = utils::utest_filename("bdft_si222");
    return default_MF(comm, mf::bdft_source, outdir, prefix);
*/
  } else if (src == "pyscf_si222") {

    auto [outdir,prefix] = utils::utest_filename("pyscf_si222");
    return default_MF(comm, mf::pyscf_source, outdir, prefix, ftype);

  } else if (src == "pyscf_h2_222") {

    auto [outdir,prefix] = utils::utest_filename("pyscf_h2_222");
    return default_MF(comm, mf::pyscf_source, outdir, prefix, ftype);

  } else if (src == "pyscf_li_222u") {

    auto [outdir,prefix] = utils::utest_filename("pyscf_li_222u");
    return default_MF(comm, mf::pyscf_source, outdir, prefix);

  } else if (src == "pyscf_h2o_mol") {

    auto [outdir,prefix] = utils::utest_filename("pyscf_h2o_mol");
    return default_MF(comm, mf::pyscf_source, outdir, prefix, ftype);

  } else {
    utils::check(false, "Unrecognized test system: {}. "
                        "Available options: qe_si211, qe_lih222, bdft_si222, pyscf_si222, pyscf_h2_222, pyscf_li_222u", src);

    auto [outdir,prefix] = utils::utest_filename("pyscf_si222");
    return default_MF(comm, mf::pyscf_source, outdir, prefix, ftype);
  }
}

} 

#endif

