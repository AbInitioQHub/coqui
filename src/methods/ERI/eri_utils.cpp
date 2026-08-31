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


#include "eri_utils.hpp"

namespace methods
{

auto make_thc(std::shared_ptr<mf::MF> mf, ptree const& pt) -> thc_reader_t {
  std::string err = "make_thc - missing required input: ";
  // required: reading just to check at this level
  auto nIpts = io::get_value_with_default<int>(pt,"nIpts",0);
  auto thresh = io::get_value_with_default<double>(pt,"thresh",0.0);
  utils::check( nIpts>0 or thresh>0.0, "Error: Must set nIpts and/or thresh");

  // generic optional
  auto storage = io::get_value_with_default<std::string>(pt,"storage","incore");
  io::tolower(storage);
  auto init = io::get_value_with_default<bool>(pt, "init", true);

  // check options
  err = std::string("make_thc - Incorrect input - ");
  utils::check(storage == "incore" or storage == "outcore", err+"storage: {}", storage);

  auto save = io::get_value_with_default<std::string>(pt,"save",(storage == "incore"?"":"./thc.eri.h5"));
  bool build_eri = (save=="" or !std::filesystem::exists(save))? true : false;

  // only mf with orbitals can be built, otherwise must be read from file
  utils::check(mf->has_orbital_set() or not build_eri, "Error in make_thc: MF types that have no orbital sets (e.g type=model), can not be built, they must be read from file. save file ({}) must be provided or could not be opened.",save);

  thc_reader_t eri = (build_eri?
                      thc_reader_t(std::move(mf), pt, false, false, init) :
                      thc_reader_t(std::move(mf), storage, save, false, init));
  return eri;
};

void make_isdf(std::shared_ptr<mf::MF> mf, ptree const& pt) {
  std::string err = "make_isdf - missing required input: ";
  // required
  auto nIpts = io::get_value_with_default<int>(pt,"nIpts",0);
  auto thresh = io::get_value_with_default<double>(pt,"thresh",1e-10);
  utils::check( nIpts>0 or thresh>0.0, "{} Must set nIpts and/or thresh", err);

  bool isdf_only = true;
  thc_reader_t isdf(std::move(mf), pt, false, isdf_only);
};


auto make_cholesky(std::shared_ptr<mf::MF> mf, ptree const& pt) -> chol_reader_t
{
  // create and return cholesky reader
  auto storage = io::get_value_with_default<std::string>(pt,"storage","outcore");
  auto path = io::get_value_with_default<std::string>(pt,"path","./");
  auto output = io::get_value_with_default<std::string>(pt,"output","chol_info.h5");
  auto read_type = io::get_value_with_default<std::string>(pt,"read_type","all");
  auto write_type = io::get_value_with_default<std::string>(pt,"write_type","multi");
  auto redo = io::get_value_with_default<bool>(pt,"overwrite",false);
  io::tolower(storage);
  io::tolower(read_type);
  io::tolower(write_type);
  utils::check(storage=="outcore", "make_cholesky - the incore version is not implemented yet!");
  utils::check(read_type == "all" or read_type == "single", "make_cholesky: Invalid value read_type:{}",read_type);
  utils::check(write_type=="multi" or write_type=="single", "make_cholesky: Invalid value write_type:{}",write_type);
  auto rtype = (read_type=="all" ? each_q : single_kpair);
  auto wtype = (write_type=="multi" ? multi_file : single_file);

  auto nq = mf->nqpts_ibz();
  bool read_chol = (chol_reader_t::check_init(path,output,nq,wtype) and not redo);

  // only mf with orbitals can be built, otherwise must be read from file
  utils::check(mf->has_orbital_set() or read_chol, "Error in make_chol: MF types that have no orbital sets (e.g type=model), can not be built, they must be read from file. output file ({}) must be provided or could not be opened.",path+"/"+output);

  return ( read_chol ?
           chol_reader_t(std::move(mf), path, output, rtype, wtype) :
           chol_reader_t(std::move(mf), pt) );
};


}