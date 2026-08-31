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


#ifndef MEANFIELD_MF_UTILS_HPP 
#define MEANFIELD_MF_UTILS_HPP 

#include <algorithm>
#include <map>

#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "utilities/concepts.hpp"
#include "IO/ptree/ptree_utilities.hpp"

#include "mean_field/MF.hpp"
#include "mean_field/bdft/bdft_readonly.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "mean_field/pyscf/pyscf_readonly.hpp"
#include "mean_field/model_hamiltonian/model_readonly.hpp"
#include "mean_field/mf_source.hpp"

namespace mf 
{ 

// MAM: leave xml_input_type as default, since it is only meaningful for qe right now, change
//      to h5_input_type when QE coqui converter is more standard
template<utils::Communicator comm_t>
inline decltype(auto) make_MF(
    const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi_context, mf_source_e mf_source,
    std::string outdir, std::string prefix, mf_input_file_type_e ftype = xml_input_type,
    double ecut = 0.0, int nbnd = -1)
{
  if(mf_source == bdft_source) {
    return MF(bdft::bdft_readonly(mpi_context, outdir, prefix, ecut, nbnd));
  } else if(mf_source == qe_source) {
    return MF(qe::qe_readonly(mpi_context, outdir, prefix, ecut, nbnd, ftype));
  } else if(mf_source == pyscf_source){
    return MF(pyscf::pyscf_readonly(mpi_context, outdir, prefix));
  } else if(mf_source == model_source){
    return MF(model::model_readonly(mpi_context, outdir, prefix, nbnd));
  }
  APP_ABORT("Error in make_MF: Unknown source.");
  // dummy return
  return MF(pyscf::pyscf_readonly(mpi_context, outdir, prefix));
}

/*
 * Creates a MF object from a provided property tree.
 * Required arguments: 
 *  - type: Type of MF object, allowed options: qe, bdft, pyscf.
 *  - prefix: prefix to file names (e.g. prefix.h5 for BDFT, prefix.xml/.save for qe, etc)
 * Optional arguments:
 *  - outdir: location of directory with files
 *  - ecut: plane wave cutoff of charge density grid, in Ha. 
 *  - nbnd: number of bands (only meaningful in QE and BDFT backends). Default: all bands. 
 *  - filetype: Default: "xml". Type of input file. Options: "xml" or "h5". Only meaningful for type="qe".  
 * Example:
 *   "mean_field":{
 *     "type": "qe",
 *     "prefix": "pwscf",
 *     "outdir": "./OUT/",
 *     "ecut": 60
 *   }
 */
template<utils::Communicator comm_t>
inline decltype(auto) make_MF(const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi_context, ptree const& pt,
                              std::string mf_type)
{
  std::string err = "mean_field - missing required input: ";
  io::tolower(mf_type);
  auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
  auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");
  auto ecut = io::get_value_with_default<double>(pt,"ecut",0.0);
  auto nbnd = io::get_value_with_default<int>(pt,"nbnd",-1);
  auto ftype = io::get_value_with_default<std::string>(pt,"filetype","h5");
  if(mf_type == "qe") {
    utils::check(ftype == "xml" or ftype == "h5", "Error: Invalid file type: {}", ftype);
    mf_input_file_type_e mf_ftype = (ftype=="xml"?xml_input_type:h5_input_type);
    return MF(qe::qe_readonly(mpi_context, outdir, prefix, ecut, nbnd, mf_ftype));
  } else if(mf_type == "bdft") {
    return MF(bdft::bdft_readonly(mpi_context, outdir, prefix, ecut, nbnd));
  } else if(mf_type == "pyscf") {
    return MF(pyscf::pyscf_readonly(mpi_context, outdir, prefix));
  } else if(mf_type == "model") {
    return MF(model::model_readonly(mpi_context, outdir, prefix, nbnd));
  }
  APP_ABORT("Error: Unknown mean_field type: {}", mf_type);
  return MF(bdft::bdft_readonly(mpi_context, outdir, prefix, ecut, nbnd));
}

/*
 * Adds a MF object to the map from the provided property tree (whose key was "mean_field").
 * If the ptree is not named (does not contain a "name" node), it generates a random name.
 * The name is returned, which can then be used to retrieve the object from the map. 
 */ 
template<utils::Communicator comm_t>
inline std::string add_mf(const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi_context,
                          ptree const& pt, std::string mf_type,
                          std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
                          bool require_name = false)
{
  static int unique_id = 0;
  std::string name;
  if(require_name)
    name = io::get_value<std::string>(pt,"name","mean_field - named input block expected.");
  else
    name = io::get_value_with_default<std::string>(pt,"name","mf_AaBbCcDd_"+std::to_string(++unique_id));

  utils::check(not mf_list.contains(name), "mean_field: Unique name are required: {}",name);
  mf_list.emplace(name,std::make_shared<mf::MF>(make_MF(mpi_context, pt, mf_type)));

  return name;
};

/*
 * Manages a list of MF objects.
 * This routine takes a property tree and searches for a node named "mean_field".
 * Only one such node can and must exist. 
 *  1. If the node has a value, it is interpreted as the name of a previously declared
 *     MF object in the list and checks that the associated object is on the list.
 *  2. If the node does not have a value, it is interpreted as the definition of a MF object.
 *     The object is created and added to the managed list.
 *  In both cases, the name of the associated MF object is returned, which is then guaranteed to
 *  exist in the list. If the ptree definition is provided and it does not contain a name,
 *  a unique name is generated. Such objects can not be reused by other input blocks since the names
 *  are assumed random. 
 */
template<utils::Communicator comm_t>
inline std::string get_mf(const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi_context,
                          ptree const& pt,
                          std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
                          std::string mf_tag="mean_field")
{
  bool found = false;
  std::string name;
  for(auto const& it : pt)
  {
    std::string cname = it.first;
    if (cname == mf_tag) {
      utils::check(not found, "Error: Only 1 {} input block allowed.", mf_tag);
      ptree mf_pt = it.second;
      auto v = mf_pt.get_value_optional<std::string>();
      if(v.has_value() and *v != "") {
        // reference to input block, check it exists in list and return
        name = *v;
        utils::check(mf_list.contains(name), 
                     "mean_field: Reference to undefined input block: {}",name);
      } else {
        // input block, add to list and return name
        utils::check(mf_pt.size()==1, "mean_field: multiple mean_field instances are not allowed.");
        auto mf_type_pt = mf_pt.begin();
        std::string mf_type = mf_type_pt->first;
        name = add_mf(mpi_context, mf_type_pt->second, mf_type, mf_list);
      }
      found=true;
    }
  }
  utils::check(found, "Error: No {} input block found.", mf_tag);
  return name;
};

/*
 * Computes a suitable value of wmax (in a.u.) from the eigenvalue bandwidth of a MF object.
 * wmax is defined as max(|emax - efermi|, |emin - efermi|), i.e. the half-bandwidth
 * measured from the Fermi energy. An optional padding factor (default 1.5) is applied
 * to provide a small safety margin for the imaginary-axis basis.
 *
 * Parameters:
 *   mf            : MF object whose eigval() is used to determine the bandwidth.
 *   padding_factor: multiplicative safety factor applied to the raw half-bandwidth (default 1.5).
 *
 * Returns: wmax in Hartree.
 */
inline double wmax_from_mf(const MF& mf, double padding_factor = 1.5)
{
  auto ev = mf.eigval();
  auto [it_min, it_max] = std::minmax_element(ev.data(), ev.data() + ev.size());
  double emin = *it_min, emax = *it_max;
  double ef   = mf.efermi();
  double wmax = std::max(std::abs(emax - ef), std::abs(emin - ef));
  if (wmax <= 0.0) {
    utils::check(false, "Error in wmax_from_mf: non-positive bandwidth detected. Cannot determine wmax from the spectrum of the MF object.");
  }
  return padding_factor * wmax;
}

} // mf

#endif
