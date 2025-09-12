#ifndef COQUI_ERI_ERI_UTILS_HPP
#define COQUI_ERI_ERI_UTILS_HPP

#include <string>
#include <filesystem>

#include "configuration.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/concepts.hpp"
#include "utilities/mpi_context.h"

#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "methods/ERI/mb_eri_context.h"

namespace methods {

  // thc builders
/**
 * Creates a thc eri object with arguments in property tree.
 * Required arguments (either thresh > 0.0 or nIpts > 0):
 *  - nIpts: "0", Number of interpolating points. 
 *  - thresh: "0.0", Threshold in the generation of interpolating points. If >0.0, either nIpts or thresh will stop the iterations. 
 *  - basis: Name of text file with DF basis. Required only with type:df.
 * Optional arguments (with default values):
 *  - save: "thc_eri.h5", Name of file with resulting thc object. For storage="incore", default is not to save.
 *  - format: "bdft". Format of resulting h5 file. {choices: "bdft"} 
 *  - storage: "incore"  Storage type. {choices: incore, outcore}
 *  - X_orbital_range: (0,nbnd), Orbital range of the left orbital. Default is the full orbital range.
 *  - Y_orbital_range: (0,nbnd), Orbital range of the right orbital. Default is the full orbital range.
 *  - basis_type: "nwchem"  Type of basis file. 
 *  - ecut: "same as MF", Plane wave cutoff used for the evaluation of coulomb matrix elements. 
 *  - matrix_block_size: 1024, Block size used in distributed arrays.
 *  - memory_frac: fraction of available memory in a node used to estimate memory requirements/utilization. 
 * Cholesky options:
 *  - chol_block_size: "8", Block size in cholesky decomposition.
 */
  auto make_thc(std::shared_ptr<mf::MF> mf, ptree const& pt) -> thc_reader_t;

  template<typename comm_t>
  std::string add_thc(const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi, ptree const& pt,
                      std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
                      std::map<std::string,std::tuple<std::string,std::unique_ptr<thc_reader_t>>> &thc_list)
  {
    static int unique_id = 0;
    std::string name;
    name = io::get_value_with_default<std::string>(pt,"name", "thc_AaBbCcDd_"+std::to_string(++unique_id));
    utils::check(not thc_list.contains(name), "interaction::thc: Unique name are required: {}",name);
    auto mf_name = mf::get_mf(mpi, pt, mf_list);
    thc_list.emplace(name,std::make_tuple(
        mf_name, std::move(std::make_unique<thc_reader_t>(make_thc(mf_list[mf_name],pt)))));
    return name;
  };

  template<typename comm_t>
  std::optional<std::string> get_thc(
      const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi,
      ptree const& base_pt, std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
      std::map<std::string,std::tuple<std::string,std::unique_ptr<thc_reader_t>>> &thc_list,
      std::string eri_tag="interaction")
  {
    bool found = false;
    std::string name;
    for(auto const& it : base_pt)
    {
      if (it.first == eri_tag) {
        ptree pt = it.second;
        auto v = pt.get_value_optional<std::string>();
        if(v.has_value() and *v != "") {
          // reference to input block, check it exists in list and return
          name = *v;
          if(thc_list.contains(name)) {
            found = true;
            break;
          }
        } else {
          auto type = io::get_value<std::string>(pt,"type",
			"interaction - missing input element: type");
          if(type=="thc") {
            utils::check(not found, "Error: Only 1 interaction input block allowed.");
            // input block, add to list and return name
            name = add_thc(mpi,pt,mf_list,thc_list);
            found=true;
            break;
          }
        }
      }
    }
    return found ? std::optional<std::string>{name} : std::nullopt;
  };
 /**
  * Creates a thc eri object with arguments in property tree.
  * Required arguments (either thresh > 0.0 or nIpts > 0):
  *  - nIpts: "0", Number of interpolating points. 
  *  - thresh: "0.0", Threshold in the generation of interpolating points. If >0.0, either nIpts or thresh will stop the iterations. 
  *  - thresh: "0.0", Threshold in cholesky decomposition. If >0.0, either nIpts or thresh will stop the iterations.
  *  - basis: Name of text file with DF basis. Required only with type:df.
  * Optional arguments (with default values):
  *  - save: "isdf.h5", Name of file with resulting thc object. For storage="incore", default is not to save.
  *  - matrix_block_size: 1024, Block size used in distributed arrays.
  *  - ecut: "same as MF", Plane wave cutoff used for the evaluation of coulomb matrix elements.
  *  - memory_frac: fraction of available memory in a node used to estimate memory requirements/utilization.
  * Cholesky options:
  *  - chol_block_size: "8", Block size in cholesky decomposition.
  */
  void make_isdf(std::shared_ptr<mf::MF> mf, ptree const& pt);


  // useful routine for unit_tests, etc
  inline ptree make_thc_ptree(double ecut = 0.0,
      int chol_block_size = 8,
      int matrix_block_size = 1024,
      double thresh = 1e-10,
      int r_blk = 1,
      double distr_tol = 0.2,
      double memory_frac = 0.75,
      bool use_least_squares = false)
  {
    ptree pt;
    auto err = std::string("make_thc_ptree - Incorrect input - ");
    utils::check(r_blk > 0, err+"Invalid r_blk:{}",r_blk);
    utils::check(chol_block_size > 0, err+"Invalid chol_block_size:{}",chol_block_size);

    pt.put("ecut", ecut);
    pt.put("thresh", thresh);
    pt.put("matrix_block_size", matrix_block_size);
    pt.put("chol_block_size", chol_block_size);
    pt.put("r_blk", r_blk);
    pt.put("distr_tol", distr_tol);
    pt.put("memory_frac", memory_frac);
    pt.put("use_least_squares", use_least_squares);
    
    return pt;
  }

  inline ptree make_thc_reader_ptree(int nIpts,
      std::string cd_dir = "",
      std::string storage = "incore",
      std::string save = "",
      std::string format = "bdft",
      double thresh = 1e-10,
      double ecut = 0.0,
      int chol_block_size = 8,
      int matrix_block_size = 1024,
      int r_blk = 1,
      double distr_tol = 0.2,
      double memory_frac = 0.75,
      bool use_least_squares = false,
      nda::range x_range = nda::range{-1,-1},
      nda::range y_range = nda::range{-1,-1},
      std::string compute = "cpu")
  {
    auto err = std::string("make_thc_reader_ptree - Incorrect input - ");
    utils::check( nIpts>0 or thresh>0.0, err+"Must set nIpts and/or thresh");
    io::tolower(storage);
    utils::check(storage == "incore" or storage == "outcore", err+"storage: {}", storage);
    io::tolower(format);
    utils::check(r_blk > 0, err+"Invalid r_blk:{}",r_blk);
    utils::check(chol_block_size > 0, err+"Invalid chol_block_size:{}",chol_block_size);

    ptree pt = make_thc_ptree(ecut,chol_block_size,matrix_block_size,thresh,r_blk,distr_tol,memory_frac,use_least_squares);

    pt.put("nIpts", nIpts);
    pt.put("storage", storage);
    pt.put("save", save);
    pt.put("format", format);
    pt.put("cd_dir", cd_dir);
    pt.put("compute", compute);
    if(x_range != nda::range{-1,-1}) {
      ptree orb;
      orb.add("", x_range.first());
      orb.add("", x_range.last());
      pt.put_child("X_orbital_range", orb);
    }
    if(y_range != nda::range{-1,-1}) {
      ptree orb;
      orb.add("", y_range.first());
      orb.add("", y_range.last());
      pt.put_child("Y_orbital_range", orb);
    }

    return pt;
  }

/*
 * Creates a cholesky eri object with arguments in property tree.
 * Requires:
 *  - type:"cholesky"
 * Optional arguments (with default values):
 *  - tol: 1e-4,  Convergence tolerance of cholesky decomposition of ERI tensor.
 *  - path: "./", Path to generated cholesky data files.
 *  - output: "chol_info.h5", Name of h5 file. 
 *            If write_type="single", this file will contain all information (meta_data + cholesky tensors).
 *            If write_type="multi", this file only contains meta_data. In this case, cholesly vectors are written
 *            in separate files, VqQ.h5 
 *  - ecut: "same as MF", Plane wave cutoff used for the evaluation of coulomb matrix elements.
 *  - chol_block_size: "32", Block size in cholesky decomposition.
 *  - read_type: "all". Whether to read all k-points at a given q-point (all), or each k-q pair separately ("single").
 *               "all" reduces IO time at the expense of extra memory requirements.
 *  - write_type: "multi" - Write separate files for the cholesky tensor of each q. 
 *                "single" - Write a single file with the cholesky tensors for all q. 
 *  - overwrite: false (default). Initialize from path+output if possible. 
 *               true. Force generation of cholesky vectors, even if oe already exists in h5 files.
 */
  auto make_cholesky(std::shared_ptr<mf::MF> mf, ptree const& pt) -> chol_reader_t;

  template<typename comm_t>
  std::string add_cholesky(
      const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi, ptree const& pt,
      std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
      std::map<std::string,std::tuple<std::string,std::unique_ptr<chol_reader_t>>> &chol_list)
  {
    static int unique_id = 0;
    std::string name;
    name = io::get_value_with_default<std::string>(pt,"name",
			"chol_AaBbCcDd_"+std::to_string(++unique_id));
    utils::check(not chol_list.contains(name), 
			"interaction::chol: Unique name are required: {}",name);
    auto mf_name = mf::get_mf(mpi, pt, mf_list);
    chol_list.emplace(name,std::make_tuple(mf_name,
		std::move(std::make_unique<chol_reader_t>(make_cholesky(mf_list[mf_name],pt)))));
    return name;
  };

  template<typename comm_t>
  std::optional<std::string> get_cholesky(
      const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi,
      ptree const& base_pt, std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
      std::map<std::string,std::tuple<std::string,std::unique_ptr<chol_reader_t>>> &chol_list,
      std::string eri_tag="interaction")
  {
    bool found = false;
    std::string name;
    for(auto const& it : base_pt)
    {
      if (it.first == eri_tag) {
        ptree pt = it.second;
        auto v = pt.get_value_optional<std::string>();
        if(v.has_value() and *v != "") {
          // reference to input block, check it exists in list and return
          name = *v;
          if(chol_list.contains(name)) {
            found = true;
            break;
          }
        } else {
          auto type = io::get_value<std::string>(pt,"type",
                        "interaction - missing input element: type");
          if(type=="cholesky") {
            utils::check(not found, "Error: Only 1 interaction input block allowed.");
            // input block, add to list and return name
            name = add_cholesky(mpi,pt,mf_list,chol_list);
            found=true;
            break;
          }
        }
      }
    }
    return found ? std::optional<std::string>{name} : std::nullopt;
  };

  // useful routine for unit_tests, etc
  inline ptree make_chol_ptree(double tol = 0.0001, double ecut = 0.0,
      int chol_block_size = 32)
  {
    ptree pt;
    auto err = std::string("make_chol_ptree - Incorrect input - ");
    utils::check(tol > 0, err+"Invalid tol:{}",tol);

    pt.put("tol", tol);
    pt.put("ecut", ecut);
    pt.put("chol_block_size", chol_block_size);

    return pt;
  }

  inline ptree make_chol_reader_ptree(double tol=0.0001,
    double ecut = 0.0,
    int chol_block = 32,
    std::string path = "./",
    std::string output = "chol_info.h5",
    chol_reading_type_e read_type = each_q,
    chol_writing_type_e write_type = multi_file,
    bool redo = false)
  {
    auto err = std::string("make_chol_reader_ptree - Incorrect input - ");
    ptree pt = make_chol_ptree(tol,ecut,chol_block);

    pt.put("storage", "outcore");
    pt.put("path", path);
    pt.put("output", output);
    pt.put("read_type", ( read_type == each_q ? "all" : "single" ) );
    pt.put("write_type", ( write_type == multi_file ? "multi" : "single" ) );
    pt.put("overwrite",redo);

    return pt;
  }


  /*
   * Get the name and the type of the integral block
   * @return std::tuple(eri_name, eri_type)
   */
  template<typename comm_t>
  auto get_eri_block(const std::shared_ptr<utils::mpi_context_t<comm_t>> &mpi, ptree const& pt,
                     std::map<std::string, std::shared_ptr<mf::MF>> &mf_list,
                     std::map<std::string,std::tuple<std::string,std::unique_ptr<thc_reader_t>>> &thc_list,
                     std::map<std::string,std::tuple<std::string,std::unique_ptr<chol_reader_t>>> &chol_list,
                     std::string eri_tag = "interaction")
  -> std::tuple<std::string, std::string>
  {
    if (auto eri_name = get_thc(mpi, pt, mf_list, thc_list, eri_tag)) 
      return std::make_tuple(*eri_name, std::string("thc") );
    else if (auto eri_name_ = get_cholesky(mpi, pt, mf_list, chol_list, eri_tag)) 
      return std::make_tuple(*eri_name_, std::string("cholesky") );
    return std::tuple(std::string(""), std::string("") );
  }
} // methods

#endif //COQUI_THC_INCORE_HPP
