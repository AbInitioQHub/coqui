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


#undef NDEBUG

#include "catch2/catch.hpp"

#include "configuration.hpp"
#include "mpi3/environment.hpp"
#include "mpi3/communicator.hpp"

#include "nda/nda.hpp"
#include "nda/h5.hpp"

#include "utilities/mpi_context.h"
#include "utilities/test_common.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "mean_field/default_MF.hpp"
#include "wannier/wan90.h"
#include <filesystem>

namespace wannier_tests {

  using utils::VALUE_EQUAL;
  using utils::ARRAY_EQUAL;
  namespace mpi3 = boost::mpi3;
  using namespace wannier;

  TEST_CASE("wannier90_library_mode_svo", "[wannier][library_mode]") {
    auto& mpi = utils::make_unit_test_mpi_context();

    auto test_library_mode = [&](std::shared_ptr<mf::MF> &mf, std::string test_name) {
      auto [outdir, prefix] = utils::utest_filename(test_name);

      // Create temporary outdir under current working directory and copy seedname.win there.
      auto tmp_outdir = std::filesystem::current_path() / ("wannier_tmp_" + prefix);
      auto source_win = std::filesystem::path(outdir) / ".." / "mlwf_dp" / (prefix + ".win");
      auto dest_win = tmp_outdir / (prefix + ".win");
      
      if (mpi->comm.root()) {
        if (std::filesystem::exists(tmp_outdir)) {
          std::filesystem::remove_all(tmp_outdir);
        }
        std::filesystem::create_directories(tmp_outdir);
        std::filesystem::copy_file(source_win, dest_win,
                                   std::filesystem::copy_options::overwrite_existing);
      }
      mpi->comm.barrier();
      
      // Build Wannier90 input parameters from wann.toml (mlwf_dp reference)
      ptree pt;
      pt.put("outdir", tmp_outdir.string());
      pt.put("prefix", prefix);  // CoQuí resolves seedname as outdir + "/" + prefix

      auto make_array_node = [](std::initializer_list<long> values) {
        ptree node;
        for(auto v : values) {
          ptree child;
          child.put_value<long>(v);
          node.push_back(std::make_pair("", child));
        }
        return node;
      };
      pt.add_child("shells.atoms", make_array_node({0L, 1L, 2L, 3L}));
      pt.add_child("shells.sort", make_array_node({0L, 1L, 1L, 1L}));
      pt.add_child("shells.l", make_array_node({2L, 1L, 1L, 1L}));
      pt.add_child("shells.dim", make_array_node({5L, 3L, 3L, 3L}));
      pt.add_child("shells.SO", make_array_node({0L, 0L, 0L, 0L}));
      pt.add_child("shells.irrep", make_array_node({0L, 0L, 0L, 0L}));

      // Sanity check: ensure shells are present in pt
      auto shell_atoms = io::get_array_with_default<long>(pt, "shells.atoms", {});
      REQUIRE(shell_atoms.size() == 4);
      auto shell_dim = io::get_array_with_default<long>(pt, "shells.dim", {});
      REQUIRE(shell_dim.size() == 4);
      
      wannier90_library_mode(*mf, pt);
      mpi->comm.barrier();

      // Load the generated output file
      auto output_file = tmp_outdir / (prefix + ".mlwf.h5");
      nda::array<ComplexType, 4> hopping_out;
      nda::array<ComplexType, 5> proj_mat_out;
      nda::array<double, 3> wan_centres_out;

      {
        h5::file file(output_file.string(), 'r');
        auto grp = h5::group(file).open_group("dft_input");
        nda::h5_read(grp, "hopping", hopping_out);
        nda::h5_read(grp, "proj_mat", proj_mat_out);
        nda::h5_read(grp, "wan_centres", wan_centres_out);
      }

      // Load the reference file (disentangled reference in mlwf_dp)
      std::string reference_file = outdir + "/../mlwf_dp/" + prefix + ".mlwf.h5";
      nda::array<ComplexType, 4> hopping_ref;
      nda::array<ComplexType, 5> proj_mat_ref;
      nda::array<double, 3> wan_centres_ref;

      {
        h5::file file(reference_file, 'r');
        auto grp = h5::group(file).open_group("dft_input");
        nda::h5_read(grp, "hopping", hopping_ref);
        nda::h5_read(grp, "proj_mat", proj_mat_ref);
        nda::h5_read(grp, "wan_centres", wan_centres_ref);
      }

      // Compare shapes
      REQUIRE(hopping_out.shape() == hopping_ref.shape());
      REQUIRE(proj_mat_out.shape() == proj_mat_ref.shape());
      REQUIRE(wan_centres_out.shape() == wan_centres_ref.shape());

      // Compare data with appropriate tolerances
      // Projection matrix comparison with relaxed tolerance due to numerical precision
      ARRAY_EQUAL(proj_mat_out, proj_mat_ref, 1e-6, 1e-8);

      // Hopping/eigenvalue comparison (should be very close)
      ARRAY_EQUAL(hopping_out, hopping_ref, 1e-6, 1e-8);

      // Wannier centres comparison
      ARRAY_EQUAL(wan_centres_out, wan_centres_ref, 1e-6, 1e-8);

      app_log(0, "wannier_tests: Successfully compared generated {}.mlwf.h5 with reference", prefix);
      mpi->comm.barrier();
      
      // Clean up: remove the temporary output directory and all generated Wannier90 files.
      if (mpi->comm.root()) {
        if (std::filesystem::exists(tmp_outdir)) {
          std::filesystem::remove_all(tmp_outdir);
        }
      }
      mpi->comm.barrier();
    };

    SECTION("sym_svo") {
      auto mf = std::make_shared<mf::MF>(mf::default_MF(mpi, "qe_svo222_sym"));
      test_library_mode(mf, "qe_svo222_sym");
    }
  }

} // wannier_tests
