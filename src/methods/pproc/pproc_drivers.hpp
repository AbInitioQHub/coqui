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


#ifndef COQUI_PPROC_DRIVERS_HPP
#define COQUI_PPROC_DRIVERS_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"

#include "IO/AppAbort.hpp"
#include "IO/app_loggers.h"
#include "IO/ptree/ptree_utilities.hpp"

#include "utilities/mpi_context.h"
#include "mean_field/MF.hpp"
#include "mean_field/mf_utils.hpp"
#include "numerics/imag_axes_ft/IAFT.hpp"
#include "methods/SCF/scf_common.hpp"
#include "numerics/ac/ac_context.h"
#include "methods/pproc/pproc_t.h"
#include "mean_field/symmetry/unfold_bz.h"
#include "hamiltonian/one_body_hamiltonian.hpp"

namespace methods {
  /**
   * Standalone Pade analytical continuation without MPI context dependency.
   * First dimension is assumed to be the Matsubara frequency axis.
   */
  template<nda::MemoryArrayOfRank<1> mesh_iw_t, nda::MemoryArray Array_iw_t>
  nda::array<ComplexType, nda::get_rank<Array_iw_t>> pade(Array_iw_t &&A_iw, mesh_iw_t &&iw_mesh,
                                                          double w_min, double w_max, long Nw, double eta,
                                                          bool is_iw_pos_only=false, int Nfit=-1) {
    constexpr int rank = nda::get_rank<Array_iw_t>;
    static_assert(rank >= 1, "pproc_drivers.hpp::pade: input array rank must be >= 1");

    auto A_w_shape = A_iw.shape();
    A_w_shape[0] = Nw;

    auto w_grid = analyt_cont::AC_t::w_grid(w_min, w_max, Nw, eta);

    nda::array<ComplexType, rank> A_w(A_w_shape);
    analyt_cont::AC_t AC("pade");
    AC.iw_to_w(A_iw, iw_mesh, A_w, w_grid, is_iw_pos_only, Nfit);

    return A_w;
  }

  /**
   * Post-processing routines with arguments in property tree.
   * @param [INPUT] pp_type - type of post-processing, allowed options: ac, unfold
   * @param [INPUT] mf - mean-field instance
   * Requires:
   *   - prefix: prefix to the scf output h5 file.
   * Optional arguments:
   *   - outdir: location of directory with files
   * Optional arguments for pp_type = ac (analytical continuation):
   *   - ac_alg: algorithm for analytical continuation, allowed options: pade
   *   - dataset: input dataset
   *   - stats: statistics, allowed options: fermion, boson
   *   - w_min: minimum real frequency
   *   - w_max: maximum real frequency
   *   - Nw: number of real frequency points
   *   - eta: shift above the real-frequency axis
   * Optional arguments for pp_type = unfold_bz (unfold BZ from irreducible to the 1st BZ)
   */
  void post_processing(std::string pp_type, std::shared_ptr<mf::MF> mf, ptree const& pt) {
    auto mpi = mf->mpi();
    if (mpi->comm.size()%mpi->node_comm.size()!=0) {
      APP_ABORT("pproc: number of processors on each node should be the same.");
    }

    std::string err = std::string("pproc - Incorrect input - ");
    auto prefix = io::get_value<std::string>(pt,"prefix",err+"prefix");
    auto outdir = io::get_value_with_default<std::string>(pt,"outdir","./");

    if (pp_type == "ac") {

      pproc_t pp(*mpi, prefix, outdir);
      auto dataset = io::get_value_with_default<std::string>(pt,"dataset","G_tskij");

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto stats   = io::get_value_with_default<std::string>(pt,"stats","fermion");
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-10.0);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",10.0);
      auto Nw      = io::get_value_with_default<int>(pt,"Nw",5000);
      auto eta     = io::get_value_with_default<double>(pt,"eta",0.01);
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", -1);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::string_to_stats_enum(stats), Nfit, eta, w_min, w_max, Nw);
      pp.analyt_cont(*mf, ac_context, dataset);

    } else if (pp_type == "unfold_bz") {

      std::string scf_output = outdir+"/"+prefix;
      unfold_bz(*mpi, *mf, scf_output);

    } else if (pp_type == "band_interpolation") {

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);
      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<long>(pt, "iteration", -1);

      std::string scf_output = outdir+"/"+prefix;

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(iteration==0 or iteration==-1,
                     "band_interpolation: iteration = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     iteration, scf_output);

        // dump mf data to "scf_output".mbpt.h5
        auto psp = hamilt::make_pseudopot(*mf);
        // For band interpolation, the only relevant parameter is ``beta`` which will be used to 
        // determine the Fermi level in ``write_mf_data``. 
        imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }

      // Check if a QP solution exist already. 
      // If not, compute QP energies on the IBZ k-mesh from dynamic self-energy and write to the checkpoint file.
      bool heff_exists = false;
      double beta = 0.0; 
      if (mpi->comm.root()) {
        h5::file file(scf_output+".mbpt.h5", 'r');
        auto grp = h5::group(file).open_group(grp_name);
        if (iteration == -1) h5::read(grp, "final_iter", iteration);
        auto iter_grp = grp.open_group("iter"+std::to_string(iteration));
        // Check if Heff_skij dataset exists
        if (iter_grp.has_dataset("Heff_skij") or (iter_grp.has_subgroup("qp_approx") and iter_grp.open_group("qp_approx").has_dataset("Heff_skij"))) {
          heff_exists = true;
        }
        h5::read(h5::group(file), "imaginary_fourier_transform/beta", beta);
      }
      std::array<double,3> buffer = {(heff_exists)? 1.0 : 0.0, double(iteration), beta};
      mpi->comm.broadcast_n(buffer.data(), buffer.size(), 0);
      heff_exists = (buffer[0] == 1.0)? true : false;
      iteration = static_cast<int>(buffer[1]);
      beta = buffer[2];
      if (!heff_exists) {
        // Compute QP energies from dynamic self-energy on IBZ 
        qp_params_t qp_params;
        qp_params.ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
        qp_params.qp_type = io::get_value_with_default<std::string>(pt, "qp_type", "sc");
        qp_params.eta     = io::get_value_with_default<double>(pt, "qp_eta", M_PI/beta);
        qp_params.Nfit    = io::get_value_with_default<int>(pt, "qp_Nfit", 18);
        qp_params.tol     = io::get_value_with_default<double>(pt, "qp_tol", 1e-8);

        pp.compute_qp_on_ibz_kmesh(*mf, qp_params, grp_name, iteration); 
      }

      pp.wannier_interpolation(*mf, pt, wannier_file, "quasiparticle", grp_name, iteration, trans_home_cell);

    } else if (pp_type == "spectral_interpolation") {

      auto ft = imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false);

      pproc_t pp(*mpi, prefix, outdir);
      auto wannier_file = io::get_value<std::string>(pt, "wannier_file", err+"wannier_file");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", ft.nw_f()/2);
      if (Nfit <= 0 or Nfit > ft.nw_f()/2) Nfit = ft.nw_f()/2;
      // default w_min and w_max for +/- 10 eV
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-0.367);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",0.367);
      utils::check(w_min < w_max, "w_min should be smaller than w_max in spectral_interpolation.");
      // (w_max - w_min) / Nw = 0.05 => Nw = (w_max - w_min) / (0.05 / 27.2114079527)
      auto Nw      = io::get_value_with_default<int>(pt, "Nw", int((w_max - w_min) * 27.2114079527 / 0.05));
      utils::check(Nw > 1, "Nw must be > 1 in spectral_interpolation (got {}).", Nw);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermion, Nfit, eta, w_min, w_max, Nw);

      pp.spectral_interpolation(*mf, pt, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "local_dos") {

      auto ft = imag_axes_ft::read_iaft(outdir+"/"+prefix+".mbpt.h5", false);

      pproc_t pp(*mpi, prefix, outdir);
      // if wannier_file is not provided, calculate local DOS in the Bloch basis
      auto wannier_file = io::get_value_with_default<std::string>(pt, "wannier_file", "");
      auto trans_home_cell = io::get_value_with_default<bool>(pt,"translate_home_cell",false);

      auto grp_name  = io::get_value_with_default<std::string>(pt,"grp_name", "scf");
      auto iteration = io::get_value_with_default<int>(pt, "iteration", -1);

      auto ac_alg  = io::get_value_with_default<std::string>(pt,"ac_alg","pade");
      auto eta     = io::get_value_with_default<double>(pt,"eta", M_PI/ft.beta());
      auto Nfit    = io::get_value_with_default<int>(pt, "Nfit", ft.nw_f()/2);
      if (Nfit <= 0 or Nfit > ft.nw_f()/2) Nfit = ft.nw_f()/2;
      // default w_min and w_max for +/- 10 eV 
      auto w_min   = io::get_value_with_default<double>(pt,"w_min",-0.367);
      auto w_max   = io::get_value_with_default<double>(pt,"w_max",0.367);
      // default to 0.05 eV resolution => Nw = (w_max - w_min) / (0.05 / 27.2114079527)
      auto Nw      = io::get_value_with_default<int>(pt, "Nw", int((w_max - w_min) * 27.2114079527 / 0.05));
      utils::check(Nw > 1, "Nw must be > 1 in spectral_interpolation (got {}).", Nw);
      analyt_cont::ac_context_t ac_context(ac_alg, imag_axes_ft::fermion, Nfit, eta, w_min, w_max, Nw);

      pp.local_density_of_state(*mf, wannier_file, ac_context, grp_name, iteration, trans_home_cell);

    } else if (pp_type == "dump_vxc") {

      std::string scf_output = outdir+"/"+prefix;
      hamilt::dump_vxc(*mpi, *mf, scf_output);

    } else if (pp_type == "dump_hartree") {

      auto scf_iter = io::get_value_with_default<int>(pt, "scf_iter", -1);

      std::string scf_output = outdir+"/"+prefix;
      auto psp = hamilt::make_pseudopot(*mf);

      if (not std::filesystem::exists(scf_output+".mbpt.h5")) {
        utils::check(scf_iter==0 or scf_iter==-1,
                     "dump_hartree: scf_iter = {} != 0 or -1 cannot be launched if {}.mbpt.h5 does not exists!",
                     scf_iter, scf_output);

        // dump mf data to "scf_output".mbpt.h5
        // 
        imag_axes_ft::IAFT ft(pt, true, mf::wmax_from_mf(*mf));
        write_mf_data(*mf, ft, *psp.get(), scf_output);
      }

      hamilt::dump_hartree(*mpi, *mf, psp.get(), scf_output, scf_iter);

    } else {
      APP_ABORT("pproc: Unkonw post-processing type: {}", pp_type);
    }
  }

} // methods

#endif //COQUI_PPROC_DRIVERS_HPP
