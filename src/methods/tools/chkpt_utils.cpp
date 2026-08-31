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


#include "chkpt_utils.h"

namespace methods {
  namespace chkpt {

template<typename communicator_t>
void write_metadata(communicator_t &comm, const mf::MF &mf, const imag_axes_ft::IAFT &ft,
                     const sArray_t<Array_view_4D_t> &sH0_skij,
                     const sArray_t<Array_view_4D_t> &sS_skij,
                     std::string output) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'w');
    h5::group grp(file);

    auto sys_grp = grp.create_group("system");
    h5::h5_write(sys_grp, "number_of_spins", mf.nspin());
    h5::h5_write(sys_grp, "number_of_kpoints", mf.nkpts());
    h5::h5_write(sys_grp, "number_of_kpoints_ibz", mf.nkpts_ibz());
    h5::h5_write(sys_grp, "number_of_orbitals", mf.nbnd());
    h5::h5_write(sys_grp, "volume", mf.volume());
    h5::h5_write(sys_grp, "madelung", mf.madelung());
    nda::h5_write(sys_grp, "kp_grid", mf.kp_grid(), false);
    nda::h5_write(sys_grp, "kpoints", mf.kpts(), false);
    nda::h5_write(sys_grp, "kpoints_crys", mf.kpts_crystal(), false);
    nda::h5_write(sys_grp, "k_weight", mf.k_weight(), false);
    nda::h5_write(sys_grp, "k_trev", mf.kp_trev(), false);
    nda::h5_write(sys_grp, "kp_to_ibz", mf.kp_to_ibz(), false);
    nda::h5_write(sys_grp, "qpoints", mf.Qpts(), false);
    nda::h5_write(sys_grp, "qk_to_k2", mf.qk_to_k2(), false);
    nda::h5_write(sys_grp, "qminus", mf.qminus(), false);
    auto sH0_loc = sH0_skij.local();
    nda::h5_write(sys_grp, "H0_skij", sH0_loc, false);
    auto sSloc = sS_skij.local();
    nda::h5_write(sys_grp, "S_skij", sSloc, false);

    auto mf_grp = grp.create_group("mean_field");
    nda::h5_write(mf_grp, "eigvals", mf.eigval(), false);

    auto iaft_grp = grp.create_group("imaginary_fourier_transform");
    std::string iaft_basis = imag_axes_ft::basis_enum_to_string(ft.basis());
    h5::h5_write(iaft_grp, "basis", iaft_basis);
    h5::h5_write(iaft_grp, "prec", ft.prec());
    h5::h5_write(iaft_grp, "eps", ft.eps());
    h5::h5_write(iaft_grp, "beta", ft.beta());
    h5::h5_write(iaft_grp, "wmax", ft.wmax());
    h5::h5_write(iaft_grp, "lambda", ft.lambda());
    auto tau_grp = iaft_grp.create_group("tau_mesh");
    nda::h5_write(tau_grp, "fermion", ft.tau_mesh(), false);
    nda::h5_write(tau_grp, "boson", ft.tau_mesh_b(), false);
    auto iwn_grp = iaft_grp.create_group("iwn_mesh");
    nda::h5_write(iwn_grp, "fermion", ft.wn_mesh(), false);
    nda::h5_write(iwn_grp, "boson", ft.wn_mesh_b(), false);
  }
  comm.barrier();
}

template<typename communicator_t, typename X_t, typename Xt_t>
void dump_scf(communicator_t &comm, long iter,
              const X_t &Dm, const Xt_t &G,
              const X_t &F, const Xt_t &Sigma,
              double mu, std::string output,
              std::string input_grp, long input_iter) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    std::string iter_grp_name = "iter" + std::to_string(iter);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto scf_grp = (grp.has_subgroup("scf"))? grp.open_group("scf") : grp.create_group("scf");
    auto iter_grp = (scf_grp.has_subgroup(iter_grp_name) )?
        scf_grp.open_group(iter_grp_name) : scf_grp.create_group(iter_grp_name);

    if (input_iter==-1) input_iter = iter-1;

    h5::h5_write(scf_grp, "final_iter", iter);
    h5::h5_write(iter_grp, "greens_func_source", input_grp);
    h5::h5_write(iter_grp, "greens_func_iteration", input_iter);
    nda::h5_write(iter_grp, "G_tskij", G.local(), false);
    nda::h5_write(iter_grp, "Sigma_tskij", Sigma.local(), false);
    nda::h5_write(iter_grp, "F_skij", F.local(), false);
    nda::h5_write(iter_grp, "Dm_skij", Dm.local(), false);
    h5::h5_write(iter_grp, "mu", mu);
  }
  comm.barrier();
}

template<typename communicator_t, typename X_4D_t, typename X_3D_t>
void dump_scf(communicator_t &comm, long iter,
              const X_4D_t &Dm_skij, const X_4D_t &Heff_skij,
              const X_4D_t &MO_skia, const X_3D_t &E_ska,
              double mu, std::string output) {
  if (comm.root()) {
    std::string filename = output + ".mbpt.h5";
    std::string iter_grp_name = "iter" + std::to_string(iter);
    h5::file file(filename, 'a');
    h5::group grp(file);
    auto scf_grp = (grp.has_subgroup("scf"))? grp.open_group("scf") : grp.create_group("scf");
    auto iter_grp = (scf_grp.has_subgroup(iter_grp_name) )?
                    scf_grp.open_group(iter_grp_name) : scf_grp.create_group(iter_grp_name);

    h5::h5_write(scf_grp, "final_iter", iter);
    nda::h5_write(iter_grp, "Dm_skij", Dm_skij.local(), false);
    nda::h5_write(iter_grp, "Heff_skij", Heff_skij.local(), false);
    nda::h5_write(iter_grp, "MO_skia", MO_skia.local(), false);
    nda::h5_write(iter_grp, "E_ska", E_ska.local(), false);
    h5::h5_write(iter_grp, "mu", mu);
  }
  comm.barrier();
}

template<typename X_t, typename Xt_t>
long read_scf(mpi3::shared_communicator node_comm,
              X_t &F, Xt_t &Sigma, double &mu,
              std::string output, std::string h5_grp, long iter) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');

    auto scf_grp = h5::group(file).open_group(h5_grp);
    if (iter == -1) h5::h5_read(scf_grp, "final_iter", iter);

    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    auto Sloc = Sigma.local();
    auto Floc = F.local();
    if (iter_grp.has_dataset("F_skij")) {
      // checkpoint from a dyson scf
      nda::h5_read(iter_grp, "F_skij", Floc);
      if (iter_grp.has_dataset("Sigma_tskij"))
        nda::h5_read(iter_grp, "Sigma_tskij", Sloc);
    } else if (iter_grp.has_dataset("Heff_skij")) {
      // checkpoint from a qp scf
      auto sys_grp = h5::group(file).open_group("system");
      nda::array<ComplexType, 4> H0(F.shape());
      nda::h5_read(iter_grp, "Heff_skij", Floc);
      nda::h5_read(sys_grp, "H0_skij", H0);
      Floc -= H0;
      Sloc() = 0.0;
    } else {
      utils::check(false, "read_scf: fail to find a scf solution from {}", output+".mbpt.h5");
    }
    h5::h5_read(iter_grp, "mu", mu);
  }
  node_comm.broadcast_n(&iter, 1, 0);
  node_comm.broadcast_n(&mu, 1, 0);
  node_comm.barrier();
  return iter;
}

template<typename shared_array_t>
void read_H0(mpi3::shared_communicator node_comm, std::string output, shared_array_t &H0) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto sys_grp = h5::group(file).open_group("system");

    auto H0_loc = H0.local();
    nda::h5_read(sys_grp, "H0_skij", H0_loc);
  }
  node_comm.barrier();
}

template<typename shared_array_t>
void read_ovlp(mpi3::shared_communicator node_comm, std::string output, shared_array_t &S) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto sys_grp = h5::group(file).open_group("system");

    auto S_loc = S.local();
    nda::h5_read(sys_grp, "S_skij", S_loc);
  }
  node_comm.barrier();
}

template<typename shared_array_t>
void read_dm(mpi3::shared_communicator node_comm, std::string output, long iter, shared_array_t &Dm) {
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');
    auto scf_grp = h5::group(file).open_group("scf");

    if (iter == -1) h5::h5_read(scf_grp, "final_iter", iter);

    utils::check(scf_grp.has_subgroup("iter"+std::to_string(iter)),
                 "read_dm: \"scf/iter{}\" h5 group does not exist.", iter);

    auto Dm_loc = Dm.local();
    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    nda::h5_read(iter_grp, "Dm_skij", Dm_loc);
  }
  node_comm.barrier();
}

template<typename X_4D_t>
long read_qpscf(mpi3::shared_communicator node_comm,
                X_4D_t &Heff_skij, double &mu, std::string output) {
  long iter;
  if (node_comm.root()) {
    std::string filename = output + ".mbpt.h5";
    h5::file file(filename, 'r');

    auto scf_grp = h5::group(file).open_group("scf");
    h5::h5_read(scf_grp, "final_iter", iter);

    auto iter_grp = scf_grp.open_group("iter"+std::to_string(iter));
    auto Heffloc = Heff_skij.local();
    if (iter_grp.has_dataset("Heff_skij")) {
      // checkpoint from a qp scf
      nda::h5_read(iter_grp, "Heff_skij", Heffloc);
    } else if (iter_grp.has_dataset("F_skij")) {
      // checkpoint from a dyson scf
      nda::h5_read(iter_grp, "F_skij", Heffloc);
      nda::array<ComplexType, 4> H0(Heff_skij.shape());
      auto sys_grp = h5::group(file).open_group("system");
      nda::h5_read(sys_grp, "H0_skij", H0);
      Heffloc += H0;
      if (iter_grp.has_dataset("Sigma_tskij")) {
        app_warning("read_qpscf: Self-energy data is found in {} although qp-scf will omit this term. "
                    "Check if this is what you want!", output+".mbpt.h5");
      }
    } else {
      utils::check(false, "read_qpscf: fail to find a scf solution from {}", output+".mbpt.h5");
    }
    h5::h5_read(iter_grp, "mu", mu);
  }
  node_comm.broadcast_n(&iter, 1, 0);
  node_comm.broadcast_n(&mu, 1, 0);
  node_comm.barrier();
  return iter;
}

template<typename X_4D_t, typename X_3D_t>
void write_qpgw_results(std::string filename, long gw_iter,
                        const X_3D_t &E_ska,
                        const X_4D_t &MO_skia,
                        const X_4D_t &Vcorr_skij,
                        double mu) {
  app_log(2, "Writing QPGW results to \"scf/iter{}\"\n", gw_iter);
  if (Vcorr_skij.communicator()->root()) {
    auto E_loc = E_ska.local();
    auto MO_loc = MO_skia.local();
    auto Vcorr_loc = Vcorr_skij.local();
    h5::file file(filename, 'a');
    auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(gw_iter));
    auto qp_grp = (iter_grp.has_subgroup("qp_approx"))?
                  iter_grp.open_group("qp_approx") : iter_grp.create_group("qp_approx");
    nda::h5_write(qp_grp, "E_ska", E_loc, false);
    nda::h5_write(qp_grp, "MO_skia", MO_loc, false);
    nda::h5_write(qp_grp, "Vcorr_skij", Vcorr_loc, false);
    h5::h5_write(qp_grp, "mu", mu);
  }
  Vcorr_skij.communicator()->barrier();
}

template<typename X_4D_t>
void read_qp_hamilt_components(X_4D_t &Vhf_skij,
                               X_4D_t &Vcorr_skij,
                               double &mu,
                               std::string filename,
                               long gw_iter) {
  h5::file file(filename, 'r');
  auto scf_grp = h5::group(file).open_group("scf/iter" + std::to_string(gw_iter));
  auto qp_grp = scf_grp.open_group("qp_approx");

  h5::read(qp_grp, "mu", mu);

  if (Vcorr_skij.node_comm()->root()) {
    auto Vhf_loc = Vhf_skij.local();
    auto Vcorr_loc = Vcorr_skij.local();

    nda::h5_read(scf_grp, "F_skij", Vhf_loc);
    if (qp_grp.has_dataset("Vcorr_skij"))
      nda::h5_read(qp_grp, "Vcorr_skij", Vcorr_loc);
    else {
      // CNY: backward compatibility... Will be removed in the near future
      nda::h5_read(qp_grp, "Vcorr_skab", Vcorr_loc);
    }
  }
  Vcorr_skij.communicator()->barrier();
}

auto read_input_iterations(std::string filename)
-> std::tuple<long, long, long, long> {
  long gw_iter;
  long weiss_f_iter;
  long weiss_b_iter;
  long embed_iter;

  h5::file file(filename, 'r');

  // gw_iter
  utils::check(h5::group(file).has_subgroup("scf"),
               "embed_t::read_input_iterations: h5 group \"scf\" does not exist in {}", filename);
  auto gw_grp = h5::group(file).open_group("scf");
  h5::h5_read(gw_grp, "final_iter", gw_iter);

  // embed_iter
  if (h5::group(file).has_subgroup("embed")) {
    auto embed_grp = h5::group(file).open_group("embed");
    h5::h5_read(embed_grp, "final_iter", embed_iter);
  } else
    embed_iter = -1;

  // weiss_f_iter
  if (h5::group(file).has_subgroup("downfold_1e")) {
    auto weiss_f_grp = h5::group(file).open_group("downfold_1e");
    h5::h5_read(weiss_f_grp, "final_iter", weiss_f_iter);
  } else
    weiss_f_iter = -1;

  // weiss_b_iter
  if (h5::group(file).has_subgroup("downfold_2e")) {
    auto weiss_b_grp = h5::group(file).open_group("downfold_2e");
    h5::h5_read(weiss_b_grp, "final_iter", weiss_b_iter);
  } else
    weiss_b_iter = -1;

  return std::make_tuple(gw_iter, weiss_f_iter, weiss_b_iter, embed_iter);
}

bool is_qp_selfenergy(std::string filename) {
  long weiss_f_iter;
  h5::file file(filename, 'r');
  auto weiss_f_grp = h5::group(file).open_group("downfold_1e");
  h5::read(weiss_f_grp, "final_iter", weiss_f_iter);
  auto iter_grp = weiss_f_grp.open_group("iter"+std::to_string(weiss_f_iter));
  if (iter_grp.has_dataset("Vcorr_gw_sIab"))
    return true;
  else
    return false;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    sigma_local_exist = true;

  }
  return sigma_local_exist;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 4> &Vcorr_dc_sIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      nda::array<ComplexType, 4> &Vhf_dc_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Vcorr_dc_sIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab") &&
      weiss_f_grp->has_dataset("Vhf_dc_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Vcorr_dc_sIab", Vcorr_dc_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
    sigma_local_exist = true;
  }
  return sigma_local_exist;
}

bool read_sigma_local(nda::array<ComplexType, 5> &Sigma_imp_wsIab,
                      nda::array<ComplexType, 5> &Sigma_dc_wsIab,
                      nda::array<ComplexType, 4> &Vhf_imp_sIab,
                      nda::array<ComplexType, 4> &Vhf_dc_sIab,
                      std::string filename, long weiss_f_iter) {
  bool sigma_local_exist = false;
  h5::file file(filename, 'r');
  auto root_grp = h5::group(file);
  std::optional<h5::group> weiss_f_grp;
  if (root_grp.has_subgroup("downfold_1e")) {
    auto df_1e_grp = root_grp.open_group("downfold_1e");
    if (df_1e_grp.has_subgroup("iter" + std::to_string(weiss_f_iter)))
      weiss_f_grp = df_1e_grp.open_group("iter" + std::to_string(weiss_f_iter));
  }

  if (weiss_f_grp &&
      weiss_f_grp->has_dataset("Sigma_imp_wsIab") &&
      weiss_f_grp->has_dataset("Sigma_dc_wsIab") &&
      weiss_f_grp->has_dataset("Vhf_imp_sIab") &&
      weiss_f_grp->has_dataset("Vhf_dc_sIab")) {

    nda::h5_read(*weiss_f_grp, "Sigma_imp_wsIab", Sigma_imp_wsIab);
    nda::h5_read(*weiss_f_grp, "Sigma_dc_wsIab", Sigma_dc_wsIab);
    nda::h5_read(*weiss_f_grp, "Vhf_imp_sIab", Vhf_imp_sIab);
    nda::h5_read(*weiss_f_grp, "Vhf_dc_sIab", Vhf_dc_sIab);
    sigma_local_exist = true;
  }
  return sigma_local_exist;
}

template<typename shared_array_t>
bool read_pi_local(shared_array_t &sPi_imp, shared_array_t &sPi_dc,
                   std::string filename, long weiss_b_iter) {
  bool pi_local_exist = false;
  auto Pi_imp = sPi_imp.local();
  auto Pi_dc = sPi_dc.local();
  if (sPi_imp.node_comm()->root()) {
    h5::file file(filename, 'r');
    auto root_grp = h5::group(file);

    std::optional<h5::group> weiss_b_grp;
    if (root_grp.has_subgroup("downfold_2e")) {
      auto df_2e_grp = root_grp.open_group("downfold_2e");

      if (weiss_b_iter == -1)
        h5::h5_read(df_2e_grp, "final_iter", weiss_b_iter);

      if (df_2e_grp.has_subgroup("iter" + std::to_string(weiss_b_iter)))
        weiss_b_grp = df_2e_grp.open_group("iter" + std::to_string(weiss_b_iter));
    }

    if (weiss_b_grp && weiss_b_grp->has_dataset("Pi_imp_wabcd")
        && weiss_b_grp->has_dataset("Pi_dc_wabcd")) {
      nda::h5_read(*weiss_b_grp, "Pi_imp_wabcd", Pi_imp);
      nda::h5_read(*weiss_b_grp, "Pi_dc_wabcd", Pi_dc);
      pi_local_exist = true;
    }
  }
  sPi_imp.node_comm()->broadcast_n(&pi_local_exist, 1, 0);
  sPi_imp.communicator()->barrier();

  return pi_local_exist;
}




/** Public template instantiation **/

template void write_metadata(
    mpi3::communicator&, const mf::MF&, const imag_axes_ft::IAFT&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
    std::string);

template void dump_scf(
    mpi3::communicator&, long,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_5D_t>&,
    double, std::string, std::string, long);

template void dump_scf(
    mpi3::communicator&, long,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_3D_t>&,
    double, std::string);

template long read_scf(
    mpi3::shared_communicator, sArray_t<Array_view_4D_t>&,
    sArray_t<Array_view_5D_t>&, double&, std::string, std::string, long);

template void read_H0(mpi3::shared_communicator, std::string, sArray_t<Array_view_4D_t>&);
template void read_ovlp(mpi3::shared_communicator, std::string, sArray_t<Array_view_4D_t>&);
template void read_dm(mpi3::shared_communicator, std::string, long, sArray_t<Array_view_4D_t>&);

template long read_qpscf(
    mpi3::shared_communicator, sArray_t<Array_view_4D_t>&,
    double&, std::string);

template void write_qpgw_results(
    std::string, long, const sArray_t<Array_view_3D_t>&,
    const sArray_t<Array_view_4D_t>&, const sArray_t<Array_view_4D_t>&, double);

template void read_qp_hamilt_components(
    sArray_t<Array_view_4D_t>&, sArray_t<Array_view_4D_t>&,
    double &, std::string, long);

template bool read_pi_local(sArray_t<Array_view_5D_t>&, sArray_t<Array_view_5D_t>&, std::string, long);

  } // chkpt
} // methods
