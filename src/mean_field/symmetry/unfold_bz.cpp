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


#include <nda/h5.hpp>
#include "numerics/sparse/csr_blas.hpp"
#include "numerics/nda_functions.hpp"
#include "numerics/imag_axes_ft/iaft_utils.hpp"
#include "utilities/kpoint_utils.hpp"
#include "unfold_bz.h"

namespace methods {
  void unfold_bz(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string scf_prefix) {
    std::string filename = scf_prefix + ".mbpt.h5";
    int iter;
    std::string scf_type;
    {
      h5::file file(filename, 'r');
      auto scf_grp = h5::group(file).open_group("scf");
      h5::h5_read(scf_grp, "final_iter", iter);
      auto iter_grp = scf_grp.open_group("iter" + std::to_string(iter));
      if ( iter_grp.has_dataset("Sigma_tskij") and iter_grp.has_dataset("F_skij") )
        scf_type = "dyson";
      else if ( iter_grp.has_dataset("E_ska") )
        scf_type = "quasiparticle";
      else
        utils::check(false, "unfold_bz: incorrect dataset from {}", filename);
    }

    app_log(2, "Brillouin zone unfolding");
    app_log(2, "------------------------");
    app_log(2, "  - scf type: {}", scf_type);
    app_log(2, "  - Monkhorst-Pack mesh:  ({},{},{})", mf.kp_grid()(0), mf.kp_grid()(1), mf.kp_grid()(2));
    app_log(2, "  - number of k-points:   {}", mf.nkpts());
    app_log(2, "  - number of irreducible k-points: {}", mf.nkpts_ibz());
    app_log(2, "  - input/output scf h5 file: {}", filename);

    if (scf_type == "dyson")
      unfold_dyson_solution(context, mf, filename, iter);
    else if (scf_type == "quasiparticle")
      unfold_qp_solution(context, mf, filename, iter);
    else
      utils::check(false, "unfold_bz: incorrect scf type {}", scf_type);
  }

  void unfold_dyson_solution(utils::mpi_context_t<mpi3::communicator> &context,
                             mf::MF &mf, std::string filename, long iter) {
    auto sF_skij_full = unfold_1e_hamiltonian(context, mf, filename,
                                              "scf/iter"+std::to_string(iter)+"/F_skij");
    auto sSigma_skij_tskij_full = unfold_dynamic_hamiltonian(context, mf, filename,
                                                             "scf/iter"+std::to_string(iter)+"/Sigma_tskij");

    if (context.comm.root()) {
      h5::file file(filename, 'a');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      nda::h5_write(iter_grp, "F_skij_unfold", sF_skij_full.local(), false);
      nda::h5_write(iter_grp, "Sigma_tskij_unfold", sSigma_skij_tskij_full.local(), false);
    }
  }

  void unfold_qp_solution(utils::mpi_context_t<mpi3::communicator> &context,
                          mf::MF &mf, std::string filename, long iter) {

    auto sE_ska_full = unfold_qp_energy(context, mf, filename, iter);
    auto sHeff_skij_full = unfold_1e_hamiltonian(context, mf,
                                                 filename, "scf/iter"+std::to_string(iter)+"/Heff_skij");

    if (context.comm.root()) {
      h5::file file(filename, 'a');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      nda::h5_write(iter_grp, "E_ska_unfold", sE_ska_full.local(), false);
      nda::h5_write(iter_grp, "Heff_skij_unfold", sHeff_skij_full.local(), false);
    }
  }

  auto unfold_dynamic_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context,
                                  mf::MF &mf, std::string filename, std::string dataset)
  -> math::shm::shared_array<nda::array_view<ComplexType,5>> {
    using math::shm::make_shared_array;
    using local_Array_view_5D_t = nda::array_view<ComplexType, 5>;

    int nts;
    {
      nda::array<double, 1> tau_mesh;
      h5::file file(filename, 'r');
      auto iaft_grp = h5::group(file).open_group("/imaginary_fourier_transform");
      auto tau_grp = iaft_grp.open_group("tau_mesh");
      nda::h5_read(tau_grp, "fermion", tau_mesh);
      nts = tau_mesh.shape(0);
    }
    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sSigma_tskij = make_shared_array<local_Array_view_5D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {nts, ns, nkpts_ibz, nbnd, nbnd});

    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto grp = h5::group(file);
      auto Sigma_loc = sSigma_tskij.local();
      nda::h5_read(grp, dataset, Sigma_loc);
    }
    context.node_comm.barrier();

    if (nkpts == nkpts_ibz) return sSigma_tskij;

    auto sSigma_tskij_full = make_shared_array<local_Array_view_5D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {nts, ns, nkpts, nbnd, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();
    auto S_tsk_full = sSigma_tskij_full.local();
    auto S_tsk = sSigma_tskij.local();
    sSigma_tskij_full.win().fence();
    for (size_t itsk=context.node_comm.rank(); itsk<nts*ns*nkpts; itsk+=context.node_comm.size()) {
      size_t it = itsk/(ns*nkpts); // itsk = it*ns*nkpts + is*nkpts + ik
      size_t is = (itsk/nkpts) % ns;
      size_t ik = itsk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);

      if (kp_trev(ik))
        S_tsk_full(it, is, ik, nda::ellipsis{}) = nda::conj(S_tsk(it, is, ik_ibz, nda::ellipsis{}));
      else
        S_tsk_full(it, is, ik, nda::ellipsis{}) = S_tsk(it, is, ik_ibz, nda::ellipsis{});
    }
    sSigma_tskij_full.win().fence();
    return sSigma_tskij_full;
  }

  auto unfold_1e_hamiltonian(utils::mpi_context_t<mpi3::communicator> &context,
                             mf::MF &mf, std::string filename, std::string dataset,
                             bool include_H0)
  -> math::shm::shared_array<nda::array_view<ComplexType,4>> {
    using math::shm::make_shared_array;
    using local_Array_view_4D_t = nda::array_view<ComplexType, 4>;

    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sHeff_skij = make_shared_array<local_Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {ns, nkpts_ibz, nbnd, nbnd});

    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto grp = h5::group(file);
      auto Heff_loc = sHeff_skij.local();
      nda::h5_read(grp, dataset, Heff_loc);
      if (include_H0) {
        auto sys_grp = h5::group(file).open_group("system");
        nda::array<ComplexType, 4> H0_skij;
        nda::h5_read(sys_grp, "H0_skij", H0_skij);
        sHeff_skij.local() += H0_skij;
      }
    }
    context.node_comm.barrier();
    if (nkpts == nkpts_ibz)
      return sHeff_skij;

    auto sHeff_skij_full = make_shared_array<local_Array_view_4D_t>(
        context.comm, context.internode_comm, context.node_comm,
        {ns, nkpts, nbnd, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();
    auto Heff = sHeff_skij.local();
    auto Heff_full = sHeff_skij_full.local();
    sHeff_skij_full.win().fence();
    for (size_t isk=context.node_comm.rank(); isk<ns*nkpts; isk+=context.node_comm.size()) {
      size_t is = isk/nkpts;
      size_t ik = isk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);

      if (kp_trev(ik))
        Heff_full(is, ik, nda::ellipsis{}) = nda::conj(Heff(is, ik_ibz, nda::ellipsis{}));
      else
        Heff_full(is, ik, nda::ellipsis{}) = Heff(is, ik_ibz, nda::ellipsis{});
    }
    sHeff_skij_full.win().fence();

    return sHeff_skij_full;
  }

  auto unfold_qp_energy(utils::mpi_context_t<mpi3::communicator> &context, mf::MF &mf, std::string filename, long iter)
  -> math::shm::shared_array<nda::array_view<ComplexType,3>> {
    using math::shm::make_shared_array;
    using local_Array_view_3D_t = nda::array_view<ComplexType, 3>;

    auto ns = mf.nspin();
    auto nkpts = mf.nkpts();
    auto nkpts_ibz = mf.nkpts_ibz();
    auto nbnd  = mf.nbnd();

    auto sE_ska = make_shared_array<local_Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                           {ns, nkpts_ibz, nbnd});
    if (context.node_comm.root()) {
      h5::file file(filename, 'r');
      auto iter_grp = h5::group(file).open_group("scf/iter"+std::to_string(iter));
      auto E_loc = sE_ska.local();
      if (iter_grp.has_subgroup("qp_approx"))
        nda::h5_read(iter_grp, "qp_approx/E_ska", E_loc);
      else
        nda::h5_read(iter_grp, "E_ska", E_loc);
    }
    context.node_comm.barrier();
    if (nkpts == nkpts_ibz)
      return sE_ska;

    auto sE_ska_full = make_shared_array<local_Array_view_3D_t>(context.comm, context.internode_comm, context.node_comm,
                                                                {ns, nkpts, nbnd});
    auto kp_to_ibz = mf.kp_to_ibz();
    auto kp_trev = mf.kp_trev();

    sE_ska_full.win().fence();
    for (size_t isk=context.comm.rank(); isk<ns*nkpts; isk+=context.comm.size()) {
      size_t is = isk/nkpts;
      size_t ik = isk%nkpts;
      size_t ik_ibz = kp_to_ibz(ik);
      if (kp_trev(ik))
        sE_ska_full.local()(is, ik, nda::range::all) = nda::conj(sE_ska.local()(is, ik_ibz, nda::range::all));
      else
        sE_ska_full.local()(is, ik, nda::range::all) = sE_ska.local()(is, ik_ibz, nda::range::all);
    }
    sE_ska_full.win().fence();
    sE_ska_full.all_reduce();

    return sE_ska_full;
  }

  void unfold_wfc(mf::MF &mf_sym, mf::MF &mf_nosym) {
    decltype(nda::range::all) all;
    utils::check(mf_sym.mf_type()==mf::mf_source_e::qe_source and mf_nosym.mf_type()==mf::mf_source_e::qe_source,
                 "unfold_wfc: mf_source != qe_source");
    utils::check(mf_sym.nkpts()==mf_nosym.nkpts(), "unfold_wfc: inconsistent nkpts between mf_sym and mf_nosym");
    utils::check(mf_sym.nspin()==mf_nosym.nspin(), "unfold_wfc: inconsistent nspin between mf_sym and mf_nosym");
    utils::check(mf_sym.nbnd()>=mf_nosym.nbnd(), "unfold_wfc: mf_sym.nbnd()<mf_nosym.nbnd()");
    utils::check(mf_sym.nbnd()>=mf_nosym.nbnd(), "unfold_wfc: mf_sym.nbnd()<mf_nosym.nbnd()");
    utils::check(mf_sym.mpi() == mf_nosym.mpi(),
                 "unfold_wfc: mf_sym and mf_nosym must have the same mpi context");

    auto mpi = mf_sym.mpi();

    size_t ns    = mf_sym.nspin();
    size_t nkpts = mf_sym.nkpts();
    size_t nbnd  = mf_sym.nbnd();
    size_t nbnd_nosym = mf_nosym.nbnd();
    auto wfc_g = mf_sym.wfc_truncated_grid();
    long ngm = wfc_g->size();
    auto fft2gv = wfc_g->fft_to_gv();  // maps index in fft grid to position in truncated grid

    // kpts_map between mf_sym and mf_nosym:
    // mf_nosym.kpts_crystal(ik) = mf_sym.kpts_crystal( kp_maps(ik) )
    nda::array<int, 1> kp_maps(nkpts);
    utils::calculate_kp_map(kp_maps, mf_nosym.kpts_crystal(), mf_sym.kpts_crystal());

    nda::array<ComplexType,1> *Xft = nullptr;
    nda::array<ComplexType,2> psi(nbnd,ngm);
    using view = nda::basic_array_view<RealType, 2, nda::C_layout, 'A', nda::default_accessor, nda::borrowed<>>;

    for (size_t isk2=mpi->comm.rank(); isk2<ns*nkpts; isk2+=mpi->comm.size()) {
      size_t is  = isk2 / nkpts;
      size_t ik2 = isk2 % nkpts;
      size_t ik2_sym = kp_maps(ik2);

      h5::file wfc_file(mf_nosym.outdir()+"/"+mf_nosym.prefix()+".save/wfc"+std::to_string(isk2+1)+".hdf5", 'a');
      auto grp = h5::group(wfc_file);

      // read miller indices for mf_nosym from file
      auto dset_info = h5::array_interface::get_dataset_info(grp,"/MillerIndices");
      long npw = dset_info.lengths[0]; 
      nda::array<int,2> mill(npw,3);
      nda::h5_read(grp, "/MillerIndices", mill);

      // index of miller indices in wfc_g fft grid
      nda::array<long,1> k2g(npw,0);
      utils::generate_k2g(mill,k2g,wfc_g->mesh());
      // shift k2g based on the difference between ik2 and ik2_sym
      utils::transform_k2g(false, mf_sym.symm_list(0),
                           mf_sym.kpts_crystal(ik2_sym)-mf_nosym.kpts_crystal(ik2),
                           wfc_g->mesh(), mf_sym.kpts(ik2_sym), k2g, Xft);

      // orbital at ik2_sym
      mf_sym.get_orbital_set('w', is, ik2_sym, nda::range(nbnd), psi(all, all));

      // move to the correct order provided by MillerIndices
      nda::array<ComplexType, 2> psi_k2(nbnd_nosym, npw);
      for ( auto [in,n] : itertools::enumerate(k2g) ) {
        long nn = fft2gv(n);
        utils::check(nn >= 0 and nn < ngm, 
                     "unfold_wfc: fail to map miller index to wfc_g truncated grid "); 
        psi_k2(all, in) = psi(nda::range(nbnd_nosym), nn);
      }

      // write orbitals to hdf5
      std::string attribute;
      view Ov({nbnd_nosym, 2*npw}, reinterpret_cast<RealType*>(psi_k2.data()) );
      {
        auto evc_ds = grp.open_dataset("/evc");
        h5::h5_read_attribute(evc_ds, "doc:", attribute);
      }
      nda::h5_write(grp, "/evc", Ov, false);
      {
        auto evc_ds = grp.open_dataset("/evc");
        h5::h5_write_attribute(evc_ds, "doc:", attribute);
      }
    }
    mpi->comm.barrier();
  }

  void check_is_G(double ni_d, double nj_d, double nk_d) {
    long ni_i = long(std::round(ni_d));
    long nj_i = long(std::round(nj_d));
    long nk_i = long(std::round(nk_d));

    utils::check(std::abs( ni_d - double(ni_i) ) < 1e-6, "check_is_G: not a G vector - ni: {}",ni_d);
    utils::check(std::abs( nj_d - double(nj_i) ) < 1e-6, "check_is_G: not a G vector - nj: {}",nj_d);
    utils::check(std::abs( nk_d - double(nk_i) ) < 1e-6, "check_is_G: not a G vector - nk: {}",nk_d);
  }


} // methods
