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


#ifndef MEANFIELD_MF_HPP 
#define MEANFIELD_MF_HPP 

#include "IO/AppAbort.hpp"
#include "utilities/check.hpp"
#include "IO/ptree/ptree_utilities.hpp"
#include "utilities/variant_helpers.hpp"
#include "utilities/concepts.hpp"

#include <map>
#include <complex>
#include "variant"

#include "mean_field/bdft/bdft_readonly.hpp"
#include "mean_field/qe/qe_readonly.hpp"
#include "mean_field/pyscf/pyscf_readonly.hpp"
#include "mean_field/model_hamiltonian/model_readonly.hpp"
#include "mean_field/mf_source.hpp"

#include "hamiltonian/pseudo/pseudopot.h"

namespace mf 
{ 

/*
 * Class to hold a generic (runtime) read-only mean field object.
 */
class MF
{
  using var_t = std::variant<bdft::bdft_readonly, qe::qe_readonly, pyscf::pyscf_readonly, model::model_readonly>;

  public:

    MF() = delete;

    // bdft interface
    explicit MF(bdft::bdft_readonly const& arg) : var(arg) {}
    explicit MF(bdft::bdft_readonly && arg) : var(std::move(arg)) {}

    MF& operator=(bdft::bdft_readonly const& arg) { var = arg; return *this; }
    MF& operator=(bdft::bdft_readonly && arg) { var = std::move(arg); return *this; }

    // qe interface
    explicit MF(qe::qe_readonly const& arg) : var(arg) {}
    explicit MF(qe::qe_readonly && arg) : var(std::move(arg)) {}

    MF& operator=(qe::qe_readonly const& arg) { var = arg; return *this; }
    MF& operator=(qe::qe_readonly && arg) { var = std::move(arg); return *this; }

    // pyscf interface
    explicit MF(pyscf::pyscf_readonly const& arg) : var(arg) {}
    explicit MF(pyscf::pyscf_readonly && arg) : var(std::move(arg) ) {}

    MF& operator=(pyscf::pyscf_readonly const& arg) { var = arg; return *this; }
    MF& operator=(pyscf::pyscf_readonly && arg) { var = std::move(arg); return *this; }

    // model hamiltonian interface
    explicit MF(model::model_readonly const& arg) : var(arg) {}
    explicit MF(model::model_readonly && arg) : var(std::move(arg) ) {}

    MF& operator=(model::model_readonly const& arg) { var = arg; return *this; }
    MF& operator=(model::model_readonly && arg) { var = std::move(arg); return *this; }    

    ~MF() = default;
    MF(MF const&) = default;
    MF(MF&&) = default;
    MF& operator=(MF const&) = default;
    MF& operator=(MF&&) = default;

    /* mpi context */
    // return a shared pointer to the mpi context
    auto mpi() const
    { return std::visit( [&](auto&& v) { return v.get_sys().mpi; }, var); }

    /* request system information copied into file */
    void save_system(std::string f) const
    { std::visit( [&](auto&& v) { v.get_sys().save(f); }, var); }
    void save_system(h5::group g) const
    { std::visit( [&](auto&& v) { v.get_sys().save(g); }, var); }

    /* return type of MF object */
    mf_source_e mf_type() const 
    { return std::visit( [&](auto&& v) { return v.get_mf_source(); }, var); }

    auto outdir() const
    { return std::visit( [&](auto&& v) { return v.get_sys().outdir; }, var); }
    auto prefix() const
    { return std::visit( [&](auto&& v) { return v.get_sys().prefix; }, var); }

    auto input_file_type() const
    { return std::visit( [&](auto&& v) { return v.get_sys().input_file_type; }, var); }
    auto filename() const
    { return std::visit( [&](auto&& v) { return v.get_sys().filename; }, var); }

    /* has orbital set */
    bool has_orbital_set() const 
    { return std::visit( [&](auto&& v) { return v.has_orbital_set(); }, var); }

    /* general */
    auto nspin() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().nspin; }, var); }
    auto npol() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().npol; }, var); }
    auto nelec() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().nelec; }, var); }
    auto noncolin() const 
    //{ return std::visit( [&](auto&& v) { return v.get_sys().noncolin; }, var); }
    { return npol() > 1; } 
    auto spinorbit() const
    { return std::visit( [&](auto&& v) { return v.get_sys().spinorbit; }, var); }
    // nuclear energy, e.g. ion-ion + frozen core + ...
    auto nuclear_energy() const
    { return std::visit( [&](auto&& v) { return v.get_sys().enuc; }, var); }
    // dimension of the system, e.g. 3 and 2.
    auto ndims() const
    { return std::visit( [&](auto&& v) { return v.get_sys().ndims; }, var); }

    /* atomic positions */
    auto number_of_atoms() const
    { return std::visit( [&](auto&& v) { return v.get_sys().natoms; }, var); }    
    auto number_of_species() const
    { return std::visit( [&](auto&& v) { return v.get_sys().nspecies; }, var); }    
    decltype(auto) atomic_id() const
    { return std::visit( [&](auto&& v) { return v.get_sys().at_ids(); }, var ); }
    int atomic_id(int i) const
    { return atomic_id()(i); } 
    decltype(auto) atomic_positions() const
    { return std::visit( [&](auto&& v) { return v.get_sys().at_pos(); }, var ); }    
    decltype(auto) atomic_positions(int i) const
    { return atomic_positions()(i,nda::range::all); }
    decltype(auto) species() const  // this will generate a copy, fix!!!
    { return std::visit( [&](auto&& v) { return v.get_sys().species; }, var ); }    

    /* FFT grid */
    auto has_wfc_grid() const 
    { return std::visit( [&](auto&& v) { return v.has_wfc_grid(); }, var); }
    auto orb_on_fft_grid() const
    { return std::visit( [&](auto&& v) { return v.get_sys().orb_on_fft_grid; }, var); }
    auto ecutrho() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().ecutrho; }, var); }
    auto fft_grid_size() const 
    { return std::visit( [&](auto&& v) { return v.fft_grid_size(); }, var); }
    decltype(auto) fft_grid_dim() const 
    { return std::visit( [&](auto&& v) { return v.fft_grid_dim(); }, var ); }
    auto fft_grid_dim(int i) const { return fft_grid_dim()(i); }
    auto nnr() const { return fft_grid_size(); } 
    decltype(auto) wfc_truncated_grid() const
    { return std::visit( [&](auto&& v) { return v.wfc_truncated_grid(); }, var ); }  

    /* cell */
    // translational vectors
    decltype(auto) lattv() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().latt(); }, var ); }
    auto lattv(int i, int j) const { return lattv()(i,j); }
    // unit cell volume
    auto volume() const {
      decltype(auto) v = this->lattv();
      return std::abs(v(0,0) * ( v(1,1)*v(2,2) - v(1,2)*v(2,1) ) -
                      v(0,1) * ( v(1,0)*v(2,2) - v(1,2)*v(2,0) ) +
                      v(0,2) * ( v(1,0)*v(2,1) - v(1,1)*v(2,0) )); 
    }
    // reciprocal vectors
    decltype(auto) recv() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().recv(); }, var ); }
    auto recv(int i, int j) const { return recv()(i,j); }
    auto madelung() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().madelung; }, var ); }

    /* kpoint info */
    auto nkpts() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nkpts; }, var ); }
    decltype(auto) kpts() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kpts(); }, var ); }
    decltype(auto) kpts(int i) const { return kpts()(i, nda::range::all); }
    auto kpts(int i, int j) const {  return kpts()(i, j); }
    decltype(auto) kpts_crystal() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kpts_crys(); }, var ); }
    decltype(auto) kpts_crystal(int i) const { return kpts_crystal()(i, nda::range::all); }
    auto kpts_crystal(int i, int j) const {  return kpts_crystal()(i, j); }
    auto nkpts_ibz() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nkpts_ibz; }, var ); }
    decltype(auto) kpts_ibz() const
    { return kpts()(nda::range(nkpts_ibz()), nda::range::all); }
    decltype(auto) kpts_ibz(int i) const { return kpts_ibz()(i, nda::range::all); }
    auto kpts_ibz(int i, int j) const {  return kpts_ibz()(i, j); }
    decltype(auto) k_weight() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().k_weight(); }, var ); }
    decltype(auto) k_weight(int i) const { return k_weight()(i); }
    auto nqpts() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nqpts; }, var ); }
    decltype(auto) Qpts() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().Qpts(); }, var ); }
    decltype(auto) Qpts(int i) const { return Qpts()(i,nda::range::all); }
    auto Qpts(int i, int j) const { return Qpts()(i,j); }
    auto nqpts_ibz() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nqpts_ibz; }, var ); }
    decltype(auto) Qpts_ibz() const
    { return Qpts()(nda::range(nqpts_ibz()), nda::range::all); }
    decltype(auto) Qpts_ibz(int i) const { return Qpts_ibz()(i,nda::range::all); }
    auto Qpts_ibz(int i, int j) const { return Qpts_ibz()(i,j); }
    decltype(auto) qk_to_k2() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qk_to_k2(); }, var ); }
    auto qk_to_k2(int i, int j) const { return qk_to_k2()(i,j); }
    decltype(auto) qminus() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qminus(); }, var ); }
    decltype(auto) twist() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().twist(); }, var ); }
    decltype(auto) kp_grid() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kp_grid(); }, var ); }

    /* symmetry */
    // symm_list should return a copy, not a reference, for safety!
    auto symm_list() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().symm_list; }, var ); }
    auto symm_list(int i) const
    { return symm_list()[i]; } 
    decltype(auto) kp_symm() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kp_symm(); }, var ); }
    auto kp_symm(int i) const
    { return kp_symm()(i); } 
    auto nkpts_trev_pairs() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nkpts_trev_pairs; }, var ); }
    decltype(auto) kp_trev() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kp_trev(); }, var ); }
    auto kp_trev(int i) const
    { return kp_trev()(i); } 
    decltype(auto) kp_trev_pair() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kp_trev_pair(); }, var ); }
    auto kp_trev_pair(int i) const
    { return kp_trev_pair()(i); } 
    decltype(auto) qp_symm() const  
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qp_symm(); }, var ); }
    auto qp_symm(int i) const
    { return qp_symm()(i); }
    decltype(auto) qp_trev() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qp_trev(); }, var ); }
    auto qp_trev(int i) const
    { return qp_trev()(i); } 
    decltype(auto) kp_to_ibz() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().kp_to_ibz(); }, var ); }
    auto kp_to_ibz(int i) const
    { return kp_to_ibz()(i); } 
    decltype(auto) qp_to_ibz() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qp_to_ibz(); }, var ); }
    auto qp_to_ibz(int i) const
    { return qp_to_ibz()(i); } 
    decltype(auto) ks_to_k() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().ks_to_k(); }, var ); }
    decltype(auto) ks_to_k(int is) const 
    { return ks_to_k()(is,nda::range::all); }
    auto ks_to_k(int is, int ik) const 
    { return ks_to_k()(is,ik); }
    decltype(auto) qsymms() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qsymms(); }, var ); }
    decltype(auto) nq_per_s() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().nq_per_s(); }, var ); }
    auto nq_per_s(int i) const
    { return nq_per_s()(i); }
    decltype(auto) Qs() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().Qs(); }, var ); }
    decltype(auto) Qs(int is) const
    { return Qs()(is,nda::range::all); }
    auto Qs(int is, int i) const
    { return Qs()(is,i); }
    decltype(auto) qs_to_q() const
    { return std::visit( [&](auto&& v) { return v.get_sys().bz().qs_to_q(); }, var ); }
    decltype(auto) qs_to_q(int is) const
    { return qs_to_q()(is,nda::range::all); }
    auto qs_to_q(int is, int iq) const
    { return qs_to_q()(is,iq); }
 

    /* bands */
    auto nbnd() const 
    { return std::visit( [&](auto&& v) { return v.nbnd(); }, var ); }
    auto nbnd_aux() const 
    { return std::visit( [&](auto&& v) { return v.nbnd_aux(); }, var ); }
    auto nspin_in_basis() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().nspin_in_basis; }, var ); }
    auto npol_in_basis() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().npol_in_basis; }, var ); }
    decltype(auto) occ() const 
    { 
      auto a = std::visit( [&](auto&& v) { return v.get_sys().occ(); }, var ); 
      return a(nda::range::all,nda::range::all,nda::range(nbnd()));
    }
    decltype(auto) eigval() const 
    { 
      auto a = std::visit( [&](auto&& v) { return v.get_sys().eigval(); }, var ); 
      return a(nda::range::all,nda::range::all,nda::range(nbnd()));
    }
    auto occ(int is, int ik, int n) const { return occ()(is,ik,n); }
    auto eigval(int is, int ik, int n) const { return eigval()(is,ik,n); }
    decltype(auto) eigval_aux() const 
    { 
      auto a = std::visit( [&](auto&& v) { return v.get_sys().eigval_aux(); }, var ); 
      return a(nda::range::all,nda::range::all,nda::range(nbnd_aux()));
    }
    auto eigval_aux(int is, int ik, int n) const { return eigval_aux()(is,ik,n); }
    decltype(auto) aux_weight() const 
    { 
      auto a = std::visit( [&](auto&& v) { return v.get_sys().aux_weight(); }, var ); 
      return a(nda::range::all,nda::range::all,nda::range(nbnd_aux()));
    }
    auto aux_weight(int is, int ik, int n) const { return aux_weight()(is,ik,n); }
    auto efermi() const 
    { return std::visit( [&](auto&& v) { return v.get_sys().efermi; }, var ); }

    decltype(auto) symmetry_rotation(long s, long k) const
    {
      auto ns = qsymms().extent(0);
      auto nk = nkpts();
      utils::check( s>0, "MF::symmetry_rotation: Symmetry index must be > 0, since s==0 is the identity and not stored.");
      utils::check( s>0 and s<ns, "out of bounds.");
      utils::check( k>=0 and k<nk, "out of bounds.");
      return std::visit( [&](auto&& v) { return v.symmetry_rotation(s,k); }, var );
    };

    // orbitals
    template<class... Args>
    void get_orbital(Args&&... args)
    { std::visit( [&](auto&& v) { v.get_orbital(std::forward<Args>(args)...); }, var ); }
    template<nda::ArrayOfRank<2> A2D>
    void get_orbital_set(char OT, int ispin, int k, nda::range b_rng, A2D&& Orb, nda::range r = {-1,-1})
    { std::visit( [&](auto&& v) { v.get_orbital_set(OT,ispin,k,b_rng,std::forward<A2D>(Orb),r); }, var ); }
    template<nda::ArrayOfRank<3> A3D>
    void get_orbital_set(char OT, int ispin, nda::range k_rng, nda::range b_rng, A3D&& Orb, nda::range r = {-1,-1})
    { std::visit( [&](auto&& v) { v.get_orbital_set(OT,ispin,k_rng,b_rng,std::forward<A3D>(Orb),r); }, var ); }
    template<nda::ArrayOfRank<4> A4D>
    void get_orbital_set(char OT, int ispin, nda::range k_rng, nda::range b_rng, nda::range p_rng, A4D&& Orb, nda::range r = {-1,-1})
    { std::visit( [&](auto&& v) { v.get_orbital_set(OT,ispin,k_rng,b_rng,p_rng,std::forward<A4D>(Orb),r); }, var ); }

    /* accessor functions for pseudopot shared pointer */
    void set_pseudopot(std::shared_ptr<hamilt::pseudopot> const& psp) 
    { return std::visit( [&](auto&& v) { v.set_pseudopot(psp); }, var ); }
    std::shared_ptr<hamilt::pseudopot> get_pseudopot() 
    { return std::visit( [&](auto&& v) { return v.get_pseudopot(); }, var ); }

    /* closes record */
    template<class... Args>
    void close() 
    { return std::visit( [&](auto&& v) { v.close(); }, var ); }
    
  private:

    std::variant<bdft::bdft_readonly,
                 qe::qe_readonly,
                 pyscf::pyscf_readonly,
                 model::model_readonly> var;
 
};

} // mf

#endif

