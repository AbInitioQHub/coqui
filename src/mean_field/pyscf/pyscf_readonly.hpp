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


#ifndef COQUI_PYSCF_READONLY_HPP
#define COQUI_PYSCF_READONLY_HPP

#include <memory>
#include <map>
#include <string>

#include <nda/nda.hpp>
#include <h5/h5.hpp>
#include <nda/h5.hpp>

#include "configuration.hpp"
#include "IO/app_loggers.h"
#include "IO/AppAbort.hpp"

#include "numerics/fft/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "numerics/sparse/sparse.hpp"

#include "mean_field/mf_source.hpp"
#include "mean_field/pyscf/pyscf_system.hpp"
#include "utilities/fortran_utilities.h"
#include "utilities/qe_utilities.hpp"
#include "utilities/concepts.hpp"
#include "grids/g_grids.hpp"
#include "hamiltonian/pseudo/pseudopot.h"

namespace mf {
  namespace pyscf {

    // TODO
    //   - Different G-meshes for wfc and FFT

    /**
     * Provide interface and read-only access to data from PySCF stored in hdf5 files
     *
     * Since quantities in AO basis are much more compact, we read all data once at the beginning
     */
    class pyscf_readonly {
    public:

      static constexpr mf_source_e mf_src = pyscf_source;
      static constexpr mf_source_e get_mf_source() { return mf_src; }
      static constexpr bool has_orbital_set() { return true; }

      // accessor functions
      auto mpi() const { return sys.mpi; }
      long nbnd() const { return sys.nbnd; }
      long nbnd_aux() const { return sys.nbnd_aux; }
      int nnr() const { return sys.nnr; }
      decltype(auto) kpts_crystal() { return sys.bz().kpts_crys(); }
      int nkpts_ibz() const { return sys.bz().nkpts_ibz; }
      int fft_grid_size() const { return fft_mesh(0)*fft_mesh(1)*fft_mesh(2); }
      decltype(auto) fft_grid_dim() const { return fft_mesh(); }
      decltype(auto) lattice() const { return sys.latt(); }
      decltype(auto) recv() const { return sys.recv(); }
      decltype(auto) kpts() { return sys.bz().kpts(); }
      int nkpts() const { return sys.bz().nkpts; }
      decltype(auto) kp_trev() { return sys.bz().kp_trev(); }
      const pyscf_system& get_sys() const {return sys; }
      bool has_wfc_grid() const { return false; }
      decltype(auto) wfc_truncated_grid() const {
        APP_ABORT("Error: wfc_truncated__grid not yet allowed on pyscf_readonly.");
        return std::addressof(wfc_g);
      }

    public:

      template<utils::Communicator comm_t>
      pyscf_readonly(std::shared_ptr<utils::mpi_context_t<comm_t>> mpi,
                     std::string outdir, std::string prefix):
        sys(std::move(mpi), outdir, prefix),
        ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
        h5files(sys.nspin_in_basis*sys.bz().nkpts, std::nullopt),
        wfc_g(1.0,nda::array<int,1>(3,3),sys.recv) {
        // no symmetries yet!
        dmat.clear();
        print_metadata();
      }

      template<utils::Communicator comm_t>
      pyscf_readonly(const pyscf_system& pyscf_sys):
        sys(pyscf_sys),
        ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
        h5files(sys.nspin_in_basis*sys.bz().nkpts, std::nullopt),
        wfc_g(1.0,nda::array<int,1>(3,3),sys.recv) {
        // no symmetries yet!
        dmat.clear();
        print_metadata();
      }

      pyscf_readonly(const pyscf_readonly& other):
        sys(other.sys),
        ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
        h5files(sys.nspin_in_basis*sys.bz().nkpts, std::nullopt),
        wfc_g(other.wfc_g), dmat( other.dmat ) {}

      template<utils::Communicator comm_t>
      pyscf_readonly(pyscf_system&& pyscf_sys):
        sys(std::move(pyscf_sys) ),
        ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
        h5files(sys.nspin_in_basis*sys.bz().nkpts, std::nullopt),
        wfc_g(1.0,nda::array<int,1>(3,3),sys.recv) {
        // no symmetries yet!
        dmat.clear();
        print_metadata();
      }

      pyscf_readonly(pyscf_readonly&& other):
        sys(std::move(other.sys) ),
        ecut(sys.ecutrho), fft_mesh(sys.fft_mesh),
        h5files(sys.nspin_in_basis*sys.bz().nkpts, std::nullopt),
        wfc_g(std::move(other.wfc_g)),
        dmat( std::move(other.dmat) ) {}

      ~pyscf_readonly() { close(); }

      pyscf_readonly& operator=(const pyscf_readonly& other) {
        this->sys = other.sys;
        this->dmat = other.dmat;
        close();
        this->ecut = other.ecut;
        this->fft_mesh = other.fft_mesh;
        this->wfc_g = other.wfc_g;
        return *this;
      }

      pyscf_readonly& operator=(pyscf_readonly&& other) {
        this->sys = std::move(other.sys);
        this->dmat = std::move(other.dmat);
        close();
        this->ecut = other.ecut;
        this->fft_mesh = other.fft_mesh;
        this->wfc_g = std::move(other.wfc_g);
        return *this;
      }

      void print_metadata() {
        app_log(1, "  PySCF reader");
        app_log(1, "  ------------");
        app_log(1, "  Number of spins          = {}", sys.nspin);
        app_log(1, "  Number of spins in basis = {}", sys.nspin_in_basis);
        app_log(1, "  Number of bands          = {}", sys.nbnd);
        app_log(1, "  Monkhorst-Pack mesh      = ({},{},{})", sys.bz().kp_grid(0), sys.bz().kp_grid(1), sys.bz().kp_grid(2));
        app_log(1, "  K-points                 = {} total, {} in the IBZ", sys.bz().nkpts, sys.bz().nkpts_ibz);
        app_log(1, "  Number of electrons      = {}", sys.nelec);
        if (sys.orb_on_fft_grid) {
          app_log(1, "  Orbital storage type     = uniform G-grid");
          app_log(1, "  Energy cutoff            = {0:.3f} a.u. | FFT mesh = ({1}.{2},{3})\n",
                  sys.ecutrho, sys.fft_mesh(0), sys.fft_mesh(1), sys.fft_mesh(2));
        } else {
          app_log(1, "  Orbital storage type     = non-uniform r-grid");
          app_log(1, "  Grid size                = {}\n", sys.nnr);
        }
      }

      /**
       * Get orbital values on a grid
       * if OT == 'r': orbitals on real space fft grid
       * if OT == 'g': orbitals on Fourier space grid
       * @param OT    - [INPUT] orbital type
       * @param ispin - [INPUT] spin index
       * @param ik    - [INPUT] k index
       * @param m     - [INPUT] orbital index
       * @param Orb   - [OUTPUT] orbital values stored on a FFT grid
       */
      template<nda::ArrayOfRank<1> A1D>
      void get_orbital(char OT, int _ispin, int ik, int m, A1D&& Orb, nda::range r = {-1,-1}) {
        utils::check(OT == 'r' or OT == 'g', "Unknown orbital type in pyscf_orbital::get_orbital.");
        utils::check(not (OT != 'r' and !sys.orb_on_fft_grid), "No FFT for orbitals on a real-space non-uniform grid.");
        int ispin = std::min(_ispin, sys.nspin_in_basis-1);
        int index = ispin*sys.bz().nkpts + ik;
        open_if_needed(index);
        check_dimensions(OT,0,Orb,r);
        orbital_from_h5(OT, index, m, Orb, r);
      }

      /**
       * Read multiple orbital values on a grid
       * @param OT    - [INPUT] orbital type
       * @param ispin - [INPUT] spin index
       * @param k     - [INPUT] k index
       * @param b_rng - [INPUT] nda range of orbitals 
       * @param Orb   - [OUTPUT] orbital values (nk, nb, grid_dim)
       * @param r_rng - [INPUT] nda range of fft grid index, default is entire grid
       */
      template<nda::ArrayOfRank<2> A2D>
      void get_orbital_set(char OT, int _ispin, int k, nda::range b_rng, A2D&& Orb, nda::range r = {-1,-1}) {
        static_assert(std::decay_t<A2D>::layout_t::is_stride_order_C(), "Layout mismatch.");
        utils::check(OT=='r' or OT=='g',"Unknown orbital type in pyscf_readonly::get_orbital.");
        utils::check(Orb.shape()[0] >= b_rng.size(), "Dimension mismatch.");
        utils::check(not (OT != 'r' and !sys.orb_on_fft_grid), "No FFT for orbitals on a real-space non-uniform grid.");
        int ispin = std::min(_ispin, sys.nspin_in_basis-1);
        size_t index = ispin*sys.bz().nkpts + k;
        open_if_needed(index);
        check_dimensions(OT,1,Orb,r);
        orbital_set_from_h5(OT, index, b_rng.first(),b_rng.size(), Orb, r);
      }

      /**
       * Read multiple orbital values on a grid
       * @param OT    - [INPUT] orbital type
       * @param ispin - [INPUT] spin index
       * @param k_rng - [INPUT] nda range of kpoints 
       * @param b_rng - [INPUT] nda range of orbitals 
       * @param Orb   - [OUTPUT] orbital values (nk, nb, grid_dim)
       * @param r_rng - [INPUT] nda range of fft grid index, default is entire grid including polarization
       */
      template<nda::ArrayOfRank<3> A3D>
      void get_orbital_set(char OT, int _ispin, nda::range k_rng, nda::range b_rng, A3D&& Orb, nda::range r = {-1,-1}) {
        static_assert(std::decay_t<A3D>::layout_t::is_stride_order_C(), "Layout mismatch.");
        utils::check(OT=='r' or OT=='g',"Unknown orbital type in pyscf_readonly::get_orbital.");
        int ispin = std::min(_ispin, sys.nspin_in_basis-1);
        utils::check(not (OT != 'r' and !sys.orb_on_fft_grid), "No FFT for orbitals on a real-space non-uniform grid.");
        utils::check(Orb.shape()[0] >= k_rng.size() and
                     Orb.shape()[1] >= b_rng.size(), "Dimension mismatch.");
        for( auto [ik,k] : itertools::enumerate(k_rng) ) {
          size_t index = ispin*sys.bz().nkpts + k;
          open_if_needed(index);
          check_dimensions(OT,2,Orb,r);
          orbital_set_from_h5(OT, index, b_rng.first(),b_rng.size(), Orb(ik, nda::ellipsis{}), r);
        }
      }

      /**
       * Read multiple orbital values on a grid
       * @param OT    - [INPUT] orbital type
       * @param ispin - [INPUT] spin index
       * @param k_rng - [INPUT] nda range of kpoints 
       * @param b_rng - [INPUT] nda range of orbitals 
       * @param p_rng - [INPUT] nda range of polarizations
       * @param Orb   - [OUTPUT] orbital values (nk, nb, np, grid_dim)
       * @param r_rng - [INPUT] nda range of fft grid index, default is entire grid
       */
      template<nda::ArrayOfRank<4> A4D>
      void get_orbital_set(char OT, int _ispin, nda::range k_rng, nda::range b_rng, nda::range p_rng, A4D&& Orb, nda::range r = {-1,-1}) {
        static_assert(std::decay_t<A4D>::layout_t::is_stride_order_C(), "Layout mismatch.");
        utils::check(p_rng.size()==1 and p_rng.first()==0, "npol > 1 not yet implemented in pyscf.");
        utils::check(Orb.is_contiguous(), "Layout mismatch.");
        constexpr MEMORY_SPACE MEM = memory::get_memory_space<std::decay_t<A4D>>();
        using view = ::nda::basic_array_view<ComplexType, 3, ::nda::C_layout, 'A', ::nda::default_accessor,
                                           ::nda::borrowed<to_nda_address_space(MEM)>>;
        auto O_ = view(std::array<long,3>{Orb.extent(0),Orb.extent(1),Orb.extent(3)},Orb.data());
        get_orbital_set(OT,_ispin,k_rng,b_rng,O_,r);
      }

      void open_if_needed(int isk) {
        std::string orb_dir = (sys.orb_on_fft_grid)? "Orb_fft" : "Orb_r";
        if( not h5files[isk].has_value() )
          h5files[isk] = std::make_optional<h5::file>(sys.outdir+"/"+orb_dir+"/Orb_"+std::to_string(isk)+".h5", 'r');
      }

      // close h5 handle for k-point k. If k<0, all open files are closed.
      void close(int isk=-1) {
        if( isk >= 0 and isk < h5files.size() ) {
          if ( h5files[isk].has_value() )
            h5files[isk] = std::nullopt; // this should close the file
        } else {
          for(auto& f: h5files)
            if( f.has_value() )
              f = std::nullopt;
        }
      }

      template<class Array>
      void check_dimensions(char OT, int dim, Array Orb, nda::range& r)
      {
        if(OT=='r' or OT=='g') { // fft mesh for potential based on ecutrho. size = nnr. LARGER
          if(r == nda::range{-1,-1}) r = {0,sys.nnr};
          utils::check(r.first()>=0 and r.last()<=sys.nnr, "Range error");
          utils::check(Orb.shape()[dim] == r.size(), "Wrong dimensions.");
          if(OT=='r')
            utils::check(r.size()==sys.nnr,"Error: range-based orbital access not allowed in real-space option.");
        }
      }

      /**
       * Read orbital values on a grid from hdf5 files.
       * If OT == 'g', Orb = Phi^{k}_{s,i}(G) on Fourier space fft grid
       * If OT == 'r', Orb = Phi^{k}_{s,i}(r) on real space fft grid
       * @param OT   - [INPUT] orbital type
       * @param isk  - [INPUT] combined (s,k) index that label the target h5df file
       * @param m    - [INPUT] orbital index
       * @param Orb  - [OUTPUT] orbital values (grid_dim)
       */
      template<nda::ArrayOfRank<1> A1D>
      void orbital_from_h5(char OT, int isk, int m, A1D&& Orb, nda::range r) {
        static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
        if constexpr (nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>) {
          orbital_from_h5_impl(isk, m, std::forward<A1D>(Orb), r);
        } else {
          utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch.");
          nda::array<ComplexType, 1> Ohost(r.size());
          orbital_from_h5_impl(isk, m, Ohost, r);
          Orb(nda::range(0, r.size())) = Ohost;
        }
        if (OT == 'r' and sys.orb_on_fft_grid) {
          auto Offt = nda::reshape(Orb,std::array<long,3>{fft_mesh(0),fft_mesh(1),fft_mesh(2)});
          math::fft::invfft(Offt);
        }
      }

      /**
       * Read a sets of orbital values on a grid from hdf5 files.
       * Phi^{ik}_{b0:b0+nb}(G or r) depends on the orbital type (OT)
       * @param OT  - [INPUT] orbital type
       * @param isk - [INPUT] combined (s,k) index that label the target h5df file
       * @param b0  - [INPUT] starting orbital index
       * @param nb  - [INPUT] number of orbitals
       * @param Orb - [OUTPUT] orbital values, (nb, grid_dim)
       */
      template<nda::ArrayOfRank<2> A2D>
      void orbital_set_from_h5(char OT, int isk, int b0, int nb, A2D&& Orb, nda::range r) {
        static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
        if constexpr (nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>)
        {
          orbital_set_from_h5_impl(isk, b0, nb, std::forward<A2D>(Orb), r);
        } else {
          utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
          utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
          nda::array<ComplexType,2> Ohost(nb,r.size());
          orbital_set_from_h5_impl(isk, b0, nb, Ohost, r);
          Orb(nda::range(0,nb),nda::range(0,r.size())) = Ohost;
        }
        if(OT=='r' and sys.orb_on_fft_grid) {
          // check_dimensions limits r to full range, so this is ok!
          utils::check(Orb.strides()[0] == Orb.shape()[1], "qe_readonly::orbital_set_from_h5: Layout mismatch.");
          auto Offt = nda::reshape(Orb,std::array<long,4>{nb,fft_mesh(0),fft_mesh(1),fft_mesh(2)});
          math::fft::invfft_many(Offt);
        }
      }

      template<nda::ArrayOfRank<1> A1D>
      void orbital_from_h5_impl(int index, int m, A1D&& Orb, nda::range r) {
        static_assert(nda::mem::on_host<std::decay_t<A1D>> or nda::mem::on_unified<std::decay_t<A1D>>, "Memory mismatch.");
        static_assert(nda::is_complex_v<typename std::decay_t<A1D>::value_type>, "Type mismatch");
        using view = nda::basic_array_view<ComplexType, 1, nda::C_layout, 'A', nda::default_accessor, nda::borrowed<>>;

        Orb() = ComplexType(0.0);
        utils::check(Orb.shape()[0] >= r.size(), "Dimension mismatch");
        {
          nda::array<ComplexType,1> Ok(sys.nnr);
          view Ok_v{{sys.nnr}, Ok.data()};
          h5_read(*(h5files[index]), "/Orb_" + std::to_string(index), Ok_v, std::tuple{m,nda::range::all});

          if(r.first() > 0 or r.last() < sys.nnr) {
            Orb(nda::range(0,r.size())) = Ok(r);
          } else {
            Orb() = Ok();
          }
        }
      }

      template<nda::ArrayOfRank<2> A2D>
      void orbital_set_from_h5_impl(int index, int b0, int nb, A2D&& Orb, nda::range r) {
        static_assert(nda::mem::on_host<std::decay_t<A2D>> or nda::mem::on_unified<std::decay_t<A2D>>, "Memory mismatch.");
        using view = nda::array_view<ComplexType, 2>;
        static_assert(nda::is_complex_v<typename std::decay_t<A2D>::value_type>, "Type mismatch");
        utils::check(b0 >= 0 and b0+nb <= sys.nbnd, "Index out of bounds.");
        utils::check(Orb.shape()[0] >= nb, "Dimension mismatch.");
        utils::check(Orb.shape()[1] >= r.size(), "Dimension mismatch.");
        Orb() = ComplexType(0.0);

        {
          // use fallback allocator
          nda::array<ComplexType,2> Ok(nb,sys.nnr);
          view Ok_v{{nb, sys.nnr}, Ok.data()};
          h5_read(*(h5files[index]), "/Orb_" + std::to_string(index), Ok_v,
                  std::tuple{nda::range(b0,b0+nb),nda::range::all});
          // implement enumerate???
          if(r.first() > 0 or r.last() < sys.nnr) {
            for(int b=0; b<nb; ++b)
              Orb(b, nda::range(0,r.size())) = Ok(b, r);
          } else {
            for(int b=0; b<nb; ++b)
              Orb( b, nda::range::all ) = Ok(b, nda::range::all);
          }
        }
      }

      decltype(auto) symmetry_rotation(long s, long k) const
      { 
        utils::check( false, "Symmetry operations are not allowed in pyscf backend.");
        long ns = sys.bz().qsymms.extent(0);
        long nk = sys.bz().nkpts;
        utils::check( s>0, "Symmetry index must be > 0, since s==0 is the identity and not stored.");
        utils::check(s>0 and s < ns, "out of bounds.");
        utils::check(k>=0 and k < nk, "out of bounds.");
        return std::make_tuple(false,std::addressof(dmat.at( 0 )));
      }

      void set_pseudopot(std::shared_ptr<hamilt::pseudopot> const& psp_) { psp = psp_; }
      std::shared_ptr<hamilt::pseudopot> get_pseudopot() { return psp; }

    private:
      pyscf_system sys;
      // plane wave cutoff of the FFT grid for AOs
      double ecut;
      // fft mesh compatible with ecut
      nda::stack_array<int, 3> fft_mesh;
      // h5 handles for k-point files
      std::vector<std::optional<h5::file>> h5files;
      // wfc g grid
      grids::truncated_g_grid wfc_g;

      // matrices that define symmetry relations between wavefunctions at different k-points
      std::vector< math::sparse::csr_matrix<ComplexType,HOST_MEMORY,int,int> > dmat;

      // shared ptr to pseudopot object. Not constructed here.
      // can be set from the outside to avoid recomputing. 
      std::shared_ptr<hamilt::pseudopot> psp;

    };

  }
}

#endif //COQUI_PYSCF_READONLY_HPP
