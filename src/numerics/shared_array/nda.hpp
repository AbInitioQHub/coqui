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


#ifndef NUMERICS_SHARED_ARRAY_NDA_HPP
#define NUMERICS_SHARED_ARRAY_NDA_HPP

#include "configuration.hpp"
#include "mpi3/communicator.hpp"
#include "mpi3/shared_window.hpp"
#include "nda/nda.hpp"
#include "numerics/distributed_array/nda.hpp"
#include "utilities/mpi_context.h"

#include "utilities/check.hpp"

namespace math {
  namespace shm {

    namespace mpi3 = boost::mpi3;

    // TODO
    //   - merge this with distributed array
    /**
     * A simple wrapper for nda arrays and MPI shared memory (from boost::mpi3)
     * still in design stage...
     */
    // nda::Array or nda::MemoryArray?
    template<::nda::MemoryArray Array_base_t>
    class shared_array {
    public:
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      static constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;

      static constexpr bool is_stride_order_Fortran() noexcept {
        return Array_view_t::layout_t::is_stride_order_Fortran();
      }
      static constexpr bool is_stride_order_C() noexcept {
        return Array_view_t::layout_t::is_stride_order_C();
      }

    private:
      using darray_t = math::nda::detail::darray<rank, mpi3::communicator>;
      static_assert ( Array_view_t::layout_t::is_stride_order_Fortran()
        or Array_view_t::layout_t::is_stride_order_C(), "Ordering mismatch.");
    public:
      shared_array(mpi3::shared_communicator* node_comm,
                   std::array<long, rank> shape) :
          _node_comm(node_comm),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*node_comm, (node_comm->root()) ? _size : 0))
      {
        check_and_init();
      }

      shared_array(mpi3::communicator *gcomm,
                   mpi3::communicator *internode_comm,
                   mpi3::shared_communicator *node_comm,
                   std::array<long, rank> shape):
          _gcomm(gcomm), _internode_comm(internode_comm), _node_comm(node_comm),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*node_comm, (node_comm->root()) ? _size : 0))
      {
        check_and_init();
      }
 
      shared_array(utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> &ctxt,
                   std::array<long, rank> shape):
          _gcomm(std::addressof(ctxt.comm)), 
          _internode_comm(std::addressof(ctxt.internode_comm)), 
          _node_comm(std::addressof(ctxt.node_comm)),
          _size(std::accumulate(shape.cbegin(), shape.cend(), (mpi3::size_t)1, std::multiplies<>{})),
          _shape(shape),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0))
      { 
        check_and_init();
      }

      shared_array(const shared_array &other) :
          _gcomm(other.communicator()), 
          _internode_comm(other.internode_comm()), 
          _node_comm(other.node_comm()),
          _size(other.size()),
          _shape(other.shape()),
          _win(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0))
      {
        check_and_init();
        node_sync();
        if (_node_comm->root()) 
          this->local() = other.local(); 
        node_sync(); 
      } 
      shared_array(shared_array &&other) = default;

      shared_array& operator=(const shared_array &other) {
        _gcomm = other.communicator();
        _internode_comm = other.internode_comm();
        _node_comm = other.node_comm();
        _size = other.size();
        _shape = other.shape();
        _win = std::move(std::make_unique<mpi3::shared_window<value_type>>(*_node_comm, (_node_comm->root()) ? _size : 0));
        node_sync();
        if (_node_comm->root()) 
          this->local() = other.local(); 
        node_sync(); 
        return *this;
      }
      shared_array& operator=(shared_array &&other) = default;

      ~shared_array() = default; 

      void check_and_init() {
        utils::check(_win->base(0) != nullptr, "shm::shared_array: win.base(0) == nullptr");
        utils::check(_win->size(0) == _size, "shm::shared_array: win.size(0) has incorrect dimension");
        if (_node_comm->size() > 1) {
          utils::check(_win->size(1) == 0, "shm::shared_array: win.size(!=0) has incorrect dimension");
        }
        // initialize array to 0.0
        set_zero();
      }

      void set_zero() {
        node_sync();
        auto[origin_i, end_i] = itertools::chunk_range(0, _size, _node_comm->size(), _node_comm->rank());
        ::nda::range i_range(origin_i, end_i);
        auto _array = Array_view_t(_shape, (value_type*) _win->base(0));
        auto array_1D = ::nda::reshape(_array, std::array<long, 1>{_size});
        _win->fence();
        array_1D(i_range) = value_type(0.0);
        _win->fence();
        node_sync();
      }

      void all_reduce() {
        node_sync();
        if (_node_comm->root()) {
          // split all_reduce() to avoid mpi count overflow
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type*)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size)? size_t(1e9) : _size-shift;
            _internode_comm->all_reduce_in_place_n(start, count, std::plus<>{});
          }
        }
        node_sync();
      }

      void broadcast_to_nodes(int src_node) {
        node_sync();
        if (_node_comm->root()) {
          for (size_t shift=0; shift<_size; shift+=size_t(1e9)) {
            value_type *start = (value_type *)_win->base(0) + shift;
            size_t count = (shift+size_t(1e9) < _size) ? size_t(1e9) : _size-shift;
            _internode_comm->broadcast_n(start, count, src_node);
          }
        }
        node_sync();
      }

      void node_sync() {
        _node_comm->barrier();
        _win->sync();
      }

      mpi3::shared_communicator* node_comm() const { return _node_comm; }
      mpi3::communicator* communicator() const { return _gcomm; }
      mpi3::communicator* internode_comm() const { return _internode_comm; }
      mpi3::shared_window<value_type>& win() { return *_win; }

      auto const& shape() const { return _shape; }
      // for ::nda::get_rank() interface
      auto const& global_shape() const { return _shape; }
      auto size() const { return _size; }

      auto local() { return Array_view_t(_shape, (value_type*) _win->base(0)); }
      auto local() const { return Array_view_t(_shape, (value_type*) _win->base(0)); }

    protected:
      mpi3::communicator *_gcomm = nullptr;
      mpi3::communicator *_internode_comm = nullptr;
      mpi3::shared_communicator *_node_comm = nullptr;
      mpi3::size_t _size;
      std::array<long, rank> _shape;
      std::unique_ptr<mpi3::shared_window<value_type>> _win;

    };

    template<::nda::MemoryArray Array_base_t>
    class distributed_shared_array : public shared_array<Array_base_t> {
    public:
      using Array_view_t = decltype(std::declval<std::decay_t < Array_base_t>>()());
      static constexpr int rank = ::nda::get_rank<Array_view_t>;
      using value_type = typename std::decay_t<Array_view_t>::value_type;

    private:
      using darray_t = math::nda::detail::darray<rank, mpi3::communicator>;

    public:
      /**
       * Constructor for distributed array among nodes
       */
      distributed_shared_array(mpi3::communicator *gcomm,
                               mpi3::communicator *internode_comm,
                               mpi3::shared_communicator *node_comm,
                               std::array<long, rank> grid,    // processor grid
                               std::array<long, rank> gshape,  // global shape
                               std::array<long, rank> lshape,  // local shape
                               std::array<long, rank> origin): // index origin of local array
          shared_array<Array_base_t>(gcomm, internode_comm, node_comm, lshape),
          _base(internode_comm, grid, gshape, origin, lshape) {
      }

      ~distributed_shared_array() = default;

      std::array<long, rank> const& global_shape() const { return _base.gextents; }
      std::array<long, rank> const& origin() const { return _base.lorigin; }
      std::array<long, rank> const& grid() const { return _base.grid; }

      auto const& local_shape() const { return this->_shape; }
      auto local_range(int dim) const {
        utils::check(dim >= 0 and dim < rank, "shared_array::range: Out or range d:{}", dim);
        return ::nda::range(_base.lorigin[dim], _base.lorigin[dim] + this->_shape[dim]);
      }

      void reset_loc_origin(int dim, long new_origin) {
        _base.lorigin[dim] = new_origin;
      }

      bool full_coverage() const { return _base.full_coverage_impl(this->_shape);}

    protected:
      darray_t _base;
    };

    /**
     * Shared memory array, one copy per node
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(mpi3::communicator &gcomm,
             mpi3::communicator &internode_comm,
             mpi3::shared_communicator &node_comm,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;

      return Array_t(std::addressof(gcomm),
                     std::addressof(internode_comm),
                     std::addressof(node_comm),
                     shape);
    }

    /**
     * Shared memory array, one copy per node
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(utils::mpi_context_t<mpi3::communicator,mpi3::shared_communicator> &ctxt,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;

      return Array_t(std::addressof(ctxt.comm),
                     std::addressof(ctxt.internode_comm),
                     std::addressof(ctxt.node_comm),
                     shape);
    }    

    /**
     * Shared memory array, one copy per node.
     * This should only be used for read-only array, i.e. no all_reduce between nodes.
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_shared_array(mpi3::shared_communicator &node_comm,
                           std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> shape) {
      using Array_t = shared_array<Array_base_t>;
      return Array_t(std::addressof(node_comm), shape);
    }

    /**
     * Distributed array among nodes.
     * Local arrays are stored in shared memory on each node.
     */
    template<::nda::MemoryArray Array_base_t>
    auto make_distributed_shared_array(mpi3::communicator &gcomm,
             mpi3::communicator &internode_comm,
             mpi3::shared_communicator &node_comm,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> grid,
             std::array<long, ::nda::get_rank<std::decay_t<Array_base_t>>> gshape) {
      static constexpr int rank = ::nda::get_rank<Array_base_t>;
      using Array_t = distributed_shared_array<Array_base_t>;

      std::array<long, rank> origin, lshape;
      long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
      utils::check(internode_comm.size() == np,
                   "distributed_shared_array: Number of nodes does not match grid: nodes:{}, grid:{}",
                   internode_comm.size(), np);
      for (int n = 0; n < rank; n++) {
        utils::check(gshape[n] >= grid[n],
                     "distributed_shared_array: Too many processors i:{}, shape:{}, grid:{}",
                     n, gshape[n], grid[n]);
      }
      long ip = long(internode_comm.rank());
      // row major over proc grid for all other cases
      for(int n = rank - 1; n >= 0; n--) {
        std::tie(origin[n], lshape[n]) =
            itertools::chunk_range(0, gshape[n],grid[n],ip%grid[n]);
        lshape[n] -= origin[n];
        ip /= grid[n];
      }
      return Array_t(std::addressof(gcomm),
                     std::addressof(internode_comm),
                     std::addressof(node_comm),
                     grid, gshape, lshape, origin);
    }

  } // shm
} // math

#endif // NUMERICS_SHARED_ARRAY_NDA_HPP
