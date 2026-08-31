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


#ifndef NUMERICS_DISTRIBUTED_ARRAY_NDA_MATRIX_HPP
#define NUMERICS_DISTRIBUTED_ARRAY_NDA_MATRIX_HPP

#include <utility>
#include <tuple>
#include "utilities/check.hpp" 
#include "nda/nda.hpp"
#include "nda/mem/address_space.hpp"
#include "numerics/distributed_array/detail/concepts.hpp"

// simple class to wrap the local part of a distributed nda array
// MAM: still in design stages
namespace math::nda 
{

/*
 * Implementation of basic grid object
 */ 
namespace detail
{

template<int Rank, typename communicator>
struct darray
{
  static constexpr int rank = Rank; 

  mutable communicator* comm;

  // processor grid
  std::array<long,rank> grid;   

  // size of global array
  std::array<long,rank> gextents;

  // origin of local array 
  std::array<long,rank> lorigin;

  // block size for slate and slate-like dispatch
  std::array<long,rank> block_size;

  darray(communicator* comm_,
         std::array<long,rank>  grid_,
         std::array<long,rank> gshape,
         std::array<long,rank> origin_,
	 std::array<long,rank> bsize) :
    comm(comm_),
    grid(grid_),
    gextents(gshape),
    lorigin(origin_),
    block_size(bsize) 
  { 
    check_dimensions();
  }

  template<typename IntArray>
  darray(communicator* comm_,
	 std::array<long,rank>  grid_,
 	 std::array<long,rank> gshape,
	 std::array<long,rank> origin_,
	 std::array<long,rank> bsize,
	 IntArray const& local_size) :
    comm(comm_),
    grid(grid_),
    gextents(gshape),
    lorigin(origin_),
    block_size(bsize)
  {
    check_dimensions(local_size);
  }

  // dummy constructor
  darray() {}
  
  ~darray() = default; 

  void reset() {
    gextents=std::array<long,rank>{0};
    lorigin=std::array<long,rank>{0};
    block_size=std::array<long,rank>{0};
  }

#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
  darray(darray const& other) :
    comm(other.comm), grid(other.grid), gextents(other.gextents), 
    lorigin(other.lorigin),block_size(other.block_size)
  { check_dimensions(); } 
  darray(darray && other) :
    comm(other.comm), grid(other.grid), gextents(other.gextents), 
    lorigin(other.lorigin),block_size(other.block_size)
  { check_dimensions(); } 
  darray& operator=(darray const& other)   
  { 
    comm = other.comm;
    grid = other.grid;
    gextents = other.gextents;
    lorigin = other.lorigin;
    block_size = other.block_size;
    check_dimensions();  
    return *this;
  }
  darray& operator=(darray && other) { *this=other; return *this; } 
#else
  darray(darray const&) = default; 
  darray(darray &&) = default; 
  darray& operator=(darray const& other) = default; 
  darray& operator=(darray && other) = default; 
#endif

  bool operator==(darray const&) const = default;
  bool operator!=(darray const&) const = default;

  // returns true if the collection of local arrays reproduces gshape without overlaps
  template<typename IntArray>
  bool full_coverage_impl([[maybe_unused]] IntArray const& local_shape) const {
    utils::check(false,"finish");
    return false;
  }

  // returns true if the distributed array is consistent with a slate matrix
  template<bool is_stride_order_C, typename IntArray>
  bool is_slate_compatible_impl(IntArray const& local_shape) const {
    if constexpr (rank != 2) {
      return false;
    } else {
      if( (block_size[0] <= 0) or
	  (block_size[1] <= 0) or
          (block_size[0] > gextents[0]/grid[0]) or
          (block_size[1] > gextents[1]/grid[1]) or
          (comm->size() != grid[0]*grid[1]) ) return false;
      long ix, iy;
      if constexpr (is_stride_order_C) {
        // row major processor distribution
        ix = comm->rank()/grid[1];
        iy = comm->rank()%grid[1];
      } else { // column major processor distribution
        ix = comm->rank()%grid[0];
        iy = comm->rank()/grid[0];
      }
      // last proc in the grid can have any dimension
      if( (ix<grid[0]-1) and (local_shape[0]%block_size[0] != 0)) return false; 
      if( (iy<grid[1]-1) and (local_shape[1]%block_size[1] != 0)) return false; 
      return true;
    }
  }

  void check_dimensions() const
  {
    long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
    utils::check( comm->size() == np, 
      "distributed_array: Number of processors does not match grid: size:{} grid:{}",comm->size(),np);
    for(int i=0; i<rank; i++) {
      utils::check(grid[i] >= 0,"distributed_array i:{} grid:{}",i,grid[i]);
      utils::check(lorigin[i] >= 0,"distributed_array i:{} origin:{}",i,lorigin[i]);
      utils::check(gextents[i] >= 0,"distributed_array i:{} extent:{}",i,gextents[i]);
    }
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    std::array<long,2*rank> tmp;
    for(int i=0; i<rank; ++i) tmp[i] = gextents[i];
    for(int i=0; i<rank; ++i) tmp[rank+i] = long(grid[i]);
    comm->broadcast_n(tmp.begin(),2*rank,0);
    for(int i=0; i<rank; ++i) {
      utils::check(tmp[i] == gextents[i],
		"distributed_array: Inconsistent global shape: i:{} local:{} ref:{}",i,gextents[i],tmp[i]);
      utils::check(tmp[rank+i] == long(grid[i]), 
    		"distributed_array: Inconsistent grid size: i:{} local:{} ref:{}",i,grid[i],tmp[rank+i]);
    }
    comm->barrier();
#endif
  }

  template<typename IntArray>
  void check_dimensions(IntArray const& local_size) const
  {
    long np = std::accumulate(grid.cbegin(), grid.cend(), 1, std::multiplies<>{});
    utils::check( comm->size() == np, 
	"distributed_array: Number of processors does not match grid: size:{} grid:{}",comm->size(),np);
    for(int i=0; i<rank; i++) {
      utils::check(grid[i] >= 0,"distributed_array i:{} grid:{}",i,grid[i]);
      utils::check(lorigin[i] >= 0,"distributed_array i:{} origin:{}",i,lorigin[i]);
      utils::check(gextents[i] >= 0,"distributed_array i:{} extent:{}",i,gextents[i]);
      utils::check(lorigin[i]+local_size[i] <= gextents[i],
		"distributed_array i:{} origin:{} local_size:{} , global:{}",
		i,lorigin[i],local_size[i],gextents[i]);
      utils::check(local_size[i] >= 0,"distributed_array i:{} local_size:{}",i,local_size[i]);
    }
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    std::array<long,2*rank> tmp;
    for(int i=0; i<rank; ++i) tmp[i] = gextents[i]; 
    for(int i=0; i<rank; ++i) tmp[rank+i] = long(grid[i]);
    comm->broadcast_n(tmp.begin(),2*rank,0);
    for(int i=0; i<rank; ++i) {
      utils::check(tmp[i] == gextents[i],
		"distributed_array: Inconsistent global shape: i:{} local:{} ref:{}",i,gextents[i],tmp[i]);
      utils::check(tmp[rank+i] == long(grid[i]), 
    		"distributed_array: Inconsistent grid size: i:{} local:{} ref:{}",i,grid[i],tmp[rank+i]);
    }
    comm->barrier();
#endif
  }
};

} // detail

/*
 * Owning version of distributed array
 */ 

template<::nda::Array Array_base_t, typename communicator_t> 
class distributed_array  
{
  public:

  using Array_t = typename std::decay_t<Array_base_t>::regular_type;
  static constexpr int rank = ::nda::get_rank<Array_t>;
  using value_type = typename std::decay_t<Array_t>::value_type;
  static constexpr bool is_stride_order_Fortran() noexcept 
    { return Array_t::layout_t::is_stride_order_Fortran(); }
  static constexpr bool is_stride_order_C() noexcept 
    { return Array_t::layout_t::is_stride_order_C(); }

  private:

  using darray_t = detail::darray<rank,communicator_t>;
  static_assert( Array_t::layout_t::is_stride_order_Fortran() or
		 Array_t::layout_t::is_stride_order_C(), "Ordering mismatch.");

  public:

  distributed_array(communicator_t* comm_,
		    std::array<long,rank>  grid_,
		    std::array<long,rank> gshape,
		    std::array<long,rank> local_size,
		    std::array<long,rank> origin_,
                    std::array<long,rank> bsize):
    base(comm_,grid_,gshape,origin_,bsize,local_size),
    A(local_size)
  {
    // initialize to zero just in case
    A() = 0;
    // enforcing slate compatibility for now
  }

  template<::nda::MemoryArray Arr>
  distributed_array(communicator_t* comm_,
                    std::array<long,rank>  grid_,
                    std::array<long,rank> gshape,
                    std::array<long,rank> origin_,
                    std::array<long,rank> bsize,
		    Arr&& A_):
    base(comm_,grid_,gshape,origin_,bsize,A_.shape()),
    A(std::forward<Array_t>(A_))
  {}

  // dummy constructor
  distributed_array() {}

  ~distributed_array() {} 

  void reset() {
    base.reset();
    std::array<long,rank> lshp;
    for( auto& v: lshp ) v=0;
    A = Array_t(lshp);
  }

  distributed_array(distributed_array const& other) = default; 
  distributed_array& operator=(distributed_array const& other) = default; 
  distributed_array(distributed_array &&) = default;
  distributed_array& operator=(distributed_array &&) = default;

  template<DistributedArrayOfRank<rank> DArr>
  distributed_array(DArr const& other) : 
    base(other.communicator(),other.grid(),other.global_shape(),
	 other.origin(),other.block_size(),other.local().shape()),
    A(other.local())
  {
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    base.comm->barrier();
#endif
  }

  template<DistributedArrayOfRank<rank> DArr>
  distributed_array& operator=(DArr const& other) 
  { 
    base.grid = other.grid();
    base.gextents = other.global_shape();
    base.lorigin = other.origin();
    base.comm = other.communicator();
    base.block_size = other.block_size();
    A = other.local();
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    base.comm->barrier();
#endif
    return *this;
  }

  bool operator==(distributed_array const&) const = default;
  bool operator!=(distributed_array const&) const = default;

  std::array<long,rank> const& global_shape() const { return base.gextents; }
  std::array<long,rank> const& origin() const { return base.lorigin; }
  std::array<long,rank> const& grid() const { return base.grid; };
  std::array<long,rank> const& block_size() const { return base.block_size; };

  // remove when contrain issue is fixed
  auto& local_() { return A; }
  auto const& local_() const { return A; }

  auto local() { return A(); }
  auto local() const { return A(); }
  auto const& local_shape() const { return A.shape(); }
  communicator_t* communicator() const { return base.comm; }

  auto local_range(int dim) const { 
    utils::check(dim >= 0 and dim < rank, "distributed_array::range: Out of range d:{}",dim); 
    return ::nda::range(base.lorigin[dim],base.lorigin[dim]+A.shape()[dim]);
  }

// returns true if the collection of local arrays reproduces gshape without overlaps
  bool full_coverage() const { return base.full_coverage_impl(A.shape());}

// returns true if the distributed array is consistent with a slate matrix
  bool is_slate_compatible() const { return base.template is_slate_compatible_impl<is_stride_order_C()>(A.shape()); }   
  protected:

  // grid/shape info
  darray_t base;

  // local array
  Array_t A;
};

/*
 * Non-owning version of distributed array
 */ 

template<::nda::Array Array_base_t, typename communicator_t> 
class distributed_array_view
{

  using Base_t = std::decay_t<Array_base_t>;

  public:

  static constexpr bool is_view = true;
  static constexpr int rank = ::nda::get_rank<Base_t>;
  using value_type = typename Base_t::value_type;

  private:

  static_assert(Base_t::layout_t::is_stride_order_Fortran() or
                Base_t::layout_t::is_stride_order_C(), "Layout mismatch.");
  using Layout = std::conditional_t< Base_t::layout_t::is_stride_order_C(),
                                     ::nda::C_stride_layout,
                                     ::nda::F_stride_layout>; 
  using OwningPolicy = ::nda::borrowed<::nda::mem::get_addr_space<Base_t>>;  

  public:

  using Array_view_t = ::nda::basic_array_view<value_type, rank, Layout, 'A', ::nda::default_accessor, OwningPolicy>;
  using Array_t = typename Array_view_t::regular_type; 
  static constexpr bool is_stride_order_Fortran() noexcept 
    { return Array_view_t::layout_t::is_stride_order_Fortran(); }
  static constexpr bool is_stride_order_C() noexcept      
    { return Array_view_t::layout_t::is_stride_order_C(); }

  private:

  using darray_t = detail::darray<rank, communicator_t>;
  static_assert( Array_view_t::layout_t::is_stride_order_Fortran() or
		 Array_view_t::layout_t::is_stride_order_C(), "Ordering mismatch.");

  public:

  distributed_array_view(communicator_t* comm_,
		    std::array<long,rank>  grid_,
		    std::array<long,rank> gshape,
		    std::array<long,rank> origin_,
                    std::array<long,rank> bsize,
		    ::nda::ArrayOfRank<rank> auto && A_) :
		    // can't keep a non-const view to a const-view, so need to take arg by mutable ref
    base(comm_,grid_,gshape,origin_,bsize,A_.shape()),
    A(A_.indexmap(),A_.data()) 
  {
  }
  
  ~distributed_array_view() {}

  void reset() {
    base.reset();
    A = Array_view_t{};  
  }

  distributed_array_view(distributed_array_view const&) = default; 
  distributed_array_view(distributed_array_view &&) = default;

  distributed_array_view& operator=(distributed_array_view const& other) {
    utils::check(base.grid==other.grid(), "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.gextents==other.global_shape(),
                                          "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.lorigin==other.origin(), "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.comm==other.communicator(), "distributed_array_view::operator= Inconsistent communicators.");
    A = other.A;
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    base.comm->barrier();
#endif
    return *this;
  }
  distributed_array_view& operator=(distributed_array_view && other) {return *this = other;}

  // MAM: operator= on the view should guard against incompatible memory spaces!
  template<DistributedArrayOfRank<rank> DArr>
  distributed_array_view& operator=(DArr & other) {
    utils::check(base.grid==other.grid(), "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.gextents==other.global_shape(), 
					  "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.lorigin==other.origin(), "distributed_array_view::operator= Inconsistent shapes.");
    utils::check(base.comm==other.communicator(), "distributed_array_view::operator= Inconsistent communicators.");
    A = other.local();  
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    base.comm->barrier();
#endif
    return *this;
  } 
  distributed_array_view& operator=(DistributedArrayOfRank<rank> auto && other) {return *this = other;}

  // changes the object the view points to
  template<DistributedArrayOfRank<rank> DArr>
  void rebind(DArr& other)
  {    
    base.grid = other.grid();
    base.gextents = other.global_shape(); 
    base.lorigin = other.origin();
    base.comm = other.communicator();
    base.block_size = other.block_size();
    A.rebind(other.local());
#if defined(SYNCHRONIZE_DISTRIBUTED_ARRAY)
    base.comm->barrier();
#endif    
  }

  bool operator==(distributed_array_view const&) const = default;
  bool operator!=(distributed_array_view const&) const = default;

  std::array<long,rank> const& global_shape() const { return base.gextents; }
  std::array<long,rank> const& origin() const { return base.lorigin; }
  std::array<long,rank> const& grid() const { return base.grid; };
  std::array<long,rank> const& block_size() const { return base.block_size; };
  
  auto local() { return A(); }
  auto local() const { return A(); }
  auto const& local_shape() const { return A.shape(); }
  communicator_t* communicator() const { return base.comm; }

  auto local_range(int dim) const {
    utils::check(dim >= 0 and dim < rank, "distributed_array::range: Out of range d:{}",dim);
    return ::nda::range(base.lorigin[dim],base.lorigin[dim]+A.shape()[dim]);
  }

  // returns true if the collection of local arrays reproduces gshape without overlaps
  bool full_coverage() const { return base.full_coverage_impl(A.shape());}

  // returns true if the distributed array is consistent with a slate matrix
  bool is_slate_compatible() const { return base.template is_slate_compatible_impl<is_stride_order_C()>(A.shape()); }

  protected:

  // grid/shape info
  darray_t base;

  // local array view
  Array_view_t A;

};

} // math::nda

#endif
