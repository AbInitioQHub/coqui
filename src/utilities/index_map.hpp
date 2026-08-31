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


#ifndef UTILITIES_INDEX_MAP_HPP
#define UTILITIES_INDEX_MAP_HPP

#include "utilities/check.hpp"
#include "itertools/itertools.hpp"

namespace utils
{

namespace detail
{

struct imap_full_iter : itertools::iterator_facade<imap_full_iter, std::tuple<long,long>> {

  long p;      // current value of linearized iterator
  long ld;     // leading dimension of 2-d range 
 
  imap_full_iter(long p_, long ld_) : p(p_),ld(ld_) {}

  void increment() {
    ++p;
  }

  bool operator==(imap_full_iter const &other) const { 
    return p == other.p and ld == other.ld; 
  }

  [[nodiscard]] decltype(auto) dereference() const { return std::tuple<long,long>{p/ld,p%ld}; }
};

}

class index_map_full
{
  public:

  using const_iterator = detail::imap_full_iter; 

  index_map_full(long n1_, long n2_, long p0_, long p1_) : n1(n1_), n2(n2_), p0(p0_), p1(p1_)
  {
    utils::check( n1 > 0 and n2 > 0 and p0 >= 0 and p1 >= 0 and
 		  p0 <= n1*n2 and p1 <= n1*n2,
                  "index_map_full(...): Inconsistent input.");
  }

  template<std::integral Int> 
  index_map_full(long n1_, long n2_, std::pair<Int,Int> const& r) : 
    n1(n1_), n2(n2_), p0(long(r.first)), p1(long(r.second))
  {
    utils::check( n1 > 0 and n2 > 0 and p0 >= 0 and p1 >= 0 and
 		  p0 <= n1*n2 and p1 <= n1*n2,
                  "index_map_full(...): Inconsistent input.");
  }

  ~index_map_full() {}

  index_map_full(index_map_full const&)=default;
  index_map_full(index_map_full &&)=default;
  index_map_full& operator=(index_map_full const&)=default;
  index_map_full& operator=(index_map_full &&)=default;

  long size() const { return p1-p0; }
    
  long map_global(long i, long j) const { 
    utils::check(i>=0 and i<n1 and 
	   	 j>=0 and j<n2, "Index out of bounds");
    return i*n2+j; 
  }

  long map(long i, long j) const {
    long p = i*n2+j-p0;
    utils::check(i>=0 and i<n1 and 
	   	 j>=0 and j<n2 and 
	   	 p>=0 and p<p1, "Index out of bounds");
    return p;    
  }

  std::tuple<long,long> map_global(long p) const {
    long i = p/n2;
    long j = p%n2;
    utils::check(p>=0 and p<n1*n2, "Index out of bounds");
    return std::make_tuple(i,j);
  }

  std::tuple<long,long> map(long p) const {
    long i = (p+p0)/n2;
    long j = (p+p0)%n2;
    utils::check(i>=0 and i<n1 and
                 j>=0 and j<n2 and
                 p>=0 and p<p1, "Index out of bounds");
    return std::make_tuple(i,j);
  }

  [[nodiscard]] const_iterator begin() noexcept { return {p0, n2}; }
  [[nodiscard]] const_iterator cbegin() const noexcept { return {p0, n2}; }

  [[nodiscard]] const_iterator end() noexcept { return {p1, n2}; }
  [[nodiscard]] const_iterator cend() const noexcept { return {p1, n2}; }

  private:

  long n1, n2, p0, p1;
};


using index_map = index_map_full;

} // utils

#endif
