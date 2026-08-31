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


#ifndef COQUI_VSPACE_HEFF_HPP
#define COQUI_VSPACE_HEFF_HPP

#include "numerics/iter_scf/diis/vspace.h"

namespace iter_scf {

class Heff {
private:
  using Array_4D = nda::array<ComplexType, 4>;

  Array_4D _Heff;
  bool inited_H = false;

public:
  Heff() = default;
  Heff(const Heff&) = default;
  Heff(Heff&&) = default;
  Heff& operator=(const Heff&) = default;
  Heff& operator=(Heff&&) = default;

  explicit Heff(const Array_4D& Heff_) : _Heff(Heff_) {
    inited_H = true;
  }

  ComplexType dot_prod(const Heff& rhs) const {
    utils::check(inited_H, "Heff: matrix is not initialized");
    auto rHeff = rhs.get_heff();

    size_t hdim = std::reduce(_Heff.shape().begin(), _Heff.shape().end(), 1, std::multiplies<size_t>());
    size_t rhdim = std::reduce(rHeff.shape().begin(), rHeff.shape().end(), 1, std::multiplies<size_t>());
    auto matvec_h = nda::reshape(_Heff, std::array<long, 2>{hdim, 1});
    auto matvec_rh = nda::reshape(rHeff, std::array<long, 2>{rhdim, 1});

    nda::array<ComplexType, 2> res(1, 1);
    nda::blas::gemm(nda::make_regular(nda::conj(nda::transpose(matvec_h))), matvec_rh, res);
    return res(0, 0);
  }

  const Array_4D& get_heff() const {
    utils::check(inited_H, "Heff: matrix is not initialized");
    return _Heff;
  }

  void set_heff(const Array_4D& H_) {
    _Heff = H_;
    inited_H = true;
  }

  void set_zero() {
    _Heff() = 0;
    inited_H = true;
  }

  Heff operator*=(std::complex<double> c) {
    utils::check(inited_H, "Heff: matrix is not initialized");
    _Heff *= c;
    return *this;
  }

  Heff operator+=(Heff& vec) {
    utils::check(inited_H, "Heff: matrix is not initialized");
    _Heff += vec.get_heff();
    return *this;
  }

  Heff operator+=(Heff&& vec) {
    utils::check(inited_H, "Heff: matrix is not initialized");
    _Heff += vec.get_heff();
    return *this;
  }

  void add(Heff&& a, ComplexType c) {
    utils::check(inited_H, "Heff: matrix is not initialized");
    _Heff += c * a.get_heff();
  }

  void read_from_file(std::string filename, const size_t vec_number) {
    h5::file file(filename, 'r');
    auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
    nda::h5_read(vec_grp, "Heff_skij", _Heff);
    inited_H = true;
  }

  void write_to_file(std::string filename, const size_t vec_number) const {
    utils::check(inited_H, "Heff: matrix is not initialized");
    h5::file file(filename, 'a');
    if(!h5::group(file).has_subgroup("vec" + std::to_string(vec_number))) {
      auto vec_grp = h5::group(file).create_group("vec" + std::to_string(vec_number));
      nda::h5_write(vec_grp, "Heff_skij", _Heff, false);
    } else {
      auto vec_grp = h5::group(file).open_group("vec" + std::to_string(vec_number));
      nda::h5_write(vec_grp, "Heff_skij", _Heff, false);
    }
  }
};

}

#endif // COQUI_VSPACE_HEFF_HPP
