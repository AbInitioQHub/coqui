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


#ifndef COQUI_QP_SOLVERS_HPP
#define COQUI_QP_SOLVERS_HPP

#include <algorithm>
#include <tuple>
#include "configuration.hpp"
#include "nda/nda.hpp"
#include "IO/app_loggers.h"

namespace methods {

template<typename function_t>
auto qp_eqn_spectral(double Vhf, function_t &Sigma, long I, double mu, double eps0, double tol, double eta)
  -> std::tuple<double, bool> {
  double w_min = eps0 - mu - 1.0;
  double w_max = eps0 - mu + 1.0;
  long Nw = 1000;

  nda::array<ComplexType, 1> w_grid(Nw);
  int i = 0;
  std::transform(w_grid.begin(), w_grid.end(), w_grid.begin(),
                 [&](const ComplexType & ) {return w_min + (i++)*(w_max - w_min)/(Nw - 1) + eta*1i;} );

  nda::array<ComplexType, 1> SigmaR_w(Nw);
  Sigma.evaluate(w_grid, SigmaR_w, I);

  double Aw_max = 0.0;
  long nmax = -1;
  for (size_t n = 0; n < Nw; ++n) {
    ComplexType Gw = 1.0 / (w_grid(n) + mu - Vhf - SigmaR_w(n));
    if (std::abs(Gw.imag()) > Aw_max) {
      nmax = n;
      Aw_max = std::abs(Gw.imag());
    }
  }

  auto Aw = [&](ComplexType w) {
    return std::abs((1.0 / ( w + mu - Vhf - Sigma.evaluate(w, I) )).imag());
  };
  ComplexType eps_qp = w_grid(nmax);
  ComplexType eps_qp_dw = eps_qp + tol;
  double Aw_dw = Aw(eps_qp_dw);
  if ( Aw_dw >= Aw_max) {
    while (Aw_dw >= Aw_max) {
      eps_qp = eps_qp_dw;
      Aw_max = Aw_dw;
      eps_qp_dw += tol;
      Aw_dw = Aw(eps_qp_dw);
    }
  } else {
    eps_qp_dw = eps_qp - tol;
    Aw_dw = Aw(eps_qp_dw);
    while (Aw_dw >= Aw_max) {
      eps_qp = eps_qp_dw;
      Aw_max = Aw_dw;
      eps_qp_dw -= tol;
      Aw_dw = Aw(eps_qp_dw);
    }
  }

  if (nmax != -1) {
    return std::make_tuple(eps_qp.real()+mu, true);
  } else {
    return std::make_tuple(eps0, false);
  }
}

template<typename function_t>
double qp_eqn_linearized(double Vhf, function_t &Sigma, long I, double mu, double eps_ks, double eta) {
  double h = 1e-6;
  double Sigma_ks = Sigma.evaluate(ComplexType(eps_ks-mu, eta), I).real();
  double dSigma = (Sigma.evaluate(ComplexType(eps_ks-mu+h, eta), I).real() - Sigma_ks);
  double Z = 1.0 / (1.0 - dSigma/h);

  return Z * ( Vhf + Sigma_ks - (1 - 1.0/Z) * eps_ks );
}

template<typename function_t>
auto qp_eqn_secant(double Vhf, function_t &Sigma, long I, double mu, double w0, int maxiter, double tol, double eta)
  -> std::tuple<double, double, bool> {
  auto qp_res = [&](ComplexType w) {
    return (w - Vhf - Sigma.evaluate(w-mu, I)).real();
  };

  bool conv = false;
  double p=0, p0, p1, q, q0, q1;
  double eps = 1e-4;

  p0 = w0;
  p1 = (p0 >= 0)? w0 * (1 + eps) + eps : w0 * (1 + eps) - eps;
  q0 = qp_res(ComplexType(p0, eta));
  q1 = qp_res(ComplexType(p1, eta));

  if (std::abs(q1) < std::abs(q0)) {
    double tmp = p1;
    p1 = p0;
    p0 = tmp;

    tmp = q1;
    q1 = q0;
    q0 = tmp;
  }

  for (long it = 0; it < maxiter; ++it) {
    if (std::abs(q1) > std::abs(q0)) {
      p = (-q0/q1 * p1 + p0) / (1 - q0/q1);
    } else {
      p = (-q1/q0 * p0 + p1) / (1 - q1/q0);
    }

    if (std::abs(p - p1) < tol) {
      conv = true;
      break;
    }

    p0 = p1;
    q0 = q1;
    p1 = p;
    q1 = qp_res(ComplexType(p1, eta));
  }

  q = qp_res(ComplexType(p, eta));
  return std::make_tuple(p, q, conv);
}

template<typename function_t>
auto qp_eqn_bisection(double Vhf, function_t &Sigma, long I, double mu, double eps0, double tol, double eta)
  -> std::tuple<double, double> {
  auto qp_res = [&](ComplexType w) {
    return (w - Vhf - Sigma.evaluate(w-mu, I)).real();
  };

  double eps1, eps2, eps_mid;
  double eps = eps0;
  double res = qp_res(ComplexType(eps0, eta));
  double delta = 0.01;

  app_log(6, "I = {0}, Vhf = {1:.12f}, Sigma = {2:.12f}, eps = {3:.12f}, res = {4:.12f}",
          I, Vhf, Sigma.evaluate(ComplexType(eps0,eta), I).real(), eps, res);
  if (std::abs(res) < tol) return std::make_tuple(eps, res);

  if (res >= 0) {
    eps2 = eps0;
    eps1 = eps0 - delta;
    double res1 = qp_res(ComplexType(eps1, eta));
    while (res1 > 0) {
      app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps1, res1);
      eps1 -= delta;
      res1 = qp_res(ComplexType(eps1, eta));
    }
  } else {
    eps1 = eps0;
    eps2 = eps0 + delta;
    double res2 = qp_res(ComplexType(eps2, eta));
    while (res2 < 0) {
      app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps2, res2);
      eps2 += delta;
      res2 = qp_res(ComplexType(eps2, eta));
    }
  }
  eps_mid = (eps1 + eps2) * 0.5;
  res = qp_res(ComplexType(eps_mid, eta));
  while (std::abs(res) > tol) {
    app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps_mid, res);
    if (res >= 0) {
      eps2 = eps_mid;
    } else {
      eps1 = eps_mid;
    }
    eps_mid = (eps1 + eps2) * 0.5;
    res = qp_res(ComplexType(eps_mid, eta));
  }
  app_log(6, "I = {0}, eps = {1:.12f}, res = {2:.12f}", I, eps_mid, res);
  eps = eps_mid;
  return std::make_tuple(eps, res);
}

} // methods

#endif //COQUI_QP_SOLVERS_HPP
