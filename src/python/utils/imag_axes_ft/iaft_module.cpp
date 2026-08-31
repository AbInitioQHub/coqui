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


#include <c2py/c2py.hpp>

#include "numerics/imag_axes_ft/iaft_utils.hpp"

namespace imag_axes_ft {

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> tau_to_w_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_ti, 
		const std::string &stats) {
		auto stats_e_ = string_to_stats_enum(stats);
		auto nw = (stats_e_ == fermion) ? iaft.nw_f() : iaft.nw_b();
		nda::array<ComplexType, 2> X_wi(nw, X_ti.shape(1));
		iaft.tau_to_w(X_ti, X_wi, stats_e_);
		return X_wi;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 1> tau_to_w_iw_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_ti, 
		const std::string &stats, 
		long iw) {
		auto stats_e_ = string_to_stats_enum(stats);
		nda::array<ComplexType, 1> Xw_i(X_ti.shape(1));
		iaft.tau_to_w(X_ti, Xw_i, stats_e_, iw);
		return Xw_i;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> tau_to_w_phsym_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_ti_pos) {
		auto nw_half = (iaft.nw_b() % 2 == 0) ? iaft.nw_b() / 2 : iaft.nw_b() / 2 + 1;
		nda::array<ComplexType, 2> X_wi_pos(nw_half, X_ti_pos.shape(1));
		iaft.tau_to_w_PHsym(X_ti_pos, X_wi_pos);
		return X_wi_pos;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> w_to_tau_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_wi, 
		const std::string &stats) {
		auto stats_e_ = string_to_stats_enum(stats);
		auto nt = (stats_e_ == fermion) ? iaft.nt_f() : iaft.nt_b();
		nda::array<ComplexType, 2> X_ti(nt, X_wi.shape(1));
		iaft.w_to_tau(X_wi, X_ti, stats_e_);
		return X_ti;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> w_to_tau_phsym_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_wi_pos) {
		auto nt_half = (iaft.nt_b() % 2 == 0) ? iaft.nt_b() / 2 : iaft.nt_b() / 2 + 1;
		nda::array<ComplexType, 2> X_ti_pos(nt_half, X_wi_pos.shape(1));
		iaft.w_to_tau_PHsym(X_wi_pos, X_ti_pos);
		return X_ti_pos;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> w_to_tau_cross_stats_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &X_wi, 
		const std::string &stats_A, 
		const std::string &stats_B) {
		auto stats_A_e = string_to_stats_enum(stats_A);
		auto stats_B_e = string_to_stats_enum(stats_B);
		auto nt = (stats_B_e == fermion) ? iaft.nt_f() : iaft.nt_b();
		nda::array<ComplexType, 2> X_ti(nt, X_wi.shape(1));
		iaft.w_to_tau(X_wi, stats_A_e, X_ti, stats_B_e);
		return X_ti;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> w_to_tau_partial_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 1> &Xw_i, 
		const std::string &stats, 
		long iwn) {
		auto stats_e_ = string_to_stats_enum(stats);
		auto nt = (stats_e_ == fermion) ? iaft.nt_f() : iaft.nt_b();
		nda::array<ComplexType, 2> X_ti(nt, Xw_i.size());
		X_ti() = ComplexType{0.0, 0.0};
		iaft.w_to_tau_partial(Xw_i, X_ti, stats_e_, iwn);
		return X_ti;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> tau_to_tau_2d(
		const IAFT &iaft, 
		const nda::array<ComplexType, 2> &A_ti, 
		const std::string &A_stats) {
		auto A_stats_e = string_to_stats_enum(A_stats);
		auto B_stats_e = (A_stats_e == fermion) ? boson : fermion;
		auto nt_B = (B_stats_e == fermion) ? iaft.nt_f() : iaft.nt_b();
		nda::array<ComplexType, 2> B_ti(nt_B, A_ti.shape(1));
		iaft.tau_to_tau(A_ti, A_stats_e, B_ti);
		return B_ti;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 1> tau_to_tau_it_2d(
		const IAFT &iaft,
		const nda::array<ComplexType, 2> &A_ti,
		const std::string &A_stats,
		long it_B) {
		auto A_stats_e = string_to_stats_enum(A_stats);
		nda::array<ComplexType, 1> Bt_i(A_ti.shape(1));
		iaft.tau_to_tau(A_ti, A_stats_e, Bt_i, it_B);
		return Bt_i;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 1> tau_to_beta_2d(
		const IAFT &iaft,
		const nda::array<ComplexType, 2> &A_ti) {
		nda::array<ComplexType, 1> A_beta_i(A_ti.shape(1));
		iaft.tau_to_beta(A_ti, A_beta_i);
		return A_beta_i;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 1> tau_to_zero_2d(
		const IAFT &iaft,
		const nda::array<ComplexType, 2> &A_ti) {
		nda::array<ComplexType, 1> A_zero_i(A_ti.shape(1));
		iaft.tau_to_zero(A_ti, A_zero_i);
		return A_zero_i;
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> construct_tau_interpolate_matrix(
		const IAFT &iaft, 
		const nda::array<double, 1> &tau_mesh_out, bool ph_sym=false) {
		return iaft.construct_tau_interpolate_matrix(tau_mesh_out, ph_sym);
	}

	C2PY_WRAP_AS_METHOD
	nda::array<ComplexType, 2> construct_w_interpolate_matrix(
		const IAFT &iaft,
		const nda::array<long, 1> &wn_mesh_out,
		const std::string &stats_str,
		bool ph_sym=false) {
		auto stats = string_to_stats_enum(stats_str);
		return iaft.construct_w_interpolate_matrix(wn_mesh_out, stats, ph_sym);
	}

	nda::array<ComplexType, 3> build_g_tau_ref(
		const nda::array<double, 1> &tau_mesh, 
		double beta, 
		int norb, bool ph_sym = false) {
		return test_utils::build_g_tau(tau_mesh, beta, norb, ph_sym);
	}

	nda::array<ComplexType, 3> build_g_iw_ref(
		const nda::array<long, 1> &wn_mesh,
		double beta, const std::string &stats_str,
		int norb, bool ph_sym = false) {
		auto stats = string_to_stats_enum(stats_str);
		return test_utils::build_g_iw(wn_mesh, beta, stats, norb, ph_sym);
	}

} // namespace imag_axes_ft

#include "iaft_module.wrap.cxx"