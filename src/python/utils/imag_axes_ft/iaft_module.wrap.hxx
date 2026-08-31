#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_iaft_module_GUARDS
#define C2PY_HXX_DECLARATION_iaft_module_GUARDS
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::ir::IR> = true;
template <>
inline constexpr auto c2py::tp_name<imag_axes_ft::ir::IR> = "iaft_module.IR";
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::IAFT> = true;
template <>
inline constexpr auto c2py::tp_name<imag_axes_ft::IAFT> = "iaft_module.IAFT";
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::stats_e> = true;
template <>
const std::map<imag_axes_ft::stats_e, str_t>
    c2py::enum_to_string<imag_axes_ft::stats_e> = {
        {imag_axes_ft::stats_e::fermion, "fermion"},
        {imag_axes_ft::stats_e::boson, "boson"}};
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::basis_e> = true;
template <>
const std::map<imag_axes_ft::basis_e, str_t>
    c2py::enum_to_string<imag_axes_ft::basis_e> = {
        {imag_axes_ft::basis_e::dlr_basis, "dlr_basis"},
        {imag_axes_ft::basis_e::ir_basis, "ir_basis"}};
#endif