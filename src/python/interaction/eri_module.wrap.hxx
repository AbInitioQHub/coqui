#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_eri_module_GUARDS
#define C2PY_HXX_DECLARATION_eri_module_GUARDS
template <> constexpr bool c2py::is_wrapped<coqui_py::ThcCoulomb> = true;
template <>
inline constexpr auto c2py::tp_name<coqui_py::ThcCoulomb> =
    "eri_module.ThcCoulomb";
template <> constexpr bool c2py::is_wrapped<coqui_py::CholCoulomb> = true;
template <>
inline constexpr auto c2py::tp_name<coqui_py::CholCoulomb> =
    "eri_module.CholCoulomb";
#endif