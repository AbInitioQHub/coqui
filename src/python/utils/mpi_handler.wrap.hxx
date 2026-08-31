#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_mpi_handler_GUARDS
#define C2PY_HXX_DECLARATION_mpi_handler_GUARDS
template <> constexpr bool c2py::is_wrapped<coqui_py::MpiHandler> = true;
template <>
inline constexpr auto c2py::tp_name<coqui_py::MpiHandler> =
    "mpi_handler.MpiHandler";
#endif