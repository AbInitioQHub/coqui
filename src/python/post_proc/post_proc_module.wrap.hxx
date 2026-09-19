#include <c2py/c2py.hpp>

#ifndef C2PY_HXX_DECLARATION_post_proc_module_GUARDS
#define C2PY_HXX_DECLARATION_post_proc_module_GUARDS
template <> constexpr bool c2py::is_wrapped<methods::lqp::result_t> = true;
template <>
inline constexpr auto c2py::tp_name<methods::lqp::result_t> =
    "post_proc_module.ResultT";
template <>
constexpr bool c2py::is_wrapped<methods::lqp::ladder_result_t> = true;
template <>
inline constexpr auto c2py::tp_name<methods::lqp::ladder_result_t> =
    "post_proc_module.LadderResultT";
#endif