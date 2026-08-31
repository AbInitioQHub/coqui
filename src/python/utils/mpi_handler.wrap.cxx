
// C.f. https://numpy.org/doc/1.21/reference/c-api/array.html#importing-the-api
#define PY_ARRAY_UNIQUE_SYMBOL _cpp2py_ARRAY_API
#ifndef CLAIR_C2PY_WRAP_GEN
#ifdef __clang__
// #pragma clang diagnostic ignored "-W#warnings"
#endif
#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#pragma GCC diagnostic ignored "-Wcast-function-type"
#pragma GCC diagnostic ignored "-Wcpp"
#endif

#define C2PY_VERSION_MAJOR 0
#define C2PY_VERSION_MINOR 1

#include <c2py/c2py.hpp>

using c2py::operator""_a;

// ==================== enums =====================

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0 = coqui_py::MpiHandler;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0> = true;
template <>
inline constexpr auto c2py::tp_name<_c2py_cls_0> = "mpi_handler.MpiHandler";
static const auto _c2py_init_0 =
    c2py::dispatcher_c_kw_t{c2py::c_constructor<_c2py_cls_0>()};
template <>
constexpr initproc c2py::tp_init<_c2py_cls_0> =
    c2py::pyfkw_constructor<_c2py_init_0>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> =
    _c2py_init_0.doc(R"DOC()DOC");
// barrier
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.barrier(); },
    "self")};

// comm_rank
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.comm_rank(); },
    "self")};

// comm_size
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.comm_size(); },
    "self")};

// internode_barrier
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.internode_barrier();
    },
    "self")};

// internode_rank
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.internode_rank();
    },
    "self")};

// internode_size
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.internode_size();
    },
    "self")};

// intranode_barrier
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.intranode_barrier();
    },
    "self")};

// intranode_rank
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.intranode_rank();
    },
    "self")};

// intranode_size
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.intranode_size();
    },
    "self")};

// root
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.root(); },
    "self")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC(
MPI barrier for the global communicator.
)DOC");
static const auto _c2py_doc_1 =
    _c2py_fun_1.doc(R"DOC(
Returns
-------
{ret_0}
   the rank of the current process in the global communicator.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_2 =
    _c2py_fun_2.doc(R"DOC(
Returns
-------
{ret_0}
   the size of the global communicator, i.e., the total number of processes.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC(
MPI barrier for the internode communicator.
)DOC");
static const auto _c2py_doc_4 =
    _c2py_fun_4.doc(R"DOC(
Returns
-------
{ret_0}
   the rank of the current process in the internode communicator.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_5 =
    _c2py_fun_5.doc(R"DOC(
Returns
-------
{ret_0}
   the size of the internode communicator, i.e., the number of nodes.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC(
MPI barrier for the intranode communicator.
)DOC");
static const auto _c2py_doc_7 =
    _c2py_fun_7.doc(R"DOC(
Returns
-------
{ret_0}
   the rank of the current process in the intranode communicator.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_8 =
    _c2py_fun_8.doc(R"DOC(
Returns
-------
{ret_0}
   the size of the intranode communicator, i.e., the number of processes within a node.
)DOC",
                    {}, {c2py::python_typename<int>()});
static const auto _c2py_doc_9 =
    _c2py_fun_9.doc(R"DOC(
Returns
-------
{ret_0}
   true if the current process is the root, false otherwise.
)DOC",
                    {}, {c2py::python_typename<bool>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
    {"barrier", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"comm_rank", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"comm_size", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"internode_barrier", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {"internode_rank", (PyCFunction)c2py::pyfkw<_c2py_fun_4>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
    {"internode_size", (PyCFunction)c2py::pyfkw<_c2py_fun_5>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
    {"intranode_barrier", (PyCFunction)c2py::pyfkw<_c2py_fun_6>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
    {"intranode_rank", (PyCFunction)c2py::pyfkw<_c2py_fun_7>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
    {"intranode_size", (PyCFunction)c2py::pyfkw<_c2py_fun_8>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
    {"root", (PyCFunction)c2py::pyfkw<_c2py_fun_9>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> =
    R"DOC(mpi handler class

The MpiHandler class encapsulates the state of a MPI environment used by CoQui.
It manages key information such as the total number of processors, node distribution,
and provides access to global, internode, and intranode communicators.

This class also offers a minimal interface for performing basic MPI operations.
It must be constructed and passed to any CoQuí routines that involve MPI parallelization.
Even in serial mode (i.e., when using a single process), this class is required to ensure
a consistent interface across all workflows.)DOC" +
    std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_0>;

// ==================== module functions ====================

//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "mpi_handler",                          /* name of module */
    R"RAWDOC(MPI handler for CoQui)RAWDOC", /* module documentation, may be NULL
                                             */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_mpi_handler() {

  if (not c2py::check_python_version("mpi_handler"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_0>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "MpiHandler");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
