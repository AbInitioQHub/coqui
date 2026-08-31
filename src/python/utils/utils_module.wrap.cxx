
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

// ==================== module functions ====================

// app_debug
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](int level, const std::string &msg) {
      return coqui_py::app_debug(level, msg);
    },
    "level", "msg")};

// app_error
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &msg) { return coqui_py::app_error(msg); }, "msg")};

// app_log
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](int level, const std::string &msg) {
      return coqui_py::app_log(level, msg);
    },
    "level", "msg")};

// app_warning
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const std::string &msg) { return coqui_py::app_warning(msg); }, "msg")};

// set_verbosity
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::MpiHandler &mpi_handler, int output_level, int debug_level) {
      return coqui_py::set_verbosity(mpi_handler, output_level, debug_level);
    },
    "mpi_handler", "output_level"_a = 2, "debug_level"_a = 0)};

// utest_filename
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](std::string src) { return coqui_py::utest_filename(src); }, "src")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 =
    _c2py_fun_4.doc(R"DOC(
Set verbosity levels for CoQui logging output.

Parameters
----------
mpi_handler : {par_0}
   - [INPUT] MPI handler to ensure logging is MPI-aware, i.e. only root prints
output_level : {par_1}
   - [INPUT] Level of output verbosity (default: 2)
debug_level : {par_2}
   - [INPUT] Level of debug verbosity (default: 0)
)DOC",
                    {{c2py::python_typename<coqui_py::MpiHandler &>()},
                     {c2py::python_typename<int>()},
                     {c2py::python_typename<int>()}});
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"app_debug", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"app_error", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"app_log", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"app_warning", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {"set_verbosity", (PyCFunction)c2py::pyfkw<_c2py_fun_4>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
    {"utest_filename", (PyCFunction)c2py::pyfkw<_c2py_fun_5>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "utils_module",                            /* name of module */
    R"RAWDOC(Utility module for CoQui)RAWDOC", /* module documentation, may be
                                                  NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_utils_module() {

  if (not c2py::check_python_version("utils_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)

#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
