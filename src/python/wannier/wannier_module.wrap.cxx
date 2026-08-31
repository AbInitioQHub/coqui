
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

// coqui2wannier90
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::wannier_interface::coqui2wannier90(mf, params);
    },
    "mf", "params")};

// mlwf_h5_from_wannier90_output
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::wannier_interface::mlwf_h5_from_wannier90_output(mf,
                                                                        params);
    },
    "mf", "params")};

// wannier90_append_win
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::wannier_interface::wannier90_append_win(mf, params);
    },
    "mf", "params")};

// wannier90_library_mode
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::wannier_interface::wannier90_library_mode(mf, params);
    },
    "mf", "params")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"coqui2wannier90", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"mlwf_h5_from_wannier90_output", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"wannier90_append_win", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"wannier90_library_mode", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "wannier_module",                                      /* name of module */
    R"RAWDOC(CoQui interface to Wannier90 library)RAWDOC", /* module
                                                              documentation, may
                                                              be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_wannier_module() {

  if (not c2py::check_python_version("wannier_module"))
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
