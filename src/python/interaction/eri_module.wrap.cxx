
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
using _c2py_cls_0 = coqui_py::ThcCoulomb;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0> = true;
template <>
inline constexpr auto c2py::tp_name<_c2py_cls_0> = "eri_module.ThcCoulomb";
static const auto _c2py_init_0 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<_c2py_cls_0, const coqui_py::Mf &, const std::string &>(
        "mf", "thc_params")};
template <>
constexpr initproc c2py::tp_init<_c2py_cls_0> =
    c2py::pyfkw_constructor<_c2py_init_0>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> =
    _c2py_init_0.doc(R"DOC()DOC");
// Np
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.Np(); },
    "self")};

// init
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 &self) -> decltype(auto) { return self.init(); }, "self")};

// initialized
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.initialized();
    },
    "self")};

// mf
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.mf(); },
    "self")};

// mpi
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.mpi(); },
    "self")};

// nbnd
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nbnd(); },
    "self")};

// nkpts
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nkpts(); },
    "self")};

// nkpts_ibz
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nkpts_ibz(); },
    "self")};

// nqpts
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nqpts(); },
    "self")};

// nqpts_ibz
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nqpts_ibz(); },
    "self")};

// nspin
static auto const _c2py_fun_10 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) { return self.nspin(); },
    "self")};

// nspin_in_basis
static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.nspin_in_basis();
    },
    "self")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8 = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9 = _c2py_fun_9.doc(R"DOC()DOC");
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");
static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
    {"Np", (PyCFunction)c2py::pyfkw<_c2py_fun_0>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_0.c_str()},
    {"init", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"initialized", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"mf", (PyCFunction)c2py::pyfkw<_c2py_fun_3>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_3.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<_c2py_fun_4>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_4.c_str()},
    {"nbnd", (PyCFunction)c2py::pyfkw<_c2py_fun_5>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
    {"nkpts", (PyCFunction)c2py::pyfkw<_c2py_fun_6>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
    {"nkpts_ibz", (PyCFunction)c2py::pyfkw<_c2py_fun_7>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
    {"nqpts", (PyCFunction)c2py::pyfkw<_c2py_fun_8>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
    {"nqpts_ibz", (PyCFunction)c2py::pyfkw<_c2py_fun_9>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
    {"nspin", (PyCFunction)c2py::pyfkw<_c2py_fun_10>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
    {"nspin_in_basis", (PyCFunction)c2py::pyfkw<_c2py_fun_11>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> =
    R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;
// --------- class _c2py_cls_1 -----------
using _c2py_cls_1 = coqui_py::CholCoulomb;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_1> = true;
template <>
inline constexpr auto c2py::tp_name<_c2py_cls_1> = "eri_module.CholCoulomb";
static const auto _c2py_init_1 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<_c2py_cls_1, const coqui_py::Mf &, const std::string &>(
        "mf", "chol_params")};
template <>
constexpr initproc c2py::tp_init<_c2py_cls_1> =
    c2py::pyfkw_constructor<_c2py_init_1>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_1> =
    _c2py_init_1.doc(R"DOC()DOC");
// mf
static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.mf(); },
    "self")};

// mpi
static auto const _c2py_fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.mpi(); },
    "self")};

static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_1>[] = {
    {"mf", (PyCFunction)c2py::pyfkw<_c2py_fun_12>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_12.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<_c2py_fun_13>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <>
const std::string c2py::tp_doc<_c2py_cls_1> =
    R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_1>;

// ==================== module functions ====================

// run_isdf
static auto const _c2py_fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &thc_params) {
      return coqui_py::run_isdf(mf, thc_params);
    },
    "mf", "thc_params")};

static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"run_isdf", (PyCFunction)c2py::pyfkw<_c2py_fun_14>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "eri_module",                                  /* name of module */
    R"RAWDOC(Interaction module for CoQui)RAWDOC", /* module documentation, may
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
PyInit_eri_module() {

  if (not c2py::check_python_version("eri_module"))
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
  if (PyType_Ready(&c2py::wrap_pytype<_c2py_cls_1>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
#define _add_type(T, N) c2py::add_type_object_to_main<T>(N, m, conv_table)
  _add_type(_c2py_cls_0, "ThcCoulomb");
  _add_type(_c2py_cls_1, "CholCoulomb");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
