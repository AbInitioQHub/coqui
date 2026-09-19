
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

// ==================== Wrapped classes =====================

template <> constexpr bool c2py::is_wrapped<coqui_py::ThcCoulomb> = true;
template <> constexpr bool c2py::is_wrapped<coqui_py::CholCoulomb> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <>
inline constexpr auto c2py::tp_name<coqui_py::ThcCoulomb> =
    "eri_module.ThcCoulomb";
static auto init_0 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<coqui_py::ThcCoulomb, const coqui_py::Mf &,
                        const std::string &>("mf", "thc_params")};
template <>
constexpr initproc c2py::tp_init<coqui_py::ThcCoulomb> =
    c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<coqui_py::ThcCoulomb> =
    init_0.doc(R"DOC()DOC");
// Np
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.Np();
    },
    "self")};

// init
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb &self) -> decltype(auto) { return self.init(); },
    "self")};

// initialized
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.initialized();
    },
    "self")};

// mf
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.mf();
    },
    "self")};

// mpi
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.mpi();
    },
    "self")};

// nbnd
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nbnd();
    },
    "self")};

// nkpts
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nkpts();
    },
    "self")};

// nkpts_ibz
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nkpts_ibz();
    },
    "self")};

// nqpts
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nqpts();
    },
    "self")};

// nqpts_ibz
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nqpts_ibz();
    },
    "self")};

// nspin
static auto const fun_10 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nspin();
    },
    "self")};

// nspin_in_basis
static auto const fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::ThcCoulomb const &self) -> decltype(auto) {
      return self.nspin_in_basis();
    },
    "self")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC()DOC");
static const auto doc_d_4 = fun_4.doc(R"DOC()DOC");
static const auto doc_d_5 = fun_5.doc(R"DOC()DOC");
static const auto doc_d_6 = fun_6.doc(R"DOC()DOC");
static const auto doc_d_7 = fun_7.doc(R"DOC()DOC");
static const auto doc_d_8 = fun_8.doc(R"DOC()DOC");
static const auto doc_d_9 = fun_9.doc(R"DOC()DOC");
static const auto doc_d_10 = fun_10.doc(R"DOC()DOC");
static const auto doc_d_11 = fun_11.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<coqui_py::ThcCoulomb>[] = {
    {"Np", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS,
     doc_d_0.c_str()},
    {"init", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS,
     doc_d_1.c_str()},
    {"initialized", (PyCFunction)c2py::pyfkw<fun_2>,
     METH_VARARGS | METH_KEYWORDS, doc_d_2.c_str()},
    {"mf", (PyCFunction)c2py::pyfkw<fun_3>, METH_VARARGS | METH_KEYWORDS,
     doc_d_3.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS,
     doc_d_4.c_str()},
    {"nbnd", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS,
     doc_d_5.c_str()},
    {"nkpts", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS,
     doc_d_6.c_str()},
    {"nkpts_ibz", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS,
     doc_d_7.c_str()},
    {"nqpts", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS,
     doc_d_8.c_str()},
    {"nqpts_ibz", (PyCFunction)c2py::pyfkw<fun_9>, METH_VARARGS | METH_KEYWORDS,
     doc_d_9.c_str()},
    {"nspin", (PyCFunction)c2py::pyfkw<fun_10>, METH_VARARGS | METH_KEYWORDS,
     doc_d_10.c_str()},
    {"nspin_in_basis", (PyCFunction)c2py::pyfkw<fun_11>,
     METH_VARARGS | METH_KEYWORDS, doc_d_11.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<coqui_py::ThcCoulomb>[] = {

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<coqui_py::ThcCoulomb> =
    R"DOC()DOC" + c2py::tp_ctor_doc<coqui_py::ThcCoulomb>;
template <>
inline constexpr auto c2py::tp_name<coqui_py::CholCoulomb> =
    "eri_module.CholCoulomb";
static auto init_1 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<coqui_py::CholCoulomb, const coqui_py::Mf &,
                        const std::string &>("mf", "chol_params")};
template <>
constexpr initproc c2py::tp_init<coqui_py::CholCoulomb> =
    c2py::pyfkw_constructor<init_1>;
template <>
const std::string c2py::tp_ctor_doc<coqui_py::CholCoulomb> =
    init_1.doc(R"DOC()DOC");
// mf
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::CholCoulomb const &self) -> decltype(auto) {
      return self.mf();
    },
    "self")};

// mpi
static auto const fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::CholCoulomb const &self) -> decltype(auto) {
      return self.mpi();
    },
    "self")};

static const auto doc_d_12 = fun_12.doc(R"DOC()DOC");
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<coqui_py::CholCoulomb>[] = {
    {"mf", (PyCFunction)c2py::pyfkw<fun_12>, METH_VARARGS | METH_KEYWORDS,
     doc_d_12.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<fun_13>, METH_VARARGS | METH_KEYWORDS,
     doc_d_13.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<coqui_py::CholCoulomb>[] = {

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<coqui_py::CholCoulomb> =
    R"DOC()DOC" + c2py::tp_ctor_doc<coqui_py::CholCoulomb>;

// ==================== module functions ====================

// run_isdf
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &thc_params) {
      return coqui_py::run_isdf(mf, thc_params);
    },
    "mf", "thc_params")};

static const auto doc_d_14 = fun_14.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"run_isdf", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS,
     doc_d_14.c_str()},
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
  if (PyType_Ready(&c2py::wrap_pytype<coqui_py::ThcCoulomb>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<coqui_py::CholCoulomb>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<coqui_py::ThcCoulomb>("ThcCoulomb", m,
                                                      conv_table);
  c2py::add_type_object_to_main<coqui_py::CholCoulomb>("CholCoulomb", m,
                                                       conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
