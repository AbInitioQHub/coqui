
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

template <> constexpr bool c2py::is_wrapped<coqui_py::Mf> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <> inline constexpr auto c2py::tp_name<coqui_py::Mf> = "mf_module.Mf";
static auto init_0 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<coqui_py::Mf, coqui_py::MpiHandler &, std::string,
                        const std::string &>("mpi", "mf_params", "mf_type")};
template <>
constexpr initproc c2py::tp_init<coqui_py::Mf> =
    c2py::pyfkw_constructor<init_0>;
template <>
const std::string c2py::tp_ctor_doc<coqui_py::Mf> = init_0.doc(R"DOC()DOC");
// ecutrho
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.ecutrho(); },
    "self")};

// ecutwfc
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.ecutwfc(); },
    "self")};

// fft_grid
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.fft_grid(); },
    "self")};

// fft_grid_wfc
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) {
      return self.fft_grid_wfc();
    },
    "self")};

// k_weights
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.k_weights(); },
    "self")};

// kp_grid
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.kp_grid(); },
    "self")};

// kpts
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.kpts(); },
    "self")};

// kpts_ibz
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.kpts_ibz(); },
    "self")};

// mf_type
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.mf_type(); },
    "self")};

// mpi
static auto const fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.mpi(); },
    "self")};

// nbnd
static auto const fun_10 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nbnd(); },
    "self")};

// nelec
static auto const fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nelec(); },
    "self")};

// nkpts
static auto const fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nkpts(); },
    "self")};

// nkpts_ibz
static auto const fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nkpts_ibz(); },
    "self")};

// npol
static auto const fun_14 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.npol(); },
    "self")};

// nqpts
static auto const fun_15 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nqpts(); },
    "self")};

// nqpts_ibz
static auto const fun_16 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nqpts_ibz(); },
    "self")};

// nspin
static auto const fun_17 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.nspin(); },
    "self")};

// nuclear_energy
static auto const fun_18 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) {
      return self.nuclear_energy();
    },
    "self")};

// outdir
static auto const fun_19 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.outdir(); },
    "self")};

// prefix
static auto const fun_20 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.prefix(); },
    "self")};

// qpts
static auto const fun_21 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.qpts(); },
    "self")};

// qpts_ibz
static auto const fun_22 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](coqui_py::Mf const &self) -> decltype(auto) { return self.qpts_ibz(); },
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
static const auto doc_d_12 = fun_12.doc(R"DOC()DOC");
static const auto doc_d_13 = fun_13.doc(R"DOC()DOC");
static const auto doc_d_14 = fun_14.doc(R"DOC()DOC");
static const auto doc_d_15 = fun_15.doc(R"DOC()DOC");
static const auto doc_d_16 = fun_16.doc(R"DOC()DOC");
static const auto doc_d_17 = fun_17.doc(R"DOC()DOC");
static const auto doc_d_18 = fun_18.doc(R"DOC()DOC");
static const auto doc_d_19 = fun_19.doc(R"DOC()DOC");
static const auto doc_d_20 = fun_20.doc(R"DOC()DOC");
static const auto doc_d_21 = fun_21.doc(R"DOC()DOC");
static const auto doc_d_22 = fun_22.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<coqui_py::Mf>[] = {
    {"ecutrho", (PyCFunction)c2py::pyfkw<fun_0>, METH_VARARGS | METH_KEYWORDS,
     doc_d_0.c_str()},
    {"ecutwfc", (PyCFunction)c2py::pyfkw<fun_1>, METH_VARARGS | METH_KEYWORDS,
     doc_d_1.c_str()},
    {"fft_grid", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS,
     doc_d_2.c_str()},
    {"fft_grid_wfc", (PyCFunction)c2py::pyfkw<fun_3>,
     METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
    {"k_weights", (PyCFunction)c2py::pyfkw<fun_4>, METH_VARARGS | METH_KEYWORDS,
     doc_d_4.c_str()},
    {"kp_grid", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS,
     doc_d_5.c_str()},
    {"kpts", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS,
     doc_d_6.c_str()},
    {"kpts_ibz", (PyCFunction)c2py::pyfkw<fun_7>, METH_VARARGS | METH_KEYWORDS,
     doc_d_7.c_str()},
    {"mf_type", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS,
     doc_d_8.c_str()},
    {"mpi", (PyCFunction)c2py::pyfkw<fun_9>, METH_VARARGS | METH_KEYWORDS,
     doc_d_9.c_str()},
    {"nbnd", (PyCFunction)c2py::pyfkw<fun_10>, METH_VARARGS | METH_KEYWORDS,
     doc_d_10.c_str()},
    {"nelec", (PyCFunction)c2py::pyfkw<fun_11>, METH_VARARGS | METH_KEYWORDS,
     doc_d_11.c_str()},
    {"nkpts", (PyCFunction)c2py::pyfkw<fun_12>, METH_VARARGS | METH_KEYWORDS,
     doc_d_12.c_str()},
    {"nkpts_ibz", (PyCFunction)c2py::pyfkw<fun_13>,
     METH_VARARGS | METH_KEYWORDS, doc_d_13.c_str()},
    {"npol", (PyCFunction)c2py::pyfkw<fun_14>, METH_VARARGS | METH_KEYWORDS,
     doc_d_14.c_str()},
    {"nqpts", (PyCFunction)c2py::pyfkw<fun_15>, METH_VARARGS | METH_KEYWORDS,
     doc_d_15.c_str()},
    {"nqpts_ibz", (PyCFunction)c2py::pyfkw<fun_16>,
     METH_VARARGS | METH_KEYWORDS, doc_d_16.c_str()},
    {"nspin", (PyCFunction)c2py::pyfkw<fun_17>, METH_VARARGS | METH_KEYWORDS,
     doc_d_17.c_str()},
    {"nuclear_energy", (PyCFunction)c2py::pyfkw<fun_18>,
     METH_VARARGS | METH_KEYWORDS, doc_d_18.c_str()},
    {"outdir", (PyCFunction)c2py::pyfkw<fun_19>, METH_VARARGS | METH_KEYWORDS,
     doc_d_19.c_str()},
    {"prefix", (PyCFunction)c2py::pyfkw<fun_20>, METH_VARARGS | METH_KEYWORDS,
     doc_d_20.c_str()},
    {"qpts", (PyCFunction)c2py::pyfkw<fun_21>, METH_VARARGS | METH_KEYWORDS,
     doc_d_21.c_str()},
    {"qpts_ibz", (PyCFunction)c2py::pyfkw<fun_22>, METH_VARARGS | METH_KEYWORDS,
     doc_d_22.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<coqui_py::Mf>[] = {

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<coqui_py::Mf> =
    R"DOC(Mf class

The Mf class encapsulates the state of a mean-field inputs to CoQuí.
This class is read-only and is responsible for accessing the input data, including

  1. Metadata of the simulated system, such as the number of bands, spins, and k-points.
  2. Single-particle basis functions whom CoQuí uses to construct a many-body Hamiltonian in CoQuí.)DOC" +
    std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<coqui_py::Mf>;

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
    "mf_module",                                  /* name of module */
    R"RAWDOC(Mean-field module for CoQui)RAWDOC", /* module documentation, may
                                                     be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *PyInit_mf_module() {

  if (not c2py::check_python_version("mf_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<coqui_py::Mf>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<coqui_py::Mf>("Mf", m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
