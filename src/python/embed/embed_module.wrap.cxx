
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

// dmft_embed
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &embed_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_hf_potentials,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_selfenergies) {
      return coqui_py::dmft_embed(mf, embed_params, projector_ksIai,
                                  band_window, kpts_crys, local_hf_potentials,
                                  local_selfenergies);
    },
    "mf", "embed_params", "projector_ksIai", "band_window", "kpts_crys",
    "local_hf_potentials", "local_selfenergies")};

// dmft_embed_with_projector_from_h5
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &embed_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_hf_potentials,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_selfenergies) {
      return coqui_py::dmft_embed_with_projector_from_h5(
          mf, embed_params, local_hf_potentials, local_selfenergies);
    },
    "mf", "embed_params", "local_hf_potentials", "local_selfenergies")};

// downfold_1e
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params) {
      return coqui_py::downfold_1e(mf, df_params);
    },
    "mf", "df_params")};

// downfold_2e
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_2e<coqui_py::ThcCoulomb>(
          eri, df_params, local_polarizabilities);
    },
    "eri", "df_params", "local_polarizabilities")};

// downfold_coulomb
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_coulomb<coqui_py::ThcCoulomb>(
          eri, df_params, projector_ksIai, band_window, kpts_crys,
          local_polarizabilities);
    },
    "eri", "df_params", "projector_ksIai", "band_window", "kpts_crys",
    "local_polarizabilities")};

// downfold_coulomb_with_projector_from_h5
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](coqui_py::ThcCoulomb &eri, const std::string &df_params,
       std::optional<std::map<
           std::string,
           nda::basic_array<std::complex<double>, 5, nda::C_layout, 'A',
                            nda::heap_basic<nda::mem::mallocator<
                                nda::mem::AddressSpace::Host>>>>>
           local_polarizabilities) {
      return coqui_py::downfold_coulomb_with_projector_from_h5<
          coqui_py::ThcCoulomb>(eri, df_params, local_polarizabilities);
    },
    "eri", "df_params", "local_polarizabilities")};

// downfold_gloc
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params,
       const nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &projector_ksIai,
       const nda::basic_array<
           long, 3, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &band_window,
       const nda::basic_array<
           double, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &kpts_crys) {
      return coqui_py::downfold_gloc(mf, df_params, projector_ksIai,
                                     band_window, kpts_crys);
    },
    "mf", "df_params", "projector_ksIai", "band_window", "kpts_crys")};

// downfold_gloc_with_projector_from_h5
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &df_params) {
      return coqui_py::downfold_gloc_with_projector_from_h5(mf, df_params);
    },
    "mf", "df_params")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");
static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"dmft_embed", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"dmft_embed_with_projector_from_h5", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"downfold_1e", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"downfold_2e", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {"downfold_coulomb", (PyCFunction)c2py::pyfkw<_c2py_fun_4>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
    {"downfold_coulomb_with_projector_from_h5",
     (PyCFunction)c2py::pyfkw<_c2py_fun_5>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_5.c_str()},
    {"downfold_gloc", (PyCFunction)c2py::pyfkw<_c2py_fun_6>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
    {"downfold_gloc_with_projector_from_h5",
     (PyCFunction)c2py::pyfkw<_c2py_fun_7>, METH_VARARGS | METH_KEYWORDS,
     _c2py_doc_7.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "embed_module",                              /* name of module */
    R"RAWDOC(Embedding module for CoQui)RAWDOC", /* module documentation, may be
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
PyInit_embed_module() {

  if (not c2py::check_python_version("embed_module"))
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
