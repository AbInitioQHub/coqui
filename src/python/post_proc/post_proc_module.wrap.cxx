
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

template <> constexpr bool c2py::is_wrapped<methods::lqp::result_t> = true;
template <>
constexpr bool c2py::is_wrapped<methods::lqp::ladder_result_t> = true;

// ==================== enums =====================

// ==================== module classes =====================

template <>
inline constexpr auto c2py::tp_name<methods::lqp::result_t> =
    "post_proc_module.ResultT";

static int synth_constructor_0(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing methods::lqp::result_t.\nNo "
                     "positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<methods::lqp::result_t> *)self)->_c =
        new methods::lqp::result_t{};
  } catch (std::exception const &e) {
    PyErr_SetString(
        PyExc_RuntimeError,
        ("Error in constructing methods::lqp::result_t from a Python dict.\n   "s +
         e.what())
            .c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<methods::lqp::result_t> *)self)->_c);
  de("Z_skab", self_c.Z_skab, false);
  de("Hqp_skab", self_c.Hqp_skab, false);
  de("V_skab", self_c.V_skab, false);
  de("E_ska", self_c.E_ska, false);
  de("Zqp_ska", self_c.Zqp_ska, false);
  de("min_eig_sk", self_c.min_eig_sk, false);
  de("resid_sk", self_c.resid_sk, false);
  de("anti_herm_A_sk", self_c.anti_herm_A_sk, false);
  de("anti_herm_B_sk", self_c.anti_herm_B_sk, false);
  de("herm_data_sk", self_c.herm_data_sk, false);
  de("sumrule_sk", self_c.sumrule_sk, false);
  de("status_sk", self_c.status_sk, false);
  de("mu", self_c.mu, true);
  de("n_fit", self_c.n_fit, true);
  de("fit_order", self_c.fit_order, true);
  de("cond", self_c.cond, true);
  return de.check();
}

template <>
constexpr initproc c2py::tp_init<methods::lqp::result_t> = synth_constructor_0;

template <>
const std::string c2py::tp_ctor_doc<methods::lqp::result_t> =
    c2py::replace_tags(
        R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
Z_skab : {par_0}

Hqp_skab : {par_1}

V_skab : {par_2}

E_ska : {par_3}

Zqp_ska : {par_4}

min_eig_sk : {par_5}

resid_sk : {par_6}

anti_herm_A_sk : {par_7}

anti_herm_B_sk : {par_8}

herm_data_sk : {par_9}

sumrule_sk : {par_10}

status_sk : {par_11}

mu : {par_12}, default=0.0

n_fit : {par_13}, default=0

fit_order : {par_14}, default=0

cond : {par_15}, default=0.0

)DOC",
        "par",
        {c2py::python_typename<
             nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<std::complex<double>, 4, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 3, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 3, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<long, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<double>(), c2py::python_typename<int>(),
         c2py::python_typename<int>(), c2py::python_typename<double>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<methods::lqp::result_t>[] = {

    {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto doc_member_0 = R"DOC()DOC";
constexpr auto doc_member_1 = R"DOC()DOC";
constexpr auto doc_member_2 = R"DOC()DOC";
constexpr auto doc_member_3 = R"DOC()DOC";
constexpr auto doc_member_4 = R"DOC()DOC";
constexpr auto doc_member_5 = R"DOC()DOC";
constexpr auto doc_member_6 = R"DOC()DOC";
constexpr auto doc_member_7 = R"DOC()DOC";
constexpr auto doc_member_8 = R"DOC()DOC";
constexpr auto doc_member_9 = R"DOC()DOC";
constexpr auto doc_member_10 = R"DOC()DOC";
constexpr auto doc_member_11 = R"DOC()DOC";
constexpr auto doc_member_12 = R"DOC()DOC";
constexpr auto doc_member_13 = R"DOC()DOC";
constexpr auto doc_member_14 = R"DOC()DOC";
constexpr auto doc_member_15 = R"DOC()DOC";
static PyObject *prop_get_dict_0(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<methods::lqp::result_t> *)self)->_c);
  c2py::pydict dic;
  dic["Z_skab"] = self_c.Z_skab;
  dic["Hqp_skab"] = self_c.Hqp_skab;
  dic["V_skab"] = self_c.V_skab;
  dic["E_ska"] = self_c.E_ska;
  dic["Zqp_ska"] = self_c.Zqp_ska;
  dic["min_eig_sk"] = self_c.min_eig_sk;
  dic["resid_sk"] = self_c.resid_sk;
  dic["anti_herm_A_sk"] = self_c.anti_herm_A_sk;
  dic["anti_herm_B_sk"] = self_c.anti_herm_B_sk;
  dic["herm_data_sk"] = self_c.herm_data_sk;
  dic["sumrule_sk"] = self_c.sumrule_sk;
  dic["status_sk"] = self_c.status_sk;
  dic["mu"] = self_c.mu;
  dic["n_fit"] = self_c.n_fit;
  dic["fit_order"] = self_c.fit_order;
  dic["cond"] = self_c.cond;
  return dic.new_ref();
}

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<methods::lqp::result_t>[] = {
    c2py::getsetdef_from_member<&methods::lqp::result_t::Z_skab,
                                methods::lqp::result_t>("Z_skab", doc_member_0),
    c2py::getsetdef_from_member<&methods::lqp::result_t::Hqp_skab,
                                methods::lqp::result_t>("Hqp_skab",
                                                        doc_member_1),
    c2py::getsetdef_from_member<&methods::lqp::result_t::V_skab,
                                methods::lqp::result_t>("V_skab", doc_member_2),
    c2py::getsetdef_from_member<&methods::lqp::result_t::E_ska,
                                methods::lqp::result_t>("E_ska", doc_member_3),
    c2py::getsetdef_from_member<&methods::lqp::result_t::Zqp_ska,
                                methods::lqp::result_t>("Zqp_ska",
                                                        doc_member_4),
    c2py::getsetdef_from_member<&methods::lqp::result_t::min_eig_sk,
                                methods::lqp::result_t>("min_eig_sk",
                                                        doc_member_5),
    c2py::getsetdef_from_member<&methods::lqp::result_t::resid_sk,
                                methods::lqp::result_t>("resid_sk",
                                                        doc_member_6),
    c2py::getsetdef_from_member<&methods::lqp::result_t::anti_herm_A_sk,
                                methods::lqp::result_t>("anti_herm_A_sk",
                                                        doc_member_7),
    c2py::getsetdef_from_member<&methods::lqp::result_t::anti_herm_B_sk,
                                methods::lqp::result_t>("anti_herm_B_sk",
                                                        doc_member_8),
    c2py::getsetdef_from_member<&methods::lqp::result_t::herm_data_sk,
                                methods::lqp::result_t>("herm_data_sk",
                                                        doc_member_9),
    c2py::getsetdef_from_member<&methods::lqp::result_t::sumrule_sk,
                                methods::lqp::result_t>("sumrule_sk",
                                                        doc_member_10),
    c2py::getsetdef_from_member<&methods::lqp::result_t::status_sk,
                                methods::lqp::result_t>("status_sk",
                                                        doc_member_11),
    c2py::getsetdef_from_member<&methods::lqp::result_t::mu,
                                methods::lqp::result_t>("mu", doc_member_12),
    c2py::getsetdef_from_member<&methods::lqp::result_t::n_fit,
                                methods::lqp::result_t>("n_fit", doc_member_13),
    c2py::getsetdef_from_member<&methods::lqp::result_t::fit_order,
                                methods::lqp::result_t>("fit_order",
                                                        doc_member_14),
    c2py::getsetdef_from_member<&methods::lqp::result_t::cond,
                                methods::lqp::result_t>("cond", doc_member_15),
    {"__dict__", (getter)prop_get_dict_0, nullptr, "", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<methods::lqp::result_t> =
    R"DOC(What a full-array solve returns: qp_matrix_t for every (spin, k), assembled into
(ns, nk, ...) arrays, plus the per-point fit diagnostics and a per-point status. This is
the single result type of every consumer, self-consistent or one-shot, C++ or Python;
produced by linearized_qp_solve().

Three (ns, nk, n, n) matrices per point, for the reasons given at qp_matrix_t. The fit's
own outputs K = F + A - mu and B are not stored: the static-versus-dynamic analyses that
want them rebuild them from Z and Hqp (K = Z^-1/2 Hqp Z^-1/2, B = 1 - Z^-1); this struct is
also what Python receives, wrapped as post_proc_module.ResultT with these member names.

Entries of a point whose status is nonzero are zero (the diagnostics are still filled for
status 1, plus min_eig for status 2). With on_failure_e::abort the driver never returns
such a point, so status_sk is then identically zero.)DOC" +
    std::string{"\n\n----------\n\n"} +
    c2py::tp_ctor_doc<methods::lqp::result_t>;
template <>
inline constexpr auto c2py::tp_name<methods::lqp::ladder_result_t> =
    "post_proc_module.LadderResultT";

static int synth_constructor_1(PyObject *self, PyObject *args,
                               PyObject *kwargs) {
  if (args and PyTuple_Check(args) and (PyTuple_Size(args) > 0)) {
    PyErr_SetString(PyExc_RuntimeError,
                    ("Error in constructing methods::lqp::ladder_result_t.\nNo "
                     "positional arguments allowed. Use keywords arguments"));
    return -1;
  }
  c2py::pydict_extractor de{kwargs};
  try {
    ((c2py::wrap<methods::lqp::ladder_result_t> *)self)->_c =
        new methods::lqp::ladder_result_t{};
  } catch (std::exception const &e) {
    PyErr_SetString(
        PyExc_RuntimeError,
        ("Error in constructing methods::lqp::ladder_result_t from a Python dict.\n   "s +
         e.what())
            .c_str());
    return -1;
  }
  auto &self_c = *(((c2py::wrap<methods::lqp::ladder_result_t> *)self)->_c);
  de("last", self_c.last, false);
  de("err_fit_sk", self_c.err_fit_sk, false);
  de("dHqp_sk", self_c.dHqp_sk, false);
  de("n_fit_history", self_c.n_fit_history, false);
  de("resid_history", self_c.resid_history, false);
  de("Zqp_history", self_c.Zqp_history, false);
  de("n_accepted", self_c.n_accepted, true);
  de("mesh_limited", self_c.mesh_limited, true);
  de("n_fit_mesh_max", self_c.n_fit_mesh_max, true);
  de("stopped_n_fit", self_c.stopped_n_fit, true);
  de("stopped_resid", self_c.stopped_resid, true);
  de("stopped_cond", self_c.stopped_cond, true);
  return de.check();
}

template <>
constexpr initproc c2py::tp_init<methods::lqp::ladder_result_t> =
    synth_constructor_1;

template <>
const std::string c2py::tp_ctor_doc<methods::lqp::ladder_result_t> =
    c2py::replace_tags(
        R"DOC(Synthesized constructor with the following keyword arguments:

Parameters
----------
last : {par_0}

err_fit_sk : {par_1}

dHqp_sk : {par_2}

n_fit_history : {par_3}

resid_history : {par_4}

Zqp_history : {par_5}

n_accepted : {par_6}, default=0

mesh_limited : {par_7}, default=false

n_fit_mesh_max : {par_8}, default=0

stopped_n_fit : {par_9}, default=0

stopped_resid : {par_10}, default=0.0

stopped_cond : {par_11}, default=0.0

)DOC",
        "par",
        {c2py::python_typename<methods::lqp::result_t>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 2, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<long, 1, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 1, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<
             nda::basic_array<double, 4, nda::C_layout, 'A',
                              nda::heap_basic<nda::mem::mallocator<
                                  nda::mem::AddressSpace::Host>>>>(),
         c2py::python_typename<int>(), c2py::python_typename<bool>(),
         c2py::python_typename<int>(), c2py::python_typename<int>(),
         c2py::python_typename<double>(), c2py::python_typename<double>()});

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<methods::lqp::ladder_result_t>[] = {

    {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto doc_member_16 = R"DOC()DOC";
constexpr auto doc_member_17 = R"DOC()DOC";
constexpr auto doc_member_18 = R"DOC()DOC";
constexpr auto doc_member_19 = R"DOC()DOC";
constexpr auto doc_member_20 = R"DOC()DOC";
constexpr auto doc_member_21 = R"DOC()DOC";
constexpr auto doc_member_22 = R"DOC()DOC";
constexpr auto doc_member_23 = R"DOC()DOC";
constexpr auto doc_member_24 = R"DOC()DOC";
constexpr auto doc_member_25 = R"DOC()DOC";
constexpr auto doc_member_26 = R"DOC()DOC";
constexpr auto doc_member_27 = R"DOC()DOC";
static PyObject *prop_get_dict_1(PyObject *self, void *) {
  auto &self_c = *(((c2py::wrap<methods::lqp::ladder_result_t> *)self)->_c);
  c2py::pydict dic;
  dic["last"] = self_c.last;
  dic["err_fit_sk"] = self_c.err_fit_sk;
  dic["dHqp_sk"] = self_c.dHqp_sk;
  dic["n_fit_history"] = self_c.n_fit_history;
  dic["resid_history"] = self_c.resid_history;
  dic["Zqp_history"] = self_c.Zqp_history;
  dic["n_accepted"] = self_c.n_accepted;
  dic["mesh_limited"] = self_c.mesh_limited;
  dic["n_fit_mesh_max"] = self_c.n_fit_mesh_max;
  dic["stopped_n_fit"] = self_c.stopped_n_fit;
  dic["stopped_resid"] = self_c.stopped_resid;
  dic["stopped_cond"] = self_c.stopped_cond;
  return dic.new_ref();
}

// ----- Method table ----

template <>
constinit PyGetSetDef c2py::tp_getset<methods::lqp::ladder_result_t>[] = {
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::last,
                                methods::lqp::ladder_result_t>("last",
                                                               doc_member_16),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::err_fit_sk,
                                methods::lqp::ladder_result_t>("err_fit_sk",
                                                               doc_member_17),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::dHqp_sk,
                                methods::lqp::ladder_result_t>("dHqp_sk",
                                                               doc_member_18),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::n_fit_history,
                                methods::lqp::ladder_result_t>("n_fit_history",
                                                               doc_member_19),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::resid_history,
                                methods::lqp::ladder_result_t>("resid_history",
                                                               doc_member_20),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::Zqp_history,
                                methods::lqp::ladder_result_t>("Zqp_history",
                                                               doc_member_21),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::n_accepted,
                                methods::lqp::ladder_result_t>("n_accepted",
                                                               doc_member_22),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::mesh_limited,
                                methods::lqp::ladder_result_t>("mesh_limited",
                                                               doc_member_23),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::n_fit_mesh_max,
                                methods::lqp::ladder_result_t>("n_fit_mesh_max",
                                                               doc_member_24),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::stopped_n_fit,
                                methods::lqp::ladder_result_t>("stopped_n_fit",
                                                               doc_member_25),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::stopped_resid,
                                methods::lqp::ladder_result_t>("stopped_resid",
                                                               doc_member_26),
    c2py::getsetdef_from_member<&methods::lqp::ladder_result_t::stopped_cond,
                                methods::lqp::ladder_result_t>("stopped_cond",
                                                               doc_member_27),
    {"__dict__", (getter)prop_get_dict_1, nullptr, "", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<methods::lqp::ladder_result_t> =
    R"DOC(Result of linearized_qp_ladder(): the last accepted rung plus what the climb measured.
Only `last` is a full solution; everything else is bookkeeping of the convergence rule.)DOC" +
    std::string{"\n\n----------\n\n"} +
    c2py::tp_ctor_doc<methods::lqp::ladder_result_t>;

// ==================== module functions ====================

// band_interpolation
static auto const fun_0 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::band_interpolation(mf, params);
    },
    "mf", "params")};

// dump_hartree
static auto const fun_1 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::dump_hartree(mf, params);
    },
    "mf", "params")};

// dump_vxc
static auto const fun_2 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::dump_vxc(mf, params);
    },
    "mf", "params")};

// linearized_qp_ladder
static auto const fun_3 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](nda::basic_array<
           std::complex<double>, 4, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           F_skab,
       nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           Sigma_tskab,
       double mu, double beta, double wmax, const std::string &basis,
       const std::string &prec, int n_fit_max, double fit_resid_tol,
       bool symmetric_window) {
      return coqui_py::post_proc::linearized_qp_ladder(
          F_skab, Sigma_tskab, mu, beta, wmax, basis, prec, n_fit_max,
          fit_resid_tol, symmetric_window);
    },
    "F_skab", "Sigma_tskab", "mu", "beta", "wmax", "basis", "prec", "n_fit_max",
    "fit_resid_tol", "symmetric_window")};

// linearized_qp_solve
static auto const fun_4 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](nda::basic_array<
           std::complex<double>, 4, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           F_skab,
       nda::basic_array<
           std::complex<double>, 5, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           Sigma_tskab,
       double mu, double beta, double wmax, const std::string &basis,
       const std::string &prec, int n_fit, int fit_order, bool symmetric_window,
       double fit_resid_tol) {
      return coqui_py::post_proc::linearized_qp_solve(
          F_skab, Sigma_tskab, mu, beta, wmax, basis, prec, n_fit, fit_order,
          symmetric_window, fit_resid_tol);
    },
    "F_skab", "Sigma_tskab", "mu", "beta", "wmax", "basis", "prec", "n_fit",
    "fit_order", "symmetric_window", "fit_resid_tol")};

// local_dos
static auto const fun_5 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::local_dos(mf, params);
    },
    "mf", "params")};

// pade
static auto const fun_6 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](nda::basic_array<
           std::complex<double>, 2, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           A_iw,
       nda::basic_array<
           std::complex<double>, 1, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           iw_mesh,
       double w_min, double w_max, long Nw, int Nfit, double eta,
       bool is_iw_pos_only) {
      return coqui_py::post_proc::pade(A_iw, iw_mesh, w_min, w_max, Nw, Nfit,
                                       eta, is_iw_pos_only);
    },
    "A_iw", "iw_mesh", "w_min", "w_max", "Nw", "Nfit", "eta",
    "is_iw_pos_only")};

// spectral_interpolation
static auto const fun_7 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::spectral_interpolation(mf, params);
    },
    "mf", "params")};

// unfold_bz
static auto const fun_8 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const coqui_py::Mf &mf, const std::string &params) {
      return coqui_py::post_proc::unfold_bz(mf, params);
    },
    "mf", "params")};

static const auto doc_d_0 = fun_0.doc(R"DOC()DOC");
static const auto doc_d_1 = fun_1.doc(R"DOC()DOC");
static const auto doc_d_2 = fun_2.doc(R"DOC()DOC");
static const auto doc_d_3 = fun_3.doc(R"DOC(
pproc_t::linearized_qp_ladder on arrays, returned as the kernel's methods::lqp::ladder_result_t
(Python class LadderResultT): the last accepted rung in `last` plus the climb's bookkeeping
(err_fit_sk, dHqp_sk, histories, n_accepted, mesh_limited, n_fit_mesh_max, stopped_*).
n_accepted == 0 means even n_fit = 2 failed the gate and `last` is empty.
)DOC");
static const auto doc_d_4 = fun_4.doc(R"DOC(
pproc_t::linearized_qp on arrays: one fit at a fixed window, returned as the kernel's own
methods::lqp::result_t (Python class ResultT). E_ska is measured from mu; status_sk (ns, nk)
is 0 ok, 1 the residual gate failed, 2 = 1 - B not positive definite, and the entries of a
failed point are zero. Sigma_tskab must be on the fermionic tau mesh of
IAFT(beta, wmax, basis, prec) and F_skab must include H0.
)DOC");
static const auto doc_d_5 = fun_5.doc(R"DOC()DOC");
static const auto doc_d_6 = fun_6.doc(R"DOC()DOC");
static const auto doc_d_7 = fun_7.doc(R"DOC()DOC");
static const auto doc_d_8 = fun_8.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"band_interpolation", (PyCFunction)c2py::pyfkw<fun_0>,
     METH_VARARGS | METH_KEYWORDS, doc_d_0.c_str()},
    {"dump_hartree", (PyCFunction)c2py::pyfkw<fun_1>,
     METH_VARARGS | METH_KEYWORDS, doc_d_1.c_str()},
    {"dump_vxc", (PyCFunction)c2py::pyfkw<fun_2>, METH_VARARGS | METH_KEYWORDS,
     doc_d_2.c_str()},
    {"linearized_qp_ladder", (PyCFunction)c2py::pyfkw<fun_3>,
     METH_VARARGS | METH_KEYWORDS, doc_d_3.c_str()},
    {"linearized_qp_solve", (PyCFunction)c2py::pyfkw<fun_4>,
     METH_VARARGS | METH_KEYWORDS, doc_d_4.c_str()},
    {"local_dos", (PyCFunction)c2py::pyfkw<fun_5>, METH_VARARGS | METH_KEYWORDS,
     doc_d_5.c_str()},
    {"pade", (PyCFunction)c2py::pyfkw<fun_6>, METH_VARARGS | METH_KEYWORDS,
     doc_d_6.c_str()},
    {"spectral_interpolation", (PyCFunction)c2py::pyfkw<fun_7>,
     METH_VARARGS | METH_KEYWORDS, doc_d_7.c_str()},
    {"unfold_bz", (PyCFunction)c2py::pyfkw<fun_8>, METH_VARARGS | METH_KEYWORDS,
     doc_d_8.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "post_proc_module",                                /* name of module */
    R"RAWDOC(Post processing module for CoQui)RAWDOC", /* module documentation,
                                                          may be NULL */
    -1, /* size of per-interpreter state of the module, or -1 if the module
           keeps state in global variables. */
    module_methods,
    NULL,
    NULL,
    NULL,
    NULL};

//--------------------- module init function -----------------------------

extern "C" __attribute__((visibility("default"))) PyObject *
PyInit_post_proc_module() {

  if (not c2py::check_python_version("post_proc_module"))
    return NULL;

  // import numpy iff 'numpy/arrayobject.h' included
#ifdef Py_ARRAYOBJECT_H
  import_array();
#endif

  PyObject *m;

  if (PyType_Ready(&c2py::wrap_pytype<c2py::py_range>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<methods::lqp::result_t>) < 0)
    return NULL;
  if (PyType_Ready(&c2py::wrap_pytype<methods::lqp::ladder_result_t>) < 0)
    return NULL;

  m = PyModule_Create(&module_def);
  if (m == NULL)
    return NULL;

  auto &conv_table = *c2py::conv_table_sptr.get();

  conv_table[std::type_index(typeid(c2py::py_range)).name()] =
      &c2py::wrap_pytype<c2py::py_range>;
  c2py::add_type_object_to_main<methods::lqp::result_t>("ResultT", m,
                                                        conv_table);
  c2py::add_type_object_to_main<methods::lqp::ladder_result_t>("LadderResultT",
                                                               m, conv_table);

  return m;
}
#endif
// CLAIR_WRAP_GEN
