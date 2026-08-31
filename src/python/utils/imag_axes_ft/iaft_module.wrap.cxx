
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

template <> constexpr bool c2py::is_wrapped<imag_axes_ft::stats_e> = true;
template <>
const std::map<imag_axes_ft::stats_e, str_t>
    c2py::enum_to_string<imag_axes_ft::stats_e> = {
        {imag_axes_ft::stats_e::fermion, "fermion"},
        {imag_axes_ft::stats_e::boson, "boson"}};
template <> constexpr bool c2py::is_wrapped<imag_axes_ft::basis_e> = true;
template <>
const std::map<imag_axes_ft::basis_e, str_t>
    c2py::enum_to_string<imag_axes_ft::basis_e> = {
        {imag_axes_ft::basis_e::dlr_basis, "dlr_basis"},
        {imag_axes_ft::basis_e::ir_basis, "ir_basis"}};

// ==================== module classes =====================

// --------- class _c2py_cls_0 -----------
using _c2py_cls_0 = imag_axes_ft::ir::IR;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_0> = true;
template <> inline constexpr auto c2py::tp_name<_c2py_cls_0> = "iaft_module.IR";
static const auto _c2py_init_0 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<_c2py_cls_0>(),
    c2py::c_constructor<_c2py_cls_0, double, double, std::string, bool>(
        "beta_", "wmax_", "prec_"_a = "high", "print_meta_log"_a = false)};
template <>
constexpr initproc c2py::tp_init<_c2py_cls_0> =
    c2py::pyfkw_constructor<_c2py_init_0>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_0> =
    _c2py_init_0.doc(R"DOC()DOC");
// determine_lambda
static auto const _c2py_fun_0 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 &self, double lbda) -> decltype(auto) {
      return self.determine_lambda(lbda);
    },
    "self", "lbda")};

// ir_file
static auto const _c2py_fun_1 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 &self, double lmbda, std::string prec_prefix)
        -> decltype(auto) { return self.ir_file(lmbda, prec_prefix); },
    "self", "lmbda", "prec_prefix")};

// metadata_log
static auto const _c2py_fun_2 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self) -> decltype(auto) {
      return self.metadata_log();
    },
    "self")};

// prec_to_eps
static auto const _c2py_fun_3 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self, const std::string &prec_) -> decltype(auto) {
      return self.prec_to_eps(prec_);
    },
    "self", "prec_")};

// prec_to_prefix
static auto const _c2py_fun_4 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_0 const &self, const std::string &prec_) -> decltype(auto) {
      return self.prec_to_prefix(prec_);
    },
    "self", "prec_")};

static const auto _c2py_doc_0 = _c2py_fun_0.doc(R"DOC()DOC");
static const auto _c2py_doc_1 = _c2py_fun_1.doc(R"DOC()DOC");
static const auto _c2py_doc_2 = _c2py_fun_2.doc(R"DOC()DOC");
static const auto _c2py_doc_3 = _c2py_fun_3.doc(R"DOC()DOC");
static const auto _c2py_doc_4 = _c2py_fun_4.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_0>[] = {
    {"determine_lambda", (PyCFunction)c2py::pyfkw<_c2py_fun_0>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_0.c_str()},
    {"ir_file", (PyCFunction)c2py::pyfkw<_c2py_fun_1>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_1.c_str()},
    {"metadata_log", (PyCFunction)c2py::pyfkw<_c2py_fun_2>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_2.c_str()},
    {"prec_to_eps", (PyCFunction)c2py::pyfkw<_c2py_fun_3>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_3.c_str()},
    {"prec_to_prefix", (PyCFunction)c2py::pyfkw<_c2py_fun_4>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_4.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

constexpr auto _c2py_doc_member_0 = R"DOC()DOC";
constexpr auto _c2py_doc_member_1 = R"DOC()DOC";
constexpr auto _c2py_doc_member_2 = R"DOC()DOC";
constexpr auto _c2py_doc_member_3 = R"DOC()DOC";
constexpr auto _c2py_doc_member_4 = R"DOC()DOC";
constexpr auto _c2py_doc_member_5 = R"DOC()DOC";
constexpr auto _c2py_doc_member_6 = R"DOC()DOC";
constexpr auto _c2py_doc_member_7 = R"DOC()DOC";
constexpr auto _c2py_doc_member_8 = R"DOC()DOC";
constexpr auto _c2py_doc_member_9 = R"DOC()DOC";
constexpr auto _c2py_doc_member_10 = R"DOC()DOC";
constexpr auto _c2py_doc_member_11 = R"DOC()DOC";
constexpr auto _c2py_doc_member_12 = R"DOC()DOC";
constexpr auto _c2py_doc_member_13 = R"DOC()DOC";
constexpr auto _c2py_doc_member_14 = R"DOC()DOC";
constexpr auto _c2py_doc_member_15 = R"DOC()DOC";
constexpr auto _c2py_doc_member_16 = R"DOC()DOC";
constexpr auto _c2py_doc_member_17 = R"DOC()DOC";
constexpr auto _c2py_doc_member_18 = R"DOC()DOC";
constexpr auto _c2py_doc_member_19 = R"DOC()DOC";
constexpr auto _c2py_doc_member_20 = R"DOC()DOC";
constexpr auto _c2py_doc_member_21 = R"DOC()DOC";
constexpr auto _c2py_doc_member_22 = R"DOC()DOC";

// ----- Member and property table ----

template <>
constinit PyGetSetDef c2py::tp_getset<_c2py_cls_0>[] = {
    c2py::getsetdef_from_member<&_c2py_cls_0::beta, _c2py_cls_0>(
        "beta", _c2py_doc_member_0),
    c2py::getsetdef_from_member<&_c2py_cls_0::wmax, _c2py_cls_0>(
        "wmax", _c2py_doc_member_1),
    c2py::getsetdef_from_member<&_c2py_cls_0::prec, _c2py_cls_0>(
        "prec", _c2py_doc_member_2),
    c2py::getsetdef_from_member<&_c2py_cls_0::eps, _c2py_cls_0>(
        "eps", _c2py_doc_member_3),
    c2py::getsetdef_from_member<&_c2py_cls_0::lambda, _c2py_cls_0>(
        "lambda", _c2py_doc_member_4),
    c2py::getsetdef_from_member<&_c2py_cls_0::nt_f, _c2py_cls_0>(
        "nt_f", _c2py_doc_member_5),
    c2py::getsetdef_from_member<&_c2py_cls_0::nt_b, _c2py_cls_0>(
        "nt_b", _c2py_doc_member_6),
    c2py::getsetdef_from_member<&_c2py_cls_0::nw_f, _c2py_cls_0>(
        "nw_f", _c2py_doc_member_7),
    c2py::getsetdef_from_member<&_c2py_cls_0::nw_b, _c2py_cls_0>(
        "nw_b", _c2py_doc_member_8),
    c2py::getsetdef_from_member<&_c2py_cls_0::wn_mesh_f, _c2py_cls_0>(
        "wn_mesh_f", _c2py_doc_member_9),
    c2py::getsetdef_from_member<&_c2py_cls_0::wn_mesh_b, _c2py_cls_0>(
        "wn_mesh_b", _c2py_doc_member_10),
    c2py::getsetdef_from_member<&_c2py_cls_0::tau_mesh_f, _c2py_cls_0>(
        "tau_mesh_f", _c2py_doc_member_11),
    c2py::getsetdef_from_member<&_c2py_cls_0::tau_mesh_b, _c2py_cls_0>(
        "tau_mesh_b", _c2py_doc_member_12),
    c2py::getsetdef_from_member<&_c2py_cls_0::Ttw_ff, _c2py_cls_0>(
        "Ttw_ff", _c2py_doc_member_13),
    c2py::getsetdef_from_member<&_c2py_cls_0::Twt_ff, _c2py_cls_0>(
        "Twt_ff", _c2py_doc_member_14),
    c2py::getsetdef_from_member<&_c2py_cls_0::Ttw_bb, _c2py_cls_0>(
        "Ttw_bb", _c2py_doc_member_15),
    c2py::getsetdef_from_member<&_c2py_cls_0::Twt_bb, _c2py_cls_0>(
        "Twt_bb", _c2py_doc_member_16),
    c2py::getsetdef_from_member<&_c2py_cls_0::Ttt_bf, _c2py_cls_0>(
        "Ttt_bf", _c2py_doc_member_17),
    c2py::getsetdef_from_member<&_c2py_cls_0::Ttt_fb, _c2py_cls_0>(
        "Ttt_fb", _c2py_doc_member_18),
    c2py::getsetdef_from_member<&_c2py_cls_0::T_beta_t_ff, _c2py_cls_0>(
        "T_beta_t_ff", _c2py_doc_member_19),
    c2py::getsetdef_from_member<&_c2py_cls_0::T_zero_t_ff, _c2py_cls_0>(
        "T_zero_t_ff", _c2py_doc_member_20),
    c2py::getsetdef_from_member<&_c2py_cls_0::Tct_ff, _c2py_cls_0>(
        "Tct_ff", _c2py_doc_member_21),
    c2py::getsetdef_from_member<&_c2py_cls_0::Tct_bb, _c2py_cls_0>(
        "Tct_bb", _c2py_doc_member_22),

    {nullptr, nullptr, nullptr, nullptr, nullptr}};

template <>
const std::string c2py::tp_doc<_c2py_cls_0> =
    R"DOC()DOC" + c2py::tp_ctor_doc<_c2py_cls_0>;
// --------- class _c2py_cls_1 -----------
using _c2py_cls_1 = imag_axes_ft::IAFT;
template <> constexpr bool c2py::is_wrapped<_c2py_cls_1> = true;
template <>
inline constexpr auto c2py::tp_name<_c2py_cls_1> = "iaft_module.IAFT";
static const auto _c2py_init_1 = c2py::dispatcher_c_kw_t{
    c2py::c_constructor<_c2py_cls_1, double, double, imag_axes_ft::basis_e,
                        std::string, bool>("beta", "wmax", "basis",
                                           "prec"_a = "medium",
                                           "print_meta_log"_a = false),
    c2py::c_constructor<_c2py_cls_1, double, double, imag_axes_ft::basis_e,
                        double, bool>("beta", "wmax", "basis", "eps",
                                      "print_meta_log"_a = false)};
template <>
constexpr initproc c2py::tp_init<_c2py_cls_1> =
    c2py::pyfkw_constructor<_c2py_init_1>;
template <>
const std::string c2py::tp_ctor_doc<_c2py_cls_1> =
    _c2py_init_1.doc(R"DOC()DOC");
// T_beta_t_ff
static auto const _c2py_fun_5 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) {
      return self.T_beta_t_ff();
    },
    "self")};

// T_zero_t_ff
static auto const _c2py_fun_6 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) {
      return self.T_zero_t_ff();
    },
    "self")};

// Tct_bb
static auto const _c2py_fun_7 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Tct_bb(); },
    "self")};

// Tct_ff
static auto const _c2py_fun_8 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Tct_ff(); },
    "self")};

// Ttt_bf
static auto const _c2py_fun_9 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Ttt_bf(); },
    "self")};

// Ttt_fb
static auto const _c2py_fun_10 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Ttt_fb(); },
    "self")};

// Ttw_bb
static auto const _c2py_fun_11 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Ttw_bb(); },
    "self")};

// Ttw_ff
static auto const _c2py_fun_12 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Ttw_ff(); },
    "self")};

// Twt_bb
static auto const _c2py_fun_13 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Twt_bb(); },
    "self")};

// Twt_ff
static auto const _c2py_fun_14 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.Twt_ff(); },
    "self")};

// basis
static auto const _c2py_fun_15 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.basis(); },
    "self")};

// beta
static auto const _c2py_fun_16 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.beta(); },
    "self")};

// construct_tau_interpolate_matrix
static auto const _c2py_fun_17 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<const imag_axes_ft::IAFT &,
               const nda::basic_array<double, 1, nda::C_layout, 'A',
                                      nda::heap_basic<nda::mem::mallocator<
                                          nda::mem::AddressSpace::Host>>> &,
               bool>(&imag_axes_ft::construct_tau_interpolate_matrix),
    "iaft", "tau_mesh_out", "ph_sym"_a = false)};

// construct_w_interpolate_matrix
static auto const _c2py_fun_18 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<const imag_axes_ft::IAFT &,
               const nda::basic_array<long, 1, nda::C_layout, 'A',
                                      nda::heap_basic<nda::mem::mallocator<
                                          nda::mem::AddressSpace::Host>>> &,
               const std::string &, bool>(
        &imag_axes_ft::construct_w_interpolate_matrix),
    "iaft", "wn_mesh_out", "stats_str", "ph_sym"_a = false)};

// eps
static auto const _c2py_fun_19 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.eps(); },
    "self")};

// lambda
static auto const _c2py_fun_20 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.lambda(); },
    "self")};

// metadata_log
static auto const _c2py_fun_21 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) {
      return self.metadata_log();
    },
    "self")};

// nt_b
static auto const _c2py_fun_22 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.nt_b(); },
    "self")};

// nt_f
static auto const _c2py_fun_23 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.nt_f(); },
    "self")};

// nw_b
static auto const _c2py_fun_24 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.nw_b(); },
    "self")};

// nw_f
static auto const _c2py_fun_25 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.nw_f(); },
    "self")};

// omega
static auto const _c2py_fun_26 = c2py::dispatcher_f_kw_t{
    c2py::cmethod([](_c2py_cls_1 const &self,
                     long n) -> decltype(auto) { return self.omega(n); },
                  "self", "n")};

// prec
static auto const _c2py_fun_27 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.prec(); },
    "self")};

// tau_mesh
static auto const _c2py_fun_28 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.tau_mesh(); },
    "self")};

// tau_mesh_b
static auto const _c2py_fun_29 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.tau_mesh_b(); },
    "self")};

// tau_mesh_f
static auto const _c2py_fun_30 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.tau_mesh_f(); },
    "self")};

// tau_to_beta_2d
static auto const _c2py_fun_31 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<
            std::complex<double>, 2, nda::C_layout, 'A',
            nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
            &>(&imag_axes_ft::tau_to_beta_2d),
    "iaft", "A_ti")};

// tau_to_tau_2d
static auto const _c2py_fun_32 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &>(&imag_axes_ft::tau_to_tau_2d),
    "iaft", "A_ti", "A_stats")};

// tau_to_tau_it_2d
static auto const _c2py_fun_33 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &, long>(&imag_axes_ft::tau_to_tau_it_2d),
    "iaft", "A_ti", "A_stats", "it_B")};

// tau_to_w_2d
static auto const _c2py_fun_34 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &>(&imag_axes_ft::tau_to_w_2d),
    "iaft", "X_ti", "stats")};

// tau_to_w_iw_2d
static auto const _c2py_fun_35 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &, long>(&imag_axes_ft::tau_to_w_iw_2d),
    "iaft", "X_ti", "stats", "iw")};

// tau_to_w_phsym_2d
static auto const _c2py_fun_36 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<
            std::complex<double>, 2, nda::C_layout, 'A',
            nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
            &>(&imag_axes_ft::tau_to_w_phsym_2d),
    "iaft", "X_ti_pos")};

// tau_to_zero_2d
static auto const _c2py_fun_37 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<
            std::complex<double>, 2, nda::C_layout, 'A',
            nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
            &>(&imag_axes_ft::tau_to_zero_2d),
    "iaft", "A_ti")};

// w_to_tau_2d
static auto const _c2py_fun_38 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &>(&imag_axes_ft::w_to_tau_2d),
    "iaft", "X_wi", "stats")};

// w_to_tau_cross_stats_2d
static auto const _c2py_fun_39 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 2, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &, const std::string &>(
        &imag_axes_ft::w_to_tau_cross_stats_2d),
    "iaft", "X_wi", "stats_A", "stats_B")};

// w_to_tau_partial_2d
static auto const _c2py_fun_40 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<std::complex<double>, 1, nda::C_layout, 'A',
                               nda::heap_basic<nda::mem::mallocator<
                                   nda::mem::AddressSpace::Host>>> &,
        const std::string &, long>(&imag_axes_ft::w_to_tau_partial_2d),
    "iaft", "Xw_i", "stats", "iwn")};

// w_to_tau_phsym_2d
static auto const _c2py_fun_41 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    c2py::cast<
        const imag_axes_ft::IAFT &,
        const nda::basic_array<
            std::complex<double>, 2, nda::C_layout, 'A',
            nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
            &>(&imag_axes_ft::w_to_tau_phsym_2d),
    "iaft", "X_wi_pos")};

// wmax
static auto const _c2py_fun_42 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.wmax(); },
    "self")};

// wn_mesh
static auto const _c2py_fun_43 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.wn_mesh(); },
    "self")};

// wn_mesh_b
static auto const _c2py_fun_44 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.wn_mesh_b(); },
    "self")};

// wn_mesh_f
static auto const _c2py_fun_45 = c2py::dispatcher_f_kw_t{c2py::cmethod(
    [](_c2py_cls_1 const &self) -> decltype(auto) { return self.wn_mesh_f(); },
    "self")};

static const auto _c2py_doc_5 = _c2py_fun_5.doc(R"DOC()DOC");
static const auto _c2py_doc_6 = _c2py_fun_6.doc(R"DOC()DOC");
static const auto _c2py_doc_7 = _c2py_fun_7.doc(R"DOC()DOC");
static const auto _c2py_doc_8 = _c2py_fun_8.doc(R"DOC()DOC");
static const auto _c2py_doc_9 = _c2py_fun_9.doc(R"DOC()DOC");
static const auto _c2py_doc_10 = _c2py_fun_10.doc(R"DOC()DOC");
static const auto _c2py_doc_11 = _c2py_fun_11.doc(R"DOC()DOC");
static const auto _c2py_doc_12 = _c2py_fun_12.doc(R"DOC()DOC");
static const auto _c2py_doc_13 = _c2py_fun_13.doc(R"DOC()DOC");
static const auto _c2py_doc_14 = _c2py_fun_14.doc(R"DOC()DOC");
static const auto _c2py_doc_15 = _c2py_fun_15.doc(R"DOC()DOC");
static const auto _c2py_doc_16 = _c2py_fun_16.doc(R"DOC()DOC");
static const auto _c2py_doc_17 = _c2py_fun_17.doc(R"DOC()DOC");
static const auto _c2py_doc_18 = _c2py_fun_18.doc(R"DOC()DOC");
static const auto _c2py_doc_19 = _c2py_fun_19.doc(R"DOC()DOC");
static const auto _c2py_doc_20 = _c2py_fun_20.doc(R"DOC()DOC");
static const auto _c2py_doc_21 = _c2py_fun_21.doc(R"DOC()DOC");
static const auto _c2py_doc_22 = _c2py_fun_22.doc(R"DOC()DOC");
static const auto _c2py_doc_23 = _c2py_fun_23.doc(R"DOC()DOC");
static const auto _c2py_doc_24 = _c2py_fun_24.doc(R"DOC()DOC");
static const auto _c2py_doc_25 = _c2py_fun_25.doc(R"DOC()DOC");
static const auto _c2py_doc_26 = _c2py_fun_26.doc(R"DOC(
Matsubara frequency iw_n = i*n*pi/beta
)DOC");
static const auto _c2py_doc_27 = _c2py_fun_27.doc(R"DOC()DOC");
static const auto _c2py_doc_28 = _c2py_fun_28.doc(R"DOC()DOC");
static const auto _c2py_doc_29 = _c2py_fun_29.doc(R"DOC()DOC");
static const auto _c2py_doc_30 = _c2py_fun_30.doc(R"DOC()DOC");
static const auto _c2py_doc_31 = _c2py_fun_31.doc(R"DOC()DOC");
static const auto _c2py_doc_32 = _c2py_fun_32.doc(R"DOC()DOC");
static const auto _c2py_doc_33 = _c2py_fun_33.doc(R"DOC()DOC");
static const auto _c2py_doc_34 = _c2py_fun_34.doc(R"DOC()DOC");
static const auto _c2py_doc_35 = _c2py_fun_35.doc(R"DOC()DOC");
static const auto _c2py_doc_36 = _c2py_fun_36.doc(R"DOC()DOC");
static const auto _c2py_doc_37 = _c2py_fun_37.doc(R"DOC()DOC");
static const auto _c2py_doc_38 = _c2py_fun_38.doc(R"DOC()DOC");
static const auto _c2py_doc_39 = _c2py_fun_39.doc(R"DOC()DOC");
static const auto _c2py_doc_40 = _c2py_fun_40.doc(R"DOC()DOC");
static const auto _c2py_doc_41 = _c2py_fun_41.doc(R"DOC()DOC");
static const auto _c2py_doc_42 = _c2py_fun_42.doc(R"DOC()DOC");
static const auto _c2py_doc_43 = _c2py_fun_43.doc(R"DOC()DOC");
static const auto _c2py_doc_44 = _c2py_fun_44.doc(R"DOC()DOC");
static const auto _c2py_doc_45 = _c2py_fun_45.doc(R"DOC()DOC");

// ----- Method table ----
template <>
PyMethodDef c2py::tp_methods<_c2py_cls_1>[] = {
    {"T_beta_t_ff", (PyCFunction)c2py::pyfkw<_c2py_fun_5>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_5.c_str()},
    {"T_zero_t_ff", (PyCFunction)c2py::pyfkw<_c2py_fun_6>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_6.c_str()},
    {"Tct_bb", (PyCFunction)c2py::pyfkw<_c2py_fun_7>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_7.c_str()},
    {"Tct_ff", (PyCFunction)c2py::pyfkw<_c2py_fun_8>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_8.c_str()},
    {"Ttt_bf", (PyCFunction)c2py::pyfkw<_c2py_fun_9>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_9.c_str()},
    {"Ttt_fb", (PyCFunction)c2py::pyfkw<_c2py_fun_10>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_10.c_str()},
    {"Ttw_bb", (PyCFunction)c2py::pyfkw<_c2py_fun_11>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_11.c_str()},
    {"Ttw_ff", (PyCFunction)c2py::pyfkw<_c2py_fun_12>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_12.c_str()},
    {"Twt_bb", (PyCFunction)c2py::pyfkw<_c2py_fun_13>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_13.c_str()},
    {"Twt_ff", (PyCFunction)c2py::pyfkw<_c2py_fun_14>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_14.c_str()},
    {"basis", (PyCFunction)c2py::pyfkw<_c2py_fun_15>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_15.c_str()},
    {"beta", (PyCFunction)c2py::pyfkw<_c2py_fun_16>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_16.c_str()},
    {"construct_tau_interpolate_matrix", (PyCFunction)c2py::pyfkw<_c2py_fun_17>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_17.c_str()},
    {"construct_w_interpolate_matrix", (PyCFunction)c2py::pyfkw<_c2py_fun_18>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_18.c_str()},
    {"eps", (PyCFunction)c2py::pyfkw<_c2py_fun_19>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_19.c_str()},
    {"lambda", (PyCFunction)c2py::pyfkw<_c2py_fun_20>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_20.c_str()},
    {"metadata_log", (PyCFunction)c2py::pyfkw<_c2py_fun_21>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_21.c_str()},
    {"nt_b", (PyCFunction)c2py::pyfkw<_c2py_fun_22>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_22.c_str()},
    {"nt_f", (PyCFunction)c2py::pyfkw<_c2py_fun_23>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_23.c_str()},
    {"nw_b", (PyCFunction)c2py::pyfkw<_c2py_fun_24>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_24.c_str()},
    {"nw_f", (PyCFunction)c2py::pyfkw<_c2py_fun_25>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_25.c_str()},
    {"omega", (PyCFunction)c2py::pyfkw<_c2py_fun_26>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_26.c_str()},
    {"prec", (PyCFunction)c2py::pyfkw<_c2py_fun_27>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_27.c_str()},
    {"tau_mesh", (PyCFunction)c2py::pyfkw<_c2py_fun_28>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_28.c_str()},
    {"tau_mesh_b", (PyCFunction)c2py::pyfkw<_c2py_fun_29>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_29.c_str()},
    {"tau_mesh_f", (PyCFunction)c2py::pyfkw<_c2py_fun_30>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_30.c_str()},
    {"tau_to_beta_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_31>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_31.c_str()},
    {"tau_to_tau_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_32>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_32.c_str()},
    {"tau_to_tau_it_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_33>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_33.c_str()},
    {"tau_to_w_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_34>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_34.c_str()},
    {"tau_to_w_iw_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_35>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_35.c_str()},
    {"tau_to_w_phsym_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_36>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_36.c_str()},
    {"tau_to_zero_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_37>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_37.c_str()},
    {"w_to_tau_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_38>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_38.c_str()},
    {"w_to_tau_cross_stats_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_39>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_39.c_str()},
    {"w_to_tau_partial_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_40>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_40.c_str()},
    {"w_to_tau_phsym_2d", (PyCFunction)c2py::pyfkw<_c2py_fun_41>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_41.c_str()},
    {"wmax", (PyCFunction)c2py::pyfkw<_c2py_fun_42>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_42.c_str()},
    {"wn_mesh", (PyCFunction)c2py::pyfkw<_c2py_fun_43>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_43.c_str()},
    {"wn_mesh_b", (PyCFunction)c2py::pyfkw<_c2py_fun_44>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_44.c_str()},
    {"wn_mesh_f", (PyCFunction)c2py::pyfkw<_c2py_fun_45>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_45.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

template <>
const std::string c2py::tp_doc<_c2py_cls_1> =
    R"DOC(A generic class for Fourier transform between imaginary axes for different types of grids on imaginary axes.
Grid information is provided from the grid driver (grid_var).)DOC" +
    std::string{"\n\n----------\n\n"} + c2py::tp_ctor_doc<_c2py_cls_1>;

// ==================== module functions ====================

// basis_enum_to_string
static auto const _c2py_fun_46 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](int basis_enum) {
      return imag_axes_ft::basis_enum_to_string(basis_enum);
    },
    "basis_enum")};

// build_g_iw_ref
static auto const _c2py_fun_47 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::basic_array<
           long, 1, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &wn_mesh,
       double beta, const std::string &stats_str, int norb, bool ph_sym) {
      return imag_axes_ft::build_g_iw_ref(wn_mesh, beta, stats_str, norb,
                                          ph_sym);
    },
    "wn_mesh", "beta", "stats_str", "norb", "ph_sym"_a = false)};

// build_g_tau_ref
static auto const _c2py_fun_48 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](const nda::basic_array<
           double, 1, nda::C_layout, 'A',
           nda::heap_basic<nda::mem::mallocator<nda::mem::AddressSpace::Host>>>
           &tau_mesh,
       double beta, int norb, bool ph_sym) {
      return imag_axes_ft::build_g_tau_ref(tau_mesh, beta, norb, ph_sym);
    },
    "tau_mesh", "beta", "norb", "ph_sym"_a = false)};

// read_iaft
static auto const _c2py_fun_49 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](std::string scf_file, bool print_meta_log) {
      return imag_axes_ft::read_iaft(scf_file, print_meta_log);
    },
    "scf_file", "print_meta_log"_a = true)};

// stats_enum_to_string
static auto const _c2py_fun_50 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](int stats_enum) {
      return imag_axes_ft::stats_enum_to_string(stats_enum);
    },
    "stats_enum")};

// string_to_basis_enum
static auto const _c2py_fun_51 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](std::string iaft_basis) {
      return imag_axes_ft::string_to_basis_enum(iaft_basis);
    },
    "iaft_basis")};

// string_to_stats_enum
static auto const _c2py_fun_52 = c2py::dispatcher_f_kw_t{c2py::cfun(
    [](std::string stats) { return imag_axes_ft::string_to_stats_enum(stats); },
    "stats")};

static const auto _c2py_doc_46 = _c2py_fun_46.doc(R"DOC()DOC");
static const auto _c2py_doc_47 = _c2py_fun_47.doc(R"DOC()DOC");
static const auto _c2py_doc_48 = _c2py_fun_48.doc(R"DOC()DOC");
static const auto _c2py_doc_49 =
    _c2py_fun_49.doc(R"DOC(
Reconstruct IAFT object from the metadata in bdft scf output

Returns
-------
{ret_0}
   IAFT
)DOC",
                     {}, {c2py::python_typename<imag_axes_ft::IAFT>()});
static const auto _c2py_doc_50 = _c2py_fun_50.doc(R"DOC()DOC");
static const auto _c2py_doc_51 = _c2py_fun_51.doc(R"DOC()DOC");
static const auto _c2py_doc_52 = _c2py_fun_52.doc(R"DOC()DOC");
//--------------------- module function table  -----------------------------

static PyMethodDef module_methods[] = {
    {"basis_enum_to_string", (PyCFunction)c2py::pyfkw<_c2py_fun_46>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_46.c_str()},
    {"build_g_iw_ref", (PyCFunction)c2py::pyfkw<_c2py_fun_47>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_47.c_str()},
    {"build_g_tau_ref", (PyCFunction)c2py::pyfkw<_c2py_fun_48>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_48.c_str()},
    {"read_iaft", (PyCFunction)c2py::pyfkw<_c2py_fun_49>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_49.c_str()},
    {"stats_enum_to_string", (PyCFunction)c2py::pyfkw<_c2py_fun_50>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_50.c_str()},
    {"string_to_basis_enum", (PyCFunction)c2py::pyfkw<_c2py_fun_51>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_51.c_str()},
    {"string_to_stats_enum", (PyCFunction)c2py::pyfkw<_c2py_fun_52>,
     METH_VARARGS | METH_KEYWORDS, _c2py_doc_52.c_str()},
    {nullptr, nullptr, 0, nullptr} // Sentinel
};

//--------------------- module struct & init error definition ------------

//// module doc directly in the code or "" if not present...
/// Or mandatory ?
static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "iaft_module",                          /* name of module */
    R"RAWDOC(IAFT module for CoQui)RAWDOC", /* module documentation, may be NULL
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
PyInit_iaft_module() {

  if (not c2py::check_python_version("iaft_module"))
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
  _add_type(_c2py_cls_0, "IR");
  _add_type(_c2py_cls_1, "IAFT");
#undef _add_type

  return m;
}
#endif
// CLAIR_WRAP_GEN
