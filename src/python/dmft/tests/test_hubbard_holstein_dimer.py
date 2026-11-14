"""
Exact-diagonalization reference model of the impurity-solver tests
(coqui.dmft.tests.hubbard_holstein_dimer). Serial only; checks the model against itself.
"""
import numpy as np
import pytest

pytest.importorskip("coqui._lib.iaft_module", reason="the DLR IAFT binding is required")

from coqui.dmft.tests.hubbard_holstein_dimer import DimerParams, DimerED, make_iaft, make_problem


def test_dimer_ed():
    """The dimer's Lehmann functions are consistent with its thermal expectation values and
    converged in the boson truncation.

    Setup: the default DimerParams (beta 20, U 2, g^2/omega0 = 0.49); IAFT wmax 20, eps 1e-10.

    Cases:
      1. Occupations: -G(tau = beta^-) from the Lehmann G equals <n_sigma> to 1e-6 (the IAFT
         transform of a function with poles inside wmax is exact to its eps).
      2. Sigma_infty: a two-point fit a + b/(i w) on the two highest DLR nodes of Sigma gives a
         within 2e-3 of U <n_-sigma> + g <b + b^dag> (the exact static limit).
      3. Truncation: n_boson 25 changes G(i w) by less than 1e-8 and the occupations by 1e-10.
      4. chi_dd(nu = 0) is positive and chi_dd(i nu) is real; Pi and W have the half-mesh shape.
      5. make_problem builds a valid ImpurityProblem whose delta_iw equals V^2/(i w - eps_b).
    """
    params = DimerParams()
    iaft = make_iaft(params)
    ed = DimerED(params)
    ref = ed.reference(iaft)
    occ = ref["occupations"]
    assert 0.9 < occ.sum() < 1.4
    g_tau = iaft.w_to_tau(ref["G_iw"], 'f')
    n_from_g = -iaft.tau_interpolate(g_tau, iaft.beta, 'f')[0]
    assert np.allclose(n_from_g.real, occ, atol=1e-6)
    wn = ref["wn_f"]
    top = np.argsort(np.abs(wn))[-2:]
    iw = 1j * np.pi * wn[top] / iaft.beta
    for spin in range(2):
        sig = ref["Sigma_iw"][top, spin]
        b_coef = (sig[0] - sig[1]) / (1.0 / iw[0] - 1.0 / iw[1])
        a_coef = sig[0] - b_coef / iw[0]
        assert abs(a_coef - ref["Sigma_infty"][spin]) < 2e-3
    bigger = DimerED(DimerParams(n_boson=25)).reference(iaft)
    assert np.abs(bigger["G_iw"] - ref["G_iw"]).max() < 1e-8
    assert np.abs(bigger["occupations"] - occ).max() < 1e-10
    assert ref["chi_dd"][0] > 0.0 and np.all(np.isreal(ref["chi_dd"]))
    nw_b_pos = len(ref["wn_b"])
    assert ref["Pi_iw"].shape == (nw_b_pos,) and ref["W_iw"].shape == (nw_b_pos,)
    problem = make_problem(params, iaft, dlr_wmax=20.0, dlr_eps=1e-10)
    iw_all = 1j * np.pi * wn / iaft.beta
    assert np.allclose(problem.delta_wsab[:, 0, 0, 0], params.V**2 / (iw_all - params.eps_b))
    assert problem.target_density == pytest.approx(occ.sum())
