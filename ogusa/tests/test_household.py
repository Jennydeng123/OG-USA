import numpy as np
import household
import pytest


def test_marg_ut_cons():
    c = np.array( [1] )
    sigma = 5
    marg1 = household.marg_ut_cons( c, sigma )
    assert (np.allclose( marg1, np.array( [1] ) ))


def test_marg_ut_labor():
    b_ellipse = 10
    upsilon = 20
    ltilde = 30
    chi_n = np.array( [5] )
    n = np.array( [1] )
    marg2 = household.marg_ut_labor( n, (b_ellipse, upsilon, ltilde, chi_n) )
    assert (np.allclose( marg2, np.array( [0.] ) ))


def test_get_cons():
    r = np.array( [0.1] )
    w = np.array( [15] )
    b = np.array( [5] )
    b_splus1 = np.array( [4] )
    n = np.array( [1] )
    BQ = np.array( [10] )
    net_tax = np.array( [100] )
    e = np.array( [100] )
    lambdas = np.array( [0.3] )
    g_y = 0.1
    get = household.get_cons( r, w, b, b_splus1, n, BQ, net_tax, (e, lambdas, g_y) )
    assert (np.allclose( get, np.array( [1434.41264966] ) ))


def test_FOC_savings():
    r = np.array( [10] )
    w = np.array( [1500] )
    b = np.array( [50] )
    b_splus1 = np.array( [40] )
    n = np.array( [10] )
    BQ = np.array( [10] )
    net_tax = np.array( [100] )
    e = np.array( [10] )
    lambdas = np.array( [0.3] )
    g_y = 0.3
    b_ellipse = 10
    upsilon = 20
    ltilde = 30
    chi_n = np.array( [5] )
    c = np.array( [1] )
    sigma = 5
    b_splus2 = np.array( [3] )
    factor = 100
    T_H = 1000
    beta = 0.5
    chi_b = np.array( [0.5] )
    theta = 0.3
    tau_bq = np.array( [1] )
    rho = np.array( [0.2] )
    J = 10
    S = 5
    etr_params = np.array( [[5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5],
                            [5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5]] )
    mtry_params = np.array( [[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                             [6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6]] )
    h_wealth = 1
    p_wealth = 2
    m_wealth = 3
    tau_payroll = 0.25
    retire = 1
    analytical_mtrs = 1

    method = 'TPI_scalar'
    FOC1 = household.FOC_savings( r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H,
                                  (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, J, S, \
                                   analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, \
                                   m_wealth, tau_payroll, retire, method) )
    assert (np.allclose( FOC1, np.array( [-4.328325914762576e-14, -4.328325914762576e-14] ) ))

    method = 'SS'
    FOC2 = household.FOC_savings( r, w, b, b_splus1, b_splus2, n, BQ, factor, T_H,
                                  (e, sigma, beta, g_y, chi_b, theta, tau_bq, rho, lambdas, J, S, \
                                   analytical_mtrs, etr_params, mtry_params, h_wealth, p_wealth, \
                                   m_wealth, tau_payroll, retire, method) )
    assert (np.allclose( FOC2, np.array( [-4.01217930e-11, -4.01217930e-11] ) ))


def test_FOC_labor():
    r = np.array( [0.1] )
    w = np.array( [15] )
    b = np.array( [5] )
    b_splus1 = np.array( [4] )
    n = np.array( [1] )
    BQ = np.array( [10] )
    net_tax = np.array( [100] )
    e = np.array( [100] )
    lambdas = np.array( [0.3] )
    g_y = 0.1
    b_ellipse = 10
    upsilon = 20
    ltilde = 30
    chi_n = np.array( [5] )
    c = np.array( [1] )
    sigma = 5
    b_splus2 = np.array( [3] )
    factor = 100
    T_H = 1000
    beta = 0.5
    chi_b = np.array( [0.5] )
    theta = 0.3
    tau_bq = np.array( [1] )
    rho = np.array( [0.2] )
    J = 10
    S = 5
    etr_params = np.array( [[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
                            [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]] )
    mtrx_params = np.array( [[0.6, 0.6, 0, 6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6],
                             [0.6, 0.6, 0, 6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6, 0.6]] )
    h_wealth = 1
    p_wealth = 2
    m_wealth = 3
    tau_payroll = 0.25
    retire = 1
    analytical_mtrs = 1
    method = 'SS'

    FOC3 = household.FOC_labor( r, w, b, b_splus1, n, BQ, factor, T_H, (e, sigma, g_y, theta,
                                                                        b_ellipse, upsilon, chi_n, ltilde, tau_bq,
                                                                        lambdas, J, S,
                                                                        analytical_mtrs, etr_params, mtrx_params,
                                                                        h_wealth, p_wealth,
                                                                        m_wealth, tau_payroll, retire, method) )
    assert (np.allclose( FOC3, np.array( [3.06862288e-08, 3.06862288e-08] ) ))
