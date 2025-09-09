# -*- coding: utf-8 -*-
"""
Created on Tue Sep  2 12:29:39 2025

@author: cdepaor
"""
from scipy.stats import truncnorm
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# --- POW_MPR_unc ---
def POW_MPR_unc(X, degree=2, max_sigma=0.2, lower_factor=0.1):
    """
    Positive predictions with multiplicative log-normal noise for a polynomial model,
    truncated smoothly at ~3 sigma above the base prediction.
    """
    model = MPoly_ss_models[2, :]
    coefs = model[0:20]
    intercept = model[20]
    r2 = model[21]

    X = np.atleast_1d(X)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()[:len(coefs)]

    y_base = np.dot(coefs, X_poly) + intercept
    std_resid = residual_std_from_r2(r2, DB_ss["Power"])

    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    upper_factor = np.exp(3 * sigma_global)

    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)
    y_out = y_base * np.exp(z[0])

    return float(y_out)


# --- PRPL_MLR_unc ---
def PRPL_MLR_unc(X, max_sigma=0.2, lower_factor=0.1):
    model = linear_ss_models[1, :]
    coefs = model[0:5]
    intercept = model[5]
    r2 = model[6]

    X = np.atleast_1d(X)
    y_base = np.dot(coefs, X) + intercept
    std_resid = residual_std_from_r2(r2, DB_ss["Propulsion"])

    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    upper_factor = np.exp(3 * sigma_global)

    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)
    y_out = y_base * np.exp(z[0])

    return float(y_out)


# --- AVIO_MLR_unc ---
def AVIO_MLR_unc(X, max_sigma=0.2, lower_factor=0.1):
    model = linear_ss_models[3, :]
    coefs = model[0:5]
    intercept = model[5]
    r2 = model[6]

    X = np.atleast_1d(X)
    y_base = np.dot(coefs, X) + intercept
    std_resid = residual_std_from_r2(r2, DB_ss["Avionics"])

    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    upper_factor = np.exp(3 * sigma_global)

    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)
    y_out = y_base * np.exp(z[0])

    return float(y_out)
