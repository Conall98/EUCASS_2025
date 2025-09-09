# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:31:27 2025

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from IAC25_DBs import *
import time
start = time.perf_counter()
degree = 2
#%% Formatting Functions

def Isaji_converter(L1): #converts ORLA landers into Isaji Landers
    L = L_isaji(L1.name, L1.mp, L1.md, L1.mprop, L1.mt, L1.dv, L1.Isp, L1.STR+L1.THER, L1.PRPLSN, L1.POW, L1.AVIO, 1, L1.OTH)
    return L
    # print(L_isaji)
    
def DB_2_class(DB, i):
    mt = DB["md"][i] + DB["mprop"][i] + DB["mp"][i]
    lander_data = L(DB["Lander"][i], DB["mp"][i], DB["md"][i], DB["mprop"][i], mt, DB["dV"][i], DB["Isp"][i], 
                       DB["Structure"][i],DB["Propulsion"][i],DB["Power"][i],DB["Avionics"][i],DB["Thermal Protection"][i],DB["Other"][i])
    
    return lander_data  

#%% Function evaluation
def R_squared(features, model, target):
    RSOS = 0
    TSOS = 0
    y_hat = np.mean(target)
    for i in range(0, len(target)):
        RSOS = RSOS + (target[i] - model(features[i, :]))  
        TSOS = TSOS + (target[i] - y_hat)**2  
    return 1 - RSOS/TSOS

def R_squared_Pow(features, model, target):
    RSOS = 0
    TSOS = 0
    y_hat = np.mean(target)
    # prediction = func(features[i, :], *STR_model)
    for i in range(0, len(target)):
        RSOS = RSOS + (target[i] - func(features[i, :], *model))**2  
        # print("data - prediction", target[i], "-", func(features[i, :], *model)**2)
        TSOS = TSOS + (target[i] - y_hat)**2  
        # print("data - average", target[i], "-", y_hat)
        # print("RSOS", RSOS)
        # print("TSOS", TSOS)
    return 1 - RSOS/TSOS

    
#%% Subsystem Sizing Rules
def f1(mp):
    return 367.29*np.log(mp) - 904.17

def f1_unc(mp):
    return np.random.normal(144.27*(mp**0.3278), 0.0116*mp+63.38)
    # return np.random.normal(144.27*(mp**0.3278), 300)

def f2(mp, md, dv, Isp):# gives mprop with mp+md
    return 1.16*(mp+md)*(np.exp(dv/(Isp*9.81)) - 1)

def f3(mp, mprop):
    x = mp+mprop
    return 12.49*x**0.55
### STRUCTURE ####

def STR_MLR(X):
    str_model = linear_ss_models[0, :]
    coefs = np.array(str_model[0:5])
    intercept = str_model[5]
    return np.dot(coefs, X) + intercept

def STR_MLR_unc(X, max_sigma=0.2, lower_factor=0.1):
    """
    Positive predictions with multiplicative log-normal noise for a single input vector,
    truncated smoothly at ~3 sigma above the base prediction.
    """
    # Get model coefficients
    str_model = linear_ss_models[0, :]
    coefs = np.array(str_model[0:5])
    intercept = str_model[5]
    r2 = str_model[6]

    # Ensure X is 1D
    X = np.atleast_1d(X)

    # Base prediction
    y_base = X @ coefs + intercept

    # Residual standard deviation
    std_resid = residual_std_from_r2(r2, DB_ss["Structure"])

    # Global sigma in log-space, capped
    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    # Truncate multiplicative factor at 3 sigma above base
    upper_factor = np.exp(3 * sigma_global)

    # Standardized truncation bounds for truncnorm
    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    # Sample multiplicative factor from truncated log-normal
    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)

    # Apply multiplicative noise
    y_out = y_base * np.exp(z[0])

    return float(y_out)

def STR_MPR(X):
    str_model = MPoly_ss_models[0, :]
    coefs = str_model[0:20]
    intercept = str_model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def STR_MPowR(X):
    str_model = MPow_ss_models[0]
    return power_func(X, *str_model)

### PROPULSION ####
def PRPL_MLR(X):
    model = linear_ss_models[1, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept
#%%
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
#%%
def PRPL_MPR(X):
    model = MPoly_ss_models[1, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def PRPL_MPowR(X):
    model = MPow_ss_models[1]
    return power_func(X, *model)

def PRPL_Ramos(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape): #FT = Fuel type (class), T is Thrust WEIGHT Requirement, MR is O/F mixture ratio
    m_prpl, m_tanks, m_engines = tn.PRPL(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
    return m_prpl

### POWER ####
def POW_MLR(X):
    model = linear_ss_models[2, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept

def POW_MPR(X):
    model = MPoly_ss_models[2, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

#%%
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
#%%
def POW_MPowR(X):
    model = MPow_ss_models[2]
    return power_func(X, *model)

def POW_SSR_battery(peak_power, average_power, mission_duration): 
    p_req = peak_power #W
    e_req = average_power*mission_duration #Wh
    
    Li_ion_cell_specific_power = 96 #W/kg #astro-batt from Airbus
    Li_ion_cell_specific_energy = 1/170 #kg/Wh
    
    # print("p_req", p_req)
    # print("e_req", p_req)
    # print("p_req/fuel_cell_specific_power", p_req/fuel_cell_specific_power)
    # print("e_req*fuel_cell_specific_energy", e_req*fuel_cell_specific_energy)
    
    return max(p_req/Li_ion_cell_specific_power, e_req*Li_ion_cell_specific_energy)

### AVIO ####
def AVIO_MLR(X):
    model = linear_ss_models[3, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept
#%%
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

#%%
def AVIO_MPR(X):
    model = MPoly_ss_models[3, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def AVIO_MPowR(X):
    model = MPow_ss_models[3]
    return power_func(X, *model)

### THERMAL ####
def THER_MLR(X):
    model = linear_ss_models[4, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept

def THER_MPR(X):
    model = MPoly_ss_models[4, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept
#%%
def THER_MLR_unc(X, max_sigma=0.2, lower_factor=0.1):
    """
    Positive predictions with multiplicative log-normal noise for a single input vector,
    truncated smoothly at ~3 sigma above the base prediction.
    """
    # Get model coefficients
    model = linear_ss_models[4, :]
    coefs = model[0:5]
    intercept = model[5]
    r2 = model[6]  # coefficient of determination

    # Ensure X is a 1D vector
    X = np.atleast_1d(X)

    # Base prediction
    y_base = np.dot(coefs, X) + intercept

    # Residual standard deviation
    std_resid = residual_std_from_r2(r2, DB_ss["Thermal Protection"])  # adjust column as needed

    # Global sigma in log-space, capped
    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    # Truncate multiplicative factor at 3 sigma above base
    upper_factor = np.exp(3 * sigma_global)

    # Standardized truncation bounds for truncnorm
    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    # Sample multiplicative factor from truncated log-normal
    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)

    # Apply multiplicative noise
    y_out = y_base * np.exp(z[0])

    return float(y_out)
#%%
def THER_MPowR(X):
    model = MPow_ss_models[4]
    return power_func(X, *model)

def THER_phys(ap_req, D_m, heater, mt):
    #p_req is the max power requirement
    #ap_req is the average power requirement
    #D_m is the duration of the mission
    T_max = 293 #requirement
    T_min = 273 #requirement
    T_surface_night = 120 #K Lunar thermal baseline
    T_surface_day = 400 #K Lunar thermal baseline
    s = 5.67E-8 #sigma steffan boltzmann constant W/m2/K4
    e = 0.3 #emissivity of the spacecraft skin
    #effective spacecraft emissivity. normally would be the same as MLI properties
    #but I'm putting this artificially higher to capture the fact that
    #not all surfaces will be covered MLI
    alp = 0.01 #absorptivity of the spacecraft skin
    qI_day = s*T_surface_day**4 # IR from the moon on the surface by stefan-boltzmann law
    qI_night = s*T_surface_night**4 # IR from the moon on the surface  by stefan-boltzmann law
    Rho = np.pi/2 # on the surface of the moon
    D = 3.44*np.sqrt((mt/16194)) #rough major diameter of the spacecraft if it is a sphere
    #D is scaled around the apollo LM using mt
    Gs = 1360 #solar flux W/m2
    a = 0.07 #lunar albedo fraction
    Ka = 0.664+0.521*Rho - 0.203*Rho#factor accounting for the spherical nature of the moon
    
    P1 = (T_max**4)*(s*e)
    P2 = Gs*alp/4
    P3 = qI_day*e*(1-np.cos(Rho))/2
    P4 = Gs*a*alp*Ka*(1-np.cos(Rho))/2
    P5 = np.pi*D**2
    # print("P1", P1, "P2", P2, "P3", P3, "P4", P4, "P5", P5)
    
    Q_TCS_hot_case = (P1-P2-P3-P4)*P5 - ap_req #the heat needed to get out by radiator to maintain the Tmax
    
    P1 = (T_min**4)*(s*e)
    P2 = Gs*alp/4
    P3 = qI_night*e*(1-np.cos(Rho))/2
    P4 = Gs*a*alp*Ka*(1-np.cos(Rho))/2
    P5 = np.pi*D**2
    # print("P1", P1, "P2", P2, "P3", P3, "P4", P4, "P5", P5)
    
    Q_TCS_cold_case = (P1 - P3)*P5 - ap_req #the heat needed to generate by heater to maintain the Tmin 
    # Q_hot_case = ((T_max**4)*(s*e) - Gs*alp/4 - qI*e*(1-np.cos(Rho))/2 - Gs*a*alp*Ka*(1-np.cos(Rho))/2)*np.pi*D**2
    
    # print("Q_TCS_cold_case", Q_TCS_cold_case,"Q_TCS_hot_case", Q_TCS_hot_case)
    
    radiator_efficiency = 50 #W/kg
    if Q_TCS_hot_case < 0:
        radiator_mass = abs(Q_TCS_hot_case/radiator_efficiency)
    elif Q_TCS_hot_case >= 0:
        radiator_mass  = 0  
    
    electric_heater_efficiency = 10000 #W/kg 
    nuclear_heater_efficiency = 24 #W/kg
    if heater == "nuclear":
        heater_efficiency = nuclear_heater_efficiency
    else:
        heater_efficiency = electric_heater_efficiency
    
    if D_m > 14:
        if Q_TCS_cold_case > 0:
            heater_mass = abs(Q_TCS_cold_case/heater_efficiency)
        elif Q_TCS_cold_case <= 0:
            heater_mass  = 0 
    elif D_m <=14:
        heater_mass = 0
    
    # print("heater_mass", heater_mass, "radiator_mass", radiator_mass)
    
    # return heater_mass + radiator_mass
    MLI_covering = 0.65*np.pi*D**2 # the spacial density of MLI from Chatgpt
    other_mass = 0.02*mt #other stuff in the TCS
    TCS_mass = heater_mass + radiator_mass + MLI_covering + other_mass    
    return TCS_mass

### OTHER ####
def OTH_MLR(X):
    model = linear_ss_models[5, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept
#%%
def OTH_MLR_unc(X, max_sigma=0.2, lower_factor=0.1):
    """
    Positive predictions with multiplicative log-normal noise for a single input vector,
    truncated smoothly at ~3 sigma above the base prediction.
    """
    # Get model coefficients
    model = linear_ss_models[5, :]
    coefs = model[0:5]
    intercept = model[5]
    r2 = model[6]  # coefficient of determination

    # Ensure X is a 1D vector
    X = np.atleast_1d(X)

    # Base prediction
    y_base = np.dot(coefs, X) + intercept

    # Residual standard deviation
    std_resid = residual_std_from_r2(r2, DB_ss["Other"])  # adjust column as needed

    # Global sigma in log-space, capped
    eps = 1e-12
    sigma_global = min(std_resid / max(np.mean([y_base]), eps), max_sigma)
    mu_global = -0.5 * sigma_global**2

    # Truncate multiplicative factor at 3 sigma above base
    upper_factor = np.exp(3 * sigma_global)

    # Standardized truncation bounds for truncnorm
    a, b = (np.log(lower_factor) - mu_global) / sigma_global, \
           (np.log(upper_factor) - mu_global) / sigma_global

    # Sample multiplicative factor from truncated log-normal
    z = truncnorm.rvs(a, b, loc=mu_global, scale=sigma_global, size=1)

    # Apply multiplicative noise
    y_out = y_base * np.exp(z[0])

    return float(y_out)
#%%
def OTH_MPR(X):
    model = MPoly_ss_models[5, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def OTH_MPowR(X):
    model = MPow_ss_models[5]
    return power_func(X, *model)
#%% model combinations
def linear_estimations(X):
    a = STR_MLR(X)
    b = PRPL_MLR(X)
    c = POW_MLR(X)
    d = AVIO_MLR(X)
    e = THER_MLR(X)
    f = OTH_MLR(X)
    return a, b, c, d, e, f

def polynomial_estimations(X):
    a = STR_MPR(X)
    b = PRPL_MPR(X)
    c = POW_MPR(X)
    d = AVIO_MPR(X)
    e = THER_MPR(X)
    f = OTH_MPR(X)
    return a, b, c, d, e, f

def powerlaw_estimations(X):
    a = STR_MPowR(X)
    b = PRPL_MPowR(X)
    c = POW_MPowR(X)
    d = AVIO_MPowR(X)
    e = THER_MPowR(X)
    f = OTH_MPowR(X)
    return a, b, c, d, e, f

def mixed_estimation(X):
    ## PRPL extra params ###
    FT = F1 #N2O4-Aerozine   
    TWR = 1.73 #same as Apollo
    tank_material = M1
    n = 4 #number of engines
    Pressure = 20 #bars
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere" 
    P_tank_shape = "sphere"
    
    ## Power extra params ##
    peak_power = 2000
    average_power = 50
    mission_duration = 10
    heater = "non-nuclear"
    
    a = STR_MLR(X)
    # b = PRPL_Ramos(X[0], X[2], X[1], FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
    b = PRPL_MLR(X)
    # c = POW_SSR_battery(peak_power, average_power, mission_duration) +0.02*X[0]
    # c = POW_MLR(X)
    c = POW_MPR(X)
    d = AVIO_MLR(X)
    # e = THER_phys(average_power, mission_duration, heater, (X[0] + X[1] + X[2]))
    e = THER_MLR(X)
    f = OTH_MLR(X)
    
    return a, b, c, d, e, f

def mixed_physical_estimation(X):
    ## PRPL extra params ###
    FT = F1 #N2O4-Aerozine   
    TWR = 1.73 #same as Apollo
    tank_material = M1
    n = 4 #number of engines
    Pressure = 20 #bars
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere" 
    P_tank_shape = "sphere"
    
    ## Power extra params ##
    peak_power = 2000
    average_power = 50
    mission_duration = 10
    heater = "non-nuclear"
    
    a = STR_MLR(X)
    b = PRPL_Ramos(X[0], X[2], X[1], FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
    # b = PRPL_MLR(X)
    c = POW_SSR_battery(peak_power, average_power, mission_duration) +0.02*X[0]
    # c = POW_MLR(X)
    # c = POW_MPR(X)
    d = AVIO_MLR(X)
    e = THER_phys(average_power, mission_duration, heater, (X[0] + X[1] + X[2]))
    # e = THER_MLR(X)
    f = OTH_MLR(X)
    
    return a, b, c, d, e, f
    
def uncertain_estimation(X):
    a = STR_MLR_unc(X)
    # print("a", a)
    b = PRPL_MLR_unc(X)
    # print("b", b)
    c = POW_MPR_unc(X)
    # print("c", c)
    d = AVIO_MLR_unc(X)
    # print("d", d)
    e = THER_MLR_unc(X)
    # print("e", e)
    f = OTH_MLR_unc(X)
    # print("f", f)
    
    return a, b, c, d, e, f
#%% The sizing Algorithm

def IAC_sizing_algorithm(mp, dv, Isp, model, max_iter=50, tol=0.01):
    """
    Shamelessly uses a blend of eerything such that the global errors are minimised
    Choose it from main
    
    """
    
    DBss = DB_ss
    # md_0 = f1(mp)
    md_0 = f1_unc(mp)
    t1 = time.perf_counter()
    
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    

    t2 = time.perf_counter()
    
    i = 0
    # tol = 0.01  # now passed as argument
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    store = []
    t3 = time.perf_counter()
    # acf = 0.01

    # lander.initial_md_guess =  md_0 
    while er > tol:
        try:
            new_input = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
            
            # ⚠️ model() might overflow
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)  # treat overflow as exception
                a, b, c, d, e, f = model(new_input)
        
            lander.STR    = a
            lander.PRPLSN = b
            lander.POW    = c
            lander.AVIO   = d
            lander.THER   = e
            lander.OTH    = f
            # 
            md_i1 = sum([a, b, c, d, e, f])
    
            mprop_i1 = f2(mp, md_i1, dv, Isp)
        
            md_i.append(md_i1)
            mprop_i.append(mprop_i1)
            
            lander.md = (float(md_i[-1:][0])) #the [0] makes sure it is a float not a 1x1 array
            # print("md_i[-1:]", md_i[-1:][0])
            
        
            lander.mprop = float(mprop_i[-1:][0])
            
            lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
            
            er = abs(1 - md_i1/md_i[i])
            # acf = md_i[i]/(md_i1) #the last iter over this iter
            # print("er:                       ", er)
            # print("initial md guess:         ", md_0)
            # print("md_i from the lase iter:  ", md_i[i])
            # print("md from this iter:        ", md_i1)
            # print("iter:                     ", i)
            i=i+1
            # print(i)
            if i>max_iter:
                print("divergence, i=", i)
                return None   # ⬅️ signal divergence safely
        
        except (FloatingPointError, OverflowError, RuntimeWarning):
            print("Numerical overflow / invalid prediction.")
            return None   # ⬅️ bail out cleanly
    
    t4 = time.perf_counter()
    # print(f"Point 1 : {t1 - start:.8e} sec")
    # print(f"Point 2 : {t2 - t1:.8e} sec")
    # print(f"Point 3 : {t3 - t2:.8e} sec")
    # print(f"Point 4   {t4 - t3:.8e} sec")

    return lander

#%%
def IAC_sizing_algorithm_star(mp, dv, Isp, model, max_iter=50, tol=0.01):
    """
    Iterative sizing algorithm with uncertainty applied robustly:
    - First iteration uses uncertain predictions for each subsystem.
    - Subsequent iterations use the deterministic model for convergence.
    - After convergence, uncertainty is re-applied once to the final subsystems.

    Parameters
    ----------
    mp, dv, Isp : float
        Input parameters for the lander.
    model : callable
        Deterministic model returning a tuple/array of six predictions (a,b,c,d,e,f).
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance for md (based on deterministic predictions).
    """
    # Default uncertain functions (only used at i=0 and final step)
    model_unc_funcs = {
        'STR': STR_MLR_unc,
        'PRPL': PRPL_MLR_unc,
        'POW': POW_MPR_unc,
        'AVIO': AVIO_MLR_unc,
        'THER': THER_MLR_unc,
        'OTH': OTH_MLR_unc
    }

    # Initial guesses
    md_0 = f1_unc(mp)
    mprop_0 = f2(mp, md_0, dv, Isp)
    mt_0 = mp + md_0 + mprop_0

    md_i = [md_0]
    mprop_i = [mprop_0]

    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)

    er = 1
    i = 0

    while er > tol:
        try:
            new_input = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])

            if i == 0:
                # First iteration uses uncertainty
                a = model_unc_funcs['STR'](new_input)
                b = model_unc_funcs['PRPL'](new_input)
                c = model_unc_funcs['POW'](new_input)
                d = model_unc_funcs['AVIO'](new_input)
                e = model_unc_funcs['THER'](new_input)
                f = model_unc_funcs['OTH'](new_input)
            else:
                # Deterministic model drives convergence
                preds = model(new_input)
                if preds is None:
                    print("Deterministic model returned None!")
                    return None
                a, b, c, d, e, f = preds

            # Update subsystem values
            lander.STR, lander.PRPLSN, lander.POW = a, b, c
            lander.AVIO, lander.THER, lander.OTH = d, e, f

            md_i1 = sum([a, b, c, d, e, f])
            mprop_i1 = f2(mp, md_i1, dv, Isp)

            md_i.append(md_i1)
            mprop_i.append(mprop_i1)

            lander.md = float(md_i[-1])
            lander.mprop = float(mprop_i[-1])
            lander.mt = float(mp + lander.md + lander.mprop)

            # Convergence check uses deterministic prediction (skip noise influence)
            if i > 0:  
                er = abs(1 - md_i1 / md_i[i])
            i += 1

            if i > max_iter:
                print("Divergence detected, iteration =", i)
                return None

        except (FloatingPointError, OverflowError, RuntimeWarning):
            print("Numerical overflow / invalid prediction.")
            return None

    # ✅ After convergence, re-apply uncertainty once to final subsystem estimates
    new_input = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
    lander.STR    = model_unc_funcs['STR'](new_input)
    lander.PRPLSN = model_unc_funcs['PRPL'](new_input)
    lander.POW    = model_unc_funcs['POW'](new_input)
    lander.AVIO   = model_unc_funcs['AVIO'](new_input)
    lander.THER   = model_unc_funcs['THER'](new_input)
    lander.OTH    = model_unc_funcs['OTH'](new_input)

    # Update final md, mprop, mt with noisy subsystems
    lander.md = sum([lander.STR, lander.PRPLSN, lander.POW, lander.AVIO, lander.THER, lander.OTH])
    lander.mprop = f2(mp, lander.md, dv, Isp)
    lander.mt = mp + lander.md + lander.mprop

    return lander
#%% Evalusation of the function
def V_ers_2(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)
    ers_dic = {"mt:      ":errors[0],
                "md:      ":errors[1],
                "mprop:   ":errors[2],
                "mp:      ":errors[3],
                "STR:     ":errors[4],
                "PRPL:    ":errors[5],
                "AVIO:    ":errors[6],
                "POW:     ":errors[7],
                "THER:   ":errors[8],
                "OTH:     ":errors[9]}
    # print(ers_dic["mp:      "])

    return np.array(errors), ers_dic

import numpy as np

def EVAL(Algo_under_test, DB, models, verbose=True): 
    """
    Evaluate an algorithm over a database DB, skipping invalid landers.
    
    Parameters
    ----------
    Algo_under_test : function
        Function returning a lander object.
    DB : dict
        Database of input parameters.
    models : object
        Model input for the algorithm.
    verbose : bool
        If True, print skipped indices.
        
    Returns
    -------
    glob_ers_means : np.ndarray
        Median of the errors for each of the 10 parameters.
    prediction_values : np.ndarray
        Array of lander properties (valid predictions only).
    skipped_indices : list
        List of DB indices that were skipped due to errors.
    """
    
    glob_ers_list = []           # store valid errors
    prediction_values_list = []  # store valid predictions
    skipped_indices = []         # store indices of failed landers

    for i in range(len(DB["mt"])):
        data = DB_2_class(DB, i)
        mp = float(DB["mp"][i])
        dv = float(DB["dV"][i])
        Isp = float(DB["Isp"][i])

        try:
            pred = Algo_under_test(mp, dv, Isp, models)
            
            # Skip if pred is None
            if pred is None:
                skipped_indices.append(i)
                continue
            
            # Collect prediction values
            pred_row = np.array([
                pred.mt,
                float(np.array(pred.md)),
                float(np.array(pred.mprop)),
                pred.mp,
                float(pred.STR),
                float(pred.PRPLSN),
                float(pred.AVIO),
                float(pred.POW),
                float(pred.THER),
                float(pred.OTH)
            ])
            prediction_values_list.append(pred_row)

            # Compute errors
            ers_i, _ = V_ers_2(data, pred, "ith error")
            glob_ers_list.append(ers_i)

        except AttributeError:
            skipped_indices.append(i)
            continue

    # Convert lists to arrays
    if len(prediction_values_list) == 0:
        prediction_values = np.zeros((0,10))
        glob_ers = np.zeros((10,0))
    else:
        prediction_values = np.vstack(prediction_values_list)
        glob_ers = np.array(glob_ers_list).T  # shape (10, n_valid)

    # Compute median error for each parameter
    glob_ers_means = np.median(np.abs(glob_ers), axis=1) if glob_ers.size > 0 else np.zeros(10)

    if verbose and skipped_indices:
        print(f"Skipped {len(skipped_indices)} landers at indices: {skipped_indices}")

    return glob_ers_means, prediction_values, skipped_indices

#%%
def monte_carlo_IAC(n, mp_samples, dv_samples, isp_samples, Algorithm, model):
    results = []
    for i in range(n):
        lander = Algorithm(mp_samples[i],
                           dv_samples[i],
                           isp_samples[i],
                           model)
        results.append(lander)

    # Collect attributes (ignore "name")
    attrs = [a for a in results[0].__dict__.keys() if a != "name"]

    collected = {}
    for attr in attrs:
        values = np.array([getattr(r, attr) for r in results])
        # Map constructor keyword if needed
        key = {"Isp": "isp"}.get(attr, attr)
        collected[key] = values

    return L(results[0].name, **collected)
#%%
def monte_carlo_IAC_star(n, mp_samples, dv_samples, isp_samples, sizing_func, model):
    """
    Run Monte Carlo simulations using the iterative sizing algorithm.
    Returns a list of successful lander objects.
    """
    results = []

    for j in range(n):
        mp = mp_samples[j]
        dv = dv_samples[j]
        isp = isp_samples[j]

        lander = sizing_func(mp, dv, isp, model)
        if lander is not None:
            results.append(lander)
        else:
            print(f"Monte Carlo iteration {j} failed, skipping.")

    return results
#%%
def plot_lander_histograms(lander, bins=30):
    """
    Plot histograms of all array-valued properties of an L object
    as subplots in a single window. Ensures mp, dv, and Isp are first.
    Adds 5th, 50th, and 95th percentile markers (better for skewed/lognormal distributions).
    
    Parameters
    ----------
    lander : L
        The L object containing Monte Carlo arrays.
    bins : int
        Number of histogram bins.
    """
    # Collect all array-valued attributes, excluding name and initial_md_guess
    all_attrs = [
        a for a, v in lander.__dict__.items()
        if a not in ("name", "initial_md_guess") and isinstance(v, np.ndarray) and v.size > 1
    ]
    
    # Put mp, dv, Isp first if they exist
    priority = ["mp", "dv", "Isp"]
    attrs = [a for a in priority if a in all_attrs] + [a for a in all_attrs if a not in priority]
    
    n_attrs = len(attrs)
    if n_attrs == 0:
        print("No array-valued properties to plot.")
        return
    
    # Compute grid size (square-ish layout)
    ncols = m.ceil(m.sqrt(n_attrs))
    nrows = m.ceil(n_attrs / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()
    fig.suptitle("Mass Estimation Uncertainty", fontsize=16, fontweight="bold")

    for i, attr in enumerate(attrs):
        values = getattr(lander, attr)

        # Histogram
        axes[i].hist(values, bins=bins, alpha=0.7, color="#006658", edgecolor="black")

        # Compute percentiles
        p5, p50, p95 = np.percentile(values, [5, 50, 95])

        # Draw vertical lines
        axes[i].axvline(p5, color="red", linestyle="--", label="5th percentile")
        axes[i].axvline(p50, color="green", linestyle="--", label="50th percentile (median)")
        axes[i].axvline(p95, color="orange", linestyle="--", label="95th percentile")

        # Formatting
        axes[i].set_title(f"{attr} ({lander.name})")
        axes[i].set_xlabel(attr)
        axes[i].set_ylabel("Frequency")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].legend()

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()
#%%
def plot_lander_scurves_with_percentiles(lander):
    """
    Plot cumulative S-curves (CDFs) of all array-valued properties of an L object
    as subplots in a single window. Ensures mp, dv, and Isp are first.
    Marks 5th, 50th, and 95th percentiles on each curve.
    
    Parameters
    ----------
    lander : L
        The L object containing Monte Carlo arrays.
    """
    # Collect all array-valued attributes, excluding name and initial_md_guess
    all_attrs = [
        a for a, v in lander.__dict__.items()
        if a not in ("name", "initial_md_guess") and isinstance(v, np.ndarray) and v.size > 1
    ]
    
    # Put mp, dv, Isp first if they exist
    priority = ["mp", "dv", "Isp"]
    attrs = [a for a in priority if a in all_attrs] + [a for a in all_attrs if a not in priority]
    
    n_attrs = len(attrs)
    if n_attrs == 0:
        print("No array-valued properties to plot.")
        return
    
    # Compute grid size (square-ish layout)
    ncols = m.ceil(m.sqrt(n_attrs))
    nrows = m.ceil(n_attrs / ncols)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.atleast_1d(axes).ravel()
    
    for i, attr in enumerate(attrs):
        values = getattr(lander, attr)
        sorted_vals = np.sort(values)
        cdf = np.arange(1, len(sorted_vals)+1) / len(sorted_vals)
        
        axes[i].plot(sorted_vals, cdf, color='blue', label='CDF')
        
        # Compute percentiles
        p5, p50, p95 = np.percentile(values, [5,50,95])
        axes[i].axvline(p5, color='red', linestyle='--', label='5th percentile')
        axes[i].axvline(p50, color='green', linestyle='--', label='50th percentile (median)')
        axes[i].axvline(p95, color='orange', linestyle='--', label='95th percentile')
        
        axes[i].set_title(f"{attr} ({lander.name})")
        axes[i].set_xlabel(attr)
        axes[i].set_ylabel("Cumulative Probability")
        axes[i].grid(True, linestyle="--", alpha=0.5)
        axes[i].set_ylim([0,1])
        axes[i].legend()
    
    # Hide unused subplots if grid > needed
    for j in range(i+1, len(axes)):
        axes[j].axis("off")
    
    plt.tight_layout()
    plt.show()


#%% ####### DEBUGGING #############

def diagnose_gaps(lander_mc, mp_samples, dv_samples, isp_samples, prop="md", bins=50):
    """
    Diagnose gaps in a Monte Carlo output property.

    Parameters
    ----------
    lander_mc : L
        Lander object with Monte Carlo arrays (e.g., md, mt, mp, etc.).
    mp_samples, dv_samples, isp_samples : array-like
        Input samples used in the Monte Carlo simulation.
    prop : str
        The property to analyze (e.g., "md", "mt", "mp").
    bins : int
        Number of bins for histogram to detect gaps.
    """
    values = getattr(lander_mc, prop)
    
    # 1. Histogram to show gaps
    plt.figure(figsize=(8,4))
    counts, edges, _ = plt.hist(values, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(f"Histogram of {prop}")
    plt.xlabel(prop)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    # 2. Identify empty bins#
    empty_bins = np.where(counts == 0)[0]
    if len(empty_bins) == 0:
        print(f"No gaps detected in {prop}.")
    else:
        print(f"Gaps detected in {prop} at bin ranges:")
        for idx in empty_bins:
            print(f"{edges[idx]:.2f} to {edges[idx+1]:.2f}")
    
    # 3. Scatter plots of inputs vs prop
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    
    axes[0].scatter(mp_samples, values, s=5, alpha=0.5)
    axes[0].set_xlabel("mp")
    axes[0].set_ylabel(prop)
    axes[0].set_title(f"{prop} vs mp")
    
    axes[1].scatter(dv_samples, values, s=5, alpha=0.5)
    axes[1].set_xlabel("dv")
    axes[1].set_ylabel(prop)
    axes[1].set_title(f"{prop} vs dv")
    
    axes[2].scatter(isp_samples, values, s=5, alpha=0.5)
    axes[2].set_xlabel("Isp")
    axes[2].set_ylabel(prop)
    axes[2].set_title(f"{prop} vs Isp")
    
    plt.tight_layout()
    plt.show()
#%%

def diagnose_gaps_with_initial_guess_highlight(lander_mc, mp_samples, dv_samples, isp_samples, prop="md", bins=50):
    """
    Diagnose gaps in a Monte Carlo output property, highlight them in scatter plots,
    show initial_md_guess points, and mark guesses that fall in the gaps.
    
    Parameters
    ----------
    lander_mc : L
        Lander object with Monte Carlo arrays (e.g., md, mt, mp, etc.).
    mp_samples, dv_samples, isp_samples : array-like
        Input samples used in the Monte Carlo simulation.
    prop : str
        The property to analyze (e.g., "md", "mt", "mp").
    bins : int
        Number of bins for histogram to detect gaps.
    
    Returns
    -------
    gap_table : pd.DataFrame
        Table of input combinations closest to the gaps.
    """
    values = getattr(lander_mc, prop)
    
    # 1. Histogram
    plt.figure(figsize=(8,4))
    counts, edges, _ = plt.hist(values, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(f"Histogram of {prop}")
    plt.xlabel(prop)
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()
    
    # 2. Identify empty bins
    empty_bins = np.where(counts == 0)[0]
    if len(empty_bins) == 0:
        print(f"No gaps detected in {prop}.")
        return None
    else:
        print(f"Gaps detected in {prop} at bin ranges:")
        gap_ranges = []
        for idx in empty_bins:
            r = (edges[idx], edges[idx+1])
            gap_ranges.append(r)
            print(f"{r[0]:.2f} to {r[1]:.2f}")
    
    # 3. Scatter plots of inputs vs property
    fig, axes = plt.subplots(1,3, figsize=(18,5))
    inputs = [mp_samples, dv_samples, isp_samples]
    input_names = ["mp", "dv", "Isp"]
    
    gap_rows = []
    
    for ax, x, name in zip(axes, inputs, input_names):
        # Normal points
        ax.scatter(x, values, s=5, alpha=0.5, color='blue', label='Normal points')
        
        # Gap points in red
        for low, high in gap_ranges:
            mask = (values >= low) & (values <= high)
            if np.any(mask):
                ax.scatter(x[mask], values[mask], color="red", s=20, alpha=0.8, label="Gap points")
                gap_rows.extend(zip(x[mask], dv_samples[mask], isp_samples[mask], values[mask]))
        
        # initial_md_guess points
        if hasattr(lander_mc, "initial_md_guess") and lander_mc.initial_md_guess is not None:
            guess_values = lander_mc.initial_md_guess
            # Mask guesses inside gaps
            in_gap_mask = np.zeros_like(guess_values, dtype=bool)
            for low, high in gap_ranges:
                in_gap_mask |= (guess_values >= low) & (guess_values <= high)
            
            # Plot guesses outside gaps in yellow
            ax.scatter(x[~in_gap_mask], guess_values[~in_gap_mask], color="yellow", s=20, alpha=0.7, label="Initial md guess")
            # Plot guesses inside gaps in black stars
            if np.any(in_gap_mask):
                ax.scatter(x[in_gap_mask], guess_values[in_gap_mask], color="black", marker="*", s=50, label="Guess in gap")
        
        ax.set_xlabel(name)
        ax.set_ylabel(prop)
        ax.set_title(f"{prop} vs {name}")
        ax.grid(True, linestyle="--", alpha=0.5)
        
        # Avoid duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
    
    plt.tight_layout()
    plt.show()
    
    # 4. Create table of gap rows
    if gap_rows:
        gap_table = pd.DataFrame(gap_rows, columns=["mp", "dv", "Isp", prop])
        print(f"\nInput combinations near the gaps ({len(gap_rows)} rows):")
        display(gap_table)
        return gap_table
    else:
        print("No points fall exactly in the empty bins; gaps are truly unpopulated.")
        return None

#%%
def monte_test(func, X, n=10000, plot=True):
    """
    Monte Carlo simulation with multiplicative log-normal noise,
    plotting mean and ±1σ, ±2σ, ±3σ based on log-space statistics.
    Only values in [0, mean + 3σ_linear] are plotted.
    """
    results = []
    for _ in range(n):
        y_sim = func(X)
        results.append(y_sim)

    results = np.array(results)  # shape (n, n_samples)
    flattened = results.flatten()

    if plot:
        # Remove negative or zero values (log-space requires positive)
        flattened = flattened[flattened > 0]

        # Log-space statistics
        log_flat = np.log(flattened)
        log_mean = np.mean(log_flat)
        log_std = np.std(log_flat, ddof=1)

        # Transform back to linear-space
        mean_val = np.exp(log_mean)
        sigma_vals = [np.exp(log_mean + k*log_std) for k in [1,2,3]]
        sigma_vals_minus = [np.exp(log_mean - k*log_std) for k in [1,2,3]]

        # Define upper limit as mean + 3σ_linear (using linear-space σ)
        std_linear = np.std(flattened, ddof=1)
        upper_limit = mean_val + 5*std_linear
        flattened_plot = flattened[flattened <= upper_limit]

        # Histogram
        plt.hist(flattened_plot, bins=50, color="skyblue", edgecolor="black")

        # Vertical dashed line at mean
        plt.axvline(mean_val, color="red", linestyle="--", linewidth=2, label="Mean")

        # Vertical dashed lines for ±1σ, ±2σ, ±3σ from log-space
        alphas = [0.8, 0.5, 0.3]
        for plus, minus, alpha in zip(sigma_vals, sigma_vals_minus, alphas):
            if minus >= 0:
                plt.axvline(minus, color="grey", linestyle="--", alpha=alpha)
            if plus <= upper_limit:
                plt.axvline(plus, color="grey", linestyle="--", alpha=alpha)

        # X-axis limits
        plt.xlim(0, upper_limit)

        plt.xlabel("Predicted values with noise")
        plt.ylabel("Frequency")
        plt.title(f"Monte Carlo Simulation (n={n})")
        plt.legend()
        plt.show()

    # return results

#%%
def residual_std_from_r2(r2, y):
    """
    Compute the residual standard deviation from R^2 and data y.

    Parameters
    ----------
    r2 : float
        Coefficient of determination of the model.
    y : array-like
        Observed data (used to compute variance of y).

    Returns
    -------
    float
        Standard deviation of the residuals.
    """
    y = np.asarray(y)
    std_y = np.std(y, ddof=1)  # sample standard deviation
    std_resid = std_y * np.sqrt(1 - r2)
    return std_resid

#%%
def plot_cost_histogram(costs, bins=30, title="Acquisition Cost Distribution"):
    """
    Plot histogram of acquisition cost samples with 5th, 50th, and 95th percentiles.
    X-axis in millions of Euro.
    """
    costs = np.asarray(costs) / 1e6  # convert to millions

    # Compute percentiles
    p5, p50, p95 = np.percentile(costs, [5, 50, 95])

    plt.figure(figsize=(8, 6))
    plt.hist(costs, bins=bins, alpha=0.7, color="#006658", edgecolor="black")

    # Add percentile markers
    plt.axvline(p5, color="red", linestyle="--", label=f"5th: {p5:.1f} M€")
    plt.axvline(p50, color="green", linestyle="--", label=f"50th: {p50:.1f} M€")
    plt.axvline(p95, color="orange", linestyle="--", label=f"95th: {p95:.1f} M€")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Acquisition Cost [Million €]")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
#%%
def plot_cost_cdf(costs, title="Acquisition Cost S-curve"):
    """
    Plot cumulative distribution (CDF / S-curve) of acquisition cost samples,
    with 5th, 50th, and 95th percentiles marked.
    X-axis in millions of Euro.
    """
    costs = np.asarray(costs) / 1e6  # convert to millions
    sorted_vals = np.sort(costs)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

    # Percentiles
    p5, p50, p95 = np.percentile(costs, [5, 50, 95])

    # Plot CDF
    plt.figure(figsize=(8, 6))
    plt.plot(sorted_vals, cdf, color="blue", label="CDF")

    # Add percentile markers
    plt.axvline(p5, color="red", linestyle="--", label=f"5th: {p5:.1f} M€")
    plt.axvline(p50, color="green", linestyle="--", label=f"50th: {p50:.1f} M€")
    plt.axvline(p95, color="orange", linestyle="--", label=f"95th: {p95:.1f} M€")

    plt.title(title, fontsize=14, fontweight="bold")
    plt.xlabel("Acquisition Cost [Million €]")
    plt.ylabel("Cumulative Probability")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()