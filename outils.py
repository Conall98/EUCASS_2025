# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:59:35 2025

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from mulitple_regression import *
import numpy as np
from DBs import *
import tank_sizing_subroutine as tn
from scipy.optimize import curve_fit
# from scipy.optimize import least_squares
import logging
import matplotlib.pyplot as plt
from sklearn import linear_model
import math as m

from sklearn.preprocessing import PolynomialFeatures
#%%
######## Regression Modelling #########
def Linregger(X, y, i):
    # print("i", i)
    model = linear_model.LinearRegression(fit_intercept=False, positive=True)
    model.fit(X, y[:, i])
    R2 = model.score(X, y[:, i])
    coefs = model.coef_
    intercept =  model.intercept_
    preds = model.predict([X[0]])
    manual_pred = np.dot(coefs, X[0]) + model.intercept_
    if abs(preds[0] - manual_pred) > 0.5:
        logging.warning("Manual Prediction and model.predict() do not agree")
        
    
    # print("predicted y", preds)
    # print("manual_preds", manual_pred)
    # print("actual y", y[0, i])
    # print(preds == manual_pred)
    return coefs, intercept, R2

def modeler(ssDB):
    X_data = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
    y_data = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
    ss_models = np.zeros([6, 7]) #six row(predictions), five coefficients + one intercept for each pred
    Rs = []
    for i in range(0, 6):
        coefs, inter, r2 = Linregger(X_data, y_data, i)
        ss_models[i, 0:5] = coefs
        ss_models[i, 5:6] = inter
        ss_models[i, 6] = r2
        Rs.append(r2)
    
    return ss_models

ss_models = modeler(DB_ss)



#%%
def flexible_modeler(X_data, target, i):
    features = len(X_data[0,:])
    # print("features", features)
    ss_model = np.zeros([1, features + 2]) #six row(predictions), five coefficients + one intercept for each pred
    Rs = []
    coefs, inter, r2 = Linregger(X_data, target, 0)
    ss_model[i, 0:features] = coefs
    ss_model[i, features:features+1] = inter
    ss_model[i, features + 1] = r2
    Rs.append(r2)
    
    return ss_model

#%%
def power_regger(X, y):
    def func(a, x, b, c):
        return a*x**(b)+c
    for i in range(0,6):
        xdata = X[:,i]
        ydata = y
        popt, pcov = curve_fit(func, xdata, ydata)

#%%        
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


#%%
def func(X, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5):
    x1, x2, x3, x4, x5 = X.T  # transpose to unpack columns
    # print("X", X.T)
    # print("coefs", a1, a2, a3, a4, a5, b1, b2, b3, b4, b5)
    A = a1*x1**b1
    B = a2*x2**b2
    C = a3*x3**b3
    D = a4*x4**b4 
    E = a5*x5**b5
    
    return A + B + C + D + E

# def func(X, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5):
#     x1, x2, x3, x4, x5 = X.T  # transpose to unpack columns
#     return (a1*x1**b1)*(a2*x2**b2)*(a3*x3**b3)*(a4*x4**b4)*(a5*x5**b5)
#%%
def multiple_power_regression(X, y):  # for predicting 6 targets using 5 features
    # xdata = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).T
    # ydata = np.array(DBss["Structure"])  # single target (1D)
    xdata = X
    ydata = y

    # Fit the function to just the "Structure" target
    initial_guess = [0.5, 0.5, 0.5, 0.5, 0.5,   # a1 to a5
                  0.5, 0.5, 0.5, 0.5, 0.5]   # b1 to b5
    bounds = (
    [0]*5 + [-np.inf]*5,  # a1–a5 >= 0, b1–b5 unrestricted
    [np.inf]*10)
    
    popt, pcov = curve_fit(func, X, y, p0=initial_guess, full_output = False, bounds=bounds, maxfev=1000000)
    # print(pcov)
    return popt
    


#%%
def MPR(X, y, degree=2):
    # Ensure X and y are both 2D
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Polynomial feature transformation
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    # Set up regression model
    model = linear_model.LinearRegression(fit_intercept=False, positive=False)
    model.fit(X_poly, y)  # Fits all outputs at once

    # Predict the first sample manually to compare
    x0 = X_poly[0].reshape(1, -1)         # (1, n_features)
    preds = model.predict(x0)             # shape (1, n_outputs)
    manual_pred = np.dot(X_poly[0], model.coef_.T) + model.intercept_

    for j in range(preds.shape[1]):
        if abs(preds[0, j] - manual_pred[j]) > 0.5:
            logging.warning(f"Disagreement on prediction in output {j}")

    # Prepare summary array: (n_outputs x [n_coefs + intercept + R2])
    n_outputs = y.shape[1]
    ss_models = np.zeros((n_outputs, X_poly.shape[1] + 2))

    for i in range(n_outputs):
        model.fit(X_poly, y[:, i])
        ss_models[i, :-2] = model.coef_
        ss_models[i, -2] = model.intercept_
        ss_models[i, -1] = model.score(X_poly, y[:, i])

    return ss_models, X_poly

#%%
def m_predict(model, features):
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(features).transpose()
    # print("model shape", np.shape(model[:, :-2]))
    # print("features shape", np.shape(X_poly))
    # print("interepts shape", np.shape(model[:, -2]))
    # for i in range(len(model[:, 0])): 
    prediction = np.dot(model[:, :-2], X_poly) #+ model[:, -2]
    # print("prediction shape", np.shape(prediction))
    return prediction 

#%% initial functions
######## The Major Mass Property functions ########
def f1(mp):
    return 367.29*np.log(mp) - 904.17

def f2(mp, md, dv, Isp):# gives mprop with mp+md
    return 1.16*(mp+md)*(np.exp(dv/(Isp*9.81)) - 1)

def f3(mp, mprop):
    x = mp+mprop
    return 12.49*x**0.55

#%% Subsyste; functions
######## The Subsystem Sizing Routines #########

def STR_MPowR(X_data, y_data):
    STR_model = multiple_power_regression(X_data, y_data[:, 0])
    # print("X_data", X_data, "y_data", y_data[:, 0])
    return STR_model

def PRPL_MPowR(X_data, y_data):
    PRPL_model = multiple_power_regression(X_data, y_data[:, 1])
    return PRPL_model

def POW_MPowR(X_data, y_data):
    POW_model = multiple_power_regression(X_data, y_data[:, 2])
    return POW_model

def AVIO_MPowR(X_data, y_data):
    AVIO_model = multiple_power_regression(X_data, y_data[:, 3])
    return AVIO_model

def THER_MPowR(X_data, y_data):
    THER_model = multiple_power_regression(X_data, y_data[:, 4])
    return THER_model

def OTH_MPowR(X_data, y_data):
    OTH_model = multiple_power_regression(X_data, y_data[:, 5])
    return OTH_model

def MPowR_initialiser(DBss):
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    return STR_MPowR(X_data, y_data), PRPL_MPowR(X_data, y_data), POW_MPowR(X_data, y_data), AVIO_MPowR(X_data, y_data), THER_MPowR(X_data, y_data), OTH_MPowR(X_data, y_data)

# MPowR_model = MPowR_initialiser(DB_ss)

#%% linear and multiple linear sizing functions ##########"
def AVIO(md, mp):
    return (md+mp)*0.0156

def AVIO2(X):
    avio_model = ss_models[3, :]
    coefs = avio_model[0:5]
    intercept = avio_model[5]
    return np.dot(coefs, X) + intercept

def AVIO_isaji(Dsm, Ncrw, md, mpower): #DSM IS MISSION DURATION
    return ((Dsm*Ncrw)**1.426)*((md/1000)**(-2.141))*mpower**0.9955 - 107.8

def STR(md):
    return (md)*0.234

def STR_mp(mp):
    return mp*0.0747

def STR2(X):
    str_model = ss_models[0, :]
    coefs = str_model[0:5]
    intercept = str_model[5]
    return np.dot(coefs, X) + intercept

def STRTPS(md, mp): #isaji structure an thermal estimation (total)
    return 1.325*(md/1000)**2.863 + (5.651E-5)*((mp+md)/1000)**5.269 +1390

def POW(md):
    return md*0.0603

def POW2(X):
    pow_model = ss_models[2, :]
    coefs = pow_model[0:5]
    intercept = pow_model[5]
    return np.dot(coefs, X) + intercept

def POW_Isaji(Dsm, Ncrw, md):
    return (Dsm**1.784)*(Ncrw**(-0.2694))*((md/1000)**1.384) + 636

# def POW_SSR_battery_cells(peak_power, average_power, mission_duration):
#     v_req = 28 #volts
#     p_req = peak_power #Watts
# #    print("p_req is of type:", type(p_req), p_req)
#     ##############Battery##################
    
#     energy_req = average_power*mission_duration #Wh
    
#     bcell_mass = 1.13 #kg
#     bcell_volume = 0.551 #U
#     bcell_voltage = 4 #V
#     bcell_ampage = 45 #Ah
#     discharge_limit = 0.8
    
#     bc_series = m.ceil(v_req/bcell_voltage)
#     bc_parallel = m.ceil((p_req/v_req)/bcell_ampage)
#     charge_req = (energy_req/v_req)/discharge_limit #Ah
    
#     total_cells = bc_parallel*bc_series
#     battery_mass = total_cells*bcell_mass
# #    print(battery_mass)
# #    print(charge_req)
#     battery_volume = total_cells*bcell_volume    


#     return np.round(battery_mass)

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
    
def POW_SSR_fuelcell(peak_power, average_power, mission_duration):
    p_req = peak_power #W
    e_req = average_power*mission_duration #Wh
    
    fuel_cell_specific_power = 101 #W/kg STS fuel cell chars
    fuel_cell_specific_energy = 0.002 #kg/Wh
    
    # print("p_req", p_req)
    # print("e_req", p_req)
    # print("p_req/fuel_cell_specific_power", p_req/fuel_cell_specific_power)
    # print("e_req*fuel_cell_specific_energy", e_req*fuel_cell_specific_energy)
    
    return p_req/fuel_cell_specific_power, e_req*fuel_cell_specific_energy

def THER(mt_0):
    return mt_0*0.0139

def THER2(X):
    ther_model = ss_models[4, :]
    coefs = ther_model[0:5]
    intercept = ther_model[5]
    return np.dot(coefs, X) + intercept


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
    

def ECLSS(Dsm, Ncrw, md):
    return 2.258*Dsm*Ncrw*((md/1000)**1.052)+544.7

def OTH_Isaji(C, md):# sets a fraction of the other mass
    return C*md

def OTH(md):
    return md*0.0316

def OTH2(X):
    oth_model = ss_models[5, :]
    coefs = oth_model[0:5]
    intercept = oth_model[5]
    return np.dot(coefs, X) + intercept


def PROP(mprop):
    return mprop*0.0887

def PROP2(X):
    prpl_model = ss_models[1, :]
    coefs = prpl_model[0:5]
    intercept = prpl_model[5]
    return np.dot(coefs, X) + intercept

def PRPL_Ramos(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape): #FT = Fuel type (class), T is Thrust WEIGHT Requirement, MR is O/F mixture ratio
    m_prpl, m_tanks, m_engines = tn.PRPL(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
    return m_prpl

def PRPL_Isaji_cryo(md, mp): #for cryogenic propellant
    m_inert = md + mp
    m_prpln = 2.702*((m_inert)/(1000))**2.785 - 0.01813*((m_inert)/(1000))**4.348
    return m_prpln

def PRPL_Isaji_storable(md, mp, FT, Isp): #direct Isaji estimation for storable propellants
    m_inert = md + mp
    
    rho = ((FT.rho_fuel) + (FT.rho_lox)*FT.MR)/FT.MR
    # Isp = FT.Isp
    # print("m_inert             ", m_inert)
    # print("m_inert**1.811      ", m_inert**1.811)
    # print("rho**(-0.4262)      ", rho**(-0.4262))
    # print("Isp**-1.201         ", Isp**-1.201)
    # # print("m_prpln             ", m_prpln)
    m_prpln = (m_inert**1.811)*(rho**(-0.4262))*(Isp**-1.201)+245.3
    
    
    return m_prpln

#%%

########## System-level sizing routines ############


def routine_Ramos_N2O4(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine
    TWR = 1.72 # same as Apollo
    tank_material = M1 #titanium
    n = 4
    Pressure = 20 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    a = lander.STR = STR_mp(mp)
    c = lander.POW = POW(md_i[i])
    d = lander.AVIO = AVIO(md_i[i], mp)
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(md_i[i])
    
    while er > tol:
        # print("mprop", mprop_i[-1:])
        # print("mp", mp)
        b = lander.PRPLSN = PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        lander.md = md_i[-1:]
        
        lander.mprop = mprop_i[-1:]
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_Ramos_N2O4_iter(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine
    TWR = 1.72 # same as Apollo
    tank_material = M1 #titanium
    n = 4
    Pressure = 20 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        a = lander.STR = STR_mp(mp)
        c = lander.POW = POW(md_i[i])
        d = lander.AVIO = AVIO(md_i[i], mp)
        e = lander.THER = THER(lander.mt)
        f = lander.OTH = OTH(md_i[i])
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        lander.md = md_i[-1:]
        
        lander.mprop = mprop_i[-1:]
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_Isaji_N2O4(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    a = lander.STR = STR_mp(mp)
    c = lander.POW = POW(md_i[i])
    d = lander.AVIO = AVIO(md_i[i], mp)
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(md_i[i])
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        
        lander.md = md_i[-1:]
        
        lander.mprop = mprop_i[-1:]
        
        # lander.mt = lander.md + lander.mprop + lander.mp
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        # print("here: ", lander.mt)
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_Isaji_N2O4_MLR(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    # X = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp]).reshape(1,-1)
    X = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
    a = lander.STR = STR2(X)
    c = lander.POW = POW2(X)
    d = lander.AVIO = AVIO2(X)
    e = lander.THER = THER2(X)
    f = lander.OTH = OTH2(X)
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_stat_MLR_iter(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    while er > tol:
        # X = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp]).reshape(1,-1)
        X = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
        a = lander.STR = STR2(X)
        b = lander.PRPLSN = PROP2(X)
        c = lander.POW = POW2(X)
        d = lander.AVIO = AVIO2(X)
        e = lander.THER = THER2(X)
        f = lander.OTH = OTH2(X)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        
        lander.md = float(md_i[-1:][0])

        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_all_linear(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = md_0
    mprop_i = mprop_0
    
    FT = F1 #N2O4-Aerozine    
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    
    b = lander.PRPLSN = PROP(lander.md)
    
    
    a = lander.STR = STR_mp(mp)
    c = lander.POW = POW(md_i)
    d = lander.AVIO = AVIO(md_i, mp)
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(md_i)
    
    md_i1 = sum([a, b, c, d, e, f])
    
    correction_scale = md_i1/md_0
    # print(correction_scale)
    
    # lander.STR = a*(md_i1/md_0)
    # lander.PRPLSN = b*(md_i1/md_0)
    # lander.POW = c*(md_i1/md_0)
    # lander.AVIO = d*(md_i1/md_0)
    # lander.THER = e*(md_i1/md_0)
    # lander.OTH = f*(md_i1/md_0)
    
    # print("iterations: ", i)
    return lander
#%%
def routine_Ramos_Cryo(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F2 #loxlh2
    Isp = FT.Isp
    TWR = 1.72 # same as Apollo
    tank_material = M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    while er > tol:
        

        b = lander.PRPLSN = PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        
        a = lander.STR = STR_mp(mp)
        c = lander.POW = POW(md_i[i])
        d = lander.AVIO = AVIO(md_i[i], mp)
        e = lander.THER = THER(mt_0)
        f = lander.OTH = OTH(md_i[i])
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_stat_MLR_noloop(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = md_0
    mprop_i = mprop_0
    
    FT = F1 #N2O4-Aerozine    
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X = np.array([lander.md, mp, lander.mprop, Isp, dv])
    a = lander.STR = STR2(X)
    b = lander.PRPLSN = PROP2(X)
    c = lander.POW = POW2(X)
    d = lander.AVIO = AVIO2(X)
    e = lander.THER = THER2(X)
    f = lander.OTH = OTH2(X)
    md_i1 = sum([a, b, c, d, e, f])
    
    correction_scale = md_i1/md_0
    
    lander.STR = a/(md_i1/md_0)
    lander.PRPLSN = b/(md_i1/md_0)
    lander.POW = c/(md_i1/md_0)
    lander.AVIO = d/(md_i1/md_0)
    lander.THER = e/(md_i1/md_0)
    lander.OTH = f/(md_i1/md_0)
    
    # print("iterations: ", i)
    return lander
#%%
def routine_I_N2O4_MLR_iter(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    acf = 0.8
    while er > tol:
        b = lander.PRPLSN = PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        X = np.array([lander.md, mp, lander.mprop, Isp, dv])
        # print("X: ", X)
        a = lander.STR = STR2(X)
        c = lander.POW = POW2(X)
        d = lander.AVIO = AVIO2(X)
        e = lander.THER = THER2(X)
        f = lander.OTH = OTH2(X)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        
        
        lander.md = float(md_i[-1:][0])*acf

        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        # print("acf:                      ", acf)
        # print("er:                       ", er)
        # print("initial md guess:         ", md_0)
        # print("md_i from the lase iter:  ", md_i[i])
        # print("md from this iter:        ", md_i1)
        # print("PRPL:                      ", b)
        # # store.append(a)
        # print("iter:                     ", i)
        i = i+1
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def Isaji_imitator(mp, dv, Isp, Dsm, Ncrw, C_other):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L_isaji("Isaji lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        X = np.array([lander.md, mp, lander.mprop, Isp, dv])
        # print("X: ", X)
        a = lander.STRTPS = STRTPS(md_i[i], mp)
        c = lander.POW = POW_Isaji(Dsm, Ncrw, md_i[i])
        d = lander.AVIO = AVIO_isaji(Dsm, Ncrw, md_i[i], c)
        e = lander.ECLSS = ECLSS(Dsm, Ncrw, md_i[i])
        f = lander.OTH = OTH_Isaji(C_other, md_i[i])
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = abs(1 - md_i1/md_i[i])
        i = i+1
        
        lander.md = float(md_i[-1:][0])

        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def Progessive_MLR_Sizing(mp, dv, Isp, DBss):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    ss_models = modeler(DBss) #initialise the models
    # print("subsystem models", ss_models)
    argmax_r2 = np.argmax(ss_models[:, -1:]) #extract model with the max R
    ss_list = ["Structure", "Propulsion", "Power", "Avionics", "Thermal Protection", "Other"]
    sorted_indices = np.argsort(-ss_models[:, -1])
    
    
    # model1.fit(X_data, DBss[ss_list[i]])  
    # A = ss_list[sorted_indices[i]]
    # A_star = np.array(DBss[A]).reshape(len(DBss[A]), 1)
    # X_data = np.append(X_data, A_star, 1)
    # print("X_data extended", X_data)
    
    #### the full process for one model ######
    X_star = np.array([lander.md, mp, lander.mprop, Isp, dv]).reshape(-1, 1).transpose() # the input features
    X_data_star = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    preds = []
    for i in range(0, len(ss_list)):
        # print("length of ss_list", len(ss_list))
        model1 = linear_model.LinearRegression(fit_intercept=False, positive=True) #initialise the model
        y_data_i = DBss[ss_list[sorted_indices[i]]] #initialise the target data
        # print("subsystem: ", ss_list[sorted_indices[i]])
        model1.fit(X_data_star, y_data_i) #fit the feature and the first target
        # print("regression score", model1.score(X_data_star, y_data_i))
        # print("features being fitted: ", len(X_data_star[0, :]))
        # print("target being fitted: ", np.shape(y_data_i))
        # print(model1)
        y1 = model1.predict(X_star) #prediction of the first target
        # print("X_star:", X_star)
        A = ss_list[sorted_indices[i]]
        A_star = np.array(DBss[A]).reshape(len(DBss[A]), 1)
        # print("target being added to features: ", A_star)
        X_data_star = np.append(X_data_star, A_star, 1) #appending the 1st target data to the feature data
        # print("new features: ", X_data_star)
        # print("X_data_star:", X_data_star)
        X_star = np.append(X_star, y1).reshape(-1, 1).transpose()#adding the predicted target to the input feautures
        # print("target", y1)
        preds.append(y1)
    
    b = lander.PRPLSN = preds[0]
    a = lander.STR = preds[5]
    c = lander.POW = preds[3]
    d = lander.AVIO = preds[4]
    e = lander.THER = preds[2]
    f = lander.OTH = preds[1]
    md_i1 = sum([a, b, c, d, e, f])
    mprop_i1 = f2(mp, md_i1, dv, Isp)
    # print(i)
    md_i.append(md_i1)
    mprop_i.append(mprop_i1)
    
    lander.md = float(md_i[-1:][0])

    lander.mprop = float(mprop_i[-1:][0])
    
    lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
    
# print("iterations: ", i)
    return lander
    

#%%
def routine_MPR_noloop(mp, dv, Isp):
    DBss = DB_ss
    md_0 = f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    
    
    
    models, X_poly = MPR(X_data, y_data, 2)
    input_array = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
    poly = PolynomialFeatures(degree=2, include_bias=False)
    input_array_transformed = poly.fit_transform(input_array.reshape(1,-1))
    
    preds = []
    # print("here", np.shape(models))
    # print("here", np.shape(input_array_transformed))
    for i in range (0, 6):
        preds.append(np.dot(models[i, :-2], input_array_transformed.reshape(-1,1))) #+ model[:, -2]
        
    b = lander.STR = preds[0]
    a = lander.PRPLSN = preds[1]
    c = lander.POW = preds[2]
    d = lander.AVIO = preds[3]
    e = lander.THER = preds[4]
    f = lander.OTH = preds[5]
    
    md_i1 = sum([a, b, c, d, e, f])
    mprop_i1 = f2(mp, md_i1, dv, Isp)
    # print(i)
    md_i.append(md_i1)
    mprop_i.append(mprop_i1)
    
    lander.md = float(md_i[-1:][0])
    
    lander.mprop = float(mprop_i[-1:][0])
    
    lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
    
    return lander
    
#%%
def routine_multiple_power_regression(mp, dv, Isp, MPowR_model):
    DBss = DB_ss
    md_0 = f1(mp)
    
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    store = []
    
    acf = 0.01
    STR_model = MPowR_model[0]
    # print("STR_model", STR_model)
    PRPL_model = MPowR_model[1]
    # print("PRPL_model", PRPL_model)
    POW_model = MPowR_model[2]
    # # print("POW_model", PRPL_model)
    AVIO_model = MPowR_model[3]
    # # print("AVIO_model", AVIO_model)
    THER_model = MPowR_model[4]
    # # print("THER_model", THER_model)
    OTH_model = MPowR_model[5]
    # print("OTH_model", OTH_model)
    while er > tol:

        
        new_input = np.array([[lander.mt, lander.mp, lander.mprop, lander.dv, lander.Isp]])
        a = lander.STR = func(new_input, *STR_model)
        # b=c=d=e=f=1
        b = lander.PRPLSN =  func(new_input, *PRPL_model)
        c = lander.POW = func(new_input, *POW_model)
        d = lander.AVIO = func(new_input, *AVIO_model)
        e = lander.THER = func(new_input, *THER_model)
        f = lander.OTH = func(new_input, *OTH_model)
        
        # print("STR", a,"PRPL",  b,"POW",  c,"AVIO",  d,"THER",  e,"OTH",  f)
        # print("STR", a)
        # 
        md_i1 = sum([a, b, c, d, e, f])*acf
        # print("uncorrected md: ", sum([a, b, c, d, e, f]))
        # print("corrected md  : ", sum([a, b, c, d, e, f])*acf)
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        
        lander.md = (float(md_i[-1:][0]))
    
        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        er = abs(1 - md_i1/md_i[i])
        # acf = md_i[i]/(md_i1) #the last iter over this iter
        
        # print("acf:                      ", acf)
        print("er:                       ", er)
        print("initial md guess:         ", md_0)
        print("md_i from the lase iter:  ", md_i[i])
        print("md from this iter:        ", md_i1)
        # # print("STR:                      ", a)
        # # store.append(a)
        print("iter:                     ", i)
        i=i+1
        if i>50:
            print("divergence, i=", i)
            break    
    
    return lander
#%%

######### Validation Functions ##########
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
# #%%
# LM_pred = EUC_AZ.routine_all_linear(5295, 2265, 311)
# LM_test = L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
# V_ers_2(LM_test, LM_pred, "all")

def V_ers_2_isaji(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STRTPS, data.PRPLSN, data.AVIO, data.POW, data.ECLSS, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STRTPS, pred.PRPLSN, pred.AVIO, pred.POW, pred.ECLSS, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)
        
    ers_dic = {"mt:      ":errors[0],
                "md:      ":errors[1],
                "mprop:   ":errors[2],
                "mp:      ":errors[3],
                "STRTPS:  ":errors[4],
                "PRPL:    ":errors[5],
                "AVIO:    ":errors[6],
                "POW:     ":errors[7],
                "ECLSS:   ":errors[8],
                "OTH:     ":errors[9]}

    return np.array(errors), ers_dic

#%%
def V_ss_plotter(data, pred, testname): #two lander objects

    dat = [data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH]
    prd = [pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]
    
    dat2 = [data.mt, data.md, data.mprop]
    prd2 = [pred.mt, pred.md, pred.mprop]

# Set category labels at adjusted positions
    x_positions = [0, 1, 
                   3, 4, 
                   6, 7, 
                   9, 10, 
                   12, 13, 
                   15, 16]
    j = 0
    k = 0
    l = 0
    plt.figure()
    while k < len(x_positions):
        # print(x_positions[k])
        plt.bar(x_positions[k], dat[j], width=1, align='center', color = "blue", label = "data")
        k=k+1
        # print(x_positions[k])
        plt.bar(x_positions[k], prd[j], width=1, align='center', color = "orange", label = "prediction")
        if j == 0:
            plt.legend()
        k=k+1
        j = j + 1


    x_labels = ["str", "prpl", "avio", "pow", "therm", "oth"]
    plt.xticks([0.5, 3.5, 6.5, 9.5, 12.5, 15.5], x_labels)
    
    plt.xlabel('Subsystems')
    plt.ylabel('Mass [kg]')
    plt.title('Sizing algorithm validation: {}'.format(testname))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
# Set category labels at adjusted positions
    x_positions2 = [0, 1, 
                   3, 4, 
                   6, 7]
    j = 0
    k = 0
    l = 0
    plt.figure()
    while k < len(x_positions2):
        # print(x_positions[k])
        plt.bar(x_positions2[k], dat2[j], width=1, align='center', color = "blue", label = "data")
        k=k+1
        # print(x_positions[k])
        plt.bar(x_positions2[k], prd2[j], width=1, align='center', color = "orange", label = "prediction")
        if j == 0:
            plt.legend()
        k=k+1
        j = j + 1


    x_labels = ["$m_t$", "$m_d$", "$m_{prop}$", ]
    plt.xticks([0.5, 3.5, 6.5], x_labels)
    
    plt.xlabel('Major mass properties')
    plt.ylabel('Mass [kg]')
    plt.title('Sizing algorithm validation: {}'.format(testname))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()    
    
#%%
def global_errors(DB):
    glob_ers = np.zeros([10,10])
    glob_ers_means = np.zeros(10)
    for i in range(0, len(DB["mt"])):
        data = DB_2_class(DB_ss, i)
        mp = float(DB_ss["mp"][i])
        dv = float(DB_ss["dV"][i])
        Isp = float(DB_ss["Isp"][i]) 
        
        ############### for N2O4 #################
        pred = routine_Ramos_N2O4(mp, dv, Isp) #making the prediction
        # pred = routine_Ramos_N2O4_iter(mp, dv, Isp) #making the prediction
        # pred = routine_all_linear(mp, dv, Isp) #making the prediction
        # pred = routine_Isaji_N2O4(mp, dv, Isp) #making the prediction
        # pred = routine_Isaji_N2O4_MLR(mp, dv, Isp) #making the prediction
        # pred = routine_stat_MLR_iter(mp, dv, Isp) #making the prediction
        # pred = routine_I_N2O4_MLR_iter(mp, dv, Isp) #making the prediction
        # pred = routine_stat_MLR_noloop(mp, dv, Isp) #making the prediction
        # pred = routine_stat_MLR_noloop(mp, dv, Isp) #making the prediction
        # pred = routine_MPR_noloop(mp, dv, Isp, DB) #making the prediction
        
        ############# for LOX/LH2 ################
        
        # pred = EUC_LH2.routine_Ramos_Cryo(mp, dV, Isp)
        # pred = EUC_LH2.routine_Isaji_cryo(mp, dV, Isp)
        # pred = EUC_LH2.routine_Ramos_Cryo_lessloop(mp, dV, Isp)
        
        ########### for lch4 ###################
        # pred = EUC_LH2.routine_Ramos_Cryo_CH4(mp, dV, Isp)
        
        
        # print(pred.test())
        ers_i, dump = V_ers_2(data, pred, "ith error")
        # print(ers_i[1])
        # print("mp_ers: ",data.mp, pred.mp)
        glob_ers[i,:] = ers_i
    
    for i in range(0, 10):
        # glob_ers_means[i] = np.mean(glob_ers[:,i])
        # glob_ers_means[i] = np.median(glob_ers[:,i])
        # glob_ers_means[i] = np.median(np.sqrt(glob_ers[:,i]**2))
        glob_ers_means[i] = np.median(abs(glob_ers[:,i]))
        # print(glob_ers[:,i])
        
    return glob_ers_means
#%%
def glob_ers_2(Algo_under_test, DB): 
    glob_ers = np.zeros([10,len(DB["mt"])])
    glob_ers_means = np.zeros(10)
    prediction_values = np.zeros([len(DB["mt"]),10])
    for i in range(0, len(DB["mt"])):
        data = DB_2_class(DB, i)
        mp = float(DB["mp"][i])
        dv = float(DB["dV"][i])
        Isp = float(DB["Isp"][i]) 
        
        if Algo_under_test == routine_multiple_power_regression:
            MPowR_model = MPowR_initialiser(DB_ss)
            pred = Algo_under_test(mp, dv, Isp, MPowR_model)
        else :
            pred = Algo_under_test(mp, dv, Isp)
            
        # print("md and mprop:", float(np.array(pred.md)), pred.mprop)
        prediction_values[i, :] = np.array([pred.mt, float(np.array(pred.md)), float(np.array(pred.mprop)), pred.mp, float(pred.STR), float(pred.PRPLSN), float(pred.AVIO), float(pred.POW), float(pred.THER), float(pred.OTH)])
        ers_i, dump = V_ers_2(data, pred, "ith error")
        # print("ers_i, mp", ers_i[3])
        glob_ers[:,i] = ers_i
        # print(glob_ers[:,0])
    for i in range(0, 10):# for the ten mass-properties
        # glob_ers_means[i] = np.mean(glob_ers[:,i])
        # glob_ers_means[i] = np.median(glob_ers[:,i])
        # glob_ers_means[i] = np.median(np.sqrt(glob_ers[:,i]**2))
        glob_ers_means[i] = np.median(abs(glob_ers[i,:]))
    
    return glob_ers_means, prediction_values
        
#%%

def MPR_errors(data, pred):
    data = data.transpose()
    ss_ers_avg = np.zeros(6)
    # print("shape of data:", np.shape(data))
    # print("shape of prediction:", np.shape(pred))
    
    for i in range(0, len(data[:, 0])):
        # print("i", i)
        raw_ers = []
        for k in range(0, len(data[0, :])):
            # print("k", k)
            # print("pred[{0}, {1}]".format(i, k), pred[i, k])
            # print("data[{0}, {1}]".format(i, k), data[i, k])
            raw_ers.append(np.round(float((pred[i, k] - data[i, k])/(data[i, k])), 4)*100)
        # print(raw_ers)
        ss_ers_avg[i] = np.median(raw_ers)
    
    return ss_ers_avg







    