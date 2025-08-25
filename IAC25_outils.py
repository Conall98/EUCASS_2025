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

#%% The Modelling
def power_func(X, a1, a2, a3, a4, a5, b1, b2, b3, b4, b5):
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

def MPowR_initialiser(DBss):
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    
    STR_model = multiple_power_regression(X_data, y_data[:, 0])
    PRPL_model = multiple_power_regression(X_data, y_data[:, 1])
    POW_model = multiple_power_regression(X_data, y_data[:, 2])
    AVIO_model = multiple_power_regression(X_data, y_data[:, 3])
    THER_model = multiple_power_regression(X_data, y_data[:, 4])
    OTH_model = multiple_power_regression(X_data, y_data[:, 5])
    
    return STR_model, PRPL_model, POW_model, AVIO_model, THER_model, OTH_model

def Multiple_linear_regression(X, y, i):
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

def MLR_initialiser(ssDB):
    X_data = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
    y_data = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
    ss_models = np.zeros([6, 7]) #six row(predictions), five coefficients + one intercept for each pred
    Rs = []
    for i in range(0, 6):
        coefs, inter, r2 = Multiple_linear_regression(X_data, y_data, i)
        ss_models[i, 0:5] = coefs
        ss_models[i, 5:6] = inter
        ss_models[i, 6] = r2
        Rs.append(r2)
    
    return ss_models

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

def MPR_initialiser(ssDB):
    X_data = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
    y_data = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
    ss_models, _ = MPR(X_data, y_data, degree=2)
    
    return ss_models
#%% Model Initialisation
linear_ss_models = MLR_initialiser(DB_ss)
MPow_ss_models = MPowR_initialiser(DB_ss)
MPoly_ss_models = MPR_initialiser(DB_ss)
    
#%% Subsystem Sizing Rules
def f1(mp):
    return 367.29*np.log(mp) - 904.17

def f2(mp, md, dv, Isp):# gives mprop with mp+md
    return 1.16*(mp+md)*(np.exp(dv/(Isp*9.81)) - 1)

def f3(mp, mprop):
    x = mp+mprop
    return 12.49*x**0.55

def STR_MPR(X):
    str_model = ss_models[0, :]
    coefs = str_model[0:5]
    intercept = str_model[5]
    return np.dot(coefs, X) + intercept

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

def PRPL_Ramos(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape): #FT = Fuel type (class), T is Thrust WEIGHT Requirement, MR is O/F mixture ratio
    m_prpl, m_tanks, m_engines = tn.PRPL(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
    return m_prpl

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



#%% The sizing Algorithm
def IAC_sizing_algorithm(mp, dv, Isp, MPowR_model):
    """
    Shamelessly uses a blend of eerything such that the global errors are minimised
    Choose it from main
    
    """
    DBss = DB_ss
    md_0 = f1(mp)
    
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine   
    TWR = 1.73 #same as Apollo
    tank_material = M1
    n = 4 #number of engines
    Pressure = 20 #bars
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere" 
    P_tank_shape = "sphere"
    
    
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

        
        # new_input = np.array([[lander.mt, lander.mp, lander.mprop, lander.dv, lander.Isp]])
        new_input = np.array([[lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp]])
        a = lander.STR = STR2(new_input)
        # b=c=d=e=f=1
        b = lander.PRPLSN =  PRPL_Ramos(new_input[0], new_input[2], new_input[1], FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        c = lander.POW = power_func(new_input, *POW_model)
        d = lander.AVIO = power_func(new_input, *AVIO_model)
        e = lander.THER = THER_phys(ap_req, D_m, heater, mt)
        f = lander.OTH = power_func(new_input, *OTH_model)
        
        # print("STR", a,"PRPL",  b,"POW",  c,"AVIO",  d,"THER",  e,"OTH",  f)
        # print("STR", a)
        # 
        md_i1 = sum([a, b, c, d, e, f])
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
    