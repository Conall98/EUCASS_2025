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

def STR_MPR(X):
    str_model = MPoly_ss_models[0, :]
    coefs = str_model[0:20]
    intercept = str_model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def STR_MPowR(X):
    str_model = MPowR_model[0]
    return power_func(new_input, *str_model)

### PROPULSION ####
def PRPL_MLR(X):
    model = linear_ss_models[1, :]
    coefs = model[0:5]
    intercept = model[5]
    return np.dot(coefs, X) + intercept

def PRPL_MPR(X):
    model = MPoly_ss_models[1, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def PRPL_MPowR(X):
    model = MPowR_model[1]
    return power_func(new_input, *model)

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

def POW_MPowR(X):
    model = MPowR_model[2]
    return power_func(new_input, *model)

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

def AVIO_MPR(X):
    model = MPoly_ss_models[3, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def AVIO_MPowR(X):
    model = MPowR_model[3]
    return power_func(new_input, *model)

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

def THER_MPowR(X):
    model = MPowR_model[4]
    return power_func(new_input, *model)

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

def OTH_MPR(X):
    model = MPoly_ss_models[5, :]
    coefs = model[0:20]
    intercept = model[20]
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X.reshape(1, -1)).ravel()    
    return np.dot(coefs, X_poly) + intercept

def OTH_MPowR(X):
    model = MPowR_model[5]
    return power_func(new_input, *model)

def linear_estimations(X):
    a = STR_MLR(X)
    b = PRPL_MLR(X)
    c = POW_MLR(X)
    d = AVIO_MLR(X)
    e = THER_MLR(X)
    f = OTH_MLR(X)
    return a, b, c, d, e, f

def polynoial_estimations(X):
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

#%% The sizing Algorithm

def IAC_sizing_algorithm(mp, dv, Isp, models, max_iter=50, tol=0.01):
    """
    Shamelessly uses a blend of eerything such that the global errors are minimised
    Choose it from main
    
    """
    
    DBss = DB_ss
    md_0 = f1(mp)
    t1 = time.perf_counter()
    
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
    t2 = time.perf_counter()
    
    i = 0
    # tol = 0.01  # now passed as argument
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    store = []
    t3 = time.perf_counter()
    acf = 0.01
    # STR_model = MPowR_model[0]
    # # print("STR_model", STR_model)
    # PRPL_model = MPowR_model[1]
    # # print("PRPL_model", PRPL_model)
    # POW_model = MPowR_model[2]
    # # # print("POW_model", PRPL_model)
    # AVIO_model = MPowR_model[3]
    # # # print("AVIO_model", PRPL_model)
    # THER_model = MPowR_model[4]
    # # # print("THER_model", THER_model)
    # OTH_model = MPowR_model[5]
    # # print("OTH_model", PRPL_model)

    
    while er > tol:
        try:
            new_input = np.array([lander.md, lander.mp, lander.mprop, lander.dv, lander.Isp])
            
            # ⚠️ models() might overflow
            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)  # treat overflow as exception
                a, b, c, d, e, f = models(new_input)
        
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
            
            lander.md = (float(md_i[-1:][0]))
        
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

def EVAL(Algo_under_test, DB, models): 
    glob_ers = np.zeros([10,len(DB["mt"])]) #ten parameters for each DB entry
    glob_ers_means = np.zeros(10) #means of the ten params
    prediction_values = np.zeros([len(DB["mt"]),10])
    for i in range(0, len(DB["mt"])):
        data = DB_2_class(DB, i)
        mp = float(DB["mp"][i])
        dv = float(DB["dV"][i])
        Isp = float(DB["Isp"][i]) 

        pred = Algo_under_test(mp, dv, Isp, models)
            
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
