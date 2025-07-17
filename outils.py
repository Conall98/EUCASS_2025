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
#%%
######## Regression Modelling #########
def Linregger(X, y, i):
    # print("i", i)
    model = linear_model.LinearRegression()
    model.fit(X, y[:, i])
    R2 = model.score(X, y[:, i])
    coefs = model.coef_
    intercept =  model.intercept_
    preds = model.predict([X[0]])
    manual_pred = np.dot(coefs, X[0]) + model.intercept_
    # print("predicted y", preds)
    # print("manual_preds", manual_pred)
    # print("actual y", y[0, i])
    # print(preds == manual_pred)
    return coefs, intercept, R2

def modeler(ssDB):
    X_data = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
    y_data = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
    ss_models = np.zeros([6, 6]) #six row(predictions), five coefficients + one intercept for each pred
    Rs = []
    for i in range(0, 6):
        coefs, inter, r = Linregger(X_data, y_data, i)
        ss_models[i, 0:5] = coefs
        ss_models[i, 5:6] = inter
        Rs.append(r)
    
    return ss_models

def polyregger(X, y):
    def func(a, x, b, c):
        return a*x**(b)+c
    for i in range(0,6):
        xdata = X[:,i]
        ydata = y
        popt, pcov = curve_fit(func, xdata, ydata)

ss_models = modeler(DB_ss)


#%%
######## The Major Mass Property functions ########
def f1(mp):
    return 367.29*np.log(mp) - 904.17

def f2(mp, md, dv, Isp):# gives mprop with mp+md
    return 1.16*(mp+md)*(np.exp(dv/(Isp*9.81)) - 1)

def f3(mp, mprop):
    x = mp+mprop
    return 12.49*x**0.55

#%%
######## The Subsystem Sizing Routines #########

def AVIO(md):
    return md*0.0774

def AVIO2(X):
    avio_model = ss_models[3, :]
    coefs = avio_model[0:5]
    intercept = avio_model[5]
    return np.dot(coefs, X) + intercept


def STR(md):
    return md*0.276

def STR_mp(mp):
    return mp*0.0747

def STR2(X):
    str_model = ss_models[0, :]
    coefs = str_model[0:5]
    intercept = str_model[5]
    return np.dot(coefs, X) + intercept


def POW(md):
    return md*0.018+311

def POW2(X):
    pow_model = ss_models[2, :]
    coefs = pow_model[0:5]
    intercept = pow_model[5]
    return np.dot(coefs, X) + intercept


def THER(mt_0):
    return mt_0*0.0197

def THER2(X):
    ther_model = ss_models[4, :]
    coefs = ther_model[0:5]
    intercept = ther_model[5]
    return np.dot(coefs, X) + intercept


def OTH(mp):
    return mp*0.0316

def OTH2(X):
    oth_model = ss_models[5, :]
    coefs = oth_model[0:5]
    intercept = oth_model[5]
    return np.dot(coefs, X) + intercept


def PROP(mt_0):
    return mt_0*0.044

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

def PRPL_Isaji_storable(md, mp, FT, Isp):
    m_inert = md + mp
    
    rho = ((FT.rho_fuel) + (FT.rho_lox)*FT.MR)/FT.MR
    # Isp = FT.Isp
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
    d = lander.AVIO = AVIO(md_i[i])
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(mp)
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
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
        d = lander.AVIO = AVIO(md_i[i])
        e = lander.THER = THER(lander.mt)
        f = lander.OTH = OTH(mp)
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
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
    d = lander.AVIO = AVIO(md_i[i])
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(mp)
    
    while er > tol:
        b = lander.PRPLSN = PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
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
        er = 1 - md_i1/md_i[i]
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
        er = 1 - md_i1/md_i[i]
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
    d = lander.AVIO = AVIO(md_i)
    e = lander.THER = THER(mt_0)
    f = lander.OTH = OTH(mp)
    
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
        d = lander.AVIO = AVIO(md_i[i])
        e = lander.THER = THER(mt_0)
        f = lander.OTH = OTH(mp)
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
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

######### Validation Functions ##########
def V_ers_2(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)
    

    return np.array(errors)
# #%%
# LM_pred = EUC_AZ.routine_all_linear(5295, 2265, 311)
# LM_test = mf.L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
# V_ers_2(LM_test, LM_pred, "all")

def V_ers_2_isaji(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STRTPS, data.PRPLSN, data.AVIO, data.POW, data.ECLSS, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STRTPS, pred.PRPLSN, pred.AVIO, pred.POW, pred.ECLSS, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)

    return np.array(errors)