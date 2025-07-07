# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:50:25 2025

@author: cdepaor
"""

import numpy as np
import tank_sizing_subroutine as tn
import pandas as pd
from mulitple_regression import *

DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
ssDB = pd.read_excel(r"subsystems database.xlsx")

#%% subsystem predictions
X = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
y = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
#%%
def modeler(X, y, i):
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

# ss_models = np.zeros([6, 6]) #six row(predictions), five coefficients + one intercept for each pred
# Rs = []
# for i in range(0, 6):
#     coefs, inter, r = modeler(X, y, i)
#     ss_models[i, 0:5] = coefs
#     ss_models[i, 5:6] = inter
#     Rs.append(r)
    
# str_model = ss_models[0, :]
# prpl_model = ss_models[1, :]
# pow_model = ss_models[2, :]
# avio_model = ss_models[3, :]
# ther_model = ss_models[4, :]
# oth_model = ss_models[5, :]
#%%
F1 = tn.Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = tn.Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = tn.Fuel("LOX/LCH4", 1141, 657, 6, 350)

M1 = tn.material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = tn.material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = tn.material("CFRP", 1420, 1260E+6) #matweb
M3 = tn.material("Aluminium 2195", 2710, 590E+6) #makeitfrom.com


#%%
def f1(mp):
    return 367.29*np.log(mp) - 904.17

def f2(mp, md, dv, Isp):# gives mprop with mp+md
    return 1.16*(mp+md)*(np.exp(dv/(Isp*9.81)) - 1)

def f3(mp, mprop):
    x = mp+mprop
    return 12.49*x**0.55

#%% subsystem functions 

def AVIO(md):
    return md*0.0774

def AVIO2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = avio_model[0:5]
    intercept = avio_model[5]
    return np.dot(coefs, X) + intercept


def STR(md):
    return md*0.276

def STR_mp(mp):
    return mp*0.0747

def STR2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = str_model[0:5]
    intercept = str_model[5]
    return np.dot(coefs, X) + intercept


def POW(md):
    return md*0.018+311

def POW2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = pow_model[0:5]
    intercept = pow_model[5]
    return np.dot(coefs, X) + intercept


def THER(mt_0):
    return mt_0*0.0197

def THER2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = ther_model[0:5]
    intercept = ther_model[5]
    return np.dot(coefs, X) + intercept


def OTH(mp):
    return mp*0.0316

def OTH2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = oth_model[0:5]
    intercept = oth_model[5]
    return np.dot(coefs, X) + intercept


def PROP(mt_0):
    return mt_0*0.044

def PROP2(md, mp, mprop, dV, Isp):
    X = [md, mp, mprop, dV, Isp]
    coefs = prpl_model[0:5]
    intercept = prpl_model[5]
    return np.dot(coefs, X) + intercept

#%%

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

#%% test case

# md = 2217
# mprop = 10292
# mp = 2000
# FT = F2
# tank_material = M1
# TWR = 1.72
# n = 4 #number of engines
# Pressure = 2 #bars
# OX_tank_shape = "sphere"
# F_tank_shape = "sphere"
# P_tank_shape = "sphere"

# test11 = PRPL_Ramos(md, mprop, mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape) #shape








