# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:50:25 2025

@author: cdepaor
"""

import numpy as np
import tank_sizing_subroutine as tn
import pandas as pd   
from sklearn import linear_model     
#%%
F1 = tn.Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = tn.Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = tn.Fuel("LOX/LCH4", 1141, 657, 3.5, 350)

M1 = tn.material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = tn.material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = tn.material("CFRP", 1420, 1260E+6) #matweb
M3 = tn.material("Aluminium 2195", 2710, 590E+6) #makeitfrom.com

DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
DB_ss = pd.read_excel(r"subsystems database.xlsx")

#%%
def MLR(X, y, verbose):
    regr = linear_model.LinearRegression()
    regr.fit(X, y)
    model_result = {"Regression Coefficients": regr.coef_,
                    "Intercept": regr.intercept_,
                    "R2":regr.score(X, y)}
    if verbose == True:
        return model_result
    else:
        return regr
#%%
X = np.array([DB_ss["md"], DB_ss["mp"], DB_ss["mprop"], DB_ss["Isp"], DB_ss["dV"]]).transpose()

y1 =  np.array(DB_ss["Avionics"])[:-2]
y2 =  np.array(DB_ss["Structure"])[:-2]
y3 =  np.array(DB_ss["Power"])[:-2]
y4 =  np.array(DB_ss["Thermal Protection"])[:-2]
y5 =  np.array(DB_ss["Other"])[:-2]
y6 =  np.array(DB_ss["Propulsion"])[:-2]

X = X[:-2,:]

M_AVIO = MLR(X, y1, False)
M_STR = MLR(X, y2, False)
M_POW = MLR(X, y3, False)
M_THER = MLR(X, y4, False)
M_OTH = MLR(X, y5, False)
M_PRPL = MLR(X, y6, False)

x_pow = np.array([0.112, 2, 2, 3])
x_cap = np.array([8.4, 144, 144, 24.9])
x_md = np.array([5295, 25000, 25000, 22074])
y_pow = np.array([495, 478, 478, 288])
X_star = np.array([x_pow, x_cap, x_md]).transpose()
M_POWPOW = MLR(X_star, y_pow, True)

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

def STR(md):
    return md*0.276

def STR_mp(mp):
    return mp*0.0747

def POW(md):
    return md*0.018+311.68

def POW_star_2(md, cap, watts):
    X =  np.array([watts, cap, md]).transpose()
    m_pow = M_POWPOW.predict(X)
    return m_pow

def THER(mt):
    return mt*0.0197

def OTH(mp):
    return mp*0.0316

def PROP(md):
    return md*0.337

def PROP_mt(mt):
    return mt*0.044

#%% MLR
def AVIO_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_AVIO.predict(X)
    return m_avio

def STR_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_STR.predict(X)
    return m_avio

def POW_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_POW.predict(X)
    return m_avio

def THER_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_THER.predict(X)
    return m_avio

def OTH_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_OTH.predict(X)
    return m_avio

def PROP_star(X):# X = md, mp, mprop, dv, Isp
    m_avio = M_PRPL.predict(X)
    return m_avio
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








