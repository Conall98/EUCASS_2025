# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:50:25 2025

@author: cdepaor
"""

import numpy as np
import tank_sizing_subroutine as tn
        
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
    return md*0.088

def STR(md):
    return md*0.276

def POW(md):
    return md*0.076

def THER(md):
    return md*0.13

def OTH(md):
    return md*0.062

def PROP(md):
    return md*0.337

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








