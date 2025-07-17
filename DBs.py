# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:59:27 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import numpy as np
import pandas as pd
#%%

class material():
    def __init__(self, name, density, sigma_max):
        self.name = name
        self.density = density
        self.sigma_max = sigma_max
        
class Fuel():
    def __init__(self, name, lox_density, fuel_density, mixture_ratio, Isp):        
        self.name = name
        self.rho_lox = lox_density
        self.rho_fuel= fuel_density 
        self.MR = mixture_ratio
        self.Isp = Isp
        
class Tank():
    def __init__(self, mass, shape, thickness, test, volume):
        self.mass = mass
        self.shape = shape
        self.thickness = thickness
        self.test = test
        self.volume = volume
        
class L:
    def __init__(self, name, mp, md, mprop, mt, dv=None, isp=None, STR = None, PRPLSN = None, POW = None, AVIO = None, THER = None, OTH = None):
        self.name = name
        self.mp = np.round(mp, 2)
        self.md = np.round(md, 2)
        self.mprop = np.round(mprop, 2)
        self.mt = np.round(mt, 2)
        self.dv = dv
        self.Isp = isp
        self.STR = STR
        self.PRPLSN = PRPLSN
        self.POW = POW
        self.AVIO = AVIO
        self.THER = THER
        self.OTH = OTH
        
    def test(self):
        mp = self.mp
        md = self.md
        dv = self.dv
        Isp = self.Isp
        mprop = self.mprop
        if mprop > (mp+md)*(np.exp(dv/(Isp*9.81)) - 1):
            margin = mprop/((mp+md)*(np.exp(dv/(Isp*9.81)) - 1))
            return "test passed. margin: {0}".format(margin)
        else:
            return "test failed"
class L_isaji:
    def __init__(self, name, mp, md, mprop, mt, dv=None, isp=None, STRTPS = None, PRPLSN = None, POW = None, AVIO = None, ECLSS = None, OTH = None):
        self.name = name
        self.mp = np.round(mp, 2)
        self.md = np.round(md, 2)
        self.mprop = np.round(mprop, 2)
        self.mt = np.round(mt, 2)
        self.dv = dv
        self.Isp = isp
        self.STRTPS = STRTPS
        self.PRPLSN = PRPLSN
        self.POW = POW
        self.AVIO = AVIO
        self.ECLSS = ECLSS
        self.OTH = OTH
        
    def test(self):
        mp = self.mp
        md = self.md
        dv = self.dv
        Isp = self.Isp
        mprop = self.mprop
        if mprop > (mp+md)*(np.exp(dv/(Isp*9.81)) - 1):
            margin = mprop/((mp+md)*(np.exp(dv/(Isp*9.81)) - 1))
            return "test passed. margin: {0}".format(margin)
        else:
            return "test failed"
#%%
######## Formatting Functions #########
def Isaji_converter(L1): #converts ORLA landers into Isaji Landers
    L = L_isaji(L1.name, L1.mp, L1.md, L1.mprop, L1.mt, L1.dv, L1.Isp, L1.STR+L1.THER, L1.PRPLSN, L1.POW, L1.AVIO, 1, L1.OTH)
    return L
    # print(L_isaji)
    
def DB_2_class(DB, i):
    mt = DB["md"][i] + DB["mprop"][i] + DB["mp"][i]
    lander_data = L(DB["Lander"][i], DB["mp"][i], DB["md"][i], DB["mprop"][i], mt, DB["dV"][i], DB["Isp"][i], 
                       DB["Structure"][i],DB["Propulsion"][i],DB["Power"][i],DB["Avionics"][i],DB["Thermal Protection"][i],DB["Other"][i])
    
    return lander_data    
#%%

F1 = Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = Fuel("LOX/LCH4", 1141, 657, 3.5, 350)

M1 = material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = material("CFRP", 1420, 1260E+6) #matweb
M3 = material("Aluminium 2195", 2710, 590E+6) #makeitfrom.com

#%%
DB_Landers = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
DB_ss = pd.read_excel(r"subsystems database.xlsx")

LM_test_isaji = L("LM_isaji", 4795, 2217, 8880, 16371, dv=2104, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
LM_test = L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)

Isaji_ers = np.array([0.641, -0.425, 1.32, 0, -1.83, -0.98, -5.33, -0.886, 4.18, 5.14])
isaji_ers_dict = {"mt:      ":Isaji_ers[0],
                  "md:      ":Isaji_ers[1],
                  "mprop:   ":Isaji_ers[2],
                  "mp:      ":Isaji_ers[3],
                  "STR:     ":Isaji_ers[4],
                  "PRPL:    ":Isaji_ers[5],
                  "AVIO:    ":Isaji_ers[6],
                  "POW:     ":Isaji_ers[7],
                  "ECLSS:   ":Isaji_ers[8],
                  "OTH:     ":Isaji_ers[9]}

 # prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]



