# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:03:19 2025

@author: cdepaor
"""
import numpy as np

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
        
#%%

F1 = Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = Fuel("LOX/LCH4", 1141, 657, 3.5, 350)

M1 = material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = material("CFRP", 1420, 1260E+6) #matweb
M3 = material("Aluminium 2195", 2710, 590E+6) #makeitfrom.com