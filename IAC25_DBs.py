# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:31:47 2025

This file contains all the objects used by the functions in outils and the routines in main
Lander, fuel, tank material, 
it also declares some initial objects: lander databases and the subsystem 
databases, the test landers, and the fuels and materials
it also imports all the modules required for all the files. 

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
import pandas as pd
from mulitple_regression import *
import tank_sizing_subroutine as tn
from scipy.optimize import curve_fit
import logging
import matplotlib.pyplot as plt
from sklearn import linear_model
import math as m
from sklearn.preprocessing import PolynomialFeatures
import warnings

#%% Formatting function
def DB_2_class(DB, i):
    mt = DB["md"][i] + DB["mprop"][i] + DB["mp"][i]
    lander_data = L(DB["Lander"][i], DB["mp"][i], DB["md"][i], DB["mprop"][i], mt, DB["dV"][i], DB["Isp"][i], 
                       DB["Structure"][i],DB["Propulsion"][i],DB["Power"][i],DB["Avionics"][i],DB["Thermal Protection"][i],DB["Other"][i])
    
    return lander_data  

#%% Objects
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

F1 = Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = Fuel("LOX/LCH4", 1141, 657, 3.5, 350)

M1 = material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = material("CFRP", 1420, 1260E+6) #matweb
M3 = material("Aluminium 2195", 2710, 590E+6) #makeitfrom.com

#%%
DB_Landers = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
DB_ss = pd.read_excel(r"subsystems database (normalised).xlsx")
# DB_ss = pd.read_excel(r"subsystems database (normalised).xlsx")

LM_test_isaji = L("LM_isaji", 4795, 2217, 8880, 16371, dv=2104, isp=311, STR = 460, PRPLSN = 492.6, POW = 356.1, AVIO = 36.3, THER = 411.2, OTH = 246)
LM_test_isaji = L_isaji("LM_isaji", 4795, 2217, 8880, 16371, dv=2104, isp=311, STRTPS = 460+411.2, PRPLSN = 492.6, POW = 356.1, AVIO = 36.3, ECLSS=214.6, OTH = 246)

LM_test = L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
ESAS_J_test = DB_2_class(DB_ss, 9)

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
    
    popt, pcov = curve_fit(power_func, X, y, p0=initial_guess, full_output = False, bounds=bounds, maxfev=1000000)
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
#%% initialising the various models

linear_ss_models = MLR_initialiser(DB_ss)
MPow_ss_models = MPowR_initialiser(DB_ss)
MPoly_ss_models = MPR_initialiser(DB_ss)