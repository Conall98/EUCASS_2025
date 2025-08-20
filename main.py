# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:59:13 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
# importing the other files
import outils as o
from DBs import *
import matplotlib.pyplot as plt
## initialising the models linear and polynomial models ###
ss_models = o.modeler(DB_ss)

A = np.array([DB_ss["md"], DB_ss["mp"], DB_ss["mprop"], DB_ss["dV"], DB_ss["Isp"]]).transpose()
B = np.array([DB_ss["Structure"], DB_ss["Propulsion"], DB_ss["Power"], DB_ss["Avionics"], DB_ss["Thermal Protection"], DB_ss["Other"]]).transpose()
MPR_models, MPR_features = o.MPR(A, B, 2)
MPowR_model = o.MPowR_initialiser(DB_ss)
#%% Main
### inputs
import outils as o
# LUT = LM_test
LUT = ESAS_J_test

mp = LUT.mp
dv = LUT.dv
Isp = LUT.Isp

test1 = o.routine_Ramos_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Ramos_N2O4_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_all_linear(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4_MLR(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_I_N2O4_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_noloop(mp, dv, Isp) #making the prediction

# test1 = o.Isaji_imitator(133.4, 2104, 311, 3.125, 2, 0.1) #making the prediction

# test1 = o.routine_MPR_noloop(mp, dv, Isp)

ers, ers_dic = o.V_ers_2(LUT, test1, "test1A") #The error comparison
o.V_ss_plotter(LUT, test1, "test1A")
# ers, ers_dic = o.V_ers_2_isaji(LM_test_isaji, test1, "test1A") #The error comparison


ers_dic #calling it

#%%Test area
# test 1: correlating single feature single target
X_data = np.array([DB_ss["mprop"]]).transpose()
y_data = np.array([DB_ss["Structure"]]).transpose()
model = o.flexible_modeler(X_data, y_data, 0)
#%% test 2: corellating all features with on target
X_data = np.array([DB_ss["md"], 
                   DB_ss["mp"], 
                   DB_ss["mprop"], 
                   DB_ss["dV"], 
                   DB_ss["Isp"], 
                   DB_ss["Propulsion"], 
                   DB_ss["Power"], 
                   DB_ss["Avionics"]]).transpose()
y_data = np.array([DB_ss["Structure"]]).transpose()
model2 = o.flexible_modeler(X_data, y_data, 0)

#%% test 3: Test of the progressive sizing regime
import outils as o
progressive_lander = o.Progessive_MLR_Sizing(mp, dv, Isp, DB_ss)

test1 = o.V_ers_2(LM_test, progressive_lander, "test1A") #The error comparison

#%% test 4: validates one routine against the whole database
# The routine is specified in the function

glob_ers_means = o.global_errors(DB_ss)

#%% Test 5
import outils as o

mp = LM_test.mp
dv = LM_test.dv
Isp = LM_test.Isp

MPR_lander = o.routine_MPR_noloop(mp, dv, Isp)

MPR_lander_ers, ers_dict = o.V_ers_2(LM_test, MPR_lander, "test1A")



# MPR_ss_models = o.MPR(X_data, y_data, 4)
# MPR_predictions = o.m_predict(MPR_ss_models, X_data)

# MPR_ers = o.MPR_errors(y_data, MPR_predictions)
# MPR_ss_models_even = o.MPR(X_data[::2], y_data[::2], 0)
# MPR_ss_models_odd = o.MPR(X_data[1::2], y_data[1::2], 0)

#%% TEST 6 ######
import outils as o
# sizing_errors1, d1 = o.glob_ers_2(o.routine_Ramos_N2O4, DB_ss)
# sizing_errors2, d2 = o.glob_ers_2(o.routine_Ramos_N2O4_iter, DB_ss)
# sizing_errors3, d3 = o.glob_ers_2(o.routine_all_linear, DB_ss)
# sizing_errors4, d4 = o.glob_ers_2(o.routine_Isaji_N2O4, DB_ss)
# sizing_errors5, d5 = o.glob_ers_2(o.routine_Isaji_N2O4_MLR, DB_ss)
# sizing_errors6, d6 = o.glob_ers_2(o.routine_stat_MLR_iter, DB_ss)
# sizing_errors7, d7 = o.glob_ers_2(o.routine_I_N2O4_MLR_iter, DB_ss)
# sizing_errors8, d8 = o.glob_ers_2(o.routine_stat_MLR_noloop, DB_ss)
# sizing_errors9, d9 = o.glob_ers_2(o.routine_MPR_noloop, DB_ss)
sizing_errors10, d10 = o.glob_ers_2(o.routine_multiple_power_regression, DB_ss)

#%%
import outils as o
LUT = LM_test
# LUT = ESAS_J_test

mp = LUT.mp
dv = LUT.dv
Isp = LUT.Isp

multi_power_lander = o.routine_multiple_power_regression(mp, dv, Isp, o.MPowR_model)
# print(storage)
# plt.figure()
# plt.plot(storage)

# multi_power_lander_ers, ers_dict = o.V_ers_2(LUT, multi_power_lander, "test1A")
# o.V_ss_plotter(LUT, multi_power_lander, "test1A")

#%% Test 8 validation using ESAS

mp = LM_test.mp
dv = LM_test.dv
Isp = LM_test.Isp

MPR_lander = o.routine_MPR_noloop(mp, dv, Isp)

MPR_lander_ers, ers_dict = o.V_ers_2(LM_test, MPR_lander, "test1A")


#%% STR investigaton
import outils as o
STR_Pow_model = o.STR_MPowR(A, B)
R2_STR = o.R_squared_Pow(A, STR_Pow_model, B[:,0])

PRPL_Pow_model = o.PRPL_MPowR(A, B)
R2_PRPL = o.R_squared_Pow(A, PRPL_Pow_model, B[:,1])

POW_Pow_model = o.POW_MPowR(A,B)
R2_POW = o.R_squared_Pow(A, POW_Pow_model, B[:,2])

AVIO_Pow_model = o.AVIO_MPowR(A,B)
R2_AVIO = o.R_squared_Pow(A, AVIO_Pow_model, B[:,3])

THER_Pow_model = o.THER_MPowR(A,B)
R2_THER = o.R_squared_Pow(A, THER_Pow_model, B[:,4])

OTH_Pow_model = o.OTH_MPowR(A,B)
R2_OTH = o.R_squared_Pow(A, OTH_Pow_model, B[:,5])

#%% Multiple Power Regression function investigation
# import outils as o
LUT = LM_test
# LUT = ESAS_J_test
mp = LUT.mp
dv = LUT.dv
Isp = LUT.Isp

target = B[:,3]
powers = o.multiple_power_regression(A, target)
R2 = o.R_squared_Pow(A, powers, target)
test_input = np.array([[LUT.md, LUT.mp, LUT.mprop, LUT.dv, LUT.Isp]])
for i in range(5):
    print("{}*{}^{}".format(np.round(powers[i], 4), test_input[:,i], np.round(powers[i+5], 2)))
print("R2", R2)
prediction = o.func(test_input, *powers)
print("prediction", prediction)

test_mds = np.linspace(0, max(DB_ss["md"]), len(target))
test_mps = np.linspace(0, max(DB_ss["mp"]), len(target))
test_x = test_mds+test_mps
pred_line = []

plt.figure()
plt.xlabel("$m_d [kg]$")
# plt.ylabel("$m_{STR} [kg]$")
for i in range(0, len(target)):
    LUT = DB_2_class(DB_ss, i)
    test_input = np.array([[LUT.md, LUT.mp, LUT.mprop, LUT.dv, LUT.Isp]])
    prediction = o.func(test_input, *powers)
    plt.scatter(LUT.md, prediction, color = "orange")
    plt.scatter(LUT.md, LUT.STR, color = "blue")
    
    test_line = np.array([[test_mds[i], 
                           test_mps[i], 
                           np.median(DB_ss["mprop"]), 
                           np.median(DB_ss["dV"]), 
                           np.median(DB_ss["Isp"])]])
    pred_line.append(o.func(test_line, *powers))

# plt.plot(test_mds, pred_line, color = "orange", linestyle = "--")


#%% POW_ssr
p_req = 2000
ap_req = 500
D_m = 4.7*24 #hrs
SSR_power_fuel_cell, SSR_fuel_cell_propellant = o.POW_SSR_fuelcell(p_req , ap_req, D_m) #(peak_power, average_power, mission_duration)
SSR_power_battery = o.POW_SSR_battery(p_req , ap_req, D_m) #(peak_power, average_power, mission_duration)

#%% THER
ap_req = 300
D_m = 15
heater = "not nuclear"
mt = 16000
o.THER_phys(ap_req, D_m, heater, mt)



