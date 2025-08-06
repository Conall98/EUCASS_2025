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
## initialising the models linear and polynomial models ###
ss_models = o.modeler(DB_ss)

A = np.array([DB_ss["md"], DB_ss["mp"], DB_ss["mprop"], DB_ss["dV"], DB_ss["Isp"]]).transpose()
B = np.array([DB_ss["Structure"], DB_ss["Propulsion"], DB_ss["Power"], DB_ss["Avionics"], DB_ss["Thermal Protection"], DB_ss["Other"]]).transpose()
MPR_models, MPR_features = o.MPR(A, B, 2)

#%% Main
### inputs

# LUT = LM_test
LUT = ESAS_J_test

mp = LUT.mp
dv = LUT.dv
Isp = LUT.Isp

# test1 = o.routine_Ramos_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Ramos_N2O4_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_all_linear(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4_MLR(mp, dv, Isp) #making the prediction
test1 = o.routine_stat_MLR_iter(mp, dv, Isp) #making the prediction
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
sizing_errors9, d9 = o.glob_ers_2(o.routine_MPR_noloop, DB_ss)


#%% Test 7 Multiple Power Regression
import outils as o
o.multiple_power_regression(DB_ss)

#%% Test 8 validation using ESAS

mp = LM_test.mp
dv = LM_test.dv
Isp = LM_test.Isp

MPR_lander = o.routine_MPR_noloop(mp, dv, Isp)

MPR_lander_ers, ers_dict = o.V_ers_2(LM_test, MPR_lander, "test1A")

#%% TEST 9 Multiple Power law regression

o.multiple_power_regression(DB_ss)























