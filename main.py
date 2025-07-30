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
#%% importing the other files
import outils as o
from DBs import *

#%% Main
### inputs
mp = LM_test.mp
dv = LM_test.dv
Isp = LM_test.Isp

Dsm = 3.125
Ncrw = 2
C_other = 0.1

test1 = o.routine_Ramos_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Ramos_N2O4_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_all_linear(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4_MLR(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_I_N2O4_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_noloop(mp, dv, Isp) #making the prediction

# test1 = o.Isaji_imitator(133.4, 2104, 311, 3.125, 2, 0.1) #making the prediction



ers, ers_dic = o.V_ers_2(LM_test, test1, "test1A") #The error comparison
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
X_data = np.array([DB_ss["md"], 
                   DB_ss["mp"], 
                   DB_ss["mprop"], 
                   DB_ss["dV"], 
                   DB_ss["Isp"], 
                   DB_ss["Propulsion"], 
                   DB_ss["Power"], 
                   DB_ss["Avionics"]]).transpose()
y_data = np.array([DB_ss["Structure"], DB_ss["Propulsion"], DB_ss["Power"], DB_ss["Avionics"], DB_ss["Thermal Protection"], DB_ss["Other"]]).transpose()
coefs, intercept, R2 = o.MPR(X_data, y_data, 4)










