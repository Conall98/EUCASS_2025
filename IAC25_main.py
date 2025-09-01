# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:30:31 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from IAC25_outils import *
#%%

mp = 2000
dv = 2500
Isp = 350
# lander = IAC_sizing_algorithm(mp, dv, Isp, linear_estimations)
lander = IAC_sizing_algorithm(mp, dv, Isp, polynoial_estimations)

#%% Testing
# X = np.array([1, 2, 3, 4, 5])
# STR_MLR(X)

mean_errors, repredictions = EVAL(IAC_sizing_algorithm, DB_ss)