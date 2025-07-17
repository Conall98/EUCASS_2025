# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:59:13 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import outils as o
from DBs import *
#%%4
### inputs
mp = LM_test.mp
dv = LM_test.dv
Isp = LM_test.Isp

Dsm = 3.125
Ncrw = 2
C_other = 0.1

# test1 = o.routine_Ramos_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Ramos_N2O4_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_all_linear(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4(mp, dv, Isp) #making the prediction
# test1 = o.routine_Isaji_N2O4_MLR(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_I_N2O4_MLR_iter(mp, dv, Isp) #making the prediction
# test1 = o.routine_stat_MLR_noloop(mp, dv, Isp) #making the prediction

test1 = o.Isaji_imitator(mp, dv, Isp, Dsm, Ncrw, C_other) #making the prediction



# ers, ers_dic = o.V_ers_2(LM_test, test1, "test1A") #The error comparison
ers, ers_dic = o.V_ers_2_isaji(LM_test_isaji, test1, "test1A") #The error comparison


ers_dic #calling it