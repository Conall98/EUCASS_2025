# -*- coding: utf-8 -*-
"""
Created on Fri Jun 20 10:36:19 2025

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
import matplotlib.pyplot as plt
import EUCASS_Subsystem_Subroutines as ESR
import tank_sizing_subroutine as tn
import matplotlib.pyplot as plt
import mat_fuel_db as mf
import EUCASS_LOXLH2 as EUC_LH2
import EUCASS_N2O4AZ as EUC_AZ
import pandas as pd
from mulitple_regression import *
DB_ss = pd.read_excel(r"subsystems database.xlsx")
#%%

def V_ss_plotter(data, pred, testname): #two lander objects

    dat = [data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH]
    prd = [pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]
    
    dat2 = [data.mt, data.md, data.mprop]
    prd2 = [pred.mt, pred.md, pred.mprop]

# Set category labels at adjusted positions
    x_positions = [0, 1, 
                   3, 4, 
                   6, 7, 
                   9, 10, 
                   12, 13, 
                   15, 16]
    j = 0
    k = 0
    l = 0
    plt.figure()
    while k < len(x_positions):
        # print(x_positions[k])
        plt.bar(x_positions[k], dat[j], width=1, align='center', color = "blue", label = "data")
        k=k+1
        # print(x_positions[k])
        plt.bar(x_positions[k], prd[j], width=1, align='center', color = "orange", label = "prediction")
        if j == 0:
            plt.legend()
        k=k+1
        j = j + 1


    x_labels = ["str", "prpl", "avio", "pow", "therm", "oth"]
    plt.xticks([0.5, 3.5, 6.5, 9.5, 12.5, 15.5], x_labels)
    
    plt.xlabel('Subsystems')
    plt.ylabel('Mass [kg]')
    plt.title('Sizing algorithm validation: {}'.format(testname))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()
    
# Set category labels at adjusted positions
    x_positions2 = [0, 1, 
                   3, 4, 
                   6, 7]
    j = 0
    k = 0
    l = 0
    plt.figure()
    while k < len(x_positions2):
        # print(x_positions[k])
        plt.bar(x_positions2[k], dat2[j], width=1, align='center', color = "blue", label = "data")
        k=k+1
        # print(x_positions[k])
        plt.bar(x_positions2[k], prd2[j], width=1, align='center', color = "orange", label = "prediction")
        if j == 0:
            plt.legend()
        k=k+1
        j = j + 1


    x_labels = ["$m_t$", "$m_d$", "$m_{prop}$", ]
    plt.xticks([0.5, 3.5, 6.5], x_labels)
    
    plt.xlabel('Major mass properties')
    plt.ylabel('Mass [kg]')
    plt.title('Sizing algorithm validation: {}'.format(testname))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()    
#%%
def NRMS(sample, prediction):
    errors = []
    for i in range(0, len(sample)):
        er = abs(sample[i] - prediction[i])
        errors.append(er)
    errors = np.array(errors)
    z = np.sqrt(np.mean(errors**2))
    nrms = z/np.mean(sample)
    return np.round(nrms, 4)
#%%
def V_ers(data, pred, testname):
    dat = [data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH, data.md, data.mprop, data.mp, data.mt]
    prd = [pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH, pred.md, pred.mprop, pred.mp, pred.mt]
    isaj = [1499, 816.3, 324.8, 688.5, 584.4, 337] #STR+Ther, PRPL, POW, AVIO, ECLSS, OTH
    error_measure_1 = []
    # print("here", prd[i])
    for k in range(0, len(dat)):
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)        
        error_measure_1.append(erm)
    error_measure_1 = {"str     ":error_measure_1[0],
                       "prp     ":error_measure_1[1],
                       "avio    ":error_measure_1[2],
                       "pow     ":error_measure_1[3],
                       "ther    ":error_measure_1[4],
                       "oth     ":error_measure_1[5],
                       "md      ":error_measure_1[6],
                       "mprop   ":error_measure_1[7],
                       "mt      ":error_measure_1[8],}
    

    return error_measure_1
    
#%%
def V_ers_2(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STR, data.PRPLSN, data.AVIO, data.POW, data.THER, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STR, pred.PRPLSN, pred.AVIO, pred.POW, pred.THER, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)
    

    return np.array(errors)
# #%%
# LM_pred = EUC_AZ.routine_all_linear(5295, 2265, 311)
# LM_test = mf.L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
# V_ers_2(LM_test, LM_pred, "all")

def V_ers_2_isaji(data, pred, testname):
    dat = [data.mt, data.md, data.mprop, data.mp, data.STRTPS, data.PRPLSN, data.AVIO, data.POW, data.ECLSS, data.OTH]
    prd = [pred.mt, pred.md, pred.mprop, pred.mp, pred.STRTPS, pred.PRPLSN, pred.AVIO, pred.POW, pred.ECLSS, pred.OTH]
    errors = []
    for k in range(0, len(dat)): #error magnitude of each quantity wrt data in %
        # print("here", len(prd[k]))
        erm = np.round(float((prd[k] - dat[k])/(dat[k])), 4)*100        
        errors.append(erm)
    

    return np.array(errors)

#%%
def global_errors(DB):
    glob_ers = np.zeros([10,10])
    glob_ers_means = np.zeros(10)
    for i in range(0, 10):
        data = DB_2_class(DB_ss, i)
        mp = float(DB_ss["mp"][i])
        dV = float(DB_ss["dV"][i])
        Isp = float(DB_ss["Isp"][i]) 
        
        ############### for N2O4 #################
        # pred = EUC_AZ.routine_all_linear(mp, dV, Isp) # the sizing algorithm under investigation 
        # pred = EUC_AZ.routine_stat_MLR_noloop(mp, dV, Isp)
        # pred = EUC_AZ.routine_stat_MLR_iter(mp, dV, Isp)
        # pred = EUC_AZ.routine_I_N2O4_MLR_iter(mp, dV, Isp)
        # pred = EUC_AZ.routine_Isaji_N2O4_MLR(mp, dV, Isp)
        # pred = EUC_AZ.routine_Isaji_N2O4(mp, dV, Isp)
        # pred = EUC_AZ.routine_Ramos_N2O4(mp, dV, Isp)
        # pred = EUC_AZ.routine_Ramos_N2O4_iter(mp, dV, Isp)
        
        
        ############# for LOX/LH2 ################
        
        # pred = EUC_LH2.routine_Ramos_Cryo(mp, dV, Isp)
        # pred = EUC_LH2.routine_Isaji_cryo(mp, dV, Isp)
        pred = EUC_LH2.routine_Ramos_Cryo_lessloop(mp, dV, Isp)
        
        ########### for lch4 ###################
        # pred = EUC_LH2.routine_Ramos_Cryo_CH4(mp, dV, Isp)
        
        
        # print(pred.test())
        ers_i = V_ers_2(data, pred, "ith error")
        # print(ers_i[1])
        # print("mp_ers: ",data.mp, pred.mp)
        glob_ers[i,:] = ers_i
    
    for i in range(0, 10):
        # glob_ers_means[i] = np.mean(glob_ers[:,i])
        # glob_ers_means[i] = np.median(glob_ers[:,i])
        # glob_ers_means[i] = np.median(np.sqrt(glob_ers[:,i]**2))
        glob_ers_means[i] = np.median(abs(glob_ers[:,i]))
        # print(glob_ers[:,i])
        
    return glob_ers_means, pred


#%%
# LM_pred = EUC_AZ.routine_all_linear(5295, 2265, 311)
# LM_test = mf.L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)
# import EUCASS_N2O4AZ as EUC_AZ
import EUCASS_LOXLH2 as EUC_LH2
Algo_errors, pred2 = global_errors(DB)
        
#%%
def DB_2_class(DB, i):
    mt = DB["md"][i] + DB["mprop"][i] + DB["mp"][i]
    lander_data = mf.L(DB["Lander"][i], DB["mp"][i], DB["md"][i], DB["mprop"][i], mt, DB["dV"][i], DB["Isp"][i], 
                       DB["Structure"][i],DB["Propulsion"][i],DB["Power"][i],DB["Avionics"][i],DB["Thermal Protection"][i],DB["Other"][i])
    
    return lander_data
#%%
def V_all_db(DB):
    errors = []
    for i in range(0, 10):
    # for i in range(0, 1):
        data = DB_2_class(DB, i)
        
        mp = float(DB_ss["mp"][i])
        dV = float(DB_ss["dV"][i])
        Isp = float(DB_ss["Isp"][i]) 
        # print(mpay, mdry, Isps)

        pred = EUC_AZ.routine_all_linear(mp, dV, Isp)
       
        # md_pred = float(pred.md[0])
        # mprop_pred = float(pred.mprop[0])
            

        ers_i, ers_2 = V_ers(data, pred, "All")
        V_ss_plotter(data, pred, "all {0}".format(i))
        errors.append(ers_2)
    return errors
    
#%%
def Isaji_comparator(L1):
    
    L_isaji = mf.L_isaji(L1.name, L1.mp, L1.md, L1.mprop, L1.mt, L1.dv, L1.Isp, L1.STR+L1.THER, L1.PRPLSN, L1.POW, L1.AVIO, 1, L1.OTH)
    return L_isaji
    # print(L_isaji)
#%%
CC = Isaji_comparator(LM_pred)
#%% #His own inputs from the paper
mp = 4795 #he has a different payload concept: he has 133, but I consider payload to be the whole ascent stage
dV = 2104 #table 6
Isp = 300 #table 2
# LM_pred = EUC_AZ.routine_all_linear(mp, dV, Isp) # the sizing algorithm under investigation 
LM_pred = EUC_AZ.routine_stat_MLR_noloop(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_stat_MLR_iter(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_I_N2O4_MLR_iter(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_Isaji_N2O4_MLR(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_Isaji_N2O4(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_Ramos_N2O4(mp, dV, Isp)
# LM_pred = EUC_AZ.routine_Ramos_N2O4_iter(mp, dV, Isp)




LM_test = mf.L("LM", 5295, 2373, 8780, 16447, dv=2265, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273)

LM_pred_isaji = Isaji_comparator(LM_pred)
LM_test_isaji = Isaji_comparator(mf.L("LM_isaji", 4795, 2217, 8880, 16371, dv=2104, isp=311, STR = 460, PRPLSN = 495, POW = 366, AVIO = 29, THER = 404, OTH = 273))

# V_ss_plotter(LM_test, LM_pred, "Apollo 17")
# # AA = V_ers(LM_test, LM_pred2, "ers")
BB =V_ers_2_isaji(LM_test_isaji, LM_pred_isaji, "ers")

