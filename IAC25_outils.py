# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:31:27 2025

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from IAC25_DBs import *

#%% Formatting Functions

def Isaji_converter(L1): #converts ORLA landers into Isaji Landers
    L = L_isaji(L1.name, L1.mp, L1.md, L1.mprop, L1.mt, L1.dv, L1.Isp, L1.STR+L1.THER, L1.PRPLSN, L1.POW, L1.AVIO, 1, L1.OTH)
    return L
    # print(L_isaji)
    
def DB_2_class(DB, i):
    mt = DB["md"][i] + DB["mprop"][i] + DB["mp"][i]
    lander_data = L(DB["Lander"][i], DB["mp"][i], DB["md"][i], DB["mprop"][i], mt, DB["dV"][i], DB["Isp"][i], 
                       DB["Structure"][i],DB["Propulsion"][i],DB["Power"][i],DB["Avionics"][i],DB["Thermal Protection"][i],DB["Other"][i])
    
    return lander_data  

#%% The subsystem sizing rules


#%% The sizing Algorithm
def IAC_sizing_algorithm(mp, dv, Isp, MPowR_model):
    """
    Shamelessly uses a blend of eerything such that the global errors are minimised
    Choose it from main
    
    """
    DBss = DB_ss
    md_0 = f1(mp)
    
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = f2(mp, md_0, dv, Isp)
    
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X_data = np.array([DBss["md"], DBss["mp"], DBss["mprop"], DBss["dV"], DBss["Isp"]]).transpose()
    y_data = np.array([DBss["Structure"], DBss["Propulsion"], DBss["Power"], DBss["Avionics"], DBss["Thermal Protection"], DBss["Other"]]).transpose()
    store = []
    
    acf = 0.01
    STR_model = MPowR_model[0]
    # print("STR_model", STR_model)
    PRPL_model = MPowR_model[1]
    # print("PRPL_model", PRPL_model)
    POW_model = MPowR_model[2]
    # # print("POW_model", PRPL_model)
    AVIO_model = MPowR_model[3]
    # # print("AVIO_model", AVIO_model)
    THER_model = MPowR_model[4]
    # # print("THER_model", THER_model)
    OTH_model = MPowR_model[5]
    # print("OTH_model", OTH_model)
    while er > tol:

        
        new_input = np.array([[lander.mt, lander.mp, lander.mprop, lander.dv, lander.Isp]])
        a = lander.STR = func(new_input, *STR_model)
        # b=c=d=e=f=1
        b = lander.PRPLSN =  func(new_input, *PRPL_model)
        c = lander.POW = func(new_input, *POW_model)
        d = lander.AVIO = func(new_input, *AVIO_model)
        e = lander.THER = func(new_input, *THER_model)
        f = lander.OTH = func(new_input, *OTH_model)
        
        # print("STR", a,"PRPL",  b,"POW",  c,"AVIO",  d,"THER",  e,"OTH",  f)
        # print("STR", a)
        # 
        md_i1 = sum([a, b, c, d, e, f])*acf
        # print("uncorrected md: ", sum([a, b, c, d, e, f]))
        # print("corrected md  : ", sum([a, b, c, d, e, f])*acf)
        mprop_i1 = f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        
        lander.md = (float(md_i[-1:][0]))
    
        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        er = abs(1 - md_i1/md_i[i])
        # acf = md_i[i]/(md_i1) #the last iter over this iter
        
        # print("acf:                      ", acf)
        print("er:                       ", er)
        print("initial md guess:         ", md_0)
        print("md_i from the lase iter:  ", md_i[i])
        print("md from this iter:        ", md_i1)
        # # print("STR:                      ", a)
        # # store.append(a)
        print("iter:                     ", i)
        i=i+1
        if i>50:
            print("divergence, i=", i)
            break    
    
    return lander    
    