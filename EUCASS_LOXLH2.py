# -*- coding: utf-8 -*-
"""
Created on Mon May 12 17:49:49 2025

@author: cdepaor
"""

import numpy as np
import EUCASS_Subsystem_Subroutines as ESR
import tank_sizing_subroutine as tn
import matplotlib.pyplot as plt
import mat_fuel_db as mf

#%%
#you have to also use your multiple linear regression technique
# and compare to double correlation technique

def routine_A(mp, dv, Isp): #the most basic subsystem looper routine
    md_0 = ESR.f1(mp)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = md_0
    mprop_i = mprop_0
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    lander.STR = ESR.STR(md_i)
    lander.PRPLSN = ESR.PROP(md_i)
    lander.POW = ESR.POW(md_i)
    lander.AVIO = ESR.AVIO(md_i)
    lander.THER = ESR.THER(md_i)
    lander.OTH = ESR.OTH(md_i)
    return lander
#%%
def routine_Ramos_Cryo(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F2 #loxlh2
    Isp = FT.Isp
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    while er > tol:
        

        b = lander.PRPLSN = ESR.PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        
        a = lander.STR = ESR.STR_mp(mp)
        c = lander.POW = ESR.POW(md_i[i])
        d = lander.AVIO = ESR.AVIO(md_i[i])
        e = lander.THER = ESR.THER(mt_0)
        f = lander.OTH = ESR.OTH(mp)
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%
def routine_Ramos_Cryo_lessloop(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F2 #loxlh2
    Isp = FT.Isp
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    a = lander.STR = ESR.STR_mp(mp)
    
    f = lander.OTH = ESR.OTH(mp)
    
    while er > tol:
        

        b = lander.PRPLSN = ESR.PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        c = lander.POW = ESR.POW(md_i[i])
        d = lander.AVIO = ESR.AVIO(md_i[i])
        e = lander.THER = ESR.THER(lander.mt)
        
        print(i)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        # print("{0}/{1}".format(md_i1, md_i[i]))
        # er = 1 - md_i1/md_i[i]
        er = 1 - md_i[i]/md_i1
        # print(np.round(md_0, 3), np.round(md_i[i], 3), np.round(md_i1, 3))
        # print(md_i1)
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_Ramos_Cryo_CH4(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F3
    Isp = FT.Isp
    # T = mf.F3 #loxlch4
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    
    
    while er > tol:
        

        b = lander.PRPLSN = ESR.PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        a = lander.STR = ESR.STR_mp(mp)
        c = lander.POW = ESR.POW(md_i[i])
        d = lander.AVIO = ESR.AVIO(md_i[i])
        e = lander.THER = ESR.THER(lander.mt)
        f = lander.OTH = ESR.OTH(mp)

        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        print(i)
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        # print(lander.mprop)
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_Isaji_cryo(mp, dv, Isp): #Uses Isaji cryogenic to size propulsion subsystem
    md_0 = ESR.f1(mp)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F2 #loxlh2
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    while er > tol:
        lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 

        b = lander.PRPLSN = ESR.PRPL_Isaji_cryo(md_i[i], mp)
        
        a = lander.STR = ESR.STR_mp(mp)
        c = lander.POW = ESR.POW(md_i[i])
        d = lander.AVIO = ESR.AVIO(md_i[i])
        e = lander.THER = ESR.THER(mt_0)
        f = lander.OTH = ESR.OTH(mp)
        
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    
    return lander


# #%% Routine_A test
# test_01 = routine_A(2000, 5000, 311)
# vars(test_01)
# test_01.test()
# #%%
# test_02 = routine_Ramos_Cryo(2000, 5000, 311)
# vars(test_02)
# test_02.test()

# #%%
# test_03 = routine_Isaji_cryo(2000, 5000, 311)
# vars(test_03)
# test_03.test()    
#%%
def routine_Isaji_cryo_2(mp, dv, Isp): #Uses Isaji cryogenic to size propulsion subsystem
    md_0 = ESR.f1(mp)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    i = 0
    tol = 0.01
    er = 1
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp) 
    a = lander.STR = ESR.STR(md_i[i])
    c = lander.POW = ESR.POW(md_i[i])
    d = lander.AVIO = ESR.AVIO(md_i[i])
    e = lander.THER = ESR.THER(md_i[i])
    f = lander.OTH = ESR.OTH(md_i[i])
    
    while er > tol: 
        b = lander.PRPLSN = ESR.PRPL_Isaji_cryo(md_i[i], mp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    
    return lander

#%%
def routine_Ramos_Cryo_2(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F2 #loxlh2
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 2 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    a = lander.STR = ESR.STR(md_i[i])
    c = lander.POW = ESR.POW(md_i[i])
    d = lander.AVIO = ESR.AVIO(md_i[i])
    e = lander.THER = ESR.THER(md_i[i])
    f = lander.OTH = ESR.OTH(md_i[i])
    
    while er > tol:
        b = lander.PRPLSN = ESR.PRPL_Ramos(md_i[i], mprop_i[i], mp, FT, TWR, tank_material, n, Pressure, OX_tank_shape, F_tank_shape, P_tank_shape)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        lander.md = md_i[-1:]
        lander.mprop = mprop_i[-1:]
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

