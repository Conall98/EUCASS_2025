# -*- coding: utf-8 -*-
"""
Created on Thu May 22 14:49:26 2025

@author: cdepaor
"""


import numpy as np
import EUCASS_Subsystem_Subroutines as ESR
import tank_sizing_subroutine as tn
import matplotlib.pyplot as plt
import mat_fuel_db as mf

    
#%%

def routine_Ramos_N2O4(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 20 #bars same as space shuttle external tank
    OX_tank_shape = "sphere"
    F_tank_shape = "sphere"
    P_tank_shape = "sphere"
    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    a = lander.STR = ESR.STR_mp(mp)
    c = lander.POW = ESR.POW(md_i[i])
    d = lander.AVIO = ESR.AVIO(md_i[i])
    e = lander.THER = ESR.THER(mt_0)
    f = lander.OTH = ESR.OTH(mp)
    
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
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_Ramos_N2O4_iter(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine
    TWR = 1.72 # same as Apollo
    tank_material = mf.M1 #titanium
    n = 4
    Pressure = 20 #bars same as space shuttle external tank
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
        # print(i)
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

def routine_Isaji_N2O4(mp, dv, Isp): #uses the ramos propulsion sizing routine
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    a = lander.STR = ESR.STR_mp(mp)
    c = lander.POW = ESR.POW(md_i[i])
    d = lander.AVIO = ESR.AVIO(md_i[i])
    e = lander.THER = ESR.THER(mt_0)
    f = lander.OTH = ESR.OTH(mp)
    
    while er > tol:
        b = lander.PRPLSN = ESR.PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        
        lander.md = md_i[-1:]
        
        lander.mprop = mprop_i[-1:]
        
        # lander.mt = lander.md + lander.mprop + lander.mp
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        # print("here: ", lander.mt)
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander
#%%

def routine_Isaji_N2O4_MLR(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    X = np.array([lander.md, lander.mp, lander.mprop, lander.Isp, lander.dv]).reshape(1,-1)
    a = lander.STR = ESR.STR2(X)
    c = lander.POW = ESR.POW2(X)
    d = lander.AVIO = ESR.AVIO2(X)
    e = lander.THER = ESR.THER2(X)
    f = lander.OTH = ESR.OTH2(X)
    
    while er > tol:
        b = lander.PRPLSN = ESR.PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        # print(i)
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
def routine_I_N2O4_MLR_iter(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    while er > tol:
        b = lander.PRPLSN = ESR.PRPL_Isaji_storable(md_i[i], mp, FT, Isp)
        X = np.array([lander.md, mp, lander.mprop, Isp, dv]).reshape(1,-1)
        a = lander.STR = ESR.STR2(X)
        c = lander.POW = ESR.POW2(X)
        d = lander.AVIO = ESR.AVIO2(X)
        e = lander.THER = ESR.THER2(X)
        f = lander.OTH = ESR.OTH2(X)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        
        lander.md = float(md_i[-1:][0])

        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_stat_MLR_iter(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = [md_0]
    mprop_i = [mprop_0]
    
    FT = mf.F1 #N2O4-Aerozine    
    i = 0
    tol = 0.01
    er = 1
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    while er > tol:
        X = np.array([lander.md, mp, lander.mprop, Isp, dv]).reshape(1,-1)
        a = lander.STR = ESR.STR2(X)
        b = lander.PRPLSN = ESR.PROP2(X)
        c = lander.POW = ESR.POW2(X)
        d = lander.AVIO = ESR.AVIO2(X)
        e = lander.THER = ESR.THER2(X)
        f = lander.OTH = ESR.OTH2(X)
        md_i1 = sum([a, b, c, d, e, f])
        mprop_i1 = ESR.f2(mp, md_i1, dv, Isp)
        # print(i)
        md_i.append(md_i1)
        mprop_i.append(mprop_i1)
        er = 1 - md_i1/md_i[i]
        i = i+1
        
        lander.md = float(md_i[-1:][0])

        lander.mprop = float(mprop_i[-1:][0])
        
        lander.mt = float(np.array(md_i[-1:])) + float(np.array(mprop_i[-1:])) + float(np.array(mp))
        
        if i>100:
            print("divergence")
            break
    # print("iterations: ", i)
    return lander

#%%
def routine_stat_MLR_noloop(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = md_0
    mprop_i = mprop_0
    
    FT = mf.F1 #N2O4-Aerozine    
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    X = np.array([lander.md, mp, lander.mprop, Isp, dv]).reshape(1,-1)
    a = lander.STR = ESR.STR2(X)
    b = lander.PRPLSN = ESR.PROP2(X)
    c = lander.POW = ESR.POW2(X)
    d = lander.AVIO = ESR.AVIO2(X)
    e = lander.THER = ESR.THER2(X)
    f = lander.OTH = ESR.OTH2(X)
    md_i1 = sum([a, b, c, d, e, f])
    
    correction_scale = md_i1/md_0
    
    lander.STR = a/(md_i1/md_0)
    lander.PRPLSN = b/(md_i1/md_0)
    lander.POW = c/(md_i1/md_0)
    lander.AVIO = d/(md_i1/md_0)
    lander.THER = e/(md_i1/md_0)
    lander.OTH = f/(md_i1/md_0)
    
    # print("iterations: ", i)
    return lander


#%%
def routine_all_linear(mp, dv, Isp):
    # print("mp in routuine_Ramos_Cryo: ", mp)    
    md_0 = ESR.f1(mp)
    # print("md_0 in routuine_Ramos_Cryo: ", md_0)
    mprop_0 = ESR.f2(mp, md_0, dv, Isp)
    # print("mprop_0 in routuine_Ramos_Cryo: ", mprop_0)
    mt_0 = mp + md_0 + mprop_0
        
    md_i = md_0
    mprop_i = mprop_0
    
    FT = mf.F1 #N2O4-Aerozine    
    
    lander = mf.L("Test lander 1", mp, md_0, mprop_0, mt_0, dv, Isp)
    
    
    
    b = lander.PRPLSN = ESR.PROP(lander.md)
    
    
    a = lander.STR = ESR.STR_mp(mp)
    c = lander.POW = ESR.POW(md_i)
    d = lander.AVIO = ESR.AVIO(md_i)
    e = lander.THER = ESR.THER(mt_0)
    f = lander.OTH = ESR.OTH(mp)
    
    md_i1 = sum([a, b, c, d, e, f])
    
    correction_scale = md_i1/md_0
    # print(correction_scale)
    
    # lander.STR = a*(md_i1/md_0)
    # lander.PRPLSN = b*(md_i1/md_0)
    # lander.POW = c*(md_i1/md_0)
    # lander.AVIO = d*(md_i1/md_0)
    # lander.THER = e*(md_i1/md_0)
    # lander.OTH = f*(md_i1/md_0)
    
    # print("iterations: ", i)
    return lander
#%% TEST

test = routine_all_linear(5295, 2265, 311)