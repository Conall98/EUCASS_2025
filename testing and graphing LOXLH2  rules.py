# -*- coding: utf-8 -*-
"""
Created on Tue May 20 11:51:13 2025

@author: cdepaor
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:28:52 2025

@author: cdepaor
"""
import numpy as np
import EUCASS_Subsystem_Subroutines as ESR
import tank_sizing_subroutine as tn
import matplotlib.pyplot as plt
import mat_fuel_db as mf
import EUCASS_LOXLH2 as EUC_LH2
import EUCASS_N2O4AZ as EUC_AZ
import pandas as pd
#%% Routine_A test
test_01 = EUC_LH2.routine_A(2000, 5000, 311)
vars(test_01)
test_01.test()
#%%
test_02 = EUC_LH2.routine_Ramos_Cryo(2000, 5000, 311)
vars(test_02)
test_02.test()

#%%
test_03 = EUC_LH2.routine_Isaji_cryo(2000, 5000, 311)
vars(test_03)
test_03.test()    

#%%
test_04 = EUC_LH2.routine_Ramos_Cryo_2(2000, 5000, 311)
vars(test_04)
test_04.test()    
#%%
test_05 = EUC_AZ.routine_Ramos_N2O4_2(2000, 5000, 301)
vars(test_05)
test_05.test()    
#%%
DB = pd.read_excel(r"C:\Users\cdepaor2\Desktop\PhD\01_Projects\EUCASS 2025\EUCASS_code\Lander DB 251 redux (alt).xlsx")

#%%
x = DB["mp"]
y = DB["md"]
dv = DB["dV"]
Isp = DB["1 stage Isp"]
x_run = np.linspace(0, 5000, 100)
y_run = []
for i in range(0, 100):
    y_run.append(ESR.f1(x_run[i]))

y_pred = []
y_pred2 = []
y_pred3 = []
y_pred4 = []
x_star = []
for i in range(0, len(x)):
    if x[i]>2:
        y_pred.append(EUC_LH2.routine_Ramos_Cryo(x[i], dv[i], Isp[i]).md)
        y_pred2.append(EUC_LH2.routine_Isaji_cryo(x[i], dv[i], Isp[i]).md)
        y_pred3.append(EUC_LH2.routine_Ramos_Cryo_2(x[i], dv[i], Isp[i]).md)
        y_pred4.append(EUC_LH2.routine_Isaji_cryo_2(x[i], dv[i], Isp[i]).md)
        x_star.append(x[i])
        

plt.figure()
plt.scatter(x, y, label = "data", color = "blue")
plt.plot(x_run, y_run, label = "trendline", color = "black", linestyle = "--")

plt.scatter(x_star, y_pred, label = "Ramos prediction", color = "orange")
plt.scatter(x_star, y_pred2, label = "Isaji prediction", color = "red")

plt.scatter(x_star, y_pred4, label = "Isaji2 prediction", color = "violet")
plt.scatter(x_star, y_pred3, label = "Ramos2 prediction", color = "pink")


plt.grid()
plt.xlabel("payload mass [kg]")
plt.ylabel("dry mass [kg]")
plt.title("New Algos predicting the data")
plt.legend()
# plt.scatter(x, y_pred)

#%%




