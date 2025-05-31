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
from mulitple_regression import *
#%% Routine_A test


#%%
DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
ssDB = pd.read_excel(r"subsystems database.xlsx")

#%% subsystem predictions
X = np.array([ssDB["md"], ssDB["mp"], ssDB["mprop"], ssDB["dV"], ssDB["Isp"]]).transpose()
y = np.array([ssDB["Structure"], ssDB["Propulsion"], ssDB["Power"], ssDB["Avionics"], ssDB["Thermal Protection"], ssDB["Other"]]).transpose()
#%%
def modeler(X, y, i):
    model = linear_model.LinearRegression()
    model.fit(X, y[:, i])
    R2 = model.score(X, y[:, i])
    coefs = model.coef_
    intercept =  model.intercept_
    preds = model.predict([X[0]])
    manual_pred = np.dot(coefs, X[0]) + model.intercept_
    # print("predicted y", preds)
    # print("manual_preds", manual_pred)
    # print("actual y", y[0, i])
    # print(preds == manual_pred)
    return coefs, intercept, R2

#%%
ss_models = np.zeros([6, 6]) #six row(predictions), five coefficients + one intercept for each pred
Rs = []
for i in range(0, 6):
    coefs, inter, r = modeler(X, y, i)
    ss_models[i, 0:5] = coefs
    ss_models[i, 5:6] = inter
    Rs.append(r)






