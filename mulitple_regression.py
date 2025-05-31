# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 14:19:22 2025

@author: cdepaor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from mpl_toolkits import mplot3d as plt3d

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

#import the dataset
DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")

def plotter(x, y, DB):
    plt.figure()
    plt.title("multiple linear regression \n X = [mp, mprop, Isp, dV], Y = [md]")
    plt.xlabel("payload mass [kg]")
    plt.ylabel("dry mass [kg]")
    plt.grid()
    plt.scatter(DB["mp"], DB["md"], color = "blue", label = "data")
    Y_preds = []
    regr = linear_model.LinearRegression()
    regr.fit(x, y)
    for i in range(0, len(DB["mp"])):
        Y_preds.append(regr.predict([x[i]]))
        
    plt.scatter(DB["mp"], Y_preds, color = "orange", label = "prediction")
    diffs = []
    for i in range(0, len(DB["md"])):
        diff_i = abs(Y_preds[i] - y[i])
        diffs.append(diff_i)
        # print(errs[i])
    diffs_a = np.array(diffs)
    RMS = np.round(np.sqrt(np.mean(diffs_a**2)), 2)
    NRMS = np.round(RMS/np.mean(y), 2)*100
    
    plt.scatter(0, 0, color = "white", label = "NRMS = {}%".format(NRMS))
    
    plt.legend()
    

    # ax = plt.figure().add_subplot(projection = "3d")
    # a = DB["mp"]
    # b = DB["mprop"]
    # c = DB["md"]
    # ax.scatter(a, b, c)
    return RMS
#%%
X = np.array([DB["mp"], DB["mprop"], DB["1 stage Isp"], DB["dV"]]).transpose()
y =  DB["md"]
# ME = plotter(X, y, DB)
# #%%


regr = linear_model.LinearRegression()
regr.fit(X, y)

md_pred = regr.predict([X])

