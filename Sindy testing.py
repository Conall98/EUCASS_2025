# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:12:22 2025

@author: cdepaor
"""
import numpy as np
import pysindy as ps
import pandas as pd
import matplotlib.pyplot as plt
#%%
DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")

#%%
#mp is the stand-in for time
mp = np.array(DB["mp"])
x1 = np.array(DB["md"])
x2 = np.array(DB["mprop"])
X = np.stack((x1, x2), axis=-1)

model = ps.SINDy(feature_names=["x1", "x2"])
model.fit(X, t=mp)
model.print()
#%%
x_model = model.predict(X)
#%%
plt.figure()
plt.scatter(mp, x_model[:,0], color="orange", label = "prediction")
plt.scatter(mp, x1, color="blue", label = "data")

Y_preds = x_model[:,0]
y = x1
diffs = []
for i in range(0, len(DB["md"])):
    diff_i = abs(Y_preds[i] - y[i])
    diffs.append(diff_i)
    # print(errs[i])
diffs_a = np.array(diffs)
RMS = np.round(np.sqrt(np.mean(diffs_a**2)), 2)
NRMS = np.round(RMS/np.mean(y), 2)*100
plt.xlabel("payload mass [kg]")
plt.ylabel("dry mass [kg]")

plt.scatter(0, 0, color = "white", label = "NRMS = {}%".format(np.round(NRMS)))
plt.title("SINDy testing \n X = [mp], Y = [md]")

plt.legend()


#%%
t = np.linspace(0, 1, 100)
x = 3 * np.exp(-2 * t)
y = 0.5 * np.exp(t)
X = np.stack((x, y), axis=-1)  # First column is x, second is y

model = ps.SINDy(feature_names=["x", "y"])
model.fit(X, t=t)
model.print()

