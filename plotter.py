# -*- coding: utf-8 -*-
"""
Created on Tue Jun 24 17:00:53 2025
TRENDLINE PLOTTER
@author: cdepaor
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%%
def trend(x, a, b):
    return x*a + b
    

def scatter_plotter(x, y, title, a, b, R):
    x_run = np.linspace(0, max(x), 100)
    y_run = trend(x_run, a, b)

    plt.figure()
    plt.scatter(x, y, label="data", color="black")
    plt.plot(x_run, y_run, label="y = {0}x + {1}".format(a, b), color="red", linestyle="--")
    plt.plot(0, 0, color="white", label="$R^2$ = {}".format(R))


    plt.minorticks_on()

    plt.grid(True, which="major", linewidth=0.8)
    plt.grid(True, which="minor", linestyle=":", linewidth=0.6)

    plt.xlabel("Dry mass [kg]")
    plt.ylabel("Avionics subsystem mass [kg]")
    plt.title("Subsystem Sizing Rules")
    plt.legend()
    plt.tight_layout()
    plt.show()

    
#%%
def histogrammer(x, binwidth, title, xlabel, ylabel): 
    fs = 14
    gs = 18
    plt.figure()
    plt.rcParams["font.family"] = "CMU Sans Serif"
    plt.title(title, fontsize = gs)
    plt.xlabel(xlabel, fontsize = fs)
    plt.ylabel(ylabel, fontsize = fs)
    plt.hist(x, bins=binwidth)

#%%
    












