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
DB_ss = pd.read_excel(r"subsystems database.xlsx")
x1 = DB_ss["mp"]
x2 = DB_ss["md"]
x3 = DB_ss["mprop"]
x4 = DB_ss["mt"]

y1 = DB_ss["Structure"]
y2 = DB_ss["Propulsion"]
y3 = DB_ss["Power"]
y4 = DB_ss["Avionics"]
y5 = DB_ss["Thermal Protection"]
y6 = DB_ss["Other"]


# scatter_plotter(x3, y2, "title", 0.088, 0, 0.8893)
scatter_plotter(x2, y4, "title", 0.0774, 0, 0.833)














