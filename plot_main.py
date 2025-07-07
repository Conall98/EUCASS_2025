# -*- coding: utf-8 -*-
"""
Created on Fri Jun 27 11:51:38 2025

@author: cdepaor
"""

from plotter import *
import pandas as pd
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
#%%
x  = pd.read_excel(r"lander year list.xlsx", sheet_name = "Feuil2")
binwidth = [1975, 1980, 19]
xlabel = "Timeline"
ylabel = "number of spacecraft"
title = "Lunar Landers Launched Towards the Moon"
histogrammer(x, binwidth, title, xlabel, ylabel)