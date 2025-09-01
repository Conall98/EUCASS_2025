# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 18:01:05 2025

@author: cdepaor
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd

DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx") 
# Example data (replace with your own x, y)
x = DB["mp"]
y = DB["md"]

# Define the model
def model(x, a, b, c):
    return a * x**b + c

ref=0.5
# Repeated fits with random subsampling
def repeated_fits(x, y, n_repeats=1000, remove_fraction = ref):
    fits = []
    n_remove = int(len(x) * remove_fraction)

    for i in range(n_repeats):
        mask = np.ones(len(x), dtype=bool)
        mask[np.random.choice(len(x), n_remove, replace=False)] = False
        x_sub, y_sub = x[mask], y[mask]

        # Now p0 has 3 parameters: a, b, c
        popt, _ = curve_fit(model, x_sub, y_sub, p0=[1, 1, 1])
        fits.append(popt)

    return np.array(fits)

# Run repeated fits
fits = repeated_fits(x, y, n_repeats=10, remove_fraction = ref)

# Common x grid
x_fit = np.linspace(min(x), max(x), 200)

# Evaluate all fits
all_curves = np.array([model(x_fit, *params) for params in fits])

# Mean and std across fits
mean_curve = np.mean(all_curves, axis=0)
std_curve = np.std(all_curves, axis=0)

# Plot data points (red dots)
plt.scatter(x, y, color="red", s=30, label="Data")

# Plot mean curve (black)
plt.plot(x_fit, mean_curve, 'k-', linewidth=2.5, label="Mean fit")

# Plot std deviation bands as grey dashed lines
alphas = [0.8, 0.5, 0.3]  # fading for ±1σ, ±2σ, ±3σ
for k, alpha in zip([1, 2, 3], alphas):
    plt.plot(x_fit, mean_curve + k*std_curve, '--', color="grey", alpha=alpha, label = "$\pm{}\sigma$".format(k))  # widened bands
    plt.plot(x_fit, mean_curve - k*std_curve, '--', color="grey", alpha=alpha)

plt.xlabel("Payload mass $m_p$ [kg]")
plt.ylabel("Dry mass $m_d$ [kg]")
plt.legend()
plt.title("Mean fit ($y = ax^b + c$) with ±σ bands")
plt.show()