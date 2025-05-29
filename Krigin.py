# -*- coding: utf-8 -*-
"""
Created on Mon May  5 14:29:40 2025

@author: cdepaor
"""

import pykrige as pk
import pandas as pd
import numpy as np
DB = pd.read_excel(r"Lander DB 251 redux (alt).xlsx")
import matplotlib.pyplot as plt
# import mulitple_regression as mr
#%%
x = DB["mp"]
y =  DB["mprop"]
z = DB["md"]

uk = pk.ok.OrdinaryKriging(x, y, z, variogram_model="spherical")

x_pred = np.linspace(0, max(x), 100)
y_pred = np.linspace(0, max(y), 100)
# x_pred = x["mp"]
z_pred, z_std = uk.execute("grid", x_pred, np.array([0.0]))

z_pred = np.squeeze(z_pred)
z_std = np.squeeze(z_std)


#%%
# Create a scatter plot with color representing z
# Create a heatmap with correct aspect ratio
plt.figure()

# Predict over the grid
gridx = x_pred
gridy = y_pred
z_pred, z_std = uk.execute("grid", gridx, gridy)

# Show interpolated background
plt.imshow(
    z_pred,
    extent=(min(x)-100, max(x)+100, min(y)-100, max(y)+100),
    origin='lower',
    aspect='auto',  # Use 'equal' if you want 1:1 scaling
    cmap='viridis'
)

# Overlay scatter points
plt.scatter(x, y, color = "black")

# Labels and colorbar
plt.xlabel('payload mass')
plt.ylabel('mprop')
plt.colorbar(label='dry mass')

# Optional: adjust layout
plt.tight_layout()
plt.show()


#%% uncertainty
plt.figure()

# Predict over the grid
gridx = x_pred
gridy = y_pred
z_pred, z_std = uk.execute("grid", gridx, gridy)

# Show interpolated background
plt.imshow(
    z_std,
    extent=(min(x)-100, max(x)+100, min(y)-100, max(y)+100),
    origin='lower',
    aspect='auto',  # Use 'equal' if you want 1:1 scaling
    cmap='viridis'
)

# Overlay scatter points
plt.scatter(x, y, color = "black")

# Labels and colorbar
plt.xlabel('payload mass')
plt.ylabel('mprop')
plt.colorbar(label='dry mass standard deviation')
plt.title("Uncertainty of Dry Mass parameter Estimation")

# Optional: adjust layout
plt.tight_layout()
plt.show()





#%%
plt.figure()
plt.plot(x_pred, z_pred, label = "predicted values", color = "red", linestyle = "--")
plt.scatter(x, z, label = "input data", color = "black")
plt.fill_between(x_pred, 
                 z_pred - 3*z_std,
                 z_pred + 3*z_std,
                 alpha = 0.3, #transperancy
                 label = "confidence interval")
uk.display_variogram_model()

#%%