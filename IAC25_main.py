# -*- coding: utf-8 -*-
"""
Created on Wed Aug 20 14:30:31 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from IAC25_outils import *
from IAC25_DBs import *
#%%

mp = 2000
dv = 2500
Isp = 350
# model = linear_estimations
# model = polynomial_estimations
# model = powerlaw_estimations
model = mixed_estimation
lander = IAC_sizing_algorithm(mp, dv, Isp, model)


#%% Testing
# X = np.array([1, 2, 3, 4, 5])
# STR_MLR(X)

mean_errors, repredictions, skipped_indices = EVAL(IAC_sizing_algorithm, DB_ss, model)


#%%
# List of models to evaluate
models = [
    ("linear", linear_estimations),
    ("polynomial", polynomial_estimations),
    ("powerlaw", powerlaw_estimations),
    ("mixed", mixed_estimation),
]

all_mean_errors = []
skipped_summary = {}

for name, model in models:
    mean_errors, repredictions, skipped_indices = EVAL(IAC_sizing_algorithm, DB_ss, model)
    
    all_mean_errors.append(mean_errors)
    skipped_summary[name] = skipped_indices
    
    if skipped_indices:
        print(f"[{name}] Skipped {len(skipped_indices)} landers at indices: {skipped_indices}")

# Convert to 2D numpy array
all_mean_errors = np.vstack(all_mean_errors).T

print("\nShape of all_mean_errors:", all_mean_errors.shape)
print("Summary of skipped indices:", skipped_summary)



#%% Monte-Carlo Class 1 uncertainties
n = 1000
mp_samples = np.random.normal(2000, 0, n)      # Monte Carlo for mp
dv_samples = np.random.normal(2500, 25, n)   # Monte Carlo for dv
isp_samples = np.random.normal(350, 5, n)    # Monte Carlo for Isp

lander_mc = monte_carlo_IAC(n, mp_samples, dv_samples, isp_samples, IAC_sizing_algorithm, model)

print(lander_mc.mp[:5])   # first 5 Monte Carlo mp values
print(lander_mc.dv[:5])   # first 5 Monte Carlo dv values
print(lander_mc.Isp[:5])  # first 5 Monte Carlo isp values

#%%
# plot_lander_histograms(lander_mc, bins=40)
plot_lander_scurves_with_percentiles(lander_mc)
