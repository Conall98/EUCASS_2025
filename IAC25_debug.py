# -*- coding: utf-8 -*-
"""
Created on Mon Sep  1 15:29:50 2025

@author: cdepaor
"""

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
from IAC25_outils import *
from IAC25_DBs import *
from IAC25_main import *

#%%
n = 1000
mp_samples = np.random.normal(2000, 1, n)      # Monte Carlo for mp
dv_samples = np.random.normal(2500, 25, n)   # Monte Carlo for dv
isp_samples = np.random.normal(350, 5, n)    # Monte Carlo for Isp

lander_mc = monte_carlo_IAC(n, mp_samples, dv_samples, isp_samples, IAC_sizing_algorithm, model)
#%% diagnose gaps
diagnose_gaps(
    lander_mc,
    mp_samples,
    dv_samples,
    isp_samples,
    prop="md",
    bins=50
)
#%%
diagnose_gaps_with_initial_guess_highlight(
    lander_mc,
    mp_samples,
    dv_samples,
    isp_samples,
    prop="md",
    bins=50
)