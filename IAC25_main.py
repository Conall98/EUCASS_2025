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
#%%

mp = 2000
dv = 2500
Isp = 350

lander = IAC_sizing_algorithm(mp, dv, Isp, MPow_ss_models)