# -*- coding: utf-8 -*-
"""
Created on Sat May 31 16:49:09 2025

@author: conal
"""

import numpy as np
import EUCASS_Subsystem_Subroutines as ESR
import tank_sizing_subroutine as tn
import matplotlib.pyplot as plt
import mat_fuel_db as mf
import EUCASS_LOXLH2 as EUC_LH2
import EUCASS_N2O4AZ as EUC_AZ
import pandas as pd
from mulitple_regression import *
#%% inputs
mp = 5295
Isp = 311
dV = 2265
#%% validation dict
LM =              {"mp": mp,
                   "md": 2373,
                   "mprop": 8780,
                   "str": 460,
                   "prpln": 495,
                   "pow": 366,
                   "avio": 29,
                   "therm": 404,
                   "oth": 273}
#%% sizing main

test_VL = EUC_AZ.routine_Ramos_N2O4(mp, dV, Isp)

VL =  {"mp": np.round(test_VL.mp, 0),
       "md": np.round(test_VL.md[0], 0),
       "mprop": np.round(test_VL.mprop[0], 0),
       "str": np.round(test_VL.STR, 0),
       "prpln": np.round(test_VL.PRPLSN, 0),
       "pow": np.round(test_VL.POW, 0),
       "avio": np.round(test_VL.AVIO, 0),
       "therm": np.round(test_VL.THER, 0),
       "oth": np.round(test_VL.OTH, 0)}


#%% Comparison

comp = {"mp_er": (VL["mp"] - LM["mp"])/LM["mp"],
        "md_er":  (VL["md"] - LM["md"])/LM["md"],
        "mprop_er":  (VL["mprop"] - LM["mprop"])/LM["mprop"],
        "str_er":  (VL["str"] - LM["str"])/LM["str"],
        "prpln_er":  (VL["prpln"] - LM["prpln"])/LM["prpln"],
        "pow_er":  (VL["pow"] - LM["pow"])/LM["pow"],
        "avio_er":  (VL["avio"] - LM["avio"])/LM["avio"],
        "therm_er":  (VL["therm"] - LM["therm"])/LM["therm"],
        "oth_er":  (VL["oth"] - LM["oth"])/LM["oth"]}





