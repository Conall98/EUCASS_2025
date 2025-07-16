# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 10:59:13 2025

@author: cdepaor
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

import outils as o
from DBs import *
#%%4

test1 = o.routine_all_linear(LM_test.mp, LM_test.md, LM_test.Isp)
o.V_ers_2(LM_test, test1, "test1A")