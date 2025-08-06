# -*- coding: utf-8 -*-
"""
Created on Tue May 13 10:28:52 2025

@author: cdepaor
"""
import numpy as np

#%%
class material():
    def __init__(self, name, density, sigma_max):
        self.name = name
        self.density = density
        self.sigma_max = sigma_max
        
class Fuel():
    def __init__(self, name, lox_density, fuel_density, mixture_ratio, Isp):        
        self.name = name
        self.rho_lox = lox_density
        self.rho_fuel= fuel_density 
        self.MR = mixture_ratio
        self.Isp = Isp
        
class Tank():
    def __init__(self, mass, shape, thickness, test, volume):
        self.mass = mass
        self.shape = shape
        self.thickness = thickness
        self.test = test
        self.volume = volume
        
#%%
F1 = Fuel("N2O2-Aerozine", 1442, 903, 1.9, 311)
F2 = Fuel("LOX/LH2", 1141, 708, 6, 450)
F3 = Fuel("LOX/LCH4", 1141, 657, 3.5, 350)

M1 = material("titanium Ti64", 4540, 880E+6) #asm mat wbe
M2 = material("Aluminium 6061", 2700, 145E+6) #asm mat web
M3 = material("CFRP", 1420, 1260E+6) #matweb
#%%

def tank_geometry(mass, density, shape, constraint=None): # constraint is the max diameter of the tank
# the only reason to have a cyllinder instead of a sphere is if you dont have enough space to make a cylinder
    # print("in tank_geometry mass is: ", mass)
    # print("volume is: ", density)
    V = mass/density
    # print("mass and density", mass, density)
    
    if shape == "sphere":
        r = (3*V/(4*np.pi))**(1/3)
        S = (4)*(np.pi*(r)**2)
        # print("sphere V", V)
        return S, V, r
        
    elif shape == "cylinder":
        r = constraint/2
        h = V/(2*np.pi*r)
        S = 2*np.pi*r*(r+h)
        return S, V, r

    elif shape == "taurus":
        d = constraint
        b_range = np.linspace(0.01, d/2, 2000)
        roots = np.zeros([3, 2])
        root_count = 0
        ys = [0]
        for b in b_range:
            y = b**3 - (d/2)*b**2 + V/(2*np.pi**2)
            if y*ys[len(ys)-1] < 0:
                a = d/2 - b
                roots[root_count, 1] = b
                roots[root_count, 0] = a
                root_count = root_count+1
            ys.append(y)
        for i in range(0,len(roots[:, 0])):
            if roots[i, 0] > roots[i, 1]:
                a = roots[i, 0]
                b = roots[i, 1]
        try:
            a
        except NameError:
            return "Taurus outer diameter constraint too small"
        else:
            S = (4*np.pi**2)*a*b
            r=a
            return S, V, r


#%%
# test_S, tank_V, tank_r = tank_geometry(10, 1000, "taurus", 0.4)

#%% material thickness
def tank_thickness(V, material, shape, pressure, constraint=None):
    sigma_max = material.sigma_max
    p = pressure*1E5
    # print("the type of p:", type(p))
    if shape == "sphere":
        
        r = (3*V/(4*np.pi))**(1/3)
        t = (p*r)/(2*sigma_max)
        return t
    
    elif shape == "cylinder":
        
        r = constraint/2
        t = p*r/sigma_max
        return t
    
    elif shape == "taurus":
        # print("here")
        # print("the type of constraint:", type(constraint))    
        # print(constraint)
        r = constraint/2
        t = p*r/sigma_max
        return t
    
#%%
# test_t = tank_thickness(1, M1, "sphere", 20)

# md = 2217
# mprop = 10292
# mp = 2000
# FT = F1
# TWR = 1.72
# n = 4

# test11 = PRPL(md, mprop, mp, FT, TWR, n)
#%% Tank mass
def tank_mass(tank_surface, tank_thickness, material):
    S = tank_surface
    t = tank_thickness
    rho_mat = material.density
    return S*t*rho_mat
    
#%%
def tank_tester(shape, material, thickness, pressure, radius): #if it is a sphere or cycllinder, radius is the radius
# if the tank is a taurus the radus is the radius of the "pipe" section
    sigma_max = material.sigma_max
    t = thickness
    p = pressure*1E5
    r = radius
    # print("shape", shape)
    if shape == "sphere":
        sigma_actual = (p*r)/(2*t)
        if sigma_actual > sigma_max:
            tank_test = "Failed"
        elif sigma_actual <= sigma_max:
            tank_test = "Passed"
        
    
    if shape == "cylinder":
        sigma_actual = (p*r)/(t)
        if sigma_actual > sigma_max:
            tank_test = "Failed"
        elif sigma_actual <= sigma_max:
            tank_test = "Passed"
        
    
    if shape == "taurus":
        sigma_actual = (p*r)/(t)
        if sigma_actual > sigma_max:
            tank_test = "Failed"
        elif sigma_actual <= sigma_max:
            tank_test = "Passed"
        
    return tank_test
    
#%% Test params


#%% test routine
# def tank_routine(mprop, tank_material, shape, pressure, constraint, density):
#     # print("mprop in tank_routine: ", mprop)
#     test_S, tank_V, tank_r = tank_geometry(mprop, density, shape, constraint)
#     test_t = tank_thickness(tank_V, tank_material, shape, pressure, constraint)
#     test_m = tank_mass(test_S, test_t, M1)
#     test_result = tank_tester(shape, tank_material, test_t, pressure, tank_r)
#     tank = Tank(np.round(test_m, 2), 
#                 shape, 
#                 np.round(test_t, 6), 
#                 test_result, 
#                 np.round(tank_V, 2))
#     return tank
def tank_routine(mprop, tank_material, shape, pressure, constraint, density):
    # print("mprop in tank_routine: ", mprop)
    # print("mprop in tank routine", mprop)
    test_S, tank_V, tank_r = tank_geometry(mprop, density, shape, constraint)
    test_t = tank_thickness(tank_V, tank_material, shape, pressure, constraint)
    test_m = tank_mass(test_S, test_t, M1)
    test_result = tank_tester(shape, tank_material, test_t, pressure, tank_r)
    tank = Tank(np.round(test_m, 2), shape, np.round(test_t, 6), test_result, np.round(tank_V, 2))
    return tank


#%%
# tank = tank_routine(mprop, tank_material, shape, pressure, constraint, density)

#%%
def PRPL(md, 
         mprop, #drymass
         mp, #payload mass
         FT, #FT = Fuel type (class),
         TWR, # TWR is Thrust WEIGHT Requirement 
         tank_material, #duh
         n, #number of engines
         Pressure, #tank pressure in bars
         OX_tank_shape, #shape of the oxidizer tank
         F_tank_shape, #shape of the fuel tank
         P_tank_shape): #shape of the pressurant tank
    # print("mprop", mprop)
    #m_engines
    # Relation - 1: (Ramos 2022)
    mt = md+mprop+mp
    W = mt*1.63 # g_moon
    Trt = TWR*W #Thrust requirement total
    Tr = Trt/n #thrust requirement per engine
    m_eng = ((Tr/9.81))/((6.098E-4)*Tr+13.44) #using the relation for Tr<50kN
    m_engines = m_eng*n
    
    #m_tanks #There will be at least three tanks
    MR = FT.MR
    # print("fuel type: ", FT.name)
    # print("MR is: ", MR)
    m_lox = (mprop*MR)/(1+MR)
    m_fuel = mprop - m_lox
    # print("m_lox compared to m_fuel compared to mprop: ", m_lox, m_fuel, mprop)
    
    rho_lox = 1141 #kg/m3
    v_lox = m_lox*FT.rho_lox
    rho_fuel = FT.rho_fuel
    v_fuel = m_fuel/rho_fuel
    v_lox = m_lox/rho_lox
    v_prop = v_lox+v_fuel
    # print("here", FT.name, FT.MR, v_fuel, m_fuel)
    P = Pressure #bar
    V = v_prop + v_lox
    R = 8.314 #J/mol/K
    T = 293 # 20C for now.
    PMM = 0.004 #pressurant kgs/mol #this is for helium specifically
    m_press = PMM*((P*V)/(R*T))# the mass of helium to pressurize both the oxygen and fuel tank volumes
    
    pressure = P #bars
    
    
    #m_tank_fuel
    shape = F_tank_shape
    constraint = 4 #meters
    density = FT.rho_fuel
    # print("m_fuel in PRPL : ", m_fuel)
    FUEL_TANK = tank_routine(m_fuel, tank_material, shape, pressure, constraint, density)
    # print("m_fuel", m_fuel)
    
    #m_tank_lox # assuming a taurus
    shape = OX_tank_shape
    constraint = 4 #meters
    density = FT.rho_lox
    LOX_TANK = tank_routine(m_lox, tank_material, shape, pressure, constraint, density)
    # print("m_lox", m_lox)
    
    #m_tank_pressurant
    shape = P_tank_shape
    constraint = 4 #meters
    density = FT.rho_fuel
    PRESSURANT_TANK = tank_routine(m_press, tank_material, shape, 50, constraint, density)
    # print("m_press", m_press)
    
    
    m_bare = LOX_TANK.mass + FUEL_TANK.mass + PRESSURANT_TANK.mass  #see the other tank mass functions you made
    #m_weld  #not included
    #m_IO #not included
    #m_sa #not included
    #m_sep #not included
    #m_tank = sum(m_sep, m_sa, m_IO, m_weld, m_bare)
    # m_tank = sum(m_bare)
    
    #m_misc
    #Relation - 1 (Ramos 2022) #15% of the engine mass. Why?
    return m_bare + m_engines, m_bare, m_engines



#%%

#%%

# md = 2217
# mprop = 10292
# mp = 2000
# FT = F2
# TWR = 1.72
# n = 4
# tank_material = M1

# test11 = PRPL(md, mprop, mp, FT, TWR, tank_material, n)














