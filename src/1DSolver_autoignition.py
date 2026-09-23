'''

# 1D COMPRESSIBLE REACTING-FLOW SOLVER WITH NSCBC

This code implements a one-dimensional compressible reacting-flow solver with
Navier-Stokes Characteristic Boundary Conditions (NSCBC) for the treatment of
inflow and outflow boundaries.

The NSCBC formulation implemented in this code is based on the characteristic
boundary-condition methodology described by Yoo and Im:

```
C. S. Yoo and H. G. Im,
"Characteristic boundary conditions for simulations of compressible
reacting flows with multi-dimensional, viscous and reaction effects,"
Combustion Theory and Modelling, Vol. 11, No. 2, pp. 259-286, 2007.
DOI: 10.1080/13647830600898995
```

The implementation accounts for characteristic wave amplitudes together with
the relevant viscous, species-diffusion, thermal-diffusion, and chemical
reaction contributions required for compressible reacting-flow boundary
treatment. The solver is intended for numerical investigation of reacting
flows, including autoignition and propagation of pressure and acoustic
disturbances through non-reflecting inflow and outflow boundaries.

---

## AUTHORS

Nishant Vinod Bhorkar (IISc)
Aritra Roy Choudhury (IISc)

---

## USAGE AND PERMISSION

This code has been developed for research and academic purposes. Any person or
organization wishing to use, modify, reproduce, redistribute, or incorporate
this code, in whole or in part, into another project or publication should
obtain prior permission from the authors.

Appropriate acknowledgement of the authors and citation of the relevant
scientific literature, including the NSCBC formulation of Yoo and Im, should
be provided in any work making use of this code.

Unauthorized redistribution or presentation of this code as original work by
another individual or organization is not permitted.

===============================================================================
'''

import pandas as pd
import numpy as np
import cantera as ct
import math
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
from matplotlib.animation import PillowWriter
import matplotlib.animation as animation
from scipy.signal import savgol_filter
from dataclasses import dataclass

density = pd.DataFrame()
momentum = pd.DataFrame()
energy = pd.DataFrame()
H2 = pd.DataFrame()
O2 = pd.DataFrame()
OH = pd.DataFrame()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MECHANISM_FILE = os.path.join(BASE_DIR, 'chem_Lietal.yaml')


@dataclass
class SimulationResult:
    pressure_probe: np.ndarray
    momentum_probe: np.ndarray
    density_probe: np.ndarray
    max_temperature: np.ndarray
    damp_history: np.ndarray
    autoignition_step: int
    autoignition_time: float | None
    dt: float


'''
##########################################################################
#############     #######  ##  ##   ##   ##    ####      #################
#############     ##       ##  ##   ###  ##   ##         #################
#############     #####    ##  ##   ## # ##   ##         #################
#############     ##       ##  ##   ##  ###   ##         #################
#############     ##        ####    ##   ##    ####      #################
##########################################################################

##########################################################################
#############     #####  #####    ##    ####  ######      ################
#############    ##      ##  ##  ## #  ##     ##          ################
#############     ##     #####  ##   # ##     ####        ################
#############      ##    ##     ###### ##     ##          ################
#############   #####    ##     ##   #  ####  ######      ################
##########################################################################
'''
def initializeQ(q_f, gas_f):
    if ICFlag == 'InltOut':
        inletVelocity = 200
        initTemperature = 1100
        initY = 'H2:0.00347816,O2:0.18401789,H2O:0.05206297,N2:0.76044098'
        initP = ct.one_atm
        for i in range(0, nx):
            gas_f.TPY = initTemperature, initP, initY

            ################################

            rho = gas_f.density
            h = gas_f.enthalpy_mass
            p = gas_f.P
            Y = gas_f.Y
            initVelocity = inletVelocity
            kEn = 0.5*initVelocity**2
            pvWrk = p/rho

            ################################

            q_f[0, i] = rho
            q_f[1, i] = rho*initVelocity
            q_f[2, i] = rho*(kEn - pvWrk + h)
            q_f[3:, i] = rho*Y
    return q_f

def applyBC(q_f, k1_from_rk1):
    k1s_f = []
    # Left boundary:                          # changeGasInit
    if leftB == 'nscbcInflow':

        inletGas = gasAll 
        R_u = ct.gas_constant

        #####################################################################
        ################          Initialized gas       #####################
        #####################################################################

        inletGas.TPX = temperature[0], pressure[0], moleFractions[:, 0]
        
        #####################################################################

        c = inletGas.sound_speed
        gamma = inletGas.cp/inletGas.cv

        #####################################################################
        #####################################################################

        L1l, L2l, L5l, Lspl = computeLLeft(q_f)

        tau0 = 4/3 * viscosity[0] * dudx[0]
        tau1 = 4/3 * viscosity[1] * dudx[1]
        tau2 = 4/3 * viscosity[2] * dudx[2]

        q0 = -thermalConductivity[0] * dTdx[0]
        q1 = -thermalConductivity[1] * dTdx[1]
        q2 = -thermalConductivity[2] * dTdx[2]

        dTaudx = ( -3 * tau0 + 4 * tau1 - tau2 ) / 2 / dx
        tauGradU = 4/3*viscosity[0]*dudx[0]
        dqdx = ( -3 * q0 + 4 * q1 - q2 ) / 2 / dx
        sumHiJi, sumHiWiOmegai = 0.0, 0.0
        Vj_0 = np.zeros(nSpecies)
        Vj_1 = np.zeros(nSpecies)
        Vj_2 = np.zeros(nSpecies)
        for j in range(nSpecies):
            if moleFractions[j, 0] > 1e-16:
                V = -mixAvDiffCoeffecient[j, 0]/moleFractions[j, 0]*dXdx[j, 0]
            else:
                V = 0
            if moleFractions[j, 1] > 1e-16:
                V1 = -mixAvDiffCoeffecient[j, 1]/moleFractions[j, 1]*dXdx[j, 1]
            else:
                V1 = 0
            if moleFractions[j, 2] > 1e-16:
                V2 = -mixAvDiffCoeffecient[j, 2]/moleFractions[j, 2]*dXdx[j, 2]
            else:
                V2 = 0
            Vj_0[j] = V
            Vj_1[j] = V1
            Vj_2[j] = V2
            J = q_f[j+3, 0]*V
            J1 = q_f[j+3, 1]*V1
            J2 = q_f[j+3, 2]*V2
            sumHiJi += (partialEnthalpies[j, 0] - inletGas.cp*temperature[0]*avgMolecularWeight[0]/inletGas.molecular_weights[j])*(-3*J + 4*J1 - J2)/2/dx
            sumHiWiOmegai += -(partialEnthalpies[j, 0] - inletGas.cp*temperature[0]*avgMolecularWeight[0]/inletGas.molecular_weights[j])*inletGas.net_production_rates[j]*inletGas.molecular_weights[j]
        
        dJdx = ( -3 * q_f[3:, 0] * Vj_0 + 4 * q_f[3:, 1] * Vj_1 - q_f[3:, 2] * Vj_2 ) / 2 / dx

        rho_RHS = -(L2l + (L5l + L1l)/c**2)
        velocity_RHS = -(L5l - L1l)/c + dTaudx/q_f[0, 0]
        pr_RHS = -(L5l + L1l) + (gamma - 1)*(tauGradU - dqdx + sumHiJi) #+ (gamma - 1)*sumHiWiOmegai 
        Y_RHS = -Lspl + dJdx/q_f[0, 0] #+ inletGas.net_production_rates*inletGas.molecular_weights/q_f[0, 0]

        if rkStep == 1: 
            k1_left = [rho_RHS, velocity_RHS, pr_RHS, Y_RHS]
            k1s_f.append(k1_left)
        if rkStep == 2: 
            k1_left = k1_from_rk1[0]
            k2_left = [rho_RHS, velocity_RHS, pr_RHS, Y_RHS]

        #####################################################################
        #####################################################################

        rhoOld = q_f[0, 0]
        velocityOld = q_f[1, 0]/rhoOld
        pressureOld = pressure[0]
        YOld = q_f[3:, 0]/rhoOld

        #####################################################################

        if rkStep == 1:
            inletRho_nscbc = rhoOld + dt*rho_RHS
            inletVelocity_nscbc = velocityOld + dt*velocity_RHS
            inletPressure_nscbc = pressureOld + dt*pr_RHS
            inletY_nscbc = YOld + dt*Y_RHS

        if rkStep == 2:
            inletRho_nscbc = rhoOld + dt/2*(k1_left[0] + k2_left[0])
            inletVelocity_nscbc = velocityOld + dt/2*(k1_left[1] + k2_left[1])
            inletPressure_nscbc = pressureOld + dt/2*(k1_left[2] + k2_left[2])
            inletY_nscbc = YOld + dt/2*(k1_left[3] + k2_left[3])

        #####################################################################
        ################    Initialize gas with new values    ###############
        #####################################################################

        inletT_nscbc = inletPressure_nscbc*inletGas.mean_molecular_weight/inletRho_nscbc/R_u
        inletGas.TPY = inletT_nscbc, inletPressure_nscbc, inletY_nscbc
        inletH_nscbc = inletGas.enthalpy_mass

        ###########################   Special step   #########################
        if count > 0:
            pressure[0] = inletPressure_nscbc
            temperature[0] = inletT_nscbc
        #######################################################################

        #####################################################################
        #####################################################################
        
        q_f[0, 0] = inletRho_nscbc
        q_f[1, 0] = inletRho_nscbc*inletVelocity_nscbc
        q_f[2, 0] = inletRho_nscbc*(0.5*inletVelocity_nscbc**2 - inletPressure_nscbc/inletRho_nscbc + inletH_nscbc)
        q_f[3:, 0] = inletRho_nscbc*inletY_nscbc

    # Right boundary:
    if rightB == 'nscbcOutflow':
        
        outletGas = gasAll                           # changeGasInit
        R_u = ct.gas_constant

        #####################################################################
        ################          Initialized gas       #####################
        #####################################################################

        outletGas.TPX = temperature[nx-1], pressure[nx-1], moleFractions[:, nx-1]
        
        #####################################################################

        c = outletGas.sound_speed
        gamma = outletGas.cp/outletGas.cv

        #####################################################################
        #####################################################################

        L1r, L2r, L5r, Lspr = computeLRight(q_f)
        
        tauNxm1 = 4/3 * viscosity[nx-1] * dudx[nx-1]
        tauNxm2 = 4/3 * viscosity[nx-2] * dudx[nx-2]
        tauNxm3 = 4/3 * viscosity[nx-3] * dudx[nx-3]

        qNxm1 = -thermalConductivity[nx-1] * dTdx[nx-1]
        qNxm2 = -thermalConductivity[nx-2] * dTdx[nx-2]
        qNxm3 = -thermalConductivity[nx-3] * dTdx[nx-3]

        dTaudx = ( 3 * tauNxm1 - 4 * tauNxm2 + tauNxm3 ) / 2 / dx
        tauGradU = 4/3*viscosity[nx-1]*dudx[nx-1]
        dqdx = ( 3 * qNxm1 - 4 * qNxm2 + qNxm3 ) / 2 / dx
        sumHiJi, sumHiWiOmegai = 0.0, 0.0

        VjNxm1 = np.zeros(nSpecies)
        VjNxm2 = np.zeros(nSpecies)
        VjNxm3 = np.zeros(nSpecies)

        #print(' At the outlet :------------------------------------')

        for j in range(nSpecies):
            if moleFractions[j, nx-1] > 1e-16:
                V = -mixAvDiffCoeffecient[j, nx-1]/moleFractions[j, nx-1]*dXdx[j, nx-1]
            else:
                V = 0
            if moleFractions[j, nx-2] > 1e-16:
                V1 = -mixAvDiffCoeffecient[j, nx-2]/moleFractions[j, nx-2]*dXdx[j, nx-2]
            else:
                V1 = 0
            if moleFractions[j, nx-3] > 1e-16:
                V2 = -mixAvDiffCoeffecient[j, nx-3]/moleFractions[j, nx-3]*dXdx[j, nx-3]
            else:
                V2 = 0
            VjNxm1[j] = V
            VjNxm2[j] = V1
            VjNxm3[j] = V2
            J = q_f[j+3, nx-1]*V
            J1 = q_f[j+3, nx-2]*V1
            J2 = q_f[j+3, nx-3]*V2
            sumHiJi += (partialEnthalpies[j, nx-1] - outletGas.cp_mass*temperature[nx-1]*avgMolecularWeight[nx-1]/outletGas.molecular_weights[j])*(3*J - 4*J1 + J2)/2/dx
            sumHiWiOmegai += -(partialEnthalpies[j, nx-1] - outletGas.cp_mass*temperature[nx-1]*avgMolecularWeight[nx-1]/outletGas.molecular_weights[j])*outletGas.net_production_rates[j]*outletGas.molecular_weights[j]
            #print('For j = ', j,', hi: ', partialEnthalpies[j, nx-1], ', cp*T*W/Wi: ', outletGas.cp_mass*temperature[nx-1]*avgMolecularWeight[nx-1]/outletGas.molecular_weights[j], ', omegai*Wi: ', outletGas.net_production_rates[j]*outletGas.molecular_weights[j])

        dJdx = ( 3 * q_f[3:, nx-1] * VjNxm1 - 4 * q_f[3:, nx-2] * VjNxm2 + q_f[3:, nx-3] * VjNxm3 ) / 2 / dx

        rho_RHS = -(L2r + (L5r + L1r)/c**2)
        velocity_RHS = -(L5r - L1r)/c + dTaudx/q_f[0, nx-1]
        pr_RHS = -(L5r + L1r) + (gamma - 1)*(tauGradU - dqdx + sumHiJi) + (gamma - 1)*sumHiWiOmegai 
        Y_RHS = -Lspr + dJdx/q_f[0, nx-1] + outletGas.net_production_rates*outletGas.molecular_weights/q_f[0, nx-1] 

        if rkStep == 1: 
            k1_right = [rho_RHS, velocity_RHS, pr_RHS, Y_RHS]
            k1s_f.append(k1_right)
        if rkStep == 2: 
            k1_right = k1_from_rk1[1]
            k2_right = [rho_RHS, velocity_RHS, pr_RHS, Y_RHS]

        #####################################################################

        rhoOld = q_f[0, nx-1]
        velocityOld = q_f[1, nx-1]/rhoOld
        pressureOld = pressure[nx-1]
        YOld = q_f[3:, nx-1]/rhoOld

        #####################################################################

        if rkStep == 1:
            outletRho_nscbc = rhoOld + dt*rho_RHS
            outletVelocity_nscbc = velocityOld + dt*velocity_RHS
            outletPressure_nscbc = pressureOld + dt*pr_RHS
            outletY_nscbc = YOld + dt*Y_RHS

        if rkStep == 2:
            outletRho_nscbc = rhoOld + dt/2*(k1_right[0] + k2_right[0])
            outletVelocity_nscbc = velocityOld + dt/2*(k1_right[1] + k2_right[1])
            outletPressure_nscbc = pressureOld + dt/2*(k1_right[2] + k2_right[2])
            outletY_nscbc = YOld + dt/2*(k1_right[3] + k2_right[3])

        #####################################################################
        ################    Initialize gas with new values    ###############
        #####################################################################

        outletT_nscbc = outletPressure_nscbc*outletGas.mean_molecular_weight/outletRho_nscbc/R_u
        outletGas.TPY = outletT_nscbc, outletPressure_nscbc, outletY_nscbc
        outletH_nscbc = outletGas.enthalpy_mass

        ###########################   Special step   #########################
        if count > 0:
            pressure[nx-1] = outletPressure_nscbc
            temperature[nx-1] = outletT_nscbc
        #######################################################################

        #####################################################################
        #####################################################################

        q_f[0, nx-1] = outletRho_nscbc
        q_f[1, nx-1] = outletRho_nscbc*outletVelocity_nscbc
        q_f[2, nx-1] = outletRho_nscbc*(0.5*outletVelocity_nscbc**2 - outletPressure_nscbc/outletRho_nscbc + outletH_nscbc)
        q_f[3:, nx-1] = outletRho_nscbc*outletY_nscbc

        # print('In the right boundary for NSCBC: ')
        # print('Energy at last cell: ', round(q_f[2, nx-1], 8))
        # print('KE: ', 0.5*outletVelocity_nscbc**2, ', PV: ', outletPressure_nscbc/outletRho_nscbc, ', H: ', outletH_nscbc)

    if rkStep == 1:
        return q_f, k1s_f
    if rkStep == 2:
        return q_f

def computeLLeft(q_f):

    ##########################################################
    ###################   Target values   ####################
    ##########################################################

    inletYtarget = [0.00347816, 0.18401789, 0.        , 0.        , 0.05206297,   0.        , 0.        , 0.        , 0.76044098] #[8.11005780E-03, 1.83162563E-01, 0.0, 0.0, 5.18209821E-02, 0.0, 0.0, 0.0, 7.56906397E-01]
    inletTTarget = 1100
    inletVelocityTarget = 200

    ##########################################################

    rho = q_f[0, 0]
    u = q_f[1, 0]/rho
    et = q_f[2, 0]/rho

    ##########################################################

    gas_f = gasAll                           # changeGasInit
    U_f = et - 0.5*u**2
    V_f = 1/rho
    Y_f = q_f[3:, 0]/rho
    gas_f.UVY = U_f, V_f, Y_f
    R_u = ct.gas_constant
    p = pressure[0]                          # pressure[0]   gas_f.P
    c = gas_f.sound_speed                    # (gamma*p/rho)**0.5
    Ma = u/c
    
    ###########################################################

    dudx_f = (-3 * q_f[1, 0]/q_f[0, 0] + 4 * q_f[1, 1]/q_f[0, 1] - q_f[1, 2]/q_f[0, 2]) / (2*dx)
    dpdx_f = (-3 * p + 4 * pressure[1] - pressure[2]) / (2*dx)
    lambda1 = (u - c)
    mixMolWt = avgMolecularWeight[0]       # avgMolecularWeight[0]  gas_f.mean_molecular_weight
    c_p = 1.0 #gas_f.cp*1000

    ##########################################################
    ##################    Define all L's    ##################
    ##########################################################

    lYil_f = np.zeros(nSpecies)

    l1l_f = lambda1 * ( dpdx_f - rho * c * dudx_f ) / 2
    l2l_f = damp1 * R_u * rho * ( temperature[0] - inletTTarget ) / mixMolWt / c / c_p / L 
    l5l_f = damp2 * rho * c**2 * ( 1 - Ma**2 ) * ( u - inletVelocityTarget ) / 2 / L
    for j in range(nSpecies):
        lYil_f[j] = damp3 * c * ( Y_f[j] - inletYtarget[j] ) / L 

    ##########################################################
    ##########################################################

    return l1l_f, l2l_f, l5l_f, lYil_f
    
def computeLRight(q_f):
    rho = q_f[0, nx-1]
    u = q_f[1, nx-1]/rho
    et = q_f[2, nx-1]/rho
    Y = q_f[3:, nx-1]/rho
    gas_f =  gasAll                           # changeGasInit
    U_f = et - 0.5*u**2
    V_f = 1/rho
    gas_f.UVY = U_f, V_f, Y
    pInf = ct.one_atm
    sigma = 0.27
    c = gas_f.sound_speed
    Ma = u/c
    k = sigma * ( 1 - Ma**2 ) * c / L /2
    lambda2 = u
    lambda5 = u + c
    lambda5pi = u

    dudx_f = ( 3 * q_f[1, nx-1] - 4 * q_f[1, nx-2] + q_f[1, nx-3] ) / ( 2 * dx )
    dpdx_f = ( 3 * pressure[nx-1] - 4 * pressure[nx-2] + pressure[nx-3] ) / ( 2 * dx )
    dRhodx_f = ( 3 * q_f[0, nx-1] - 4 * q_f[0, nx-2] + q_f[0, nx-3] ) / ( 2 * dx )
    dYidx_f = ( 3 * q_f[3:, nx-1]/q_f[0, nx-1] - 4 * q_f[3:, nx-2]/q_f[0, nx-2] + q_f[3:, nx-3]/q_f[0, nx-3] ) / ( 2 * dx )

    ##############################################

    lYir_f = np.zeros(nSpecies)

    l1r_f = k*(pressure[nx-1] - pInf)
    l2r_f = lambda2*(dRhodx_f - dpdx_f/c**2)
    l5r_f = lambda5*(dpdx_f + rho*c*dudx_f)/2

    for j in range(nSpecies):
        lYir_f[j] = lambda5pi*dYidx_f[j]

    return l1r_f, l2r_f, l5r_f, lYir_f

def computeAux(q_f):
    gas_f =  gasAll                           # changeGasInit
    for i in range(nx):
        rho = q_f[0, i]
        u = q_f[1, i]/rho
        et = q_f[2, i]/rho
        Y = q_f[3:, i]/rho
        U_f = et - 0.5*u**2
        V_f = 1/rho
        gas_f.UVY = U_f, V_f, Y

        ##############################################

        UU[i] = U_f
        VV[i] = V_f

        ##############################################

        if count == 0 :
            pressure[i] = gas_f.P
            temperature[i] = gas_f.T
        else:
            if i != 0 and i != nx-1: 
                pressure[i] = gas_f.P
                temperature[i] = gas_f.T
        
        ##############################################

        # pressure[i] = gas_f.P
        # temperature[i] = gas_f.T

        ###############################################

        viscosity[i] = gas_f.viscosity * 1.0
        thermalConductivity[i] = gas_f.thermal_conductivity
        mixAvDiffCoeffecient[:, i] = gas_f.mix_diff_coeffs
        avgMolecularWeight[i] = gas_f.mean_molecular_weight
        moleFractions[:, i] = gas_f.Y
        partialEnthalpies[:, i] = gas_f.partial_molar_enthalpies/gas_f.molecular_weights
        ##############################################

def computeDerivatives(u_f, x_f, T_f):
    for i in range(2, nx-2):
        dudx[i] = (-u_f[i+2] + 8*u_f[i+1] - 8*u_f[i-1] + u_f[i-2])/(12*dx)
        dTdx[i] = (-T_f[i+2] + 8*T_f[i+1] - 8*T_f[i-1] + T_f[i-2])/(12*dx)
        for j in range(nSpecies):
            dXdx[j, i] = (-x_f[j, i+2] + 8*x_f[j, i+1] - 8*x_f[j, i-1] + x_f[j, i-2])/(12*dx)
    dudx[0] = (-3 * u_f[0] + 4*u_f[1] - u_f[2])/(2*dx)
    dTdx[0] = (-3 * T_f[0] + 4*T_f[1] - T_f[2])/(2*dx)
    dudx[nx-1] = (3 * u_f[nx-1] - 4*u_f[nx-2] + u_f[nx-3])/(2*dx)
    dTdx[nx-1] = (3 * T_f[nx-1] - 4*T_f[nx-2] + T_f[nx-3])/(2*dx)
    dudx[1] = (u_f[2] - u_f[0])/2/dx
    dTdx[1] = (T_f[2] - T_f[0])/2/dx
    dudx[nx-2] = (u_f[nx-1] - u_f[nx-3])/2/dx
    dTdx[nx-2] = (T_f[nx-1] - T_f[nx-3])/2/dx
    for j in range(nSpecies):
        dXdx[j, 0] = (-3 * x_f[j, 0] + 4*x_f[j, 1] - x_f[j, 2])/(2*dx)
        dXdx[j, nx-1] = (3 * x_f[j, nx-1] - 4*x_f[j, nx-2] + x_f[j, nx-3])/(2*dx)
        dXdx[j, 1] = (x_f[j, 2] - x_f[j, 0])/2/dx
        dXdx[j, nx-2] = (x_f[j, nx-1] - x_f[j, nx-3])/2/dx

def computeRHS(q_f):
    c_f = np.zeros((nVars, nx))
    d_f = np.zeros((nVars, nx))
    s_f = np.zeros((nVars, nx))
    rhs_f = np.zeros((nVars, nx))
    
    c_f, d_f, s_f = computeCDS(q_f)
    for j in range(nVars):
        if j != indexInert:
            for i in range(1, nx-1):
                dcdx_f = computeDcdx(c_f, j, i)
                dddx_f = computeDddx(d_f, j, i)
                rhs_f[j, i] = -dcdx_f - dddx_f + s_f[j, i]
    return rhs_f

def computeCDS(q_f):
    c_ff = np.zeros((nVars, nx))
    d_ff = np.zeros((nVars, nx))
    s_ff = np.zeros((nVars, nx))
    gas_f =  gasAll                           # changeGasInit
    for i in range(nx):
        rho = q_f[0, i]
        u = q_f[1, i]/rho
        et = q_f[2, i]/rho
        Y = q_f[3:, i]/rho
        U_f = et - 0.5*u**2
        V_f = 1/rho
        gas_f.UVY = U_f, V_f, Y
        p = gas_f.P
        ##############################################
        cRho = rho*u
        cMom = rho*u**2 + p
        cEne = u*(rho*et + p)
        cY = rho*u*Y
        ##############################################
        c_ff[0, i] = cRho
        c_ff[1, i] = cMom
        c_ff[2, i] = cEne
        c_ff[3:, i] = cY
        ##############################################
        dRho = 0
        dMom = -4/3*viscosity[i]*dudx[i]
        sumHiji = 0
        for j in range(nSpecies):
            if moleFractions[j, i] > 1e-16:
                V = -mixAvDiffCoeffecient[j, i]/moleFractions[j, i]*dXdx[j, i]
            else:
                V = 0
            J = rho*Y[j]*V
            d_ff[j+3, i] = J
            sumHiji += partialEnthalpies[j, i]*J
        dEne = dMom*u + sumHiji - thermalConductivity[i]*dTdx[i]
        d_ff[0, i] = dRho
        d_ff[1, i] = dMom
        d_ff[2, i] = dEne
        ##############################################
        s_ff[0, i] = 0
        s_ff[1, i] = 0
        s_ff[2, i] = 0
        s_ff[3:, i] = gas_f.net_production_rates*gas_f.molecular_weights

        ###############################################
        ##############   Special step   ###############

        
        # tauDamper = 0.016*L
        # damper = 1 - np.exp(-i*dx / tauDamper)
        # if i*dx > 0.1*L:
        #     damper = 1.0
        # s_ff[3:, i] = s_ff[3:, i]*damper

    return c_ff, d_ff, s_ff

def correctSpecies(q_f):
    for i in range(nx):
        sumYi = 0
        for j in range(nSpecies):
            if j+3 != indexInert:
                if q_f[j+3, i] < 0:
                    q_f[j+3, i] = 0
                sumYi += q_f[j+3, i]/q_f[0, i]
        #         if q_f[0, i] == 0:
        #             print('q_f[j+3, i]: ', q_f[j+3, i], ' q_f[0, i]: ', q_f[0, i])
        # #print('sumYi: ', sumYi)
        q_f[indexInert, i] = (1 - sumYi)*q_f[0, i]
    #print('q_f[indexInert]: ', q_f[indexInert, :])
    return q_f

def writeData():
    density[count] = Q[0, :]
    momentum[count] = Q[1, :]
    energy[count] = Q[2, :]
    H2[count] = Q[4, :]
    O2[count] = Q[6, :]
    OH[count] = Q[7, :]
    density.to_csv('density.csv')
    momentum.to_csv('momentum.csv')
    energy.to_csv('energy.csv')
    H2.to_csv('H2.csv')
    O2.to_csv('O2.csv')
    OH.to_csv('OH.csv')
    print('Data written to file')

def readIC():
    density = pd.read_csv('density.csv')
    momentum = pd.read_csv('momentum.csv')
    energy = pd.read_csv('energy.csv')
    H2 = pd.read_csv('H2.csv')
    O2 = pd.read_csv('O2.csv')
    OH = pd.read_csv('OH.csv')
    cols = density.columns[1:]
    print(density[cols[2]])
    Q[0, :] = density[cols[2]]
    Q[1, :] = momentum[cols[2]]
    Q[2, :] = energy[cols[2]]
    print('Data read from file')

def limitMom(q_f):
    for i in range(nx):
        if q_f[1, i] < 0:
            q_f[1, i] = 0
        if q_f[1, i] > 100:
            q_f[1, i] = 100
    return q_f

def plotData():

    idxFrom = 0
    idxTo = nx

    plt.clf()
    plt.gcf().set_size_inches(12, 12)
    
    plt.subplot(6, 1, 1)
    plt.plot(Q[1, idxFrom:idxTo], color='b')
    plt.title('Momentum')

    plt.subplot(6, 1, 2)
    plt.plot(Q[2, idxFrom:idxTo], color='b')
    plt.title('Energy')

    plt.subplot(6, 1, 3)
    plt.plot(pressure[idxFrom:idxTo], color='b')
    plt.axvline(x=int(nx*0.1), color='r', linestyle='--', label='Probe')
    # plt.ylim(100000, 140000)
    plt.title('Pressure')

    plt.subplot(6, 1, 4)
    plt.plot(Q[3, idxFrom:idxTo]/Q[0, idxFrom:idxTo]/0.00347816, color='b', label='H2')
    plt.plot(Q[4, idxFrom:idxTo]/Q[0, idxFrom:idxTo]/0.18401789, color='r', label='O2')
    plt.legend()
    plt.title('Species')

    plt.subplot(6, 1, 5)
    plt.plot(Q[0, idxFrom:idxTo], color='b')
    plt.title('Density')
    
    plt.subplot(6, 1, 6)
    plt.plot(pProbe, color='b', label='probe')
    plt.xlim(-10000, endCount+10000)
    plt.title('Pressure probe')

    maxTempText = f'Tmax: {maxTemperature:.2f} K'
    ignitionText = f'Autoignition: {autoignitionDetected}'
    plt.suptitle('Iteration: ' + str(count)+', CFL: '+str(dt*np.max(Q[1, :]/Q[0, :])/dx) + ', ' + maxTempText + ', ' + ignitionText)

    # Save images as a jpeg 
    # Make a directory 
    if count % 100 == 0 or count < 6:
        plt.savefig(os.path.join(plotDirectory, str(count) + '_thStep.jpeg'), dpi=300)
        plt.close(plt.gcf())

def computeDcdx(c_ff, jj, ii):
    cDiscFlag = 'central'
    if cDiscFlag == 'central':
        if ii == 0:
            dcdx_ff = (-3 * c_ff[jj, ii] + 4*c_ff[jj, ii+1] - c_ff[jj, ii+2])/(2*dx)
        elif ii == nx-1:
            dcdx_ff = (3 * c_ff[jj, ii] - 4*c_ff[jj, ii-1] + c_ff[jj, ii-2])/(2*dx)
        elif ii == 1:
            dcdx_ff = (c_ff[jj, ii+1] - c_ff[jj, ii-1])/2/dx
        elif ii == nx-2:
            dcdx_ff = (c_ff[jj, ii] - c_ff[jj, ii-2])/2/dx
        else:
            dcdx_ff = (-c_ff[jj, ii+2] + 8*c_ff[jj, ii+1] - 8*c_ff[jj, ii-1] + c_ff[jj, ii-2])/(12*dx)
    elif cDiscFlag == 'upwind':
        if ii == 0:
            dcdx_ff = 0 #(-3 * c_ff[jj, ii] + 4*c_ff[jj, ii+1] - c_ff[jj, ii+2])/(2*dx)
        elif ii == 1:
            dcdx_ff = (c_ff[jj, ii] - c_ff[jj, ii-1])/dx
        else:
            dcdx_ff = (3 * c_ff[jj, ii] - 4*c_ff[jj, ii-1] + c_ff[jj, ii-2])/(2*dx)
    return dcdx_ff

def computeDddx(d_ff, jj, ii):
    if ii == 0:
        dddx_ff = (-3 * d_ff[jj, ii] + 4*d_ff[jj, ii+1] - d_ff[jj, ii+2])/(2*dx)
    elif ii == nx-1:
        dddx_ff = (3 * d_ff[jj, ii] - 4*d_ff[jj, ii-1] + d_ff[jj, ii-2])/(2*dx)
    elif ii == 1:
        dddx_ff = (d_ff[jj, ii+1] - d_ff[jj, ii-1])/2/dx
    elif ii == nx-2:
        dddx_ff = (d_ff[jj, ii] - d_ff[jj, ii-2])/2/dx
    else:
        dddx_ff = (-d_ff[jj, ii+2] + 8*d_ff[jj, ii+1] - 8*d_ff[jj, ii-1] + d_ff[jj, ii-2])/(12*dx)
    return dddx_ff

def printData():
    print('Count = ', count)
    print(' Density: ')
    print(Q[0, :10])
    print(' Momentum: ')
    print(Q[1, :10])
    print(' Energy: ')
    print(Q[2, :10])
    print('---------------------------------------------------------------------')

def applyFilter(q_f):

    window = 5
    poly = 3

    filtFrom = 0
    filtTo = nx-1

    density = q_f[0, filtFrom:filtTo]
    energy = q_f[2, filtFrom:filtTo]
    momentum = q_f[1, filtFrom:filtTo]
    smoothDensity = savgol_filter(density, window, poly)
    smoothEnergy = savgol_filter(energy, window, poly)
    smoothMomentum = savgol_filter(momentum, window, poly)
    q_f[0, filtFrom:filtTo] = smoothDensity
    q_f[2, filtFrom:filtTo] = smoothEnergy
    q_f[1, filtFrom:filtTo] = smoothMomentum
    for i in range(nSpecies):
        sp = q_f[i+3, filtFrom:filtTo]
        smoothSpecies = savgol_filter(sp, window, poly)
        q_f[i+3, filtFrom:filtTo] = smoothSpecies
    return q_f
    

###########################################################
###########################################################


def write_probe_files():
    with open('pressureProbe27.txt', 'w') as f:
        for p in pProbe:
            f.write(str(p) + '\n')
    with open('momentumProbe27.txt', 'w') as f:
        for m in momentumProbe:
            f.write(str(m) + '\n')
    with open('densityProbe27.txt', 'w') as f:
        for d in densityProbe:
            f.write(str(d) + '\n')


def initialize_problem(end_count=200000, damp_init=(0.27, 0.27, 0.27), plot_dir='images_0.27'):
    global gasAll, L, nx, dx, dt, gasInit, ICFlag, leftB, rightB, C_disFlag
    global indexInert, nSpecies, nVars, Q, QNew, k1, k2, k3, k4, Q1, Q2, Q3, Q4
    global animPlots, pressure, temperature, dudx, dXdx, dTdx, viscosity
    global thermalConductivity, mixAvDiffCoeffecient, avgMolecularWeight
    global moleFractions, partialEnthalpies, UU, VV, count, pProbe
    global momentumProbe, densityProbe, maxTemperatureHistory, dampHistory
    global autoignitionDetected, autoignitionTemperatureThreshold, autoignitionTime
    global maxTemperature, endCount, damp1, damp2, damp3, plotDirectory
    global density, momentum, energy, H2, O2, OH

    density = pd.DataFrame()
    momentum = pd.DataFrame()
    energy = pd.DataFrame()
    H2 = pd.DataFrame()
    O2 = pd.DataFrame()
    OH = pd.DataFrame()

    gasAll = ct.Solution(MECHANISM_FILE)

    L = 0.05
    nx = 256
    dx = L/(nx-1)
    dt = 0.5e-8

    gasInit = ct.Solution(MECHANISM_FILE)
    ICFlag = 'InltOut'

    leftB = 'nscbcInflow'
    rightB = 'nscbcOutflow'

    C_disFlag = 'upwind'

    indexInert = 3 + gasInit.species_index('N2')

    nSpecies = gasInit.n_species
    nVars = 3 + nSpecies
    Q = np.zeros((nVars, nx))
    QNew = np.zeros((nVars, nx))
    k1, k2, k3, k4 = np.zeros((nVars, nx)), np.zeros((nVars, nx)), np.zeros((nVars, nx)), np.zeros((nVars, nx))
    Q1, Q2, Q3, Q4 = np.zeros((nVars, nx)), np.zeros((nVars, nx)), np.zeros((nVars, nx)), np.zeros((nVars, nx))

    animPlots = []

    pressure = np.zeros(nx)
    temperature = np.zeros(nx)
    dudx = np.zeros(nx)
    dXdx = np.zeros((nSpecies, nx))
    dTdx = np.zeros(nx)
    viscosity = np.zeros(nx)
    thermalConductivity = np.zeros(nx)
    mixAvDiffCoeffecient = np.zeros((nSpecies, nx))
    avgMolecularWeight = np.zeros(nx)
    moleFractions = np.zeros((nSpecies, nx))
    partialEnthalpies = np.zeros((nSpecies, nx))
    UU = np.zeros(nx)
    VV = np.zeros(nx)

    count = 0
    Q = initializeQ(Q, gasInit)
    computeAux(Q)
    computeDerivatives(Q[1, :], moleFractions, temperature)

    pProbe = []
    momentumProbe = []
    densityProbe = []
    maxTemperatureHistory = []
    dampHistory = []
    autoignitionDetected = False
    autoignitionTemperatureThreshold = 1400.0
    autoignitionTime = None
    maxTemperature = float(np.max(temperature))
    endCount = end_count
    damp1, damp2, damp3 = map(float, damp_init)
    plotDirectory = plot_dir


def update_controller(controller, p_target, candidate_damps, control_interval, history_length):
    global damp1, damp2, damp3

    if controller is None or p_target is None:
        return
    if count % control_interval != 0:
        return
    if len(pProbe) < history_length:
        return

    current_damp = (damp1, damp2, damp3)
    if candidate_damps is None:
        candidate_damps = [current_damp]

    time_since_ignition = 0.0
    if autoignitionTime is not None:
        time_since_ignition = count * dt - autoignitionTime

    damp1, damp2, damp3 = controller.choose_damping(
        pressure_history=pProbe[-history_length:],
        p_target=p_target,
        max_temperature=maxTemperature,
        time_since_ignition=time_since_ignition,
        current_damp=current_damp,
        candidate_damps=candidate_damps,
    )


def run_simulation(
    p_target=None,
    damp_init=(0.27, 0.27, 0.27),
    controller=None,
    candidate_damps=None,
    control_interval=10,
    end_count=200000,
    write_files=True,
    make_plots=True,
    plot_dir='images_0.27',
    verbose=True,
):
    global Q, QNew, RHS, rkStep, count, maxTemperature, autoignitionDetected, autoignitionTime

    initialize_problem(end_count=end_count, damp_init=damp_init, plot_dir=plot_dir)
    if make_plots:
        os.makedirs(plotDirectory, exist_ok=True)

    history_length = getattr(controller, 'history_length', 8)
    autoignitionStep = -1

    while count < endCount:
        if verbose and count % 1000 == 0:
            print('Count = ', count, ' ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        if make_plots and (count % 200 == 0 or count < 6):
            plotData()

        rkStep = 1

        RHS = computeRHS(Q)
        QNew = Q + dt*RHS
        Q, a = applyBC(Q, 0)
        QNew[:, 0] = Q[:, 0]
        QNew[:, nx-1] = Q[:, nx-1]

        Q = QNew
        Q = correctSpecies(Q)

        if count % 30 == 0 and count > 1000:
            Q = applyFilter(Q)

        computeAux(Q)

        maxTemperature = float(np.max(temperature))
        if not autoignitionDetected and maxTemperature >= autoignitionTemperatureThreshold:
            autoignitionDetected = True
            autoignitionStep = count
            autoignitionTime = count * dt
            if verbose:
                print(
                    'Autoignition detected at count =',
                    count,
                    ', time =',
                    autoignitionTime,
                    's, Tmax =',
                    maxTemperature,
                    'K'
                )

        if autoignitionDetected:
            update_controller(controller, p_target, candidate_damps, control_interval, history_length)

        pProbe.append(pressure[int(nx*0.1)])
        momentumProbe.append(Q[1, int(nx*0.1)])
        densityProbe.append(Q[0, int(nx*0.1)])
        maxTemperatureHistory.append(maxTemperature)
        dampHistory.append([damp1, damp2, damp3])

        computeDerivatives(Q[1, :], moleFractions, temperature)

        if write_files and count % 20 == 0:
            write_probe_files()

        count += 1

    if write_files:
        write_probe_files()

    return SimulationResult(
        pressure_probe=np.asarray(pProbe, dtype=float),
        momentum_probe=np.asarray(momentumProbe, dtype=float),
        density_probe=np.asarray(densityProbe, dtype=float),
        max_temperature=np.asarray(maxTemperatureHistory, dtype=float),
        damp_history=np.asarray(dampHistory, dtype=float),
        autoignition_step=autoignitionStep,
        autoignition_time=autoignitionTime,
        dt=dt,
    )


if __name__ == "__main__":
    run_simulation()
