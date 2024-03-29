import re
import sys
import time
import numpy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep
from pyranda.pyranda import pyrandaRestart

from mixtureEOS import *

def up_mixture(pysim):
    cmd = """
    # Compute Mixture Quantities
    :R1: = :Ru:/:M1:
    :R2: = :Ru:/:M2:
    :cp1: = :gam1:/(:gam1:-1.)*:R1:
    :cp2: = :gam2:/(:gam2:-1.)*:R2:
    :cp: = :cp1:*:Y1: + :cp2:*:Y2:
    :cv1: = 1./(:gam1:-1.)*:R1:
    :cv2: = 1./(:gam2:-1.)*:R2:
    :cv: = :cv1:*:Y1: + :cv2:*:Y2:
    :gam: = :cp:/:cv:

    :M: = 1./(:Y1:/:M1:+:Y2:/:M2:)
    :R: = :Ru:/:M:
    """
    pysim.parse(cmd)

def get_eos(pysim,mat,eostype=1):

    if eostype == 1:
        if mat == 0:
            gamma = pysim.variables['gam'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M'].data
        elif mat == 1:
            gamma = pysim.variables['gam1'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M1'].data
        elif mat == 2:
            gamma = pysim.variables['gam2'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M2'].data
        my_eos = gammaEOS(Ru,M,gamma)
    elif eostype == 2:
        if mat == 0:
            gamma = pysim.variables['gam'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M'].data
            pinf = pysim.variables['p_inf'].data
        elif mat == 1:
            gamma = pysim.variables['gam1'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M1'].data
            pinf = pysim.variables['p_inf1'].data
        elif mat == 2:
            gamma = pysim.variables['gam2'].data
            Ru = pysim.variables['Ru'].data
            M = pysim.variables['M2'].data
            pinf = pysim.variables['p_inf2'].data
        my_eos = stiffenedGasEOS(Ru,M,gamma,pinf)
    elif eostype == 3:
        leos = pysim.packages['LEOS']
        if mat == 0:
            matID = pysim.variables['matID'].data
        elif mat == 1:
            matID = pysim.variables['matID1'].data
        elif mat == 2:
            matID = pysim.variables['matID2'].data
        my_eos = leosEOS(leos,matID)

    return my_eos

def mat_T(pysim,mat,eostype=1):
    eos = get_eos(pysim,mat,eostype)

    if mat == 0:
        rho = pysim.variables['rho'].data
        e = pysim.variables['e'].data
    elif mat == 1:
        rho = pysim.variables['rho1'].data
        e = pysim.variables['e1'].data
    elif mat == 2:
        rho = pysim.variables['rho2'].data
        e = pysim.variables['e2'].data

    [T, dum1, dum2] = eos.get_T(rho,e)

    if mat == 0:
        pysim.variables['T'].data = T
    elif mat == 1:
        pysim.variables['T1'].data = T
    elif mat == 2:
        pysim.variables['T2'].data = T

def getPwater(pysim,matype1):

    rho = pysim.variables['rho1'].data
    e   = pysim.variables['e1'].data
    eos = get_eos(pysim,1,matype1)

    [T,d1,d2]   = eos.get_T(rho,e)
    [P,d1,d2]   = eos.get_p(rho,T)

    return P
    
        
def mat_p(pysim,mat,eostype=1):
    eos = get_eos(pysim,mat,eostype)

    if mat == 0:
        rho = pysim.variables['rho'].data
        T = pysim.variables['T'].data
    elif mat == 1:
        rho = pysim.variables['rho1'].data
        T = pysim.variables['T1'].data
    elif mat == 2:
        rho = pysim.variables['rho2'].data
        T = pysim.variables['T2'].data

    [p, dum1, dum2] = eos.get_p(rho,T)

    if mat == 0:
        pysim.variables['p'].data = p
    elif mat == 1:
        pysim.variables['p1'].data = p
    elif mat == 2:
        pysim.variables['p2'].data = p

def mat_e(pysim,mat,eostype=1):
    eos = get_eos(pysim,mat,eostype)

    if mat == 0:
        rho = pysim.variables['rho'].data
        T = pysim.variables['T'].data
    elif mat == 1:
        rho = pysim.variables['rho1'].data
        T = pysim.variables['T1'].data
    elif mat == 2:
        rho = pysim.variables['rho2'].data
        T = pysim.variables['T2'].data

    [e, dum1, dum2] = eos.get_e(rho,T)

    if mat == 0:
        pysim.variables['e'].data = e
    elif mat == 1:
        pysim.variables['e1'].data = e
    elif mat == 2:
        pysim.variables['e2'].data = e

def mat_T_from_p(pysim,mat,eostype=1):
    eos = get_eos(pysim,mat,eostype)

    if mat == 0:
        rho = pysim.variables['rho'].data
        p = pysim.variables['p'].data
    elif mat == 1:
        rho = pysim.variables['rho1'].data
        p = pysim.variables['p1'].data
    elif mat == 2:
        rho = pysim.variables['rho2'].data
        p = pysim.variables['p2'].data

    T = eos.get_T_from_p(rho,p)

    if mat == 0:
        pysim.variables['T'].data = T
    elif mat == 1:
        pysim.variables['T1'].data = T
    elif mat == 2:
        pysim.variables['T2'].data = T


def comp_mixtureEOS(pysim,eostype1,eostype2):
    eps_comp = 1.e-32
    # At each zone; these wont change
    rhoY1  = pysim.variables['rhoY1'].data
    rhoY2  = pysim.variables['rhoY2'].data
    rho = pysim.variables['rho'].data
    ie  = pysim.variables['e'].data
    Y1  = pysim.variables['Y1'].data
    Y2  = pysim.variables['Y2'].data
    
    # Pointers to things that WILL change
    rho1 = pysim.variables['rho1'].data
    e1   = pysim.variables['e1'].data
    sv1 = 1.0/(rho1+eps_comp)

    rho2 = pysim.variables['rho2'].data
    e2   = pysim.variables['e2'].data
    sv2 = 1.0/(rho2+eps_comp)
    
    # Mixture equillibration 
    p    = pysim.variables['p'].data
    T    = pysim.variables['T'].data

    # Volume fractions for masks
    V1  = pysim.variables['V1'].data
    V2  = pysim.variables['V2'].data

    
    eos1 = get_eos(pysim,1,eostype1)
    eos2 = get_eos(pysim,2,eostype2)
    
    materials = []

    # MATERIAL TYPE
    materials.append( material(rho1,e1,sv1,rhoY1,Y1,eos1) )
    materials.append( material(rho2,e2,sv2,rhoY2,Y2,eos2) )
        
    # HACK
    mat1_T_floor_on = pysim.variables['T_floor_on1'].data
    mat1_T_floor = pysim.variables['T_floor1'].data
    mat1_p_floor_on = pysim.variables['p_floor_on1'].data
    mat1_p_floor = pysim.variables['p_floor1'].data
    mat1_rho_floor_on = pysim.variables['rho_floor_on1'].data
    mat1_rho_floor = pysim.variables['rho_floor1'].data
    if mat1_T_floor_on:
        materials[0].set_floor_T(mat1_T_floor)
    if mat1_p_floor_on:
        materials[0].set_floor_P(mat1_p_floor)
    if mat1_rho_floor_on:
        materials[0].set_floor_rho(mat1_rho_floor)
    mat2_T_floor_on = pysim.variables['T_floor_on2'].data
    mat2_T_floor = pysim.variables['T_floor2'].data
    mat2_p_floor_on = pysim.variables['p_floor_on2'].data
    mat2_p_floor = pysim.variables['p_floor2'].data
    mat2_rho_floor_on = pysim.variables['rho_floor_on2'].data
    mat2_rho_floor = pysim.variables['rho_floor2'].data
    if mat2_T_floor_on:
        materials[1].set_floor_T(mat2_T_floor)
    if mat2_p_floor_on:
        materials[1].set_floor_P(mat2_p_floor)
    if mat2_rho_floor_on:
        materials[0].set_floor_rho(mat2_rho_floor)
    
        
    [rho,p,T,ie,materials,converged,convergedP,convergedT] = mixtureEOS(rho,p,T,ie,materials)

    pysim.var('rho1').data = materials[0].rho
    pysim.var('rho2').data = materials[1].rho

    pysim.var('e1').data = materials[0].e
    pysim.var('e2').data = materials[1].e

                  
    # After each iteration
    pysim.var('V1').data =  pysim.var('rhoY1').data / (pysim.var('rho1').data +eps_comp)#* mix + high 
    pysim.var('V2').data =  pysim.var('rhoY2').data / (pysim.var('rho2').data +eps_comp)#* mix + low

            
    pysim.var('p').data = p #* mix + high * materials[0].P + low * materials[1].P
    pysim.var('T').data = T #* mix + high * materials[0].T + low * materials[1].T
        

def comp_mixture_cs(pysim,eostype1,eostype2):
    Y1  = pysim.variables['Y1'].data
    Y2  = pysim.variables['Y2'].data

    rho1 = pysim.variables['rho1'].data
    rho2 = pysim.variables['rho2'].data
    T    = pysim.variables['T'].data
    
    eos1 = get_eos(pysim,1,eostype1)
    eos2 = get_eos(pysim,2,eostype2)

    cs1 = eos1.get_cs(rho1,T)
    cs2 = eos2.get_cs(rho2,T)

    cs = np.sqrt(np.abs(Y1*cs1**2 + Y2*cs2**2))
    
    pysim.var('cs').data = cs 


def add_multimat_functions(ss):
    ss.addUserDefinedFunction("update_mixture",up_mixture)
    ss.addUserDefinedFunction("comp_mixtureEOS",comp_mixtureEOS)
    ss.addUserDefinedFunction("comp_mixture_cs",comp_mixture_cs)
    ss.addUserDefinedFunction("mat_p",mat_p)
    ss.addUserDefinedFunction("mat_T",mat_T)
    ss.addUserDefinedFunction("mat_e",mat_e)
        
        


def multimat1d_eom(ss,BCs="",eps_mul=1.2,C_mu=0.002,C_beta=1.0,C_k=0.01,C_D=0.003,C_Y=50,C_DY=0.003,RZ=False,Gamma="max(sqrt(:u:*:u:+:v:*:v:))"):
    # To use this eom, must define mat_type1 and mat_type2 in parms

    ss.addUserDefinedFunction("update_mixture",up_mixture)
    ss.addUserDefinedFunction("comp_mixtureEOS",comp_mixtureEOS)
    ss.addUserDefinedFunction("comp_mixture_cs",comp_mixture_cs)
    ss.addUserDefinedFunction("mat_p",mat_p)
    ss.addUserDefinedFunction("mat_T",mat_T)
    ss.addUserDefinedFunction("mat_e",mat_e)
    ss.addUserDefinedFunction("mat_T_from_p",mat_T_from_p)
    ss.addUserDefinedFunction("getPwater",getPwater)
    # ss.addUserDefinedFunction("get_eos",get_eos)
    # Define the equations of motion
    eom = """
# Primary Equations
ddt(:rhoY1:) = :mass1:
ddt(:rhoY2:) = :mass2:
ddt(:rhou:) = :xmom:
ddt(:rhov:) = :ymom:
ddt(:Et:) = :energy:

:unrho: = :rhoY1:+:rhoY2:
:unY1: = :rhoY1:/:unrho:
:unY2: = :rhoY2:/:unrho:

:eps_comp: = 1.0e-32

# Filter conservative variables
:rhoY1:  = fbar(:rhoY1:)
:rhoY2:  = fbar(:rhoY2:)
:rhou:   = fbar(:rhou:)
:rhov:   = fbar(:rhov:)
:Et:     = fbar(:Et:)

# Update primative variables
:rho: = :rhoY1:+:rhoY2:
:Y1: = :rhoY1:/:rho:
:Y2: = :rhoY2:/:rho:
:u: = :rhou:/:rho:
:v: = :rhov:/:rho:
:ke: = 1./2.*:rho:*(:u:*:u: + :v:*:v:)
:e: = (:Et: - :ke:)/:rho:

# Inject BCs
INSERT_BCS

# Compute Mixture
:converged: = 1 + 3d()
:convergedP: = 1 + 3d()
:convergedT: = 1 + 3d()
comp_mixtureEOS(mat_type1,mat_type2)

## Min pressure (WATER CASE ONLY)
#:fail: = ((:p: < 0.9*my_p0) &  ( :Y1: > .999 ))
#:p:     = where( :fail: , my_p0, :p:)
#:p1: = :p:
#mat_T_from_p(1,mat_type1)
#mat_e(1,mat_type1)
#:e:     = where( :fail:, :e1:, :e:)
#
:ke: = 1./2.*:rho:*(:u:*:u: + :v:*:v:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:Et:  = :rho:*:e:  + :ke:

# Compute derived quantities
comp_mixture_cs(mat_type1,mat_type2)
:h1: = :e1: + :p:/(:rho1:+:eps_comp:) 
:h2: = :e2: + :p:/(:rho2:+:eps_comp:) 
[:ux:,:uy:,:uz:] = grad(:u:)
[:vx:,:vy:,:vz:] = grad(:v:)
:udiv:           = div(:u:,:v:)
# Compute Remaining terms
#[:Tx:,:Ty:,:Tz:]     = grad(:T:)
[:Tx:,:Ty:,:Tz:]     = grad(:e:)
[:Y1x:,:Y1y:,:Y1z:]  =  grad(:Y1:)
[:Y2x:,:Y2y:,:Y2z:]  =  grad(:Y2:)
[:rhoY1x:,:rhoY1y:,:rhoY1z:]  =  grad(:rhoY1:)
[:rhoY2x:,:rhoY2y:,:rhoY2z:]  =  grad(:rhoY2:)
[:V1x:,:V1y:,:V1z:]  =  grad(:V1:)
[:V2x:,:V2y:,:V2z:]  =  grad(:V2:)
## Fix directions for RZ

##GEOM_FLUX##

##
:Sxx: = :ux:
:Sxy: = 0.5*(:uy:+:vx:)
:Syy: = :vy:
:Smag: = sqrt(:Sxx:*:Sxx: + :Syy:*:Syy: + 2.*:Sxy:*:Sxy:)
:rhoe: = :rho:*:e:
:rhoke: = :rho:*:ke:

## Fictive failure force for pressure relaxation from Pflooring
:pwater: = getPwater(mat_type1)
[:fffx:,:fffy:,:fffz:]   = grad( :pwater: - :p: )
:fffx: = .01 * :fffx: * :V1:
:fffy: = .01 * :fffy: * :V1:
:fffz: = .01 * :fffz: * :V1:


# Default Regularization
:J1_out: = 0.0
:J2_out: = 0.0
:Fx_out: = 0.0
:Fy_out: = 0.0
:H_out: = 0.0


# Artifician bulk viscosity
:C_mu:   = 0.0 #0.002
:C_beta: = 0.0 # 1.
:C_k:    = 0.0 # 0.01
:C_D:    = 0.0 # 0.003
:C_Y:    = 0.0 # 100.

:C_mu:   = my_C_mu
:C_beta: = my_C_beta
:C_k:    = my_C_k
:C_D:    = my_C_D
:C_Y:    = my_C_Y
:C_DY:   = my_C_DY

:mu:   = :C_mu:*gbar( :rho:*abs(ring(:Smag:)) )
:mu: = :mu: + my_MU_Water * numpy.maximum(:Y1:,0) + my_MU_Air * numpy.maximum(:Y2:,0)
:beta: = :C_beta:*gbar(:rho:*abs(ring(:udiv:)))
#:kap:  = :C_k:*gbar( :rho:*:cs:*:cs:*:cs:*abs(ring(:T:))/(:T:*:T:*gridLen+1.0e-12) )
#:kap:  = :C_k:*gbar(  sqrt( abs( ring( :e: ) ) ) ) / gridLen
:kap:  = :C_k:*gbar( :rho: *  :cs: * abs( ring( :T: ) ) ) / ( :T: * gridLen )

:taudia: = (:beta:-2./3.*:mu:)*:udiv: - :p:
:tauxx: = 2.*:mu:*:Sxx: + :taudia:
:tauxy: = 2.*:mu:*:Sxy:
:tauyy: = 2.*:mu:*:Syy: + :taudia:
:qcx: = -:kap:*:Tx:
:qcy: = -:kap:*:Ty:


# ######### Diffusivity from Cook(2009) with cs instead of gridLen/dt
# :D11:  = :C_D:*gbar(abs(ring(:Y1:)))/gridLen*:cs:
# :D12:  = :C_Y:*(abs(:Y1:)-1.+abs(1.-:Y1:))*gridLen*:cs:
# :D21:  = :C_D:*gbar(abs(ring(:Y2:)))/gridLen*:cs:
# :D22:  = :C_Y:*(abs(:Y2:)-1.+abs(1.-:Y2:))*gridLen*:cs:
# :D1:   =  :D11: + :D12: 
# :D2:   =  :D21: + :D22:
# :sumDYx: = :D1:*:Y1x: + :D2:*:Y2x:
# :sumDYy: = :D1:*:Y1y: + :D2:*:Y2y:
# :J1x: = -:rho:*(:D1:*:Y1x:-:Y1:*:sumDYx:)
# :J2x: = -:rho:*(:D2:*:Y2x:-:Y2:*:sumDYx:)
# :J1y: = -:rho:*(:D1:*:Y1y:-:Y1:*:sumDYy:)
# :J2y: = -:rho:*(:D2:*:Y2y:-:Y2:*:sumDYy:)
# :FJux:  = (:J1x: + :J2x:)*:u:
# :FJuy:  = (:J1y: + :J2y:)*:u:
# :FJvx:  = (:J1x: + :J2x:)*:v:
# :FJvy:  = (:J1y: + :J2y:)*:v:
# :FJEtx: = (:J1x: + :J2x:)*(:u:*:u:+:v:*:v:)/2.
# :FJEty: = (:J1y: + :J2y:)*(:u:*:u:+:v:*:v:)/2.

#### Jain(2022)
# :D11:  = :C_D:*gbar(abs(:cs:*ring(:Y1:)))/gridLen
# :D12:  = :C_Y:*:cs:/2.*(abs(:Y1:)-1.+abs(1.-:Y1:))*gridLen
# :D21:  = :C_D:*gbar(abs(:cs:*ring(:Y2:)))/gridLen
# :D22:  = :C_Y:*:cs:/2.*(abs(:Y2:)-1.+abs(1.-:Y2:))*gridLen
# :D1:   =  numpy.maximum(:D11:,:D12:) 
# :D2:   =  numpy.maximum(:D21:,:D22:)
# :sumDYx: = :D1:*:Y1x: + :D2:*:Y2x:
# :sumDYy: = :D1:*:Y1y: + :D2:*:Y2y:
# :J1x: = -:rho:*(:D1:*:rhoY1x:-:Y1:*:sumDYx:)
# :J2x: = -:rho:*(:D2:*:rhoY2x:-:Y2:*:sumDYx:)
# :J1y: = -:rho:*(:D1:*:rhoY1y:-:Y1:*:sumDYy:)
# :J2y: = -:rho:*(:D2:*:rhoY2y:-:Y2:*:sumDYy:)
# :FJux:  = (:J1x: + :J2x:)*:u:
# :FJuy:  = (:J1y: + :J2y:)*:u:
# :FJvx:  = (:J1x: + :J2x:)*:v:
# :FJvy:  = (:J1y: + :J2y:)*:v:
# :FJEtx: = (:J1x: + :J2x:)*(:u:*:u:+:v:*:v:)/2.
# :FJEty: = (:J1y: + :J2y:)*(:u:*:u:+:v:*:v:)/2.

#### Brill/Olson with Y and V
:D11:  = :C_D:*gbar(abs(:cs:*ring(:V1:)))/gridLen
:D12:  = :C_Y:*:cs:/2.*(abs(:V1:)-1.+abs(1.-:V1:))*gridLen
:D21:  = :C_D:*gbar(abs(:cs:*ring(:V2:)))/gridLen
:D22:  = :C_Y:*:cs:/2.*(abs(:V2:)-1.+abs(1.-:V2:))*gridLen
:D13:  = :C_DY:*gbar(abs(ring(:Y1:)))/gridLen*:cs:
:D23:  = :C_DY:*gbar(abs(ring(:Y2:)))/gridLen*:cs:
:D1:   =  numpy.maximum(:D11:,:D12:) 
:D2:   =  numpy.maximum(:D21:,:D22:)
:D1:   =  numpy.maximum(:D1:,:D13:) 
:D2:   =  numpy.maximum(:D2:,:D23:)
:sumDYx: = :D1:*:rhoY1x: + :D2:*:rhoY2x:
:sumDYy: = :D1:*:rhoY1y: + :D2:*:rhoY2y:
:J1x: = -(:D1:*:rhoY1x:)#-:Y1:*:sumDYx:)
:J2x: = -(:D2:*:rhoY2x:)#-:Y2:*:sumDYx:)
:J1y: = -(:D1:*:rhoY1y:)#-:Y1:*:sumDYy:)
:J2y: = -(:D2:*:rhoY2y:)#-:Y2:*:sumDYy:)
:FJux:  = (:J1x: + :J2x:)*:u:
:FJuy:  = (:J1y: + :J2y:)*:u:
:FJvx:  = (:J1x: + :J2x:)*:v:
:FJvy:  = (:J1y: + :J2y:)*:v:
:FJEtx: = (:J1x: + :J2x:)*(:u:*:u:+:v:*:v:)/2.
:FJEty: = (:J1y: + :J2y:)*(:u:*:u:+:v:*:v:)/2.

:J1xD: = :J1x:
:J2xD: = :J2x:
:J1yD: = :J1y:
:J2yD: = :J2y:


### Conservative Regularization Jain(2022) - no min enforcement - Flux integration
:Gamma: = my_Gamma
:eps: = gridLen*my_eps_mul
:s1: = :V1:*(1.-:V1:)
:s2: = :V2:*(1.-:V2:)
:gV1x: = gbar(:V1x:) # Need to use for normals
:gV2x: = gbar(:V2x:) # Need to use for normals
:gV1y: = gbar(:V1y:) # Need to use for normals
:gV2y: = gbar(:V2y:) # Need to use for normals
:nhat1x: = :gV1x:/(sqrt(:gV1x:*:gV1x:+:gV1y:*:gV1y:)+1.e-10)
:nhat1y: = :gV1y:/(sqrt(:gV1x:*:gV1x:+:gV1y:*:gV1y:)+1.e-10)
:nhat2x: = :gV2x:/(sqrt(:gV2x:*:gV2x:+:gV2y:*:gV2y:)+1.e-10)
:nhat2y: = :gV2y:/(sqrt(:gV2x:*:gV2x:+:gV2y:*:gV2y:)+1.e-10)

:regV: = 1.e-6
:LV1: = 1.0
:LV2: = 1.0
:LV1: = where(:V1: > :regV:,1,0)
:LV1: = where(:V1: < 1.-:regV:,:LV1:,0)
:LV2: = where(:V2: > :regV:,1,0)
:LV2: = where(:V2: < 1.-:regV:,:LV2:,0)
:a1x: = :Gamma:*(:eps:*:V1x:-:s1:*:nhat1x:)*:LV1:
:a2x: = :Gamma:*(:eps:*:V2x:-:s2:*:nhat2x:)*:LV2:
:a1y: = :Gamma:*(:eps:*:V1y:-:s1:*:nhat1y:)*:LV1:
:a2y: = :Gamma:*(:eps:*:V2y:-:s2:*:nhat2y:)*:LV2:

# :J1x: = :J1x: - my_rho10*:a1x:
# :J2x: = :J2x: - my_rho20*:a2x:
# :J1y: = :J1y: - my_rho10*:a1y:
# :J2y: = :J2y: - my_rho20*:a2y:

:rhoa1x: = :rho1:*:a1x:
:rhoa2x: = :rho2:*:a2x:
:rhoa1y: = :rho1:*:a1y:
:rhoa2y: = :rho2:*:a2y:

:J1x: = :J1x: - :rhoa1x:
:J2x: = :J2x: - :rhoa2x:
:J1y: = :J1y: - :rhoa1y:
:J2y: = :J2y: - :rhoa2y:
:FJux:  = (:J1x: + :J2x:)*:u:
:FJuy:  = (:J1y: + :J2y:)*:u:
:FJvx:  = (:J1x: + :J2x:)*:v:
:FJvy:  = (:J1y: + :J2y:)*:v:
:FJEtx: = (:J1x: + :J2x:)*(:u:*:u:+:v:*:v:)/2.
:FJEty: = (:J1y: + :J2y:)*(:u:*:u:+:v:*:v:)/2.

# Compute qd
:qdx: = :h1:*:J1x: + :h2:*:J2x:
:qdy: = :h1:*:J1y: + :h2:*:J2y:

#  SURFACE_TENSION
:mix:       = :V1:*(1.0-:V1:) * 4.0
:curv:      =  ( ( ddx(:gV1x:) + ddy(:gV1y:) + ddz(:gV1y:) ) * :mix: )  # Note both gv1y in y and z
:stenx:     =  -surface_tension * :curv: * :nhat1x: #/ :rho: 
:steny:     =  -surface_tension * :curv: * :nhat1y: #/ :rho: 
:Fx_out:    =  :stenx:*:rho:
:Fy_out:    =  :steny:*:rho:
# Missing energy term?

:dtC: = dt.courant(:u:,:v:,0.0,:cs:)
:dtC: = numpy.minimum(dt.courant(:u:,0.0,:v:,:cs:),:dtC:)
:dtBeta: = 0.2 * dt.diff(:beta:,:rho:)
:dtMu: = 0.2 * dt.diff(:mu:,:rho:)
:dtD1: = 0.2 * dt.diff(:D1:,1.0)
:dtD2: = 0.2 * dt.diff(:D2:,1.0)
:dta1: = 0.2 * dt.courant(:a1x:,:a1y:,0.0,0)
:dta1bri: = numpy.minimum(:dta1:, 0.2 * dt.courant(:a1x:,0.0,:a1y:,0))
:dta2: = 0.2 * dt.courant(:a2x:,:a2y:,0.0,0)
:dta2bri: = numpy.minimum(:dta2:, 0.2 * dt.courant(:a2x:,0.0,:a2y:,0))
:dta1vis: = 0.02 * dt.diff(:Gamma:*:eps:*:LV1:,1.0)
:dta2vis: = 0.02 * dt.diff(:Gamma:*:eps:*:LV2:,1.0)
:dta1shp: = 0.02 * dt.courant(:Gamma:*:LV1:*:nhat1x:,0.,:Gamma:*:LV1:*:nhat1y:,0.)
:dta2shp: = 0.02 * dt.courant(:Gamma:*:LV2:*:nhat2x:,0.,:Gamma:*:LV2:*:nhat1y:,0.)
:dta1: = numpy.minimum(:dta1vis:,:dta1shp:)
:dta2: = numpy.minimum(:dta2vis:,:dta2shp:)
:dta1: = numpy.minimum(:dta1:,:dta1bri:)
:dta2: = numpy.minimum(:dta2:,:dta2bri:)
:dt: = numpy.minimum( :dtC:,:dtBeta:)
:dt: = numpy.minimum( :dt:, :dtMu:)
:dt: = numpy.minimum( :dt:, :dtD1:)
:dt: = numpy.minimum( :dt:, :dtD2:)
:dt: = numpy.minimum( :dt:, :dta1:)
:dt: = numpy.minimum( :dt:, :dta2:)

# RHS of EOM
##GEOM_RHS##


""".replace('INSERT_BCS',BCs).replace('my_eps_mul',str(eps_mul)).replace('my_C_mu',str(C_mu)).replace('my_C_beta',str(C_beta)).replace('my_C_k',str(C_k)).replace('my_C_DY',str(C_DY)).replace('my_C_D',str(C_D)).replace('my_C_Y',str(C_Y)).replace('my_Gamma',Gamma)


    # For RZ (axisymmetric) modifiy EOM
    if RZ:
        geom_rhs = """
#  Keep 'v' as the second velocity and use "y" as the "z" direction
:mass1:   = -div(:rhoY1:*:u: + :J1x:, :w:, :rhoY1:*:v: + :J1y:) + :J1_out:
:mass2:   = -div(:rhoY2:*:u: + :J2x:, :w:, :rhoY2:*:v: + :J2y:) + :J2_out:
:energy:  = -div((:Et:-:tauxx:)*:u: - :tauxy:*:v: + :qcx: + :qdx: + :FJEtx: , :w: , (:Et:-:tauyy:)*:v: -:tauxy:*:u: + :qcy: + :qdy: + :FJEty:) + :H_out:
[:fxx:,:fxy:,:fxz:] = [:rhou:*:u: - :tauxx: + :FJux: , :w:       , :rhou:*:v: - :tauxy: + :FJuy: ]
[:fyx:,:fyy:,:fyz:] = [      :w:                     , -:taudia: , :w:                           ]
[:fzx:,:fzy:,:fzz:] = [:rhov:*:u: - :tauxy: + :FJvx: , :w:       , :rhov:*:v: - :tauyy: + :FJvy: ]
[:xmom:,:dum:,:ymom:] = divT(:fxx:,:fxy:,:fxz:,:fyx:,:fyy:,:fyz:,:fzx:,:fzy:,:fzz:)
:xmom: = -:xmom: + :Fx_out: - :fffx:
:ymom: = -:ymom: + :Fy_out: - :fffz:
"""
        flux = """
:uy: = :uz:
:vy: = :vz:
:udiv: = :ux: + :vy:
:Ty: = :Tz:
:Y1y: = :Y1z:
:Y2y: = :Y2z:
:rhoY1y: = :rhoY1z:
:rhoY2y: = :rhoY2z:
:V1y: = :V1z:
:V2y: = :V2z:
"""

        
    else:
        geom_rhs = """
:mass1: = -div(:rhoY1:*:u: + :J1x: ,  :rhoY1:*:v: + :J1y:) + :J1_out:
:mass2: = -div(:rhoY2:*:u: + :J2x: ,  :rhoY2:*:v: + :J2y:) + :J2_out:
:xmom:  = -div(:rhou:*:u: - :tauxx: + :FJux: , :rhou:*:v: - :tauxy: + :FJuy:) + :Fx_out:
:ymom:  = -div(:rhov:*:u: - :tauxy: + :FJvx: , :rhov:*:v: - :tauyy: + :FJvy:) + :Fy_out:
:energy: = -div((:Et:-:tauxx:)*:u: - :tauxy:*:v: + :qcx: + :qdx: + :FJEtx: , (:Et:-:tauyy:)*:v: -:tauxy:*:u: + :qcy: + :qdy: + :FJEty:) + :H_out:
"""
        flux = ""

    eom = eom.replace("##GEOM_RHS##",geom_rhs)
    eom = eom.replace("##GEOM_FLUX##",flux)
            


    
    return eom
