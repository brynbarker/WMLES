import os
import numpy as np
import time
import scipy
import scipy.optimize
import matplotlib as mpl
import matplotlib.colors as mcol
import matplotlib.pyplot as plt

from mixtureEOS import *


def get_T_from_p_func(T,rho,p,mat):

    [pmat,temp,temp]=mat.get_p(rho,T)
    return p-pmat

def get_T_from_p(mat,rho,p):
    Tguess = p
    T = scipy.optimize.fsolve(get_T_from_p_func,Tguess,args=(rho,p,mat))[0]
    
    return T

def jump_cond_func_post(input,M,rho1,p1,mat):

    (rho2,p2,u1) = input

    T1 = get_T_from_p(mat,rho1,p1)
    T2 = get_T_from_p(mat,rho2,p2)

    cs2 =  mat.get_cs(rho2,T2)
    u2 = M*cs2

    [e1,temp,temp] = mat.get_e(rho1,T1)
    [e2,temp,temp] = mat.get_e(rho2,T2)

    h1 = e1 + p1/rho1
    h2 = e2 + p2/rho2

    eqn1 = rho1*u1 - (rho2*u2)
    eqn2 = p1+rho1*u1*u1 - (p2+rho2*u2*u2)
    eqn3 = h1+u1*u1/2. - (h2+u2*u2/2.)

    out = [eqn1,eqn2,eqn3]

    return out

def calculate_jump_set_post(M,rho1,p1,mat):

    # 0.6597633136094674, 0.554016620498615, 1.4095311747144714

    # print('Solving Jump')
    [rho2,p2,u1] = \
    scipy.optimize.fsolve(jump_cond_func_post,[rho1,p1,100.0],args=(M,rho1,p1,mat))
    # print('Solved Jump')

    # print('Solving T')
    T2 = get_T_from_p(mat,rho2,p2)
    # print('Solved T')
    cs2 =  mat.get_cs(rho2,T2)
    u2 = M*cs2

    return (rho2, u2, p2, u1)

def jump_cond_func_pre(input,M,rho2,p2,mat):

    (rho1,p1,u1) = input

    T1 = get_T_from_p(mat,rho1,p1)
    T2 = get_T_from_p(mat,rho2,p2)

    cs2 =  mat.get_cs(rho2,T2)
    u2 = M*cs2

    [e1,temp,temp] = mat.get_e(rho1,T1)
    [e2,temp,temp] = mat.get_e(rho2,T2)

    h1 = e1 + p1/rho1
    h2 = e2 + p2/rho2

    eqn1 = rho1*u1 - (rho2*u2)
    eqn2 = p1+rho1*u1*u1 - (p2+rho2*u2*u2)
    eqn3 = h1+u1*u1/2. - (h2+u2*u2/2.)

    out = [eqn1,eqn2,eqn3]

    return out

def jump_cond_func_pre_p(input,p1,rho2,p2,mat):

    (rho1,u1,u2) = input

    T1 = get_T_from_p(mat,rho1,p1)
    T2 = get_T_from_p(mat,rho2,p2)

    cs2 =  mat.get_cs(rho2,T2)
    # u2 = M*cs2

    [e1,temp,temp] = mat.get_e(rho1,T1)
    [e2,temp,temp] = mat.get_e(rho2,T2)

    h1 = e1 + p1/rho1
    h2 = e2 + p2/rho2

    eqn1 = rho1*u1 - (rho2*u2)
    eqn2 = p1+rho1*u1*u1 - (p2+rho2*u2*u2)
    eqn3 = h1+u1*u1/2. - (h2+u2*u2/2.)

    out = [eqn1,eqn2,eqn3]

    return out

def calculate_jump_set_pre(M,rho2,p2,mat):

    # 0.6597633136094674, 0.554016620498615, 1.4095311747144714

    # print('Solving Jump')
    [rho1,p1,u1] = \
    scipy.optimize.fsolve(jump_cond_func_pre,[rho2,p2,100.0],args=(M,rho2,p2,mat))
    # print('Solved Jump')

    # print('Solving T')
    T2 = get_T_from_p(mat,rho2,p2)
    # print('Solved T')
    cs2 =  mat.get_cs(rho2,T2)
    u2 = M*cs2

    return (rho1, u1, p1, u2)

def calculate_jump_set_pre_p(p1,rho2,p2,mat):

    # 0.6597633136094674, 0.554016620498615, 1.4095311747144714

    # print('Solving Jump')
    [rho1,u1,u2] = \
    scipy.optimize.fsolve(jump_cond_func_pre_p,[rho2,1000.0,300.0],args=(p1,rho2,p2,mat))
    # print('Solved Jump')

    # print('Solving T')
    # T2 = get_T_from_p(mat,rho2,p2)
    # # print('Solved T')
    # cs2 =  mat.get_cs(rho2,T2)
    # u2 = M*cs2

    return (rho1, u1, u2)

def jump_cond_func_bjo(input,M1,rho1,p1,mat):

    (rho2,p2,M2) = input

    T1 = get_T_from_p(mat,rho1,p1)
    T2 = get_T_from_p(mat,rho2,p2)

    cs1 =  mat.get_cs(rho1,T1)
    cs2 =  mat.get_cs(rho2,T2)
    u1 = M1*cs1
    u2 = M2*cs2

    [e1,temp,temp] = mat.get_e(rho1,T1)
    [e2,temp,temp] = mat.get_e(rho2,T2)

    h1 = e1 + p1/rho1
    h2 = e2 + p2/rho2

    eqn1 = rho1*u1 - (rho2*u2)
    eqn2 = p1+rho1*u1*u1 - (p2+rho2*u2*u2)

    # Density or not?
    #eqn3 = h1+u1*u1/2. - (h2+u2*u2/2.)
    eqn3 = rho1*(h1+u1*u1/2.) - rho2*(h2+u2*u2/2.)

    #out = [eqn1,eqn2,eqn3]

    out = eqn1**2 + eqn2**2 + eqn3**2
    
    return out

def calculate_jump_bjo(M1,rho1,p1,mat):

    # 0.6597633136094674, 0.554016620498615, 1.4095311747144714

    #[rho2,p2,u1] = \
    #scipy.optimize.fsolve(jump_cond_func,[rho1,p1,1.0],args=(M,rho1,p1,mat))
    #[rho2,p2,M2] = scipy.optimize.fsolve(jump_cond_func,[1.001*rho1,1.001*p1,.1],args=(M1,rho1,p1,mat))
    sol = scipy.optimize.minimize(jump_cond_func,[1.001*rho1,1.001*p1,.1],args=(M1,rho1,p1,mat))
    [rho2,p2,M2] = sol.x
    
    
    T2  = get_T_from_p(mat,rho2,p2)
    cs2 =  mat.get_cs(rho2,T2)
    u2  = M2*cs2
    
    #return (rho2, u2, p2, M2)
    return (M2,rho2, p2, u2)



P = 1.0
gamma = 1.4
Ru = 1.0
M = 100.
rho = 1.0
Pinf = 6.e3

my_mat = gammaEOS(Ru,M,gamma)
my_mat2 = stiffenedGasEOS(Ru,M,gamma,Pinf)


Tout = get_T_from_p(my_mat,rho,P)
T = P/(rho*Ru/M)

T2out = get_T_from_p(my_mat2,rho,P)
T2 = (P+Pinf)/(rho*Ru/M)

# print([T,Tout,T2,T2out])
