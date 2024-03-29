import time
import numpy as npy
import scipy.optimize
from pyranda import *
import matplotlib.pyplot as plt

toptimes = []
bottomtimes = []
N = [20,30,40,50,60,70,80,90,100]#,120,140,160,180,200,240,280,320,360,400]
for nx in N:
    ny = nx
    mesh = """
    xdom = (0,1,%s)
    ydom = (0,1,%s)
    """ % (nx,ny)
    
    ss = pyrandaSim("test",mesh)
    ss.EOM("")
    
    # Add in IBM
    ss.addPackage( pyrandaIBM(ss) )
    
    
    # Offset
    delta = .1
    eps = ss.mesh.GridLen * npy.sqrt(2)
    
    ss.addVar('phi','scalar')
    ss.addVar('u','scalar')
    ss.addVar('v','scalar')
    ss.addVar('gphix','scalar')
    ss.addVar('gphiy','scalar')
    ss.addVar('rad','scalar')
    ss.addVar('tmp','scalar')
    ss.addVar('Uwm','scalar')
    ss.addVar('Uwm2','scalar')
    ss.addVar('u0','scalar')
    ss.addVar('u0s2','scalar')
    ss.addVar('u0w2','scalar')
    
    
    initPhi = """
    :rad:  = sqrt( (meshx-.5)**2 + (meshy-.5)**2 )
    :phi:  = :rad: - .25
    [:gphix:,:gphiy:,:tmp:] = grad( :phi: )
    """
    ss.parse(initPhi)
    
    def compute_Uwm(pysim, vel, phi, gphi):
        u, v = vel
        phix, phiy = gphi
    
        norm = u*phix + v*phiy 
        u -= norm*phix
        v -= norm*phiy
    
        Uwm = npy.sqrt(u*u + v*v)
        return Uwm
    
    def compute_u0(pysim,Uwm):
        hwm = delta
        Uinf = 0.5
        nu = 10*hwm*Uinf/5e4
        deltax = 0.15*hwm
        #nu = 7.415414466563401e-05 # Re ~ 4e5
        #nu = 3.7077072332817006e-12
        #hwm = 0.3070183607238605
        #deltax = 0.04605275410857908
    
        kappa = 0.41
        A = 10.
        u_tau = npy.zeros_like(Uwm.flatten())
        for index, uwm in enumerate(Uwm.flatten()):
            def g(ut):
                val = uwm/ut - 1/kappa*npy.log(hwm*ut/nu)-A
                valprime = -1/ut * (uwm/ut - 1/kappa)
                return val, valprime
            sol = scipy.optimize.root(g, 1e-6,jac=True)
            if not sol.success:
                import pdb
                pdb.set_trace()
            u_tau[index] = sol.x
        u0 = u_tau/kappa * (npy.log(u_tau*deltax/nu)-1) + u_tau*A
        return u0.reshape(Uwm.shape)
    
    # The ibmS operator preserving gradient information?
    #  Try reducing the strength of the diffusion/filter
    immersed_CFL = 0.5
    def smooth(pysim,SDF,gDx,gDy,gDz,val_in,epsi=0.0,new=False):    
        val = val_in * 1.0
        GridLen = pysim.mesh.GridLen
        alpha = 0.5  # < .4 is unstable!
        for i in range(5):
            [tvx,tvy,tvz] = pysim.grad(val)
            term = tvx*gDx+tvy*gDy+tvz*gDz
            val = npy.where( SDF <= epsi , val + immersed_CFL*GridLen*term , val )
            Tval = pysim.gfilter(val)
            Tval = Tval * alpha + val * (1.0 - alpha)
            val = npy.where( SDF <= epsi , Tval, val )        
            return val
    
    
    ss.addUserDefinedFunction("ibmG",smooth)
    ss.addUserDefinedFunction("computeU0",compute_u0)
    ss.addUserDefinedFunction("computeUwm",compute_Uwm)
    
    
    initUV = """
    :u:    = sin(  meshx * 4 * pi ) * cos( meshy * 4 * pi )
    :v:    = sin( -meshy * 4 * pi ) * cos( meshx * 4 * pi )
    :Uwm: = computeUwm([:u:,:v:],:phi:,[:gphix:,:gphiy:])
    :u0:   = computeU0(:Uwm:)
    :Uwm2: = :Uwm:*1.
    """
    ss.parse(initUV)
    
    # First, treat Uwm as a scalar to tranport
    #   from phi = \delta to \phi = 0
    
    start = time.time()
    moveUwm2= """
    :Uwm2: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0,:Uwm2: )
    """ % delta
    for i in range(100):
        ss.parse(moveUwm2)
    uwm2u0="""
    :u0s2: = computeU0(:Uwm2:)
    """
    ss.parse(uwm2u0)
    toptime = time.time()-start
    toptimes.append(toptime)
    
    # Next, treat u0 as a scalar to tranport
    #   from phi = \delta to \phi = 0
    
    start = time.time()
    uwm2u0="""
    :u0w2: = computeU0(:Uwm:)
    """
    ss.parse(uwm2u0)
    moveUwm2= """
    :u0w2: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0, :u0w2:)
    """ % delta
    for i in range(100):
        ss.parse(moveUwm2)
    bottomtime = time.time()-start
    bottomtimes.append(bottomtime)
    
for nx,tt,bt in zip(N,toptimes,bottomtimes):
    
    print('-'*30+'\nN = '+str(nx)+'\n'+'-'*30)
    print('Top time:    {}'.format(tt))
    print('Bottom time: {}'.format(bt))

plt.plot(N,toptimes,label=r'$u_0(\Phi(U_{wm}))$')
plt.plot(N,bottomtimes,label=r'$\Phi(u_0)$')
plt.legend()
plt.show()
import pdb; pdb.set_trace()

