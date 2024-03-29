import numpy as npy
from pyranda import *
import matplotlib.pyplot as plt

nx = 200
ny = 200

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

# Make a probe set for diagnostics
nProbes = 100
theta = npy.linspace(0,2.0*npy.pi,nProbes+1)[:-1]
R0 = 0.25
x = R0 * npy.cos( theta ) + 0.5
y = R0 * npy.sin( theta ) + 0.5
probesInner = pyrandaProbes(ss,x=x,y=y,z=None)

nProbes = 100
theta = npy.linspace(0,2.0*npy.pi,nProbes+1)[:-1]
R0 = 0.25 + delta
x = R0 * npy.cos( theta ) + 0.5
y = R0 * npy.sin( theta ) + 0.5
probesOuter = pyrandaProbes(ss,x=x,y=y,z=None)


ss.addVar('phi','scalar')
ss.addVar('u','scalar')
ss.addVar('v','scalar')
ss.addVar('gphix','scalar')
ss.addVar('gphiy','scalar')
ss.addVar('rad','scalar')
ss.addVar('tmp','scalar')
ss.addVar('umag','scalar')
ss.addVar('uwm','scalar')
ss.addVar('uwm2','scalar')


initPhi = """
:rad:  = sqrt( (meshx-.5)**2 + (meshy-.5)**2 )
:phi:  = :rad: - .25
[:gphix:,:gphiy:,:tmp:] = grad( :phi: )
"""
ss.parse(initPhi)

def inside_zero(pysim,vel,phi):

    u, v = vel
    u_new = npy.where(phi < 1e-7, 0, u)
    v_new = npy.where(phi < 1e-7, 0, v)

    return [u_new,v_new]
ss.addUserDefinedFunction("inside_zero",inside_zero)


initUV_a = """
:u:    = sin(  meshx * 0.25 * pi ) * cos( meshy * 0.25 * pi )
:v:    = sin( -meshy * 0.25 * pi ) * cos( meshx * 0.25 * pi )
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
:uwm2: = 1.*:umag:
"""
initUV_b = """
:u:    = sin(  meshx * 1.*pi ) * cos( meshy *1.* pi )
:v:    = sin( -meshy * 1.*pi ) * cos( meshx *1.* pi )
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
:uwm2: = 1.*:umag:
"""
initUV_c = """
:u:    = meshy
:v:    = numpy.random.rand(meshy.shape[0],meshy.shape[1],meshy.shape[2])*1e-4
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
:uwm2: = 1.*:umag:
"""
initUV_d = """
:u:    = numpy.ones_like(meshy)
:v:    = numpy.ones_like(meshy)
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
:uwm2: = 1.*:umag:
"""

titles = ['sin(pi x /4)*cos(pi y /4)','sin(pi x) * cos(pi y)','horizontal flow','constant']

convergence = npy.zeros((4,2,20))

for field,initUV in enumerate([initUV_a,initUV_b,initUV_c,initUV_d]):
    ss.parse(initUV)
    outer = probesOuter.get('umag')


    # First, treat umag as a scalar to tranport
    #   from phi = \delta to \phi = 0
    moveU = """
    :uwm: = ibmS( :uwm:, :phi: - %s, [:gphix:,:gphiy:,0.0] )
    """ % delta
    for j in range(20):
        for i in range(10):
            ss.parse(moveU)

    

        innerU = probesInner.get('uwm')

        err = npy.linalg.norm(outer-innerU)
        convergence[field,0,j] = err

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

    moveU2 = """
    :uwm2: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0, :uwm2: )
    """ % delta
    for j in range(20):
        for i in range(10):
            ss.parse(moveU2)


        innerU2 = probesInner.get('uwm2')

        err = npy.linalg.norm(outer-innerU2)
        convergence[field,1,j] = err

    plt.subplot(2,2,field+1)
    plt.semilogy(convergence[field,0,:],label='alpha=1')
    plt.semilogy(convergence[field,1,:],label='alpha=.5')
    plt.legend()
    plt.title(titles[field])
plt.show()
plt.pause(.1)
