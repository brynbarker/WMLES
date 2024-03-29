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


initPhi = """
:rad:  = sqrt( (meshx-.5)**2 + (meshy-.5)**2 )
:phi:  = :rad: - .25
[:gphix:,:gphiy:,:tmp:] = grad( :phi: )
"""
ss.parse(initPhi)



initUV = """
:u:    = sin(  meshx * 2 * pi ) * cos( meshy * 2 * pi )
:v:    = sin( -meshy * 2 * pi ) * cos( meshx * 2 * pi )
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm:  = :umag: * 1.0
"""
ss.parse(initUV)


# First, treat umag as a scalar to tranport
#   from phi = \delta to \phi = 0
moveU = """
:uwm: = ibmS( :uwm:, :phi: - %s, [:gphix:,:gphiy:,0.0] )
""" % delta
for i in range(1000):
    ss.parse(moveU)


ss.plot.figure(1)
ss.plot.contourf('umag',32)
ss.plot.contour('phi',[0])
ss.plot.contour('phi',[delta])


ss.plot.figure(2)
ss.plot.contourf('uwm',32)
ss.plot.contour('phi',[0])
ss.plot.contour('phi',[delta])

innerU = probesInner.get('uwm')
outerU = probesOuter.get('uwm')

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
:uwm: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0, :uwm: )
""" % delta
for i in range(100):
    ss.parse(moveU2)


ss.plot.figure(3)
ss.plot.contourf('uwm',32)
ss.plot.contour('phi',[0])
ss.plot.contour('phi',[delta])




outerU2 = probesOuter.get('uwm')

plt.figure(4)
plt.plot( theta, outerU , 'k--',label='predicted alpha=1.0')
plt.plot( theta, outerU2, 'b--',label='predicted alpha=0.5')
plt.plot( theta, innerU, 'k-',label='actual')

plt.legend()
plt.show(block=False)
import pdb
pdb.set_trace()
plt.pause(.1)
