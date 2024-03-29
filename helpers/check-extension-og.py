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
ss.addVar('uwm1','scalar')
ss.addVar('uwm2','scalar')


initPhi = """
:rad:  = sqrt( (meshx-.5)**2 + (meshy-.5)**2 )
:phi:  = :rad: - .25
[:gphix:,:gphiy:,:tmp:] = grad( :phi: )
"""
ss.parse(initPhi)

def compute_u0(pysim,vel, phi, gphi):
    hwm = delta
    Uinf = 0.5
    nu = 10*hwm*Uinf/5e4
    deltax = 0.15*hwm
    #nu = 7.415414466563401e-05 # Re ~ 4e5
    #nu = 3.7077072332817006e-12
    #hwm = 0.3070183607238605
    #deltax = 0.04605275410857908

    u, v, w = vel
    phix, phiy, phiz = gphi

    norm = u*phix + v*phiy + z*phiz
    u -= norm*phix
    v -= norm*phiy
    w -= norm*phiz

    Uwm = npy.sqrt(u*u + v*v + w*w)

    kappa = 0.41
    A = 10.
    u_tau = npy.zeros_like(u.flatten())
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
    return u0.reshape(u.shape)
ss.addUserDefinedFunction("compute_u0",compute_u0)


initUV = """
:u:    = sin(  meshx * 0.25 * pi ) * cos( meshy * 0.25 * pi )
:v:    = sin( -meshy * 0.25 * pi ) * cos( meshx * 0.25 * pi )
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm:  = :umag: * 1.0
:uwm1:  = :umag: * 1.0
:uwm2:  = :umag: * 1.0
"""
ss.parse(initUV)
outer = probesOuter.get('umag')


# First, treat umag as a scalar to tranport
#   from phi = \delta to \phi = 0
moveU = """
:uwm: = ibmS( :uwm:, :phi: - %s, [:gphix:,:gphiy:,0.0] )
""" % delta
for i in range(100):
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

def smooth1(pysim,SDF,delta,gDx,gDy,gDz,val_in,epsi=0.0,new=False):    
    GridLen = pysim.mesh.GridLen
    SDF1 = SDF - delta
    SDF2 = SDF1 - npy.sqrt(2)*GridLen
    
    tmpx = (delta-SDF)*gDx
    tmpy = (delta-SDF)*gDy

    shftx = (tmpx/GridLen).astype(int)-(tmpx<0)
    shfty = (tmpy/GridLen).astype(int)-(tmpy<0)

    offsetx = tmpx/GridLen-shftx
    offsety = tmpy/GridLen-shfty

    ax, ay, az = SDF.shape

    for xind in range(ax):
        for yind in range(ay):
            if SDF[xind,yind,0] <= epsi:
                xstart = lo[0]+xind+shftx
                ystart = lo[0]+yind+shfty
                x_in, y_in = npy.meshgrid(npy.array([xstart,xstart+1]),npy.array([ystart,ystart+1]))
                phis = pysim.userDefined['get_phi'](x_in,y_in)-delta
                temp = phis - (phis>0)*max(phis.flatten()) - (phis<0)*min(phis.flatten())
                inside, outside = phis[numpy.argmax(temp)], phis[numpy.argmin(temp)]


    val = val_in * 1.0
    alpha = 0.5
    for i in range(5):
        [tvx,tvy,tvz] = pysim.grad(val)
        term = tvx*gDx+tvy*gDy+tvz*gDz
        Tterm = pysim.gfilter(term)
        val = npy.where( SDF <= epsi , alpha*Tval+(1-alpha)*val + immersed_CFL*GridLen*Tterm , val )
        return val



ss.addUserDefinedFunction("ibmP",smooth1)

moveU2 = """
:uwm2: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0, :uwm2: )
""" % delta
for i in range(100):
    ss.parse(moveU2)
#
#moveU1 = """
#:uwm1: = ibmP( :phi: - {}, {}, :gphix:,:gphiy:,0.0, :uwm1: )
#""".format(delta, delta)
#for i in range(100):
#    ss.parse(moveU1)


ss.plot.figure(3)
ss.plot.contourf('uwm2',32)
ss.plot.contour('phi',[0])
ss.plot.contour('phi',[delta])

#ss.plot.figure(4)
#ss.plot.contourf('uwm1',32)
#ss.plot.contour('phi',[0])
#ss.plot.contour('phi',[delta])


innerU2 = probesInner.get('uwm2')
outerU2 = probesOuter.get('uwm2')

#innerU1 = probesInner.get('uwm1')
#outerU1 = probesOuter.get('uwm1')


plt.figure(4)
plt.subplot(221)
plt.plot( theta, outer , 'k',label=r'|u| at $\phi=\delta$')
plt.plot( theta, outerU , 'r-',label=r'$\Phi(|u|)$ at $\phi=\delta$')
plt.plot( theta, innerU, 'r--',label=r'$\Phi(|u|)$ at $\phi=0$')
plt.title(r'$\alpha = 1.0$')
plt.legend()
plt.subplot(222)
plt.plot( theta, outer , 'k',label=r'|u| at $\phi=\delta$')
plt.plot( theta, outerU2 , 'r-',label=r'$\Phi(|u|)$ at $\phi=\delta$')
plt.plot( theta, innerU2, 'r--',label=r'$\Phi(|u|)$ at $\phi=0$')
plt.title(npy.linalg.norm(outer-innerU2))
plt.title(r'$\alpha = 1.0$')
plt.legend()
plt.subplot(223)
plt.plot(theta, abs(outer-outerU), 'k', label=r'L2 normed error at $\phi=\delta$')
plt.plot(theta, abs(outer-innerU), 'r--', label=r'L2 normed error at $\phi=0$')
plt.title(npy.linalg.norm(outer-innerU))
plt.legend()
plt.subplot(224)
plt.plot(theta, abs(outer-outerU2), 'k', label=r'L2 normed error at $\phi=\delta$')
plt.plot(theta, abs(outer-innerU2), 'r--', label=r'L2 normed error at $\phi=0$')
plt.title(npy.linalg.norm(outer-innerU2))
plt.legend()
#plt.subplot(133)
#plt.plot( theta, outer , 'k',label='umag')
#plt.plot( theta, outerU1 , 'r-',label='predicted')
#plt.plot( theta, innerU1, 'r--',label='actual')
#plt.title(npy.linalg.norm(outer-innerU1))

#print(npy.linalg.norm(innerU1-innerU2))

plt.legend()
plt.show(block=False)
import pdb
pdb.set_trace()
plt.pause(.1)
