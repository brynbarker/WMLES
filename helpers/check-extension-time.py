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
ss.addVar('t','scalar')


initPhi = """
:rad:  = sqrt( (meshx-.5)**2 + (meshy-.5)**2 )
:phi:  = :rad: - .25
[:gphix:,:gphiy:,:tmp:] = grad( :phi: )
:t: = 0.*:rad:
"""
ss.parse(initPhi)

def inside_zero(pysim,vel,phi):

    u, v = vel
    u_new = npy.where(phi < 1e-7, 0, u)
    v_new = npy.where(phi < 1e-7, 0, v)

    return [u_new,v_new]
ss.addUserDefinedFunction("inside_zero",inside_zero)

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

moveU = """
:uwm: = ibmS( :uwm:, :phi: - %s, [:gphix:,:gphiy:,0.0] )
""" % delta

moveU2 = """
:uwm: = ibmG( :phi: - %s, :gphix:,:gphiy:,0.0, :uwm: )
""" % delta

initUV_a = """
:u:    = sin(  meshx * 0.25 * pi ) * cos( meshy * 0.25 * pi )*exp(-:t:)
:v:    = sin( -meshy * 0.25 * pi ) * cos( meshx * 0.25 * pi )*exp(-:t:)
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
"""
initUV_b = """
:u:    = sin(  meshx * 1.*pi ) * cos( meshy *1.* pi )*exp(-:t:)
:v:    = sin( -meshy * 1.*pi ) * cos( meshx *1.* pi )*exp(-:t:)
: : = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
"""
initUV_c = """
:u:    = meshy*exp(-:t:)
:v:    = numpy.random.rand(meshy.shape[0],meshy.shape[1],meshy.shape[2])*1e-4*exp(-:t:)
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
"""
initUV_d = """
:u:    = numpy.ones_like(meshy)*exp(-:t:)
:v:    = numpy.ones_like(meshy)*exp(-:t:)
:u: = where(:phi: < 1e-7, 0., :u:)
:v: = where(:phi: < 1e-7, 0., :v:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
:uwm: = 1.*:umag:
"""

updateUV_a = """
:u: = where(:phi: < 1e-7, :u:, sin(  meshx * .25*pi ) * cos( meshy *.25* pi )*exp(-:t:))
:v: = where(:phi: < 1e-7, :v:,sin( -meshy * .25*pi ) * cos( meshx *.25* pi )*exp(-:t:))
:umag: = where(:phi:<1e-7,:uwm:,sqrt( :u:*:u: + :v:*:v: ))
:uwm: = 1.*:umag:
"""
updateUV_b = """
:u: = where(:phi: < 1e-7, :u:, sin(  meshx * 1.*pi ) * cos( meshy *1.* pi )*exp(-:t:))
:v: = where(:phi: < 1e-7, :v:,sin( -meshy * 1.*pi ) * cos( meshx *1.* pi )*exp(-:t:))
:umag: = where(:phi:<1e-7,:uwm:,sqrt( :u:*:u: + :v:*:v: ))
:uwm: = 1.*:umag:
"""
updateUV_c = """
:u: = where(:phi: < 1e-7, :u:, meshy*exp(-:t:))
:v: = where(:phi: < 1e-7, :v:,numpy.random.rand(meshy.shape[0],meshy.shape[1],meshy.shape[2])*1e-4*exp(-:t:))
:umag: = where(:phi:<1e-7,:uwm:,sqrt( :u:*:u: + :v:*:v: ))
:uwm: = 1.*:umag:
"""
updateUV_d = """
:u: = where(:phi: < 1e-7, :u:, numpy.ones_like(meshy)*exp(-:t:))
:v: = where(:phi: < 1e-7, :v:,numpy.ones_like(meshy)*exp(-:t:))
:umag: = where(:phi:<1e-7,:uwm:,sqrt( :u:*:u: + :v:*:v: ))
:uwm: = 1.*:umag:
"""

titles = ['sin(pi x /4)*cos(pi y /4)','sin(pi x) * cos(pi y)','horizontal flow','constant']

convergence = npy.zeros((4,2,20))


delta_t_vals = 2**npy.arange(-7,1)

inits = [initUV_a,initUV_b,initUV_c,initUV_d]
updates = [updateUV_a,updateUV_b,updateUV_c,updateUV_d]

for alpha_ind, alpha in enumerate([0.5,1.]):
    move = moveU if alpha_ind else moveU2
    all_erros = np.zeros((len(delta_t_vals),4,10,100)
    for i,delta_t in enumerate(delta_t_vals):
        for field,(initUV,updateUV) in enumerate(zip(inits,updates)):
            #initialize
            ss.parse(initUV)

            # First, treat umag as a scalar to tranport
            #   from phi = \delta to \phi = 0
            for _ in range(100):
                ss.parse(move)

            
            for it in range(100):
                time = delta_t*(1+it) 
                setT = """
                :t: = :t: + %s
                """ % delta_t
                ss.parse(setT)
                ss.parse(updateUV)
                outer = probesOuter.get('umag')

                for tau_step in range(10):
                    ss.parse(move)
                
                    innerU = probesInner.get('uwm')
                    err = npy.linalg.norm(outer-innerU)
                    all_errs[i,field,tau_step,it] = err 


            plt.subplot(len(delta_t_vals),4,i*4+field+1)
            for tau_step in range(10):
                plt.semilogy(all_errs[i,field,tau_step,:],label='tau_step {}'.format(tau_step+1))
            plt.legend()
            if field == 0:
                plt.ylabel('delta_t = {}'.formate(delta_t))
            if i == 0:
                plt.title(titles[field])
            if i == len(delta_t_vals)-1:
                plt.xlabel('timestep')

    plt.suptitle('alpha = {}'.format(alpha))
    plt.savefig('alpha{}.png',dpi=300)
delta_t = 1.
num_t_steps = 1
irse
/nitUV_a = """
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
plt.show()
plt.pause(.1)
