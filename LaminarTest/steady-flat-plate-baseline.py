import sys
from mpi4py import MPI
import numpy 
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM

test = False
testName = None


## Define a mesh
Npts = 128 
L = numpy.pi * 2.0  
dim = 2
gamma = 1.4
yp = L/10.

problem = 'steady-flat-plate-baseline'

Lp = L * (Npts-1.0) / Npts
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = numpy.array([True, False, False])
mesh_options['dim'] = 3
mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
mesh_options['xn'] = [ Lp   , Lp    ,  L-Lp ]
mesh_options['nn'] = [ Npts , Npts ,  1  ]
mesh_options['pn'] = [4,4,1]
#mesh_options['pn'] = [14,14,1]


# Initialize a simulation object on a mesh
ss = pyrandaSim(problem,mesh_options)
ss.addPackage( pyrandaBC(ss) )
ss.addPackage( pyrandaIBM(ss) )
ss.addPackage( pyrandaTimestep(ss) )


rho0 = 1.0
p0   = 1.0
gamma = 1.4
mach = 2.0
s0 = numpy.sqrt( p0 / rho0 * gamma )
u0 = s0 * mach
e0 = p0/(gamma-1.0) + rho0*.5*u0*u0

def print_BL_params(pysim, u):
    nx, ny, nz = pysim.mesh.options['nn']
    uleft = numpy.zeros((ny))
    uinf = numpy.zeros((nx))
    if pysim.PyMPI.x1proc:
        lo = pysim.PyMPI.chunk_3d_lo[1]
        hi = pysim.PyMPI.chunk_3d_hi[1]
        uleft[lo:hi+1] = u[0,:,0].copy()
    if pysim.PyMPI.ynproc:
        lo = pysim.PyMPI.chunk_3d_lo[0]
        hi = pysim.PyMPI.chunk_3d_hi[0]
        uinf[lo:hi+1] = u[:,-1,0].copy()
    uleft = pysim.PyMPI.comm.allreduce(uleft,op=MPI.SUM)
    uinf = pysim.PyMPI.comm.allreduce(uinf,op=MPI.SUM)
    if pysim.PyMPI.master:
        if min(uleft.flatten()) == max(uleft.flatten()):
            print('u is constant')
            nu, hwm = 0., 0.
        else:
            Uinf = numpy.mean(uinf)
            x1, y1, z1 = pysim.mesh.options['x1']
            xn, yn, zn = pysim.mesh.options['xn']
            domain = numpy.linspace(y1,yn,uleft.size)
            d99 = numpy.interp(0.99*Uinf, uleft, domain)
            nu = d99*Uinf/5e4
            hwm = 0.1*d99 
            deltax = 0.15*hwm
            print(hwm,pysim.dx)

ss.addUserDefinedFunction("printBLparams",print_BL_params)

# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)                  - ddy(:rho:*:v:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)   - ddy(:rhou:*:v:)
ddt(:rhov:) =  -ddx(:rhov:*:u:)                 - ddy(:rhov:*:v: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: - :tx:*:kappa:) - ddy( (:Et: + :p: - :tau:)*:v: - :ty:*:kappa: )
# Level set equation
#ddt(:phi:)  =  - :gx: * :u1: - :gy: * :v1: 
# Conservative filter of the EoM
:rho:       =  fbar( :rho:  )
:rhou:      =  fbar( :rhou: )
:rhov:      =  fbar( :rhov: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:u:         =  :rhou: / :rho:
:v:         =  :rhov: / :rho:
:p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
:T:         = :p: / (:rho: * :R: )
# Artificial bulk viscosity (old school way)
:div:       =  ddx(:u:) + ddy(:v:)
:beta:      =  gbar( ring(:div:) * :rho: ) * 7.0e-2
:tau:       = :beta:*:div:
[:tx:,:ty:,:tz:] = grad(:T:)
:kappa:     = gbar( ring(:T:)* :rho:*:cv:/(:T: * :dt: ) ) * 1.0e-3
# Apply constant BCs
printBLparams(:u:)
[:u:,:v:,:w:] = ibmWall( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [:u1:,:u2:,0.0] )
:rho: = ibmS( :rho: , :phi:, [:gx:,:gy:,:gz:] )
:p:   = ibmS( :p:   , :phi:, [:gx:,:gy:,:gz:] )
bc.extrap(['rho','p'],['y1','yn'])
bc.extrap(['u'],['yn'])
#bc.const(['u'],['x1','y1','yn'],u0)
bc.const(['v'],['y1','yn'],0.0)
bc.const(['u'],['y1'],0.0)
#bc.const(['rho'],['x1','y1','yn'],rho0)
#bc.const(['p'],['x1','y1','yn'],p0)
:Et:  = :p: / ( :gamma: - 1.0 )  + .5*:rho:*(:u:*:u: + :v:*:v:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)
:dtB: = 0.2* dt.diff(:beta:,:rho:)
:dt: = numpy.minimum(:dt:,:dtB:)
:umag: = sqrt( :u:*:u: + :v:*:v: )
"""
eom = eom.replace('u0',str(u0)).replace('p0',str(p0)).replace('rho0',str(rho0))


# Add the EOM to the solver
ss.EOM(eom)

# Initialize variables
ic = """
:gamma: = 1.4
:R: = 1.0
:cp: = :R: / (1.0 - 1.0/:gamma: )
:cv: = :cp: - :R:
:phi: = meshy - 2*pi/8.
:rho: = 1.0 + 3d()
:p:  =  1.0 + 3d() #exp( -(meshx-1.5)**2/.25**2)*.1
#:u: = where( :phi:>0.0, mach * sqrt( :p: / :rho: * :gamma:) , 0.0 )
:u: = mach * sqrt( :p: / :rho: * :gamma:)
:u: = gbar( gbar( :u: ) )
:v: = 0.0 + 3d()
:Et: = :p:/( :gamma: - 1.0 ) + .5*:rho:*(:u:*:u: + :v:*:v:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)*.1
[:gx:,:gy:,:gz:] = grad( :phi: )
:gx: = gbar( :gx: )
:gy: = gbar( :gy: )
"""
ic = ic.replace('mach',str(mach))


# Set the initial conditions
ss.setIC(ic)


# Write a time loop
time = 0.0
viz = True

# Approx a max dt and stopping time
tt = 5.0 #

# Start time loop
cnt = 1
viz_freq = 10
max_steps = 100

wvars = ['p','rho','u','v','phi']
if not test:
    ss.write( wvars )
CFL = 1.0
dt = ss.variables['dt'].data * CFL * .1

while tt > time:
    
    # Update the EOM and get next dt
    time = ss.rk4(time,dt)
    dt = min(ss.variables['dt'].data * CFL, 1.1*dt)
    dt = min(dt, (tt - time) )

    # Print some output
    ss.iprint("%s -- %s --- %f" % (cnt,time,dt)  )
    cnt += 1
    #import pdb; pdb.set_trace()
    if viz and (not test):
        ss.write(wvars)
        if (cnt%viz_freq == 1): 
            ss.write( wvars )
    if cnt > max_steps:
        break

                   

# Curve test.  Write file and print its name at the end
if test:
    x = ss.mesh.coords[0].data
    xx   =  ss.PyMPI.zbar( x )
    pvar = 'p'
    v = ss.PyMPI.zbar( ss.variables[pvar].data )
    ny = ss.PyMPI.ny
    v1d =  v[:,int(ny/2)]
    x1d = xx[:,int(ny/2)]
    fname = testName + '.dat'
    numpy.savetxt( fname  , (x1d,v1d) )
    print(fname)
