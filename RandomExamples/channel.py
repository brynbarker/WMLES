import sys
import numpy 
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM



## Define a mesh
Lx = numpy.pi * 6.0  
Ly = numpy.pi
Nx = 256
Ny = 256
dim = 2
gamma = 1.4

problem = 'channel_flow'

Lxp = Lx * (Nx-1.0) / Nx
Lyp = Ly * (Ny-1.0) / Ny
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = numpy.array([True, False, True])
mesh_options['dim'] = 3
mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
mesh_options['xn'] = [ Lxp   , Lyp    , 1. ]
mesh_options['nn'] = [ Nx , Ny ,  1  ]
#mesh_options['pn'] = [ 16, 2, 1]


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

def debug(pysim):
    uvec = pysim.var('u').data
    import pdb; pdb.set_trace()

ss.addUserDefinedFunction('debug',debug)

#def noslip(pysim, u):
#    if pysim.PyMPI.y1proc:
#        u[:,0,:] = 0.
#    if pysim.PyMPI.ynproc:
#        u[:,-1,:] = 0.
#    return u
#ss.addUserDefinedFunction('noslip',noslip)

def noslip(pysim, u, v, rho, p):
    if pysim.PyMPI.y1proc:
        u[:,0,:] = 0.
        v[:,0,:] = 0.
        #rho[:,0,:] = rho0
        #p[:,0,:] = p0
    if pysim.PyMPI.ynproc:
        u[:,-1,:] = 0.
        v[:,-1,:] = 0.
        #rho[:,-1,:] = rho0
        #p[:,-1,:] = p0
    if pysim.PyMPI.x1proc:
        pass#u[0,:,:] = u0
        #v[0,:,:] = 0.
        #rho[0,:,:] = rho0
        #p[0,:,:] = p0
    return u,v*0.,rho,p
ss.addUserDefinedFunction('noslip',noslip)
# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)                  - ddy(:rho:*:v:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)   - ddy(:rhou:*:v:)
ddt(:rhov:) =  -ddx(:rhov:*:u:)                 - ddy(:rhov:*:v: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: - :tx:*:kappa:) - ddy( (:Et: + :p: - :tau:)*:v: - :ty:*:kappa: )
# Level set equation
# Conservative filter of the EoM
#[:u:,:v:,:rho:,:p:] = noslip(:u:,:v:,:rho:,:p:)
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
#[:u:,:v:,:w:] = ibmV( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [:u1:,:u2:,0.0] )
#:rho: = ibmS( :rho: , :phi:, [:gx:,:gy:,:gz:] )
#:p:   = ibmS( :p:   , :phi:, [:gx:,:gy:,:gz:] )
#bc.extrap(['rho','p','u'],['xn'])
#bc.const(['u'],['x1'],u0)
bc.const(['u'],['y1','yn'],0.0)
bc.const(['v'],['x1','xn','y1','yn'],0.0)
bc.const(['rho'],['x1','y1','yn'],rho0)
bc.const(['p'],['x1','y1','yn'],p0)
#[:u:,:v:,:rho:,:p:] = noslip(:u:,:v:,:rho:,:p:)
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
rad = sqrt( (meshx-pi)**2  +  (meshy-pi)**2 ) 
#:phi: = rad - pi/4.0
:rho: = 1.0 + 3d()
:p:  =  1.0 + 3d() #exp( -(meshx-1.5)**2/.25**2)*.1
#:u: = where( :phi:>0.5, mach * sqrt( :p: / :rho: * :gamma:) , 0.0 )
:u: = mach * sqrt( :p: / :rho: * :gamma:)
:u: = gbar( gbar( :u: ) )
:v: = 0.0 + 3d()
:Et: = :p:/( :gamma: - 1.0 ) + .5*:rho:*(:u:*:u: + :v:*:v:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)*.1
#[:gx:,:gy:,:gz:] = grad( :phi: )
#:gx: = gbar( :gx: )
#:gy: = gbar( :gy: )
"""
ic = ic.replace('mach',str(mach))

# Set the initial conditions
ss.setIC(ic)

# Write a time loop
time = 0.0
viz = True

# Approx a max dt and stopping time
tt = 1.5 #

# Start time loop
cnt = 1
viz_freq = 1

wvars = ['p','rho','u','v','Et']
ss.write( wvars )
CFL = 0.1
dt = ss.variables['dt'].data * CFL * .1

while tt > time:
    
    # Update the EOM and get next dt
    time = ss.rk4(time,dt)
    dt = min(ss.variables['dt'].data * CFL, 1.1*dt)
    dt = min(dt, (tt - time) )

    # Print some output
    ss.iprint("%s -- %s --- %f" % (cnt,time,dt)  )
    ss.write(wvars)
    cnt += 1
    #if viz and (not test):
    #    if (cnt%viz_freq == 1): 
    #        ss.write( wvars )
