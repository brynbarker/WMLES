import sys
import numpy 
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM


test = False
testName = None


## Define a mesh
Npts = 64
L = numpy.pi * 2.0  
dim = 2
gamma = 1.4

problem = 'ellipse-ib-test'

Lp = L * (Npts-1.0) / Npts
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = numpy.array([False, False, False])
mesh_options['dim'] = 3
mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
mesh_options['xn'] = [ Lp   , Lp    ,  Lp ]
mesh_options['nn'] = [ Npts , Npts ,  1  ]


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

def get_phi(pysim, x_in, y_in):
    r = x_in.shape[0]
    A = numpy.array([[numpy.cos(numpy.pi/12), -numpy.sin(numpy.pi/12)],[numpy.sin(numpy.pi/12), numpy.cos(numpy.pi/12)]])
    a = numpy.pi/4
    b = numpy.pi/8
    scale = numpy.array([[1/a,0],[0,1/b]])

    def E(theta):
        v = numpy.array([a*numpy.cos(theta),b*numpy.sin(theta)])
        coord = A.T.dot(v) + numpy.pi
        if coord.size > 2:
            return coord
        return coord.flatten()

    def dE(theta):
        v = numpy.array([-a*numpy.sin(theta),b*numpy.cos(theta)])
        grad = A.T.dot(v)
        if grad.size > 2:
            return grad
        return grad.flatten()

    def angle(theta):
        return numpy.sum( (x-E(theta))*dE(theta), axis=0 ) 

    phi = numpy.zeros_like(x_in)
    for ind in range(r):
        x = numpy.vstack((x_in[ind],y_in[ind])).reshape((2,-1))
        d = numpy.inf*numpy.ones((r))
        for guess in numpy.linspace(0,numpy.pi*2,5)[:-1]:
            theta = scipy.optimize.root(angle, guess*numpy.ones((r))).x
            new_d = numpy.linalg.norm(E(theta)-x, axis=0)

            mask = new_d < d
            d[mask] = new_d[mask]

        s = numpy.sign( numpy.linalg.norm( scale.dot(A.dot(x-numpy.pi)),axis=0 ) - 1.0 )
        phi[ind] = (d*s).reshape((1,-1)).T

    return phi

ss.addUserDefinedFunction("get_phi",get_phi)

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
[:u:,:v:,:w:] = ibmV( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [:u1:,:u2:,0.0] )
:rho: = ibmS( :rho: , :phi:, [:gx:,:gy:,:gz:] )
:p:   = ibmS( :p:   , :phi:, [:gx:,:gy:,:gz:] )
bc.extrap(['rho','p','u'],['xn'])
bc.const(['u'],['x1','y1','yn'],u0)
bc.const(['v'],['x1','xn','y1','yn'],0.0)
bc.const(['rho'],['x1','y1','yn'],rho0)
bc.const(['p'],['x1','y1','yn'],p0)
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
:phi: = get_phi(meshx, meshy)
:rho: = 1.0 + 3d()
:p:  =  1.0 + 3d() #exp( -(meshx-1.5)**2/.25**2)*.1
:u: = where( :phi:>0.5, mach * sqrt( :p: / :rho: * :gamma:) , 0.0 )
#:u: = mach * sqrt( :p: / :rho: * :gamma:)
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
tt = 1.5 #

# Start time loop
cnt = 1
viz_freq = 25

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
    if viz and (not test):
        if (cnt%viz_freq == 1): 
            ss.write( wvars )
                   

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
