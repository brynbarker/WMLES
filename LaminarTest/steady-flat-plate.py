import sys
from mpi4py import MPI
import numpy 
import scipy
import scipy.optimize
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM, pyrandaRestart

test = False
testName = None


## Define a mesh
Npts = 56#504
L = numpy.pi * 2.0  
dim = 2
gamma = 1.4
yp = L/10.

problem = 'steady-flat-plate'
restart = 0
#import pdb; pdb.set_trace()
if not restart:
    Lp = L * (Npts-1.0) / Npts
    mesh_options = {}
    mesh_options['coordsys'] = 0
    mesh_options['periodic'] = numpy.array([True, False, False])
    mesh_options['dim'] = 3
    mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
    mesh_options['xn'] = [ Lp   , Lp    ,  L-Lp ]
    mesh_options['nn'] = [ Npts , Npts ,  1  ]
    #mesh_options['pn'] = [2,2,1]
    #mesh_options['pn'] = [14,14,1]
    mesh_options['pn'] = [7,7,1]


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

    def get_BL_params(pysim, u, params):
        if len(params) == 3:
            return params

        Uinf = numpy.max(u)
        d99 = 1.5
        nu = d99*Uinf/5e4
        hwm = 0.1*d99 
        deltax = 0.15*hwm
        return [nu, hwm, deltax]
        #x1, y1, z1 = pysim.mesh.options['x1']
        #xn, yn, zn = pysim.mesh.options['xn']
        #ystart = yn - (yn-y1)/pysim.PyMPI.py
        #num_yn = numpy.array([0.0])
        #if pysim.PyMPI.ynproc:
        #    Uinf = numpy.mean(u[:,-1,0])
        #    num_yn[0] = 1.
        #    domain = numpy.linspace(ystart,yn,u.shape[1])
        #    d99 = numpy.interp(0.99*Uinf, u[0,:,0], domain)
        #    nu = d99*Uinf/5e4
        #    hwm = 0.1*d99 
        #else:
        #    nu, hwm = 0., 0.

        nu = pysim.PyMPI.comm.allreduce(numpy.array([nu]),op=MPI.SUM) / \
             pysim.PyMPI.comm.allreduce(num_yn,op=MPI.SUM)
        hwm = pysim.PyMPI.comm.allreduce(numpy.array([hwm]),op=MPI.SUM) / \
              pysim.PyMPI.comm.allreduce(num_yn,op=MPI.SUM)
        
        deltax = 0.15*hwm[0]
        if pysim.PyMPI.master == 1:
            print(nu[0],hwm[0],deltax)
            print(pysim.dx,pysim.dy,pysim.dz)
            import pdb; pdb.set_trace()
        return [nu[0], hwm[0], deltax]

    ss.addUserDefinedFunction("get_BL_params",get_BL_params)

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
:BLparams: = get_BL_params(:u:,:BLparams:)
#[:u:,:v:,:w:] = ibmV( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [:u1:,:u2:,0.0] )
[:u:,:v:,:w:,:utau:,:Uwm:,:ubc:] = ibmWM( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], :BLparams:,0,[:u1:,:u2:,0.0] )
:rho: = ibmS( :rho: , :phi:, [:gx:,:gy:,:gz:] )
:p:   = ibmS( :p:   , :phi:, [:gx:,:gy:,:gz:] )
bc.extrap(['rho','p','u'],['y1','yn'])
#bc.const(['u'],['x1','y1','yn'],u0)
bc.const(['v'],['y1','yn'],0.0)
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
:BLparams: = []
:gamma: = 1.4
:R: = 1.0
:cp: = :R: / (1.0 - 1.0/:gamma: )
:cv: = :cp: - :R:
:phi: = meshy - pi/5.
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
    CFL = 1.0
    dt = ss.variables['dt'].data * CFL * .1
if restart:
    [ ss , local_vars] = pyrandaRestart( problem )  # Restore simulation object
    locals().update(  local_vars )                  # Restore local vars saved
    time = ss.time                                  # Resume time
    dt   = ss.deltat                                # Last deltat

viz = True

# Approx a max dt and stopping time
tt = 500.0 #

# Start time loop
cnt = 1
viz_freq = 10
max_steps = 300

wvars = ['p','rho','u','v','phi']
if not test:
    ss.write( wvars )
CFL = 1.0

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
        if (cnt%viz_freq == 1): 
            ss.write( wvars )
            ss.writeRestart()
    if cnt > 50:
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
