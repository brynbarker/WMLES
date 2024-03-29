import sys
import numpy 
from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM, pyrandaTBL, pyrandaRestart

try:
    local = int(sys.argv[1])
except: 
    local = 0
restart = True
## Define a mesh
problem = 'myTBL'

nx = 100
ny = 64
nz = 16

Lx = 10.0
Ly = 5.0
Lz = 0.5


problem = 'tbl-wmles-'+'local'*local+'global'*(1-local)
nx = 400
ny = 128
nz = 32

# from get-pn.py 
nx = 78
ny = 204+12
nz = 8

IB_y = 12

nx = 80
ny = 208 + 16
nz = 16
IB_y = 16

Lx = 25.0
Ly = 10.0
Lz = 1.0

# delta99 = 3.07 so we need dy <= 0.1ish
#nu = 7.415414466563401e-05 # Re ~ 4e5
##nu = 3.7077072332817006e-12
#hwm = 0.3070183607238605
#deltax = 0.04605275410857908

Re = 5.e4
uinf = 1.20985794067
d99 = 3.33297650325
nu = d99*uinf/Re
hwm = 0.1*d99
deltax = 0.15*hwm

utau, nu = 0.040894041383302195, 8.06485617707006e-05


delBL = 2.5
#Re = 1.0e12
mach = 0.3
p0 = 1.0
rho0 = .1
gamma = 1.4
u0 = mach * numpy.sqrt( p0 / rho0 * gamma)
mu0 = u0 * delBL * rho0 / Re

CFL = 0.4

#Re = u0 del0 den / ( mu0

[ ss , local_vars] = pyrandaRestart( problem )  # Restore simulation object
locals().update(  local_vars )                  # Restore local vars saved
ss.pyrandaTBL.setup(restart=True)
time = ss.time                                  # Resume time
dt   = ss.deltat                                # Last deltat


# Approx a max dt and stopping time
tt = 60.0 #

# Start time loop
viz_freq = 250
pvar = 'umag'

#TBL.DFinflow()
wvars = ['p','rho','u','v','w','Et','cs']

#for i in range(1,nx):
#    ss.var('u').data[i,:,:] = ss.var('u').data[0,:,:]



ss.write(wvars)
ss.writeToFile(problem, utau,nu,first=True)

if 1:
    while tt > time:

        # Update the EOM and get next dt
        time = ss.rk4(time,dt)
        dt = min( ss.variables['dt'].data * CFL, dt)
        dt = min(dt, (tt - time) )
        dtCFL = ss.variables['dtC'].data
    
        # Print some output
        ss.iprint("%s -- %.6e --- %.4e --- CFL: %.4e" % (ss.cycle,time,dt,dt/dtCFL)  )

        if (ss.cycle%viz_freq == 0) :
            ss.write(wvars)
            ss.writeRestart()
            ss.writeToFile(problem,utau,nu)

#ss.writeRestart()
            

