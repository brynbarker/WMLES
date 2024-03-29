import sys
import numpy 
from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM, pyrandaTBL, pyrandaRestart
import os

problem = 'tbl-wrles-correct-mesh'
DataDir = "/p/lustre1/barker38/WRLES"
DataDir = os.path.join(DataDir,problem)

delBL = 2.5
Re = 5.e4
mach = 0.3
p0 = 1.0
rho0 = .1
gamma = 1.4
u0 = mach * numpy.sqrt( p0 / rho0 * gamma)
mu0 = u0 * delBL * rho0 / Re

utau, nu = 0.040894041383302195, 8.06485617707006e-05

[ ss , local_vars] = pyrandaRestart( problem )  # Restore simulation object
locals().update(  local_vars )                  # Restore local vars saved
BL_data = {}
BL_data['umean']  = '../TBL-Analysis/TBL_data/umean.dat'
BL_data['uumean'] = '../TBL-Analysis/TBL_data/uumean.dat'
BL_data['vvmean'] = '../TBL-Analysis/TBL_data/vvmean.dat'
BL_data['wwmean'] = '../TBL-Analysis/TBL_data/wwmean.dat'
BL_data['uvmean'] = '../TBL-Analysis/TBL_data/uvmean.dat'
# Setup the 2d filters ( 6 total.  Inner (u,v,w) and outer (u,v,w) )
ss.packages['TBL'].UIx = 10
ss.packages['TBL'].UIy = [ 10, 15 ]
ss.packages['TBL'].UIz = 10
ss.packages['TBL']     
ss.packages['TBL'].VIx = 8
ss.packages['TBL'].VIy = [ 15, 20 ]
ss.packages['TBL'].VIz = 10
ss.packages['TBL']   
ss.packages['TBL'].WIx = 8
ss.packages['TBL'].WIy = [ 10, 20 ]
ss.packages['TBL'].WIz = 10
ss.packages['TBL']
ss.packages['TBL'].Nbuff = 20 * 2
ss.packages['TBL'].BL_data = BL_data
ss.packages['TBL'].U_in    = u0 / 18.93   # Max val data / mean flow u0 in pyranda
ss.packages['TBL'].del_BL  = 2.5 / 100.0  # Data val at BL / physical location
ss.packages['TBL'].tauX    = 2.5
ss.packages['TBL'].setup(restart=True)
time = ss.time                                  # Resume time
dt   = ss.deltat                                # Last deltat

viz = True

# Approx a max dt and stopping time
tt = 60.0 #

# Start time loop
viz_freq = 250
pvar = 'umag'

#TBL.DFinflow()
wvars = ['p','rho','u','v','w','Et','cs']

if 1:
    CFL = 0.4
    #dt = ss.var('dt').data * CFL
    while tt > time:

        # Update the EOM and get next dt
        time = ss.rk4(time,dt)
        dt = min( ss.variables['dt'].data * CFL, dt)
        dt = min(dt, (tt - time) )
        dtCFL = ss.variables['dtC'].data
    
        # Print some output
        ss.iprint("%s -- %.6e --- %.4e --- CFL: %.4e" % (ss.cycle,time,dt,dt/dtCFL)  )

        if (ss.cycle%viz_freq == 0) :
            ss.write(wvars,path=DataDir)
            ss.writeRestart(path=DataDir)
            ss.writeToFile(problem,utau, nu)

#ss.writeRestart()
            

