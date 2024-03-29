import sys
import numpy 
from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM, pyrandaTBL
import os

storageDir = '/p/lustre1/barker38/WMLES'

## Define a mesh
problem = 'tbl-3d-wrles-ib'
path = os.path.join(storageDir,problem)
nx = 200
ny = 400
nz = 20

Lx = 25.0
Ly = 10.0
Lz = 1.0


delBL = 2.5
Re = 5.e4
mach = 0.3
p0 = 1.0
rho0 = .1
gamma = 1.4
u0 = mach * numpy.sqrt( p0 / rho0 * gamma)
mu0 = u0 * delBL * rho0 / Re

utau, nu = 0.040894041383302195, 8.06485617707006e-05

IB_y = 20
#Re = u0 del0 den / ( mu0
def wallMesh(i,j,k):
    if j < IB_y:
        y = Ly / float(4*IB_y)*float(j-IB_y)
    else:
        alpha = numpy.log( Ly + 1.0 ) / ( float(ny-IB_y) - 1.0 )
        y = numpy.exp( alpha* float( j-IB_y ) ) - 1.0
    
    x = Lx * ( float(i) / (nx-1) )
    z = Lz * ( float(k) / (nz-1) )
    return x,y,z

    alpha = numpy.log( Ly + 1.0 ) / ( float(ny) - 1.0 )
    
    x = Lx * ( float(i) / (nx-1) )
    y = numpy.exp( alpha* float( j ) ) - 1.0
    z = Lz * ( float(k) / (nz-1) )
    return x,y,z

mesh_options = {}
mesh_options['coordsys'] = 3
mesh_options['function'] = wallMesh
mesh_options['periodic'] = numpy.array([False, False, True])
mesh_options['periodicGrid'] = True  # (grid linked to flow periodicity)
mesh_options['x1'] = [ 0.0    , -2.5    ,  0.0 ]
mesh_options['xn'] = [ Lx     , Ly     ,  Lz  ]
mesh_options['nn'] = [ nx     , ny     ,  nz  ]
mesh_options['pn'] = [10,20,1]


# Initialize a simulation object on a mesh
ss = pyrandaSim(problem,mesh_options)
ss.addPackage( pyrandaBC(ss) )
ss.addPackage( pyrandaTimestep(ss) )
ss.addPackage( pyrandaIBM(ss) )

TBL = pyrandaTBL(ss)
# Incoming BL data
BL_data = {}
BL_data['umean']  = '../TBL-Analysis/TBL_data/umean.dat'
BL_data['uumean'] = '../TBL-Analysis/TBL_data/uumean.dat'
BL_data['vvmean'] = '../TBL-Analysis/TBL_data/vvmean.dat'
BL_data['wwmean'] = '../TBL-Analysis/TBL_data/wwmean.dat'
BL_data['uvmean'] = '../TBL-Analysis/TBL_data/uvmean.dat'
# Setup the 2d filters ( 6 total.  Inner (u,v,w) and outer (u,v,w) )
TBL.UIx = 10
TBL.UIy = [ 10, 15 ]
TBL.UIz = 10
        
TBL.VIx = 8
TBL.VIy = [ 15, 20 ]
TBL.VIz = 10
      
TBL.WIx = 8
TBL.WIy = [ 10, 20 ]
TBL.WIz = 10

TBL.Nbuff = 20 * 2
TBL.BL_data = BL_data
TBL.U_in    = u0 / 18.93   # Max val data / mean flow u0 in pyranda
TBL.del_BL  = 2.5 / 100.0  # Data val at BL / physical location
TBL.tauX    = 2.5
ss.addPackage( TBL )


# Import 3D Euler-curvilinear
#from equation_library import euler_3d
from equation_library import euler_3d_dir as euler_3d

Re = 5.e4
uinf = 1.20985794067
d99 = 3.33297650325
nu = d99*uinf/Re
hwm = 0.1*d99
deltax = 0.15*hwm

euler_3d += """
bc.const(['v','w'],['y1'],0.0)
[:u:,:v:,:w:] = ibmV( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [:u1:,:u2:,0.0])
bc.extrap(['u'],['y1'])
bc.extrap(['rho','p'],['y1','yn'])
bc.extrap(['rho','p'],['xn'])
bc.const(['rho'],['x1'],rho0)
bc.const(['p'],['x1'],p0)
TBL.inflow()
# Sponge outflow
:wgt: = ( 1.0 + tanh( (meshx-Lx*(1.0-0.025))/ (.025*Lx) ) ) * 0.5
:u: = :u:*(1-:wgt:) + gbarx(:u:)*:wgt:
:v: = :v:*(1-:wgt:) + gbarx(:v:)*:wgt:
:w: = :w:*(1-:wgt:) + gbarx(:w:)*:wgt:
bc.extrap(['u','v','w'],['xn'])
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:rhow: = :rho:*:w:
:Et:   =  .5*:rho:*(:u:*:u: + :v:*:v: + :w:*:w:)  + :p: / ( :gamma: - 1.0 )
"""

euler_3d = euler_3d.replace('mu0',str(mu0))
euler_3d = euler_3d.replace('rho0',str(rho0))
euler_3d = euler_3d.replace('p0',str(p0))
euler_3d = euler_3d.replace('Lx',str(Lx))
euler_3d = euler_3d.replace('nuvar',str(nu))
euler_3d = euler_3d.replace('hwm',str(hwm))
euler_3d = euler_3d.replace('deltax',str(deltax))

# Add the EOM to the solver
ss.EOM(euler_3d)

# Initialize variables
ic = """
:gamma: = gam0
#:R: = 1.0
#:cp: = :R: / (1.0 - 1.0/:gamma: )
#:cv: = :cp: - :R:
:phi: = meshy
[:gx:,:gy:,:gz:] = grad( :phi: )
:gx: = gbar( :gx: )
:gy: = gbar( :gy: )
:gz: = gbar( :gz: )
:rho: = 3d(rho0)
:p:   = 3d(p0)
:ptmp: = 3d(p0)
:u: = mach * sqrt( :p: / :rho: * :gamma:)
bc.const( ['u','v','w'] , ['y1'] , 0.0 )
:u: = gbar( gbar( :u: ) )
:Et: = :p:/( :gamma: - 1.0 ) + .5*:rho:*(:u:*:u: + :v:*:v: + :w:*:w:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:rhov: = :rho:*:w:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)
TBL.setup()
:tblRHOu: = 3d(rho0)
:tblRHOv: = 3d(rho0)
:tblRHOw: = 3d(rho0)
# Mesh metrics from parcops
:dAdx: = meshVar('dAx')
:dAdy: = meshVar('dAy')
:dAdz: = meshVar('dAz')
:dBdx: = meshVar('dBx')
:dBdy: = meshVar('dBy')
:dBdz: = meshVar('dBz')
:dCdx: = meshVar('dCx')
:dCdy: = meshVar('dCy')
:dCdz: = meshVar('dCz')
:detJ: = meshVar('dtJ')
:dA: = meshVar('d1')
:dB: = meshVar('d2')
:dC: = meshVar('d3')
"""
ic = ic.replace('mach',str(mach))
ic = ic.replace('rho0',str(rho0))
ic = ic.replace('p0',str(p0))
ic = ic.replace('gam0',str(gamma))
ic = ic.replace('BL_data',str(BL_data))

# Set the initial conditions
ss.setIC(ic)
    

# Write a time loop
time = 0.0
viz = True

# Approx a max dt and stopping time
tt = 60.0 #

# Start time loop
viz_freq = 250
pvar = 'umag'

#TBL.DFinflow()
wvars = ['p','rho','u','v','w','Et','cs']

#for i in range(1,nx):
#    ss.var('u').data[i,:,:] = ss.var('u').data[0,:,:]


TBL.DFinflow()

ss.parse(":rhou: = :rho:*:u:")
ss.write(wvars,path=path)
ss.writeToFile(problem,utau,nu,first=True)

if 1:
    CFL = 0.2
    dt = ss.var('dt').data * CFL
    #dt = 1.7845e-03
    while tt > time:

        # Update the EOM and get next dt
        #dt = 1.7845e-03
        time = ss.rk4(time,dt)
        #time = time + dt
        #ss.time = time*1.0
        #TBL.DFinflow()
        #ss.cycle += 1
        dt = min( ss.variables['dt'].data * CFL, dt)
        dt = min(dt, (tt - time) )
        dtCFL = ss.variables['dtC'].data
    
        # Print some output
        ss.iprint("%s -- %.6e --- %.4e --- CFL: %.4e" % (ss.cycle,time,dt,dt/dtCFL)  )

        if (ss.cycle%viz_freq == 0) :
            ss.write(wvars,path=path)
            ss.writeRestart(path=path)
            ss.writeToFile(problem,utau, nu)

#ss.writeRestart()
            
