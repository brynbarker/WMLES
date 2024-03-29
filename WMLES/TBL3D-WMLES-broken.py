import sys
import numpy 
from pyranda import pyrandaSim, pyrandaBC, pyrandaTimestep, pyrandaIBM, pyrandaTBL

try:
    local = int(sys.argv[1])
except: 
    local = 0

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

nx = 96#80
ny = 288#208 + 16
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
def wallMesh(i,j,k):
    if j < IB_y:
        y = Ly / float(4*IB_y)*float(j-IB_y)
    else:
        alpha = numpy.log( Ly + 1.0 ) / ( float(ny-IB_y) - 1.0 )
        y = numpy.exp( alpha* float( j-IB_y ) ) - 1.0
    
    x = Lx * ( float(i) / (nx-1) )
    z = Lz * ( float(k) / (nz-1) )

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
mesh_options['x1'] = [ 0.0    , 0.0    ,  0.0 ]
mesh_options['xn'] = [ Lx     , Ly     ,  Lz  ]
mesh_options['nn'] = [ nx     , ny     ,  nz  ]
mesh_options['pn'] = [6,18,1]
#mesh_options['pn'] = [5,14,1]
mesh_options['IB_y'] = IB_y

# Initialize a simulation object on a mesh
ss = pyrandaSim(problem,mesh_options)
ss.addPackage( pyrandaBC(ss) )
ss.addPackage( pyrandaIBM(ss) )
ss.addPackage( pyrandaTimestep(ss) )

TBL = pyrandaTBL(ss)
# Incoming BL data
BL_data = {}
#datadir = '/usr/workspace/barker38/scratch/TBL/
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

euler_3d += """
bc.const(['u','v','w'],['y1'],0.0)
#[:u:,:v:,:w:,:utau:,:Uwm:,:ubc:] = ibmWM( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [nuvar,hwm,deltax],LOCAL)
#[:u:,:v:,:w:,:utau:,:Uwm:,:ubc:] = ibmWM( [:u:,:v:,:w:], :phi:, [:gx:,:gy:,:gz:], [nuvar,hwm,deltax],LOCAL)
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
euler_3d = euler_3d.replace('LOCAL',str(local))
# Add the EOM to the solver
ss.EOM(euler_3d)

# Initialize variables
ic = """
:gamma: = gam0
#:R: = 1.0
#:cp: = :R: / (1.0 - 1.0/:gamma: )
#:cv: = :cp: - :R:
:rho: = 3d(rho0)
:p:   = 3d(p0)
:phi: = meshy
[:gx:,:gy:,:gz:] = grad( :phi: )
:gx: = gbar( :gx: )
:gy: = gbar( :gy: )
:gz: = gbar( :gz: )
:u: = mach * sqrt( :p: / :rho: * :gamma:)
bc.const( ['u','v','w'] , ['y1'] , 0.0 )
:u: = gbar( gbar( :u: ) )
:Et: = :p:/( :gamma: - 1.0 ) + .5*:rho:*(:u:*:u: + :v:*:v: + :w:*:w:)
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
:rhov: = :rho:*:w:
:cs:  = sqrt( :p: / :rho: * :gamma: )
:dt: = dt.courant(:u:,:v:,:w:,:cs:)
:tblRHOu: = 3d(rho0)
:tblRHOv: = 3d(rho0)
:tblRHOw: = 3d(rho0)
TBL.setup()
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
dt = ss.var('dt').data * CFL

TBL.DFinflow()
ss.parse(":rhou: = :rho:*:u:")

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
            

