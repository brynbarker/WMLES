from __future__ import print_function
import re
import sys
import time
import numpy 
import matplotlib.pyplot as plt
from matplotlib import cm

from pyranda import pyrandaSim, pyrandaBC



test = False

testName = None

Npts = 20


## Define a mesh
L = numpy.pi * 2.0  
gamma = 1.4
dim = 2

problem = 'flow'

Lp = L * (Npts-1.0) / Npts
mesh_options = {}
mesh_options['coordsys'] = 0
mesh_options['periodic'] = numpy.array([True, False, True])
mesh_options['dim'] = 3
mesh_options['x1'] = [ 0.0 , 0.0  ,  0.0 ]
mesh_options['xn'] = [ Lp   , Lp    ,  Lp ]
mesh_options['nn'] = [ Npts, Npts ,  1  ]
#mesh_options['pn'] = [7,7,1]

# Initialize a simulation object on a mesh
ss = pyrandaSim(problem,mesh_options)
ss.addPackage( pyrandaBC(ss) )

# Define the equations of motion
eom ="""
# Primary Equations of motion here
ddt(:rho:)  =  -ddx(:rho:*:u:)                  - ddy(:rho:*:v:)
ddt(:rhou:) =  -ddx(:rhou:*:u: + :p: - :tau:)   - ddy(:rhou:*:v:)
ddt(:rhov:) =  -ddx(:rhov:*:u:)                 - ddy(:rhov:*:v: + :p: - :tau:)
ddt(:Et:)   =  -ddx( (:Et: + :p: - :tau:)*:u: ) - ddy( (:Et: + :p: - :tau:)*:v: )
# Conservative filter of the EoM
:rho:       =  fbar( :rho:  )
:rhou:      =  fbar( :rhou: )
:rhov:      =  fbar( :rhov: )
:Et:        =  fbar( :Et:   )
# Update the primatives and enforce the EOS
:u:         =  :rhou: / :rho:
:v:         =  :rhov: / :rho:
:p:         =  ( :Et: - .5*:rho:*(:u:*:u: + :v:*:v:) ) * ( :gamma: - 1.0 )
# Artificial bulk viscosity (old school way)
:div:       =  ddx(:u:) + ddy(:v:)
:beta:      =  gbar(abs(ring(:div:))) * :rho: * 7.0e-2
:tau:       =  :beta:*:div:
# Apply constant BCs
bc.extrap(['rho','Et'],['x1','xn','y1','yn'])
bc.const(['u','v'],['y1'],0.0)
bc.extrap(['u','v'],['yn'])
#bc.const(['u'],['x1'],1.0)
"""

print(eom)

# Add the EOM to the solver
ss.EOM(eom)


# Initialize variables
ic = """
rad = sqrt( (meshx-pi)**2  +  (meshy-pi)**2 )
:gamma:=1.4
:rho:=1.+3d()
:Et: = 1+3d()
:u: = 1.+3d()#sin(meshy)
#:u: = gbar ( gbar ( :u:) )
:rhou: = :rho:*:u:
:rhov: = :rho:*:v:
"""

# Set the initial conditions
ss.setIC(ic)
    
# Length scale for art. viscosity
# Initialize variables
x = ss.mesh.coords[0].data
y = ss.mesh.coords[1].data
z = ss.mesh.coords[2].data


# Write a time loop
time = 0.0
viz = True

# Approx a max dt and stopping time
v = 1.0
dt_max = v / ss.mesh.nn[0] * 0.75
tt = 10.#L/v * .125 #dt_max

# Mesh for viz on master
xx   =  ss.PyMPI.zbar( x )
yy   =  ss.PyMPI.zbar( y )
ny = ss.PyMPI.ny

# Start time loop
dt = dt_max
cnt = 1
viz_freq = 25
wvars = ['p','rho','u','v','Et']
ss.write(wvars)
while tt > time:

    # Update the EOM and get next dt
    time = ss.rk4(time,dt)
    dt = min(dt_max, (tt - time) )

    
    # Print some output
    ss.iprint("%s -- %s" % (cnt,time)  )
    ss.write(wvars)
    cnt += 1
    #if viz and (not test):
    #    if (ss.PyMPI.master and (cnt%viz_freq == 1)) and True:
    #        ss.write(wvars)

# Curve test.  Write file and print its name at the end
if test:
    v = ss.PyMPI.zbar( ss.variables[pvar].data )
    v1d =  v[:,int(ny/2)]
    x1d = xx[:,int(ny/2)]
    fname = testName + '.dat'
    numpy.savetxt( fname  , (x1d,v1d) )
    print(fname)
