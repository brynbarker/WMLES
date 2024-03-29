import numpy as np
import sys
from mesh_sizing import run

method = str(sys.argv[1])

ib = True if method=='wmles' else False
    

Nvals = run(method,ib=ib)
if Nvals is not None:
    nx, ny, nz = Nvals
    print '\t'.join([str(v) for v in [nx,ny,nz,nx*nx*nz]])
else:
    raise ValueError('can\'t get mesh sizing')

try: 
    hi = int(sys.argv[2])
except:
    hi = 64
low = 8
ops = np.arange(low,hi+1)

for min_num_nodes in range(256):
    num_patches = float(nx*ny*nz)/float(hi**3)
    if num_patches <= min_num_nodes * 56:
        break

print '-'*100
print 'N\t(Nx, Ny, Nz)\t(pnx, pny, pnz)\tPatch size\tntasks\tLeftover Processors'
print '-'*100

success = False
combos = []
sizes = []
leftovers = []
for nodes in range(min_num_nodes, 520):
    max_num_procs = nodes*56
    for Nx in range(nx,nx+25):
        for Ny in range(ny,ny+25):
            for Nz in range(nz,nz+25):

                pnx_ops = [Nx/x for x in ops if Nx%x==0]
                pny_ops = [Ny/y for y in ops if Ny%y==0]
                pnz_ops = [Nz/z for z in ops if Nz%z==0]
                
                if method == 'wmles':
                    pny_ops = [Ny/y+1 for y in ops[4:] if Ny%y==0]

                for pnx in pnx_ops:
                    xdim = Nx/pnx
                    for pny in pny_ops:
                        ydim = Ny/pny
                        if method == 'wmles': 
                            ydim = Ny/(pny-1)
                        for pnz in pnz_ops:
                            zdim = Nz/pnz

                            num_procs = pnx*pny*pnz
                            p_size = xdim*ydim*zdim
                            diff = max_num_procs-num_procs
                            
                            if num_procs <= max_num_procs and diff < 20:
                                L=[nodes,(Nx,Ny,Nz),(pnx,pny,pnz),(xdim,ydim,zdim),num_procs,diff,p_size]
                                combos.append(L)
                                sizes.append(p_size)
                                leftovers.append(diff)
                                success = True

    if success:
        break

smallest = min(sizes)
small_combos = [L for L in combos if L[-1]==smallest]
low_leftovers = min([L[-2] for L in small_combos])
for L in small_combos:
    if L[-2] == low_leftovers:
        print '\t'.join([str(v) for v in L[:-1]])
            
