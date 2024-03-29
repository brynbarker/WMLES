import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

def run(which='all',ib=False):
    y,u = np.loadtxt('/usr/workspace/barker38/scratch/helpers/u-profile.dat')
    uinf = u[-1]
    d99 = np.interp(0.99*uinf, u, y)
    Re = 5e4
    nu = d99*uinf/Re
    hwm = 0.1*d99
    deltax = 0.15*hwm

    Uwm = np.interp(hwm,y,u)
    guess = 1e-14
    kappa = 0.41
    A = 5.
    def g(ut):
        val = Uwm/ut - 1/kappa*np.log(hwm*ut/nu)-A
        valprime = -1/ut * (Uwm/ut - 1/kappa)
        return val, np.diag(valprime)
    sol = scipy.optimize.root(g,guess,jac=True)
    utau = sol.x[0]
    print(utau,nu,hwm)

    #Re = 5e4
    #uinf = 1.20985794067
    #d99 = 3.33297650325
    #nu = d99*uinf/Re
    #hwm = 0.1*d99
    #deltax = 0.15*hwm
    #utau = 0.04089404

    Ly = 10.
    Lx = 25.
    Lz = 1.

    def wallMesh(j,n,ind):

        if ind == 1:
            alpha = np.log( Ly + 1.0 ) / ( float(n) - 1.0 )
            y = np.exp( alpha* float( j ) ) - 1.0
            return y
        
        if ind==0: 
            x = Lx * ( float(j) / (n-1) )
            return x
        if ind == 2:
            z = Lz * ( float(j) / (n-1) )
            return z
        return x,y,z
    #dom = np.arange(1000)
    #ran = [wallMesh(d,1000,1) for d in dom]
    #plt.plot(dom,ran)
    #plt.show()
    bestNlist = []

    s = utau/nu
    # nu = 8.06485617705e-05
    maxN = 2000
    bestN = False
    flags =['dns','wrles','wmles']
    for flag in flags:
        prod = 1.
        if which=='all':
            print '-'*50
            print flag
        if flag=='wmles':
            yin = 0.3*hwm
            yout = 0.04*d99
            xin = hwm
            xout= .1*d99
            zin = .8*hwm
            zout = .08*d99
        if flag=='wrles':
            yin = 1./s
            yout = 0.1*d99
            xin = 18./s
            xout= 1.
            zin = 20./s
            zout = 1.
        if flag=='dns':
            yin = 1./s
            yout = 1.
            xin = 20./s
            xout= 1.
            zin = 10./s
            zout = 1.

        inners = [xin,yin,zin]
        outers = [xout,yout,zout]
        lengths = [Lx,Ly,Lz]

        dims = ['x','y','z']

        for ind,(dim, inner, outer, length) in enumerate(zip(dims,inners,outers,lengths)):
            if which=='all':
                print dim, (inner,outer,length)
            bestN = False
            minN = 5 + 40*ib
            for N in range(minN,maxN):
                Nib = N - 40*ib
                loc = wallMesh(1,Nib,ind)
                dx = loc
                if N == maxN-1 and which=='all':print dx,inner

                out1 = wallMesh(Nib-1,Nib,ind)
                out2 = wallMesh(Nib-2,Nib,ind)
                out_dx = out1-out2
                if N == maxN-1 and which=='all':print out_dx, outer

                if loc < inner and out_dx < outer:
                    prod *= N
                    if which=='all':
                        print 'best N = '+str(N+1)+'\tdelta '+dim+' = '+str(dx)+ '\tdelta {} + = {}'.format(dim,dx*s)+'   '+str(out_dx)
                    elif which==flag: 
                        bestNlist.append(N+1)
                    bestN = True
                    if ind != 1:
                        break
                    prev = loc

                    if flag != 'wmles': break
                    near_wall = [True]
                    j=1
                    while dx < inner:
                        loc = wallMesh(j,Nib+1,ind)
                        dx = loc-prev
                        near_wall.append(dx< inner)
                        j += 1
                    if which=='all':print near_wall, j

                    break
        

            if not bestN: print 'need to try larger N values'
            if which=='all':print '\n'
        if which=='all':print prod
    if which=='all':print '-'*50
    print(which,bestNlist)
    if len(bestNlist)==3: return bestNlist
    else: return None

if __name__ == "__main__":
    run()
