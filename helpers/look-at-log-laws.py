import numpy as np
import matplotlib.pyplot as plt
#
#wrles_data = '/usr/workspace/barker38/scratch/TBL-Analysis/iTBL_output_data.dat'
#gwmles_data = '/usr/workspace/barker38/scratch/TBL-Analysis/TBL3D_output'
#lwmles_data = '/usr/workspace/barker38/scratch/TBL-Analysis/TBL3D_output'
#
#filename = lambda f: '/usr/workspace/barker38/scratch/TBL-Analysis/tbl-{}-output.dat'.format(f)
#
#files = ['wmles-local','wmles-global','wrles']
#colors = ['C0','C1','C2']
#
#fig = plt.figure()
#for find,f in enumerate(files):
#    with open(filename(f),'r') as output:
#        file_data = output.readlines()
#        labels = file_data[0].split()
#        data = np.array([d.split() for d in file_data[1:]],dtype=float)
#        for ind,label in enumerate(labels[1:]):
#            plt.subplot(1,5,ind+1)
#            plt.plot(data[:,0],data[:,ind+1],c=colors[find],label=f)
#            if find == len(files)-1:
#                plt.xlabel('time')
#                plt.legend()
#                plt.title(label)

filename = '/usr/workspace/barker38/scratch/TBL-Analysis/TBL_data/umean.dat'
#data = np.loadtxt(filename)
with open(filename,'r') as tmp:
    data = tmp.readlines()[1:]
print(data)
data = [d.split(', ') for d in data]
yplus = [float(d[0]) for d in data]
uplus = [float(d[1].strip('\n')) for d in data]
plt.semilogx(yplus,uplus)
        
plt.show() 


#            U_wm = numpy.sqrt(u_h*u_h + v_h*v_h + w_h*w_h)
#      
#            # solve for u_tau
#            kappa = 0.41
#            A = 5.0
#      
#            # root finding for whole array at once isn't working for some reason
#            u_tau = numpy.zeros_like(u_h)
#            for j in range(0,len(U_wm),40):
#                u_wm = U_wm[j:j+40]
#                guess = 1e-14*numpy.ones_like(u_wm)
#                def g(ut):
#                    val = u_wm/ut - 1/kappa*numpy.log(hwm*ut/nu)-A
#                    valprime = -1/ut * (u_wm/ut - 1/kappa)
#                    return val, numpy.diag(valprime)
#                sol = scipy.optimize.root(g,guess,jac=True)
