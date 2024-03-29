import subprocess
subprocess.check_call([sys.executable,'-m','pip','install','numpy'])
subprocess.check_call([sys.executable,'-m','pip','install','matplotlib'])

import numpy as np
import matplotlib.pyplot as plt

def plot_profile():
    y,u = np.loadtxt('u-profile.dat')
    uinf = u[-1]
    d99 = np.interp(0.99*uinf, u, y)
    Re = 5e4
    nu = d99*uinf/Re
    hwm = 0.1*d99
    deltax = 0.15*hwm
    plt.semilogx(y,u)
    plt.semilogx(hwm, np.interp(hwm,y,u),'*')
    plt.semilogx(d99, 0.99*uinf, '*')
    plt.title('Uinf = {}'.format(uinf))
    plt.show()

plot_profile()
