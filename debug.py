import numpy as np
import matplotlib.pyplot as plt

utau = np.loadtxt('debug.dat')
fig = plt.figure()
plt.matshow(utau)
plt.show()
