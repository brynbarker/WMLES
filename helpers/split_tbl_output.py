import numpy as np
import os

datadir = "TBL3D_output"

datafiles = [f for f in os.listdir(datadir) if f.endswith('dat')]

localdir = os.path.join(datadir,'local-wmles') 
globaldir = os.path.join(datadir,'global-wmles') 

for f in datafiles:
    localdata = []
    globaldata = []

    fname = os.path.join(datadir,f)
    with open(fname,'r') as fdata:
        lines = fdata.readlines()

    localdata.append(lines[0])
    globaldata.append(lines[0])
    localtime = 0.0

    for line in lines[1:]:
        time = float(line.split()[0])
        if time > localtime:
            localdata.append(line)
            localtime = time
        else:
            globaldata.append(line)

    with open(os.path.join(localdir,f),'w') as localf:
        localf.write('\n'.join(localdata)) 
    with open(os.path.join(globaldir,f),'w') as globalf:
        globalf.write('\n'.join(globaldata)) 

    print(len(lines),len(localdata)+len(globaldata)-1)
