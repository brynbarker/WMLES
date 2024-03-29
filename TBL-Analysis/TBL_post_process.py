import numpy as np
import visit_utils as visit
import os

datafile = 'myTBL_long/pyranda.visit'
data_names = ['uumean','uvmean','vvmean','wwmean']
data_eqs = ['u*u','u*v','v*v','w*w']
outputdir = 'TBL_data_postproc'

metadata = visit.GetMetaData(datafile)
num_scalars = metadata.GetNumScalars()
num_vectors = metadata.GetNumVectors()

visit.OpenDatabase(datafile)
visit.AddPlot("Pseudocolor", 'u')
visit.DrawPlots()
visit.SetQueryOutputToValue()
grid_info = np.array(visit.Query('Grid Information')).reshape((-1,5))
num_nodes = np.prod(np.min(grid_info[:,2:],axis=0)-1.)*grid_info.shape[0]

output = {d:[] for d in data_names}

nts = visit.TimeSliderGetNStates()
for ts in range(nts):
    visit.TimeSliderSetState(ts)
    tval = visit.Query("Time")


    for (name, eq) in zip(data_names, data_eqs):
        visit.DefineScalarExpression(name, eq)
        visit.ChangeActivePlotsVar(name)
        var_sum = visit.Query('Variable Sum')
        var_mean = var_sum / num_nodes
        output[name].append([tval, var_mean])

visit.CloseDataBase(datafile)

for name in data_names:
    fname = os.path.join(outputdir,name + '.dat')
    header = name + ' data\ntime\tvalue\n'
    np.savetxt(fname, output[name], header=header)

