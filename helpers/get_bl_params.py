import sys
import os

def u_profile(datafile):
    import numpy as np
    import visit
    OpenDatabase(datafile)
    SetActiveWindow(1)
    DeleteActivePlots()
    last_state = TimeSliderGetNStates()-1
    TimeSliderSetState(last_state)
    AddPlot("Pseudocolor", "u", 1, 1)
    DrawPlots()
    AddOperator("DataBinning", 1)
    SetActivePlots(0)
    SetActivePlots(0)
    DataBinningAtts = DataBinningAttributes()
    DataBinningAtts.numDimensions = DataBinningAtts.Three  # One, Two, Three
    DataBinningAtts.dim1BinBasedOn = DataBinningAtts.X  # X, Y, Z, Variable
    DataBinningAtts.dim1Var = "default"
    DataBinningAtts.dim1SpecifyRange = 1
    DataBinningAtts.dim1MinRange = 5
    DataBinningAtts.dim1MaxRange = 20
    DataBinningAtts.dim1NumBins = 1
    DataBinningAtts.dim2BinBasedOn = DataBinningAtts.Y  # X, Y, Z, Variable
    DataBinningAtts.dim2Var = "default"
    DataBinningAtts.dim2SpecifyRange = 0
    DataBinningAtts.dim2MinRange = 0
    DataBinningAtts.dim2MaxRange = 1
    DataBinningAtts.dim2NumBins = 100
    DataBinningAtts.dim3BinBasedOn = DataBinningAtts.Z  # X, Y, Z, Variable
    DataBinningAtts.dim3Var = "default"
    DataBinningAtts.dim3SpecifyRange = 0
    DataBinningAtts.dim3MinRange = 0
    DataBinningAtts.dim3MaxRange = 1
    DataBinningAtts.dim3NumBins = 1
    DataBinningAtts.outOfBoundsBehavior = DataBinningAtts.Clamp  # Clamp, Discard
    DataBinningAtts.reductionOperator = DataBinningAtts.Average  # Average, Minimum, Maximum, StandardDeviation, Variance, Sum, Count, RMS, PDF
    DataBinningAtts.varForReduction = "u"
    DataBinningAtts.emptyVal = 0
    DataBinningAtts.outputType = DataBinningAtts.OutputOnBins  # OutputOnBins, OutputOnInputMesh
    DataBinningAtts.removeEmptyValFromCurve = 0
    SetOperatorOptions(DataBinningAtts, 0, 1)
    DrawPlots()
    Query("Lineout", end_point=(5, 10, 0), num_samples=100, start_point=(5, 0, 0), use_sampling=0)
    SetQueryOutputToValue()
    domain = np.linspace(0,10,11)
    coords = []
    SetActiveWindow(2)
    for ind in range(100):
        coord = PickByNode(domain=0, element=ind)['point']
        if coord[1] > 0.:
            print(coord)
            
            coords.append(list(coord))
    np.savetxt('u-profile.dat',np.array(coords).T)
    return np.array(coords)

def plot_profile():
    import numpy as np
    import matplotlib.pyplot as plt
    import scipy.optimize
    y,u = np.loadtxt('u-profile.dat')
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
    utau = sol.x

    plt.plot(y*utau/nu,u*utau/nu)
    plt.plot(hwm*utau/nu, Uwm*utau/nu,'*')
    plt.plot(d99*utau/nu, 0.99*uinf*utau/nu, '*')
    plt.title('d99 = {}'.format(d99))
    print('Re  :\t{}\nuinf:\t{}\nd99 :\t{}\nhwm :\t{}\nnu  :\t{}\ndelt:\t{}\nUwm :\t{}\nutau:\t{}'.format(Re,uinf,d99,hwm,nu,deltax,Uwm,utau))
    plt.show()


if __name__=="__main__":
    if len(sys.argv)==2:
        plot_profile()
    else:
        #datafile = '/usr/workspace/barker38/scratch/WRLES/tbl-wrles/pyranda.visit'
        datafile = '/usr/workspace/barker38/scratch/WRLES/myTBL_long2/pyranda.visit'
        os.system('\n')
        data = u_profile(datafile)
        sys.exit()
