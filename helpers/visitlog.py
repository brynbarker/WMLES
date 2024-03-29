# Visit 3.3.3 log file
ScriptVersion = "3.3.3"
if ScriptVersion != Version():
    print "This script is for VisIt %s. It may not work with version %s" % (ScriptVersion, Version())
ShowAllWindows()
OpenDatabase("/usr/workspace/barker38/scratch/WRLES/myTBL_long2/pyranda.visit", 0)
# The UpdateDBPluginInfo RPC is not supported in the VisIt module so it will not be logged.
SetActiveWindow(1)
DeleteActivePlots()
AddPlot("Pseudocolor", "u", 1, 1)
DrawPlots()
AddOperator("DataBinning", 1)
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
SetOperatorOptions(DataBinningAtts, -1, 1)
DrawPlots()
Query("Lineout", end_point=(5, 10, 0), num_samples=100, start_point=(5, 0, 0), use_sampling=0)
SetActiveWindow(2)
PickByNode(domain=0, element=0)
