import sys
import os

def animate(loc,param):
    #datafile = '/usr/workspace/barker38/{}/pyranda.visit'.format(loc)
    datafile = '/p/lustre1/barker38/{}/pyranda.visit'.format(loc)
    fname = loc.split('/')[-1]
    import visit
    from visit_utils import encoding

    aatts = AnnotationAttributes()
    aatts.axes3D.visible = 0
    aatts.axes3D.triadFlag = 0
    aatts.axes3D.bboxFlag = 0
    aatts.userInfoFlag = 0
    aatts.databaseInfoFlag = 0
    aatts.legendInfoFlag = 0
    SetAnnotationAttributes(aatts)

    # set basis save options
    swatts = SaveWindowAttributes()
    # The 'family' option controls if visit automatically adds a frame number to 
    # the rendered files. For this example we will explicitly manage the output name.
    swatts.family = 0
    # select PNG as the output file format
    swatts.format = swatts.PNG 
    # set the width of the output image
    swatts.width = 1024 
    # set the height of the output image
    swatts.height = 512

    OpenDatabase(datafile)
    SetActiveWindow(1)
    DeleteActivePlots()
    AddPlot("Pseudocolor", param, 1, 1)
    DrawPlots()
    nts = TimeSliderGetNStates()
    tmp_folder = 'tmp-for-animation'
    os.makedirs(tmp_folder)
    for ts in range(nts):
        TimeSliderSetState(ts)
        swatts.fileName = "{}/{}-%04d.png".format(tmp_folder,fname) % ts
        SetSaveWindowAttributes(swatts)
        # render the image to a PNG file
        SaveWindow()

    input_pattern = "{}/{}-%04d.png".format(tmp_folder,fname)
    output_movie = "/usr/workspace/barker38/scratch/animations/{}.mp4".format(fname)
    encoding.encode(input_pattern,output_movie,fdup=4)
    for f in os.listdir(tmp_folder):
        fpath = os.path.join(tmp_folder,f)
        os.remove(fpath)
    os.rmdir(tmp_folder)


        
    
if __name__=="__main__":
    last = str(sys.argv[-1])
    if len(last) > 5:
        loc = last
        param = 'u'
    else:
        param = last
        loc = str(sys.argv[-1])
    animate(loc,param)
    sys.exit()
