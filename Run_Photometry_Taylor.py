import numpy as np
import matplotlib.pyplot as plt
import os, sys
import Photometry_Aperture_TaylorVersion as APhotometry
import Photometry_PSF as PSFPhotometry
import Photometry_Companion as CPhotometry
import Photometry_PLD as PLDPhotometry

def create_folder(fullname):
    solved = 'no'
    while(solved == 'no'):
        if not os.path.exists(fullname):
            os.makedirs(fullname)
            solved = 'yes'
        else :
            folder = fullname.split('/')[-1]
            print('Error:', folder, 'already exists! Are you sure you want to overwrite this folder? (y/n)')
            answer = input()
            if (answer=='y'):
                solved = 'yes'
            else:
                print('What would you like the new folder name to be?')
                folder = input()
                fullname = '/'.join(fullname.split('/')[0:-1])+'/'+folder
    return fullname

if (len(sys.argv) < 2):
    rootpath = input("Root Path (including trailing slash): ")
    addStack = bool(input("Would you like to add a background correcting stack? (True/False): "))
    if addStack:
        stackPath = input("Enter the path for the background correcting stack: ")
    planet = input("Planet's Name: ")
    channel = input("Channel (ch1/ch2): ")
    subarray = bool(input("Subarray (True/False): "))
    AOR_snip = input("AOR Snip: ")
    maskStars = input("Optional comma separated coordinates of any stars to mask from background subtraction\n"+
                      "(e.g.: 01h02m03.4s, +01d02m03.4s, ...): ")
    if maskStars is not None:
        maskStars = list(filter(None, maskStars.split(',')))
        maskStars = [coord.strip() for coord in maskStars]
    photometry = input("Photometry Method: ")
    if photometry == "Aperture":
        radius = float(input("Aperture Radius: "))
        shape = input("Aperture Shape: ")
        edge = input("Aperture Edge: ")
        moveCentroid = input("Center Aperture on FWM Centroid? (True/False): ")
else:
    #test if given settings file exists
    fname = sys.argv[1]
    valid = os.path.isfile(fname)
    while not valid:
        print("Error: Settings file \""+fname+"\" does not exist!")
        fname = input("Please enter a new settings file name: ")
        valid = os.path.isfile(fname)
    #read in settings
    with open(fname) as file:
        lines = file.readlines()
        lines = [line.strip() for line in lines] #remove endlines
        headers = np.array([line.split(':')[0].lower().strip() for line in lines]) #get headers
        settings = [line.split(':')[-1].strip() for line in lines] #get inputs
        rootpath = settings[np.where(headers=='root path')[0][0]]
        addStack = bool(settings[np.where(headers=='add stack')[0][0]])
        stackPath = settings[np.where(headers=='stack path')[0][0]]
        ignoreFrames = settings[np.where(headers=='ignore frames')[0][0]]
        if ignoreFrames is not None and ignoreFrames != '':
            ignoreFrames = np.array(ignoreFrames.split(","), dtype=int)
        else:
            ignoreFrames = []
        planet = settings[np.where(headers=='planet')[0][0]]
        channel = settings[np.where(headers=='channel')[0][0]]
        subarray = bool(settings[np.where(headers=='subarray')[0][0]])
        AOR_snip = settings[np.where(headers=='aor snip')[0][0]]
        maskStars = settings[np.where(headers=='mask stars')[0][0]]
        if maskStars is not None:
            maskStars = list(filter(None, maskStars.split(',')))
            maskStars = [coord.strip() for coord in maskStars]
        photometry = settings[np.where(headers=='photometry method')[0][0]]
        if photometry == "Aperture":
            radius = float(settings[np.where(headers=='radius')[0][0]])
            shape = settings[np.where(headers=='shape')[0][0]]
            edge = settings[np.where(headers=='edge')[0][0]]
            moveCentroid = bool(settings[np.where(headers=='move centroid')[0][0]])
            if moveCentroid is None:
                moveCentroid = False

if channel=='ch1':
    folder='3um'
else:
    folder='4um'
if photometry=='Aperture':
    folder += edge+shape+"_".join(str(radius).split('.'))
    if moveCentroid:
        folder += '_movingCentroid'
datapath   = rootpath+planet+'/data/'+channel
savepath   = rootpath+planet+'/analysis/'+channel+'/'+folder
plot_name = 'lightcurve_'+planet+'.pdf'


# create save folder
savepath = create_folder(savepath)
# prepare filenames for saved data
save_full = channel+'_datacube_full_AORs'+AOR_snip[1:]+'.dat'
save_bin = channel+'_datacube_binned_AORs'+AOR_snip[1:]+'.dat'

# Call requested function
if   (photometry == 'Aperture'):
    APhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
                               save_full=save_full, save_bin=save_bin, planet=planet,
                               r=radius, shape=shape, edge=edge, plot_name=plot_name,
                               addStack=addStack, stackPath=stackPath, ignoreFrames=ignoreFrames,
                               moveCentroid=moveCentroid)
elif (photometry == 'PSFfit'):
    PSFPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray)
elif (photometry == 'Companion'):
    CPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray, r = radius)
elif (photometry == 'PLD'):
    PLDPhotometry.get_pixel_lightcurve(datapath, savepath, AOR_snip, channel, subarray)
elif (photometry == 'Routine'):
    Routine.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray, r = radius*2+0.5)
else:
    print('Sorry,', photometry, 'is not supported by this pipeline!')
