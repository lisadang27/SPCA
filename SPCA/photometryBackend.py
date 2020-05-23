import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys
from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel

# SPCA libraries
from . import Photometry_Aperture as APhotometry
from . import Photometry_PSF as PSFPhotometry
from . import Photometry_Companion as CPhotometry
from . import Photometry_PLD as PLDPhotometry


# FIX - check if the folder is empty, and if it is, just overwrite it
def create_folder(fullname, auto=False, overwrite=False):
    """Create a folder unless it exists.

    Args:
        fullname (string): Full path to the folder to be created.
        auto (bool, optional): If the folder already exists, should the folder just be skipped (True) or should the user be asked whether they want to overwrite the folder or change the folder name (False, Default).
        overwrite (bool, optional): Whether you want to overwrite the folder if it already exists

    Returns:
        string: The final name used for the folder.

    """
    
    solved = 'no'
    while(solved == 'no'):
        if not os.path.exists(fullname):
            # Folder doesn't exist yet and can be safely written to
            os.makedirs(fullname)
            solved = 'yes'
        elif len(os.listdir(fullname))==0:
            # Folder exists but is empty and can be safely written to
            solved = 'yes'
        else:
            if overwrite:
                solved = 'yes'
            elif auto:
                fullname = None
                solved = 'yes'
            else:
                folder = fullname.split('/')[-1]
                print('Warning:', folder, 'already exists! Are you sure you want to overwrite this folder? (y/n)')
                answer = input()
                if (answer=='y'):
                    solved = 'yes'
                else:
                    print('What would you like the new folder name to be?')
                    folder = input()
                    fullname = '/'.join(fullname.split('/')[0:-1])+'/'+folder
    return fullname

def run_photometry(photometryMethod, basepath, planet, channel, subarray, AOR_snip, rerun_photometry=False,
                   addStack=False, bin_data=True, bin_size=64, 
                   ignoreFrames=None, maskStars=None,
                   stamp_size=3,
                   shape='Circular', edge='Exact', moveCentroid=False, radius=3):
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []
    
    stackPath = basepath+'Calibration/' #folder containing properly named correction stacks (will be automatically selected)
    
    if channel=='ch1':
        folder='3um'
    else:
        folder='4um'
    if photometryMethod=='Aperture':
        folder += edge+shape+"_".join(str(np.round(radius, 2)).split('.'))
        if moveCentroid:
            folder += '_movingCentroid'
    elif photometryMethod=='PLD':
        folder += 'PLD_'+str(int(stamp_size))+'x'+str(int(stamp_size))
    folder += '/'
    datapath   = basepath+planet+'/data/'+channel
    
    savepath = basepath+planet+'/analysis/'+channel+'/'
    if addStack:
        savepath += 'addedStack/'
    else:
        savepath += 'addedBlank/'
    if ignoreFrames != []:
        savepath += 'ignore/'
    else:
        savepath += 'noIgnore/'
    savepath   += folder
    plot_name = 'lightcurve_'+planet+'.pdf'

    print('Starting:', savepath)
    
    # create save folder
    savepath = create_folder(savepath, True, rerun_photometry)
    
    if savepath == None:
        # This photometry has already been run
        return
    
    # prepare filenames for saved data
    save_full = channel+'_datacube_full_AORs'+AOR_snip[1:]+'.dat'
    save_bin = channel+'_datacube_binned_AORs'+AOR_snip[1:]+'.dat'

    # Call requested function
    if   (photometryMethod == 'Aperture'):
        APhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
                                   save=True, save_full=save_full,
                                   bin_data=bin_data, bin_size=bin_size, save_bin=save_bin,
                                   planet=planet,
                                   r=radius, shape=shape, edge=edge, plot=False, plot_name=plot_name,
                                   addStack=addStack, stackPath=stackPath, ignoreFrames=ignoreFrames,
                                   moveCentroid=moveCentroid, maskStars=maskStars)
    elif (photometryMethod == 'PSFfit'):
        PSFPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray)
    elif (photometryMethod == 'Companion'):
        CPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray, r = radius)
    elif (photometryMethod == 'PLD'):
        PLDPhotometry.get_pixel_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
                                           save=True, save_full=save_full,
                                           bin_data=bin_data, bin_size=bin_size, save_bin=save_bin,
                                           plot=False, plot_name=plot_name, planet=planet,
                                           stamp_size=stamp_size, addStack=addStack, stackPath=stackPath,
                                           ignoreFrames=ignoreFrames, maskStars=maskStars)
    else:
        print('Sorry,', photometryMethod, 'is not supported by this pipeline!')
        
    return

'''Get list of directories'''
def get_fnames(directory, tag='um'):
    '''
    Find paths to all the fits files.

    Parameters
    ----------

    directory : string object
        Path to the directory containing all the Spitzer data.

    AOR_snip  : string object
        Common first characters of data directory eg. 'r579'

    ch        : string objects
        Channel used for the observation eg. 'ch1' for channel 1    

    Returns
    -------

    fname     : list
        List of paths to all bcd.fits files.

    len(fnames): int
        Number of fits file found.
    '''
    lst      = os.listdir(directory)
    Run_list = [k for k in lst if tag in k]
    return sorted(Run_list)

def get_full_data(foldername, channel, AOR_snip):
    path = foldername + '/'+channel+'_datacube_full_AORs'+AOR_snip+'.dat'
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # Flux from circular aperture (MJy/str)
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # time in days?
    xdata    = np.loadtxt(path, usecols=[3], skiprows=1)     # x-centroid (15 = center of 15th pixel)
    ydata    = np.loadtxt(path, usecols=[4], skiprows=1)     # y-centroid (15 = center of 15th pixel)
    psfwx    = np.loadtxt(path, usecols=[5], skiprows=1)     # psf width in pixel size (FWHM of 2D Gaussian)
    psfwy    = np.loadtxt(path, usecols=[6], skiprows=1)     # psf width in pixel size (FWHM of 2D Gaussian)    
    return flux, time, xdata, ydata, psfwx, psfwy

def get_data(folderdata, channel, AOR_snip):
    path = folderdata + '/'+channel+'_datacube_binned_AORs'+AOR_snip+'.dat'
    path2= folderdata + '/popt.dat'
    
    #Loading Data (Aperture)
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # Flux from circular aperture (MJy/str)
    flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # Flux uncertainty from circular aperture (MJy/str)
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # Time in days
    xdata    = np.loadtxt(path, usecols=[4], skiprows=1)     # x-centroid (15 = center of 15th pixel)
    ydata    = np.loadtxt(path, usecols=[6], skiprows=1)     # y-centroid (15 = center of 15th pixel)
    psfwx    = np.loadtxt(path, usecols=[8], skiprows=1)     # psf width in pixel size (FWHM of 2D Gaussian)
    psfwy    = np.loadtxt(path, usecols=[10], skiprows=1)    # psf width in pixel size (FWHM of 2D Gaussian)
    
    return flux, flux_err, time, xdata, ydata, psfwx, psfwy 

def highpassflist(signal, highpassWidth):
    g = Box1DKernel(highpassWidth)
    smooth=convolve(np.asarray(signal), g,boundary='extend')
    return smooth

def get_RMS(Run_list, channel, AOR_snip, highpassWidth, trim=False, trimStart=0, trimEnd=0):
    RMS_list = np.empty(len(Run_list))
    for i in range(len(Run_list)):
        foldername = Run_list[i]
        flux, flux_err, time, xdata, ydata, psfwx, psfwy = get_data(foldername, channel, AOR_snip)
        if trim:
            flux = np.delete(flux, np.where(np.logical_and(time > trimStart, time < trimEnd))[0])
            time = np.delete(time, np.where(np.logical_and(time > trimStart, time < trimEnd))[0])
            
        order = np.argsort(time)
        flux = flux[order]
        time = time[order]
            
        smooth = highpassflist(flux, highpassWidth)
        smoothed = (flux - smooth)+np.nanmean(flux)
        RMS_list[i] = np.sqrt(np.nanmean((flux-smooth)**2.))/np.nanmean(smoothed)
        path = foldername + '/RMS_Scatter.pdf'
        fig, axes = plt.subplots(ncols = 1, nrows = 2, sharex = True, figsize = (10,6))
        fig.suptitle('RMS = '+ str(RMS_list[i]))
        axes[0].plot(time, flux, 'k.', alpha = 0.15, label='Measured Flux')
        axes[0].plot(time, smooth, '+', label = 'Filtered')
        axes[0].set_ylabel('Relative Flux')
        axes[1].plot(time, (smoothed/np.nanmean(smoothed)-1)*1e2, 'k.', alpha =0.1)
        axes[1].set_xlim(np.nanmin(time), np.nanmax(time))
        axes[1].axhline(y=0, color='b', linewidth = 1)
        axes[1].set_ylabel('Residual (%)')
        axes[1].set_xlabel('Time since IRAC turn on(days)')
        fig.subplots_adjust(hspace=0)
        fig.savefig(path)
        plt.close()
    return RMS_list



def comparePhotometry(basepath, planet, channel, AOR_snip, ignoreFrames, addStack, highpassWidth = 5, trim=False, trimStart=None, trimEnd=False):
    
    datapath = basepath+planet+'/analysis/'+channel+'/'
    figpath  = basepath+planet+'/analysis/photometryComparison/'+channel+'/'
    if addStack:
        datapath += 'addedStack/'
        figpath += 'addedStack/'
    else:
        datapath += 'addedBlank/'
        figpath += 'addedBlank/'
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    if ignoreFrames != []:
        datapath += 'ignore/'
        figpath += 'ignore/'
    else:
        datapath += 'noIgnore/'
        figpath += 'noIgnore/'
    
    if not os.path.exists(figpath):
        os.makedirs(figpath)
    
    Run_list = get_fnames(datapath)
    # Remove PLD runs from comparing photometry
    Run_list = [Run for Run in Run_list if 'PLD' not in Run]
    Radius = np.array([float(Run_list[i].split('_')[0][-1] + '.' 
                             + Run_list[i].split('_')[1][:]) for i in range(len(Run_list))])
    Run_list = [datapath + st for st in Run_list]
    
    
    if trim:
        RMS = get_RMS(Run_list, channel, AOR_snip[1:], highpassWidth, trim, trimStart, trimEnd)
    else:
        RMS = get_RMS(Run_list, channel, AOR_snip[1:], highpassWidth)
        
    plt.figure(figsize = (10,4))

    exact_moving = np.array(['exact' in Run_list[i].lower() and 'moving' in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)
    soft_moving =  np.array(['soft' in Run_list[i].lower() and 'moving' in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)
    hard_moving =  np.array(['hard' in Run_list[i].lower() and 'moving' in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)

    exact = np.array(['exact' in Run_list[i].lower() and 'moving' not in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)
    soft =  np.array(['soft' in Run_list[i].lower() and 'moving' not in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)
    hard =  np.array(['hard' in Run_list[i].lower() and 'moving' not in Run_list[i].lower() for i in range(len(Run_list))], dtype=bool)

    if np.any(exact_moving):
        plt.plot(Radius[exact_moving],  RMS[exact_moving]*1e6, 'o-', label = 'Circle: Exact Edge, Moving')
    if np.any(soft_moving):
        plt.plot(Radius[soft_moving],  RMS[soft_moving]*1e6, 'o-', label = 'Circle: Soft Edge, Moving')
    if np.any(hard_moving):
        plt.plot(Radius[hard_moving],  RMS[hard_moving]*1e6, 'o-', label = 'Circle: Hard Edge, Moving')

    if np.any(exact):
        plt.plot(Radius[exact],  RMS[exact]*1e6, 'o-', label = 'Circle: Exact Edge')
    if np.any(soft):
        plt.plot(Radius[soft],  RMS[soft]*1e6, 'o-', label = 'Circle: Soft Edge')
    if np.any(hard):
        plt.plot(Radius[hard],  RMS[hard]*1e6, 'o-', label = 'Circle: Hard Edge')

    plt.xlabel('Aperture Radius')
    plt.ylabel('RMS Scatter (ppm)')
    plt.legend(loc='best')

    if channel=='ch2':
        fname = figpath + '4um'
    else:
        fname = figpath + '3um'
    fname += '_Photometry_Comparison.pdf'
    plt.savefig(fname)
    plt.show()
    
    
    if np.any(exact_moving):
        print('Exact Moving - Best RMS (ppm):', np.round(np.nanmin(RMS[exact_moving])*1e6, decimals=2))
        print('Exact Moving - Best Aperture Radius:', Radius[exact_moving][np.where(RMS[exact_moving]==np.nanmin(RMS[exact_moving]))[0][0]])
        print()
    if np.any(soft_moving):
        print('Soft Moving - Best RMS (ppm):', np.round(np.nanmin(RMS[soft_moving])*1e6, decimals=2))
        print('Soft Moving - Best Aperture Radius:', Radius[soft_moving][np.where(RMS[soft_moving]==np.nanmin(RMS[soft_moving]))[0][0]])
        print()
    if np.any(hard_moving):
        print('Hard Moving - Best RMS (ppm):', np.round(np.nanmin(RMS[hard_moving])*1e6, decimals=2))
        print('Hard Moving - Best Aperture Radius:', Radius[hard_moving][np.where(RMS[hard_moving]==np.nanmin(RMS[hard_moving]))[0][0]])
        print()
    if np.any(exact):
        print('Exact - Best RMS (ppm):', np.round(np.nanmin(RMS[exact])*1e6, decimals=2))
        print('Exact - Best Aperture Radius:', Radius[exact][np.where(RMS[exact]==np.nanmin(RMS[exact]))[0][0]])
        print()
    if np.any(soft):
        print('Soft - Best RMS (ppm):', np.round(np.nanmin(RMS[soft])*1e6, decimals=2))
        print('Soft - Best Aperture Radius:', Radius[soft][np.where(RMS[soft]==np.nanmin(RMS[soft]))[0][0]])
        print()
    if np.any(hard):
        print('Hard - Best RMS (ppm):', np.round(np.nanmin(RMS[hard])*1e6, decimals=2))
        print('Hard - Best Aperture Radius:', Radius[hard][np.where(RMS[hard]==np.nanmin(RMS[hard]))[0][0]])
        
    
    optionsSelected = np.array(['' for i in range(len(RMS))], dtype=str)
    optionsSelected[exact_moving] = 'Exact Moving'
    optionsSelected[soft_moving] = 'Soft Moving'
    optionsSelected[hard_moving] = 'Hard Moving'
    optionsSelected[exact] = 'Exact'
    optionsSelected[soft] = 'Soft'
    optionsSelected[hard] = 'Hard'
    
    
    print('Best photometry:', Run_list[np.argmin(RMS)])
    
    with open(figpath+'best_option.txt', 'w') as file:
        file.write(Run_list[np.argmin(RMS)])
    
    return np.nanmin(RMS), Run_list[np.argmin(RMS)]

