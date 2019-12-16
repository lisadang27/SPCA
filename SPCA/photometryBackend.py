import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys
from matplotlib import rc
import matplotlib.patches
import time
from matplotlib.ticker import MaxNLocator
import os, sys
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils import aperture_photometry
from photutils import CircularAperture
from numpy import std
import glob
import csv
import operator
import matplotlib.ticker as mtick
from photutils.datasets import make_4gaussians_image
#from photutils.morphology import (centroid_com,centroid_1dg,centroid_2dg)
from time import time
#from scipy.linalg.fblas import dgemm
from astropy.convolution import convolve, Box1DKernel
import collections
from astropy.convolution import Gaussian1DKernel
import astropy.constants as constants
import astropy.units as units


lib_path = os.path.abspath(os.path.join('../'))
sys.path.append(lib_path)

# SPCA libraries
from . import Photometry_Aperture_TaylorVersion as APhotometry
from . import Photometry_PSF as PSFPhotometry
from . import Photometry_Companion as CPhotometry
from . import Photometry_PLD as PLDPhotometry




def create_folder(fullname, auto=False):
    solved = 'no'
    while(solved == 'no'):
        if not os.path.exists(fullname):
            os.makedirs(fullname)
            solved = 'yes'
        else :
            if auto:
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

def run_photometry(basepath, addStack, planet, channel, subarray, AOR_snip, ignoreFrames, maskStars, photometryMethod, shape, edge, moveCentroid, radius):
    
    stackPath = basepath+'Calibration/' #folder containing properly named correction stacks (will be automatically selected)
    
    if channel=='ch1':
        folder='3um'
    else:
        folder='4um'
    if photometryMethod=='Aperture':
        folder += edge+shape+"_".join(str(np.round(radius, 2)).split('.'))
        if moveCentroid:
            folder += '_movingCentroid'
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
    savepath = create_folder(savepath, True)
    
    if savepath == None:
        # This photometry has already been run
        return
    
    # prepare filenames for saved data
    save_full = channel+'_datacube_full_AORs'+AOR_snip[1:]+'.dat'
    save_bin = channel+'_datacube_binned_AORs'+AOR_snip[1:]+'.dat'

    # Call requested function
    if   (photometryMethod == 'Aperture'):
        APhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
                                   save_full=save_full, save_bin=save_bin, planet=planet,
                                   r=radius, shape=shape, edge=edge, plot=False, plot_name=plot_name,
                                   addStack=addStack, stackPath=stackPath, ignoreFrames=ignoreFrames,
                                   moveCentroid=moveCentroid)
    elif (photometryMethod == 'PSFfit'):
        PSFPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray)
    elif (photometryMethod == 'Companion'):
        CPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray, r = radius)
    elif (photometryMethod == 'PLD'):
        PLDPhotometry.get_pixel_lightcurve(datapath, savepath, AOR_snip, channel, subarray)
    elif (photometryMethod == 'Routine'):
        Routine.get_lightcurve(datapath, savepath, AOR_snip, channel, subarray, r = radius*2+0.5)
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
    Run_list = [k for k in lst if tag==k[:len(tag)]]
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
        smoothed = (flux - smooth)+np.mean(flux)
        RMS_list[i] = np.sqrt(np.mean((flux-smooth)**2.))/np.mean(smoothed)
        path = foldername + '/RMS_Scatter.pdf'
        fig, axes = plt.subplots(ncols = 1, nrows = 2, sharex = True, figsize = (10,6))
        fig.suptitle('RMS = '+ str(RMS_list[i]))
        axes[0].plot(time, flux, 'k.', alpha = 0.15, label='Measured Flux')
        axes[0].plot(time, smooth, '+', label = 'Filtered')
        axes[0].set_ylabel('Relative Flux')
        axes[1].plot(time, (smoothed/np.mean(smoothed)-1)*1e2, 'k.', alpha =0.1)
        axes[1].set_xlim(np.min(time), np.max(time))
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
    
    Run_list = get_fnames(datapath, tag=AOR_snip)
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

    #plt.axhline(y=1.39485316958, color='orange', linewidth = 1, label = 'PSF Fitting')
    #plt.axhline(y=RMS45, color='orange', linewidth = 1)
    plt.xlabel('Aperture Radius')
    plt.ylabel('RMS Scatter (ppm)')
    #plt.ylim(ymin=0)
    plt.legend(loc='best')

#     if highpassWidth == 5:
#         if planet=='WASP-12b' and channel=='ch2':
#             plt.ylim(1100,1300)
#         elif planet=='WASP-12b' and channel=='ch1':
#             plt.ylim(1150, 1400)
#         elif planet=='WASP-12b_old' and channel=='ch1':
#             plt.ylim(1350, 1500)
#         elif planet=='WASP-12b_old' and channel=='ch2':
#             plt.ylim(1150, 1250)
#     else:
#         if planet=='WASP-12b' and channel=='ch2':
#             plt.ylim(1500,1700)
#         elif planet=='WASP-12b_old' and channel=='ch2':
#             "break"
#         elif planet=='WASP-12b' and channel=='ch1':
#             plt.ylim(2400, 2900)
#         elif planet=='WASP-12b_old' and channel=='ch1':
#             plt.ylim(2750, 3300)

    if channel=='ch2':
        fname = figpath + '4um'
    else:
        fname = figpath + '3um'
    fname += '_Photometry_Comparison.pdf'
    plt.savefig(fname)
    plt.show()
    
    
    if np.any(exact_moving):
        print('Exact Moving - Best RMS (ppm):', np.round(np.min(RMS[exact_moving])*1e6, decimals=2))
        print('Exact Moving - Best Aperture Radius:', Radius[exact_moving][np.where(RMS[exact_moving]==np.min(RMS[exact_moving]))[0][0]])
        print()
    if np.any(soft_moving):
        print('Soft Moving - Best RMS (ppm):', np.round(np.min(RMS[soft_moving])*1e6, decimals=2))
        print('Soft Moving - Best Aperture Radius:', Radius[soft_moving][np.where(RMS[soft_moving]==np.min(RMS[soft_moving]))[0][0]])
        print()
    if np.any(hard_moving):
        print('Hard Moving - Best RMS (ppm):', np.round(np.min(RMS[hard_moving])*1e6, decimals=2))
        print('Hard Moving - Best Aperture Radius:', Radius[hard_moving][np.where(RMS[hard_moving]==np.min(RMS[hard_moving]))[0][0]])
        print()
    if np.any(exact):
        print('Exact - Best RMS (ppm):', np.round(np.min(RMS[exact])*1e6, decimals=2))
        print('Exact - Best Aperture Radius:', Radius[exact][np.where(RMS[exact]==np.min(RMS[exact]))[0][0]])
        print()
    if np.any(soft):
        print('Soft - Best RMS (ppm):', np.round(np.min(RMS[soft])*1e6, decimals=2))
        print('Soft - Best Aperture Radius:', Radius[soft][np.where(RMS[soft]==np.min(RMS[soft]))[0][0]])
        print()
    if np.any(hard):
        print('Hard - Best RMS (ppm):', np.round(np.min(RMS[hard])*1e6, decimals=2))
        print('Hard - Best Aperture Radius:', Radius[hard][np.where(RMS[hard]==np.min(RMS[hard]))[0][0]])
        
    
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
    
    return np.min(RMS), Run_list[np.argmin(RMS)]

