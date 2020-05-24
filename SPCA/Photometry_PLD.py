import numpy as np
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.stats import sigma_clip

import os, sys, csv, glob, warnings

from .Photometry_Common import get_fnames, get_stacks, get_time, sigma_clipping, bgsubtract, binning_data

def binning_data2D(data, size):
    """Median bin PLD stamps.

    Args:
        data (2D array): Array of PLD stamps to be binned.
        size (int): Size of bins.

    Returns:
        tuple: binned_data (2D array) Array of binned PLD stamps.
            binned_data_std (2D array) Array of standard deviation for each entry in binned_data.
    
    """
    
    data = np.ma.masked_invalid(data)
    h, w = data.shape
    reshaped_data   = data.reshape((int(h/size), size, w))
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.ma.std(reshaped_data, axis=1)
    return binned_data, binned_data_std

def get_pixel_values(image, P, cx = 15, cy = 15, nbx = 3, nby = 3):
    """Median bin PLD stamps.

    Args:
        image (2D array): Image to be cut into PLD stamps.
        P (ndarray): Previously made PLD stamps to append new stamp to.
        cx (int, optional): x-coordinate of the center of the PLD stamp. Default is 15.
        cy (int, optional): y-coordinate of the center of the PLD stamp. Default is 15.
        nbx (int, optional): Number of pixels to use along the x-axis for the PLD stamp.
        nby (int, optional): Number of pixels to use along the y-axis for the PLD stamp.

    Returns:
        ndarray: Updated array of binned PLD stamps including the new stamp.
    
    """
    
    image_data = np.ma.masked_invalid(image)
    h, w, l = image_data.shape
    P_tmp = np.empty(shape=(h, nbx*nby))
    deltax = int((nbx-1)/2)
    deltay = int((nbx-1)/2)
    for i in range(h):
        P_tmp[i,:]   = image_data[i, (cx-deltax):(cx+deltax+1), (cy-deltay):(cy+deltay+1)].flatten()
#         np.array([image_data[i, cx-deltax,cy-1], image_data[i, cx-deltax,cy], image_data[i, cx-deltax,cy+1],
#             image_data[i,cx,cy-1], image_data[i,cx,cy], image_data[i, cx,cy+1],
#             image_data[i,cx+1,cy-1], image_data[i,cx+1,cy], image_data[i,cx+1,cy+1]])
    P = np.append(P, P_tmp, axis = 0)
    return P

def get_pixel_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
    save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
    bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
    plot_name= 'Lightcurve.pdf', planet = 'CoRoT-2b', stamp_size = 3, addStack = False,
    stackPath = '', ignoreFrames = None, maskStars = None):
    
    """Given a directory, looks for data (bcd.fits files), opens them and performs PLD "photometry".

    Args:
        datapath (string): Directory where the spitzer data is stored.
        savepath (string): Directory the outputs will be saved.
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
        subarray (bool): True if observation were taken in subarray mode. False if observation were taken in full-array mode.
        save (bool, optional): True if you want to save the outputs. Default is True.
        save_full (string, optional): Filename of the full unbinned output data. Default is '/ch2_datacube_full_AORs579.dat'.
        bin_data (bool, optional): True you want to get binned data. Default is True.
        bin_size (int, optional): If bin_data is True, the size of the bins. Default is 64.
        save_bin (string, optional): Filename of the full binned output data. Default is '/ch2_datacube_binned_AORs579.dat'.
        plot (bool, optional): True if you want to plot the time resolved lightcurve. Default is True.
        plot_name (string, optional): If plot and save is True, the filename of the plot to be saved as. Default is True.
        planet (string, optional): The name of the planet. Default is CoRoT-2b.
        stamp_size (int, optional): The size of PLD stamp to use (returns stamps that are stamp_size x stamp_size).
            Only 3 and 5 are currently supported.
        addStack (bool, optional): Whether or not to add a background subtraction correction stack. Default is False.
        stackPath (string, optional): Path to the background subtraction correction stack.
        ignoreFrames (list, optional) A list of frames to be masked when performing aperature photometry (e.g. first
            frame to remove first-frame systematic).
        maskStars (list, optional): An array-like object where each element is an array-like object with the RA and DEC
            coordinates of a nearby star which should be masked out when computing background subtraction.

    Raises: 
        Error: If Photometry method is not supported/recognized by this pipeline.
    
    """
    
    #Fix: Throw actual errors in these cases
    if stamp_size!=3 and stamp_size!=5:
        print('Error: Only stamp sizes of 3 and 5 are currently allowed.')
        return
    if not subarray:
        print('Error: Full frame photometry is not yet supported.')
        return
    
    if savepath[-1]!='/':
        savepath += '/'
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []

    # Ignore warning
    warnings.filterwarnings('ignore')

    # get list of filenames and nb of files
    fnames, lens = get_fnames(datapath, AOR_snip, channel)
    if addStack:
        stacks = get_stacks(stackPath, datapath, AOR_snip, channel)

    tossed        = 0                                # Keep tracks of number of frame discarded 
    badframetable = []                               # list of filenames of the discarded frames
    time          = []                               # time array
    bg_flux       = []                               # background flux
    bg_err        = []                               # background flux error 
    P             = np.empty((0,int(stamp_size**2)))

    if subarray:
        for i in range(len(fnames)):
            # open fits file
            hdu_list = fits.open(fnames[i])
            image_data0 = hdu_list[0].data
            # get time
            time = get_time(hdu_list, time, ignoreFrames)
            #add background correcting stack if requested
            if addStack:
                while i > np.sum(lens[:j+1]):
                    j+=1 #if we've moved onto a new AOR, increment j
                stackHDU = fits.open(stacks[j])
                image_data0 += stackHDU[0].data
            #ignore any consistently bad frames
            if ignoreFrames != []:
                image_data0[ignoreFrames] = np.nan
            h, w, l = image_data0.shape
            # convert MJy/str to electron count
            convfact = hdu_list[0].header['GAIN']*hdu_list[0].header['EXPTIME']/hdu_list[0].header['FLUXCONV']
            image_data1 = convfact*image_data0
            # sigma clip
            fname = fnames[i]
            image_data2, tossed, badframetable = sigma_clipping(image_data1, i ,fname[fname.find('ch2/bcd/')+8:],
                                                                tossed=tossed)
            
            if maskStars != []:
                # Mask any other stars in the frame to avoid them influencing the background subtraction
                hdu_list[0].header['CTYPE3'] = 'Time-SIP' #Just need to add a type so astropy doesn't complain
                w = WCS(hdu_list[0].header, naxis=[1,2])
                mask = image_data2.mask
                for st in maskStars:
                    coord = SkyCoord(st[0], st[1])
                    x,y = np.rint(skycoord_to_pixel(coord, w)).astype(int)
                    x = x+np.arange(-1,2)
                    y = y+np.arange(-1,2)
                    x,y = np.meshgrid(x,y)
                    mask[x,y] = True
                image_data2 = np.ma.masked_array(image_data2, mask=mask)
            
            # bg subtract
            image_data3, bg_flux, bg_err = bgsubtract(image_data2, bg_flux, bg_err)
            # get pixel peak index
            P = get_pixel_values(image_data3, P, cx=15, cy=15, nbx=stamp_size, nby=stamp_size)
    else:
        # FIX: The full frame versions of the code will have to go here
        pass
        
    
    if bin_data:
        binned_P, binned_P_std = binning_data2D(P, bin_size)
        binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)
        binned_bg, binned_bg_std = binning_data(np.asarray(bg_flux), bin_size)
        binned_bg_err, binned_bg_err_std = binning_data(np.asarray(bg_err), bin_size)
        
        #sigma clip binned data to remove wildly unacceptable data
        binned_flux = binned_P.sum(axis=1)
        try:
            # Need different versions for different versions of astropy...
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, maxiters=2)
        except TypeError:
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, iters=2)
        if np.ma.is_masked(binned_flux_mask):
            binned_time[binned_flux_mask==binned_flux] = np.nan
            binned_time_std[binned_flux_mask==binned_flux] = np.nan
            binned_bg[binned_flux_mask==binned_flux] = np.nan
            binned_bg_std[binned_flux_mask==binned_flux] = np.nan
            binned_bg_err[binned_flux_mask==binned_flux] = np.nan
            binned_bg_err_std[binned_flux_mask==binned_flux] = np.nan
            binned_P_std[binned_flux_mask==binned_flux] = np.nan
            binned_P[binned_flux_mask==binned_flux] = np.nan

    if plot:
        if bin_data:
            plotx = binned_time
            ploty0 = binned_P
            ploty2 = binned_P_std
            nrows=3
        else:
            plotx = time
            ploty0 = P
            nrows = 2
        fig, axes = plt.subplots(nrows = nrows, ncols = 1, sharex = True, figsize=(nrows*5,10))
        fig.suptitle(planet, fontsize="x-large")
        for i in range(int(stamp_size**2)):
            axes[0].plot(binned_time, binned_P[:,i], '+', label = '$P_'+str(i+1)+'$')
        axes[0].set_ylabel("Pixel Flux (MJy/pixel)")
        axes[0].legend()
        axes[1].set_ylabel('Sum Flux (MJy/pixel)')
        axes[1].plot(binned_time, np.sum(binned_P, axis = 1), '+')
        if bin_data:
            for i in range(int(stamp_size**2)):
                axes[2].plot(binned_time, binned_P_std[:,i], '+', label = '$Pstd_'+str(i+1)+'$')
            axes[2].set_xlabel("Time since IRAC turn-on (days)")
        fig.subplots_adjust(hspace=0)

        if save:
            pathplot = savepath + plot_name
            fig.savefig(pathplot)
        plt.show()
        plt.close()
        

    if save:
        FULL_data = np.c_[P, time, bg_flux, bg_err]
        FULL_head = ''
        for i in range(int(stamp_size**2)):
            FULL_head += 'P'+str(i+1)+', '
        FULL_head += 'time, bg, bg_err'
        pathFULL  = savepath + save_full
        np.savetxt(pathFULL, FULL_data, header = FULL_head)
        if bin_data:
            BINN_data = np.c_[binned_P, binned_P_std, binned_time, binned_time_std,
                              binned_bg, binned_bg_std, binned_bg_err, binned_bg_err_std]
            BINN_head = ''
            for i in range(int(stamp_size**2)):
                BINN_head += 'P'+str(i+1)+', '
            for i in range(int(stamp_size**2)):
                BINN_head += 'P'+str(i+1)+'_std, '
            BINN_head = 'time, time_std, bg, bg_std, bg_err, bg_err_std'
            pathBINN  = savepath + save_bin
            np.savetxt(pathBINN, BINN_data, header = BINN_head)
    
    return

import unittest

if __name__ == '__main__':
    print('No unit tests currently implemented.')
