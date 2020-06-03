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

from .Photometry_Common import get_fnames, get_stacks, get_time, sigma_clipping, bgsubtract, bin_array, create_folder

def bin_array2D(data, size):
    """Median bin PLD stamps.

    Args:
        data (2D array): Array of PLD stamps to be binned.
        size (int): Size of bins.

    Returns:
        tuple: binned_data (2D array) Array of binned PLD stamps.
            binned_data_std (2D array) Array of standard deviation for each entry in binned_data.
    
    """
    
    h, w = data.shape
    
    # Iterate pixel-by-pixel to do the binning since there is sadly no axis argument in the binning function
    binned_data = []
    binned_data_std = []
    for i in range(w):
        result = bin_array(data[:,i], size)
        binned_data.append(result[1])
        binned_data_std.append(result[1])
        
    binned_data = np.array(binned_data).T
    binned_data_std = np.array(binned_data_std).T
    
    return binned_data, binned_data_std

def get_pixel_values(image, cx = 15, cy = 15, nbx = 3, nby = 3):
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
    deltax = int((nbx-1)/2)
    deltay = int((nbx-1)/2)
    P = image_data[:, (cx-deltax):(cx+deltax+1), (cy-deltay):(cy+deltay+1)].reshape(h, -1)
    return P

def get_pixel_lightcurve(datapath, savepath, AOR_snip, channel,
    save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
    bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
    plot_name= 'Lightcurve.pdf', planet = 'CoRoT-2b', stamp_size = 3, addStack = False,
    stackPath = '', ignoreFrames = None, maskStars = None, rerun_photometry=False):
    
    """Given a directory, looks for data (bcd.fits files), opens them and performs PLD "photometry".

    Args:
        datapath (string): Directory where the spitzer data is stored.
        savepath (string): Directory the outputs will be saved.
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
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
        rerun_photometry (bool, optional): Whether to overwrite old photometry if it exists. Default is False.

    Raises: 
        Error: If Photometry method is not supported/recognized by this pipeline.
    
    """
    
    #Fix: Throw actual errors in these cases
    if stamp_size!=3 and stamp_size!=5:
        print('Error: Only stamp sizes of 3 and 5 are currently allowed.')
        return
    
    if save and savepath[-1]!='/':
        savepath += '/'
    
    if save:
        if channel=='ch1':
            folder='3um'
        else:
            folder='4um'
        folder += f'PLD_{stamp_size}x{stamp_size}/'

        # create save folder
        savepath = create_folder(savepath+folder, True, rerun_photometry)
        if savepath == None:
            # This photometry has already been run and shouldn't be rerun
            return
        print('Starting:', savepath)
    
    while datapath[-1]=='/':
        datapath=datapath[:-1]
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []
    
    # Ignore warning
    warnings.filterwarnings('ignore')

    # get list of filenames and nb of files
    fnames, lens = get_fnames(datapath, AOR_snip)
    if addStack:
        stacks = get_stacks(stackPath, datapath, AOR_snip)

    time = []
    image_stack = np.zeros((0,32,32))
    
    # data reduction & aperture photometry part
    j=0 #counter to keep track of which correction stack we're using
    for i in range(len(fnames)):
        # open fits file
        with fits.open(fnames[i]) as hdu_list:
            if len(hdu_list[0].data.shape)==2:
                # Reshape fullframe data so that it can be used with our routines
                hdu_list[0].data = hdu_list[0].data[np.newaxis,217:249,9:41]

            image_data = hdu_list[0].data
            header = hdu_list[0].header
            # get time
            time = np.append(time, get_time(hdu_list, ignoreFrames))
            #add background correcting stack if requested
            if addStack:
                while i > np.sum(lens[:j+1]):
                    j+=1 #if we've moved onto a new AOR, increment j
                stackHDU = fits.open(stacks[j])
                image_data += stackHDU[0].data
            
            # convert MJy/str to electron count
            convfact = (hdu_list[0].header['GAIN']*hdu_list[0].header['EXPTIME']
                        /hdu_list[0].header['FLUXCONV'])
            image_stack = np.append(image_stack, convfact*image_data, axis=0)
    
    time = np.array(time)
    
    #ignore any consistently bad frames in datacubes
    for i in ignoreFrames:
        l = image_stack.shape[0]
        ignore_inds = (i + 64*np.arange(int(l/64)+int(l%64>i))).astype(int)
        image_stack[l] = np.nan
    
    # sigma clip along full time axis
    image_stack = sigma_clipping(image_stack)

    # Mask any other stars in the frame to avoid them influencing the background subtraction
    if maskStars != []:
        header['CTYPE3'] = 'Time-SIP' #Just need to add a type so astropy doesn't complain
        w = WCS(header, naxis=[1,2])
        mask = np.ma.getmaskarray(image_stack)
        for st in maskStars:
            coord = SkyCoord(st[0], st[1])
            x,y = np.rint(skycoord_to_pixel(coord, w)).astype(int)
            x = x+np.arange(-1,2)
            y = y+np.arange(-1,2)
            x,y = np.meshgrid(x,y)
            mask[:,x,y] = True
        image_stack = np.ma.masked_array(image_stack, mask=mask)

    # bg subtract
    image_stack, bg_flux, bg_err = bgsubtract(image_stack)
    
    # get pixel peak index
    P = get_pixel_values(image_stack, cx=15, cy=15, nbx=stamp_size, nby=stamp_size)
    
    if bin_data:
        binned_P, binned_P_std = bin_array2D(P, bin_size)
        binned_time, binned_time_std = bin_array(time, bin_size)
        binned_bg, binned_bg_std = bin_array(bg_flux, bin_size)
        binned_bg_err, binned_bg_err_std = bin_array(bg_err, bin_size)
        
        #sigma clip binned data to remove wildly unacceptable data
        binned_flux = binned_P.sum(axis=1)
        try:
            # Need different versions for different versions of astropy...
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, maxiters=2)
        except TypeError:
            binned_flux_mask = sigma_clip(binned_flux, sigma=10, iters=2)
        if np.ma.is_masked(binned_flux_mask):
            binned_time[binned_flux_mask!=binned_flux] = np.nan
            binned_time_std[binned_flux_mask!=binned_flux] = np.nan
            binned_bg[binned_flux_mask!=binned_flux] = np.nan
            binned_bg_std[binned_flux_mask!=binned_flux] = np.nan
            binned_bg_err[binned_flux_mask!=binned_flux] = np.nan
            binned_bg_err_std[binned_flux_mask!=binned_flux] = np.nan
            binned_P_std[binned_flux_mask!=binned_flux] = np.nan
            binned_P[binned_flux_mask!=binned_flux] = np.nan

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
