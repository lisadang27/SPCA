import numpy as np
import matplotlib.pyplot as plt

from astropy.stats import sigma_clip

from .Photometry_Common import bin_array, create_folder, prepare_images

import warnings
warnings.filterwarnings('ignore')

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

def get_lightcurve(basepath, AOR_snip, channel, planet, stamp_sizes=[3,5], save=True,
                   bin_data=True, bin_size=64, showPlots=False, savePlots=True,
                   addStack = False, ignoreFrames = None, maskStars = None, ncpu=4):
    
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
    for stamp_size in stamp_sizes:
        if stamp_size!=3 and stamp_size!=5:
            print(f'Error: Stamp size {stamp_size} not permitted')
            print('Only stamp sizes of 3 and 5 are currently allowed.')
            return
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []
    
    if basepath[-1]!='/':
        basepath += '/'
    
    stackPath = basepath+'Calibration/' #folder containing properly named correction stacks (will be automatically selected)
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
    
    # prepare filenames for saved data
    save_full = channel+'_datacube_full_AORs'+AOR_snip[1:]+'.dat'
    save_bin = channel+'_datacube_binned_AORs'+AOR_snip[1:]+'.dat'
    
    # Ignore warning
    # warnings.filterwarnings('ignore')

    # Prepare all of the images
    image_stack, bg, bg_err, time = prepare_images(datapath, savepath, AOR_snip, ignoreFrames,
                                                   False, 1, False, False,
                                                   addStack, stackPath, maskStars, ncpu)
    
    savepath_raw = savepath
    for stamp_size in stamp_sizes:
        # get pixel peak index
        print(f'\tGetting {stamp_size}x{stamp_size} stamps... ', end='', flush=True)
        P = get_pixel_values(image_stack, cx=15, cy=15, nbx=stamp_size, nby=stamp_size)

        if bin_data:
            print('Binning... ', end='', flush=True)
            binned_P, binned_P_std = bin_array2D(P, bin_size)
            binned_time, binned_time_std = bin_array(time, bin_size)
            binned_bg, binned_bg_std = bin_array(bg, bin_size)
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

        if save or savePlots:
            print('Saving... ', end='', flush=True)
            if channel=='ch1':
                folder='3um'
            else:
                folder='4um'
            folder += f'PLD_{stamp_size}x{stamp_size}/'

            # create save folder
            savepath = create_folder(savepath_raw+folder, True, True)
                
        if savePlots or showPlots:
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
            axes[0].set_ylabel("Pixel Flux (electrons)")
            axes[0].legend()
            axes[1].set_ylabel('Sum Flux (electrons)')
            axes[1].plot(binned_time, np.sum(binned_P, axis = 1), '+')
            if bin_data:
                for i in range(int(stamp_size**2)):
                    axes[2].plot(binned_time, binned_P_std[:,i], '+', label = '$Pstd_'+str(i+1)+'$')
                axes[2].set_xlabel("Time (BMJD)")
            fig.subplots_adjust(hspace=0)

            if savePlots:
                pathplot = savepath + 'Lightcurve.pdf'
                fig.savefig(pathplot)
            if showPlots:
                plt.show()
            plt.close()


        if save:
            FULL_data = np.c_[P, time, bg, bg_err]
            FULL_head = ''.join([f'P{i+1}, ' for i in range(int(stamp_size**2))])
            FULL_head += 'time, bg, bg_err'
            pathFULL  = savepath + save_full
            np.savetxt(pathFULL, FULL_data, header = FULL_head)
            if bin_data:
                BINN_data = np.c_[binned_P, binned_P_std, binned_time, binned_time_std,
                                  binned_bg, binned_bg_std, binned_bg_err, binned_bg_err_std]
                BINN_head = ''.join([f'P{i+1}, ' for i in range(int(stamp_size**2))])
                BINN_head += ''.join([f'P{i+1}_std, ' for i in range(int(stamp_size**2))])
                BINN_head = 'time, time_std, bg, bg_std, bg_err, bg_err_std'
                pathBINN  = savepath + save_bin
                np.savetxt(pathBINN, BINN_data, header = BINN_head)
    
    image_stack = None
    
    print('Done.', flush=True)
    
    return

import unittest

if __name__ == '__main__':
    print('No unit tests currently implemented.')
