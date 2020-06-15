import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

from astropy.stats import sigma_clip

from photutils import aperture_photometry
from photutils import CircularAperture, EllipticalAperture, RectangularAperture
from photutils.utils import calc_total_error

from multiprocessing import Pool
from functools import partial

from collections import Iterable

from .Photometry_Common import bin_array, create_folder, prepare_images, clip_data
from .make_plots import plot_photometry

import os, warnings
warnings.filterwarnings('ignore')

# Make into a global variable so that A_Photometry can be run with multiprocessing without
# needing to pickle image_stack - this is critical for datasets with many GB of data!
image_stack = np.zeros((0,32,32))

# We need to resort to some hackery to make a tqdm progress bar work with multiprocessing
results = []
func = lambda arg: arg
pbar = None
def wrapMyFunc(arg):
    global func
    return arg, func(arg)

def update(outputs):
    # note: input comes from async `wrapMyFunc`
    global results
    results[outputs[0]] = outputs[1]  # put answer into correct index of result list
    global pbar
    pbar.update()
    return

# Get the center by fitting a 2D gaussian
def gaussian(height, center_x, center_y, width_x, width_y, x, y):
    #Using code based on that from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    """Returns a gaussian function with the given parameters"""
    return height/(2*np.pi*width_x*width_y)*np.exp(-(((center_x-x)/width_x)**2+((center_y-y)/width_y)**2)/2)

def moments(starbox):
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution by calculating its
    moments """
    w, l = starbox.shape
    
    # get centroid
    Y, X    = np.mgrid[:w,:l]
    cx      = np.nansum(X*starbox)/np.nansum(starbox)
    cy      = np.nansum(Y*starbox)/np.nansum(starbox)
    
    X2, Y2  = (X - cx)**2, (Y - cy)**2
    with np.errstate(invalid='ignore'):
        widx    = np.sqrt(np.nansum(X2*starbox)/np.nansum(starbox))
        widy    = np.sqrt(np.nansum(Y2*starbox)/np.nansum(starbox))
        
    return np.max(starbox), cx, cy, widx, widy

def fitgaussian(bounds, scale, i):
    #Using code based on that from https://scipy-cookbook.readthedocs.io/items/FittingData.html
    """Returns (height, x, y, width_x, width_y)
    the gaussian parameters of a 2D distribution found by a fit"""
    
    global image_stack
    lbx, ubx, lby, uby = bounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    
    params = moments(image_stack[i, lbx:ubx, lby:uby])
    errorfunction = lambda p: np.ravel(gaussian(*p, *np.indices(image_stack[i, lbx:ubx, lby:uby].shape)[::-1,:,:])
                                       - image_stack[i, lbx:ubx, lby:uby])
    p, success = optimize.leastsq(errorfunction, params)
    return p

def fit_2DGaussian(image_stack, scale=1, bounds=(13, 18, 13, 18), defaultCentroid=['median','median'],
                   defaultPSFW=['median','median'], ncpu=4):
    """Gets the centroid of the target by flux weighted mean and the PSF width of the target.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds,
            i.e, give centroid in terms of the pixel value of the initial image.
        bounds (tuple, optional): Bounds of box around the target to exclude background . Default is (14, 18, 14, 18).
        defaultCentroid (list, optional): Default location for sigma clipped centroids.
            Default is median centroid position.
        defaultPSFW (list, optional): Default width for sigma clipped PSF widths.
            Default is median of widths.
    
    Returns:
        tuple: xo, yo, wx, wy (list, list, list, list). The updated lists of x-centroid, y-centroid,
            PSF width (x-axis), and PSF width (y-axis).
    """
    
    lbx, ubx, lby, uby = bounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    
    # Resorting to a bit of hackery to get tqdm to work with multiprocessing
    global pbar
    global results
    global func
    N = image_stack.shape[0]
    pbar = tqdm(total=N)
    results = [None] * N  # result list of correct size
    func = partial(fitgaussian, bounds, scale)

    pool = Pool(ncpu)
    for i in range(N):
        pool.apply_async(wrapMyFunc, args=(i,), callback=update)
    pool.close()
    pool.join()
    pbar.close()
    sys.stderr.flush()
    
    results = np.array(results)
    flux, xmean, ymean, widx, widy = results.T
    results = None
    cx = xmean+lbx
    cy = ymean+lby
    
    try:
        cx = sigma_clip(cx, sigma=5, maxiters=5, cenfunc=np.ma.median)
        cy = sigma_clip(cy, sigma=5, maxiters=5, cenfunc=np.ma.median)
    except TypeError:
        cx = sigma_clip(cx, sigma=5, iters=5, cenfunc=np.ma.median)
        cy = sigma_clip(cy, sigma=5, iters=5, cenfunc=np.ma.median)
    
    # Set masked centroids to default position
    if defaultCentroid[0]=='median':
        defaultCentroid[0] = np.ma.median(cx)
    if defaultCentroid[1]=='median':
        defaultCentroid[1] = np.ma.median(cy)
    cx[np.ma.getmaskarray(cx)] = defaultCentroid[0]
    cy[np.ma.getmaskarray(cy)] = defaultCentroid[1]
    
    xo = cx/scale
    yo = cy/scale
    
    try:
        widx    = sigma_clip(widx, sigma=5, maxiters=5, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=5, maxiters=5, cenfunc=np.ma.median)
    except TypeError:
        widx    = sigma_clip(widx, sigma=5, iters=5, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=5, iters=5, cenfunc=np.ma.median)
    
    # Set masked PSF widths to default width
    if defaultPSFW[0]=='median':
        defaultPSFW[0] = np.ma.median(widx)
    if defaultPSFW[1]=='median':
        defaultPSFW[1] = np.ma.median(widy)
    widx[np.ma.getmaskarray(widx)] = defaultPSFW[0]
    widx[np.ma.getmaskarray(widx)] = defaultPSFW[1]
    
    wx = widx/scale
    wy = widy/scale
    
    flux = sigma_clip(flux, sigma=5, maxiters=3, cenfunc=np.ma.median)
    
    return flux, xo, yo, wx, wy

def get_lightcurve(basepath, AOR_snip, channel, planet,
                   save=True, highpassWidth=5*64, bin_data=True, bin_size=64,
                   showPlots=False, savePlots=True,
                   oversamp=False, scale=2, saveoversamp=True, reuse_oversamp=True,
                   addStack = False, ignoreFrames = None,
                   maskStars = None, ncpu=4, image_stack_input=None, bg=None, bg_err=None, time=None):
    """Given a directory, looks for data (bcd.fits files), opens them and performs PSF photometry.

    Args:
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
        planet (string, optional): The name of the planet.
        save (bool, optional): True if you want to save the outputs. Default is True.
        bin_data (bool, optional): True you want to get binned data. Default is True.
        bin_size (int, optional): If bin_data is True, the size of the bins. Default is 64.
        oversamp (bool, optional): True if you want to oversample the image by a factor of 2. Default is False.
        save_oversamp (bool, optional): True if you want to save oversampled images. Default is True.
        reuse_oversamp (bool, optional): True if you want to reuse oversampled images that were previously saved.
            Default is False.
        ignoreFrames (list, optional) A list of frames to be masked when performing aperature photometry (e.g. first
            frame to remove first-frame systematic).
        maskStars (list, optional): An array-like object where each element is an array-like object with the RA and DEC
            coordinates of a nearby star which should be masked out when computing background subtraction.
        ncpu (int, optional): The number of aperture radii to try at the same time with multiprocessing. Default is 4.

    Raises: 
        Error: If Photometry method is not supported/recognized by this pipeline.
    
    """
    
    if ignoreFrames is None:
        ignoreFrames = []
        
    if maskStars is None:
        maskStars = []
    
    if basepath[-1]!='/':
        basepath += '/'
    
    # prepare filenames for saved data
    save_full = channel+'_datacube_full_AORs'+AOR_snip[1:]+'.dat'
    save_bin = channel+'_datacube_binned_AORs'+AOR_snip[1:]+'.dat'
    
    # Access the global variable
    global image_stack
    if image_stack_input is None:
        # Prepare all of the images
        image_stack, bg, bg_err, time = prepare_images(basepath, planet, channel, AOR_snip, ignoreFrames,
                                                       oversamp, scale, reuse_oversamp, saveoversamp,
                                                       addStack, maskStars, ncpu)
    else:
        image_stack = image_stack_input
    
    bg = clip_data(bg, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    bg_err = clip_data(bg_err, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    
    print('\tFitting Gaussians...', flush=True)
    flux, xo, yo, xw, yw = fit_2DGaussian(image_stack, scale = 1, bounds = (13, 18, 13, 18), ncpu=ncpu)
    
    # Clear up some RAM
    image_stack = None
    
    flux[flux<0.1*np.ma.median(flux)] = np.nan
    flux[flux<0.1*np.ma.median(flux)].mask = True

    flux = clip_data(flux, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    xo = clip_data(xo, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    xw = clip_data(xw, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    yo = clip_data(yo, highpassWidth, sigma1=10, sigma2=5, maxiters=3)
    yw = clip_data(yw, highpassWidth, sigma1=10, sigma2=5, maxiters=3)

    if bin_data:
        binned_flux, binned_flux_std = bin_array(flux, bin_size)
        binned_time, binned_time_std = bin_array(time, bin_size)
        binned_xo, binned_xo_std     = bin_array(xo, bin_size)
        binned_yo, binned_yo_std     = bin_array(yo, bin_size)
        binned_xw, binned_xw_std     = bin_array(xw, bin_size)
        binned_yw, binned_yw_std     = bin_array(yw, bin_size)
        binned_bg, binned_bg_std     = bin_array(bg, bin_size)

        # Do a rolling median based sigma clipping to remove bad data
        binned_flux = clip_data(binned_flux, highpassWidth/bin_size, sigma1=10, sigma2=5, maxiters=3)

    if save or savePlots:
        print('\tSaving... ', end='', flush=True)
        # create save folder
        if channel=='ch1':
            folder='3um'
        else:
            folder='4um'
        folder += 'PSF/'

        savepath = basepath+planet+'/analysis/'+channel+'/'
        if addStack:
            savepath += 'addedStack/'
        else:
            savepath += 'addedBlank/'
        if ignoreFrames != []:
            savepath += 'ignore/'
        else:
            savepath += 'noIgnore/'

        savepath = savepath+folder

        savepath = create_folder(savepath, True, True)

    if savePlots or showPlots:
        if bin_data:
            plotx = binned_time
            ploty0 = binned_flux
            ploty1 = binned_xo
            ploty2 = binned_yo
            ploty3 = binned_xw
            ploty4 = binned_yw
        else:
            plotx = time
            ploty0 = flux
            ploty1 = xo
            ploty2 = yo
            ploty3 = xw
            ploty4 = yw

        fig, axes = plt.subplots(nrows=5, ncols=1, sharex=True, figsize=(15,15))

        axes[0].set_title(planet, fontsize="x-large")
        axes[0].plot(plotx, ploty0,'k+')
        axes[0].set_ylabel("Stellar Flux (electrons)")

        axes[1].plot(plotx, ploty1, 'k+')
        axes[1].set_ylabel("$x_0$")

        axes[2].plot(plotx, ploty2, 'k+')
        axes[2].set_ylabel("$y_0$")

        axes[3].plot(plotx, ploty3, 'k+')
        axes[3].set_ylabel("$x_w$")

        axes[4].plot(plotx, ploty4, 'k+')
        axes[4].set_ylabel("$y_w$")

        fig.subplots_adjust(hspace=0)
        axes[4].set_xlabel("Time (BMJD))")
        axes[4].ticklabel_format(useOffset=False)

        if savePlots:
            # Save the plot if requested
            pathplot = savepath + 'Lightcurve.pdf'
            fig.savefig(pathplot)
        if showPlots:
            plt.show()
        plt.close()

    # Save the data if requested
    if save:
        FULL_data = np.c_[flux, time, xo, yo, xw, yw, bg]
        FULL_head = 'Flux, Time, x-centroid, y-centroid, x-PSF width, y-PSF width, bg flux'
        pathFULL  = savepath+save_full
        np.savetxt(pathFULL, FULL_data, header=FULL_head)
        if bin_data:
            BIN_data = np.c_[binned_flux, binned_flux_std, binned_time, binned_time_std,
                            binned_xo, binned_xo_std, binned_yo, binned_yo_std,
                            binned_xw, binned_xw_std, binned_yw, binned_yw_std,
                            binned_bg, binned_bg_std]
            BIN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std'
            BIN_head += ', x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, bg flux, bg flux std'
            pathBIN  = savepath+save_bin
            np.savetxt(pathBIN, BIN_data, header=BIN_head)
    
    print('Done.', flush=True)
    
    return
