#--------------------------------------------------------------
#Author: Lisa Dang
#Created: 2016-11-09 1:21 AM EST
#Last Modified: 
#Title: PSF Fitting
#--------------------------------------------------------------

import numpy as np
from scipy import interpolate
from scipy import stats as st
from scipy import optimize as opt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from astropy.io import fits
from astropy.stats import sigma_clip

import glob
import csv
import time as tim
import os, sys
import warnings

def get_fnames(directory, AOR_snip, ch):
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
    AOR_list = [k for k in lst if AOR_snip in k] 
    #AOR_list = np.delete(AOR_list, [0, 4])                # used to ignore calibration data sets
    fnames   = []
    for i in range(len(AOR_list)):
        path = directory + '/' + AOR_list[i] + '/' + ch +'/bcd' 
        fnames.extend([filename for filename in glob.glob(os.path.join(path, '*bcd.fits'))])
    fnames.sort()
    return fnames, len(fnames)

def get_time(hdu_list, time):
    '''
    Gets the time stamp for each image.

    Parameters
    ----------

    hdu_list : 
        content of fits file.

    time     : 1D array
        Time array to append new time stamps to.

    Returns
    -------

    time     : 1D array
        Time array with new values appended.
    '''
    h, w, l = hdu_list[0].data.shape
    t       = np.linspace(hdu_list[0].header['AINTBEG'] + hdu_list[0].header['EXPTIME']/2, hdu_list[0].header['ATIMEEND'] - hdu_list[0].header['EXPTIME']/2, h)
    sec2day = 1.0/(3600.0*24.0)
    t       = sec2day*t
    time.extend(t)
    return time

def sigma_clipping(image_data, filenb = 0 , fname = ['not provided'], tossed = 0, badframetable = [], bounds = (11, 19, 11, 19)):
    '''
    Sigma clips bad pixels and mask entire frame if the sigma clipped
    pixel is too close to the target.

    Parameters
    ----------

    image_data: 3D array 
        Data cube of images (2D arrays of pixel values).

    filenb    : int (optional)
        Index of current file in the 'fname' list (list of names of files)
        to keep track of the files that were tossed out. Default is 0.

    fname     : list (optional)
        list (list of names of files) to keep track of the files that were 
        tossed out. 

    tossed    : int (optional)
        Total number of image tossed out. Default is 0 if none provided.

    badframetable: list (optional)
        List of file names and frame number of images tossed out from 'fname'.

    bounds    : tuple (optional)
        Bounds of box around the target. Default is (11, 19 ,11, 19).


    Returns
    -------

    sig_clipped_data: 3D array
        Data cube of sigma clipped images (2D arrays of pixel values).

    tossed    : int
        Updated total number of image tossed out.

    badframetable: list
        Updated list of file names and frame number of images tossed 
        out from 'fname'.
    '''
    lbx, ubx, lby, uby = bounds
    h, w, l = image_data.shape
    # mask invalids
    image_data2 = np.ma.masked_invalid(image_data)
    # make mask to mask entire bad frame
    x = np.ones(shape = (w, l))
    mask = np.ma.make_mask(x)
    sig_clipped_data = sigma_clip(image_data2, sigma=4, iters=2, cenfunc=np.ma.median, axis = 0)
    for i in range (h):
        oldstar = image_data[i, lbx:ubx, lby:uby]
        newstar = sig_clipped_data[i, lbx:ubx, lby:uby]
        truth   = newstar==oldstar
        if(truth.sum() < truth.size):
            sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
            badframetable.append([i,filenb,fname])
            tossed += 1
    return sig_clipped_data, tossed, badframetable

def bgsubtract(img_data, bg_err = [], bounds = (11, 19, 11, 19)):
    '''
    Measure the background level and subtracts the background from
    each frame.

    Parameters
    ----------

    img_data  : 3D array 
        Data cube of images (2D arrays of pixel values).

    bg_err    : 1D array (optional)
        Array of uncertainties on background measurements for previous images.
        Default if none given is an empty list

    bounds    : tuple (optional)
        Bounds of box around the target to exclude from the background level
        measurements. Default is (11, 19 ,11, 19).


    Returns
    -------

    bgsub_data: 3D array
        Data cube of sigma clipped images (2D arrays of pixel values).

    bg_err    : 1D array
        Updated array of uncertainties on background measurements for previous 
        images.
    '''
    lbx, ubx, lby, uby = bounds
    image_data = np.copy(img_data)
    h, w, l = image_data.shape
    x = np.zeros(shape = image_data.shape)
    x[:, lbx:ubx,lby:uby] = 1
    mask   = np.ma.make_mask(x)
    masked = np.ma.masked_array(image_data, mask = mask)
    masked = np.reshape(masked, (h, w*l))
    bg_med = np.reshape(np.ma.median(masked, axis=1), (h, 1, 1))
    bgsub_data = image_data - bg_med
    bg_err.extend(np.ma.std(masked, axis=1))
    bgsub_data = np.ma.masked_invalid(bgsub_data)
    return bgsub_data, bg_err

def oversampling(image_data, a = 2):
    '''
    First, substitutes all invalid/sigmaclipped pixel by interpolating the value.
    Then oversamples the image.

    Parameters
    ----------

    image_data: 3D array 
        Data cube of images (2D arrays of pixel values).

    a         : int (optional)
        Sampling factor, e.g. if a = 2, there will be twice as much data points in
        the x and y axis. Default is 2. (Do not recommend larger than 2)

    Returns
    -------
    image_over: 3D array
        Data cube of oversampled images (2D arrays of pixel values).
    '''
    l, h, w = image_data.shape
    gridx, gridy = np.mgrid[0:h:1/a, 0:w:1/a]
    image_over = np.empty((l, h*a, w*a))
    for i in range(l):
        image_masked = np.ma.masked_invalid(image_data[i,:,:])
        points       = np.ma.nonzero(image_masked)
        image_compre = np.ma.compressed(image_masked)
        image_over[i,:,:] = interpolate.griddata(points, image_compre, (gridx, gridy), method = 'linear')
    return image_over

def Gaussian2D(position, amp, xo, yo, sigx, sigy):
    '''
    Create a 2D Gaussian array.

    Parameters
    ----------

    position  : 3D array 
        Meshgrids of x and y indices of pixels. position[:,:,0] = x and
        position[:,:,1] = y.

    amp       : float
        Amplitude of the 2D Gaussian.

    xo        : float
        x value of the peak of the 2D Gaussian.

    yo        : float
        y value of the peak of the 2D Gaussian.

    sigx      : float
        Width of the 2D Gaussian along the x axis.

    sigy      : float
        Width of the 2D Gaussian along the y axis.

    Returns
    -------
    PSF.ravel(): 1D array
        z values of the 2D Gaussian raveled.
    '''
    centroid = [yo, xo]
    cov = [[sigy**2, 0],[0, sigx**2]]
    rv = st.multivariate_normal(mean = centroid, cov = cov)
    PSF = amp*(rv.pdf(position))
    return PSF.ravel()

def imagefit(image_data, popt = [], pcov = [], tossed =0, scale = 1,
    tbounds = (9, 20, 9, 20), 
    pinit = (18000, 15, 15, 2, 2), 
    ub = (25000, 15.3, 15.3, 5, 5),
    lb = (3000, 14.6, 14.6, 0.3, 0.3)):
    '''
    CFit a 2D Gaussian on the image.

    Parameters
    ----------

    image_data: 3D array
        Data cube of images (2D arrays of pixel values).

    popt      : 2D array (optional)
        Array of optimized parameters to append to.

    pcov      : 3D array (optional)
        Array of covariance matrix to append to.

    tossed    : int (optional)
        Total number of image tossed out. Default is 0 if none provided. Default is 0.

    scale     : int (optional)
        If the image is over sampled, scaling factor for centroid and bounds, 
        i.e, give centroid in terms of the pixel value of the initial image.
        Default is 1.

    bounds    :

    pinit     :

    ub        :

    lb        :
 

    Returns
    -------
    PSF.ravel(): 1D array
        z values of the 2D Gaussian raveled.
    '''
    lbx, ubx, lby, uby = tbounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    l, h, w = image_data.shape
    x, y = np.mgrid[lbx/scale:ubx/scale:1/scale, lby/scale:uby/scale:1/scale]
    position = np.empty(x.shape + (2,))
    position[:,:,0] = x 
    position[:,:,1] = y
    popt_tmp = np.empty((l,len(pinit)))
    pcov_tmp = np.empty((l,len(pinit), len(pinit)))
    for i in range(l):
        dataravel = image_data[i,lbx:ubx,lby:uby].ravel()
        try:
            popt_tmp[i,:], pcov_tmp[i,:,:] = opt.curve_fit(Gaussian2D, position, dataravel, p0=pinit, bounds = (lb,ub))
        except RuntimeError:
            print("Error - curve_fit failed")
            popt_tmp[i,:] = np.nan
            tossed += 1 
    popt = np.append(popt, popt_tmp, axis = 0)
    pcov = np.append(pcov, pcov_tmp, axis = 0)
    return popt, pcov, tossed

def binning_data(data, size):
    '''
    Median bin an array.

    Parameters
    ----------
    data     : 1D array
        Array of data to be binned.

    size     : int
        Size of bins.

    Returns
    -------
    binned_data: 1D array
        Array of binned data.

    binned_data: 1D array
        Array of standard deviation for each entry in binned_data.
    '''
    data = np.ma.masked_invalid(data) 
    reshaped_data   = data.reshape((len(data)/size, size))
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.std(reshaped_data, axis=1)
    return binned_data, binned_data_std

def get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
    save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
    bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
    plot_name= 'CoRoT-2b.pdf', oversamp = False, **kwargs):
    '''
    Given a directory, looks for data (bcd.fits files), opens them and performs photometry.

    Parameters
    ----------
    datapath : string object
        Directory where the spitzer data is stored.

    savepath : string object
        Directory the outputs will be saved.

    AORsnip  : string objects
        Common first characters of data directory eg. 'r579'

    channel  : string objects
        Channel used for the observation eg. 'ch1' for channel 1

    subarray : bool
        True if observation were taken in subarray mode. False if 
        observation were taken in full-array mode.

    save     : bool (optional)
        True if you want to save the outputs. Default is True.

    save_full: string object (optional)
        Filename of the full unbinned output data. Default is 
        '/ch2_datacube_full_AORs579.dat'.

    bin_data : bool (optional)
        True you want to get binned data. Default is True.

    bin_size : int (optional)
        If bin_data is True, the size of the bins. Default is 64.

    save_bin : string object (optional)
        Filename of the full binned output data. Default is 
        '/ch2_datacube_binned_AORs579.dat'.

    plot     : bool (optional)
        True if you want to plot the time resolved lightcurve. 
        Default is True.

    plot_name: string object (optional)
        If plot and save is True, the filename of the plot to be 
        saved as. Default is True.

    oversamp : bool (optional)
        True if you want to oversample you image. Default is True.

    **kwargs : dictionary
        Argument passed onto other functions.

    Raises
    ------
    Error      : 
        If Photometry method is not supported/recognized by this pipeline.
    '''

    # Ignore warning and starts timing
    warnings.filterwarnings('ignore')
    tic = tim.clock()

    # get list of filenames and nb of files
    fnames, nfiles = get_fnames(datapath, AOR_snip, channel)

    # variables declaration 
    percent       = 0                                # to show progress while running the code
    tossed        = 0                                # Keep tracks of number of frame discarded 
    badframetable = []                               # list of filenames of the discarded frames
    time          = []                               # time array
    bg_err        = []                               # background flux error 
    popt          = np.empty(shape = (0,5))
    pcov          = np.empty(shape = (0,5,5)) 
    #xo            = []                               # centroid value along the x-axis
    #yo            = []                               # centroid value along the y-axis
    #xw            = []                               # PSF width along the x-axis
    #yw            = []                               # PSF width along the y-axis
    #aperture_sum  = []                               # flux obtained from aperture photometry
    #aperture_sum_err = []                            c# error on flux obtained from aperture photometry

    #image_data_full=np.zeros((64*nfiles, 32, 32))
    #factor = np.zeros(64*nfiles)

    # data reduction & aperture photometry part
    if (subarray == True):
        for i in range(nfiles):
            # open fits file
            hdu_list = fits.open(fnames[i])
            image_data0 = hdu_list[0].data
            h, w, l = image_data0.shape
            # get time
            time = get_time(hdu_list, time)
            # convert MJy/str to electron count
            convfact = hdu_list[0].header['GAIN']*hdu_list[0].header['EXPTIME']/hdu_list[0].header['FLUXCONV']
            image_data1 = convfact*image_data0
            # sigma clip
            fname = fnames[i]
            image_data2, tossed, badframetable = sigma_clipping(image_data1, i ,fname[fname.find('ch2/bcd/')+8:], tossed=tossed, **kwargs)
            # bg subtract
            image_data3, bg_err = bgsubtract(image_data2, bg_err)
            # oversampling & Photometry
            if (oversamp == True):
                image_data3 = np.ma.masked_invalid(oversampling(image_data3))
                popt, pcov, tossed = imagefit(image_data3,popt, pcov,  tossed, scale = 2)
            else:
                popt, pcov, tossed = imagefit(image_data3, popt, pcov, tossed)
            print('Status:', i, 'out of', nfiles)


    elif (subarray == False):
        print('Sorry this part is undercontruction!')
    
    if (bin_data == True):
        binned_flux, binned_flux_std = binning_data(popt[:,0], bin_size)
        binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)
        binned_xo, binned_xo_std     = binning_data(popt[:,1], bin_size)
        binned_yo, binned_yo_std     = binning_data(popt[:,2], bin_size)
        binned_xw, binned_xw_std     = binning_data(popt[:,3], bin_size)
        binned_yw, binned_yw_std     = binning_data(popt[:,4], bin_size)
    
    if (plot == True):
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15,5))
        fig.suptitle("CoRoT-2b", fontsize="x-large")
        axes[0].plot(binned_time, binned_flux,'k+')
        axes[0].set_ylabel("Stellar Flux (MJy/pixel)")

        axes[1].plot(binned_time, binned_xo, '+')
        axes[1].set_ylabel("$x_0$")

        axes[2].plot(binned_time, binned_yo, 'r+')
        axes[2].set_xlabel("Time since IRAC turn-on (days)")
        axes[2].set_ylabel("$y_0$")
        fig.subplots_adjust(hspace=0)
        if (save == True):
            pathplot = savepath + '/' + plot_name
            fig.savefig(pathplot)
        else :
            plt.show()

    if (save == True):
        FULL_data = np.c_[popt.T, time]
        FULL_head = 'Flux, x-centroid, y-centroid, x-PSF width, y-PSF width, time'
        BINN_data = np.c_[binned_flux, binned_flux_std, binned_time, binned_time_std, binned_xo, binned_xo_std, binned_yo, binned_yo_std, binned_xw, binned_xw_std, binned_yw, binned_yw_std]
        BINN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std, x-PSF width, x-PSF width std, y-PSF width, y-PSF width std'
        pathFULL  = savepath + save_full
        pathBINN  = savepath + save_bin
        np.savetxt(pathFULL, FULL_data, header = FULL_head)
        np.savetxt(pathBINN, BINN_data, header = BINN_head)
    
    toc = tim.clock()
    print('Number of discarded frames:', tossed)
    print('Time:', toc-tic, 'seconds')

if __name__=='__main__': main()