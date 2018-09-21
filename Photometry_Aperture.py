import numpy as np

import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.ticker 
from matplotlib import rc
from matplotlib.ticker import MaxNLocator

import time
import os, sys

from astropy.io import fits
from astropy.stats import sigma_clip

from photutils import aperture_photometry
from photutils import CircularAperture
from photutils.datasets import make_4gaussians_image
from photutils.utils import calc_total_error

import glob
import csv
import operator
import warnings

#import error

#from photutils.morphology import centroid_1dg,centroid_2dg
#np.set_printoptions(threshold=np.nan)

def create_folder(fname):
    solved = 'no'
    while(solved == 'no'):
        #path = 	os.path.dirname(os.path.abspath(__file__)) + '/' + fname
        path = fname
        if not os.path.exists(path):
            os.makedirs(path)
            solved = 'yes'
        else :
            print('Error:', fname, 'already exist! Are you sure you want to overwrite this folder? (y/n)')
            answer = input()
            if (answer=='y'):
                solved = 'yes'
            else:
                print('What would you like the new folder name to be?')
                fname = input()
    return fname

def get_fnames(directory, AOR_snip, ch):
    '''
    Find paths to all the fits files.

    Parameters
    ----------

    :type directory : string object
    :param directory: Path to the directory containing all the Spitzer data.

    :type AOR_snip : string object
    :param AOR_snip: Common first characters of data directory eg. 'r579'

    :type ch : string objects
    :param ch: Channel used for the observation eg. 'ch1' for channel 1	

    Returns
    -------

    :return: fname - (list) - List of paths to all bcd.fits files.
    '''
    lst      = os.listdir(directory)
    AOR_list = [k for k in lst if AOR_snip in k] 
    #AOR_list = np.delete(AOR_list, [0, 4])                # used to ignore calibration data sets
    fnames   = []
    for i in range(len(AOR_list)):
        path = directory + '/' + AOR_list[i] + '/' + ch +'/bcd'	
        fnames.extend([filename for filename in glob.glob(os.path.join(path, '*bcd.fits'))])
    fnames.sort()
    return fnames

def get_time(hdu_list, time):
    '''
    Gets the time stamp for each image.

    Parameters
    ----------

    :type hdu_list : list
    :param hdu_list: content of fits file.

    :type time : 1D array
    :param time: Array of existing time stamps.

    Returns
    -------

    :return: time (1D array) - Updated time stamp array

    '''
    h, w, l = hdu_list[0].data.shape
    sec2day = 1.0/(3600.0*24.0)
    step    = hdu_list[0].header['FRAMTIME']*sec2day
    t       = np.linspace(hdu_list[0].header['BMJD_OBS'] + step/2, hdu_list[0].header['BMJD_OBS'] + (h-1)*step, h)
    time.extend(t)
    return time

def sigma_clipping(image_data, filenb = 0 , fname = ['not provided'], tossed = 0, badframetable = [], bounds = (13, 18, 13, 18)):
    '''
    Sigma clips bad pixels and mask entire frame if the sigma clipped
    pixel is too close to the target.

    Parameters
    ----------

    :param image_data: (3D Array) - Data cube of images (2D arrays of pixel values).

    :param filenb: (optional) - Index of current file in the 'fname' list (list of names of files) to keep track of the files that were tossed out. Default is 0.

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
    :returns: sigma_clipped_data (3D array) - Data cube of sigma clipped images (2D arrays of pixel values).
    :returns: tossed (int) - Updated total number of image tossed out.
    :returns: badframetable (list) - Updated list of file names and frame number of images tossed out from 'fname'.
    '''
    lbx, ubx, lby, uby = bounds
    h, w, l = image_data.shape
    # mask invalids
    image_data2 = np.ma.masked_invalid(image_data)
    # make mask to mask entire bad frame
    x = np.ones(shape = (w, l))
    mask = np.ma.make_mask(x)
    sig_clipped_data = sigma_clip(image_data2, sigma=4, iters=4, cenfunc=np.nanmedian, axis = 0)
    for i in range (h):
        oldstar = image_data[i, lbx:ubx, lby:uby]
        newstar = sig_clipped_data[i, lbx:ubx, lby:uby]
        truth   = newstar==oldstar
        if(truth.sum() < truth.size):
            sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
            badframetable.append([i,filenb,fname])
            tossed += 1
    return sig_clipped_data, tossed, badframetable

def bgsubtract(img_data, bg_flux = [], bg_err = [], bounds = (11, 19, 11, 19)):
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

    bg_flux   : 1D array
        Updated array of background flux measurements for previous 
        images.

    bg_err    : 1D array
        Updated array of uncertainties on background measurements for previous 
        images.
    '''
    lbx, ubx, lby, uby = bounds
    image_data = np.ma.copy(img_data)
    mask1 = image_data.mask
    h, w, l = image_data.shape
    x = np.zeros(shape = image_data.shape)
    x[:, lbx:ubx,lby:uby] = 1
    mask   = np.ma.make_mask(x+mask1)
    masked0 = np.ma.masked_array(image_data, mask = mask)
    masked = np.reshape(masked0, (h, w*l))
    bg_med = np.reshape(np.ma.median(masked, axis=1), (h, 1, 1))
    bgsub_data = image_data - bg_med
    bgsub_data = np.ma.masked_invalid(bgsub_data)
    bg_flux.extend(bg_med.ravel())
    bg_err.extend(np.ma.std(masked, axis=1))
    return bgsub_data, bg_flux, bg_err

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
        mask         = np.ma.getmask(image_masked)
        points       = np.where(mask == False)
        #points       = np.ma.nonzero(image_masked)
        image_compre = np.ma.compressed(image_masked)
        image_over[i,:,:] = interpolate.griddata(points, image_compre, (gridx, gridy), method = 'linear')
    return image_over/(a**2)

def centroid_FWM(image_data, xo = [], yo = [], wx = [], wy = [], scale = 1, bounds = (13, 18, 13, 18)):
    '''
    Gets the centroid of the target by flux weighted mean and the PSF width
    of the target.

    Parameters:
    -----------

        img_data :(3D array) 
            Data cube of images (2D arrays of pixel values).

        xo        : list (optional)
            List of x-centroid obtained previously. Default if none given is an 
            empty list.

        yo        : list (optional)
            List of y-centroids obtained previously. Default if none given is an 
            empty list.

        wx        : list (optional)
            List of PSF width (x-axis) obtained previously. Default if none given 
            is an empty list.

        wy        : list (optional)
            List of PSF width (x-axis) obtained previously. Default if none given 
            is an empty list.

        scale     : int (optional)
            If the image is oversampled, scaling factor for centroid and bounds, 
            i.e, give centroid in terms of the pixel value of the initial image.

        bounds    : tuple (optional)
            Bounds of box around the target to exclude background . Default is (11, 19 ,11, 19).
    
    Returns:
    --------

        xo        : list
            Updated list of x-centroid obtained previously.

        yo        : list
            Updated list of y-centroids obtained previously.

        wx        : list
            Updated list of PSF width (x-axis) obtained previously.

        wy        : list
            Updated list of PSF width (x-axis) obtained previously.
    '''
    lbx, ubx, lby, uby = bounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    starbox = image_data[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape
    # get centroid	
    X, Y    = np.mgrid[:w,:l]
    cx      = (np.sum(np.sum(X*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lbx
    cy      = (np.sum(np.sum(Y*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lby
    cx      = sigma_clip(cx, sigma=4, iters=2, cenfunc=np.ma.median)
    cy      = sigma_clip(cy, sigma=4, iters=2, cenfunc=np.ma.median)
    xo.extend(cx/scale)
    yo.extend(cy/scale)
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    widx    = np.sqrt(np.sum(np.sum(X2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    widy    = np.sqrt(np.sum(np.sum(Y2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    widx    = sigma_clip(widx, sigma=4, iters=2, cenfunc=np.ma.median)
    widy    = sigma_clip(widy, sigma=4, iters=2, cenfunc=np.ma.median)
    wx.extend(widx/scale)
    wy.extend(widy/scale)
    return xo, yo, wx, wy

def A_photometry(image_data, bg_err, factor = 1, ape_sum = [], ape_sum_err = [],
    cx = 15, cy = 15, r = 2.5, a = 5, b = 5, w_r = 5, h_r = 5, 
    theta = 0, shape = 'Circular', method='center'):
    '''
    Performs aperture photometry, first by creating the aperture (Circular,
    Rectangular or Elliptical), then it sums up the flux that falls into the 
    aperture.

    Parameters
    ==========

    image_data: 3D array 
        Data cube of images (2D arrays of pixel values).

    bg_err   : 1D array
        Array of uncertainties on pixel value.

    factor   : float (optional)
        Electron count to photon count factor. Default is 1 if none given.

    ape_sum  : 1D array (optional)
        Array of flux to append new flux values to. If 'None', the new values
        will be appended to an empty array

    ape_sum_err: 1D array (optional)
        Array of flux uncertainty to append new flux uncertainty values to. If 
        'None', the new values will be appended to an empty array.

    cx       : float or 1D array (optional)
        x-coordinate of the center of the aperture. Dimension must be equal to 
        dimension of cy. Default is 15.

    cy       : float or 1D array (optional)
        y-coordinate of the center of the aperture. Default is 15.

    r        : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Circular', c_radius is 
        the radius for the circular aperture. Default is 2.5.

    a        : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Elliptical', e_semix is
        the semi-major axis for elliptical aperture (x-axis). Default is 5.

    b        : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Elliptical', e_semiy is
        the semi-major axis for elliptical aperture (y-axis). Default is 5.

    w_r      : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Rectangular', r_widthx is
        the full width for rectangular aperture (x-axis). Default is 5.

    h_r      : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Rectangular', r_widthy is
        the full height for rectangular aperture (y-axis). Default is 5.

    theta    : int (optional)
        If phot_meth is 'Aperture' and ap_shape is 'Elliptical' or
        'Rectangular', theta is the angle of the rotation angle in radians 
        of the semimajor axis from the positive x axis. The rotation angle 
        increases counterclockwise. Default is 0.

    shape    : string object (optional)
        If phot_meth is 'Aperture', ap_shape is the shape of the aperture. 
        Possible aperture shapes are 'Circular', 'Elliptical', 'Rectangular'. 
        Default is 'Circular'.

    method   : string object (optional)
        If phot_meth is 'Aperture', apemethod is the method used to 
        determine the overlap of the aperture on the pixel grid. Possible 
        methods are 'exact', 'subpixel', 'center'. Default is 'exact'.

    Returns
    -------
    ape_sum  : 1D array
        Array of flux with new flux appended.

    ape_sum_err: 1D array
        Array of flux uncertainties with new flux uncertainties appended.

    '''
    l, h, w = image_data.shape
    # central position of aperture
    # position = np.c_[cx, cy] # remove when uncommenting below
    if (type(cx) is list):
        position = np.c_[cx, cy]
    else:
        position = np.c_[(cx*np.ones(l)), (cy*np.ones(l))]
    tmp_sum = []
    tmp_err = []
    # performing aperture photometry
    for i in range(l):
        #aperture = CircularAperture(position[i], r=r) # remove when uncommenting below
        if   (shape == 'Circular'):
            aperture = CircularAperture(position[i], r=r)
        elif (shape == 'Elliptical'):
            aperture = EllipticalAperture(position[i], a=a, b=b, theta=theta)
        elif (shape == 'Rectangular'):
            aperture = RectangularAperture(position[i], w=w_r, h=h_r, theta=theta)
        data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
        phot_table = aperture_photometry(image_data[i,:,:],aperture, error=data_error, method=method)
        tmp_sum.extend(phot_table['aperture_sum']*factor)
        tmp_err.extend(phot_table['aperture_sum_err']*factor)
    # removing outliers
    tmp_sum = sigma_clip(tmp_sum, sigma=4, iters=2, cenfunc=np.ma.median)
    tmp_err = sigma_clip(tmp_err, sigma=4, iters=2, cenfunc=np.ma.median)
    ape_sum.extend(tmp_sum)
    ape_sum_err.extend(tmp_err)
    return ape_sum, ape_sum_err

def get_pixel_values(image_data0, P, box = 3):
    img = np.ma.masked_invalid(image_data0)
    h, w, l = img.shape
    P_tmp = np.empty(shape=(h, box**2))
    x0, y0 = 15, 15
    if box == 3:
        for i in range(h):
            P_tmp[i,:]   = np.array([img[i,x0-1,y0-1], img[i,x0-1,  y0], img[i,x0-1,y0+1],
                                     img[i,x0  ,y0-1], img[i,x0  ,  y0], img[i,x0  ,y0+1],
                                     img[i,x0+1,y0-1], img[i,x0+1,  y0], img[i,x0+1,y0+1]])
    elif box == 5:
        for i in range(h):
            P_tmp[i,:]   = np.array([img[i,x0-2,y0-2], img[i,x0-2,y0-1], img[i,x0-2,  y0], img[i,x0-2,y0+1], img[i,x0-2,y0+2],
                                     img[i,x0-1,y0-2], img[i,x0-1,y0-1], img[i,x0-1,  y0], img[i,x0-1,y0+1], img[i,x0-1,y0+2],
                                     img[i,x0  ,y0-2], img[i,x0  ,y0-1], img[i,x0  ,  y0], img[i,x0  ,y0+1], img[i,x0  ,y0+2],
                                     img[i,x0+1,y0-2], img[i,x0+1,y0-1], img[i,x0+1,  y0], img[i,x0+1,y0+1], img[i,x0+1,y0+2],
                                     img[i,x0+2,y0-2], img[i,x0+2,y0-1], img[i,x0+2,  y0], img[i,x0+2,y0+1], img[i,x0+2,y0+2]])
    else:
        raise RuntimeError('Sorry, not supported only box = 3,5 possible!')
        
    P = np.append(P, P_tmp, axis = 0)
    return P

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
    reshaped_data   = data.reshape((int(len(data)/size), size))
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.std(reshaped_data, axis=1)
    return binned_data, binned_data_std

def binning_data2D(data, size):
    data = np.ma.masked_invalid(data)
    h, w = data.shape
    reshaped_data   = data.reshape((int(h/size), size, w))
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.ma.std(reshaped_data, axis=1)
    return binned_data, binned_data_std