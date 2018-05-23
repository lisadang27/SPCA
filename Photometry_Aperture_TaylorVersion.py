#--------------------------------------------------------------
#Author: Lisa Dang
#Created: 2016-11-09 1:21 AM EST
#Branched: 2017-05-18 from main to taylor
#Last Modified: 2017-10-24 by Taylor Bell
#Title: Aperture Photometry
#--------------------------------------------------------------
import numpy as np
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from astropy.io import fits
from astropy.stats import sigma_clip
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord

from photutils import aperture_photometry
from photutils import CircularAperture, EllipticalAperture, RectangularAperture
from photutils.utils import calc_total_error

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

def get_time(hdu_list, time, ignoreFrames):
    '''
    Gets the time stamp for each image.

    Parameters
    ----------

    :type hdu_list : list
    :param hdu_list: content of fits file.

    :type time : 1D array
    :param time: Array of existing time stamps.
    
    :type ignoreFrames : 1D array
    :param ignoreFrames: Array of frames to ignore (consistently bad frames).

    Returns
    -------

    :return: time (1D array) - Updated time stamp array

    '''
    h, w, l = hdu_list[0].data.shape
    sec2day = 1.0/(3600.0*24.0)
    step    = hdu_list[0].header['FRAMTIME']*sec2day
    t       = np.linspace(hdu_list[0].header['BMJD_OBS'] + step/2, hdu_list[0].header['BMJD_OBS'] + (h-1)*step, h)
    t = np.delete(t, ignoreFrames, axis=0)
    time.extend(t)
    return time

def sigma_clipping(image_data, filenb = 0 , fname = ['not provided'], tossed = 0, badframetable = [], bounds = (13, 18, 13, 18), sigma=4, iters=2):
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
    sig_clipped_data = sigma_clip(image_data2, sigma=sigma, iters=iters, cenfunc=np.ma.median, axis = 0)
    for i in range (h):
        if np.ma.is_masked(sig_clipped_data[i, lbx:ubx, lby:uby]):
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
    h, w, l = image_data.shape
    x = np.zeros(shape = image_data.shape)
    x[:, lbx:ubx,lby:uby] = 1
    mask   = np.ma.make_mask(x)
    masked = np.ma.masked_array(image_data, mask = mask)
    masked = np.reshape(masked, (h, w*l))
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

def centroid_FWM(image_data, xo = [], yo = [], wx = [], wy = [], scale = 1, bounds = (14, 18, 14, 18)):
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

    cx       : int (optional)
        x-coordinate of the center of the aperture. Default is 15.

    cy       : int (optional)
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
        methods are 'exact', 'subpixel', 'center'. Default is 'center'.

    Returns
    -------
    ape_sum  : 1D array
        Array of flux with new flux appended.

    ape_sum_err: 1D array
        Array of flux uncertainties with new flux uncertainties appended.

    '''
    l, h, w  = image_data.shape
    position = [cx, cy]
    tmp_sum  = []
    tmp_err  = []
    if   (shape == 'Circular'):
        aperture = CircularAperture(position, r=r)
    elif (shape == 'Elliptical'):
        aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
    elif (shape == 'Rectangular'):
        aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
    for i in range(l):
        data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
        phot_table = aperture_photometry(image_data[i,:,:], aperture, error=data_error, method=method)#, pixelwise_error=False)
        tmp_sum.extend(phot_table['aperture_sum']*factor)
        tmp_err.extend(phot_table['aperture_sum_err']*factor)
    # removing outliers
    tmp_sum = sigma_clip(tmp_sum, sigma=4, iters=2, cenfunc=np.ma.median)
    tmp_err = sigma_clip(tmp_err, sigma=4, iters=2, cenfunc=np.ma.median)
    ape_sum.extend(tmp_sum)
    ape_sum_err.extend(tmp_err)
    return ape_sum, ape_sum_err

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
    reshaped_data   = data.reshape(int(len(data)/size), size)
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.std(reshaped_data, axis=1)
    return binned_data, binned_data_std


def get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
    save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
    bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
    plot_name= 'Lightcurve.pdf', oversamp = False, saveoversamp = True, reuse_oversamp = False,
    planet = 'CoRoT-2b', r = 2.5, shape = 'Circular', edge='hard', addStack = False,
    stackPath = '', ignoreFrames = [], maskStars = [], **kwargs):

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
        
    shape    : string object (optional)
        If phot_meth is 'Aperture', ap_shape is the shape of the aperture. 
        Possible aperture shapes are 'Circular', 'Elliptical', 'Rectangular'. 
        Default is 'Circular'.
    
    edge     : string object (optional)
        A string specifying the type of aperture edge to be used.
        Options are 'hard', 'soft', and 'exact' which correspond
        to the 'center', 'subpixel', and 'exact' methods. Default is
        'hard'.

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
        True if you want to oversample you image. Default is False.
        
    save_oversamp : bool (optional)
        True if you want to save oversampled images. Default is True.
        
    reuse_oversamp : bool (optional)
        True if you want to reuse oversampled images that were previously
        saved. Default is False.
        
    planet : string object (optional)
        The name of the planet. Default is CoRoT-2b.
        
    r : float (optional)
        The radius to use for aperture photometry in units of pixels.
        Default is 2.5 pixels.
        
    ignoreFrames : array-like (optional)
        An array-like object, with a list of frames to be masked when
        performing aperature photometry (e.g. first frame to remove
        first-frame systematic).
        
    maskStars : array-like (optional)
        An array-like object where each element is an array-like object
        with the RA and DEC coordinates of a nearby star which should be
        masked out when computing background subtraction.

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
    
    if edge.lower()=='hard':
        method = 'center'
    elif edge.lower()=='soft':
        method = 'subpixel'
    elif edge.lower()=='exact':
        method = 'exact'
    else:
        print("No such method \""+edge+"\". Using hard edged aperture")
        method = 'center'

    # get list of filenames and nb of files
    fnames = get_fnames(datapath, AOR_snip, channel)
    
    bin_size = bin_size - len(ignoreFrames)

    # variables declaration 
    percent       = 0                                # to show progress while running the code
    tossed        = 0                                # Keep tracks of number of frame discarded 
    badframetable = []                               # list of filenames of the discarded frames
    flux          = []                               # flux obtained from aperture photometry
    flux_err      = []                               # error on flux obtained from aperture photometry
    time          = []                               # time array
    xo            = []                               # centroid value along the x-axis
    yo            = []                               # centroid value along the y-axis
    xw            = []                               # PSF width along the x-axis
    yw            = []                               # PSF width along the y-axis
    bg_flux       = []                               # background flux
    bg_err        = []                               # background flux error 
    
    # variables declaration for binned data
    binned_flux          = []                        # binned flux obtained from aperture photometry
    binned_flux_std      = []                        # std.dev in binned error on flux obtained from aperture photometry
    binned_time          = []                        # binned time array
    binned_time_std      = []                        # std.dev in binned time array
    binned_xo            = []                        # binned centroid value along the x-axis
    binned_xo_std        = []                        # std.dev in binned centroid value along the x-axis
    binned_yo            = []                        # binned centroid value along the y-axis
    binned_yo_std        = []                        # std.dev in binned centroid value along the y-axis
    binned_xw            = []                        # binned PSF width along the x-axis
    binned_xw_std        = []                        # std.dev in binned PSF width along the x-axis
    binned_yw            = []                        # binned PSF width along the y-axis
    binned_yw_std        = []                        # std.dev in binned PSF width along the y-axis
    binned_bg            = []                        # binned background flux
    binned_bg_std        = []                        # std.dev in binned background flux
    binned_bg_err        = []                        # binned background flux error 
    binned_bg_err_std    = []                        # std.dev in binned background flux error 
    
    # data reduction & aperture photometry part
    if (subarray == True):
        for i in range(len(fnames)):
            # open fits file
            hdu_list = fits.open(fnames[i])
            image_data0 = hdu_list[0].data
            # get time
            time = get_time(hdu_list, time, ignoreFrames)
            #add background correcting stack if requested
            if addStack:
                stackHDU = fits.open(stackPath)
                image_data0 += stackHDU[0].data
            #ignore any consistently bad frames
            if ignoreFrames != []:
                image_data0 = np.delete(image_data0, ignoreFrames, axis=0)
            h, w, l = image_data0.shape
            # convert MJy/str to electron count
            convfact = hdu_list[0].header['GAIN']*hdu_list[0].header['EXPTIME']/hdu_list[0].header['FLUXCONV']
            image_data1 = convfact*image_data0
            # sigma clip
            fname = fnames[i]
            image_data2, tossed, badframetable = sigma_clipping(image_data1, i ,fname[fname.find('/bcd/')+5:], 
                                                                badframetable=badframetable, tossed=tossed)
            
            if maskStars is not None:
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
            # oversampling
            if (oversamp == True):
                if (reuse_oversamp):
                    savename = savepath + '/Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
                    if os.path.isfile(savename):
                        image_data3 = np.load(savename)
                    else:
                        print('Warning: Oversampled images were not previously saved! Making new ones now...')
                        image_data3 = np.ma.masked_invalid(oversampling(image_data3))
                        if (saveoversamp == True):
                            # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                            image_data3.dump(savename)
                else:
                    image_data3 = np.ma.masked_invalid(oversampling(image_data3))
                    
                if (saveoversamp == True):
                    # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                    savename = savepath + '/Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
                    image_data3.dump(savename)
                
                # Aperture Photometry
                # get centroids & PSF width
                xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw, scale = 2)
                # convert electron count to Mjy/str
                ecnt2Mjy = - hdu_list[0].header['PXSCAL1']*hdu_list[0].header['PXSCAL2']*(1/convfact) 
                # aperture photometry
                flux, flux_err = A_photometry(image_data3, bg_err[-h:], ecnt2Mjy, flux, flux_err, cx=2*15, cy=2*15,
                                              r=2*r, a=2*5, b=2*5, w_r=2*5, h_r=2*5, shape=shape, method=method, **kwargs)
            else :
                # get centroids & PSF width
                xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw)
                # convert electron count to Mjy/str
                ecnt2Mjy = - hdu_list[0].header['PXSCAL1']*hdu_list[0].header['PXSCAL2']*(1/convfact) 
                # aperture photometry
                flux, flux_err = A_photometry(image_data3, bg_err[-h:], ecnt2Mjy, flux, flux_err, r=r, shape=shape, 
                                              method=method, **kwargs)
            if ((i+1)%100 == 0 or i+1==len(fnames) or i==0):
                print('Status:', i+1, 'out of', len(fnames))

    elif (subarray == False):
        print('Sorry this part is undercontruction!')

    if (bin_data == True):
        binned_flux, binned_flux_std = binning_data(np.asarray(flux), bin_size)
        binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)
        binned_xo, binned_xo_std     = binning_data(np.asarray(xo), bin_size)
        binned_yo, binned_yo_std     = binning_data(np.asarray(yo), bin_size)
        binned_xw, binned_xw_std     = binning_data(np.asarray(xw), bin_size)
        binned_yw, binned_yw_std     = binning_data(np.asarray(yw), bin_size)
        binned_bg, binned_bg_std     = binning_data(np.asarray(bg_flux), bin_size)
        binned_bg_err, binned_bg_err_std = binning_data(np.asarray(bg_err), bin_size)

        #sigma clip binned data to remove wildly unacceptable data
        binned_flux_mask = sigma_clip(binned_flux, sigma=10, iters=2)
        if np.ma.is_masked(binned_flux_mask):
            binned_time = binned_time[binned_flux_mask==binned_flux]
            binned_time_std = binned_time_std[binned_flux_mask==binned_flux]
            binned_xo = binned_xo[binned_flux_mask==binned_flux]
            binned_xo_std = binned_xo_std[binned_flux_mask==binned_flux]
            binned_yo = binned_yo[binned_flux_mask==binned_flux]
            binned_yo_std = binned_yo_std[binned_flux_mask==binned_flux]
            binned_xw = binned_xw[binned_flux_mask==binned_flux]
            binned_xw_std = binned_xw_std[binned_flux_mask==binned_flux]
            binned_yw = binned_yw[binned_flux_mask==binned_flux]
            binned_yw_std = binned_yw_std[binned_flux_mask==binned_flux]
            binned_bg = binned_bg[binned_flux_mask==binned_flux]
            binned_bg_std = binned_bg_std[binned_flux_mask==binned_flux]
            binned_bg_err = binned_bg_err[binned_flux_mask==binned_flux]
            binned_bg_err_std = binned_bg_err_std[binned_flux_mask==binned_flux]
            binned_flux_std = binned_flux_std[binned_flux_mask==binned_flux]
            print('Clipped', len(binned_flux)-np.sum(binned_flux_mask==binned_flux), 'binned points (>10 sigma outliers)')
            binned_flux = binned_flux[binned_flux_mask==binned_flux]

    if (plot == True):
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15,5))
        fig.suptitle(planet, fontsize="x-large")
        
        axes[0].plot(binned_time, binned_flux,'k+', color='black')
        #axes[0].plot(binned_time_clipped, binned_flux_clipped,'k+', color='black')
        #axes[0].plot(time, flux,'k+', color='lightgrey')
        axes[0].set_ylabel("Stellar Flux (MJy/str)")

        axes[1].plot(binned_time, binned_xo, '+', color='black')
        #axes[1].plot(binned_time_clipped, binned_xo_clipped, '+', color='black')
        #axes[1].plot(time, xo, '+', color='lightgrey')
        axes[1].set_ylabel("$x_0$")

        axes[2].plot(binned_time, binned_yo, 'r+', color='black')
        #axes[2].plot(binned_time_clipped, binned_yo_clipped, 'r+', color='black')
        #axes[2].plot(time, yo, 'r+', color='lightgrey')
        axes[2].set_xlabel("Time (BMJD))")
        axes[2].set_ylabel("$y_0$")
        fig.subplots_adjust(hspace=0)
        axes[2].ticklabel_format(useOffset=False)
        
        if (save == True):
            pathplot = savepath + '/' + plot_name
            fig.savefig(pathplot)
        else :
            plt.show()

    if (save == True):
        FULL_data = np.c_[flux, flux_err, time, xo, yo, xw, yw, bg_flux, bg_err]
        FULL_head = 'Flux, Flux Uncertainty, Time, x-centroid, y-centroid, x-PSF width, y-PSF width, bg flux, bg flux err'
        BINN_data = np.c_[binned_flux, binned_flux_std, binned_time, binned_time_std, binned_xo, binned_xo_std, binned_yo,
                          binned_yo_std, binned_xw, binned_xw_std, binned_yw, binned_yw_std,  binned_bg, binned_bg_std,
                          binned_bg_err, binned_bg_err_std]
        BINN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std, x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, bg flux, bg flux std, bg flux err, bg flux err std]'
        pathFULL  = savepath +'/'+ save_full
        pathBINN  = savepath +'/'+ save_bin
        np.savetxt(pathFULL, FULL_data, header = FULL_head)
        np.savetxt(pathBINN, BINN_data, header = BINN_head)

    toc = tim.clock()
    print('Number of discarded frames:', tossed)
    print('Time:', toc-tic, 'seconds')
