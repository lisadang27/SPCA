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

from photutils import aperture_photometry
from photutils import CircularAperture, EllipticalAperture, RectangularAperture
from photutils.utils import calc_total_error

import os, sys, csv, glob, warnings

from multiprocessing import Pool
from functools import partial

from collections import Iterable

from .Photometry_Common import get_fnames, get_stacks, get_time, oversampling
from .Photometry_Common import sigma_clipping, bgsubtract, noisepixparam, bin_array, create_folder

def centroid_FWM(image_data, scale=1, bounds=(13, 18, 13, 18)):
    """Gets the centroid of the target by flux weighted mean and the PSF width of the target.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds,
            i.e, give centroid in terms of the pixel value of the initial image.
        bounds (tuple, optional): Bounds of box around the target to exclude background . Default is (14, 18, 14, 18).
    
    Returns:
        tuple: xo, yo, wx, wy (list, list, list, list). The updated lists of x-centroid, y-centroid,
            PSF width (x-axis), and PSF width (y-axis).
    """
    
    lbx, ubx, lby, uby = np.array(bounds)*scale
    starbox = image_data[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape
    # get centroid
    Y, X    = np.mgrid[:w,:l]
    cx      = np.nansum(X*starbox, axis=(1,2))/np.nansum(starbox, axis=(1,2)) + lbx
    cy      = np.nansum(Y*starbox, axis=(1,2))/np.nansum(starbox, axis=(1,2)) + lby
    try:
        cx      = sigma_clip(cx, sigma=4, maxiters=2, cenfunc=np.ma.median)
        cy      = sigma_clip(cy, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        cx      = sigma_clip(cx, sigma=4, iters=2, cenfunc=np.ma.median)
        cy      = sigma_clip(cy, sigma=4, iters=2, cenfunc=np.ma.median)
    xo = cx/scale
    yo = cy/scale
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    with np.errstate(invalid='ignore'):
        widx    = np.sqrt(np.nansum(X2*starbox, axis=(1,2))/(np.nansum(starbox, axis=(1,2))))
        widy    = np.sqrt(np.nansum(Y2*starbox, axis=(1,2))/(np.nansum(starbox, axis=(1,2))))
    try:
        widx    = sigma_clip(widx, sigma=4, maxiters=2, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        widx    = sigma_clip(widx, sigma=4, iters=2, cenfunc=np.ma.median)
        widy    = sigma_clip(widy, sigma=4, iters=2, cenfunc=np.ma.median)
    wx = widx/scale
    wy = widy/scale
    return xo, yo, wx, wy

def A_photometry(image_data, bg_err, 
                 cx = 15, cy = 15, r = 2.5, a = 5, b = 5, w_r = 5, h_r = 5,
                 theta = 0, scale = 1, shape = 'Circular', method='center'):
    """Performs aperture photometry, first by creating the aperture then summing the flux within the aperture.

    Args:
        image_data (3D array): Data cube of images (2D arrays of pixel values).
        bg_err (1D array): Array of uncertainties on pixel value.
        cx (int/array, optional): x-coordinate(s) of the center of the aperture. Default is 15.
        cy (int/array, optional): y-coordinate(s) of the center of the aperture. Default is 15.
        r (int, optional): If shape is 'Circular', r is the radius for the circular aperture. Default is 2.5.
        a (int, optional): If shape is 'Elliptical', a is the semi-major axis for elliptical aperture (x-axis). Default is 5.
        b (int, optional): If shape is 'Elliptical', b is the semi-major axis for elliptical aperture (y-axis). Default is 5.
        w_r (int, optional): If shape is 'Rectangular', w_r is the full width for rectangular aperture (x-axis). Default is 5.
        h_r (int, optional): If shape is 'Rectangular', h_r is the full height for rectangular aperture (y-axis). Default is 5.
        theta (int, optional): If shape is 'Elliptical' or 'Rectangular', theta is the angle of the rotation angle in radians
            of the semimajor axis from the positive x axis. The rotation angle increases counterclockwise. Default is 0.
        scale (int, optional): If the image is oversampled, scaling factor for centroid and bounds,
            i.e, give centroid in terms of the pixel value of the initial image.
        shape (string, optional): shape is the shape of the aperture. Possible aperture shapes are 'Circular',
            'Elliptical', 'Rectangular'. Default is 'Circular'.
        method (string, optional): The method used to determine the overlap of the aperture on the pixel grid. Possible 
            methods are 'exact', 'subpixel', 'center'. Default is 'center'.

    Returns:
        tuple: ape_sum (1D array) Array of flux with new flux appended.
            ape_sum_err (1D array) Array of flux uncertainties with new flux uncertainties appended.

    """
    
    # The following are all single values
    r, a, b, w_r, h_r = np.array([r, a, b, w_r, h_r])*scale
    
    # The following could be arrays
    cx *= scale
    cy *= scale
    
    l, h, w  = image_data.shape
    tmp_sum  = []
    tmp_err  = []
    movingCentroid = (isinstance(cx, Iterable) and isinstance(cy, Iterable))
    if not movingCentroid:
        position = [cx, cy]
        if   (shape == 'Circular'):
            aperture = CircularAperture(position, r=r)
        elif (shape == 'Elliptical'):
            aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
        elif (shape == 'Rectangular'):
            aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
    for i in range(l):
        if movingCentroid:
            position = [cx[i], cy[i]]
            if   (shape == 'Circular'):
                aperture = CircularAperture(position, r=r)
            elif (shape == 'Elliptical'):
                aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
            elif (shape == 'Rectangular'):
                aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
        data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
        phot_table = aperture_photometry(image_data[i,:,:], aperture, error=data_error, method=method)
        tmp_sum.extend(phot_table['aperture_sum'])
        tmp_err.extend(phot_table['aperture_sum_err'])
    # removing outliers
    try:
        ape_sum = sigma_clip(tmp_sum, sigma=4, maxiters=2, cenfunc=np.ma.median)
        ape_err = sigma_clip(tmp_err, sigma=4, maxiters=2, cenfunc=np.ma.median)
    except TypeError:
        ape_sum = sigma_clip(tmp_sum, sigma=4, iters=2, cenfunc=np.ma.median)
        ape_err = sigma_clip(tmp_err, sigma=4, iters=2, cenfunc=np.ma.median)
        
    return ape_sum, ape_err

def bin_all_data(flux, time, xo, yo, xw, yw, bg_flux, bg_err, npp, bin_size):
    binned_flux, binned_flux_std = bin_array(flux, bin_size)
    binned_time, binned_time_std = bin_array(time, bin_size)
    binned_xo, binned_xo_std     = bin_array(xo, bin_size)
    binned_yo, binned_yo_std     = bin_array(yo, bin_size)
    binned_xw, binned_xw_std     = bin_array(xw, bin_size)
    binned_yw, binned_yw_std     = bin_array(yw, bin_size)
    binned_bg, binned_bg_std     = bin_array(bg_flux, bin_size)
    binned_bg_err, binned_bg_err_std = bin_array(bg_err, bin_size)
    binned_npp, binned_npp_std     = bin_array(npp, bin_size)

    #sigma clip binned data to remove wildly unacceptable data
    try:
        binned_flux_mask = sigma_clip(binned_flux, sigma=5, maxiters=2)
    except TypeError:
        binned_flux_mask = sigma_clip(binned_flux, sigma=5, iters=2)
    if np.ma.is_masked(binned_flux_mask):
        mask_pos = binned_flux_mask!=binned_flux
        binned_time[mask_pos] = np.nan
        binned_time_std[mask_pos] = np.nan
        binned_xo[mask_pos] = np.nan
        binned_xo_std[mask_pos] = np.nan
        binned_yo[mask_pos] = np.nan
        binned_yo_std[mask_pos] = np.nan
        binned_xw[mask_pos] = np.nan
        binned_xw_std[mask_pos] = np.nan
        binned_yw[mask_pos] = np.nan
        binned_yw_std[mask_pos] = np.nan
        binned_bg[mask_pos] = np.nan
        binned_bg_std[mask_pos] = np.nan
        binned_bg_err[mask_pos] = np.nan
        binned_bg_err_std[mask_pos] = np.nan
        binned_npp[mask_pos] = np.nan
        binned_npp_std[mask_pos] = np.nan
        binned_flux_std[mask_pos] = np.nan
        binned_flux[mask_pos] = np.nan
        
    return (binned_flux, binned_flux_std, binned_time, binned_time_std,
            binned_xo, binned_xo_std, binned_yo, binned_yo_std,
            binned_xw, binned_xw_std, binned_yw, binned_yw_std,
            binned_bg, binned_bg_std, binned_bg_err, binned_bg_err_std,
            binned_npp, binned_npp_std)

def try_aperture(image_stack, time, xo, yo, xw, yw, bg_flux, bg_err,
                 shape='Circular', edge='exact', scale=1,
                 bin_data=True, bin_size=64, plot=False,
                 save=False, rerun_photometry=False, savepath='',
                 save_bin='', save_full='',
                 moveCentroid=True, channel='ch2', planet='',
                 r=2.5):
    
    edge_tmp = edge.lower()
    if edge_tmp=='hard' or edge_tmp=='center' or edge_tmp=='centre':
        method = 'center'
    elif edge_tmp=='soft' or edge_tmp=='subpixel':
        method = 'subpixel'
    elif edge_tmp=='exact':
        method = 'exact'
    else:
        # FIX: Throw an actual error
        print("Warning: No such method \""+edge+"\".",
              "Using hard edged aperture instead.")
        method = 'center'
    
    if save:
        if channel=='ch1':
            folder='3um'
        else:
            folder='4um'
        folder += edge+shape+"_".join(str(np.round(r, 2)).split('.'))
        if moveCentroid:
            folder += '_movingCentroid'
        folder += '/'

        # create save folder
        savepath = create_folder(savepath+folder, True, rerun_photometry)
        if savepath == None:
            # This photometry has already been run and shouldn't be rerun
            return
        print('Starting:', savepath)
    
    # aperture photometry
    if moveCentroid:
        flux, flux_err = A_photometry(image_stack, bg_err, 
                                      cx=xo, cy=yo,
                                      r=r, shape=shape, method=method, scale=scale)
    else:
        flux, flux_err = A_photometry(image_stack, bg_err,
                                      cx=15, cy=15,
                                      r=r, shape=shape, method=method, scale=scale)

    npp = noisepixparam(image_stack)
        
    order = np.argsort(time)
    flux = flux[order]
    flux_err = flux_err[order]
    time = time[order]
    xo = xo[order]
    yo = yo[order]
    xw = xw[order]
    yw = yw[order]
    bg_flux = bg_flux[order]
    bg_err = bg_err[order]
    npp = npp[order]
        
    if bin_data:
        (binned_flux, binned_flux_std, binned_time, binned_time_std,
         binned_xo, binned_xo_std, binned_yo, binned_yo_std,
         binned_xw, binned_xw_std, binned_yw, binned_yw_std,
         binned_bg, binned_bg_std, binned_bg_err, binned_bg_err_std,
         binned_npp, binned_npp_std) = bin_all_data(flux, time, xo, yo,
                                                    xw, yw, bg_flux, bg_err,
                                                    npp, bin_size)

    if plot:
        if bin_data:
            plotx = binned_time
            ploty0 = binned_flux
            ploty1 = binned_xo
            ploty2 = binned_yo
        else:
            plotx = time
            ploty0 = flux
            ploty1 = xo
            ploty2 = yo
            
        fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15,5))
        fig.suptitle(planet, fontsize="x-large")
        
        axes[0].plot(plotx, ploty0,'k+', color='black')
        axes[0].set_ylabel("Stellar Flux (electrons)")

        axes[1].plot(plotx, ploty1, '+', color='black')
        axes[1].set_ylabel("$x_0$")

        axes[2].plot(plotx, ploty2, 'r+', color='black')
        axes[2].set_xlabel("Time (BMJD))")
        axes[2].set_ylabel("$y_0$")
        fig.subplots_adjust(hspace=0)
        axes[2].ticklabel_format(useOffset=False)
        
        if save:
            pathplot = savepath + plot_name
            fig.savefig(pathplot)
        
        plt.show()
        plt.close()

    FULL_data = np.c_[flux, flux_err, time, xo, yo, xw, yw, bg_flux, bg_err, npp]

    if bin_data:
        BIN_data = np.c_[binned_flux, binned_flux_std,
                         binned_time, binned_time_std,
                         binned_xo, binned_xo_std,
                         binned_yo, binned_yo_std, binned_xw, binned_xw_std,
                         binned_yw, binned_yw_std,  binned_bg, binned_bg_std,
                         binned_bg_err, binned_bg_err_std]
    
    if save:
        FULL_head = 'Flux, Flux Uncertainty, Time, x-centroid, y-centroid, x-PSF width, y-PSF width, bg flux, bg flux err'
        FULL_head += ', Noise Pixel Parameter'
        pathFULL  = savepath + save_full
        np.savetxt(pathFULL, FULL_data, header = FULL_head)
        if bin_data:
            BIN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std'
            BIN_head += ', x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, bg flux, bg flux std'
            BIN_head += ', bg flux err, bg flux err std, Noise Pixel Parameter, Noise Pixel Parameter std'
            pathBIN  = savepath + save_bin
            np.savetxt(pathBIN, BIN_data, header = BIN_head)
    else:
        if bin_data:
            return FULL_data, BIN_data
        else:
            return FULL_data

def get_lightcurve(datapath, savepath, AOR_snip, channel,
                   save = True, save_full = '/ch2_datacube_full_AORs579.dat',
                   bin_data = True, bin_size = 64,
                   save_bin = '/ch2_datacube_binned_AORs579.dat',
                   plot = True, plot_name= 'Lightcurve.pdf',
                   oversamp = False, saveoversamp = True, reuse_oversamp = False,
                   planet = 'CoRoT-2b',
                   rs = [2.5], shapes=['Circular'], edges=['hard'],
                   addStack = False, stackPath = '', ignoreFrames = None,
                   maskStars = None, moveCentroids=[True], rerun_photometry=False,
                   ncpu=4):
    """Given a directory, looks for data (bcd.fits files), opens them and performs photometry.

    Args:
        datapath (string): Directory where the spitzer data is stored.
        savepath (string): Directory the outputs will be saved.
        AORsnip (string):  Common first characters of data directory eg. 'r579'
        channel (string): Channel used for the observation eg. 'ch1' for channel 1
        shapes (iterable, optional): The aperture shapes to try. Possible
            aperture shapes are 'Circular', 'Elliptical', 'Rectangular'.
            Default is just ['Circular'].
        edges (iterable, optional): The aperture edges to try. Options are 'hard',
            'soft', and 'exact' which correspond to the 'center', 'subpixel',
            and 'exact' methods in astropy. Default is just ['hard'].
        save (bool, optional): True if you want to save the outputs. Default is True.
        save_full (string, optional): Filename of the full unbinned output data. Default is '/ch2_datacube_full_AORs579.dat'.
        bin_data (bool, optional): True you want to get binned data. Default is True.
        bin_size (int, optional): If bin_data is True, the size of the bins. Default is 64.
        save_bin (string, optional): Filename of the full binned output data. Default is '/ch2_datacube_binned_AORs579.dat'.
        plot (bool, optional): True if you want to plot the time resolved lightcurve. Default is True.
        plot_name (string, optional): If plot and save is True, the filename of the plot to be saved as. Default is True.
        oversamp (bool, optional): True if you want to oversample the image by a factor of 2. Default is False.
        save_oversamp (bool, optional): True if you want to save oversampled images. Default is True.
        reuse_oversamp (bool, optional): True if you want to reuse oversampled images that were previously saved.
            Default is False.
        planet (string, optional): The name of the planet. Default is CoRoT-2b.
        rs (iterable, optional): The radii to try for aperture photometry in units of pixels. Default is just 2.5 pixels.
        ignoreFrames (list, optional) A list of frames to be masked when performing aperature photometry (e.g. first
            frame to remove first-frame systematic).
        maskStars (list, optional): An array-like object where each element is an array-like object with the RA and DEC
            coordinates of a nearby star which should be masked out when computing background subtraction.
        moveCentroids (iterable, optional): True if you want the centroid to be
            centered on the flux-weighted mean centroids (will default to median
            centroid when a NaN is returned), otherwise aperture will be centered
            on 15,15 (or 30,30 for 2x oversampled images). Default is [True].
        rerun_photometry (bool, optional): Whether to overwrite old photometry if it exists. Default is False.
        ncpu (int, optional): The number of aperture radii to try at the same time with multiprocessing. Default is 4.

    Raises: 
        Error: If Photometry method is not supported/recognized by this pipeline.
    
    """
    
    if not isinstance(rs, Iterable):
        rs = [rs]
    if not isinstance(shapes, Iterable):
        shapes = [shapes]
    if not isinstance(edges, Iterable):
        edges = [edges]
    if not isinstance(moveCentroids, Iterable):
        moveCentroids = [moveCentroids]
    
    for i in range(len(shapes)):
        shape = shapes[i]
        if shape!='Circular' and shape!='Elliptical' and shape!='Rectangular':
            # FIX: Throw an actual error
            print('Warning: No such aperture shape "'+shape+'". Using Circular aperture instead.')
            shapes[i]='Circular'
    
    if save and savepath[-1]!='/':
        savepath += '/'
        
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
    
    # oversampling
    if oversamp:
        if reuse_oversamp:
            savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
            if os.path.isfile(savename):
                image_stack = np.load(savename)
            else:
                print('Warning: Oversampled images were not previously saved! Making new ones now...')
                image_stack = np.ma.masked_invalid(oversampling(image_stack))
                if (saveoversamp == True):
                    # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                    image_stack.dump(savename)
        else:
            image_stack = np.ma.masked_invalid(oversampling(image_stack))

        if saveoversamp:
            # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
            savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
            image_stack.dump(savename)

        scale = 2
    else:
        scale = 1
        
    # get centroids & PSF width
    xo, yo, xw, yw = centroid_FWM(image_stack, scale=scale)
    xo[np.ma.getmaskarray(xo)] = np.ma.median(xo)
    yo[np.ma.getmaskarray(yo)] = np.ma.median(yo)
    
    # perform aperture photometry
    for moveCentroid in moveCentroids:
        for edge in edges:
            for shape in shapes:
                with Pool(ncpu) as pool:
                    func = partial(try_aperture, image_stack, time, xo, yo,
                                   xw, yw, bg_flux, bg_err,
                                   shape, edge, scale,
                                   bin_data, bin_size, plot,
                                   save, rerun_photometry, savepath, save_bin,
                                   save_full, moveCentroid, 
                                   channel, planet)
                    results = pool.map(func, rs)

    return results




import unittest

class TestAperturehotometryMethods(unittest.TestCase):

    # Test that centroiding gives the expected values and doesn't swap x and y
    def test_FWM_centroiding(self):
        fake_images = np.zeros((4,32,32))
        for i in range(fake_images.shape[0]):
            fake_images[i,14+i,15] = 2
        xo, yo, _, _ = centroid_FWM(fake_images)
        self.assertTrue(np.all(xo==np.ones_like(xo)*15.))
        self.assertTrue(np.all(yo==np.arange(14,18)))

    # Test that circular aperture photometry properly follows the input centroids and gives the expected values
    def test_circularAperture(self):
        fake_images = np.zeros((4,32,32))
        for i in range(fake_images.shape[0]):
            fake_images[i,14+i,15] = 2
        xo = np.ones(fake_images.shape[0])*15
        yo = np.arange(14,18)
        flux, _ = A_photometry(fake_images, np.zeros_like(xo), cx=xo, cy=yo, r=1.,
                               shape='Circular', method='center')
        self.assertTrue(np.all(flux==np.ones_like(flux)*2.))

if __name__ == '__main__':
    unittest.main()
