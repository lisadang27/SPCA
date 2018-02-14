#--------------------------------------------------------------
#Author: Lisa Dang
#Created: 2016-11-09 1:21 AM EST
#Last Modified: 
#Title: Aperture Photometry
#--------------------------------------------------------------
import numpy as np
from scipy import interpolate

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches

from astropy.io import fits
from astropy.stats import sigma_clip

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
	fnames   = []
	for i in range(len(lst)):
		path = directory + '/' + lst[i] 
		fnames.extend([path])
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
	t       = np.linspace(hdu_list[0].header['AINTBEG'] + hdu_list[0].header['EXPTIME']/2, hdu_list[0].header['ATIMEEND'] - hdu_list[0].header['EXPTIME']/2, h)
	sec2day = 1.0/(3600.0*24.0)
	t       = sec2day*t
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
	sig_clipped_data = sigma_clip(image_data2, sigma=4, iters=2, cenfunc=np.ma.median, axis = 0)
	for i in range (h):
		oldstar = image_data[i, lbx:ubx, lby:uby]
		newstar = sig_clipped_data[i, lbx:ubx, lby:uby]
		truth   = newstar==oldstar
		if(truth.sum() < truth.size):
			sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
			badframetable.append([i,filenb,fname])
			tossed += 1
			print(tossed)
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
	return np.ma.masked_invalid(bgsub_data), bg_err


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
	xo.extend(cx/scale)
	yo.extend(cy/scale)
	# get PSF widths
	X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
	cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
	X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
	widx    = np.sqrt(np.sum(np.sum(X2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
	widy    = np.sqrt(np.sum(np.sum(Y2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
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
    	methods are 'exact', 'subpixel', 'center'. Default is 'exact'.

    Returns
    -------
    ape_sum  : 1D array
    	Array of flux with new flux appended.

    ape_sum_err: 1D array
    	Array of flux uncertainties with new flux uncertainties appended.

	'''
	l, h, w = image_data.shape
	position=[cx, cy]
	if   (shape == 'Circular'):
		aperture = CircularAperture(position, r=r)
	elif (shape == 'Elliptical'):
		aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
	elif (shape == 'Rectangular'):
		aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
	for i in range(l):
		data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
		phot_table = aperture_photometry(image_data[i,:,:],aperture, error=data_error, pixelwise_error=False)
		ape_sum.extend(phot_table['aperture_sum']*factor)
		ape_sum_err.extend(phot_table['aperture_sum_err']*factor)
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
	reshaped_data   = data.reshape((len(data)/size, size))
	binned_data     = np.ma.median(reshaped_data, axis=1)
	binned_data_std = np.std(reshaped_data, axis=1)
	return binned_data, binned_data_std


def get_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
	save = True, save_full = '/ch2_datacube_full_AORs579.dat', bin_data = True, 
	bin_size = 64, save_bin = '/ch2_datacube_binned_AORs579.dat', plot = True, 
	plot_name= 'CoRoT-2b.pdf', oversamp = True, saveover = True, **kwargs):

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
	fnames= get_fnames(datapath, AOR_snip, channel)

	# get pre-computed variables
	pre_path = 'C:/Users/Lisa/Documents/Exoplanets/High_Precision_Photometry/Run13/ch2_datacube_full_AORs579.dat'

	time     = np.loadtxt(pre_path, usecols=[2], skiprows=1)
	bg_flux  = np.loadtxt(pre_path, usecols=[11], skiprows=1)
	bg_err   = np.loadtxt(pre_path, usecols=[12], skiprows=1)

	# variables declaration 
	percent       = 0                                # to show progress while running the code
	tossed        = 0                                # Keep tracks of number of frame discarded 
	badframetable = []                               # list of filenames of the discarded frames
	xo            = []                               # centroid value along the x-axis
	yo            = []                               # centroid value along the y-axis
	xw            = []                               # PSF width along the x-axis
	yw            = []                               # PSF width along the y-axis
	aperture_sum  = []                               # flux obtained from aperture photometry
	aperture_sum_err = []                            # error on flux obtained from aperture photometry
	xo2           = []                               # centroid value along the x-axis
	yo2           = []                               # centroid value along the y-axis
	xw2           = []                               # PSF width along the x-axis
	yw2           = []                               # PSF width along the y-axis

	#image_data_full=np.zeros((64*nfiles, 32, 32))
	#factor = np.zeros(64*nfiles)

	# data reduction & aperture photometry part
	if (subarray == True):
		for i in range(len(fnames)):
			# open fits file
			print('this')
			print(fnames[i])
			image_data3 = np.load(fnames[i])
			# oversampling
			#if (oversamp == True):
			#	image_data3 = np.ma.masked_invalid(oversampling(image_data3))
			#	if (saveover == True):
			#		savename = fnames[i]
			#		new_name = savepath + '/Oversampled/' + savename[57:86]
			#		image_data3.dump(new_name)
			# Aperture Photometry
			if (oversamp == True):
				# get centroids & PSF width
				xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw, scale = 2)
				# propersccaling
				#xo, yo, xw, yw = xo-0.25, yo-0.25, xw-0.25, yw-0.25
				# get centroids & PSF width for companion
				xo2, yo2, xw2, yw2 = centroid_FWM(image_data3, xo2, yo2, xw2, yw2, bounds = (12,15,11,14), scale = 2)
				#xo2, yo2, xw2, yw2 = xo2-0.25, yo2-0.25, xw2-0.25, yw2-0.25
				# convert electron count to Mjy/str
				ecnt2Mjy = 0.029691810510039204
				# aperture photometry
				aperture_sum, aperture_sum_err = A_photometry(image_data3, bg_err[-64:], ecnt2Mjy, aperture_sum, aperture_sum_err, cx = 30.5, cy = 30.5, a = 2*5, b = 2*5, w_r = 2*5, h_r = 2*5, **kwargs)
			else :
				# get centroids & PSF width
				xo, yo, xw, yw = centroid_FWM(image_data3, xo, yo, xw, yw)
				# get centroids & PSF width for companion
				xo2, yo2, xw2, yw2 = centroid_FWM(image_data3, xo2, yo2, xw2, yw2, bounds = (12, 15, 11, 14))
				# convert electron count to Mjy/str
				ecnt2Mjy = 0.029691810510039204 
				# aperture photometry
				aperture_sum, aperture_sum_err = A_photometry(image_data3, bg_err[-64:], ecnt2Mjy, aperture_sum, aperture_sum_err, **kwargs)
			print('Status:', i, 'out of', len(fnames))

		print(np.shape(aperture_sum))
		print(np.shape(aperture_sum_err))
		print(np.shape(time))
		print(np.shape(xo))
		print(np.shape(yo))
		print(np.shape(xw))
		print(np.shape(yw))
		print(np.shape(xo2))
		print(np.shape(yo2))
		print(np.shape(xw2))
		print(np.shape(yw2))

	elif (subarray == False):
		print('Sorry this part is undercontruction!')

	if (bin_data == True):
		binned_flux, binned_flux_std = binning_data(np.asarray(aperture_sum), bin_size)
		binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)
		binned_xo, binned_xo_std     = binning_data(np.asarray(xo)-0.25, bin_size)
		binned_yo, binned_yo_std     = binning_data(np.asarray(yo)-0.25, bin_size)
		binned_xw, binned_xw_std     = binning_data(np.asarray(xw)-0.25, bin_size)
		binned_yw, binned_yw_std     = binning_data(np.asarray(yw)-0.25, bin_size)
		binned_xo2, binned_xo2_std   = binning_data(np.asarray(xo2)-0.25, bin_size)
		binned_yo2, binned_yo2_std   = binning_data(np.asarray(yo2)-0.25, bin_size)
		binned_xw2, binned_xw2_std   = binning_data(np.asarray(xw2)-0.25, bin_size)
		binned_yw2, binned_yw2_std   = binning_data(np.asarray(yw2)-0.25, bin_size)

	if (plot == True):
		fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(15,5))
		fig.suptitle("CoRoT-2b", fontsize="x-large")
		axes[0].plot(binned_time, binned_flux,'k+')
		axes[0].set_ylabel("Stellar Flux (MJy/pixel)")

		axes[1].plot(binned_time, binned_xo, '+')
		#axes[1].plot(binned_time, binned_xo2, '+')
		axes[1].set_ylabel("$x_0$")

		axes[2].plot(binned_time, binned_yo, 'r+')
		#axes[2].plot(binned_time, binned_yo2, 'r+')
		axes[2].set_xlabel("Time since IRAC turn-on (days)")
		axes[2].set_ylabel("$y_0$")
		fig.subplots_adjust(hspace=0)
		if (save == True):
			pathplot = savepath + '/' + plot_name
			fig.savefig(pathplot)
		else :
			plt.show()

	if (save == True):
		FULL_data = np.c_[aperture_sum, aperture_sum_err, time, xo, yo, xw, yw, xo2, yo2, xw2, yw2]
		FULL_head = 'Flux, Flux Uncertainty, Time, x-centroid, y-centroid, x-PSF width, y-PSF width'
		BINN_data = np.c_[binned_flux, binned_flux_std, binned_time, binned_time_std, binned_xo, binned_xo_std, binned_yo, binned_yo_std, binned_xw, binned_xw_std, binned_yw, binned_yw_std, binned_xo2, binned_xo2_std, binned_yo2, binned_yo2_std, binned_xw2, binned_xw2_std, binned_yw2, binned_yw2_std]
		BINN_head = 'Flux, Flux std, Time, Time std, x-centroid, x-centroid std, y-centroid, y-centroid std, x-PSF width, x-PSF width std, y-PSF width, y-PSF width std, x2-centroid, x2-centroid std, y2-centroid, y2-centroid std, x2-PSF width, x2-PSF width std, y2-PSF width, y2-PSF width std'
		pathFULL  = savepath + save_full
		pathBINN  = savepath + save_bin
		np.savetxt(pathFULL, FULL_data, header = FULL_head)
		np.savetxt(pathBINN, BINN_data, header = BINN_head)

	toc = tim.clock()
	print('Number of discarded frames:', tossed)
	print('Time:', toc-tic, 'seconds')

if __name__=='__main__': main()