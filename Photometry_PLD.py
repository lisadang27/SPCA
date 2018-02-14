#--------------------------------------------------------------
#Author: Lisa Dang
#Created: 2016-10-13 1:29 PM EST
#Last Modified: 2016 - 11 - 08 EST
#Title: Get Pixel Light Curve for Pixel-Level-Decorrelation
#--------------------------------------------------------------

import numpy as np
from scipy import interpolate

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
	AOR_list = [k for k in lst if AOR_snip in k]                # used to ignore calibration data sets
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

    :return: time (1D array) - Updated time stamp array in days

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

def binning_data2D(data, size):
	data = np.ma.masked_invalid(data)
	h, w = data.shape
	reshaped_data   = data.reshape((h/size, size, w))
	binned_data     = np.ma.median(reshaped_data, axis=1)
	binned_data_std = np.ma.std(reshaped_data, axis=1)
	return binned_data, binned_data_std

def get_pixel_values(image_data0, P, nbx = 3, nby = 3):
	image_data = np.ma.masked_invalid(image_data0)
	#if (n % 2):
	h, w, l = image_data.shape
	P_tmp = np.empty(shape=(h, nbx*nby))
	for i in range(h):
		xmax, ymax = 15, 15
		#(xmax, ymax) = np.unravel_index(image_data[i,:,:].argmax(), image_data[i,:,:].shape)
		#print(xmax,ymax)
		P_tmp[i,:]   = np.array([image_data[i, xmax-1,ymax-1], image_data[i, xmax-1,ymax], image_data[i, xmax-1,ymax+1],
			image_data[i,xmax,ymax-1], image_data[i,xmax,ymax], image_data[i, xmax,ymax+1],
			image_data[i,xmax+1,ymax-1], image_data[i,xmax+1,ymax], image_data[i,xmax+1,ymax+1]])
	P = np.append(P, P_tmp, axis = 0)
	return P

def get_pixel_lightcurve(datapath, savepath, AOR_snip, channel, subarray,
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
	fnames= get_fnames(datapath, AOR_snip, channel)

		# variables declaration 
	percent       = 0                                # to show progress while running the code
	tossed        = 0                                # Keep tracks of number of frame discarded 
	badframetable = []                               # list of filenames of the discarded frames
	time          = []                               # time array
	bg_err        = []                               # background flux error 
	P             = np.empty(shape = (0,9))

	if (subarray == True):
		for i in range(len(fnames)):
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
			image_data2, tossed, badframetable = sigma_clipping(image_data1, i ,fname[fname.find('ch2/bcd/')+8:], **kwargs, tossed=tossed)
			# bg subtract
			image_data3, bg_err = bgsubtract(image_data2, bg_err)	
			# get pixel peak index
			P = get_pixel_values(image_data3, P)
			# reconvert to MJy/str
		ecnt2Mjy = - hdu_list[0].header['PXSCAL1']*hdu_list[0].header['PXSCAL2']*(1/convfact)
		P = ecnt2Mjy*P

		print('P shape', P.shape)

	elif (subarray == False):
		print('Sorry this part is undercontruction!')
	print(P.shape)
	if (bin_data == True):
		binned_P, binned_P_std = binning_data2D(P, bin_size)
		binned_time, binned_time_std = binning_data(np.asarray(time), bin_size)

	if (plot == True):
		fig, axes = plt.subplots(nrows = 3, ncols = 1, sharex = True, figsize=(15,10))
		fig.suptitle("CoRoT-2b", fontsize="x-large")
		axes[0].plot(binned_time, binned_P[:,0], '+', label = '$P_1$')
		axes[0].plot(binned_time, binned_P[:,1], '+', label = '$P_2$')
		axes[0].plot(binned_time, binned_P[:,2], '+', label = '$P_3$')
		axes[0].plot(binned_time, binned_P[:,3], '+', label = '$P_4$')
		axes[0].plot(binned_time, binned_P[:,4], '+', label = '$P_5$')
		axes[0].plot(binned_time, binned_P[:,5], '+', label = '$P_6$')
		axes[0].plot(binned_time, binned_P[:,6], '+', label = '$P_7$')
		axes[0].plot(binned_time, binned_P[:,7], '+', label = '$P_8$')
		axes[0].plot(binned_time, binned_P[:,8], '+', label = '$P_9$')
		axes[0].set_ylabel("Pixel Flux (MJy/pixel)")
		axes[0].legend()
		axes[1].set_ylabel('Sum Flux (MJy/pixel)')
		axes[1].plot(binned_time, np.sum(binned_P, axis = 1), '+')
		axes[2].plot(binned_time, binned_P_std[:,0], '+', label = '$Pstd_1$')
		axes[2].plot(binned_time, binned_P_std[:,1], '+', label = '$Pstd_2$')
		axes[2].plot(binned_time, binned_P_std[:,2], '+', label = '$Pstd_3$')
		axes[2].plot(binned_time, binned_P_std[:,3], '+', label = '$Pstd_4$')
		axes[2].plot(binned_time, binned_P_std[:,4], '+', label = '$Pstd_5$')
		axes[2].plot(binned_time, binned_P_std[:,5], '+', label = '$Pstd_6$')
		axes[2].plot(binned_time, binned_P_std[:,6], '+', label = '$Pstd_7$')
		axes[2].plot(binned_time, binned_P_std[:,7], '+', label = '$Pstd_8$')
		axes[2].plot(binned_time, binned_P_std[:,8], '+', label = '$Pstd_9$')
		axes[2].set_xlabel("Time since IRAC turn-on (days)")
		fig.subplots_adjust(hspace=0)

		if (save == True):
			pathplot = savepath + '/' + plot_name
			fig.savefig(pathplot)
			plt.show()

		else :
			plt.show()

	if (save == True):
		FULL_data = np.c_[P, time, bg_err]
		FULL_head = 'P1, P2, P3, P4, P5, P6, P7, P8, P9, time, bg_err'
		BINN_data = np.c_[binned_P, binned_P_std, binned_time, binned_time_std]
		BINN_head = 'P1, P2, P3, P4, P5, P6, P7, P8, P9, P1_std, P2_std, P3_std, P4_std, P5_std, P6_std, P7_std, P8_std, P9_std, time, time_std,'
		pathFULL  = savepath + save_full
		pathBINN  = savepath + save_bin
		np.savetxt(pathFULL, FULL_data, header = FULL_head)
		np.savetxt(pathBINN, BINN_data, header = BINN_head)

	toc = tim.clock()
	print('Number of discarded frames:', tossed)
	print('Time:', toc-tic, 'seconds')

if __name__=='__main__': main()

				

