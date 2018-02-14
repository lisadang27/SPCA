import numpy as np

import matplotlib.pyplot as plt

import os, sys

from astropy.convolution import convolve, Box1DKernel, Gaussian1DKernel



def get_binned_data(foldername, snip):
	'''
	Retrieve data

	Parameters
	----------

	foldername : string object
		Path to the directory containing all the Spitzer data.

	snip  : string object
		Common first characters of data directory eg. 'r579'

	Returns
	-------

	flux            : 1D array
		Flux extracted for each frame

	time             : 1D array
		Time stamp for each frame

	xdata            : 1D array
		X-coordinate of the centroid for each frame

	ydata            : 1D array
		Y-coordinate of the centroid for each frame   

	psfwx            : 1D array
		X-width of the target's PSF for each frame     

	psfwy            : 1D array
	Y-width of the target's PSF for each frame     

	'''
	path = foldername + '/' + snip
	flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # Flux from circular aperture (MJy/str)
	time     = np.loadtxt(path, usecols=[2], skiprows=1)     # time in days?
	xdata    = np.loadtxt(path, usecols=[3], skiprows=1)     # x-centroid (15 = center of 15th pixel)
	ydata    = np.loadtxt(path, usecols=[4], skiprows=1)     # y-centroid (15 = center of 15th pixel)
	psfwx    = np.loadtxt(path, usecols=[5], skiprows=1)     # psf width in pixel size (FWHM of 2D Gaussian)
	psfwy    = np.loadtxt(path, usecols=[6], skiprows=1)     # psf width in pixel size (FWHM of 2D Gaussian)    
	return flux, time, xdata, ydata, psfwx, psfwy


def get_fnames(directory):
	'''
	Find paths to all the fits files.

	Parameters
	----------

	directory : string object
		Path to the directory containing all the Spitzer data.

	Returns
	-------

	fname     : list
		List of paths to all bcd.fits files.

	len(fnames): int
		Number of fits file found.
	'''
	lst      = os.listdir(directory)
	Run_list = [k for k in lst if 'Run' in k]
	return Run_list, len(Run_list)

def highpassflist(signal):
	'''
	Smoothes out the lightcurve using a 1D box kernel

	Parameters
	----------

	signal : 1D array
		signal to be smoothed out.
	Returns
	-------

	smooth : 1D array
		smoothed out signal.
	'''
	#g = Gaussian1DKernel(stddev=10)
	g = Box1DKernel(10)
	smooth = convolve(np.asarray(signal), g,boundary='extend')
	return smooth

def RMS(x):
	'''
	Smoothes out the lightcurve using a 1D box kernel

	Parameters
	----------

	x      : 1D array
		signal to use to calculate RMs.

	Returns
	-------
	x_RMS  : float
		RMS value.
	'''
	n=len(x)
	x2 = np.multiply(x,x)
	sumx2 =np.sum(x2)
	x_RMS = np.sqrt((1.0/n)*sumx2)
	return x_RMS