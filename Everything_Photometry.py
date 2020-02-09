#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from functools import partial

# SPCA libraries
from SPCA import frameDiagnosticsBackend
from SPCA import photometryBackend

# The number of CPU threads you want to use for running photometry methods in parallel
ncpu = 7

# The names of all the planets you want analyzed (without spaces)
planets = ['CoRoT-2b', 'HAT-P-7b', 'HAT-P-7b', 'HD149026b', 'HD149026b', 'KELT-16b', 'KELT-9b', 'MASCARA-1b', 'Qatar1b', 'Qatar1b', 'WASP-14b', 'WASP-14b', 'WASP-18b', 'WASP-18b', 'WASP-19b', 'WASP-19b', 'WASP-33b', 'WASP-33b', 'WASP-43b', 'WASP-43b']
# WASP-103b is full-frame, 55Cnce is going to be tough

#folder containing data from each planet
basepath = '/homes/picaro/bellt/research/'

#####################################################################
# Parameters to set how you want your photometry done
#####################################################################

#################
# General settings
#################

# Whether or not to bin data points together
bin_data = True
# Number of data points to bin together
bin_size = 64

# Do you want the frame diagnostics to automatically remove certain frames within a data cube?
allowIgnoreFrames = [True, False]

# The number of sigma a frame's median value must be off from the median frame in order to be added to ignore_frames
nsigma = 4

# Was the data collected in subarray mode (full frame photometry is currently not supported)
subarray = True

# An array-like object where each element is an array-like object with the RA and DEC coordinates of a nearby star which should be masked out when computing background subtraction.
maskStars = None

# Whether to use Aperture photometry or PLD photometry (PSF photometry currently not supported)
photometryMethods = ['Aperture', 'PLD']

#################
# The purpose of this stack is to remove artifacts from bad background subtraction from the Spitzer data pipeline
# If set to True, the following option requires contacting someone from IPAC in order to get your correction stack.
# In general, this stack doesn't significantly impact the results, so we have not implemented this ourselves.
#################
# If True, the correction stack must be in the folder basepath+'/Calibration'
# The correction stack must also have the name format 'correction_stack_for_BCDs_that_used_SPITZER_I#_SDARK#'
#    where # should be replaced with the channel number, and SDARK# should be replaced with a string that 
#    specifies which sdark file was used. This can be found within the cal directory if you downloaded that
#    from SHA.
addStacks = [False]

#################
# Settings for aperture photometry
# Can leave untouched if not using aperture photometry; these settings would be ignored
#################

# Aperture radii to try
radii = np.linspace(2.,6.,21,endpoint=True)

# Aperture shape to try. Possible aperture shapes are 'Circular', 'Elliptical', 'Rectangular'
shape = 'Circular'

# Aperture edges to try. Possible options are 'Exact' (pixels are weighted by the fraction that lies within the aperture), 'Hard' (pixel is only included if its centre is in the aperture), and 'Soft' (approximates exact)
edges = ['Exact', 'Hard']

# Whether or not to keep the aperture centred at the centroid (otherwise keeps centred at the middle of the subarray)
moveCentroids = [False, True]

# How wide should the boxcar filter be that smooths the raw data to select the best aperture
highpassWidth = 5

#################
# Settings for PLD
# Can leave untouched if not using PLD; these settings would be ignored
#################

# Size of PLD stamp to use (only 3 and 5 currently supported)
stamp_sizes = [3, 5]

#################
# Settings for photometry comparisons
#################

# Trim data between some start and end point (good for bad starts or ends to data)
trim = False
trimStart = 5.554285e4
trimEnd = 5.5544266e4



#####################################################################
# Everything below is automated
#####################################################################

for planet in planets:
    
    #bit of AOR to pick out which folders contain AORs that should be analyzed
    with open(basepath+planet+'/analysis/aorSnippet.txt', 'r') as file:
        AOR_snip = file.readline().strip()
    
    channels = [name for name in os.listdir(basepath+planet+'/data/') if os.path.isdir(basepath+planet+'/data/'+name) and 'ch' in name]

    for channel in channels:
        print('Starting planet', planet, 'channel', channel)
        
        minRMSs = []
        phoptions = []

        for addStack in addStacks:
            if True in allowIgnoreFrames:
                # Perform frame diagnostics to figure out which frames within a datacube are consistently bad
                print('Analysing', channel, 'for systematically bad frames...')
                ignoreFrames = frameDiagnosticsBackend.run_diagnostics(planet, channel, AOR_snip,
                                                                       basepath, addStack, nsigma)
            else:
                ignoreFrames = []

            for allowIgnoreFrame in np.sort(allowIgnoreFrames)[::-1]:
                if allowIgnoreFrame:
                    print('Using ignoreFrames')
                    ignoreFrames_temp = ignoreFrames
                else:
                    print('Overwriting ignoreFrames to []')
                    ignoreFrames_temp = []

                # Try all of the different photometry methods
                print('Trying the many different photometries...')
                for photometryMethod in photometryMethods:
                    if photometryMethod=='PLD':
                        for stamp_size in stamp_sizes:
                            photometryBackend.run_photometry(photometryMethod, basepath, planet, channel, subarray,
                                                             AOR_snip, addStack, bin_data, bin_size, ignoreFrames_temp,
                                                             maskStars, stamp_size)
                    elif photometryMethod=='Aperture':
                        with multiprocessing.Pool(ncpu) as pool:
                            for moveCentroid in moveCentroids:
                                for edge in edges:
                                    func = partial(photometryBackend.run_photometry, photometryMethod, basepath, planet,
                                                   channel, subarray, AOR_snip,
                                                   addStack, bin_data, bin_size, ignoreFrames_temp, maskStars,
                                                   stamp_size, shape, edge, moveCentroid)
                                    pool.map(func, radii)

                        print('Selecting the best aperture photometry method from this suite...')
                        minRMS, phoption = photometryBackend.comparePhotometry(basepath, planet, channel, AOR_snip,
                                                                               ignoreFrames_temp, addStack, highpassWidth,
                                                                               trim, trimStart, trimEnd)

                        minRMSs.append(minRMS)
                        phoptions.append(phoption)

        if photometryMethods != ['PLD']:
            bestPhOption = phoptions[np.argmin(minRMSs)]

            if (False in allowIgnoreFrames) and not np.sort(allowIgnoreFrames)[::-1][np.argmin(minRMSs)]:
                # Best photometry didn't use ignoreFrames
                ignoreFrames = []

            print('The best overall aperture photometry method is:')
            print(bestPhOption)
            print('With an RMS of:')
            print(str(np.round(np.min(minRMSs)*1e6,1)))

            with open(basepath+planet+'/analysis/'+channel+'/bestPhOption.txt', 'a') as file:
                file.write(bestPhOption+'\n')
                file.write('IgnoreFrames = '+str(ignoreFrames)[1:-1]+'\n')
                file.write(str(np.round(np.min(minRMSs)*1e6,1))+'\n\n')

        if 'PLD' in photometryMethods:
            with open(basepath+planet+'/analysis/'+channel+'/PLD_ignoreFrames.txt', 'a') as file:
                file.write('IgnoreFrames = '+str(ignoreFrames)[1:-1]+'\n')

print('Done!')          
