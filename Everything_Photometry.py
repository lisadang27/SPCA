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

# Do you want to add a correction stack to fix bad backgrounds
addStacks = [False]

# Do you want the frame diagnostics to automatically remove certain frames within a data cube?
allowIgnoreFrames = [True, False]

# How bad does a value need to be before you clip it
nsigma = 3

# Was the data collected in subarray mode (currently only subarray data can be used)
subarray = True

# An array-like object where each element is an array-like object with the RA and DEC coordinates of a nearby star which should be masked out when computing background subtraction.
maskStars = None

# Whether to use Aperture photometry or PLD (currently only aperture photometry can be used)
photometryMethod = 'Aperture'

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
        minRMSs = []
        phoptions = []

        for addStack in addStacks:
            # Perform frame diagnostics to figure out which frames within a datacube are consistently bad
            print('Analysing', channel, 'for systematically bad frames...')
            ignoreFrames = frameDiagnosticsBackend.run_diagnostics(planet, channel, AOR_snip, basepath, addStack, nsigma)

            for allowIgnoreFrame in np.sort(allowIgnoreFrames)[::-1]:
                if allowIgnoreFrame:
                    print('Using ignoreFrames')
                else:
                    print('Overwriting ignoreFrames to []')
                    ignoreFrames = []

                # Try all of the different photometry methods
                print('Trying the many different photometries...')
                pool = multiprocessing.Pool(ncpu)
                for moveCentroid in moveCentroids:
                    for edge in edges:
                        func = partial(photometryBackend.run_photometry, basepath, addStack, planet, channel, subarray, AOR_snip, ignoreFrames, maskStars, photometryMethod, shape, edge, moveCentroid)
                        pool.map(func, radii)
                pool.close()

                print('Selecting the best photometry method...')
                minRMS, phoption = photometryBackend.comparePhotometry(basepath, planet, channel, AOR_snip, ignoreFrames, addStack, highpassWidth, trim, trimStart, trimEnd)

                minRMSs.append(minRMS)
                phoptions.append(phoption)

        bestPhOption = phoptions[np.argmin(minRMSs)]

        print('The best overall photometry method is:')
        print(bestPhOption)
        print('With an RMS of:')
        print(str(np.round(np.min(minRMSs)*1e6,1)))

        with open(basepath+planet+'/analysis/'+channel+'/bestPhOption.txt', 'a') as file:
            file.write(bestPhOption+'\n')
            file.write('IgnoreFrames = '+str(ignoreFrames)[1:-1]+'\n')
            file.write(str(np.round(np.min(minRMSs)*1e6,1))+'\n\n')            
