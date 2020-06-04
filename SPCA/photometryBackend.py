import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, sys
from astropy.io import fits
from astropy.stats import sigma_clip

# SPCA libraries
from . import Photometry_Aperture as APhotometry
from . import Photometry_PSF as PSFPhotometry
from . import Photometry_Companion as CPhotometry
from . import Photometry_PLD as PLDPhotometry


def run_photometry(photometryMethod, basepath, ncpu, planet, channel, AOR_snip, rerun_photometry=False,
                   addStack=False, bin_data=True, bin_size=64, ignoreFrames=None, maskStars=None,
                   stamp_size=3, shapes=['Circular'], edges=['Hard', 'Exact'], moveCentroids=[True], radii=[3],
                   onlyBest=True, highpassWidth=5*64, oversamp=False):
    
    if ignoreFrames is None:
        ignoreFrames = []
    if maskStars is None:
        maskStars = []

    # Call requested function
    if   (photometryMethod == 'Aperture'):
        APhotometry.get_lightcurve(basepath, AOR_snip, channel, planet,
                                   True, onlyBest, highpassWidth, bin_data, bin_size,
                                   False, True, oversamp, True, True,
                                   radii, edges, addStack, ignoreFrames,
                                   maskStars, moveCentroids, ncpu)
    elif (photometryMethod == 'PSFfit'):
        PSFPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel)
    elif (photometryMethod == 'Companion'):
        CPhotometry.get_lightcurve(datapath, savepath, AOR_snip, channel,
                                   r = radius)
    elif (photometryMethod == 'PLD'):
        PLDPhotometry.get_pixel_lightcurve(datapath, savepath, AOR_snip, channel,
                                           save=True, save_full=save_full,
                                           bin_data=bin_data, bin_size=bin_size,
                                           save_bin=save_bin,
                                           plot=False, plot_name='', planet=planet,
                                           stamp_size=stamp_size, 
                                           addStack=addStack, stackPath=stackPath,
                                           ignoreFrames=ignoreFrames, 
                                           maskStars=maskStars, rerun_photometry=rerun_photometry)
    else:
        print('Sorry,', photometryMethod, 'is not supported by this pipeline!')
        
    return
