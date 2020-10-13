import numpy as np
import scipy.optimize as spopt

import matplotlib.pyplot as plt

from astropy.stats import sigma_clip

import inspect
from functools import partial

import os, sys
lib_path = os.path.abspath(os.path.join('../'))
sys.path.append(lib_path)

from copy import deepcopy

# SPCA libraries
import SPCA
from SPCA import astro_models, detec_models, bliss

# FIX: Add a docstring for this function
def signal_params():
    
    p0_obj = {'name': 'planet', 't0': 0.0, 't0_err': 0.0, 'per': 1.0, 'per_err': 0.0,
              'rp': 0.1, 'rp_err': 0.0, 'a': 8.0, 'a_err': 0.0, 'inc': 90.0, 'inc_err': 0.0, 'ecosw': 0.0, 'ecosw_err': 0.0,
              'esinw': 0.0, 'esinw_err': 0.0, 'q1': 0.01, 'q2': 0.01, 'fp': 0.003, 'fp_err': 0.0, 'A': 0.35, 'B': 0.0,
              'C': 0.0, 'D': 0.0, 'r2': None, 'r2off': 0.0, 'c1': 1.0}
    
    p0_obj.update(dict([['c'+str(i), 0.0] for i in range(2,22)]))
    p0_obj.update(dict([['p1_1', 1.0] for i in range(1,10)]))
    p0_obj.update(dict([['p'+str(i)+'_1', 0.03] for i in range(2,10)]))
    p0_obj.update(dict([['p'+str(i)+'_1', 0.01] for i in range(10,26)]))
    p0_obj.update(dict([['p'+str(i)+'_2', 0.01] for i in range(1,26)]))
    p0_obj.update({'gpAmp': 0.35, 'gpLx': -1.0, 'gpLy': -1.0})
    p0_obj.update({'d1': 1.0, 'd2': 0.0, 'd3': 0.0, 'm1': 0.0})
    p0_obj.update({'s0':0, 's0break':0, 's1':0, 's1break':0, 's2':0, 's2break':0, 's3':0, 's3break':0, 's4':0, 's4break':0})
    p0_obj.update({'sigF': 0.0003, 'mode': '', 'Tstar': None, 'Tstar_err': None})
    
    params = np.array(['t0', 'per', 'rp', 'a', 'inc', 'ecosw', 'esinw', 'q1', 'q2', 'fp', 
                            'A', 'B', 'C', 'D', 'r2', 'r2off'])
    params = np.append(params, ['c'+str(i) for i in range(1,22)])
    params = np.append(params, ['p'+str(i)+'_1' for i in range(1,26)])
    params = np.append(params, ['p'+str(i)+'_2' for i in range(1,26)])
    params = np.append(params, ['gpAmp', 'gpLx', 'gpLy'])
    params = np.append(params, ['d1', 'd2', 'd3', 'm1'])
    params = np.append(params, ['s0', 's0break', 's1', 's1break', 's2', 's2break', 's3', 's3break', 's4', 's4break'])
    params = np.append(params, ['sigF'])
 
    fancyParams = np.array([r'$t_0$', r'$P_{\rm orb}$', r'$R_p/R_*$', r'$a/R_*$', r'$i$',
                            r'$e \cos(\omega)$', r'$e \sin(\omega)$', r'$q_1$', r'$q_2$', r'$f_p$', r'$A$', r'$B$',
                            r'$C$', r'$D$', r'$R_{p,2}/R_*$', r'$R_{p,2}/R_*$ Offset'])
    fancyParams = np.append(fancyParams, ['$C_'+str(i)+'$' for i in range(1,22)])
    fancyParams = np.append(fancyParams, [r'$p_{'+str(i)+'-1}$' for i in range(1,26)])
    fancyParams = np.append(fancyParams, [r'$p_{'+str(i)+'-2}$' for i in range(1,26)])
    fancyParams = np.append(fancyParams, [r'$GP_{amp}$', r'$GP_{Lx}$', r'$GP_{Ly}$'])
    fancyParams = np.append(fancyParams, [r'$D_1$', r'$D_2$', r'$D_3$', r'$M_1$'])
    fancyParams = np.append(fancyParams, [r'$S_0$', r'$S_{0, break}$', r'$S_1$', r'$S_{1, break}$', r'$S_2$', r'$S_{2, break}$', r'$S_3$', r'$S_{3, break}$', r'$S_4$', r'$S_{4, break}$'])
    fancyParams = np.append(fancyParams, [r'$\sigma_F$'])
    
    p0_obj.update({'params': params, 'fancyParams': fancyParams,
                   'checkPhasePhis':np.linspace(-np.pi,np.pi,1000)})

    return p0_obj

def get_data(foldername, filename, mode, foldername_aper='', foldername_psf='', cut=0):
    """Retrieve binned data.

    Args:
        path (string): Full path to the data file output by photometry routine.
        mode (string): The string specifying the detector and astrophysical model to use.
        path_aper (string, optional): Full path to the data file output by aperture photometry routine.
        cut (int, optional): Number of data points to remove from the start of the arrays.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    if 'pld' in mode.lower():
        if '3x3' in mode.lower():
            npix = 3
        elif '5x5' in mode.lower():
            npix = 5
        else:
            # FIX, throw an actual error
            print('Error: only 3x3 and 5x5 boxes for PLD are supported.')
            return
        stamp    = np.loadtxt(foldername+filename, usecols=np.arange(int(npix**2)), skiprows=1) # electrons
        time     = np.loadtxt(foldername+filename, usecols=[int(2*npix**2)], skiprows=1)         # BMJD
        
        stamp = stamp[cut:]
        time_pld = time[cut:]
        
        # Sigma clip per data cube (also masks invalids)
        # Convert masks into which indices to keep
        mask_pld = np.logical_not(sigma_clip(stamp.sum(axis=1), sigma=6).mask)
        
    
    if 'pldaper' in mode.lower() or 'pld' not in mode.lower():
        if 'pld' not in mode.lower():
            foldername_aper = foldername
        flux     = np.loadtxt(foldername_aper+filename, usecols=[0], skiprows=1) # electrons
        time     = np.loadtxt(foldername_aper+filename, usecols=[2], skiprows=1) # BMJD
        xdata    = np.loadtxt(foldername_aper+filename, usecols=[4], skiprows=1) # pixel
        ydata    = np.loadtxt(foldername_aper+filename, usecols=[6], skiprows=1) # pixel
        psfxw = np.loadtxt(foldername_aper+filename, usecols=[8], skiprows=1)    # pixel
        psfyw = np.loadtxt(foldername_aper+filename, usecols=[10], skiprows=1)   # pixel
        
        flux = flux[cut:]
        time = time[cut:]
        xdata = xdata[cut:]
        ydata = ydata[cut:]
        psfxw = psfxw[cut:]
        psfyw = psfyw[cut:]
        
        # Sigma clip per data cube (also masks invalids)
        try:
            FLUX_clip  = sigma_clip(flux, sigma=6, maxiters=1)
            XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, maxiters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, maxiters=1)
        except TypeError:
            FLUX_clip  = sigma_clip(flux, sigma=6, iters=1)
            XDATA_clip = sigma_clip(xdata, sigma=6, iters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, iters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, iters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, iters=1)

        # Combine masks for aperture photometry
        MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask
        # Convert masks into which indices to keep
        mask_aper = np.logical_not(MASK)
        
    if 'psfx' in mode.lower():
        foldername_psf = '/'.join(foldername.split('/')[:-2])+'/'+foldername_psf
        xdata    = np.loadtxt(foldername_psf+filename, usecols=[4], skiprows=1) # pixel
        ydata    = np.loadtxt(foldername_psf+filename, usecols=[6], skiprows=1) # pixel
        psfxw = np.loadtxt(foldername_psf+filename, usecols=[8], skiprows=1)    # pixel
        psfyw = np.loadtxt(foldername_psf+filename, usecols=[10], skiprows=1)   # pixel
        
        xdata = xdata[cut:]
        ydata = ydata[cut:]
        psfxw = psfxw[cut:]
        psfyw = psfyw[cut:]
        
        # Sigma clip per data cube (also masks invalids)
        try:
            XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, maxiters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, maxiters=1)
        except TypeError:
            XDATA_clip = sigma_clip(xdata, sigma=6, iters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, iters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, iters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, iters=1)

        # Combine masks for aperture photometry
        MASK  = XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask
        # Convert masks into which indices to keep
        mask_psf = np.logical_not(MASK)
        
    # Combine masks in needed
    if 'pldaper' in mode.lower():
        mask = np.logical_and(mask_pld, mask_aper)
    elif 'pld' in mode.lower():
        mask = mask_pld
    elif 'psfx' in mode.lower():
        mask = np.logical_and(mask_psf, mask_aper)
    else:
        mask = mask_aper
        
        
    # Apply masks
    if 'pld' in mode.lower():
        #Transpose pixel stamp array for easier use
        stamp = stamp[mask].T
        
        if 'pldaper' not in mode.lower():
            time = time[mask]
            flux = np.sum(stamp, axis=0).reshape(1,-1)
            
        #Normalize stamp pixel values by the sum of the stamp
        stamp /= np.sum(stamp, axis=0)
        
    if 'pldaper' in mode.lower() or 'pld' not in mode.lower():
        flux = flux[mask]
        time = time[mask]
        xdata = xdata[mask]
        ydata = ydata[mask]
        psfxw = psfxw[mask]
        psfyw = psfyw[mask]
        
        factor = 1/(np.median(flux))
        flux = factor*flux
        
        # redefining the zero centroid position
        if 'bliss' not in mode.lower():
            mid_x, mid_y = np.mean(xdata), np.mean(ydata)
            xdata -= mid_x
            ydata -= mid_y
    
    if 'pld' in mode.lower():
        if 'pld2' in mode.lower() or 'pldaper2' in mode.lower():
            # Add on the 2nd order PLD pixel lightcurves
            stamp2 = stamp**2
            stamp2 /= stamp2.sum(axis=0)
            stamp = np.append(stamp, stamp, axis=0)
        
        return stamp, flux, time
    else:
        return flux, time, xdata, ydata, psfxw, psfyw

def get_full_data(foldername, filename, mode, foldername_aper='', foldername_psf='', cut=0, nFrames=64, ignore=np.array([])):
    """Retrieve unbinned data.
    
    Args:
        path (string): Full path to the unbinned data file output by photometry routine.
        mode (string): The string specifying the detector and astrophysical model to use.
        path_aper (string, optional): Full path to the data file output by aperture photometry routine.
        cut (int, optional): Number of data points to remove from the start of the arrays.
        nFrames (int, optional): The number of frames that were binned together in the binned data.
        ignore (ndarray, optional): Array specifying which frames were found to be bad and should be ignored.
    
    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).
    
    """
    
    if 'pld' in mode.lower():
        if '3x3' in mode.lower():
            npix = 3
        elif '5x5' in mode.lower():
            npix = 5
        else:
            # FIX, throw an actual error
            print('Error: only 3x3 and 5x5 stamps for PLD are supported.')
            return
        stamp     = np.loadtxt(foldername+filename, usecols=np.arange(int(npix**2)), skiprows=1)       # electrons
        time     = np.loadtxt(foldername+filename, usecols=[int(npix**2)], skiprows=1)     # BMJD
        
        order = np.argsort(time)
        stamp = stamp[order][int(cut*nFrames):]
        time = time[order][int(cut*nFrames):]
        
        # Clip bad frames
        MASK  = sigma_clip(stamp.sum(axis=1), sigma=6).mask+np.isnan(time)
        # Convert masks into which indices to keep
        mask_pld = np.logical_not(MASK)
    
    if 'pldaper' in mode.lower() or 'pld' not in mode.lower():
        if 'pld' not in mode.lower():
            foldername_aper = foldername
        flux     = np.loadtxt(foldername_aper+filename, usecols=[0], skiprows=1)     # electrons
        time     = np.loadtxt(foldername_aper+filename, usecols=[1], skiprows=1)     # hours
        xdata    = np.loadtxt(foldername_aper+filename, usecols=[2], skiprows=1)     # pixels
        ydata    = np.loadtxt(foldername_aper+filename, usecols=[3], skiprows=1)     # pixels
        psfxw    = np.loadtxt(foldername_aper+filename, usecols=[4], skiprows=1)     # pixels
        psfyw    = np.loadtxt(foldername_aper+filename, usecols=[5], skiprows=1)     # pixels
        
        order = np.argsort(time)
        flux = flux[order][int(cut*nFrames):]
        time = time[order][int(cut*nFrames):]
        xdata = xdata[order][int(cut*nFrames):]
        ydata = ydata[order][int(cut*nFrames):]
        psfxw = psfxw[order][int(cut*nFrames):]
        psfyw = psfyw[order][int(cut*nFrames):]
        
        # Sigma clip per data cube (also masks invalids)
        try:
            FLUX_clip  = sigma_clip(flux, sigma=6, maxiters=1)
            XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, maxiters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, maxiters=1)
        except TypeError:
            FLUX_clip  = sigma_clip(flux, sigma=6, iters=1)
            XDATA_clip = sigma_clip(xdata, sigma=6, iters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, iters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, iters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, iters=1)
        
        mask_nan = np.isnan(flux)
        
        # Combine aperture masks
        MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask + mask_nan
        # Convert masks into which indices to keep
        mask_aper = np.logical_not(MASK)
    
    if 'psfx' in mode.lower():
        foldername_psf = '/'.join(foldername.split('/')[:-2])+'/'+foldername_psf
        xdata    = np.loadtxt(foldername_psf+filename, usecols=[2], skiprows=1)     # pixels
        ydata    = np.loadtxt(foldername_psf+filename, usecols=[3], skiprows=1)     # pixels
        psfxw    = np.loadtxt(foldername_psf+filename, usecols=[4], skiprows=1)     # pixels
        psfyw    = np.loadtxt(foldername_psf+filename, usecols=[5], skiprows=1)     # pixels
        
        xdata = xdata[int(cut*nFrames):]
        ydata = ydata[int(cut*nFrames):]
        psfxw = psfxw[int(cut*nFrames):]
        psfyw = psfyw[int(cut*nFrames):]
        
        # Sigma clip per data cube (also masks invalids)
        try:
            XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, maxiters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, maxiters=1)
        except TypeError:
            XDATA_clip = sigma_clip(xdata, sigma=6, iters=1)
            YDATA_clip = sigma_clip(ydata, sigma=6, iters=1)
            PSFXW_clip = sigma_clip(psfxw, sigma=6, iters=1)
            PSFYW_clip = sigma_clip(psfyw, sigma=6, iters=1)
        
        # Combine aperture masks
        MASK  = XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask
        # Convert masks into which indices to keep
        mask_psf = np.logical_not(MASK)
    
    # Combine masks in needed
    if 'pldaper' in mode.lower():
        mask = np.logical_and(mask_pld, mask_aper)
    elif 'pld' in mode.lower():
        mask = mask_pld
    elif 'psfx' in mode.lower():
        mask = np.logical_and(mask_psf, mask_aper)
    else:
        mask = mask_aper
    
    # Apply masks
    if 'pld' in mode.lower():
        #Transpose pixel stamp array for easier use
        stamp = stamp[mask].T
        
        if 'pldaper' not in mode.lower():
            time = time[mask]
            flux = np.sum(stamp, axis=0).reshape(1,-1)
        
        #Normalize stamp pixel values by the sum of the stamp
        stamp /= np.sum(stamp, axis=0)
    
    if 'pldaper' in mode.lower() or 'pld' not in mode.lower():
        flux = flux[mask]
        time = time[mask]
        xdata = xdata[mask]
        ydata = ydata[mask]
        psfxw = psfxw[mask]
        psfyw = psfyw[mask]
        
        factor = 1/(np.median(flux))
        flux = factor*flux
        
        # redefining the zero centroid position
        if 'bliss' not in mode.lower():
            mid_x, mid_y = np.mean(xdata), np.mean(ydata)
            xdata -= mid_x
            ydata -= mid_y
    
    if 'pld' in mode.lower():
        if 'pld2' in mode.lower() or 'pldaper2' in mode.lower():
            # Add on the 2nd order PLD pixel lightcurves
            stamp2 = stamp**2
            stamp2 /= stamp2.sum(axis=0)
            stamp = np.append(stamp, stamp, axis=0)
        
        return stamp, flux, time
    else:
        return flux, time, xdata, ydata, psfxw, psfyw 
    
def expand_dparams(dparams, mode):
    """Add any implicit dparams given the mode (e.g. GP parameters if using a Polynomial model).

    Args:
        dparams (ndarray): A list of strings specifying which parameters shouldn't be fit.
        mode (string): The string specifying the detector and astrophysical model to use.

    Returns:
        ndarray: The updated dparams array.

    """
    
    modeLower = mode.lower()
    
    if 'ellipse' not in modeLower:
        dparams = np.append(dparams, ['r2', 'r2off'])
    elif 'offset' not in modeLower:
        dparams = np.append(dparams, ['r2off'])

    if 'v2' not in modeLower:
        dparams = np.append(dparams, ['C', 'D'])

    if 'poly' not in modeLower:
        dparams = np.append(dparams, ['c'+str(int(i)) for i in range(22)])  
    elif 'poly2' in modeLower:
        dparams = np.append(dparams, ['c'+str(int(i)) for i in range(7,22)])
    elif 'poly3' in modeLower:
        dparams = np.append(dparams, ['c'+str(int(i)) for i in range(11,22)])
    elif 'poly4' in modeLower:
        dparams = np.append(dparams, ['c'+str(int(i)) for i in range(16,22)])
        
    if 'ecosw' in dparams and 'esinw' in dparams:
        dparams = np.append(dparams, ['ecc', 'anom', 'w'])
        
    if 'psfw' not in modeLower:
        dparams = np.append(dparams, ['d1', 'd2', 'd3'])
        
    if 'hside' not in modeLower:
        dparams = np.append(dparams, ['s0', 's0break', 's1', 's1break', 's2', 's2break', 's3', 's3break', 's4', 's4break'])
        
    if 'tslope' not in modeLower:
        dparams = np.append(dparams, ['m1'])
        
    if 'pld' not in mode.lower():
        dparams = np.append(dparams, ['p0_0'])
        dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(1,26)])
        dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(1,26)])
    elif 'pld' in mode.lower():
        if 'pld1' in mode.lower() or 'pldaper1' in mode.lower():
            if '3x3' in mode.lower():
                dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(10,26)])
            dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(1,26)])
        elif '3x3' in mode.lower():
            dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(10,26)])
            dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(10,26)])
        
    if 'gp' not in modeLower:
        dparams = np.append(dparams, ['gpAmp', 'gpLx', 'gpLy'])
    
    return dparams

# FIX: Add a docstring for this function
def get_p0(dparams, obj):
    """Initialize the p0 variable to the defaults.

    Args:
        dparams (ndarray): A list of strings specifying which parameters shouldn't be fit.
        obj (object): An object containing the default values for all fittable parameters. #FIX: change this to dict later

    Returns:
        tuple: p0 (ndarray; the initialized values),\
            fit_params (ndarray; the names of the fitted variables),
            fancy_labels (ndarray; the nicely formatted names of the fitted variables)
    
    """
    
    function_params = obj['params']
    fancy_names = obj['fancyParams']
    
    fit_params = np.array([sa for sa in function_params if not any(sb in sa for sb in dparams)])
    fancy_labels = np.array([fancy_names[i] for i in range(len(function_params)) if not function_params[i] in dparams])
    p0 = np.zeros(len(fit_params),dtype=float)
    for i in range(len(fit_params)):
        p0[i] = obj[fit_params[i]]
    return p0, fit_params, fancy_labels

# FIX: Add a docstring for this function
def lnprior_gaussian(p0, priorInds, priors, errs):
    prior = 0
    for i in range(len(priorInds)):
        prior -= 0.5*(((p0[priorInds[i]] - priors[i])/errs[i])**2.)
    return prior

# FIX: Add a docstring for this function
def lnprior_uniform(p0, priorInds, limits):
    if priorInds is None or len(priorInds)==0:
        # Need to evaluate this first, otherwise the next line would fail
        return 0
    elif np.any(np.logical_or(np.array(limits)[:,0] > p0[priorInds],
                            np.array(limits)[:,1] < p0[priorInds])):
        return -np.inf
    else:
        return 0

# FIX: Add a docstring for this function
def lnprior_gamma(p0, priorInd, shape, rate):
    if priorInd is not None:
        x = p0[priorInd]**2
        alpha = shape
        beta = rate
        return np.log(beta**alpha * x**(alpha-1) * np.exp(-beta*x) / np.math.factorial(alpha-1))
    else:
        return 0

# FIX: Add a docstring for this function
def lnprior_custom(p0, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd):
    # Combine all the different priors
    return (lnprior_gaussian(p0, gpriorInds, priors, errs)+
            lnprior_uniform(p0, upriorInds, uparams_limits)+
            lnprior_gamma(p0, gammaInd, 1, 100))


# FIX - check if sigF in p0, otherwise use a fixed value passed in through signal_input or something
def lnlike(p0, flux, mode, signal_func, signal_inputs):
    """Evaluate the ln-likelihood at the position p0.
    
    Note: We assumine that we are always fitting for the photometric scatter (sigF). 

    Args:
        p0 (ndarray): The array containing the n-D position to evaluate the log-likelihood at.
        p0_labels (ndarray): An array containing the names of the fitted parameters.
        signalfunc (function): The super function to model the astrophysical and detector functions.
        signal_input (list): The collection of other assorted variables required for signalfunc beyond just p0.

    Returns:
        float: The ln-likelihood evaluated at the position p0.
    
    """
    
    if 'gp' in mode.lower():
        temp_signal_inputs = deepcopy(signal_inputs)
        gpInd = np.where([partial_func.func.__name__=='detec_model_GP'
                          for partial_func in temp_signal_inputs[-3]])[0][0]
        temp_signal_inputs[-1][gpInd][-1]=False
        model, gp = signal_func(p0, *temp_signal_inputs)
        
        return gp.log_likelihood(flux-model)
    else:
        # define model
        model = signal_func(p0, *signal_inputs)
        return loglikelihood(flux, model, p0[-1])
    
def lnprob(p0, flux, mode, p0_labels, signal_func, signal_inputs,
           gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd, 
           positivity_func=None, positivity_labels=None):
    """Evaluate the ln-probability of the signal function at the position p0, including priors.

    Args:
        p0 (ndarray): The array containing the n-D position to evaluate the log-likelihood at.
        p0_labels (ndarray): An array containing the names of the fitted parameters.
        signalfunc (function): The super function to model the astrophysical and detector functions.
        lnpriorfunc (function): The function to evaluate the default ln-prior.
        signal_input (list): The collection of other assorted variables required for signalfunc beyond just p0.
        checkPhasePhis (ndarray): The phase angles to use when checking that the phasecurve is always positive.
        lnpriorcustom (function, optional): An additional function to evaluate the a user specified ln-prior function
            (default is None).

    Returns:
        float: The ln-probability evaluated at the position p0.
    
    """
    
    lp = 0
    
    # Evalute the prior first since this is much quicker to compute
    if positivity_func is not None:
        lp = positivity_func(**dict([[label, p0[i]] for i, label in enumerate(p0_labels) if label in positivity_labels]))
        if not np.isfinite(lp):
            return -np.inf
    
    lp += lnprior_custom(p0, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd)
    if not np.isfinite(lp):
        return -np.inf
    
    lp += lnlike(p0, flux, mode, signal_func, signal_inputs)
    if not np.isfinite(lp):
        return -np.inf
    else:
        return lp

def chi2(data, fit, err):
    """Compute the chi-squared statistic.

    Args:
        data (ndarray): The real y values.
        fit (ndarray): The fitted y values.
        err (ndarray or float): The y error(s).

    Returns:
        float: The chi-squared statistic.
    
    """
    
    #using inverse sigma since multiplying is faster than dividing
    inv_err = err**-1
    return np.sum(((data - fit)*inv_err)**2)

def loglikelihood(data, fit, err):
    """Compute the lnlikelihood.

    Args:
        data (ndarray): The real y values.
        fit (ndarray): The fitted y values.
        err (ndarray or float): The y error(s).

    Returns:
        float: The lnlikelihood.
    
    """
    
    #using inverse sigma since multiplying is faster than dividing
    inv_err = err**-1
    len_fit = len(fit)
    return -0.5*np.sum(((data - fit)*inv_err)**2) + len_fit*np.log(inv_err) - len_fit*np.log(np.sqrt(2*np.pi))

def evidence(logL, Npar, Ndat):
    """Compute the Bayesian evidence.

    Args:
        logL (float): The lnlikelihood.
        Npar (int): The number of fitted parameters.
        Ndat (int): The number of data fitted.

    Returns:
        float: The Bayesian evidence.
    
    """
    
    return logL - (Npar/2.)*np.log(Ndat)

def BIC(logL, Npar, Ndat):
    """Compute the Bayesian Information Criterion.

    Args:
        logL (float): The lnlikelihood.
        Npar (int): The number of fitted parameters.
        Ndat (int): The number of data fitted.

    Returns:
        float: The Bayesian Information Criterion.
    
    """
    
    return -2.*evidence(logL, Npar, Ndat)


def binValues(values, binAxisValues, nbin, assumeWhiteNoise=False):
    """Bin values and compute their binned noise.

    Args:
        values (ndarray): An array of values to bin.
        binAxisValues (ndarray): Values of the axis along which binning will occur.
        nbin (int): The number of bins desired.
        assumeWhiteNoise (bool, optional): Divide binned noise by sqrt(nbinned) (True) or not (False, default).

    Returns:
        tuple: binned (ndarray; the binned values),
            binnedErr (ndarray; the binned errors)
    
    """
    
    bins = np.linspace(np.nanmin(binAxisValues), np.nanmax(binAxisValues), nbin)
    digitized = np.digitize(binAxisValues, bins)
    binned = np.array([np.nanmedian(values[digitized == i]) for i in range(1, nbin)])
    binnedErr = np.nanmean(np.array([np.nanstd(values[digitized == i]) for i in range(1, nbin)]))
    if assumeWhiteNoise:
        binnedErr /= np.sqrt(len(values)/nbin)
    return binned, binnedErr

def binnedNoise(x, y, nbin):
    """Compute the binned noise (not assuming white noise)

    Args:
        x (ndarray): The values along the binning axis.
        y (ndarray): The values which should be binned.
        nbin (int): The number of bins desired.

    Returns:
        ndarray: The binned noise (not assuming white noise).
    
    """
    
    bins = np.linspace(np.min(x), np.max(x), nbin)
    digitized = np.digitize(x, bins)
    y_means = np.array([np.nanmean(y[digitized == i]) for i in range(1, nbin)])
    return np.nanstd(y_means)

def getIngressDuration(p0_mcmc, p0_labels, p0_obj, intTime):
    """Compute the transit/eclipse ingress duration in units of datapoints.
    
    Warning - this assumes a circular orbit!

    Args:
        p0_mcmc (ndarray): The array containing the fitted values.
        p0_labels (ndarray): The array containing all of the names of the fittable parameters.
        p0_obj (object): The object containing the default values for non-fitted variables.
        intTime (float): The integration time of each measurement.

    Returns:
        float: The transit/eclipse ingress duration in units of datapoints.
    
    """
    
    if 'rp' in p0_labels:
        rpMCMC = p0_mcmc[np.where(p0_labels == 'rp')[0][0]]
    else:
        rpMCMC = p0_obj['rp']
    if 'a' in p0_labels:
        aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
    else:
        aMCMC = p0_obj['a']
    if 'per' in p0_labels:
        perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
    else:
        perMCMC = p0_obj['per']

    return (2*rpMCMC/(2*np.pi*aMCMC/perMCMC))/intTime

def getOccultationDuration(p0_mcmc, p0_labels, p0_obj, intTime):
    """Compute the full transit/eclipse duration in units of datapoints.
    
    Warning - this assumes a circular orbit!

    Args:
        p0_mcmc (ndarray): The array containing the fitted values.
        p0_labels (ndarray): The array containing all of the names of the fittable parameters.
        p0_obj (object): The object containing the default values for non-fitted variables.
        intTime (float): The integration time of each measurement.

    Returns:
        float: The full transit/eclipse duration in units of datapoints
    
    """
    
    if 'rp' in p0_labels:
        rpMCMC = p0_mcmc[np.where(p0_labels == 'rp')[0][0]]
    else:
        rpMCMC = p0_obj['rp']
    if 'a' in p0_labels:
        aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
    else:
        aMCMC = p0_obj['a']
    if 'per' in p0_labels:
        perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
    else:
        perMCMC = p0_obj['per']

    return (2/(2*np.pi*aMCMC/perMCMC))/intTime
