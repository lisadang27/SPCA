import numpy as np
import scipy.optimize as spopt

import matplotlib.pyplot as plt

from astropy.stats import sigma_clip

import inspect

import os, sys
lib_path = os.path.abspath(os.path.join('../'))
sys.path.append(lib_path)

# SPCA libraries
import SPCA
from SPCA import astro_models, detec_models, bliss

class signal_params(object):
    # class constructor
    def __init__(self, name='planet', t0=0., per=1., rp=0.1,
                 a=8., inc=90., ecosw=0.0, esinw=0.0, q1=0.01, q2=0.01,
                 fp=0.001, A=0.1, B=0.0, C=0.0, D=0.0, sigF=0.0003, mode=''):
        self.name    = name
        self.t0      = t0
        self.t0_err  = 0.0
        self.per     = per
        self.per_err = 0.0
        self.rp      = rp
        self.a       = a
        self.a_err   = 0.0
        self.inc     = inc
        self.inc_err = 0.0
        self.ecosw = ecosw
        self.esinw = esinw
        self.q1    = q1
        self.q2    = q2
        self.fp    = fp
        self.A     = A
        self.B     = B
        self.C     = C
        self.D     = D
        self.r2    = rp
        self.r2off = 0.0
        self.c1    = 1.0     # Poly coeff 
        self.c2    = 0.0     # Poly coeff 
        self.c3    = 0.0     # Poly coeff 
        self.c4    = 0.0     # Poly coeff 
        self.c5    = 0.0     # Poly coeff 
        self.c6    = 0.0     # Poly coeff 
        self.c7    = 0.0     # Poly coeff 
        self.c8    = 0.0     # Poly coeff 
        self.c9    = 0.0     # Poly coeff 
        self.c10   = 0.0     # Poly coeff 
        self.c11   = 0.0     # Poly coeff 
        self.c12   = 0.0     # Poly coeff 
        self.c15   = 0.0     # Poly coeff 
        self.c13   = 0.0     # Poly coeff 
        self.c14   = 0.0     # Poly coeff 
        self.c16   = 0.0     # Poly coeff 
        self.c17   = 0.0     # Poly coeff 
        self.c18   = 0.0     # Poly coeff 
        self.c19   = 0.0     # Poly coeff 
        self.c20   = 0.0     # Poly coeff 
        self.c21   = 0.0     # Poly coeff 
        self.d1    = 1.0     # PSF width coeff 
        self.d2    = 0.0     # PSF width coeff 
        self.d3    = 0.0     # PSF width coeff 
        self.s1    = 0.0     # step function coeff 
        self.s2    = 0.0     # step function coeff
        self.m1    = 0.0     # tslope coeff 
        self.p1_1  = 1.0     # PLD coefficient
        self.p2_1  = 1.0     # PLD coefficient
        self.p3_1  = 1.0     # PLD coefficient
        self.p4_1  = 1.0     # PLD coefficient
        self.p5_1  = 1.0     # PLD coefficient
        self.p6_1  = 1.0     # PLD coefficient
        self.p7_1  = 1.0     # PLD coefficient
        self.p8_1  = 1.0     # PLD coefficient
        self.p9_1  = 1.0     # PLD coefficient
        self.p10_1 = 0.0     # PLD coefficient
        self.p11_1 = 0.0     # PLD coefficient
        self.p12_1 = 0.0     # PLD coefficient
        self.p13_1 = 0.0     # PLD coefficient
        self.p14_1 = 0.0     # PLD coefficient
        self.p15_1 = 0.0     # PLD coefficient
        self.p16_1 = 0.0     # PLD coefficient
        self.p17_1 = 0.0     # PLD coefficient
        self.p18_1 = 0.0     # PLD coefficient
        self.p19_1 = 0.0     # PLD coefficient
        self.p20_1 = 0.0     # PLD coefficient
        self.p21_1 = 0.0     # PLD coefficient
        self.p22_1 = 0.0     # PLD coefficient
        self.p23_1 = 0.0     # PLD coefficient
        self.p24_1 = 0.0     # PLD coefficient
        self.p25_1 = 0.0     # PLD coefficient        
        self.p1_2  = 0.0     # PLD coefficient
        self.p2_2  = 0.0     # PLD coefficient
        self.p3_2  = 0.0     # PLD coefficient
        self.p4_2  = 0.0     # PLD coefficient
        self.p5_2  = 0.0     # PLD coefficient
        self.p6_2  = 0.0     # PLD coefficient
        self.p7_2  = 0.0     # PLD coefficient
        self.p8_2  = 0.0     # PLD coefficient
        self.p9_2  = 0.0     # PLD coefficient
        self.p10_2 = 0.0     # PLD coefficient
        self.p11_2 = 0.0     # PLD coefficient
        self.p12_2 = 0.0     # PLD coefficient
        self.p13_2 = 0.0     # PLD coefficient
        self.p14_2 = 0.0     # PLD coefficient
        self.p15_2 = 0.0     # PLD coefficient
        self.p16_2 = 0.0     # PLD coefficient
        self.p17_2 = 0.0     # PLD coefficient
        self.p18_2 = 0.0     # PLD coefficient
        self.p19_2 = 0.0     # PLD coefficient
        self.p20_2 = 0.0     # PLD coefficient
        self.p21_2 = 0.0     # PLD coefficient
        self.p22_2 = 0.0     # PLD coefficient
        self.p23_2 = 0.0     # PLD coefficient
        self.p24_2 = 0.0     # PLD coefficient
        self.p25_2 = 0.0     # PLD coefficient
        self.gpAmp = -2.     # GP covariance amplitude
        self.gpLx  = -2.     # GP lengthscale in x
        self.gpLy  = -2.     # GP lengthscale in y
        self.sigF  = sigF    # White noise
        self.mode  = mode
        self.Tstar = None
        self.Tstar_err = None
        
        # labels for all the possible fit parameters
        self.params = np.array(['t0', 'per', 'rp', 'a', 'inc', 'ecosw', 'esinw', 'q1', 'q2', 'fp', 
                             'A', 'B', 'C', 'D', 'r2', 'r2off'])
        
        self.params = np.append(self.params, ['c'+str(i) for i in range(1,22)])
        self.params = np.append(self.params, ['d1', 'd2', 'd3', 's1', 's2', 'm1'])
        self.params = np.append(self.params, ['p'+str(i)+'_1' for i in range(1,26)])
        self.params = np.append(self.params, ['p'+str(i)+'_2' for i in range(1,26)])
        self.params = np.append(self.params, ['gpAmp', 'gpLx', 'gpLy', 'sigF'])

        # fancy labels for plot purposed  for all possible fit parameters
        self.fancyParams = np.array([r'$t_0$', r'$P_{\rm orb}$', r'$R_p/R_*$', r'$a/R_*$', r'$i$',
                                       r'$e \cos(\omega)$', r'$e \sin(\omega)$', r'$q_1$', r'$q_2$', r'$f_p$', r'$A$', r'$B$',
                                       r'$C$', r'$D$', r'$R_{p,2}/R_*$', r'$R_{p,2}/R_*$ Offset'])
        self.fancyParams = np.append(self.fancyParams, ['$C_'+str(i)+'$' for i in range(1,22)])
        self.fancyParams = np.append(self.fancyParams, [r'$D_1$', r'$D_2$', r'$D_3$', r'$S_1$', r'$S_2$', r'$M_1$'])
        self.fancyParams = np.append(self.fancyParams, ['$p_{'+str(i)+'-1}$' for i in range(1,22)])
        self.fancyParams = np.append(self.fancyParams, ['$p_{'+str(i)+'-2}$' for i in range(1,22)])
        self.fancyParams = np.append(self.fancyParams, [r'$GP_{amp}$', r'$GP_{Lx}$', r'$GP_{Ly}$', r'$\sigma_F$'])
        

def get_data(path, mode='', cut=0):
    """Retrieve binned data.

    Args:
        path (string): Full path to the data file output by photometry routine.
        mode (string): The string specifying the detector and astrophysical model to use.
        cut (int): Number of data points to remove from the start of the arrays.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
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
        stamp     = np.loadtxt(path, usecols=np.arange(int(npix**2)), skiprows=1)       # mJr/str
        time     = np.loadtxt(path, usecols=[int(2*npix**2)], skiprows=1)     # BMJD
        
        order = np.argsort(time)
        stamp = stamp[order][cut:]
        time = time[order][cut:]
        
        mask = np.logical_not(sigma_clip(stamp.sum(axis=1), sigma=6).mask)
        
        #Transpose pixel stamp array for easier use
        stamp = stamp[mask].T
        time = time[mask]
        
        stamp /= np.median(np.sum(stamp, axis=0))
        
        # Could replace this with aperture photometry flux instead if we wanted
        flux = np.sum(stamp, axis=0)
        
        #Normalize stamp pixel values by the sum of the stamp
        stamp /= flux
        
        return stamp, flux, time
    
    else:
        flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
        flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
        time     = np.loadtxt(path, usecols=[2], skiprows=1)     # BMJD
        xdata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixel
        ydata    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixel
        psfxw = np.loadtxt(path, usecols=[8], skiprows=1)     # pixel
        psfyw = np.loadtxt(path, usecols=[10], skiprows=1)    # pixel

        factor = 1/(np.median(flux))
        flux = factor*flux
        flux_err = factor*flux
        
        order = np.argsort(time)
        flux = flux[order][cut:]
        flux_err = flux_err[order][cut:]
        time = time[order][cut:]
        xdata = xdata[order][cut:]
        ydata = ydata[order][cut:]
        psfxw = psfxw[order][cut:]
        psfyw = psfyw[order][cut:]
        
        # Sigma clip per data cube (also masks invalids)
        FLUX_clip  = sigma_clip(flux, sigma=6, maxiters=1)
        FERR_clip  = sigma_clip(flux_err, sigma=6, maxiters=1)
        XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
        YDATA_clip = sigma_clip(ydata, sigma=3.5, maxiters=1)
        PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
        PSFYW_clip = sigma_clip(psfyw, sigma=3.5, maxiters=1)
        
        # Ultimate Clipping
        MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask
        mask = np.logical_not(MASK)
        
        flux = flux[mask]
        flux_err = flux_err[mask]
        time = time[mask]
        xdata = xdata[mask]
        ydata = ydata[mask]
        psfxw = psfxw[mask]
        psfyw = psfyw[mask]
    
        # redefining the zero centroid position
        if 'bliss' not in mode.lower():
            mid_x, mid_y = np.nanmean(xdata), np.nanmean(ydata)
            xdata -= mid_x
            ydata -= mid_y
    
        return flux, flux_err, time, xdata, ydata, psfxw, psfyw

def get_full_data(path, mode='', cut=0, nFrames=64, ignore=np.array([])):
    """Retrieve unbinned data.

    Args:
        path (string): Full path to the unbinned data file output by photometry routine.
        mode (string): The string specifying the detector and astrophysical model to use.
        cut (int): Number of data points to remove from the start of the arrays.
        ignore (ndarray): Array specifying which frames were found to be bad and should be ignored.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
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
        stamp     = np.loadtxt(path, usecols=np.arange(int(npix**2)), skiprows=1)       # mJr/str
        time     = np.loadtxt(path, usecols=[int(npix**2)], skiprows=1)     # BMJD
        
        order = np.argsort(time)
        stamp = flux[order][int(cut*nFrames):]
        time = time[order][int(cut*nFrames):]
        
        # Clip bad frames
        ind = np.array([])
        for i in ignore:
            ind = np.append(ind, np.arange(i, len(stamp), nFrames))
        mask_id = np.zeros(len(stamp))
        mask_id[ind.astype(int)] = 1
        mask_id = np.ma.make_mask(mask_id)

        # Ultimate Clipping
        MASK  = sigma_clip(flux.sum(axis=1), sigma=6).mask + mask_id
        mask = np.logical_not(MASK)
        
        #Transpose pixel stamp array for easier use
        stamp = stamp[mask].T
        time = time[mask]
        
        stamp /= np.median(np.sum(stamp, axis=0))
        
        # Could replace this with aperture photometry flux instead if we wanted
        flux = np.sum(stamp, axis=0)
        
        #Normalize stamp pixel values by the sum of the stamp
        stamp /= flux
        
        return stamp, flux, time
    
    else:
        #Loading Data
        flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
        flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
        time     = np.loadtxt(path, usecols=[2], skiprows=1)     # hours
        xdata    = np.loadtxt(path, usecols=[3], skiprows=1)     # pixels
        ydata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixels
        psfxw    = np.loadtxt(path, usecols=[5], skiprows=1)     # pixels
        psfyw    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixels

        order = np.argsort(time)
        flux = flux[order][int(cut*nFrames):]
        flux_err = flux_err[order][int(cut*nFrames):]
        time = time[order][int(cut*nFrames):]
        xdata = xdata[order][int(cut*nFrames):]
        ydata = ydata[order][int(cut*nFrames):]
        psfxw = psfxw[order][int(cut*nFrames):]
        psfyw = psfyw[order][int(cut*nFrames):]
        
        # Sigma clip per data cube (also masks invalids)
        FLUX_clip  = sigma_clip(flux, sigma=6, maxiters=1)
        FERR_clip  = sigma_clip(flux_err, sigma=6, maxiters=1)
        XDATA_clip = sigma_clip(xdata, sigma=6, maxiters=1)
        YDATA_clip = sigma_clip(ydata, sigma=3.5, maxiters=1)
        PSFXW_clip = sigma_clip(psfxw, sigma=6, maxiters=1)
        PSFYW_clip = sigma_clip(psfyw, sigma=3.5, maxiters=1)
        
        # Clip bad frames
        ind = np.array([])
        for i in ignore:
            ind = np.append(ind, np.arange(i, len(flux), nFrames))
        mask_id = np.zeros(len(flux))
        mask_id[ind.astype(int)] = 1
        mask_id = np.ma.make_mask(mask_id)
    
        # Ultimate Clipping
        MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask + mask_id
        mask = np.logical_not(MASK)
    
        flux = flux[mask]
        flux_err = flux_err[mask]
        time = time[mask]
        xdata = xdata[mask]
        ydata = ydata[mask]
        psfxw = psfxw[mask]
        psfyw = psfyw[mask]
        
        # redefining the zero centroid position
        if 'bliss' not in mode.lower():
            mid_x, mid_y = np.nanmean(xdata), np.nanmean(ydata)
            xdata -= mid_x
            ydata -= mid_y
    
        return flux, flux_err, time, xdata, ydata, psfxw, psfyw

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
        dparams = np.append(dparams, ['s1', 's2'])
        
    if 'tslope' not in modeLower:
        dparams = np.append(dparams, ['m1'])
        
    if 'pld' not in mode.lower():
        dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(1,26)])
        dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(1,26)])
    elif 'pld' in mode.lower():
        if 'pld1' in mode.lower():
            if '3x3' in mode.lower():
                dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(10,26)])
            dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(1,26)])
        elif 'pld2' in mode.lower() and '3x3' in mode.lower():
            dparams = np.append(dparams, ['p'+str(int(i))+'_1' for i in range(10,26)])
            dparams = np.append(dparams, ['p'+str(int(i))+'_2' for i in range(10,26)])
        
    if 'gp' not in modeLower:
        dparams = np.append(dparams, ['gpAmp', 'gpLx', 'gpLy'])
    
    return dparams


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
    
    function_params = p0_obj.params
    fancy_names = p0_obj.fancyParams
    
    fit_params = np.array([sa for sa in function_params if not any(sb in sa for sb in dparams)])
    fancy_labels = np.array([fancy_names[i] for i in range(len(function_params)) if not any(sb in function_params[i] for sb in dparams)])
    p0 = np.zeros(len(fit_params),dtype=float)
    for i in range(len(fit_params)):
        # FIX: switch to using dictionaries to cut out this instance of eval
        p0[i] = eval('obj.'+ fit_params[i])
    return p0, fit_params, fancy_labels

def get_lambdaparams(function):
    return inspect.getargspec(function).args[1:]

def get_fitted_params(function, dparams):
    if function.__name__=='detec_model_GP':
        if 'sigF' in dparams:
            params = []
        else:
            params = ['sigF']
    else:
        params = get_lambdaparams(function)
        params = [param for param in params if param not in dparams]
    return params

# FIX - this is currently empty!!!
def load_past_params(path):
    """Load the fitted parameters from a previous run.

    Args:
        path (string): Path to the file containing past mcmc result (must be a table saved as .npy file).

    Returns:
        ndarray: p0 (the previously fitted values)
    
    """
    
    return

# FIX - keep trying to think of ways of removing any/all instances of eval...
def make_lambdafunc(function, dparams=[], obj=[], debug=False):
    """Create a lambda function called dynamic_funk that will fix the parameters listed in dparams with the values in obj.

    Note: The module where the original function is needs to be loaded in this file.
    
    Args:
        function (string): Name of the original function.
        dparams (list, optional): List of all input parameters the user does not wish to fit (default is None.)
        obj (string, optional): Object containing all initial and fixed parameter values (default is None.)
        debug (bool, optional): If true, will print mystr so the user can read the command because executing it (default is False).

    Returns:
        function: dynamic_funk (the lambda function with fixed parameters.)
    
    """
    
    module   = function.__module__
    namefunc = function.__name__
    # get list of params you wish to fit
    function_params  = np.asarray(inspect.getargspec(function).args)
    index    = np.in1d(function_params, dparams)
    fit_params  = function_params[np.where(index==False)[0]]
    # assign value to fixed variables
    varstr  = ''
    for label in function_params:
        if label in dparams and label != 'r2':
            tmp = 'obj.' + label
            varstr += str(eval(tmp)) + ', '
        elif label in dparams and label == 'r2':
            varstr += 'rp' + ', '
        else:
            varstr += label + ', '
    #remove extra ', '
    varstr = varstr[:-2]
    
    parmDefaults = inspect.getargspec(function).defaults
    if parmDefaults is not None:
        parmDefaults = np.array(parmDefaults, dtype=str)
        nOptionalParms = len(parmDefaults)
        if np.all(index[-nOptionalParms:]):
            # if all optional parameters are in dparams, remove them from this list
            nOptionalParms = 0
        elif np.any(index[-nOptionalParms:]):
            parmDefaults = parmDefaults[np.logical_not(index[-nOptionalParms:])]
            nOptionalParms = len(parmDefaults)    
    else:
        nOptionalParms = 0
    
    # generate the line to execute
    mystr = 'global dynamic_funk; dynamic_funk = lambda '
    for i in range(len(fit_params)-nOptionalParms):
        mystr = mystr + fit_params[i] +', '
    # add in any optional parameters
    for i in range(nOptionalParms):
        mystr = mystr + fit_params[len(fit_params)-nOptionalParms+i] + '=' + parmDefaults[i] + ', '
    #remove extra ', '
    mystr = mystr[:-2]
    #mystr = mystr +': '+namefunc+'(' + varstr + ')'
    if module == 'helpers':
        mystr = mystr +': '+namefunc+'(' + varstr + ')'
    else: 
        mystr = mystr +': '+module+'.'+namefunc+'(' + varstr + ')'
    # executing the line
    exec(mystr)
    if debug:
        print(mystr)
        print()
    return dynamic_funk

def lnprior_gaussian(p0, priorInds, priors, errs):
    prior = 0
    for i in range(len(priorInds)):
        prior -= 0.5*(((p0[priorInds[i]] - priors[i])/errs[i])**2.)
    return prior

def lnprior_uniform(p0, priorInds, limits):
    if priorInds == []:
        return 0
    elif np.any(np.logical_or(np.array(limits)[:,0] < p0[priorInds],
                            np.array(limits)[:,1] > p0[priorInds])):
        return -np.inf
    else:
        return 0

def lnprior_gamma(p0, priorInd, shape, rate):
    if priorInd is not None:
        x = np.exp(p0[priorInd])
        alpha = shape
        beta = rate
        return np.log(beta**alpha * x**(alpha-1) * np.exp(-beta*x) / np.math.factorial(alpha-1))
    else:
        return 0


# FIX - is it possible to remove the assumption that we're always fitting for sigF? What if we wrap everything with super functions that lazily evaluate freezings, rather than making a lambda function at the start?
def lnlike(p0, signalfunc, signal_input):
    """Evaluate the ln-likelihood at the position p0.
    
    Note: We assumine that we are always fitting for the photometric scatter (sigF). 

    Args:
        p0 (ndarray): The array containing the n-D position to evaluate the log-likelihood at.
        signalfunc (function): The super function to model the astrophysical and detector functions.
        signal_input (list): The collection of other assorted variables required for signalfunc beyond just p0.

    Returns:
        float: The ln-likelihood evaluated at the position p0.
    
    """
    
    flux = signal_input[0]
    mode = signal_input[-1]
    
    if 'gp' in mode.lower():
        model, gp = signalfunc(signal_input, *p0, predictGp=False, returnGp=True)
        
        return gp.log_likelihood(flux-model)
    else:
        # define model
        model = signalfunc(signal_input, *p0)
        return loglikelihood(flux, model, p0[-1])
    

def lnprob(p0, signalfunc, lnpriorfunc, signal_input, checkPhasePhis, lnpriorcustom=None):
    """Evaluate the ln-probability of the signal function at the position p0, including priors.

    Args:
        p0 (ndarray): The array containing the n-D position to evaluate the log-likelihood at.
        signalfunc (function): The super function to model the astrophysical and detector functions.
        lnpriorfunc (function): The function to evaluate the default ln-prior.
        signal_input (list): The collection of other assorted variables required for signalfunc beyond just p0.
        checkPhasePhis (ndarray): The phase angles to use when checking that the phasecurve is always positive.
        lnpriorcustom (function, optional): An additional function to evaluate the a user specified ln-prior function
            (default is None).

    Returns:
        float: The ln-probability evaluated at the position p0.
    
    """
    
    # Evalute the prior first since this is much quicker to compute
    lp = lnpriorfunc(*p0, signal_input[-1], checkPhasePhis)

    if (lnpriorcustom is not None):
        lp += lnpriorcustom(p0)
    if not np.isfinite(lp):
        return -np.inf
    else:
        lp += lnlike(p0, signalfunc, signal_input)

    if np.isfinite(lp):
        return lp
    else:
        return -np.inf

def lnprior(t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
            c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9,  c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21,
            d1, d2, d3, s1, s2, m1,
            p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, p11_1, p12_1, p13_1, p14_1, p15_1,
            p16_1, p17_1, p18_1, p19_1, p20_1, p21_1, p22_1, p23_1, p24_1, p25_1,
            p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2,
            p16_2, p17_2, p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2,
            gpAmp, gpLx, gpLy, sigF,
            mode, checkPhasePhis):
    """Check that the parameters are physically plausible.

    Args:
        t0 (float): Time of inferior conjunction.
        per (float): Orbital period.
        rp (float): Planet radius (in units of stellar radii).
        a (float): Semi-major axis (in units of stellar radii).
        inc (float): Orbital inclination (in degrees).
        ecosw (float): Eccentricity multiplied by the cosine of the longitude of periastron (value between -1 and 1).
        esinw (float): Eccentricity multiplied by the sine of the longitude of periastron (value between -1 and 1).
        q1 (float): Limb darkening coefficient 1, parametrized to range between 0 and 1.
        q2 (float): Limb darkening coefficient 2, parametrized to range between 0 and 1.
        fp (float): Planet-to-star flux ratio.
        A (float): Amplitude of the first-order cosine term.
        B (float): Amplitude of the first-order sine term.
        C (float): Amplitude of the second-order cosine term. Default=0.
        D (float): Amplitude of the second-order sine term. Default=0.
        r2 (float): Planet radius along sub-stellar axis (in units of stellar radii). Default=None.
        r2off (float): Angle to the elongated axis with respect to the sub-stellar axis (in degrees). Default=None.
        c1--c21 (float): The polynomial model amplitudes.
        d1 (float): The constant offset term. #FIX - I don't think this should be here.
        d2 (float): The slope in sensitivity with the PSF width in the x direction.
        d3 (float): The slope in sensitivity with the PSF width in the y direction.
        s1 (float): The amplitude of the heaviside step function.
        s2 (float): The location of the step in the heaviside function.
        m1 (float): The slope in sensitivity over time with respect to time[0].
        p1_1--p25_1 (float): The 1st order PLD coefficients for 3x3 or 5x5 PLD stamps.
        p1_2--p25_2 (float): The 2nd order PLD coefficients for 3x3 or 5x5 PLD stamps.
        gpAmp (float): The natural logarithm of the GP covariance amplitude.
        gpLx (float): The natural logarithm of the GP covariance lengthscale in x.
        gpLy (float): The natural logarithm of the GP covariance lengthscale in y.
        sigF (float): The white noise in units of F_star.
        mode (string): The string specifying the detector and astrophysical model to use.
        checkPhasePhis (ndarray): The phase angles to use when checking that the phasecurve is always positive.

    Returns:
        float: The default ln-prior evaluated at the position p0.
    
    """
    
    check = astro_models.check_phase(checkPhasePhis, A, B, C, D)
    if ((0 < rp < 1) and (0 < fp < 1) and (0 < q1 < 1) and (0 < q2 < 1) and
        (-1 < ecosw < 1) and (-1 < esinw < 1) and (check == False) and (sigF > 0)
        and (m1 > -1)):
        return 0.0
    else:
        return -np.inf

# FIX - move this to make_plots.py
def walk_style(ndim, nwalk, samples, interv, subsamp, labels, fname=None):
    """Make a plot showing the evolution of the walkers throughout the emcee sampling.

    Args:
        ndim (int): Number of free parameters
        nwalk (int): Number of walkers
        samples (ndarray): The ndarray accessed by calling sampler.chain when using emcee
        interv (int): Take every 'interv' element to thin out the plot
        subsamp (int): Only show the last 'subsamp' steps
        labels (ndarray): The fancy labels for each dimension
        fname (string, optional): The savepath for the plot (or None if you want to return the figure instead).

    Returns:
        None
    
    """
    
    # get first index
    beg   = len(samples[0,:,0]) - subsamp
    end   = len(samples[0,:,0]) 
    step  = np.arange(beg,end)
    step  = step[::interv] 
    
    # number of columns and rows of subplots
    ncols = 4
    nrows = int(np.ceil(ndim/ncols))
    sizey = 2*nrows
    
    # plotting
    plt.figure(figsize = (15, 2*nrows))
    for ind in range(ndim):
        plt.subplot(nrows, ncols, ind+1)
        sig1 = (0.6827)/2.*100
        sig2 = (0.9545)/2.*100
        sig3 = (0.9973)/2.*100
        percentiles = [50-sig3, 50-sig2, 50-sig1, 50, 50+sig1, 50+sig2, 50+sig3]
        neg3sig, neg2sig, neg1sig, mu_param, pos1sig, pos2sig, pos3sig = np.percentile(samples[:,:,ind][:,beg:end:interv], percentiles, axis=0)
        plt.plot(step, mu_param)
        plt.fill_between(step, pos3sig, neg3sig, facecolor='k', alpha = 0.1)
        plt.fill_between(step, pos2sig, neg2sig, facecolor='k', alpha = 0.1)
        plt.fill_between(step, pos1sig, neg1sig, facecolor='k', alpha = 0.1)
        plt.title(labels[ind])
        plt.xlim(np.min(step), np.max(step))
        if ind < (ndim - ncols):
            plt.xticks([])
        else: 
            plt.xticks(rotation=25)
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    else:
        # FIX - return the figure instead
        plt.show()
    plt.close()
    return    

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
        rpMCMC = p0_obj.rp
    if 'a' in p0_labels:
        aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
    else:
        aMCMC = p0_obj.a
    if 'per' in p0_labels:
        perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
    else:
        perMCMC = p0_obj.per

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
        rpMCMC = p0_obj.rp
    if 'a' in p0_labels:
        aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
    else:
        aMCMC = p0_obj.a
    if 'per' in p0_labels:
        perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
    else:
        perMCMC = p0_obj.per

    return (2/(2*np.pi*aMCMC/perMCMC))/intTime
