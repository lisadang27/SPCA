import numpy as np
import scipy.optimize as spopt

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

from astropy.stats import sigma_clip

import inspect

import astro_models
import detec_models
import bliss

from numba import jit


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
        self.c1    = 1.0
        self.c2    = 0.0
        self.c3    = 0.0
        self.c4    = 0.0
        self.c5    = 0.0
        self.c6    = 0.0
        self.c7    = 0.0
        self.c8    = 0.0
        self.c9    = 0.0
        self.c10   = 0.0
        self.c11   = 0.0
        self.c12   = 0.0
        self.c15   = 0.0
        self.c13   = 0.0
        self.c14   = 0.0
        self.c16   = 0.0
        self.c17   = 0.0
        self.c18   = 0.0
        self.c19   = 0.0
        self.c20   = 0.0
        self.c21   = 0.0
        self.d1    = 1.0
        self.d2    = 0.0
        self.d3    = 0.0
        self.s1    = 0.0
        self.s2    = 0.0
        self.m1    = 0.0
        self.gpAmp = -2.
        self.gpLx  = -6.
        self.gpLy  = -6.
        self.sigF  = sigF
        self.mode  = mode
        self.Tstar = None
        self.Tstar_err = None

def get_data(path):
    """Retrieve binned data.

    Args:
        path (string): Full path to the data file output by photometry routine.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    #Loading Data
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
    flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # BMJD
    xdata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixel
    ydata    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixel
    psfxwdat = np.loadtxt(path, usecols=[8], skiprows=1)     # pixel
    psfywdat = np.loadtxt(path, usecols=[10], skiprows=1)    # pixel
    
    factor = 1/(np.median(flux))
    flux = factor*flux
    flux_err = factor*flux
    return flux, flux_err, time, xdata, ydata, psfxwdat, psfywdat

def get_full_data(foldername, filename):
    """Retrieve unbinned data.

    Args:
        foldername (string): Full path to the data file output by photometry routine.
        filename (string): File name of the unbinned data file output by photometry routine.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    path = foldername + filename
    #Loading Data
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
    flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # hours
    xdata    = np.loadtxt(path, usecols=[3], skiprows=1)     # pixels
    ydata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixels
    psfxw    = np.loadtxt(path, usecols=[5], skiprows=1)     # pixels
    psfyw    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixels
    
    #remove bad values so that BLISS mapping will work
    mask = np.where(np.logical_and(np.logical_and(np.logical_and(np.isfinite(flux), np.isfinite(flux_err)), 
                                                  np.logical_and(np.isfinite(xdata), np.isfinite(ydata))),
                                   np.logical_and(np.isfinite(psfxw), np.isfinite(psfyw))))
    
    return flux[mask], flux_err[mask], time[mask], xdata[mask], ydata[mask], psfxw[mask], psfyw[mask]

def clip_full_data(FLUX, FERR, TIME, XDATA, YDATA, PSFXW, PSFYW, nFrames=64, cut=0, ignore=np.array([])):
    """Sigma cip the unbinned data.

    Args:
        flux (ndarray): Flux extracted for each frame.
        flux_err (ndarray): uncertainty on the flux for each frame.
        time (ndarray): Time stamp for each frame.
        xdata (ndarray): X-coordinate of the centroid for each frame.
        ydata (ndarray): Y-coordinate of the centroid for each frame.
        psfwx (ndarray): X-width of the target's PSF for each frame.
        psfwy (ndarray): Y-width of the target's PSF for each frame.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    # chronological order
    index = np.argsort(TIME)
    FLUX  = FLUX[index]
    FERR  = FERR[index]
    TIME  = TIME[index]
    XDATA = XDATA[index]
    YDATA = YDATA[index]
    PSFXW = PSFXW[index]
    PSFYW = PSFYW[index]

    # crop the first AOR (if asked)
    FLUX  = FLUX[int(cut*nFrames):]
    FERR  = FERR[int(cut*nFrames):]
    TIME  = TIME[int(cut*nFrames):]
    XDATA = XDATA[int(cut*nFrames):]
    YDATA = YDATA[int(cut*nFrames):]
    PSFXW = PSFXW[int(cut*nFrames):]
    PSFYW = PSFYW[int(cut*nFrames):]

    # Sigma clip per data cube (also masks invalids)
    FLUX_clip  = sigma_clip(FLUX, sigma=6, iters=1)
    FERR_clip  = sigma_clip(FERR, sigma=6, iters=1)
    XDATA_clip = sigma_clip(XDATA, sigma=6, iters=1)
    YDATA_clip = sigma_clip(YDATA, sigma=3.5, iters=1)
    PSFXW_clip = sigma_clip(PSFXW, sigma=6, iters=1)
    PSFYW_clip = sigma_clip(PSFYW, sigma=3.5, iters=1)

    # Clip bad frames
    ind = np.array([])
    for i in ignore:
        ind = np.append(ind, np.arange(i, len(FLUX), nFrames))
    mask_id = np.zeros(len(FLUX))
    mask_id[ind.astype(int)] = 1
    mask_id = np.ma.make_mask(mask_id)

    # Ultimate Clipping
    MASK  = FLUX_clip.mask + XDATA_clip.mask + YDATA_clip.mask + PSFXW_clip.mask + PSFYW_clip.mask + mask_id
    #FLUX  = np.ma.masked_array(FLUX, mask=MASK)
    #XDATA = np.ma.masked_array(XDATA, mask=MASK)
    #YDATA = np.ma.masked_array(YDATA, mask=MASK)
    #PSFXW = np.ma.masked_array(PSFXW, mask=MASK)
    #PSFYW = np.ma.masked_array(PSFYW, mask=MASK)
    
    #remove bad values so that BLISS mapping will work
    FLUX  = FLUX[np.logical_not(MASK)]
    FERR  = FERR[np.logical_not(MASK)]
    TIME  = TIME[np.logical_not(MASK)]
    XDATA = XDATA[np.logical_not(MASK)]
    YDATA = YDATA[np.logical_not(MASK)]
    PSFXW = PSFXW[np.logical_not(MASK)]
    PSFYW = PSFYW[np.logical_not(MASK)]

    # normalizing the flux
    FERR  = FERR/np.ma.median(FLUX)
    FLUX  = FLUX/np.ma.median(FLUX)
    return FLUX, FERR, TIME, XDATA, YDATA, PSFXW, PSFYW

def time_sort_data(flux, flux_err, time, xdata, ydata, psfxw, psfyw, cut=0):
    """Sort the data in time and cut off any bad data at the start of the observations (e.g. ditered AOR).

    Args:
        flux (ndarray): Flux extracted for each frame.
        flux_err (ndarray): uncertainty on the flux for each frame.
        time (ndarray): Time stamp for each frame.
        xdata (ndarray): X-coordinate of the centroid for each frame.
        ydata (ndarray): Y-coordinate of the centroid for each frame.
        psfwx (ndarray): X-width of the target's PSF for each frame.
        psfwy (ndarray): Y-width of the target's PSF for each frame.

    Returns:
        tuple: flux (ndarray; Flux extracted for each frame),
            flux_err (ndarray; uncertainty on the flux for each frame),
            time (ndarray; Time stamp for each frame),
            xdata (ndarray; X-coordinate of the centroid for each frame),
            ydata (ndarray; Y-coordinate of the centroid for each frame), 
            psfwx (ndarray; X-width of the target's PSF for each frame), 
            psfwy (ndarray; Y-width of the target's PSF for each frame).

    """
    
    # sorting chronologically
    index      = np.argsort(time)
    time0      = time[index]
    flux0      = flux[index]
    flux_err0  = flux_err[index]
    xdata0     = xdata[index]
    ydata0     = ydata[index]
    psfxw0     = psfxw[index]
    psfyw0     = psfyw[index]

    # chop off dithered-calibration AOR if requested
    time       = time0[cut:]
    flux       = flux0[cut:]
    flux_err   = flux_err0[cut:]
    xdata      = xdata0[cut:]
    ydata      = ydata0[cut:]
    psfxw      = psfxw0[cut:]
    psfyw      = psfyw0[cut:]
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
        
    if 'gp' not in modeLower:
        dparams = np.append(dparams, ['gpAmp', 'gpLx', 'gpLy'])
    
    return dparams


def get_p0(function_params, fancy_names, dparams, obj):
    """Initialize the p0 variable to the defaults.

    Args:
        function_params (ndarray): Array of strings listing all parameters required by a function.
        fancy_names (ndarray): Array of fancy (LaTeX or nicely formatted) strings labelling each parameter for plots.
        dparams (ndarray): A list of strings specifying which parameters shouldn't be fit.
        obj (object): An object containing the default values for all fittable parameters. #FIX: change this to dict later

    Returns:
        tuple: p0 (ndarray; the initialized values),\
            fit_params (ndarray; the names of the fitted variables),
            fancy_labels (ndarray; the nicely formatted names of the fitted variables)
    
    """
    
    fit_params = np.array([sa for sa in function_params if not any(sb in sa for sb in dparams)])
    fancy_labels = np.array([fancy_names[i] for i in range(len(function_params)) if not any(sb in function_params[i] for sb in dparams)])
    p0 = np.zeros(len(fit_params),dtype=float)
    for i in range(len(fit_params)):
        # FIX: switch to using dictionaries to cut out this instance of eval
        p0[i] = eval('obj.'+ fit_params[i])
    return p0, fit_params, fancy_labels

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
    return dynamic_funk


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
    
    return lp + lnlike(p0, signalfunc, signal_input)

def lnprior(t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
            c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9,  c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21,
            d1, d2, d3, s1, s2, m1,
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

@jit(nopython=True, parallel=False)
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

@jit(nopython=True, parallel=False)
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

@jit(nopython=True, parallel=False)
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

# FIX - move this to make_plots.py
def triangle_colors(all_data, _data, transit_data, secondEcl_data, fname=None):
    """Make a triangle plot like figure to help look for any residual correlations in the data.

    Args:
        all_data (list): A list of the all of the xdata, ydata, psfxw, psfyw, flux, residuals.
        all_data (list): A list of the all of the xdata, ydata, psfxw, psfyw, flux, residuals.
        firstEcl_data (list): A list of the xdata, ydata, psfxw, psfyw, flux, residuals during the first eclipse.
        transit_data (list): A list of the xdata, ydata, psfxw, psfyw, flux, residuals during the transit.
        secondEcl_data (list): A list of the xdata, ydata, psfxw, psfyw, flux, residuals during the second eclipse.
        fname (string, optional): The savepath for the plot (or None if you want to return the figure instead).

    Returns:
        None
    
    """
    
    label = [r'$x_0$', r'$y_0$', r'$\sigma _x$', r'$\sigma _y$', r'$F$', r'Residuals']
    
    fig = plt.figure(figsize = (8,8))
    gs  = gridspec.GridSpec(len(all_data)-1,len(all_data)-1)
    i = 0
    for k in range(np.sum(np.arange(len(all_data)))):
        j= k - np.sum(np.arange(i+1))
        ax = fig.add_subplot(gs[i,j])
        ax.plot(all_data[j], all_data[i+1],'k.', markersize = 0.2)
        l1 = ax.plot(firstEcl_data[j], firstEcl_data[i+1],'.', color = '#66ccff', markersize = 0.7, label='$1^{st}$ secondary eclipse')
        l2 = ax.plot(transit_data[j], transit_data[i+1],'.', color = '#ff9933', markersize = 0.7, label='transit')
        l3 = ax.plot(secondEcl_data[j], secondEcl_data[i+1],'.', color = '#0066ff', markersize = 0.7, label='$2^{nd}$ secondary eclipse')
        if (j == 0):
            plt.setp(ax.get_yticklabels(), rotation = 45)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune = 'both'))
            ax.set_ylabel(label[i+1])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if (i == len(all_data)-2):
            plt.setp(ax.get_xticklabels(), rotation = 45)
            plt.axhline(y=0, color='k', linestyle='dashed')
            ax.xaxis.set_major_locator(MaxNLocator(5, prune = 'both'))
            ax.set_xlabel(label[j])
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        if(i == j):
            i += 1
    handles = [l1,l2,l3]
    
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    
    if fname is not None:
        fig.savefig(path, bbox_inches='tight')
        plt.close(fig)
        return
    else:
        return fig

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