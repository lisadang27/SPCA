import numpy as np
import scipy.optimize as spopt
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

import astro_models

def get_data(path):
    '''
    Retrieve binned data
    
    Parameters
    ----------
    
    path : string object
        Full path to the data file output by photometry routine

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
    path = foldername + filename
    #Loading Data
    flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # mJr/str
    flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # mJr/str
    time     = np.loadtxt(path, usecols=[2], skiprows=1)     # hours
    xdata    = np.loadtxt(path, usecols=[3], skiprows=1)     # hours
    ydata    = np.loadtxt(path, usecols=[4], skiprows=1)     # hours
    
    return flux, flux_err, time, xdata, ydata

def detec_model_poly(input_dat, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
                     c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0):
    
    xdata, ydata, mid_x, mid_y, mode = input_dat
    x = xdata - mid_x
    y = ydata - mid_y
    
    if 'Poly2' in mode:
        pos = np.vstack((np.ones_like(x),x,y,x**2,x*y,y**2))
        detec = np.array([c1, c2, c3, c4, c5, c6])
    elif 'Poly3' in mode:
        pos = np.vstack((np.ones_like(x),x,y,x**2,x*y,y**2,
                        x**3,y*x**2,x*y**2,y**3))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,])
    elif 'Poly4' in mode:
        pos = np.vstack((np.ones_like(x),x,y,x**2,x*y,y**2,
                        x**3,y*x**2,x*y**2,y**3,
                        x**4,x**3*y,y**2*x**2,x*y**3,y**4))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15])
    elif 'Poly5' in mode:
        pos = np.vstack((np.ones_like(x),x,y,x**2,x*y,y**2,
                        x**3,y*x**2,x*y**2,y**3,
                        x**4,x**3*y,y**2*x**2,x*y**3,y**4,
                        x**5,x**4*y,x**3*y**2,x**2*y**3,x*y**4,y**5))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15,
                          c16, c17, c18, c19, c20, c21])

    return np.dot(detec[np.newaxis,:], pos).reshape(-1)

def signal(time, xdata, ydata, mid_x, mid_y, per, p0, mode):
    if 'Poly' in mode:
        input_data = (xdata, ydata, mid_x, mid_y, mode)
        start = 11
        if 'ellipsoid' in mode:
            start += 1
        elif 'v2' in mode:
            start += 2
        p0_detec = p0[start:-1]
        return astro_models.ideal_lightcurve(time, p0, per, mode)*detec_model_poly(input_data, *p0_detec)
    else:
        print('CANNOT DEAL WITH NON-POLY DETECTOR MAPS YET!!!')
        return np.zeros_like(time)

def lnprior(p0, initial, errs, time, per, mode):
    t0, rp, a, inc, ecosw, esinw, q1, q2, fp = p0[:9]
    t0_err, rp_err, a_err, inc_err = errs
    check = astro_models.check_phase(time, p0, per, mode)
    lgpri_t0 = -0.5*(((t0 - initial[0])/t0_err)**2.0)
    lgpri_rp = 0 # -0.5*(((rp - initial[1])/(rp_err))**2.0)
    lgpri_a = -0.5*(((a - initial[2])/a_err)**2.0)
    lgpri_i = -0.5*(((inc - initial[3])/inc_err)**2.0)
    # uniform prior for the rest
    if ((0 < fp < 1) and (0 < q1 < 1) and (0 < q2 < 1) and 
        (-1 < ecosw < 1) and (-1 < esinw < 1) and (check == False)):
        return 0.0 + lgpri_t0 + lgpri_rp + lgpri_a + lgpri_i
    else:
        return -np.inf

def lnprob(p0, time, flux, xdata, ydata, mid_x, mid_y, per, initial, errs, mode):
    lp = lnprior(p0, initial, errs, time, per, mode)
    if not np.isfinite(lp):
        return -np.inf
    model = signal(time, xdata, ydata, mid_x, mid_y, per, p0, mode)
    flux_err = p0[-1]
    lnlike = -0.5*np.sum((flux-model)**2)/(flux_err**2) - len(flux)*np.log(flux_err)
    if not np.isfinite(lnlike):
        return -np.inf
    return lp + lnlike

#Initial ML fit to detector parameters
def fit_residual(flux, time, xdata, ydata, p0, u_detec, l_detec, per, mode):
    # Use initial LC parameters
    lcurve = astro_models.ideal_lightcurve(time, p0, per, mode)
    # Get residuals
    residuals = flux/lcurve
    # Fit residual
    coord = (xdata, ydata)
    start = 11
    if 'ellipsoid' in mode:
        start += 1
    elif 'v2' in mode:
        start += 2
    order = int(mode[mode.find('Poly')+4])
    if order == 2:
        func = det4fit_Poly2
    elif order == 3:
        func = det4fit_Poly3
    elif order == 4:
        func = det4fit_Poly4
    elif order ==5:
        func = det4fit_Poly5
    p0_detec = p0[start:-1]
    popt, pcov = spopt.curve_fit(func, coord, residuals, p0=p0_detec, bounds = (l_detec, u_detec))
    return popt

def walk_style(ndim, nwalk, samples, interv, subsamp, labels, fname=None):
    '''
    input:
        ndim    = number of free parameters
        nwalk   = number of walkers
        samples = samples chain
        interv  = take every 'interv' element to thin out the plot
        subsamp = only show the last 'subsamp' steps
    '''
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
        mu_param = np.mean(samples[:,:,ind][:,beg:end:interv], axis = 0)
        std_param = np.std(samples[:,:,ind][:,beg:end:interv], axis = 0)
        plt.plot(step, mu_param)
        plt.fill_between(step, mu_param + 3*std_param, mu_param - 3*std_param, facecolor='k', alpha = 0.1)
        plt.fill_between(step, mu_param + 2*std_param, mu_param - 2*std_param, facecolor='k', alpha = 0.1)
        plt.fill_between(step, mu_param + 1*std_param, mu_param - 1*std_param, facecolor='k', alpha = 0.1)
        plt.title(labels[ind])
        plt.xlim(np.min(step), np.max(step))
        if ind < (ndim - ncols):
            plt.xticks([])
        else: 
            plt.xticks(rotation=25)
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    plt.show()
    return    

def chi2(data, fit, err):
    return np.sum(((data - fit)/err)**2)

def loglikelihood(data, fit, err):
    return -0.5*chi2(data, fit, err) - np.sum(np.log(err)) #sum acts as N

def BIC(logL, Npar, Ndat):
    return logL - (Npar/2)*np.log(Ndat)

def triangle_colors(data1, data2, data3, data4, label, path):
    fig = plt.figure(figsize = (8,8))
    gs  = gridspec.GridSpec(len(data1)-1,len(data1)-1)
    i = 0
    for k in range(np.sum(np.arange(len(data1)))):
        j= k - np.sum(np.arange(i+1))
        ax = fig.add_subplot(gs[i,j])
        ax.plot(data1[j], data1[i+1],'k.', markersize = 0.2)
        l1 = ax.plot(data2[j], data2[i+1],'.', color = '#66ccff', markersize = 0.7, label='$1^{st}$ secondary eclipse')
        l2 = ax.plot(data3[j], data3[i+1],'.', color = '#ff9933', markersize = 0.7, label='transit')
        l3 = ax.plot(data4[j], data4[i+1],'.', color = '#0066ff', markersize = 0.7, label='$2^{nd}$ secondary eclipse')
        if (j == 0):
            plt.setp(ax.get_yticklabels(), rotation = 45)
            ax.yaxis.set_major_locator(MaxNLocator(5, prune = 'both'))
            ax.set_ylabel(label[i+1])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if (i == len(data1)-2):
            plt.setp(ax.get_xticklabels(), rotation = 45)
            plt.axhline(y=0, color='k', linestyle='dashed')
            #ax.plot(bins[j], res[j], '.', color='#ff5050', markersize = 3)  # plot bin residual
            ax.xaxis.set_major_locator(MaxNLocator(5, prune = 'both'))
            ax.set_xlabel(label[j])
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        if(i == j):
            i += 1
    handles = [l1,l2,l3]
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    fig.savefig(path, bbox_inches='tight')
    return