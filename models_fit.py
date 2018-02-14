import numpy as np
import batman
import scipy
import scipy.stats as sp
import scipy.optimize as spopt

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

def get_data(folderdata, filename):
	'''
	Retrieve binned data

	Parameters
	----------

	foldername : string object
		Path to the directory containing all the Spitzer data.

	filename   : string object
		filename of the data file output by photometry routine
		e.g. ch2_datacube_binned_AORs579.dat

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

	path = folderdata + '/' + filename

	#Loading Data
	flux     = np.loadtxt(path, usecols=[0], skiprows=1)     # MJr/str
	flux_err = np.loadtxt(path, usecols=[1], skiprows=1)     # MJr/str
	time     = np.loadtxt(path, usecols=[2], skiprows=1)     # BMJD
	xdata    = np.loadtxt(path, usecols=[4], skiprows=1)     # pixel
	ydata    = np.loadtxt(path, usecols=[6], skiprows=1)     # pixel
	psfxwdat = np.loadtxt(path, usecols=[8], skiprows=1)     # pixel
	psfywdat = np.loadtxt(path, usecols=[10], skiprows=1)    # pixel

	factor   = 1/(np.median(flux))
	flux     = factor*flux
	flux_err = factor*flux
	return flux, flux_err, time, xdata, ydata, psfxwdat, psfywdat

def get_full_data(folderdata, filename):
    path = folderdata + '/' + filename
    #Loading Data
    flux  = np.loadtxt(path, usecols=[0], skiprows=1)         # photon count
    time  = np.loadtxt(path, usecols=[2], skiprows=1)         # hours
    xdata = np.loadtxt(path, usecols=[3], skiprows=1)         # hours
    ydata = np.loadtxt(path, usecols=[4], skiprows=1)         # hours
    # scaling
    #factor = 1/(np.median(flux))
    #flux = factor*flux
    #flux_err = factor*flux
    return flux, time, xdata, ydata


def phase_variation(anom, w, A, B, C=0, D=0):
	'''
	Params
	------
	anom  : 1D array
		True anomaly (obtained from batman)
	w     : float
		Longitude of periastrion in RADIAN
	A     : float
		Phase varation coefficient
	B     : float
		Phase varation coefficient
	return
	------
	phase : 1D array
		phase variation
	'''
	theta = anom + w + np.pi/2
	phase = 1 + A*(np.cos(theta)-1) + (B*np.sin(theta)) + C*(np.cos(2*theta)-1) + (D*np.sin(2*theta))
	return phase

def transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2):
	params     = batman.TransitParams()                   #object to store transit parameters
	params.t0  = t0                                       #time of inferior conjunction
	params.per = per                                      #orbital period
	params.rp  = rp                                       #planet radius (in units of stellar radii)
	params.a   = a                                        #semi-major axis (in units of stellar radii)
	params.inc = inc                                      #orbital inclination (in degrees)
	params.ecc = ecc                                      #eccentricity
	params.w   = np.rad2deg(w)                            #longitude of periastron (in degrees)
	params.limb_dark = "quadratic"                        #limb darkening model
	params.u   = [u1, u2]                                 #limb darkening coefficients

	m          = batman.TransitModel(params, time)        #initializes model
	flux       = m.light_curve(params)
	t_sec      = m.get_t_secondary(params)
	anom       = m.get_true_anomaly()
	return flux, t_sec, anom

def eclipse(time, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec):
	params     = batman.TransitParams()                   #object to store transit parameters
	params.t0  = t0                                       #time of inferior conjunction
	params.per = per                                      #orbital period
	params.rp  = rp                                       #planet radius (in units of stellar radii)
	params.a   = a                                        #semi-major axis (in units of stellar radii)
	params.inc = inc                                      #orbital inclination (in degrees)
	params.ecc = ecc                                      #eccentricity
	params.w   = np.rad2deg(w)                            #longitude of periastron (in degrees)
	params.limb_dark = "quadratic"                        #limb darkening model
	params.u   = [u1, u2]                                 #limb darkening coefficients
	params.fp  = fp                                       #planet/star brightnes
	params.t_secondary = t_sec
    
	m          = batman.TransitModel(params, time, transittype="secondary")  #initializes model
	flux       = m.light_curve(params)
	return flux

def fplanet_model(time, anom, A, B, C, D, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec):
	phase = phase_variation(anom, w, A, B, C, D)
	eclip = eclipse(time, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec)
	flux  = phase*(eclip - 1)
	return flux

def phase_curve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C=0, D=0):
	# retransformation
	ecc = np.sqrt(ecosw**2 + esinw**2)
	if (ecosw <= 0):
		w = np.arctan(esinw/ecosw) + np.pi                          # this is in radians!!
	else:
		w = np.arctan(esinw/ecosw)
	u1  = 2*np.sqrt(q1)*q2
	u2  = np.sqrt(q1)*(1-2*q2)
	# create transit first and use orbital paramater to get time of superior conjunction
	transit, t_sec, anom = transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2)
	# create light curve of the planet
	fplanet = fplanet_model(time, anom, A, B, C, D, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec)
	# add both light curves
	f_total = transit + fplanet
	return f_total

def detec_poly(input_dat, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
	c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0, order=2):
	xdata, ydata, mid_x, mid_y = input_dat
	x = xdata - mid_x
	y = ydata - mid_y
	if (order ==2):
		det_sens = np.array(c1 + 
			c2*x     + c3*y       + 
			c4*x**2  + c5*x*y     + c6*y**2)
	elif(order==3):
		det_sens = np.array(c1 + 
			c2*x     + c3*y       + 
			c4*x**2  + c5*x*y     + c6*y**2       + 
			c7*x**3  + c8*x**2*y  + c9*x*y**2     + c10*y**3)
	elif(order==4):
		det_sens = np.array(c1 + 
			c2*x     + c3*y       + 
			c4*x**2  + c5*x*y     + c6*y**2       + 
			c7*x**3  + c8*x**2*y  + c9*x*y**2     + c10*y**3      + 
			c11*x**4 + c12*x**3*y + c13*x**2*y**2 + c14*x*y**3    + c15*y**4)
	elif(order==5):
		det_sens = np.array(c1 + 
			c2*x     + c3*y       + 
			c4*x**2  + c5*x*y     + c6*y**2       + 
			c7*x**3  + c8*x**2*y  + c9*x*y**2     + c10*y**3      + 
			c11*x**4 + c12*x**3*y + c13*x**2*y**2 + c14*x*y**3    + c15*y**4   + 
			c16*x**5 + c17*x**4*y + c18*x**3*y**2 + c19*x**2*y**3 + c20*x*y**4 + c21*y**5)
	return np.array(det_sens)

def signal_poly(time, per, t0, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B,
	       xdata, ydata, mid_x, mid_y, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
	       c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0, C=0, D=0):
	pcurve = phase_curve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D)
	detect = detec_poly((xdata, ydata, mid_x, mid_y), c1, c2, c3, c4, c5, c6,c7, c8, c9, c10, 
		c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21)
	return pcurve*detect

def walk_style(ndim, nwalk, samples, interv, subsamp, labels):
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
    #plt.subplot(nrows, ncols, ind)
    #fig.tight_layout()
    plt.figure(figsize = (15, 2*nrows))
    for ind in range(ndim):
        plt.subplot(nrows, ncols, ind+1)
        # get indices for subplots
        #i = int(ind/ncols) # row number
        #j = ind % ncols    # col number
        # get mean and standard deviation
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
    return    

def get_chi2(data, fit, err):
    N     = len(data)
    denom = (data - fit)**2
    numer = err**2
    chi2  = np.sum(denom/numer)
    return chi2

def get_lnlike(data, fit, err):
    inv_sigma2 = 1.0/(err**2)
    return -0.5*(np.sum((data-fit)**2*inv_sigma2 - np.log(inv_sigma2)))

def get_BIC(logL, Npar, Ndat):
    BIC = Npar*np.log(Ndat) - 2*logL
    return BIC

def triangle_colors(data1, data2, data3, data4, label):
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
    #fig.legend(handles, ['$1^{st}$ secondary eclipse', 'transit', '$2^{nd}$ secondary eclipse'],bbox_to_anchor = [0.5, -0.05], loc = 'upper center')
    #fig.colorbar()
    return