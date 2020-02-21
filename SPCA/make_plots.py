import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.ticker import MaxNLocator
from matplotlib import gridspec

import os,sys
lib_path = os.path.abspath(os.path.join('../MCcubed/rednoise/'))
sys.path.append(lib_path)

try:
    from mc3.stats import time_avg
except ImportError:
    print('Warning: time_avg from mc3.stats failed to import. To install it, run the getMCcubed.sh script.')

# SPCA libraries
from . import bliss
from . import helpers


def plot_photometry(time0, flux0, xdata0, ydata0, psfxw0, psfyw0, 
                    time, flux, xdata, ydata, psfxw, psfyw, breaks=[], savepath=None, peritime='', showPlot=False):
    """Makes a multi-panel plot from photometry outputs.
    
    Args:
        time0 (1D array): Array of time stamps. Discarded points not removed.
        flux0 (1D array): Array of flux values for each time stamps. Discarded points not removed.
        xdata0 (1D array): Initial modelled the fluxes for each time stamps. Discarded points not removed.
        ydata0 (1D array): Initial modelled astrophysical flux variation for each time stamps. Discarded points not removed.
        psfxw0 (1D array): Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        psfyw0 (1D array): Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        time (1D array): Array of time stamps. Discarded points removed.
        flux (1D array): Array of flux values for each time stamps. Discarded points removed.
        xdata (1D array): Initial modelled the fluxes for each time stamps. Discarded points removed.
        ydata (1D array): Initial modelled astrophysical flux variation for each time stamps. Discarded points removed.
        psfxw (1D array): Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        psfyw (1D array): Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        break (1D array): Time of the breaks from one AOR to another.
        savepath (string): Path to directory where the plot will be saved
        pertime (float): Time of periapsis
    
    Returns:
        None
        
    """
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))

    axes[0].plot(time0, flux0,  'r.', markersize=1, alpha = 0.7)
    axes[0].plot(time, flux,  'k.', markersize=2, alpha = 1.0)
    axes[0].set_ylabel("Relative Flux $F$")
    axes[0].set_xlim((np.min(time0), np.max(time0)))

    axes[1].plot(time0, xdata0,  'r.', markersize=1, alpha = 0.7)
    axes[1].plot(time, xdata,  'k.', markersize=2, alpha = 1.0)
    axes[1].set_ylabel("x-centroid $x_0$")

    axes[2].plot(time0, ydata0,  'r.', markersize=1, alpha = 0.7)
    axes[2].plot(time, ydata, 'k.', markersize=2, alpha = 1.0)
    axes[2].set_ylabel("y-centroid $y_0$")

    axes[3].plot(time0, psfxw0,  'r.', markersize=1, alpha = 0.7)
    axes[3].plot(time, psfxw, 'k.', markersize=2, alpha = 1.0)
    axes[3].set_ylabel("x PSF-width $\sigma _x$")

    axes[4].plot(time0, psfyw0,  'r.', markersize=1, alpha = 0.7)
    axes[4].plot(time, psfyw,  'k.', markersize=2, alpha = 1.0)
    axes[4].set_ylabel("y PSF-width $\sigma _y$")
    axes[4].set_xlabel('Time (BMJD)')

    for i in range(5):
        for j in range(len(breaks)):
            axes[i].axvline(x=breaks[j], color ='k', alpha=0.3, linestyle = 'dashed')

    fig.subplots_adjust(hspace=0)
    
    if savepath!=None:
        pathplot = savepath + '01_Raw_data.pdf'
        fig.savefig(pathplot, bbox_inches='tight')
    
    if showPlot:
        plt.show()
        
    plt.close()
    return

def plot_centroids(xdata0, ydata0, xdata, ydata, savepath='', showPlot=False):
    """Makes a multi-panel plot from photometry outputs.
    
    Args:
        xdata0 (1D array): Initial modelled the fluxes for each time stamps. Discarded points not removed.
        ydata0 (1D array): Initial modelled astrophysical flux variation for each time stamps. Discarded points not removed.
        xdata (1D array): Initial modelled the fluxes for each time stamps. Discarded points removed.
        ydata (1D array): Initial modelled astrophysical flux variation for each time stamps. Discarded points removed.
        savepath (string): Path to directory where the plot will be saved
    
    Returns:
        None
    
    """
    
    fig = plt.figure(figsize=(6, 6))
    ax = plt.gca()

    ax.set_title('Distribution of centroids')
    
    ax.plot(xdata0, ydata0,  'r.', markersize=1, alpha = 0.7)
    ax.plot(xdata, ydata,  'k.', markersize=2, alpha = 1.0)
    ax.set_ylabel("$y$")
    ax.set_xlabel("$x$")

    fig.subplots_adjust(hspace=0)
    
    if savepath is not None:
        pathplot = savepath + 'Centroids.pdf'
        fig.savefig(pathplot, bbox_inches='tight')
    
    if showPlot:
        plt.show()
        
    plt.close()
    return


def plot_knots(xdata, ydata, flux, astroModel, knot_nrst_lin,
               tmask_good_knotNdata, knots_x, knots_y, 
               knots_x_mesh, knots_y_mesh, nBin, knotNdata, savepath=None, showPlot=False):
    '''Plot the Bliss map'''
    
    fB_avg = bliss.map_flux_avgQuick(flux, astroModel, knot_nrst_lin, nBin, knotNdata)
    delta_xo, delta_yo = knots_x[1] - knots_x[0], knots_y[1] - knots_y[0]
    
    star_colrs = knotNdata[tmask_good_knotNdata]
    
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.scatter(xdata, ydata,color=(0,0,0),alpha=0.2,s=2,marker='.')
    plt.gca().set_aspect((knots_x[-1]-knots_x[0])/(knots_y[-1]-knots_y[0]))
    plt.xlabel('Pixel x',size='x-large');
    plt.ylabel('Pixel y',size='x-large');
    plt.title('Knot Mesh',size='large')
    plt.xlim([knots_x[0] - 0.5*delta_xo, knots_x[-1] + 0.5*delta_xo])
    plt.ylim([knots_y[0] - 0.5*delta_yo, knots_y[-1] + 0.5*delta_yo])
    plt.locator_params(axis='x',nbins=8)
    plt.locator_params(axis='y',nbins=8)
    my_stars = plt.scatter(knots_x_mesh[tmask_good_knotNdata],
                           knots_y_mesh[tmask_good_knotNdata],
                           c=star_colrs, cmap=matplotlib.cm.Purples,
                           edgecolor='k',marker='*',s=175,vmin=1)
    plt.colorbar(my_stars, label='Linked Centroids',shrink=0.75)
    plt.scatter(knots_x_mesh[tmask_good_knotNdata == False],
                knots_y_mesh[tmask_good_knotNdata == False], 
                color=(1,0.75,0.75), marker='x',s=35)
    legend = plt.legend(('Centroids','Good Knots','Bad Knots'),
                        loc='lower right',bbox_to_anchor=(0.975,0.025),
                        fontsize='small',fancybox=True)
    legend.legendHandles[1].set_color(matplotlib.cm.Purples(0.67)[:3])
    legend.legendHandles[1].set_edgecolor('black')

    if savepath!=None:
        pathplot = savepath+'BLISS_Knots.pdf'
        plt.savefig(pathplot, bbox_inches='tight')

    if showPlot:
        plt.show()
    
    plt.close()
    return

def plot_init_guess(time, data, astro, detec_full, savepath=None):
    """Makes a multi-panel plots for the initial light curve guesses.
    
    Args:
        time (1D array): Time stamps.
        data (1D array): Flux values for each time stamps.
        astro (1D array): Initial modelled astrophysical flux variation for each time stamps.
        detec_full (1D array): Initial modelled flux variation due to the detector for each time stamps.
        savepath (str): Path to directory where the plot will be saved.
    
    Returns:
        None
    
    """
    
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,9))
    
    axes[0].plot(time, data, '.', label='data')
    axes[0].plot(time, astro*detec_full, '.', label='guess')
    
    axes[1].plot(time, data/detec_full, '.', label='Corrected')
    axes[1].plot(time, astro, '.', label='Astrophysical')
    
    axes[2].plot(time, data/detec_full, '.', label='Corrected')
    axes[2].plot(time, astro, '.', label='Astrophysical')
    axes[2].set_ylim(0.998, 1.005)
    
    axes[3].plot(time, data/detec_full-astro, '.', label='residuals')
    axes[3].axhline(y=0, linewidth=2, color='black')
    
    axes[0].set_ylabel('Relative Flux')
    axes[2].set_ylabel('Relative Flux')
    axes[2].set_xlabel('Time (BMJD)')
    
    axes[0].legend(loc=3)
    axes[1].legend(loc=3)
    axes[2].legend(loc=3)
    axes[3].legend(loc=3)
    axes[3].set_xlim(np.min(time), np.max(time))
    
    fig.subplots_adjust(hspace=0)
    if savepath is not None:
        pathplot = savepath + '02_Initial_Guess.pdf'
        fig.savefig(pathplot, bbox_inches='tight')
        
    if showPlot:
        plt.show()
        
    plt.close()
    return

def walk_style(ndim, nwalk, samples, interv, subsamp, labels, fname=None, showPlot=False):
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
    
    if showPlot:
        plt.show()
    
    plt.close()
    return
    
# FIX - add docstring for this function
def plot_bestfit(p0_mcmc, time, flux, mode, p0_obj, p0_astro, p0_labels, signal_inputs, astrofunc, signalfunc,
                 breaks, savepath, plotTrueAnomaly=False, nbin=None, showPlot=False, fontsize=24,plot_peritime=False):
    
    ind_a = len(p0_astro) # index where the astro params end

    # generate the models from best-fit parameters
    mcmc_signal = signalfunc(signal_inputs, **dict([[p0_labels[i], p0_mcmc[i]] for i in range(len(p0_labels))]))
    astro = astrofunc(time, **dict([[p0_astro[i], p0_mcmc[:ind_a][i]] for i in range(len(p0_astro))]))
    detec = mcmc_signal/astro

#     time2 = np.linspace(np.min(time), np.max(time), 1000)
    
#     #for higher-rez red curve
#     astro  = astrofunc(time2, **dict([[p0_astro[i], p0_mcmc[:ind_a][i]] for i in range(len(p0_astro))]))
    
    if plotTrueAnomaly:
        # FIX: convert time to true anomaly for significantly eccentric planets!!
        # Use p0_mcmc if there, otherwise p0_obj
        x = time
#         x2 = time2
    else:
        x = time
#         x2 = time2
    
    if nbin is not None:
        x_binned, _ = helpers.binValues(x, x, nbin)
        calibrated_binned, calibrated_binned_err = helpers.binValues(flux/detec, x, nbin, assumeWhiteNoise=True)
        residuals_binned, residuals_binned_err = helpers.binValues(flux/detec-astro, x, nbin, assumeWhiteNoise=True)
    
    fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize=(8, 14))
    
    axes[0].set_xlim(np.nanmin(x), np.nanmax(x))
    axes[0].plot(x, flux, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[0].plot(x, astro*detec, '.', color = 'r', markersize = 2.5, alpha = 0.4)
    axes[0].set_ylabel('Raw Flux', fontsize=fontsize)

    axes[1].plot(x, flux/detec, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[1].plot(x, astro, color = 'r', linewidth=2)
    if nbin is not None:
        axes[1].errorbar(x_binned, calibrated_binned, yerr=calibrated_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[1].set_ylabel('Calibrated Flux', fontsize=fontsize)
    
    axes[2].axhline(y=1, color='k', linewidth = 2, linestyle='dashed', alpha = 0.5)
    axes[2].plot(x, flux/detec, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[2].plot(x, astro, color = 'r', linewidth=2)
    if nbin is not None:
        axes[2].errorbar(x_binned, calibrated_binned, yerr=calibrated_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[2].set_ylabel('Calibrated Flux', fontsize=fontsize)
    axes[2].set_ylim(ymin=1-3*np.nanstd(flux/detec - astro))

    axes[3].plot(x, flux/detec - astro, 'k.', markersize = 4, alpha = 0.15)
    axes[3].axhline(y=0, color='r', linewidth = 2)
    if nbin is not None:
        axes[3].errorbar(x_binned, residuals_binned, yerr=residuals_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[3].set_ylabel('Residuals', fontsize=fontsize)
    axes[3].set_xlabel('Time (BMJD)', fontsize=fontsize)

    if plot_peritime:
        # FIX - compute peritime
        print('Have not yet implemented this!')
        
        for i in range(len(axes)):
            axes[i].xaxis.set_tick_params(labelsize=fontsize)
            axes[i].yaxis.set_tick_params(labelsize=fontsize)
            axes[i].axvline(x=peritime, color ='C1', alpha=0.8, linestyle = 'dashed')
            for j in range(len(breaks)):
                axes[i].axvline(x=(breaks[j]), color ='k', alpha=0.3, linestyle = 'dashed')
    
    fig.align_ylabels()
    
    fig.subplots_adjust(hspace=0)
    
    if savepath is not None:
        plotname = savepath + 'MCMC_'+mode+'_2.pdf'
        fig.savefig(plotname, bbox_inches='tight')
        
    if showPlot:
        plt.show()
    
    plt.close()
    return

def plot_rednoise(residuals, minbins, ingrDuration, occDuration, intTime, mode, savepath=None, showPlot=True, showtxt=True, savetxt=False, fontsize=10):
    
    maxbins = int(np.rint(residuals.size/minbins))
    
    try:
        rms, rmslo, rmshi, stderr, binsz = time_avg(residuals, maxbins)
    except:
        rms = []
        for i in range(minbins,len(residuals)):
            rms.append(helpers.binnedNoise(np.arange(len(residuals)),residuals,i))
        rms = np.array(rms)[::-1]

        binsz = len(residuals)/np.arange(minbins,len(residuals))[::-1]

        #In case there is a NaN or something while binning
        binsz = binsz[np.isfinite(rms)]
        rms = rms[np.isfinite(rms)]
        rmslo = np.zeros_like(rms)
        rmshi = rmslo
        stderr = np.std(residuals)/np.sqrt(binsz)
    
    plt.clf()
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.errorbar(binsz, rms, yerr=[rmslo, rmshi], fmt="k-", ecolor='0.5', capsize=0, label="Data RMS")
    ax.plot(binsz, stderr, c='red', label="Gaussian std.")
    ylim = ax.get_ylim()
    ax.plot([ingrDuration,ingrDuration],ylim, color='black', ls='--', alpha=0.6)
    ax.plot([occDuration,occDuration],ylim, color='black', ls='-.', alpha=0.6)
    ax.set_ylim(ylim)
    
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    
    plt.xlabel(r'N$_{\rm binned}$', fontsize=fontsize)
    plt.ylabel('RMS', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    if savepath is not None:
        plotname = savepath + 'MCMC_'+mode+'_RedNoise.pdf'
        plt.savefig(plotname, bbox_inches='tight')
    if showplot:
        plt.show()
        
    plt.clf()
    plt.close()
    
    #Ingress Duration
    sreal = rms[np.where(binsz<=ingrDuration)[0][-1]]*1e6
    s0 = stderr[np.where(binsz<=ingrDuration)[0][-1]]*1e6
    outStr = 'Over Ingress ('+str(round(ingrDuration*intTime*24*60, 1))+' min):\n'
    outStr += 'Expected Noise (ppm)\t'+'Observed Noise (ppm)\n'
    outStr += str(s0)+'\t'+str(sreal)+'\n'
    outStr += 'Observed/Expected\n'
    outStr += str(sreal/s0)+'\n\n'
    #Occultation Duration
    sreal = rms[np.where(binsz<=occDuration)[0][-1]]*1e6
    s0 = stderr[np.where(binsz<=occDuration)[0][-1]]*1e6
    outStr += 'Over Transit/Eclipse ('+str(round(occDuration*intTime*24*60, 1))+' min):\n'
    outStr += 'Expected Noise (ppm)\t'+'Observed Noise (ppm)\n'
    outStr += str(s0)+'\t'+str(sreal)+'\n'
    outStr += 'Observed/Expected\n'
    outStr += str(sreal/s0)

    if showtxt:
        print(outStr)
    if savetxt:
        fname = savepath + 'MCMC_'+mode+'_RedNoise.txt'
        with open(fname,'w') as file:
            file.write(outStr)
    
    return

def triangle_colors(all_data, firstEcl_data, transit_data, secondEcl_data, fname=None, showPlot=False):
    """Make a triangle plot like figure to help look for any residual correlations in the data.

    Args:
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
        fig.savefig(fname, bbox_inches='tight')
    
    if showPlot:
        plt.show()
    
    plt.close()
    return

