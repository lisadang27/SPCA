# Some code to make plots look a bit nicer
import matplotlib
from distutils.spawn import find_executable
fontsize = 20
ticklen = 5
params = {
    'font.size': fontsize,
    'axes.titlesize': fontsize,
    'xtick.labelsize': fontsize,
    'ytick.labelsize': fontsize,
    'axes.labelsize': fontsize,
    'legend.fontsize': fontsize,
    'xtick.major.size' : ticklen,
    'ytick.major.size' : ticklen,
    'xtick.minor.size' : ticklen/2,
    'ytick.minor.size' : ticklen/2
}
if find_executable('latex'):
    params['pgf.texsystem'] = 'xelatex'
    params['pgf.preamble'] = r'''\usepackage{fontspec}
    \setmainfont{Linux Libertine O}'''
    params['text.usetex'] = True
    params['pgf.rcfonts'] = False
matplotlib.rcParams.update(params)

# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.ticker import MaxNLocator, ScalarFormatter
from matplotlib import gridspec
import pickle

import os,sys
lib_path = os.path.abspath(os.path.join('../MCcubed/rednoise/'))
sys.path.append(lib_path)

from mc3.stats import time_avg

# SPCA libraries
from . import bliss, helpers, astro_models


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

    axes[0].plot(time0-5e4, flux0,  'r.', markersize=1, alpha = 0.7)
    axes[0].plot(time-5e4, flux,  'k.', markersize=2, alpha = 1.0)
    axes[0].set_ylabel(r'$\rm Relative~Flux$')
    axes[0].set_xlim((np.nanmin(time0-5e4), np.nanmax(time0-5e4)))

    axes[1].plot(time0-5e4, xdata0,  'r.', markersize=1, alpha = 0.7)
    axes[1].plot(time-5e4, xdata,  'k.', markersize=2, alpha = 1.0)
    axes[1].set_ylabel(r'$x_0$')

    axes[2].plot(time0-5e4, ydata0,  'r.', markersize=1, alpha = 0.7)
    axes[2].plot(time-5e4, ydata, 'k.', markersize=2, alpha = 1.0)
    axes[2].set_ylabel(r'$y_0$')

    axes[3].plot(time0-5e4, psfxw0,  'r.', markersize=1, alpha = 0.7)
    axes[3].plot(time-5e4, psfxw, 'k.', markersize=2, alpha = 1.0)
    axes[3].set_ylabel(r'$\sigma_x$')

    axes[4].plot(time0-5e4, psfyw0,  'r.', markersize=1, alpha = 0.7)
    axes[4].plot(time-5e4, psfyw,  'k.', markersize=2, alpha = 1.0)
    axes[4].set_ylabel(r'$\sigma_y$')
    axes[4].set_xlabel(r'$\rm Time~(BJD-2450000.5)$')

    for i in range(5):
        for j in range(len(breaks)):
            axes[i].axvline(x=breaks[j], color ='k', alpha=0.3, linestyle = 'dashed')

    fig.subplots_adjust(hspace=0)
    fig.align_ylabels()
    
    if savepath!=None:
        # saving plot as pdf
        pathplot = savepath + '01_Raw_data.pdf'
        fig.savefig(pathplot, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: Time, Flux, xdata, ydata, psfwx, psfwy, Time0, Flux0, xdata0, ydata0, psfwx0, psfwy0'
        data = [header, time0, flux0, xdata0, ydata0, psfxw0, psfyw0, time, flux, xdata, ydata, psfxw, psfyw]
        pathdata = savepath + '01_Raw_data.pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
    
    if showPlot:
        plt.show()
        
    plt.close()
    return fig

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
        # saving plot as pdf
        pathplot = savepath + 'Centroids.pdf'
        fig.savefig(pathplot, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: xdata0, ydata0, xdata, ydata'
        data = [header, xdata0, ydata0, xdata, ydata]
        pathdata = savepath + 'Centroids.pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    if showPlot:
        plt.show()
        
    plt.close()
    return


def plot_knots(xdata, ydata, 
               tmask_good_knotNdata, knots_x, knots_y, 
               knots_x_mesh, knots_y_mesh, knotNdata, savepath=None, showPlot=False, fontsize=16):
    '''Plot the Bliss map'''
    
    delta_xo, delta_yo = knots_x[1] - knots_x[0], knots_y[1] - knots_y[0]
    
    star_colrs = knotNdata[tmask_good_knotNdata]
    
    plt.figure(figsize=(12,6))

    plt.subplot(121)
    plt.scatter(xdata, ydata, color=(0,0,0), alpha=0.2, s=2, marker='.')
    plt.gca().set_aspect((knots_x[-1]-knots_x[0])/(knots_y[-1]-knots_y[0]))
    plt.xlabel(r'${\rm Pixel}~x$', fontsize=fontsize);
    plt.ylabel(r'${\rm Pixel}~y$', fontsize=fontsize);
    plt.title(r'$\rm Knot~Mesh$',size=fontsize)
    plt.xlim([knots_x[0] - 0.5*delta_xo, knots_x[-1] + 0.5*delta_xo])
    plt.ylim([knots_y[0] - 0.5*delta_yo, knots_y[-1] + 0.5*delta_yo])
    plt.gca().xaxis.set_tick_params(labelsize=fontsize*0.8)
    plt.gca().yaxis.set_tick_params(labelsize=fontsize*0.8)
    plt.locator_params(axis='x',nbins=8)
    plt.locator_params(axis='y',nbins=8)
    my_stars = plt.scatter(knots_x_mesh[tmask_good_knotNdata],
                           knots_y_mesh[tmask_good_knotNdata],
                           c=star_colrs, cmap=matplotlib.cm.Purples,
                           edgecolor='k',marker='*',s=175,vmin=1)
    cb = plt.colorbar(my_stars, shrink=0.75)
    cb.set_label(label=r'$\rm Linked~Centroids$', size=fontsize)
    plt.scatter(knots_x_mesh[tmask_good_knotNdata == False],
                knots_y_mesh[tmask_good_knotNdata == False], 
                color=(1,0.75,0.75), marker='x',s=35)
    legend = plt.legend((r'$\rm Centroids$',r'$\rm Good~Knots$',r'$\rm Bad~Knots$'),
                        loc='lower right',bbox_to_anchor=(0.975,0.025),
                        fontsize=fontsize*0.75,fancybox=True)
    legend.legendHandles[1].set_color(matplotlib.cm.Purples(0.67)[:3])
    legend.legendHandles[1].set_edgecolor('black')

    if savepath!=None:
        pathplot = savepath+'BLISS_Knots.pdf'
        plt.savefig(pathplot, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: xdata, ydata, tmask_good_knotNdata, knots_x, knots_y, knots_x_mesh, knots_y_mesh, knotNdata'
        data = [header, xdata, ydata, tmask_good_knotNdata, knots_x, knots_y, knots_x_mesh, knots_y_mesh, knotNdata]
        pathdata = savepath + 'BLISS_Knots.pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    if showPlot:
        plt.show()
    
    plt.close()
    return

def walk_style(chain, labels, interv=10, fname=None, showPlot=False, fontsize=15):
    """Make a plot showing the evolution of the walkers throughout the emcee sampling.

    Args:
        chain (ndarray): The ndarray accessed by calling sampler.chain when using emcee
        labels (ndarray): The fancy labels for each dimension
        interv (int): Take every 'interv' element to thin out the plot
        name (string, optional): The savepath for the plot (or None if you want to return the figure instead).
        showPlot (bool, optional): Whether or not you want to show the plotted figure.

    Returns:
        None
    
    """
    
    nwalk = chain.shape[0]
    ndim = chain.shape[-1]
    
    # get first index
    beg   = 0
    end   = len(chain[0,:,0]) 
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
        neg3sig, neg2sig, neg1sig, mu_param, pos1sig, pos2sig, pos3sig = np.percentile(chain[:,:,ind][:,beg:end:interv],
                                                                                       percentiles, axis=0)
        plt.plot(step, mu_param)
        plt.fill_between(step, pos3sig, neg3sig, facecolor='k', alpha = 0.1)
        plt.fill_between(step, pos2sig, neg2sig, facecolor='k', alpha = 0.1)
        plt.fill_between(step, pos1sig, neg1sig, facecolor='k', alpha = 0.1)
        plt.title(labels[ind], fontsize=fontsize)
        plt.xlim(np.min(step), np.max(step))
        if ind < (ndim - ncols):
            plt.xticks([])
        else: 
            plt.xticks(rotation=25)
        
        y_formatter = ScalarFormatter(useOffset=False)
        plt.gca().yaxis.set_major_formatter(y_formatter)
        plt.gca().xaxis.set_tick_params(labelsize=fontsize*0.8)
        plt.gca().yaxis.set_tick_params(labelsize=fontsize*0.8)        
    
    if fname != None:
        plt.savefig(fname, bbox_inches='tight')
    
    if showPlot:
        plt.show()
    
    plt.close()
    return
    
# FIX - add docstring for this function
def plot_model(time, flux, astro, detec, breaks, savepath=None, plotName='Initial_Guess.pdf',
               plotTrueAnomaly=False, nbin=None, showPlot=False, fontsize=20, plot_peritime=False):
    
    mcmc_signal = astro*detec
    
    if plotTrueAnomaly:
        # FIX: convert time to true anomaly for significantly eccentric planets!!
        # Use p0_mcmc if there, otherwise p0_obj
        x = time
    else:
        x = time-5e4
    
    if nbin is not None:
        x_binned, _ = helpers.binValues(x, x, nbin)
        calibrated_binned, calibrated_binned_err = helpers.binValues(flux/detec, x, nbin, assumeWhiteNoise=True)
        residuals_binned, residuals_binned_err = helpers.binValues(flux/detec-astro, x, nbin, assumeWhiteNoise=True)
    
    fig, axes = plt.subplots(ncols = 1, nrows = 4, sharex = True, figsize=(8, 14))
    
    axes[0].set_xlim(np.nanmin(x), np.nanmax(x))
    axes[0].plot(x, flux, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[0].plot(x, astro*detec, '.', color = 'r', markersize = 2.5, alpha = 0.4)
    axes[0].set_ylabel(r'$\rm Raw~Flux$', fontsize=fontsize)

    axes[1].plot(x, flux/detec, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[1].plot(x, astro, color = 'r', linewidth=2)
    if nbin is not None:
        axes[1].errorbar(x_binned, calibrated_binned, yerr=calibrated_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[1].set_ylabel(r'$\rm Calibrated~Flux$', fontsize=fontsize)
    
    axes[2].axhline(y=1, color='k', linewidth = 2, linestyle='dashed', alpha = 0.5)
    axes[2].plot(x, flux/detec, '.', color = 'k', markersize = 4, alpha = 0.15)
    axes[2].plot(x, astro, color = 'r', linewidth=2)
    if nbin is not None:
        axes[2].errorbar(x_binned, calibrated_binned, yerr=calibrated_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[2].set_ylabel(r'$\rm Calibrated~Flux$', fontsize=fontsize)
    axes[2].set_ylim(ymin=1-3*np.nanstd(flux/detec - astro), ymax=np.max(astro)+3*np.nanstd(flux/detec - astro))

    axes[3].plot(x, flux/detec - astro, 'k.', markersize = 4, alpha = 0.15)
    axes[3].axhline(y=0, color='r', linewidth = 2)
    if nbin is not None:
        axes[3].errorbar(x_binned, residuals_binned, yerr=residuals_binned_err, fmt='.',
                         color = 'blue', markersize = 10, alpha = 1)
    axes[3].set_ylabel(r'$\rm Residuals$', fontsize=fontsize)
    axes[3].set_xlabel(r'$\rm Time~(BJD-2450000.5)$', fontsize=fontsize)

    for i in range(len(axes)):
        axes[i].xaxis.set_tick_params(labelsize=fontsize*0.8)
        axes[i].yaxis.set_tick_params(labelsize=fontsize*0.8)
    
    if plot_peritime:
        # FIX - compute peritime
        print('Have not yet implemented this!')
        
        for i in range(len(axes)):
            axes[i].axvline(x=peritime, color ='C1', alpha=0.8, linestyle = 'dashed')
    
    for i in range(len(axes)):
        for j in range(len(breaks)):
            axes[i].axvline(x=(breaks[j]), color ='k', alpha=0.3, linestyle = 'dashed')
    
    fig.align_ylabels()
    
    fig.subplots_adjust(hspace=0)
    
    if savepath is not None:
        plotname = savepath + plotName
        fig.savefig(plotname, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: time, flux, astro, detec, breaks'
        data = [header, time, flux, astro, detec, breaks]
        pathdata = savepath + plotName[:-3] + 'pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
        
    if showPlot:
        plt.show()
    
    plt.close()
    return

def plot_rednoise(residuals, minbins, ingrDuration, occDuration, intTime, mode, savepath=None, showPlot=True, showtxt=True, savetxt=False, fontsize=20):
    
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
    ax.errorbar(binsz, rms, yerr=[rmslo, rmshi], fmt="k-", ecolor='0.5', capsize=0, label=r'$\rm Data~RMS$')
    ax.plot(binsz, stderr, c='red', label=r'$\rm Gaussian~std.$')
    ylim = ax.get_ylim()
    ax.plot([ingrDuration,ingrDuration],ylim, color='black', ls='--', alpha=0.6)
    ax.plot([occDuration,occDuration],ylim, color='black', ls='-.', alpha=0.6)
    ax.set_ylim(ylim)
    
    ax.xaxis.set_tick_params(labelsize=fontsize)
    ax.yaxis.set_tick_params(labelsize=fontsize)
    
    plt.xlabel(r'$N_{\rm binned}$', fontsize=fontsize)
    plt.ylabel(r'$\rm RMS$', fontsize=fontsize)
    plt.legend(loc='best', fontsize=fontsize)
    if savepath is not None:
        plotname = savepath + 'MCMC_'+mode+'_RedNoise.pdf'
        plt.savefig(plotname, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: residuals, minbins, ingrDuration, occDuration, intTime, mode'
        data = [residuals, minbins, ingrDuration, occDuration, intTime, mode]
        pathdata = savepath + 'MCMC_'+mode+'_RedNoise.pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    if showPlot:
        plt.show()
        
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
        print(outStr, flush=True)
    if savetxt:
        fname = savepath + 'MCMC_'+mode+'_RedNoise.txt'
        with open(fname,'w') as file:
            file.write(outStr)
    
    return

def triangle_colors(all_data, firstEcl_data, transit_data, secondEcl_data, fname=None, showPlot=False, fontsize=15):
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

    label = [r'$x_0$', r'$y_0$', r'$\sigma_x$', r'$\sigma_y$', r'$F$', r'$\rm Residuals$']

    fig = plt.figure(figsize = (8,8))
    axs = []
    gs  = gridspec.GridSpec(len(all_data)-1,len(all_data)-1)
    i = 0
    for k in range(np.sum(np.arange(len(all_data)))):
        j= k - np.sum(np.arange(i+1))
        ax = fig.add_subplot(gs[i,j])
        axs.append(ax)
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
            ax.set_xlabel(label[j], fontsize=fontsize)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
        if(i == j):
            i += 1
    handles = [l1,l2,l3]

    for i in range(len(axs)):
        axs[i].xaxis.set_tick_params(labelsize=fontsize*0.8)
        axs[i].yaxis.set_tick_params(labelsize=fontsize*0.8)
    
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(wspace=0)
    fig.align_ylabels(axs)
    fig.align_xlabels(axs)
    
    if fname is not None:
        fig.savefig(fname, bbox_inches='tight')
        # saving data used in the plot as pkl file
        header = 'HEADER: all_data, firstEcl_data, transit_data, secondEcl_data (structure of each list: xdata, ydata, psfxw, psfyw, flux, residuals)'
        data = [all_data, firstEcl_data, transit_data, secondEcl_data]
        pathdata = fname[:-3] +'pkl'
        with open(pathdata, 'wb') as outfile:
            pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)

    if showPlot:
        plt.show()
    
    plt.close()
    return

def look_for_residual_correlations(time, flux, xdata, ydata, psfxw, psfyw, residuals,
                                   p0_mcmc, p0_labels, p0_obj, mode, savepath=None, showPlot=False, fontsize=15):
    if 't0' in p0_labels:
        t0MCMC = p0_mcmc[np.where(p0_labels == 't0')[0][0]]
    else:
        t0MCMC = p0_obj['t0']
    if 'per' in p0_labels:
        perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
    else:
        perMCMC = p0_obj['per']
    if 'rp' in p0_labels:
        rpMCMC = p0_mcmc[np.where(p0_labels == 'rp')[0][0]]
    else:
        rpMCMC = p0_obj['rp']
    if 'a' in p0_labels:
        aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
    else:
        aMCMC = p0_obj['a']
    if 'inc' in p0_labels:
        incMCMC = p0_mcmc[np.where(p0_labels == 'inc')[0][0]]
    else:
        incMCMC = p0_obj['inc']
    if 'ecosw' in p0_labels:
        ecoswMCMC = p0_mcmc[np.where(p0_labels == 'ecosw')[0][0]]
    else:
        ecoswMCMC = p0_obj['ecosw']
    if 'esinw' in p0_labels:
        esinwMCMC = p0_mcmc[np.where(p0_labels == 'esinw')[0][0]]
    else:
        esinwMCMC = p0_obj['esinw']
    if 'q1' in p0_labels:
        q1MCMC = p0_mcmc[np.where(p0_labels == 'q1')[0][0]]
    else:
        q1MCMC = p0_obj['q1']
    if 'q2' in p0_labels:
        q2MCMC = p0_mcmc[np.where(p0_labels == 'q2')[0][0]]
    else:
        q2MCMC = p0_obj['q2']
    if 'fp'in p0_labels:
        fpMCMC = p0_mcmc[np.where(p0_labels == 'fp')[0][0]]
    else:
        fpMCMC = p0_obj['fp']

    eccMCMC = np.sqrt(ecoswMCMC**2 + esinwMCMC**2)
    wMCMC   = np.arctan2(esinwMCMC, ecoswMCMC)
    u1MCMC  = 2*np.sqrt(q1MCMC)*q2MCMC
    u2MCMC  = np.sqrt(q1MCMC)*(1-2*q2MCMC)

    trans, t_sec, true_anom = astro_models.transit_model(time, t0MCMC, perMCMC, rpMCMC,
                                                         aMCMC, incMCMC, eccMCMC, wMCMC,
                                                         u1MCMC, u2MCMC)
    # generating secondary eclipses model
    eclip = astro_models.eclipse(time, t0MCMC, perMCMC, rpMCMC, aMCMC, incMCMC, eccMCMC, wMCMC,
                                 fpMCMC, t_sec)

    # get in-transit indices
    ind_trans  = np.where(trans!=1)
    # get in-eclipse indices
    ind_eclip  = np.where((eclip!=(1+fpMCMC)))
    # seperating first and second eclipse
    ind_ecli1 = ind_eclip[0][np.where(ind_eclip[0]<int(len(time)/2))]
    ind_ecli2 = ind_eclip[0][np.where(ind_eclip[0]>int(len(time)/2))]

    data1 = [xdata, ydata, psfxw, psfyw, flux, residuals]
    data2 = [xdata[ind_ecli1], ydata[ind_ecli1], psfxw[ind_ecli1], psfyw[ind_ecli1], flux[ind_ecli1], residuals[ind_ecli1]]
    data3 = [xdata[ind_trans], ydata[ind_trans], psfxw[ind_trans], psfyw[ind_trans], flux[ind_trans], residuals[ind_trans]]
    data4 = [xdata[ind_ecli2], ydata[ind_ecli2], psfxw[ind_ecli2], psfyw[ind_ecli2], flux[ind_ecli2], residuals[ind_ecli2]]

    if savepath is not None:
        plotname = savepath + 'MCMC_'+mode+'_7.pdf'
    else:
        plotname = None
    triangle_colors(data1, data2, data3, data4, plotname, showPlot, fontsize)
    
    return
