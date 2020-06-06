import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from astropy.stats import sigma_clip

# SPCA libraries
from .Photometry_Aperture import get_lightcurve
from .Photometry_Common import create_folder

import warnings
warnings.filterwarnings('ignore')

def run_diagnostics(planet, channel, AOR_snip, basepath, addStack, ncpu=4, nsigma=3, showPlot=False, savePlot=True):
    """Run frame diagnostics and choose which frames within a datacube are consistently bad and should be discarded.

    Args:
        planet (string): The name of the planet.
        channel (string): The channel being analyzed.
        AOR_snip (string): AOR snippet used to figure out what folders contain data.
        basepath (string): The full path to the folder containing folders for each planet.
        addStack (bool): Whether or not to add a background subtraction correction stack (will be automatically selected if present).
        nsigma (float): The number of sigma a frame's median value must be off from the median frame in order to be added to ignore_frames.

    Returns:
        list: The frames whose photometry is typically nsigma above/below the median frame and should be removed from photometry.

    """
    
    flux, time, xo, yo, xw, yw, bg, npp = get_lightcurve(basepath, AOR_snip, channel, planet,
                                                         save=False, bin_data=False,
                                                         showPlots=False, savePlots=False,
                                                         oversamp=False, r=[2.4], edges=['exact'],
                                                         addStack=addStack, moveCentroids=[True],
                                                         ncpu=ncpu)[0]
    
    try:
        flux  = sigma_clip(flux, sigma=5, maxiters=5).reshape(-1,64)
        bg    = sigma_clip(bg, sigma=5, maxiters=5).reshape(-1,64)
        xdata = sigma_clip(xo, sigma=5, maxiters=5).reshape(-1,64)
        ydata = sigma_clip(yo, sigma=5, maxiters=5).reshape(-1,64)
        xw    = sigma_clip(xw, sigma=5, maxiters=5).reshape(-1,64)
        yw    = sigma_clip(yw, sigma=5, maxiters=5).reshape(-1,64)
        npp   = sigma_clip(npp, sigma=5, maxiters=5).reshape(-1,64)
    except TypeError:
        flux  = sigma_clip(flux, sigma=5, iters=5).reshape(-1,64)
        bg    = sigma_clip(bg, sigma=5, iters=5).reshape(-1,64)
        xdata = sigma_clip(xdata, sigma=5, iters=5).reshape(-1,64)
        ydata = sigma_clip(ydata, sigma=5, iters=5).reshape(-1,64)
        xw    = sigma_clip(xw, sigma=5, iters=5).reshape(-1,64)
        yw    = sigma_clip(yw, sigma=5, iters=5).reshape(-1,64)
        npp   = sigma_clip(npp, sigma=5, iters=5).reshape(-1,64)
        
    flux_med = np.ma.median(flux, axis=0)
    bg_med = np.ma.median(bg, axis=0)
    xdata_med = np.ma.median(xdata, axis=0)
    ydata_med = np.ma.median(ydata, axis=0)
    xw_med = np.ma.median(xw, axis=0)
    yw_med = np.ma.median(yw, axis=0)
    npp_med = np.ma.median(npp, axis=0)

    meanflux = np.ma.median(flux_med)
    meanbg = np.ma.median(bg_med)
    meanxdata = np.ma.median(xdata_med)
    meanydata = np.ma.median(ydata_med)
    meanxw = np.ma.median(xw_med)
    meanyw = np.ma.median(yw_med)
    meannpp = np.ma.median(npp_med)

    flux_med /= meanflux
    bg_med /= meanbg
    xdata_med /= meanxdata
    ydata_med /= meanydata
    xw_med /= meanxw
    yw_med /= meanyw
    npp_med /= meannpp
    
    sigmaflux = np.ma.std(sigma_clip(flux_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmabg = np.ma.std(sigma_clip(bg_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmaxdata = np.ma.std(sigma_clip(xdata_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmaydata = np.ma.std(sigma_clip(ydata_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmaxw = np.ma.std(sigma_clip(xw_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmayw = np.ma.std(sigma_clip(yw_med, sigma=5, maxiters=5, cenfunc=np.ma.median))
    sigmanpp = np.ma.std(sigma_clip(npp_med, sigma=5, maxiters=5, cenfunc=np.ma.median))

    if savePlot or showPlot:
        nb = np.arange(64)
        fig, axes = plt.subplots(ncols=1, nrows=7, sharex=True, figsize=(5.5,10))

        axes[0].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[1].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[2].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[3].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[4].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[5].axhline(y=1, color='k', alpha=0.4, linewidth=1)
        axes[6].axhline(y=1, color='k', alpha=0.4, linewidth=1)

        axes[0].axhline(y= 1 + nsigma*sigmaflux , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[1].axhline(y= 1 + nsigma*sigmabg , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[2].axhline(y= 1 + nsigma*sigmaxdata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[3].axhline(y= 1 + nsigma*sigmaydata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[4].axhline(y= 1 + nsigma*sigmaxw, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[5].axhline(y= 1 + nsigma*sigmayw, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[6].axhline(y= 1 + nsigma*sigmanpp , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')

        axes[0].axhline(y= 1 - nsigma*sigmaflux , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[1].axhline(y= 1 - nsigma*sigmabg   , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[2].axhline(y= 1 - nsigma*sigmaxdata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[3].axhline(y= 1 - nsigma*sigmaydata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[4].axhline(y= 1 - nsigma*sigmaxw, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[5].axhline(y= 1 - nsigma*sigmayw, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
        axes[6].axhline(y= 1 - nsigma*sigmanpp , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')

        flux_markers = list(np.where(np.logical_or(flux_med<1-nsigma*sigmaflux, flux_med>1+nsigma*sigmaflux))[0])
        bg_markers = list(np.where(np.logical_or(bg_med<1-nsigma*sigmabg, bg_med>1+nsigma*sigmabg))[0])
        xdata_markers = list(np.where(np.logical_or(xdata_med<1-nsigma*sigmaxdata, xdata_med>1+nsigma*sigmaxdata))[0])
        ydata_markers = list(np.where(np.logical_or(ydata_med<1-nsigma*sigmaydata, ydata_med>1+nsigma*sigmaydata))[0])
        xw_markers = list(np.where(np.logical_or(xw_med<1-nsigma*sigmaxw, xw_med>1+nsigma*sigmaxw))[0])
        yw_markers = list(np.where(np.logical_or(yw_med<1-nsigma*sigmayw, yw_med>1+nsigma*sigmayw))[0])
        npp_markers = list(np.where(np.logical_or(npp_med<1-nsigma*sigmanpp, npp_med>1+nsigma*sigmanpp))[0])
        flux_other_markers = np.ma.concatenate((bg_markers, xdata_markers, ydata_markers,
                                                xw_markers, yw_markers, npp_markers))
        flux_other_markers = list(np.setdiff1d(flux_other_markers, flux_markers).astype(int))
        axes[0].plot(nb, flux_med , 'k', mec ='r', marker='s', markevery=flux_markers,fillstyle='none')
        axes[0].plot(nb, flux_med , 'k', mec ='b', marker='s', markevery=flux_other_markers,fillstyle='none')
        axes[1].plot(nb, bg_med   , 'k', mec ='r', marker='s', markevery=bg_markers,fillstyle='none')
        axes[2].plot(nb, xdata_med, 'k', mec ='r', marker='s', markevery=xdata_markers,fillstyle='none')
        axes[3].plot(nb, ydata_med, 'k', mec ='r', marker='s', markevery=ydata_markers,fillstyle='none')
        axes[4].plot(nb, xw_med, 'k', mec ='r', marker='s', markevery=xw_markers,fillstyle='none')
        axes[5].plot(nb, yw_med, 'k', mec ='r', marker='s', markevery=yw_markers,fillstyle='none')
        axes[6].plot(nb, npp_med , 'k', mec ='r', marker='s', markevery=npp_markers,fillstyle='none')

        #axes[0].set_ylim(0.99, 1.01)
        axes[0].set_ylim(np.ma.min([(np.ma.min(flux_med)-1)*4/3+1, 1-(nsigma+1)*sigmaflux]),
                         np.ma.max([(np.ma.max(flux_med)-1)*4/3+1, 1+(nsigma+1)*sigmaflux]))
        axes[1].set_ylim(np.ma.min([(np.ma.min(bg_med)-1)*4/3+1, 1-(nsigma+1)*sigmabg]),
                         np.ma.max([(np.ma.max(bg_med)-1)*4/3+1, 1+(nsigma+1)*sigmabg]))
        axes[2].set_ylim(np.ma.min([(np.ma.min(xdata_med)-1)*4/3+1, 1-(nsigma+1)*sigmaxdata]),
                         np.ma.max([(np.ma.max(xdata_med)-1)*4/3+1, 1+(nsigma+1)*sigmaxdata]))
        axes[3].set_ylim(np.ma.min([(np.ma.min(ydata_med)-1)*4/3+1, 1-(nsigma+1)*sigmaydata]),
                         np.ma.max([(np.ma.max(ydata_med)-1)*4/3+1, 1+(nsigma+1)*sigmaydata]))
        axes[4].set_ylim(np.ma.min([(np.ma.min(xw_med)-1)*4/3+1, 1-(nsigma+1)*sigmaxw]),
                         np.ma.max([(np.ma.max(xw_med)-1)*4/3+1, 1+(nsigma+1)*sigmaxw]))
        axes[5].set_ylim(np.ma.min([(np.ma.min(yw_med)-1)*4/3+1, 1-(nsigma+1)*sigmayw]),
                         np.ma.max([(np.ma.max(yw_med)-1)*4/3+1, 1+(nsigma+1)*sigmayw]))
        axes[6].set_ylim(np.ma.min([(np.ma.min(npp_med)-1)*4/3+1, 1-(nsigma+1)*sigmanpp]),
                         np.ma.max([(np.ma.max(npp_med)-1)*4/3+1, 1+(nsigma+1)*sigmanpp]))

        axes[0].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[1].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[2].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[3].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[4].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[5].yaxis.set_major_locator(MaxNLocator(4,prune='both'))
        axes[6].yaxis.set_major_locator(MaxNLocator(4,prune='both'))

        axes[0].set_ylabel(r'$F$', fontsize=14)
        axes[1].set_ylabel(r'$b$', fontsize=14)
        axes[2].set_ylabel(r'$x_0$', fontsize=14)
        axes[3].set_ylabel(r'$y_0$', fontsize=14)
        axes[4].set_ylabel(r'$\sigma_x$', fontsize=14)
        axes[5].set_ylabel(r'$\sigma_y$', fontsize=14)
        axes[6].set_ylabel(r'$\beta$', fontsize=14)

        axes[0].ticklabel_format(useOffset=False)
        axes[1].ticklabel_format(useOffset=False)
        axes[2].ticklabel_format(useOffset=False)
        axes[3].ticklabel_format(useOffset=False)
        axes[4].ticklabel_format(useOffset=False)
        axes[5].ticklabel_format(useOffset=False)
        axes[6].ticklabel_format(useOffset=False)

        axes[6].set_xlim(-0.5,63.5)
        axes[6].set_xlabel('Frame Number', fontsize=14)
        fig.subplots_adjust(hspace=0)
        if savePlot:
            savepath = f'{basepath}{planet}/analysis/frameDiagnostics/{channel}/'
            if addStack:
                savepath += 'addedStack/'
            else:
                savepath += 'addedBlank/'
            # create save folder
            savepath = create_folder(savepath, True, True)
    
            fname = savepath + 'Frame_Diagnostics1.pdf'
            fig.savefig(fname, bbox_inches='tight')
        if showPlot:
            plt.show()
        plt.close()

    print('Ignore Frames:', flux_markers)
    print('Bad by:', np.array(((flux_med-1)/sigmaflux)[flux_markers]), 'sigma')

    return flux_markers
