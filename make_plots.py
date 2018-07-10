
import numpy as np
import matplotlib.pyplot as plt

def plot_photometry(time0, flux0, xdata0, ydata0, psfxw0, psfyw0, 
                    time, flux, xdata, ydata, psfxw, psfyw, breaks, savepath):
    '''
    Makes a multi-panel plot from photometry outputs.
    params:
    -------
        time0 : 1D array 
            array of time stamps. Discarded points not removed.
        flux0 : 1D array
            array of flux values for each time stamps. Discarded points not removed.
        xdata0 : 1D array
            initial modelled the fluxes for each time stamps. Discarded points not removed.
        ydata0: 1D array
            initial modelled astrophysical flux variation for each time stamps. 
            Discarded points not removed.
        psfxw0: 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        psfyw0: 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points not removed.
        time  : 1D array 
            array of time stamps. Discarded points removed.
        flux  : 1D array
            array of flux values for each time stamps. Discarded points removed.
        xdata  : 1D array
            initial modelled the fluxes for each time stamps. Discarded points removed.
        ydata : 1D array
            initial modelled astrophysical flux variation for each time stamps. Discarded points removed.
        psfxw : 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        psfyw : 1D array
            Point-Spread-Function (PSF) width along the x-direction. Discarded points removed.
        break : 1D array
            time of the breaks from one AOR to another.
        savepath : str
            path to directory where the plot will be saved
    returns:
    --------
        none
    '''
    
    fig, axes = plt.subplots(5, 1, sharex=True, figsize=(10, 12))
    #fig.suptitle("XO-3b Observation")

def plot_detec_syst(time, data, init):
    plt.figure(figsize=(10,3))
    plt.plot(time, data, '+', label='data')
    plt.plot(time, init, '+', label='guess')
    plt.title('Initial Guess')
    plt.xlabel('Time (BMJD)')
    plt.ylabel('Relative Flux')	
    
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
    pathplot = savepath + '01_Raw_data.pdf'
    fig.savefig(pathplot, bbox_inches='tight')
    return


def plot_init_guess(time, data, init, astro, detec, savepath):
    '''
    Makes a multi-panel plots for the initial light curve guesses.
    params:
    -------
        time  : 1D array 
            array of time stamps
        data  : 1D array
            array of flux values for each time stamps
        init  : 1D array
            initial modelled the fluxes for each time stamps
        astro : 1D array
            initial modelled astrophysical flux variation for each time stamps
        detec : 1D array
            initial modelled flux variation due to the detector for each time stamps
        savepath : str
            path to directory where the plot will be saved
    returns:
    --------
        none
    '''
    
    fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,9))
    #fig.suptitle('Initial Guess')
    
    axes[0].plot(time, data, '.', label='data')
    axes[0].plot(time, init, '.', label='guess')
    
    axes[1].plot(time, data/detec, '.', label='Corrected')
    axes[1].plot(time, astro, '.', label='Astrophysical')
    
    axes[2].plot(time, data/detec, '.', label='Corrected')
    axes[2].plot(time, astro, '.', label='Astrophysical')
    axes[2].set_ylim(0.998, 1.005)
    
    axes[3].plot(time, data-init, '.', label='residuals')
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
    pathplot = savepath + '02_Initial_Guess.pdf'
    fig.savefig(pathplot, bbox_inches='tight')
    return
