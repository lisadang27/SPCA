import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os, sys
from astropy.io import fits
from astropy.stats import sigma_clip
import glob

# SPCA libraries
from .Photometry_Aperture import sigma_clipping
from .Photometry_Aperture import bgsubtract
from .Photometry_Aperture import centroid_FWM
from .Photometry_Aperture import A_photometry

def get_stacks(stackpath, datapath, AOR_snip, ch):
    """Find paths to all the correction stack FITS files.

    Args:
        stackpath (string): Full path to the folder containing background correction stacks.
        datapath (string): Full path to the data folder containing the AOR folders with images to be corrected.
        AOR_snip (string): AOR snippet used to figure out what folders contain data.
        ch (string): String specifying which channel is being used.

    Returns:
        ndarray: The calibration FITS file that should be used for background subtraction correction.

    """
    
    stacks = np.array(os.listdir(stackpath))
    locs = np.array([stacks[i].find('SPITZER_I') for i in range(len(stacks))])
    good = np.where(locs!=-1)[0] #filter out all files that don't fit the correct naming convention for correction stacks
    offset = 11 #legth of the string "SPITZER_I#_"
    keys = np.array([stacks[i][locs[i]+offset:].split('_')[0] for i in good]) #pull out just the key that says what sdark this stack is for

    data_list = os.listdir(datapath)
    AOR_list = [a for a in data_list if AOR_snip==a[:len(AOR_snip)]]
    calFiles = []
    for i in range(len(AOR_list)):
        path = datapath + '/' + AOR_list[i] + '/' + ch +'/cal/'
        if not os.path.isdir(path):
            print('Error: Folder \''+path+'\' does not exist, so automatic correction stack selection cannot be performed')
            return []
        fname = glob.glob(path+'*sdark.fits')[0]
        loc = fname.find('SPITZER_I')+offset
        key = fname[loc:].split('_')[0]
        calFiles.append(os.path.join(stackpath, stacks[list(good)][np.where(keys == key)[0][0]]))
    return np.array(calFiles)

# Noise pixel param
def noisepixparam(image_data):
    """Compute the noise pixel parameter.

    Args:
        image_data (ndarray): FITS images stack.

    Returns:
        list: The noise pixel parameter for each image in the stack.

    """
    
    lb= 14
    ub= 18
    npp=[]
    # Its better to operate on the copy of desired portion of image_data than on image_data itself.
    # This reduces the risk of modifying image_data accidently. Arguements are passed as pass-by-object-reference.
    stx=np.ndarray((64,4,4))
    
    np.copyto(stx,image_data[:,lb:ub,lb:ub])
    for img in stx:
        #To find noise pixel parameter for each frame. For eqn, refer Knutson et al. 2012
        numer= np.ma.sum(img)
        numer=np.square(numer)
        denom=0.0
        temp = np.square(img)
        denom = np.ma.sum(img)
        param= numer/denom
        npp.append(param)
    return np.array(npp)

def bgnormalize(image_data):
    """Compute the normalized background from each stack.

    Args:
        image_data (ndarray): FITS images stack.

    Returns:
        ndarray: The background in each frame of a datacube, normalized by the median within that datacube.

    """
    
    xmask = np.ma.make_mask(np.zeros((64,32,32)), shrink=False)
    xmask[:,13:18,13:18]=True
    masked= np.ma.masked_array(image_data, mask=xmask)
    
    #Normalize by average background from whole datecube
    return np.ma.median(masked.reshape(masked.shape[0], -1), axis=1)/np.ma.median(masked)

def load_data(path, AOR):
    """Compute the normalized background from each stack.

    Args:
        image_data (ndarray): FITS images stack.

    Returns:
        tuple: flux (ndarray; the aperture sum from each frame, normalized by the median flux from its datacube),
            bg (ndarray; the background flux from each frame, normalized by the median background from its datacube),
            xdata (ndarray; the x-centroid from each frame, normalized by the median x-centroid from its datacube),
            ydata (ndarray; the y-centroid from each frame, normalized by the median y-centroid from its datacube),
            psfwx (ndarray; the x PSF width from each frame, normalized by the median x PSF width from its datacube),
            psfwy (ndarray; the y PSF width from each frame, normalized by the median y PSF width from its datacube),
            beta (ndarray; the noise pixel parameter from each frame, normalized by the median noise pixel parameter from its datacube).

    """
    
    pathflux  = path + 'flux'  + AOR + '.npy'
    pathbg    = path + 'bg'    + AOR + '.npy'
    pathxdata = path + 'xdata' + AOR + '.npy'
    pathydata = path + 'ydata' + AOR + '.npy'
    pathpsfwx = path + 'psfwx' + AOR + '.npy'
    pathpsfwy = path + 'psfwy' + AOR + '.npy'
    pathbeta  = path + 'beta'  + AOR + '.npy'
    
    flux  = np.load(pathflux)
    bg    = np.load(pathbg)
    xdata = np.load(pathxdata)
    ydata = np.load(pathydata)
    psfwx = np.load(pathpsfwx)
    psfwy = np.load(pathpsfwy)
    beta  = np.load(pathbeta )
    
    return flux, bg, xdata, ydata, psfwx, psfwy, beta 

def get_stats(data, median_arr, std_arr):
    """Compute the median and std. dev. from an array of data and add it to the previously computed values.

    Args:
        data (ndarray): The array to get information from.
        median_arr (ndarray): The previously computed median values to be appended to.
        std_arr (ndarray): The previously computed std. dev. values to be appended to.

    Returns:
        tuple: median_arr (ndarray; the median of the data),
            std_arr (ndarray; the std. dev. of the data).

    """
    
    for i in range(np.array(data).shape[0]):
        median_arr = np.append(median_arr, [np.ma.std(data[i], axis=0)], axis = 0)
        std_arr    = np.append(std_arr, [np.ma.median(data[i], axis=0)], axis = 0)
    return median_arr, std_arr

def run_diagnostics(planet, channel, AOR_snip, basepath, addStack, nsigma=3):
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
    
    savepath = basepath+planet+'/analysis/frameDiagnostics/'
    datapath = basepath+planet+'/data/'+channel+'/'
    stackpath = basepath+'Calibration/' #folder containing properly named correction stacks (will be automatically selected)
    
    if addStack:
        stacks = get_stacks(stackpath, datapath, AOR_snip, channel)
        savepath += channel+'/addedStack/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
    else:
        savepath += channel+'/addedBlank/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)

    dirs_all = os.listdir(datapath)
    dirs = [k for k in dirs_all if AOR_snip==k[:len(AOR_snip)]]
    print ('Found the following AORs', dirs)
    tossed = 0
    counter=0
    ct = 0

    for i in range(len(dirs)):

        direc = dirs[i]

        if addStack:
            stack = fits.open(stacks[i], mode='readonly')
            skydark = stack[0].data
        
        path = datapath+direc+'/'+channel+'/bcd'
        
        print('Analysing', direc)
        normbg=[]
        normf=[]
        normx=[]
        normy=[]
        normpsfwx=[]
        normpsfwy=[]
        normnpp=[]
        for filename in glob.glob(os.path.join(path, '*bcd.fits')):
            #print filename
            f=fits.open(filename,mode='readonly')
            # get data and apply sky dark correction
            image_data0 = f[0].data
            if addStack:
                image_data0 += skydark
            # convert MJy/str to electron count
            convfact=f[0].header['GAIN']*f[0].header['EXPTIME']/f[0].header['FLUXCONV']
            image_data1=image_data0*convfact        
            #sigma clip
            image_data2, tossed, _ = sigma_clipping(image_data1, tossed, sigma=4, maxiters=5)
            #b should be calculated on sigmaclipped data
            normbg.append(bgnormalize(image_data2))
            #bg subtract
            image_data3, _, _ = bgsubtract(image_data2)
            #centroid
            xo, yo, psfwx, psfwy = centroid_FWM(image_data3)
            ape_sum, _ = A_photometry(np.ma.masked_invalid(image_data3), np.zeros_like(image_data3))
            npp = noisepixparam(image_data3)     
            
            normf.append(np.ma.masked_invalid(ape_sum)/np.ma.median(np.ma.masked_invalid(ape_sum)))
            normx.append(xo/np.ma.median(xo))
            normy.append(yo/np.ma.median(yo))
            normpsfwx.append(psfwx/np.ma.median(psfwx))
            normpsfwy.append(psfwy/np.ma.median(psfwy))
            normnpp.append(npp/np.ma.median(npp))
            ct+=1
        counter+=1

        pathFULL  = savepath
        pathflux  = pathFULL + 'flux' + direc
        pathbg    = pathFULL + 'bg' + direc
        pathx     = pathFULL + 'xdata' + direc
        pathy     = pathFULL + 'ydata' + direc

        pathpsfx  = pathFULL + 'psfwx' + direc
        pathpsfy  = pathFULL + 'psfwy' + direc
        pathbeta  = pathFULL + 'beta' + direc
        np.save(pathflux, normf)
        np.save(pathbg, normbg)
        np.save(pathx, normx)
        np.save(pathy, normy)
        np.save(pathpsfx, normpsfwx)
        np.save(pathpsfy, normpsfwy)
        np.save(pathbeta, normnpp)
    
    #endfor
    
    
    AOR = [a for a in os.listdir(datapath) if AOR_snip==a[:len(AOR_snip)]]
    data = [np.asarray(load_data(savepath, a)) for a in AOR]
    nb_data = [len(data[i]) for i in range(len(data))]
    data = [np.where(np.isfinite(data[i]), data[i], 99999) for i in range(len(data))]
    try:
        flux  = np.array([sigma_clip(data[i][0], sigma=4, maxiters=5) for i in range(len(data))])
        bg    = np.array([sigma_clip(data[i][1], sigma=4, maxiters=5) for i in range(len(data))])
        xdata = np.array([sigma_clip(data[i][2], sigma=4, maxiters=5) for i in range(len(data))])
        ydata = np.array([sigma_clip(data[i][3], sigma=4, maxiters=5) for i in range(len(data))])
        psfwx = np.array([sigma_clip(data[i][4], sigma=4, maxiters=5) for i in range(len(data))])
        psfwy = np.array([sigma_clip(data[i][5], sigma=4, maxiters=5) for i in range(len(data))])
        beta  = np.array([sigma_clip(data[i][6], sigma=4, maxiters=5) for i in range(len(data))])
    except TypeError:
        flux  = np.array([sigma_clip(data[i][0], sigma=4, iters=5) for i in range(len(data))])
        bg    = np.array([sigma_clip(data[i][1], sigma=4, iters=5) for i in range(len(data))])
        xdata = np.array([sigma_clip(data[i][2], sigma=4, iters=5) for i in range(len(data))])
        ydata = np.array([sigma_clip(data[i][3], sigma=4, iters=5) for i in range(len(data))])
        psfwx = np.array([sigma_clip(data[i][4], sigma=4, iters=5) for i in range(len(data))])
        psfwy = np.array([sigma_clip(data[i][5], sigma=4, iters=5) for i in range(len(data))])
        beta  = np.array([sigma_clip(data[i][6], sigma=4, iters=5) for i in range(len(data))])
    fluxval, bgval, xdataval, ydataval, psfwxval, psfwyval, betaval = np.empty((0,64)), np.empty((0,64)), np.empty((0,64)), \
    np.empty((0,64)), np.empty((0,64)), np.empty((0,64)), np.empty((0,64))
    fluxerr, bgerr, xdataerr, ydataerr, psfwxerr, psfwyerr, betaerr = np.empty((0,64)), np.empty((0,64)), np.empty((0,64)), \
    np.empty((0,64)), np.empty((0,64)), np.empty((0,64)), np.empty((0,64))

    fluxval , fluxerr = get_stats(flux , fluxval, fluxerr)
    bgval   , bgerr   = get_stats(bg   , bgval,   bgerr  )
    xdataval, xdataerr= get_stats(xdata, xdataval, xdataerr)
    ydataval, ydataerr= get_stats(ydata, ydataval, ydataerr)
    psfwxval, psfwxerr= get_stats(psfwx, psfwxval, psfwxerr)
    psfwyval, psfwyerr= get_stats(psfwy, psfwyval, psfwyerr)
    betaval , betaerr = get_stats(beta , betaval , betaerr)

    bgmed = np.ma.median(bg[0], axis = 0)

    flux_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        flux_all = np.append(flux_all, flux[i], axis = 0)
        try:
            flux_all = sigma_clip(flux_all, sigma=4, maxiters=5)
        except TypeError:
            flux_all = sigma_clip(flux_all, sigma=4, iters=5)

    bg_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        bg_all = np.append(bg_all, bg[i], axis = 0)
        bg_all = np.where(np.isfinite(bg_all), bg_all, 99999)
        try:
            bg_all = sigma_clip(bg_all, sigma=4, maxiters=5)
        except TypeError:
            bg_all = sigma_clip(bg_all, sigma=4, iters=5)

    xdata_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        xdata_all = np.append(xdata_all, xdata[i], axis = 0)
        try:
            xdata_all = sigma_clip(xdata_all, sigma=4, maxiters=5)
        except TypeError:
            xdata_all = sigma_clip(ydata_all, sigma=4, iters=5)

    ydata_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        ydata_all = np.append(ydata_all, ydata[i], axis = 0)
        try:
            ydata_all = sigma_clip(ydata_all, sigma=4, maxiters=5)
        except TypeError:
            ydata_all = sigma_clip(ydata_all, sigma=4, iters=5)

    psfwx_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        psfwx_all = np.append(psfwx_all, psfwx[i], axis = 0)
        try:
            psfwx_all = sigma_clip(psfwx_all, sigma=4, maxiters=5)
        except TypeError:
            psfwx_all = sigma_clip(psfwx_all, sigma=4, iters=5)

    psfwy_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        psfwy_all = np.append(psfwy_all, psfwy[i], axis = 0)
        try:
            psfwy_all = sigma_clip(psfwy_all, sigma=4, maxiters=5)
        except TypeError:
            psfwy_all = sigma_clip(psfwy_all, sigma=4, iters=5)

    beta_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        beta_all = np.append(beta_all, beta[i], axis = 0)
        try:
            beta_all = sigma_clip(beta_all, sigma=4, maxiters=5)
        except TypeError:
            beta_all = sigma_clip(beta_all, sigma=4, iters=5)

    flux_med = np.ma.median(flux_all, axis=0)
    bg_med = np.ma.median(bg_all, axis=0)
    xdata_med = np.ma.median(xdata_all, axis=0)
    ydata_med = np.ma.median(ydata_all, axis=0)
    psfwx_med = np.ma.median(psfwx_all, axis=0)
    psfwy_med = np.ma.median(psfwy_all, axis=0)
    beta_med = np.ma.median(beta_all, axis=0)

    bgall = np.ma.concatenate([bg[i] for i in range(bg.shape[0])], axis=0)

    bgmed, bgstd = np.ma.median(bgall, axis = 0), np.ma.std(bgall, axis = 0)

    meanflux, sigmaflux = np.ma.median(flux_med), np.ma.std(flux_med)
    meanbg, sigmabg = np.ma.median(bg_med), np.ma.std(bg_med)
    meanxdata, sigmaxdata = np.ma.median(xdata_med), np.ma.std(xdata_med)
    meanydata, sigmaydata = np.ma.median(ydata_med), np.ma.std(ydata_med)
    meanpsfwx, sigmapsfwx = np.ma.median(psfwx_med), np.ma.std(psfwx_med)
    meanpsfwy, sigmapsfwy = np.ma.median(psfwy_med), np.ma.std(psfwy_med)
    meanbeta, sigmabeta = np.ma.median(beta_med), np.ma.std(beta_med)

    flag = False
    while(flag == False):
        index = np.where(flux_med < (meanflux - nsigma*sigmaflux))
        index = np.append(index, np.where(flux_med > (meanflux + nsigma*sigmaflux)))
        sigmaflux2 = np.ma.std(np.delete(flux_med, index))
        flag = (sigmaflux2 == sigmaflux)
        sigmaflux = sigmaflux2


    flag = False
    while(flag == False):
        index = np.where(bg_med < (meanbg - nsigma*sigmabg))
        index = np.append(index, np.where(bg_med > (meanbg + nsigma*sigmabg)))
        sigmabg2 = np.ma.std(np.delete(bg_med, index))
        flag = (sigmabg2 == sigmabg)
        sigmabg = sigmabg2

    flag = False
    while(flag == False):
        index = np.where(xdata_med < (meanxdata - nsigma*sigmaxdata))
        index = np.append(index, np.where(xdata_med > (meanxdata + nsigma*sigmaxdata)))
        sigmaxdata2 = np.ma.std(np.delete(xdata_med, index))
        flag = (sigmaxdata2 == sigmaxdata)
        sigmaxdata = sigmaxdata2

    flag = False
    while(flag == False):
        index = np.where(ydata_med < (meanydata - nsigma*sigmaydata))
        index = np.append(index, np.where(ydata_med > (meanydata + nsigma*sigmaydata)))
        sigmaydata2 = np.ma.std(np.delete(ydata_med, index))
        flag = (sigmaydata2 == sigmaydata)
        sigmaydata = sigmaydata2

    flag = False
    while(flag == False):
        index = np.where(psfwx_med < (meanpsfwx - nsigma*sigmapsfwx))
        index = np.append(index, np.where(psfwx_med > (meanpsfwx + nsigma*sigmapsfwx)))
        sigmapsfwx2 = np.ma.std(np.delete(psfwx_med, index))
        flag = (sigmapsfwx2 == sigmapsfwx)
        sigmapsfwx = sigmapsfwx2

    flag = False
    while(flag == False):
        index = np.where(psfwy_med < (meanpsfwy - nsigma*sigmapsfwy))
        index = np.append(index, np.where(psfwy_med > (meanpsfwy + nsigma*sigmapsfwy)))
        sigmapsfwy2 = np.ma.std(np.delete(psfwy_med, index))
        flag = (sigmapsfwy2 == sigmapsfwy)
        sigmapsfwy = sigmapsfwy2

    flag = False
    while(flag == False):
        index = np.where(beta_med < (meanbeta - nsigma*sigmabeta))
        index = np.append(index, np.where(beta_med > (meanbeta + nsigma*sigmabeta)))
        sigmabeta2 = np.ma.std(np.delete(beta_med, index))
        flag = (sigmabeta2 == sigmabeta)
        sigmabeta = sigmabeta2

    flux_med /= meanflux
    bg_med /= meanbg
    xdata_med /= meanxdata
    ydata_med /= meanydata
    psfwx_med /= meanpsfwx
    psfwy_med /= meanpsfwy
    beta_med /= meanbeta



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
    axes[4].axhline(y= 1 + nsigma*sigmapsfwx, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[5].axhline(y= 1 + nsigma*sigmapsfwy, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[6].axhline(y= 1 + nsigma*sigmabeta , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')

    axes[0].axhline(y= 1 - nsigma*sigmaflux , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[1].axhline(y= 1 - nsigma*sigmabg   , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[2].axhline(y= 1 - nsigma*sigmaxdata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[3].axhline(y= 1 - nsigma*sigmaydata, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[4].axhline(y= 1 - nsigma*sigmapsfwx, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[5].axhline(y= 1 - nsigma*sigmapsfwy, color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')
    axes[6].axhline(y= 1 - nsigma*sigmabeta , color='#6495ED', alpha=0.7, linewidth=2, linestyle = 'dashed')

    flux_markers = list(np.where(np.logical_or(flux_med<1-nsigma*sigmaflux, flux_med>1+nsigma*sigmaflux))[0])
    bg_markers = list(np.where(np.logical_or(bg_med<1-nsigma*sigmabg, bg_med>1+nsigma*sigmabg))[0])
    xdata_markers = list(np.where(np.logical_or(xdata_med<1-nsigma*sigmaxdata, xdata_med>1+nsigma*sigmaxdata))[0])
    ydata_markers = list(np.where(np.logical_or(ydata_med<1-nsigma*sigmaydata, ydata_med>1+nsigma*sigmaydata))[0])
    psfwx_markers = list(np.where(np.logical_or(psfwx_med<1-nsigma*sigmapsfwx, psfwx_med>1+nsigma*sigmapsfwx))[0])
    psfwy_markers = list(np.where(np.logical_or(psfwy_med<1-nsigma*sigmapsfwy, psfwy_med>1+nsigma*sigmapsfwy))[0])
    beta_markers = list(np.where(np.logical_or(beta_med<1-nsigma*sigmabeta, beta_med>1+nsigma*sigmabeta))[0])
    flux_other_markers = np.ma.concatenate((bg_markers, xdata_markers, ydata_markers, psfwx_markers, psfwy_markers, beta_markers))
    flux_other_markers = list(np.setdiff1d(flux_other_markers, flux_markers).astype(int))
    axes[0].plot(nb, flux_med , 'k', mec ='r', marker='s', markevery=flux_markers,fillstyle='none')
    axes[0].plot(nb, flux_med , 'k', mec ='b', marker='s', markevery=flux_other_markers,fillstyle='none')
    axes[1].plot(nb, bg_med   , 'k', mec ='r', marker='s', markevery=bg_markers,fillstyle='none')
    axes[2].plot(nb, xdata_med, 'k', mec ='r', marker='s', markevery=xdata_markers,fillstyle='none')
    axes[3].plot(nb, ydata_med, 'k', mec ='r', marker='s', markevery=ydata_markers,fillstyle='none')
    axes[4].plot(nb, psfwx_med, 'k', mec ='r', marker='s', markevery=psfwx_markers,fillstyle='none')
    axes[5].plot(nb, psfwy_med, 'k', mec ='r', marker='s', markevery=psfwy_markers,fillstyle='none')
    axes[6].plot(nb, beta_med , 'k', mec ='r', marker='s', markevery=beta_markers,fillstyle='none')

    #axes[0].set_ylim(0.99, 1.01)
    axes[0].set_ylim(np.ma.min([(np.ma.min(flux_med)-1)*4/3+1, 1-(nsigma+1)*sigmaflux]), np.ma.max([(np.ma.max(flux_med)-1)*4/3+1, 1+(nsigma+1)*sigmaflux]))
    axes[1].set_ylim(np.ma.min([(np.ma.min(bg_med)-1)*4/3+1, 1-(nsigma+1)*sigmabg]), np.ma.max([(np.ma.max(bg_med)-1)*4/3+1, 1+(nsigma+1)*sigmabg]))
    axes[2].set_ylim(np.ma.min([(np.ma.min(xdata_med)-1)*4/3+1, 1-(nsigma+1)*sigmaxdata]), np.ma.max([(np.ma.max(xdata_med)-1)*4/3+1, 1+(nsigma+1)*sigmaxdata]))
    axes[3].set_ylim(np.ma.min([(np.ma.min(ydata_med)-1)*4/3+1, 1-(nsigma+1)*sigmaydata]), np.ma.max([(np.ma.max(ydata_med)-1)*4/3+1, 1+(nsigma+1)*sigmaydata]))
    axes[4].set_ylim(np.ma.min([(np.ma.min(psfwx_med)-1)*4/3+1, 1-(nsigma+1)*sigmapsfwx]), np.ma.max([(np.ma.max(psfwx_med)-1)*4/3+1, 1+(nsigma+1)*sigmapsfwx]))
    axes[5].set_ylim(np.ma.min([(np.ma.min(psfwy_med)-1)*4/3+1, 1-(nsigma+1)*sigmapsfwy]), np.ma.max([(np.ma.max(psfwy_med)-1)*4/3+1, 1+(nsigma+1)*sigmapsfwy]))
    axes[6].set_ylim(np.ma.min([(np.ma.min(beta_med)-1)*4/3+1, 1-(nsigma+1)*sigmabeta]), np.ma.max([(np.ma.max(beta_med)-1)*4/3+1, 1+(nsigma+1)*sigmabeta]))

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
    axes[4].set_ylabel(r'$\sigma _x$', fontsize=14)
    axes[5].set_ylabel(r'$\sigma _y$', fontsize=14)
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
    fname = savepath + 'Frame_Diagnostics1.pdf'
    fig.savefig(fname, bbox_inches='tight')
    plt.show()


    print('Ignore Frames:', flux_markers)
    print('Bad by:', np.array(((flux_med-1)/sigmaflux)[flux_markers]), 'sigma')

    return flux_markers
