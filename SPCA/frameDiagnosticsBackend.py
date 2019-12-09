import warnings
warnings.simplefilter("ignore", UserWarning)

import numpy as np
import matplotlib
from matplotlib import rc
import matplotlib.pyplot as plt
import matplotlib.patches
import time
from matplotlib.ticker import MaxNLocator
import os, sys
from astropy.io import fits
from astropy.stats import sigma_clip
from photutils import aperture_photometry
from photutils import CircularAperture
from numpy import std
import glob
import csv
import operator
import warnings
import matplotlib.ticker as mtick
from photutils.datasets import make_4gaussians_image

def get_stacks(stackpath, datapath, AOR_snip, ch):
    '''
    Find paths to all the fits files.

    Parameters
    ----------


    Returns
    -------

    :return: 
    '''
    stacks = np.array(os.listdir(stackpath))
    locs = np.array([stacks[i].find('SPITZER_I') for i in range(len(stacks))])
    good = np.where(locs!=-1)[0] #filter out all files that don't fit the correct naming convention for correction stacks
    offset = 11 #legth of the string "SPITZER_I#_"
    keys = np.array([stacks[i][locs[i]+offset:].split('_')[0] for i in good]) #pull out just the key that says what sdark this stack is for

    data_list = os.listdir(datapath)
    AOR_list = [a for a in data_list if AOR_snip in a]
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
    return calFiles



def sigma_clipping(image_data, tossed, bounds = ( 14, 18, 14, 18)):#,fname):
    lbx, ubx, lby, uby = bounds
    h, w, l = image_data.shape
    # mask invalids
    image_data2 = np.ma.masked_invalid(image_data)
    # make mask to mask entire bad frame
    x = np.ones(shape = (w, l))
    mask = np.ma.make_mask(x)
    sig_clipped_data = sigma_clip(image_data2, sigma=4, maxiters=4, cenfunc=np.ma.median, axis = 0)
    for i in range (h):
        oldstar = image_data[i, lbx:ubx, lby:uby]
        newstar = sig_clipped_data[i, lbx:ubx, lby:uby]
        truth   = newstar==oldstar
        if(np.ma.sum(truth) < truth.size):
            sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
            tossed += 1
            #print('tossed:', tossed)
    return sig_clipped_data

def bgsubtract(image_data):
    bgsubimg=image_data
    x=np.ndarray ( shape=(64,32,32), dtype=bool)
    xmask=np.ma.make_mask(x,copy=True, shrink=True, dtype=np.bool)
    xmask[:,:,:]= False
    xmask[:,14:18,14:18]=True
    masked= np.ma.masked_array(bgsubimg, mask=xmask)
    n=0
    #Background subtraction for each frame
    while(n<64):
        bg_avg=np.ma.median(masked[n])
        bgsubimg[n]= bgsubimg[n,:,:] - bg_avg
        n+=1
    return bgsubimg

def centroid_FWM(image_data, scale = 1, bounds = (14, 18, 14, 18)):
    lbx, ubx, lby, uby = bounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    starbox = image_data[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape
    # get centroid  
    X, Y    = np.mgrid[:w,:l]
    cx      = (np.ma.sum(np.ma.sum(X*starbox, axis=1), axis=1)/(np.ma.sum(np.ma.sum(starbox, axis=1), axis=1))) + lbx
    cy      = (np.ma.sum(np.ma.sum(Y*starbox, axis=1), axis=1)/(np.ma.sum(np.ma.sum(starbox, axis=1), axis=1))) + lby
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    widx    = np.ma.sqrt(np.ma.sum(np.ma.sum(X2*starbox, axis=1), axis=1)/(np.ma.sum(np.ma.sum(starbox, axis=1), axis=1)))
    widy    = np.ma.sqrt(np.ma.sum(np.ma.sum(Y2*starbox, axis=1), axis=1)/(np.ma.sum(np.ma.sum(starbox, axis=1), axis=1)))
    return cx.ravel(), cy.ravel(), widx.ravel(), widy.ravel()

def A_photometry(image_data, factor = 0.029691810510039204,
    cx = 15, cy = 15, r = 2.5, a = 5, b = 5, w_r = 5, h_r = 5, 
    theta = 0, shape = 'Circular', method='center'):
    l, h, w = image_data.shape
    position= [cx, cy]
    ape_sum = []
    if   (shape == 'Circular'):
        aperture = CircularAperture(position, r=r)
    elif (shape == 'Elliptical'):
        aperture = EllipticalAperture(position, a=a, b=b, theta=theta)
    elif (shape == 'Rectangular'):
        aperture = RectangularAperture(position, w=w_r, h=h_r, theta=theta)
    for i in range(l):
        phot_table = aperture_photometry(image_data[i,:,:], aperture)
        ape_sum.extend(phot_table['aperture_sum']*factor)
    return ape_sum


# Noise pixel param
def noisepixparam(image_data,edg):
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
    return npp  

def bgnormalize(image_data,normbg):
    x=np.ndarray( shape=(64,32,32))
    xmask=np.ma.make_mask(x,copy=True, shrink=True)
    xmask[:,:,:]= False
    xmask[:,13:18,13:18]=True
    masked= np.ma.masked_array(image_data, mask=xmask)
    bgsum = np.zeros(64)
    # Replace for loop with one line code
    for i in range (64):
        bgsum[i] = np.ma.mean(masked[i]) #np.ma.mean
    #background average for the datecube
    bgdcbavg= np.ma.median(bgsum)
    #Normalize
    bgsum=bgsum/bgdcbavg
    normbg.append(bgsum)
    
def normstar(ape_sum,normf):
    starmean=np.ma.median(ape_sum)
    ape_sum=ape_sum/starmean
    normf.append(ape_sum)
    
def normxycent(xo,yo,normx,normy):
    xo=xo/np.ma.median(xo)
    yo=yo/np.ma.median(yo)
    normx.append(xo)
    normy.append(yo)

def normpsfwidth(psfwx,psfwy,normpsfwx,normpsfwy):
    psfwx=psfwx/np.ma.median(psfwx)
    psfwy=psfwy/np.ma.median(psfwy)
    normpsfwx.append(psfwx)
    normpsfwy.append(psfwy) 

def normnoisepix(npp,normnpp):
    npp = npp/np.ma.median(npp)
    normnpp.append(npp)

def stackit(normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp):
    normf=np.ma.median(normf,axis=0)
    normbg=np.ma.median(normbg, axis=0)
    normx=np.ma.median(normx,axis=0)
    normy=np.ma.median(normy,axis=0)
    normpsfwx=np.ma.median(normpsfwx,axis=0)
    normpsfwy=np.ma.median(normpsfwy,axis=0)
    normnpp=np.ma.median(normnpp,axis=0)
    return normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp
 
def plotcurve(xax,f,b,X,Y,wx,wy,npp,direc,ct, f_med, f_std, b_med, b_std, x_med, x_std, y_med, y_std, xw_med, xw_std, yw_med, yw_std, npp_med, npp_std, savepath, channel):
    devfactor=2
    fmed=np.ma.median(f)
    fstdev=np.ma.std(f)
    lb=fmed-devfactor*fstdev
    ub=fmed+devfactor*fstdev
    avoid=[]
    i=0
    for x in (0,57):
        if( f[x] <=lb or f[x]>=ub):
            avoid.append(x)
    #print (avoid)
    fig, axes = plt.subplots(nrows=7, ncols=1, sharex=True)
    fig.set_figheight(8)
    plt.minorticks_on()
    fig.subplots_adjust(hspace = 0.001)
    plt.rc('font', family='serif')
    
    plt.xlim(0,64)
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    if 0 not in (avoid):
        axes[0].plot(xax,f,color='k', mec ='b', marker='s', markevery=[0],fillstyle='none')
    if 57 not in (avoid):
        axes[0].plot(xax,f,color='k', mec ='b', marker='s', markevery=[57],fillstyle='none')

    axes[0].plot(xax,f,color='k', mec ='b', marker='s', markevery=[0],fillstyle='none')
    axes[0].set_ylabel(r'$F$',fontsize=16)
    axes[0].yaxis.set_major_formatter(y_formatter)
    axes[0].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[0].axhline(y = f_med, color='black', linewidth = 1, label = 'Median')
    axes[0].axhline(y = f_med - f_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[0].axhline(y = f_med + f_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

    bmed=np.ma.median(b)
    bstdev=np.ma.std(b)    
    blb=bmed-devfactor*bstdev
    bub=bmed+devfactor*bstdev

    axes[1].plot(xax,b,color='k', mec ='b', marker='s', markevery=[57],fillstyle='none')
    axes[1].plot(xax,b,color='k', mec ='b', marker='s', markevery=[0],fillstyle='none')
    axes[1].set_ylabel(r'$b$',fontsize=16)
    axes[1].yaxis.set_major_formatter(y_formatter)
    axes[1].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[1].axhline(y = b_med, color='black', linewidth = 1, label = 'Median')
    axes[1].axhline(y = b_med - b_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[1].axhline(y = b_med + b_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

    axes[2].plot(xax,X,color='k', mec ='b',marker='s', markevery=[57],fillstyle='none')
    axes[2].plot(xax,X,color='k', mec ='b',marker='s', markevery=[0],fillstyle='none')
    axes[2].set_ylabel(r'$x_0$',fontsize=16)
    axes[2].yaxis.set_major_formatter(y_formatter)
    axes[2].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[2].axhline(y = x_med, color='black', linewidth = 1, label = 'Median')
    axes[2].axhline(y = x_med - x_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[2].axhline(y = x_med + x_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

    axes[3].plot(xax,Y,color='k' , mec ='b', marker='s', markevery=[57],fillstyle='none')
    axes[3].plot(xax,Y,color='k' , mec ='b', marker='s', markevery=[0],fillstyle='none')    
    axes[3].set_ylabel(r'$y_0$',fontsize=16)
    axes[3].yaxis.set_major_formatter(y_formatter)
    axes[3].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[3].axhline(y = y_med, color='black', linewidth = 1, label = 'Median')
    axes[3].axhline(y = y_med - y_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[3].axhline(y = y_med + y_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

    axes[4].plot(xax,wx,color='k' , mec ='b', marker='s', markevery=[57],fillstyle='none')
    axes[4].plot(xax,wx,color='k' , mec ='b', marker='s', markevery=[0], fillstyle='none')
    axes[4].set_ylabel(r'$\sigma_x$',fontsize=16)
    axes[4].yaxis.set_major_formatter(y_formatter)
    axes[4].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[4].axhline(y = xw_med, color='black', linewidth = 1, label = 'Median')
    axes[4].axhline(y = xw_med - xw_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[4].axhline(y = xw_med + xw_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

    axes[5].plot(xax,wy,color='k' , mec ='b', marker='s', markevery=[57],fillstyle='none')
    axes[5].plot(xax,wy,color='k' , mec ='b', marker='s', markevery=[0],fillstyle='none')
    axes[5].set_ylabel(r'$\sigma_y$', fontsize=16)
    axes[5].yaxis.set_major_formatter(y_formatter)
    axes[5].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[5].axhline(y = yw_med, color='black', linewidth = 1, label = 'Median')
    axes[5].axhline(y = yw_med - yw_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[5].axhline(y = yw_med + yw_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)    

    axes[6].plot(xax,npp,color='k' , mec ='b', marker='s', markevery=[57],fillstyle='none')
    axes[6].plot(xax,npp,color='k' , mec ='b', marker='s', markevery=[0],fillstyle='none')
    axes[6].set_ylabel(r'$\beta$', fontsize=16)
    axes[6].set_xlabel('Frame number',fontsize=16)
    axes[6].yaxis.set_major_formatter(y_formatter)
    axes[6].yaxis.set_major_locator(MaxNLocator(prune='both',nbins=5))
    axes[6].axhline(y = npp_med, color='black', linewidth = 1, label = 'Median')
    axes[6].axhline(y = npp_med - npp_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[6].axhline(y = npp_med + npp_std, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    axes[6].set_xlim((-0.5, 63.5))

    if channel == 'ch1':
        wav='3.6'
    else:
        wav='4.5'
    plt.savefig(savepath+wav+'_' +str(ct)+'_' +direc+'.pdf', bbox_inches='tight', dpi=200)
    
def load_data(path, AOR):
    pathflux  = path + 'flux'  + AOR + '.npy'
    pathbg    = path + 'bg'    + AOR + '.npy'
    pathxdata = path + 'xdata' + AOR + '.npy'
    pathydata = path + 'ydata' + AOR + '.npy'
    pathpsfwx = path + 'psfwx' + AOR + '.npy'
    pathpsfwy = path + 'psfwy' + AOR + '.npy'
    pathbeta  = path + 'beta'  + AOR + '.npy'
    
    flux  = np.load(pathflux )
    bg    = np.load(pathbg   )
    xdata = np.load(pathxdata)
    ydata = np.load(pathydata)
    psfwx = np.load(pathpsfwx)
    psfwy = np.load(pathpsfwy)
    beta  = np.load(pathbeta )
    
    return flux, bg, xdata, ydata, psfwx, psfwy, beta 

def sigclip(data, sigma=3, maxiters=5):
    new_data = sigma_clip(data, sigma=sigma, maxiters=maxiters)
#     print(data.shape)
#     print(np.where(data!=new_data)[0])
    return new_data

def get_stats(data, median_arr, std_arr):
    for i in range(np.array(data).shape[0]):
        median = np.ma.median(data[i], axis = 0)
        std    = np.ma.std(data[i], axis = 0)
        median_arr = np.append(median_arr, [median], axis = 0)
        std_arr    = np.append(std_arr, [std], axis = 0)
    return median_arr, std_arr

def run_diagnostics(planet, channel, AOR_snip, basepath, addStack, nsigma=3):
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
    dirs = [k for k in dirs_all if AOR_snip in k]
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
            image_data2=sigma_clipping(image_data1, tossed)
            #b should be calculated on sigmaclipped data
            bgnormalize(image_data2,normbg)
            #bg subtract
            image_data3=bgsubtract(image_data2)
            #centroid
            xo, yo, psfwx,psfwy = centroid_FWM(image_data3) # xo, yo, psxfwx, psfwy are temp
            ape_sum = A_photometry(np.ma.masked_invalid(image_data3))
            npp=noisepixparam(image_data3,4)        
            normstar(np.ma.masked_invalid(ape_sum),normf)
            normxycent(xo,yo,normx,normy)
            normpsfwidth(psfwx,psfwy,normpsfwx,normpsfwy)
            normnoisepix(npp,normnpp)
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
    
    
    AOR = [a for a in os.listdir(datapath) if AOR_snip in a]
    data = [np.asarray(load_data(savepath, a)) for a in AOR]
    nb_data = [len(data[i]) for i in range(len(data))]
    data = [np.where(np.isfinite(data[i]), data[i], 99999) for i in range(len(data))]

    flux  = np.array([sigclip(data[i][0]) for i in range(len(data))])
    bg    = np.array([sigclip(data[i][1]) for i in range(len(data))])
    xdata = np.array([sigclip(data[i][2]) for i in range(len(data))])
    ydata = np.array([sigclip(data[i][3]) for i in range(len(data))])
    psfwx = np.array([sigclip(data[i][4]) for i in range(len(data))])
    psfwy = np.array([sigclip(data[i][5]) for i in range(len(data))])
    beta  = np.array([sigclip(data[i][6]) for i in range(len(data))])

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
        flux_all = sigma_clip(flux_all, sigma=4, maxiters=5)

    bg_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        bg_all = np.append(bg_all, bg[i], axis = 0)
        bg_all = np.where(np.isfinite(bg_all), bg_all, 99999)
        bg_all = sigma_clip(bg_all, sigma=2, maxiters=2)

    xdata_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        xdata_all = np.append(xdata_all, xdata[i], axis = 0)
        xdata_all = sigma_clip(xdata_all, sigma=4, maxiters=5)

    ydata_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        ydata_all = np.append(ydata_all, ydata[i], axis = 0)
        ydata_all = sigma_clip(ydata_all, sigma=4, maxiters=5)

    psfwx_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        psfwx_all = np.append(psfwx_all, psfwx[i], axis = 0)
        psfwx_all = sigma_clip(psfwx_all, sigma=4, maxiters=5)

    psfwy_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        psfwy_all = np.append(psfwy_all, psfwy[i], axis = 0)
        psfwy_all = sigma_clip(psfwy_all, sigma=4, maxiters=5)

    beta_all = np.empty((0, 64))
    for i in range(np.array(flux).shape[0]):
        beta_all = np.append(beta_all, beta[i], axis = 0)
        beta_all = sigma_clip(beta_all, sigma=4, maxiters=5)

    flux_med, flux_err = np.ma.median(flux_all, axis = 0), np.ma.std(flux_all, axis = 0)/1374
    bg_med, bg_err = np.ma.median(bg_all, axis = 0), np.ma.std(bg_all, axis = 0)/1374
    xdata_med, xdata_err = np.ma.median(xdata_all, axis = 0), np.ma.std(xdata_all, axis = 0)/1374
    ydata_med, ydata_err = np.ma.median(ydata_all, axis = 0), np.ma.std(ydata_all, axis = 0)/1374
    psfwx_med, psfwx_err = np.ma.median(psfwx_all, axis = 0), np.ma.std(psfwx_all, axis = 0)/1374
    psfwy_med, psfwy_err = np.ma.median(psfwy_all, axis = 0), np.ma.std(psfwy_all, axis = 0)/1374
    beta_med, beta_err = np.ma.median(beta_all, axis = 0), np.ma.std(beta_all, axis = 0)/1374

    # if planet == 'WASP-12b':
    #     bgall = np.concatenate((bg[0], bg[1], bg[2]), axis=0)
    # else:
    #     bgall = np.concatenate((bg[0], bg[1]), axis=0)

    bgall = np.ma.concatenate([bg[i] for i in range(bg.shape[0])], axis=0)
    # bgall = bg.reshape(-1,bg.shape[2])

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
        np.append(index, np.where(flux_med > (meanflux + nsigma*sigmaflux)))
        sigmaflux2 = np.ma.std(np.delete(flux_med, index))
        flag = (sigmaflux2 == sigmaflux)
        sigmaflux = sigmaflux2


    flag = False
    while(flag == False):
        index = np.where(bg_med < (meanbg - nsigma*sigmabg))
        np.append(index, np.where(bg_med > (meanbg + nsigma*sigmabg)))
        sigmabg2 = np.ma.std(np.delete(bg_med, index))
        flag = (sigmabg2 == sigmabg)
        sigmabg = sigmabg2

    flag = False
    while(flag == False):
        index = np.where(xdata_med < (meanxdata - nsigma*sigmaxdata))
        np.append(index, np.where(xdata_med > (meanxdata + nsigma*sigmaxdata)))
        sigmaxdata2 = np.ma.std(np.delete(xdata_med, index))
        flag = (sigmaxdata2 == sigmaxdata)
        sigmaxdata = sigmaxdata2

    flag = False
    while(flag == False):
        index = np.where(ydata_med < (meanydata - nsigma*sigmaydata))
        np.append(index, np.where(ydata_med > (meanydata + nsigma*sigmaydata)))
        sigmaydata2 = np.ma.std(np.delete(ydata_med, index))
        flag = (sigmaydata2 == sigmaydata)
        sigmaydata = sigmaydata2

    flag = False
    while(flag == False):
        index = np.where(psfwx_med < (meanpsfwx - nsigma*sigmapsfwx))
        np.append(index, np.where(psfwx_med > (meanpsfwx + nsigma*sigmapsfwx)))
        sigmapsfwx2 = np.ma.std(np.delete(psfwx_med, index))
        flag = (sigmapsfwx2 == sigmapsfwx)
        sigmapsfwx = sigmapsfwx2

    flag = False
    while(flag == False):
        index = np.where(psfwy_med < (meanpsfwy - nsigma*sigmapsfwy))
        np.append(index, np.where(psfwy_med > (meanpsfwy + nsigma*sigmapsfwy)))
        sigmapsfwy2 = np.ma.std(np.delete(psfwy_med, index))
        flag = (sigmapsfwy2 == sigmapsfwy)
        sigmapsfwy = sigmapsfwy2

    flag = False
    while(flag == False):
        index = np.where(beta_med < (meanbeta - nsigma*sigmabeta))
        np.append(index, np.where(beta_med > (meanbeta + nsigma*sigmabeta)))
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