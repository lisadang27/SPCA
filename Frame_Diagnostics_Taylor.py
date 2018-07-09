# To justify why 1st and 58 frames are outliers
# Do an aperture photometry and verify.

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






#####################################################################
#Only these values need to be personalized

planet = 'WASP-12b_old' #planet name
channel = 'ch2'         #Spitzer IRAC channel
AOR_snip = 'r4'         #bit of AOR to pick out which folders contain AORs that should be analyzed
basepath = '/home/taylor/Documents/Research/spitzer/'+planet+'/'  #folder containing data to be analyzed
savepath = basepath+'analysis/FrameDiagnostics/'
stackpath = '/home/taylor/Documents/Research/spitzer/Calibration/' #folder containing properly names correction stacks (will be automatically selected)
addStack = True        #do you want to add a correction stack to fix bad backgrounds

#####################################################################








def get_stacks(stackpath, basepath, AOR_snip, ch):
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

    datapath = basepath+'data/'+ch

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



def sigma_clipping(image_data, bounds = ( 14, 18, 14, 18)):#,fname):
    global tossed
    lbx, ubx, lby, uby = bounds
    h, w, l = image_data.shape
    # mask invalids
    image_data2 = np.ma.masked_invalid(image_data)
    # make mask to mask entire bad frame
    x = np.ones(shape = (w, l))
    mask = np.ma.make_mask(x)
    sig_clipped_data = sigma_clip(image_data2, sigma=4, iters=4, cenfunc=np.ma.median, axis = 0)
    for i in range (h):
        oldstar = image_data[i, lbx:ubx, lby:uby]
        newstar = sig_clipped_data[i, lbx:ubx, lby:uby]
        truth   = newstar==oldstar
        if(truth.sum() < truth.size):
            sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
            tossed += 1
            print('tossed:', tossed)
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
    cx      = (np.sum(np.sum(X*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lbx
    cy      = (np.sum(np.sum(Y*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lby
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    widx    = np.sqrt(np.sum(np.sum(X2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    widy    = np.sqrt(np.sum(np.sum(Y2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
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
        numer= np.nansum(img)
        numer=np.square(numer)
        denom=0.0
        temp = np.square(img)
        denom = np.nansum(img)
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
        bgsum[i] = np.nanmean(masked[i]) #np.ma.mean
    #background average for the datecube
    bgdcbavg= np.nanmedian(bgsum)
    #Normalize
    bgsum=bgsum/bgdcbavg
    normbg.append(bgsum)
    
def normstar(ape_sum,normf):
    starmean=np.ma.median(ape_sum)
    ape_sum=ape_sum/starmean
    normf.append(ape_sum)
    
def normxycent(xo,yo,normx,normy):
    xo=xo/np.nanmedian(xo)
    yo=yo/np.nanmedian(yo)
    normx.append(xo)
    normy.append(yo)

def normpsfwidth(psfwx,psfwy,normpsfwx,normpsfwy):
    psfwx=psfwx/np.nanmedian(psfwx)
    psfwy=psfwy/np.nanmedian(psfwy)
    normpsfwx.append(psfwx)
    normpsfwy.append(psfwy) 

def normnoisepix(npp,normnpp):
    npp = npp/np.nanmedian(npp)
    normnpp.append(npp)

def stackit(normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp):
    normf=np.nanmedian(normf,axis=0)
    normbg=np.nanmedian(normbg, axis=0)
    normx=np.nanmedian(normx,axis=0)
    normy=np.nanmedian(normy,axis=0)
    normpsfwx=np.nanmedian(normpsfwx,axis=0)
    normpsfwy=np.nanmedian(normpsfwy,axis=0)
    normnpp=np.nanmedian(normnpp,axis=0)
    return normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp
 
def plotcurve(xax,f,b,X,Y,wx,wy,npp,direc,ct, f_med, f_std, b_med, b_std, x_med, x_std, y_med, y_std, xw_med, xw_std, yw_med, yw_std, npp_med, npp_std, savepath, channel):
    devfactor=2
    fmed=np.nanmedian(f)
    fstdev=np.std(f)
    lb=fmed-devfactor*fstdev
    ub=fmed+devfactor*fstdev
    avoid=[]
    i=0
    for x in (0,57):
        if( f[x] <=lb or f[x]>=ub):
            avoid.append(x)
    print (avoid)
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

    bmed=np.nanmedian(b)
    bstdev=np.std(b)    
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
    plt.savefig(savepath+channel+'/'+wav+'_' +str(ct)+'_' +direc+'.pdf', bbox_inches='tight', dpi=200)



if addStack:
    stacks = get_stacks(stackpath, basepath, AOR_snip, channel)

outerpath = basepath+'data/'+channel+'/'
dirs_all = os.listdir(outerpath)
dirs = [k for k in dirs_all if AOR_snip in k]
print (dirs)
tossed = 0
counter=0
ct = 0

for i in range(len(dirs)):

    direc = dirs[i]

    if addStack:
        stack = fits.open(stacks[i], mode='readonly')
        skydark = stack[0].data

    print (direc)
    normbg=[]
    normf=[]
    normx=[]
    normy=[]
    normpsfwx=[]
    normpsfwy=[]
    normnpp=[]
    path = outerpath+direc+'/'+channel+'/bcd'
    print (path)
    for filename in glob.glob(os.path.join(path, '*bcd.fits')):
        print (ct)
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
        image_data2=sigma_clipping(image_data1)
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
    print (ct)
    counter+=1
    
    pathFULL  = basepath+'analysis/FrameDiagnostics/'+channel+'/'
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

    normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp = stackit(normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp)

    frameno=np.arange(0,64)
    f_med = np.nanmean(normf)
    f_std = np.nanstd(normf)
    b_med = np.nanmean(normbg)
    b_std = np.nanstd(normbg)
    x_med = np.nanmean(normx)
    x_std = np.nanstd(normx)
    y_med = np.nanmean(normy)
    y_std = np.nanstd(normy)
    xw_med = np.nanmean(normpsfwx)
    xw_std = np.nanstd(normpsfwx)
    yw_med = np.nanmean(normpsfwy)
    yw_std = np.nanstd(normpsfwy)
    npp_med = np.nanmean(normnpp)
    npp_std = np.nanstd(normnpp)
    plotcurve(frameno,normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp,'all',ct, f_med, f_std, b_med, b_std, x_med, x_std, y_med, y_std, xw_med, xw_std, yw_med, yw_std, npp_med, npp_std, savepath, channel)
    plotcurve(frameno,normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp,'all01',ct, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, savepath, channel)
