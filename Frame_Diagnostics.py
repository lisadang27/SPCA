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
#from photutils.morphology import centroid_1dg,centroid_2dg
#np.set_printoptions(threshold=np.nan)

tossed=0

def sigma_clipping(image_data, bounds = (13, 18, 13, 18)):#,fname):
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
            #badframetable.append([i,filenb,fname])
            tossed += 1
            print('tossed:', tossed)
    return sig_clipped_data

def bgsubtract(image_data):
    bgsubimg=image_data
    x=np.ndarray ( shape=(64,32,32), dtype=bool)
    xmask=np.ma.make_mask(x,copy=True, shrink=True, dtype=np.bool)
    xmask[:,:,:]= False
    xmask[:,11:19,11:19]=True
    #xmask[:,0:1,:]=True
    masked= np.ma.masked_array(bgsubimg, mask=xmask)
    n=0
    #Background subtraction for each frame
    while(n<64):
        bg_avg=np.ma.median(masked[n])

        # np.nanmedian functions weirdly for some reason.
        # below code to calculate median by ignoring nan values
        '''
        copy=[]
        for i in range (32):
            for j in range (32):
                if(masked[n][i][j]!= np.nan):
                    copy.append( masked[n,i,j])
        bg_avg=np.median(copy)
        '''
        bgsubimg[n]= bgsubimg[n,:,:] - bg_avg
        n+=1
    return bgsubimg

def centroid_FWM(image_data, scale = 1, bounds = (13, 18, 13, 18)):
    lbx, ubx, lby, uby = bounds
    lbx, ubx, lby, uby = lbx*scale, ubx*scale, lby*scale, uby*scale
    starbox = image_data[:, lbx:ubx, lby:uby]
    h, w, l = starbox.shape
    # get centroid  
    X, Y    = np.mgrid[:w,:l]
    cx      = (np.sum(np.sum(X*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lbx
    cy      = (np.sum(np.sum(Y*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1))) + lby
    #xo.extend(cx/scale)
    #yo.extend(cy/scale)
    # get PSF widths
    X, Y    = np.repeat(X[np.newaxis,:,:], h, axis=0), np.repeat(Y[np.newaxis,:,:], h, axis=0)
    cx, cy  = np.reshape(cx, (h, 1, 1)), np.reshape(cy, (h, 1, 1))
    X2, Y2  = (X + lbx - cx)**2, (Y + lby - cy)**2
    widx    = np.sqrt(np.sum(np.sum(X2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    widy    = np.sqrt(np.sum(np.sum(Y2*starbox, axis=1), axis=1)/(np.sum(np.sum(starbox, axis=1), axis=1)))
    #wx.extend(widx/scale)
    #wy.extend(widy/scale)
    return cx.ravel(), cy.ravel(), widx.ravel(), widy.ravel()

'''def centroid(image_data):
    # Refer: Intra-Pixel Gain Variations and High-Precision Photometry with the Infrared Array Camera (IRAC)
    cx=np.zeros(64)
    cy=np.zeros(64)
    starbox = image_data[:, 13:18, 13:18]
    h,w = np.shape(starbox[0,:,:])
    x = np.arange(0,w)
    y = np.arange(0,h)
    X,Y = np.meshgrid(x,y)
    for i in range(64):
        cx[i]=(np.sum(X*starbox[i,:,:])/np.sum(starbox[i,:,:]))+13
        cy[i]=(np.sum(Y*starbox[i,:,:])/np.sum(starbox[i,:,:]))+13
    return cx,cy'''

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
        #data_error = calc_total_error(image_data[i,:,:], bg_err[i], effective_gain=1)
        phot_table = aperture_photometry(image_data[i,:,:], aperture)
        #if (phot_table['aperture_sum_err'] > 0.000001):
        ape_sum.extend(phot_table['aperture_sum']*factor)
            #ape_sum_err.extend(phot_table['aperture_sum_err']*factor)
        #else:
            #ape_sum.extend([np.nan])
            #ape_sum_err.extend([np.nan])
    return ape_sum

#edg is the edge(in pixels) of the starbox
'''def psfwidth(image_data,xo,yo,edg):
    psfxw=np.zeros(64)
    psfyw=np.zeros(64)
    lb= 14
    ub= 18
    stx=np.ndarray((64,4,4))
    np.copyto(stx,image_data[:,lb:ub,lb:ub])
    for i in range(64):
        denom=0.0
        numerx=0.0
        numery=0.0
        for j in range(edg):
            for k in range(edg):
                f=stx[i][j][k]
                # lower bound to be added 
                numerx+=f*(j-xo[i]+lb)*(j-xo[i]+lb)
                numery+=f*(k-yo[i]+lb)*(k-yo[i]+lb)

        denom=np.nansum(stx[i,:,:])
        widx=numerx/denom
        widy=numery/denom
        widx=np.sqrt(widx)
        widy=np.sqrt(widy)
        psfxw[i]=widx
        psfyw[i]=widy
    
    return psfxw,psfyw'''

# Noise pixel param
def noisepixparam(image_data,edg):
    lb= 13
    ub= 18
    npp=[]
    # Its better to operate on the copy of desired portion of image_data than on image_data itself.
    # This reduces the risk of modifying image_data accidently. Arguements are passed as pass-by-object-reference.
    stx=np.ndarray((64,5,5))
    
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
    #print masked[5][12:19,12:19]
    bgsum = np.zeros(64)
    # Replace for loop with one line code
    for i in range (64):
        bgsum[i] = np.nanmean(masked[i]) #np.ma.mean
    #background average for the datecube
    bgdcbavg= np.nanmedian(bgsum)
    #Normalize
    bgsum=bgsum/bgdcbavg
    normbg.append(bgsum)
    
    #print " normal ", bgsum[5]
    #bg_avg = np.mean(bgsum)
    #bgsum=bgsum/
def normstar(ape_sum,normf):
    starmean=np.ma.median(ape_sum)
    ape_sum=ape_sum/starmean
    normf.append(ape_sum)
    #print min(enumerate(normf), key=operator.itemgetter(1))
    
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
 
def plotcurve(xax,f,b,X,Y,wx,wy,npp,direc,ct, f_med, f_std, b_med, b_std, x_med, x_std, y_med, y_std, xw_med, xw_std, yw_med, yw_std, npp_med, npp_std):
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
    
    #fig.subplots_adjust(.15,.15,.9,.9,0,0)
    plt.xlim(0,64)
    y_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    #axes[0].plot(xax,f,color='k', mec ='r', marker='x', markevery=avoid,fillstyle='none')
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

    #axes[0].axhline(y = fmed, color='black', linewidth = 1, label = 'Median')
    #axes[0].axhline(y = lb, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    #axes[0].axhline(y = ub, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

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
    #axes[1].axhline(y = bmed, color='black', linewidth = 1, label = 'Median')
    #axes[1].axhline(y = blb, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)
    #axes[1].axhline(y = bub, color='black', linewidth = 1, label = '$2 \sigma$', alpha = 0.3)

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


    plt.savefig('C:/Users/Lisa/Desktop/4.5_'+str(ct)+'_'+direc+'.pdf',bbox_inches='tight',dpi=200)
    #plt.savefig('C:/Users/Lisa/Documents/Exoplanets/high_precision_photometry/Plots/4.5_'+str(ct)+'_'+direc+'.pdf',bbox_inches='tight',dpi=200)
    #plt.savefig('4.5_'+str(ct)+'_'+direc+'.pdf',bbox_inches='tight',dpi=200)

# Ignore warning and starts timing
warnings.filterwarnings('ignore')
#outerpath='/home/hema/Documents/mcgill/handy/aorkeys-20-selected_AORs'
#outerpath = 'C:/Users/Lisa/Desktop'
outerpath = 'D:/Spitzer_Data/XO-3b/Phase_ch2'
dirs_all = os.listdir(outerpath)
dirs = [k for k in dirs_all if 'r464' in k]
#dirs = np.delete(dirs, [0, 4])
#dirs=os.listdir(outerpath)
print (dirs)
counter=0
ct = 0

# Sky dark correction
#darkpath = 'C:/Users/Lisa/Dropbox/CoRoT-2b/Spitzer/correction_stack_MJy_per_ster__add_this_to_each_bcd.fits'
#hdu_list = fits.open(darkpath, mode='readonly')
#skydark  = hdu_list[0].data

for direc in dirs :
    print (direc)
    #Normalised and stacked
    #if(counter==2):
    #   break
    normbg=[]
    normf=[]
    normx=[]
    normy=[]
    normpsfwx=[]
    normpsfwy=[]
    normnpp=[]
    #path='/home/hema/Documents/mcgill/handy/aorkeys-20-selected_AORs/'+direc+'/ch2/bcd'
    path = 'D:/Spitzer_Data/XO-3b/Phase_ch2/'+direc+'/ch2/bcd'
    #path = 'C:/Users/Lisa/Desktop/'+direc+'/ch2/bcd'
    print (path)
    #xn=1
    for filename in glob.glob(os.path.join(path, '*bcd.fits')):
        print (ct)
        #print filename
        f=fits.open(filename,mode='readonly')
        # get data and apply sky dark correction
        image_data0 = f[0].data # + skydark
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
        #apply gaussian 2d fit and find the centroid
        #gx, gy = centroidg2d(image_data3)
        #aperture photmetry
        ape_sum = A_photometry(np.ma.masked_invalid(image_data3))
        #psfwx,psfwy=psfwidth(image_data3,xo,yo,4)
        npp=noisepixparam(image_data3,4)        
        normstar(np.ma.masked_invalid(ape_sum),normf)
        #print(l(normf))
        normxycent(xo,yo,normx,normy)
        normpsfwidth(psfwx,psfwy,normpsfwx,normpsfwy)
        normnoisepix(npp,normnpp)
        ct+=1
        #sprint(normf)
    print (ct)
    #Since we are appending and not extending lists, we needn't reshape it
    #normf,normbg,normx,normy,normpsfwx,normpsfwy=reshapelists(normf,normbg,normx,normy,normpsfwx,normpsfwy,ct)
    #normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp=stackit(normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp)
    #frameno=np.arange(0,64)
    #plotcurve(frameno,normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp,direc,ct)
    counter+=1
    
    pathFULL  = 'C:/Users/Lisa/Desktop/Phase_Curves/ch2/Frame_Diagnostics/'
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

    #normf,normbg,normx,normy,normpsfwx,normpsfwy=reshapelists(normf,normbg,normx,normy,normpsfwx,normpsfwy,ct)
    normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp = stackit(normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp)

# SAVE VALUES
FULL_data = np.c_[normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp]
#save it is plots...
pathFULL2  = pathFULL+'Frame_Diag_Results_pre.dat'
FULL_head = 'Flux, bg level, x, y, psfwx, psfwy, npp'
np.savetxt(pathFULL2, FULL_data, header = FULL_head)


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
plotcurve(frameno,normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp,'all',ct, f_med, f_std, b_med, b_std, x_med, x_std, y_med, y_std, xw_med, xw_std, yw_med, yw_std, npp_med, npp_std)
plotcurve(frameno,normf,normbg,normx,normy,normpsfwx,normpsfwy,normnpp,'all01',ct, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05, 1, 0.05)