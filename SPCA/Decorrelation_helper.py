import scipy
import scipy.stats as sp
import scipy.optimize as spopt

import emcee
import corner

from astropy import constants as const
from astropy import units

import numpy as np
import time as t
import os, sys, errno
import csv
from tqdm import tqdm

from multiprocessing import Pool
from threadpoolctl import threadpool_limits

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.time
from astropy.stats import sigma_clip
from astropy.table import Table
from astropy.io import fits

import urllib.request

# SPCA libraries
from SPCA import helpers, astro_models, make_plots, make_plots_custom, detec_models, bliss, freeze


# FIX: Add a docstring for this function
def downloadExoplanetArchive():
    #Download the most recent masterfile of the best data on each target
    try:
        _ = urllib.request.urlretrieve('http://www.astro.umontreal.ca/~adb/masterfile.ecsv', './masterfile.ecsv')
    except:
        print('Unable to download the most recent Exoplanet Archive data - analyses will resort to a previously downloaded version if available.')
    return

# FIX: Add a docstring for this function
def loadArchivalData(rootpath, planet, channel):
    if os.path.exists('./masterfile.ecsv'):
        data = Table.to_pandas(Table.read('./masterfile.ecsv'))
    else:
        # Fix: throw a proper error
        print('ERROR: No previously downloaded Exoplanet Archive data - try again when you are connected to the internet.')
        print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), './masterfile.ecsv'))
        return

    names = np.array(data['pl_hostname'])+np.array(data['pl_letter'])
    names = np.array([name.replace(' ','').replace('-', '').replace('_','') for name in names])


    # make params obj
    p0_obj  = helpers.signal_params() 

    # Personalize object default object values
    p0_obj['name'] = planet

    indices = np.where(names==planet.replace(' ','').replace('-', '').split('_')[0])[0]
    if len(indices)==0:
        # Fix: throw a proper error
        print('ERROR: Planet', planet, 'not found in the Exoplanet Archive - please load your own prior values.')
        return
    
    nameIndex = indices[0]

    if np.isfinite(data['pl_ratror'][nameIndex]):
        p0_obj['rp'] = data['pl_ratror'][nameIndex]
    else:
        p0_obj['rp'] = data['pl_rads'][nameIndex]/data['st_rad'][nameIndex]

    if np.isfinite(data['pl_ratdor'][nameIndex]):
        p0_obj['a'] = data['pl_ratdor'][nameIndex]
        p0_obj['a_err'] = np.mean([data['pl_ratdorerr1'][nameIndex],
                                   np.abs(data['pl_ratdorerr2'][nameIndex])])
    else:
        p0_obj['a'] = data['pl_orbsmax'][nameIndex]*const.au.value/data['st_rad'][nameIndex]/const.R_sun.value
        p0_obj['a_err'] = np.sqrt(
            (np.mean([data['pl_orbsmaxerr1'][nameIndex], np.abs(data['pl_orbsmaxerr2'][nameIndex])])*const.au.value
             /data['st_rad'][nameIndex]/const.R_sun.value)**2
            + (data['pl_orbsmax'][nameIndex]*const.au.value
               /data['st_rad'][nameIndex]**2/const.R_sun.value
               *np.mean([data['st_raderr1'][nameIndex], np.abs(data['st_raderr2'][nameIndex])]))**2
        )
    p0_obj['per'] = data['pl_orbper'][nameIndex]
    p0_obj['per_err'] = np.mean([data['pl_orbpererr1'][nameIndex],
                                 np.abs(data['pl_orbpererr2'][nameIndex])])
    p0_obj['t0'] = data['pl_tranmid'][nameIndex]-2.4e6-0.5
    p0_obj['t0_err'] = np.mean([data['pl_tranmiderr1'][nameIndex],
                                np.abs(data['pl_tranmiderr2'][nameIndex])])
    p0_obj['inc'] = data['pl_orbincl'][nameIndex]
    p0_obj['inc_err'] = np.mean([data['pl_orbinclerr1'][nameIndex],
                                 np.abs(data['pl_orbinclerr2'][nameIndex])])
    p0_obj['Tstar'] = data['st_teff'][nameIndex]
    p0_obj['Tstar_err'] = np.mean([data['st_tefferr1'][nameIndex],
                                   np.abs(data['st_tefferr2'][nameIndex])])
    
    p0_obj['logg'] = data['st_logg'][nameIndex]
    p0_obj['feh'] = data['st_metfe'][nameIndex]

    e = data['pl_orbeccen'][nameIndex]
    argp = data['pl_orblper'][nameIndex]

    if e != 0:

        if not np.isfinite(argp):
            print('Randomly generating an argument of periastron...')
            argp = np.random.uniform(0.,360.,1)

        p0_obj['ecosw'] = e/np.sqrt(1+np.tan(argp*np.pi/180.)**2)
        if 90 < argp < 270:
            p0_obj['ecosw']*=-1
        p0_obj['esinw'] = np.tan(argp*np.pi/180.)*p0_obj['ecosw']
        
    # Get the stellar brightness temperature to allow us to invert Plank equation later
    p0_obj['tstar_b'], p0_obj['tstar_b_err'] = getTstarBright(rootpath, planet, channel, p0_obj)
    
    return p0_obj

# FIX: Add a docstring for this function
def loadCustomData(rootpath, planet, channel, rp, a, per, t0, inc, e, argp, Tstar, logg, feh, rp_err=np.inf, a_err=np.inf, t0_err=np.inf, per_err=np.inf, inc_err=np.inf, e_err=np.inf, argp_err=np.inf, Tstar_err=np.inf):
    # make params obj
    p0_obj  = helpers.signal_params() 

    # Personalize object default object values
    p0_obj['name'] = planet
    p0_obj['rp'] = rp
    p0_obj['rp_err'] = np.abs(rp_err)
    p0_obj['a'] = a
    p0_obj['a_err'] = np.abs(a_err)
    p0_obj['per'] = per
    p0_obj['per_err'] = np.abs(per_err)
    p0_obj['t0'] = t0-2.4e6-0.5
    p0_obj['t0_err'] = np.abs(t0_err)
    p0_obj['inc'] = inc
    p0_obj['inc_err'] = np.abs(inc_err)
    p0_obj['Tstar'] = Tstar
    p0_obj['Tstar_err'] = np.abs(Tstar_err)
    p0_obj['logg'] = logg
    p0_obj['feh'] = feh

    if e != 0:

        if not np.isfinite(argp):
            print('Randomly generating an argument of periastron...')
            argp = np.random.uniform(0.,360.,1)

        p0_obj['ecosw'] = e/np.sqrt(1+np.tan(argp*np.pi/180.)**2)
        if 90 < argp < 270:
            p0_obj['ecosw']*=-1
        p0_obj['esinw'] = np.tan(argp*np.pi/180.)*p0_obj['ecosw']
        
    # Get the stellar brightness temperature to allow us to invert Plank equation later
    p0_obj['tstar_b'], p0_obj['tstar_b_err'] = getTstarBright(rootpath, planet, channel, p0_obj)
    
    return p0_obj

# FIX: Add a docstring for this function
def getTstarBright(rootpath, planet, channel, p0_obj):
    # Get the phoenix file ready to compute the stellar brightness temperature
    teffStr = p0_obj['Tstar']
    if teffStr <= 7000:
        teffStr = teffStr - (teffStr%100) + np.rint((teffStr%100)/100)*100
    elif teffStr > 7000:
        teffStr = teffStr - (teffStr%200) + np.rint((teffStr%200)/200)*200
    elif teffStr > 12000:
        teffStr = 12000
    teffStr = str(int(teffStr)).zfill(5)

    logg = p0_obj['logg']
    if np.isnan(logg):
        logg = 4.5
    logg = logg - (logg%0.5) + np.rint((logg%0.5)*2)/2.
    logg = -logg
    feh = p0_obj['feh']
    if np.isnan(feh):
        feh = 0.
    feh = (feh - (feh%0.5) + np.rint((feh%0.5)*2)/2.)
    if feh<-2.:
        feh = (feh - (feh%1) + np.rint((feh%1)))

    webfolder = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'
    phoenixPath = rootpath+planet+'/phoenix/'
    phoenixWavFile = phoenixPath+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    if not os.path.exists(phoenixPath):
        os.mkdir(phoenixPath)
        try:
            _ = urllib.request.urlretrieve(webfolder+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', phoenixWavFile)
        except:
            # Fix: throw a proper error
            print('ERROR: No previously downloaded PHOENIX data - try again when you are connected to the internet.')
            exit()
        print('Done download.')

    webfolder += 'PHOENIX-ACES-AGSS-COND-2011/Z'+("{0:+.01f}".format(feh) if feh!=0 else '-0.0')+'/'

    webfile = ('lte'+teffStr
             +("{0:+.02f}".format(logg) if logg!=0 else '-0.00')
             +("{0:+.01f}".format(feh) if feh!=0 else '-0.0')
             +'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

    phoenixSpectraFile = phoenixPath+webfile

    if not os.path.exists(phoenixSpectraFile):
        print('Downloading relevant PHOENIX spectra...')
        try:
            _ = urllib.request.urlretrieve(webfolder+webfile, phoenixSpectraFile)
        except:
            # Fix: throw a proper error
            print('ERROR: No previously downloaded PHOENIX data - try again when you are connected to the internet.')
            exit()
        print('Done download.')
    
    
    with fits.open(phoenixSpectraFile) as f:
        fStar = f[0].data*1e-1 # 'erg/s/cm^2/cm' to kg/s^3
    with fits.open(phoenixWavFile) as f:
        wavStar = f[0].data*1e-4 # angstrom to micron

    def planck(wav, T):
        intensity = ((2.0*const.h.value*const.c.value**2) /
                     (wav**5 * (np.exp(const.h.value*const.c.value/(wav*const.k_B.value*T)) - 1.0)))
        return intensity
    def fluxDiff(temp, fStarSummed, wavs):
        #factor of pi likely needed to account for emitting area (pi*rstar^2 where rstar=1)
        return (np.sum(planck(wavs, temp)*np.pi)-fStarSummed)**2
    temps = np.linspace(3000, 11000, 801, endpoint=True)
    if channel == 'ch1':
        incides = np.where(np.logical_and(wavStar < 4., wavStar > 3.))[0]
    else:
        incides = np.where(np.logical_and(wavStar < 5., wavStar > 4.))[0]
    diffs = [fluxDiff(temp, np.sum(fStar[incides]), wavStar[incides]*1e-6) for temp in temps]
    tstar_b = temps[np.argmin(diffs)]
    
    # Assuming uncertainty on brightness temperature is close to uncertainty on effective temperature
    return tstar_b, p0_obj['Tstar_err']

# FIX: Add a docstring for this function
def findPhotometry(rootpath, planet, channel, mode, pldIgnoreFrames=True, pldAddStack=False):
    AOR_snip = ''
    with open(rootpath+planet+'/analysis/aorSnippet.txt') as f:
        AOR_snip = f.readline().strip()

    mainpath   = rootpath+planet+'/analysis/'+channel+'/'
    ignoreFrames = np.array([])
    if 'pld' in mode.lower():
        foldername = mainpath

        if pldIgnoreFrames:
            with open(mainpath+'PLD_ignoreFrames.txt') as f:
                line = f.readline()
            ignoreFrames = np.array(line.strip().replace(' ','').split('=')[1].split(','))
            if np.all(ignoreFrames==['']):
                ignoreFrames = np.array([]).astype(int)
            else:
                ignoreFrames = ignoreFrames.astype(int)
        else:
            ignoreFrames = np.array([]).astype(int)

        if pldAddStack:
            foldername += 'addedStack/'
        else:
            foldername += 'addedBlank/'
        if len(ignoreFrames)==0:
            foldername += 'noIgnore/'
        else:
            foldername += 'ignore/'
        if channel=='ch2':
            foldername += '4um'
        else:
            foldername += '3um'
        foldername += 'PLD_'
        foldername += mode.split('x')[0][-1]+'x'+mode.split('x')[1][0]+'/'

    else:
        phoption = ''
        rms = None
        with open(mainpath+'bestPhOption.txt') as f:
            lines = f.readlines()
        for i in range(len(lines)):
            if phoption=='' and lines[i][0]=='/':
                foldername = rootpath+lines[i][lines[i].find(planet):].strip()+'/'
                phoption = lines[i].split('/')[-1].strip()
                i += 1
                ignoreFrames = np.array(lines[i].strip().split('=')[1].replace(' ','').split(','))
                if np.all(ignoreFrames==['']):
                    ignoreFrames = np.array([]).astype(int)
                else:
                    ignoreFrames = ignoreFrames.astype(int)
                i += 1
                rms = float(lines[i])
            elif phoption!='' and lines[i][0]=='/':
                if float(lines[i+2]) < rms:
                    foldername = rootpath+lines[i][lines[i].find(planet):].strip()+'/'
                    phoption = lines[i].split('/')[-1].strip()
                    i += 1
                    ignoreFrames = np.array(lines[i].split('=')[1].replace(' ','').split(','))
                    if np.all(ignoreFrames==['']):
                        ignoreFrames = np.array([]).astype(int)
                    else:
                        ignoreFrames = ignoreFrames.astype(int)
                    i += 1
                    rms = float(lines[i])
                else:
                    i += 3
    
    
    # path where outputs are saved
    savepath   = foldername + mode + '/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    aors = os.listdir(rootpath+planet+'/data/'+channel)
    aors = np.sort([aor for aor in aors if AOR_snip==aor[:len(AOR_snip)]])
    AOR_snip = AOR_snip[1:]
    
    # path to photometry outputs
    filename   = channel + '_datacube_binned_AORs'+AOR_snip+'.dat'
    filename_full  = channel + '_datacube_full_AORs'+AOR_snip+'.dat'
    # Path to previous mcmc results (optional)
    path_params = foldername + mode + '/ResultMCMC_'+mode+'_Params.npy'
    
    return foldername, filename, filename_full, savepath, path_params, AOR_snip, aors, ignoreFrames

# FIX: Add a docstring for this function
def find_breaks(rootpath, planet, channel, aors):
    breaks = []
    for aor in aors:
        rawfiles = np.sort(os.listdir(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'))
        rawfiles  = [rawfile for rawfile in rawfiles if '_bcd.fits' in rawfile]
        rawImage = fits.open(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'+rawfiles[0])

        # Get the time of the first exposure of each AOR after the first
        #     - this allows us to plot dashed lines where AOR breaks happen and where jump discontinuities happen
        breaks.append(rawImage[0].header['BMJD_OBS'] + rawImage[0].header['FRAMTIME']/2/3600/24)
        rawHeader = rawImage[0].header
        rawImage.close()
    
    # Remove the first break which is just the start of observations
    return np.sort(breaks)[1:]

# FIX: Add a docstring for this function
def get_photon_limit(flux, nFrames, ignoreFrames):
    
    return 1/np.sqrt(np.median(flux))/np.sqrt(nFrames-len(ignoreFrames))*1e6

# FIX: Add a docstring for this function
def get_photon_limit_oldData(rootpath, datapath, datapath_aper, planet, channel, mode, aors, nFrames, ignoreFrames):
    #This function is to be used with old photometry to allow for recalculating photon noise limits given the
    #    old data had incorrect units for making photon noise calculations
    aor = aors[-1]
    rawfiles = np.sort(os.listdir(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'))
    rawfiles  = [rawfile for rawfile in rawfiles if '_bcd.fits' in rawfile]
    with fits.open(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'+rawfiles[0]) as rawImage:
        rawHeader = rawImage[0].header
    
    if 'pld' in mode.lower() and 'pldaper' not in mode.lower():
        if '3x3' in mode.lower():
            npix = 3
        elif '5x5' in mode.lower():
            npix = 5
        flux = np.loadtxt(datapath, usecols=list(np.arange(npix**2).astype(int)), skiprows=1)     # mJr/str
        flux = np.sum(flux, axis=1)
    else:
        if 'pldaper' in mode.lower():
            datapath = datapath_aper
        # Calculate the photon noise limit
        flux = np.loadtxt(datapath, usecols=[0], skiprows=1)     # mJr/str
    
    convfact = rawHeader['GAIN']*rawHeader['EXPTIME']/rawHeader['FLUXCONV']
    ecnt2Mjy = - rawHeader['PXSCAL1']*rawHeader['PXSCAL2']*(1/convfact) 
    
    flux /= ecnt2Mjy
    
    return 1/np.sqrt(np.median(flux))/np.sqrt(nFrames-len(ignoreFrames))*1e6

# FIX: Add a docstring for this function
def get_detector_functions(mode):
    signalfunc = detec_models.signal

    if 'poly' in mode.lower():
        detecfunc = detec_models.detec_model_poly
    elif 'pld' in mode.lower():
        detecfunc = detec_models.detec_model_PLD
    elif 'bliss' in mode.lower():
        detecfunc = detec_models.detec_model_bliss
    elif 'gp' in mode.lower():
        detecfunc = detec_models.detec_model_GP
    else:
        raise NotImplementedError('Only Polynomial, PLD, BLISS, and GP models are currently implemented! \nmode=\''+mode+'\' does not include \'poly\', \'Poly\', \'PLD\', \'pld\', \'bliss\', \'BLISS\', \'gp\', or \'GP\'.')
        
    return signalfunc, detecfunc

# FIX: Add a docstring for this function
def setup_gpriors(gparams, p0_obj):
    priors = []
    errs = []
    if 't0' in gparams:
        priors.append(p0_obj['t0'])
        errs.append(p0_obj['t0_err'])
    if 'per' in gparams:
        priors.append(p0_obj['per'])
        errs.append(p0_obj['per_err'])
    if 'a' in gparams:
        priors.append(p0_obj['a'])
        errs.append(p0_obj['a_err'])
    if 'inc' in gparams:
        priors.append(p0_obj['inc'])
        errs.append(p0_obj['inc_err'])
        
    return priors, errs

# FIX: Add a docstring for this function
def reload_old_fit(path_params, p0_obj):
    Table_par = np.load(path_params)                  # table of best-fit params from prev. run
    nparams   = p0_obj['params'][np.logical_not(np.in1d(p0_obj['params'], dparams))]   # get the name list of params to be fitted
    for name in nparams:
        try:
            p0_obj[name]  = Table_par[name][0]
        except Exception as e:
            # FIX: throw a more meaningful error message
            print("type error: " + str(e))            # catch errors if you use values from fun with less params

    return

# FIX: Add a docstring for this function
def print_MCMC_results(time, flux, time_full, flux_full, chain, lnprobchain, p0_labels, p0_astro, mode, channel, p0_obj, signal_inputs, signal_inputs_full, signalfunc, astrofunc, usebestfit, savepath, sigF_photon_ppm, nFrames, secondOrderOffset, compFactor=0):
    #print the results

    ndim = chain.shape[-1]
    samples = chain.reshape((-1, ndim))
    
    MCMC_Results = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
    p0_mcmc = np.median(samples, axis=0)

    # taking max lnprob params instead of median bc degeneracy
    if usebestfit: 
        maxk, maxiter = np.unravel_index((lnprobchain).argmax(), (lnprobchain).shape)
        p0_mcmc = chain[maxk, maxiter,:]
        for i in range(len(p0_mcmc)):
            MCMC_Results[i] = (p0_mcmc[i], MCMC_Results[i][1], MCMC_Results[i][2])

    # adjust fp, sigF, rp, r2 for dilution due to any nearby companion
    if np.any(p0_labels == 'fp'):
        for i in range(3):
            MCMC_Results[np.where(p0_labels == 'fp')[0][0]][i] *= compFactor
    if np.any(p0_labels == 'sigF'):
        for i in range(3):
            MCMC_Results[np.where(p0_labels == 'sigF')[0][0]][i] *= compFactor
    if np.any(p0_labels == 'rp'):
        for i in range(3):
            MCMC_Results[np.where(p0_labels == 'rp')[0][0]][i] *= np.sqrt(compFactor)
    if np.any(p0_labels == 'r2'):
        for i in range(3):
            MCMC_Results[np.where(p0_labels == 'r2')[0][0]][i] *= np.sqrt(compFactor)
            
            
    # printing output from MCMC
    out = "MCMC result:\n\n"
    for i in range(len(p0_mcmc)):
        out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format(p0_labels[i],MCMC_Results[i][0], MCMC_Results[i][1], MCMC_Results[i][2])

    # getting and printing the phase offset
    As = samples[:,np.where(p0_labels == 'A')[0][0]][:,np.newaxis]
    Bs = samples[:,np.where(p0_labels == 'B')[0][0]][:,np.newaxis]
    phis = np.linspace(-np.pi,np.pi,1000)
    offsets = []
    # Doing this in steps to not overflow RAM
    stepSizeOffsets = int(1e2)
    if ('A' in p0_labels)  and ('B' in p0_labels) and (('C' not in p0_labels and 'D' not in p0_labels) or not secondOrderOffset):
        for i in range(int(len(As)/stepSizeOffsets)):
            offsets.extend(-phis[np.argmax(1 + As[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*(np.cos(phis)-1) + Bs[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*np.sin(phis),axis=1)]*180/np.pi)
        if len(As)%stepSizeOffsets != 0:
            offsets.extend(-phis[np.argmax(1 + As[-len(As)%stepSizeOffsets:]*(np.cos(phis)-1) + Bs[-len(As)%stepSizeOffsets:]*np.sin(phis),axis=1)]*180/np.pi)
        offset = np.percentile(np.array(offsets), [16, 50, 84])[[1,2,0]]
        offset[1] -= offset[0]
        offset[2] = offset[0]-offset[2]
        out += '{:>8} = {:>16}  +{:>16}  -{:>16} degrees east\n'.format('Offset', offset[0], offset[1], offset[2])
    elif ('A' in p0_labels)  and ('B' in p0_labels) and ('C' in p0_labels) and ('D' in p0_labels):
        Cs = samples[:,np.where(p0_labels == 'C')[0][0]][:,np.newaxis]
        Ds = samples[:,np.where(p0_labels == 'D')[0][0]][:,np.newaxis]
        for i in range(int(len(As)/stepSizeOffsets)):
            offsets.extend(-phis[np.argmax(1 + As[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*(np.cos(phis)-1) + Bs[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*np.sin(phis) + Cs[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*(np.cos(2*phis)-1) + Ds[i*stepSizeOffsets:(i+1)*stepSizeOffsets]*np.sin(2*phis),axis=1)]*180/np.pi)
        if len(As)%stepSizeOffsets != 0:
            offsets.extend(-phis[np.argmax(1 + As[-len(As)%stepSizeOffsets:]*(np.cos(phis)-1) + Bs[-len(As)%stepSizeOffsets:]*np.sin(phis),axis=1)]*180/np.pi)
        offset = np.percentile(np.array(offsets), [16, 50, 84])[[1,2,0]]
        offset[1] -= offset[0]
        offset[2] = offset[0]-offset[2]
        out += '{:>8} = {:>16}  +{:>16}  -{:>16} degrees east\n'.format('Offset', offsets[0], offsets[1], offsets[2])

    # print the R2/Rp ratio
    if ('ellipse' in mode.lower()) and ('rp' in p0_labels) and ('r2' in p0_labels):
        out += '{:>8} = {:>16}\n'.format('R2/Rp', p0_mcmc[np.where(p0_labels == 'r2')[0][0]]/p0_mcmc[np.where(p0_labels == 'rp')[0][0]])

    if channel == 'ch1':
        wav = 3.6*1e-6
    elif channel == 'ch2':
        wav = 4.5*1e-6
    if 'fp' in p0_labels:
        fp_MCMC = samples[:,np.where(p0_labels == 'fp')[0][0]]*compFactor
    else:
        fp_MCMC = p0_obj['fp']
    if 'rp' in p0_labels:
        rp_MCMC = samples[:,np.where(p0_labels == 'rp')[0][0]]*np.sqrt(compFactor)
    else:
        rp_MCMC = p0_obj['rp']

    tstar_bs = np.random.normal(p0_obj['tstar_b'], p0_obj['tstar_b_err'])

    tday = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_bs))-1)/(fp_MCMC/rp_MCMC**2)))**-1
    tnight = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_bs))-1)/(fp_MCMC*(1-2*As[:,0])/rp_MCMC**2)))**-1

    out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Day: ', np.median(tday), np.percentile(tday, 84)-np.median(tday), np.median(tday)-np.percentile(tday, 16))
    out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Night: ', np.nanmedian(tnight), np.nanpercentile(tnight, 84)-np.nanmedian(tnight), np.nanmedian(tnight)-np.nanpercentile(tnight, 16))
    out += 'For T_{*,b} = '+str(p0_obj['tstar_b'])+'\n'

    print(out)
    with open(savepath+'MCMC_RESULTS_'+mode+'.txt','w') as file:
        file.write(out) 
    
    
    mcmc_signal = signalfunc(signal_inputs, **dict([[p0_labels[i], p0_mcmc[i]] for i in range(len(p0_mcmc))]))
    mcmc_lightcurve = astrofunc(time, **dict([[p0_astro[i], p0_mcmc[:len(p0_astro)][i]] for i in range(len(p0_astro))]))
    mcmc_detec = mcmc_signal/mcmc_lightcurve
    residuals = flux/mcmc_detec - mcmc_lightcurve
    
    #####################
    # Compute the BIC for binned and unbinned data
    #####################
    
    #Binned data
    data = flux/mcmc_detec
    astro  = mcmc_lightcurve
    if 'sigF' in p0_labels:
        sigFMCMC = p0_mcmc[np.where(p0_labels == 'sigF')[0][0]]
    else:
        sigFMCMC = p0_obj['sigF']
    if 'bliss' in mode.lower():
        nKnotsUsed = len(signal_inputs[-4][signal_inputs[-2]])
        ndim_eff = ndim+nKnotsUsed
    else:
        ndim_eff = ndim
    chisB = helpers.chi2(data, astro, sigFMCMC)
    logLB = helpers.loglikelihood(data, astro, sigFMCMC)
    EB = helpers.evidence(logLB, ndim, len(data))
    BICB = -2*EB

    out = """Binned data:
    chi2 = {0}
    chi2datum = {1}
    Likelihood = {2}
    Evidence = {3}
    BIC = {4}""".format(chisB, chisB/len(flux), logLB, EB, BICB)

    if 'gp' not in mode.lower() and nFrames!=1:
        #Unbinned data
        '''Get model'''
        
        astro_full   = astrofunc(time_full, **dict([[p0_astro[i], p0_mcmc[:len(p0_astro)][i]] for i in range(len(p0_astro))]))

        if 'bliss' in mode.lower():
            flux_full, time_full, xdata_full, ydata_full, psfxw_full, psfyw_full, mode, blissNBin, savepath = signal_inputs_full
            signal_inputs_full = bliss.precompute(flux_full, time_full, xdata_full, ydata_full,
                                                  psfxw_full, psfyw_full, mode,
                                                  astro_full, blissNBin, savepath, False)
        elif 'gp' in mode.lower():
            flux_full, xdata_full, ydata_full, time_full = signal_inputs_full
            signal_inputs_full = [flux_full, xdata_full, ydata_full, time_full, True, astro_full]
        
        signal_full = signalfunc(signal_inputs_full, **dict([[p0_labels[i], p0_mcmc[i]] for i in range(len(p0_mcmc))]))
        detec_full = signal_full/astro_full
        data_full = flux_full/detec_full
        
        '''Get Fitted Uncertainty'''
        ferr_full = sigFMCMC*np.sqrt(nFrames)

        N = len(data_full)
        if 'bliss' in mode.lower():
            nKnotsUsed_full = len(signal_inputs_full[-4][signal_inputs_full[-2]])
            ndim_eff_full = ndim+nKnotsUsed_full
        else:
            ndim_eff_full = ndim

        chis = helpers.chi2(data_full, astro_full, ferr_full)
        logL = helpers.loglikelihood(data_full, astro_full, ferr_full)
        E = helpers.evidence(logL, ndim_eff_full, N)
        BIC = -2*E

        out += """

Unbinned data:
    chi2 = {0}
    chi2datum = {1}
    Likelihood = {2}
    Evidence = {3}
    BIC = {4}""".format(chis, chis/len(flux_full), logL, E, BIC)

    with open(savepath+'EVIDENCE_'+mode+'.txt','w') as file:
        file.write(out)
    print(out)
    
    ###########################
    # Save computed parameters to be used later in table generation
    ###########################
    
    ResultMCMC_Params = Table()

    for i in range(len(p0_labels)):
        ResultMCMC_Params[p0_labels[i]] = MCMC_Results[i]

    ResultMCMC_Params['offset'] = offset
    ResultMCMC_Params['tDay'] = [np.nanmedian(tday), np.nanpercentile(tday, 84)-np.nanmedian(tday), np.nanmedian(tday)-np.nanpercentile(tday, 16)]
    ResultMCMC_Params['tNight'] = [np.nanmedian(tnight), np.nanpercentile(tnight, 84)-np.nanmedian(tnight), np.nanmedian(tnight)-np.nanpercentile(tnight, 16)]

    ResultMCMC_Params['chi2B'] = [chisB]
    ResultMCMC_Params['chi2datum'] = [chisB/len(flux)]
    ResultMCMC_Params['logLB'] = [logLB]
    ResultMCMC_Params['evidenceB'] = [EB]
    ResultMCMC_Params['sigF_photon_ppm'] = [sigF_photon_ppm]

    if 'gp' not in mode.lower():
        ResultMCMC_Params['chi2'] = [chis]
        ResultMCMC_Params['logL'] = [logL]
        ResultMCMC_Params['evidence'] = [E]

    pathres = savepath + 'ResultMCMC_'+mode+'_Params.npy'
    np.save(pathres, ResultMCMC_Params)
    
    return p0_mcmc, MCMC_Results, residuals

# FIX: Add a docstring for this function
def plot_walkers(savepath, mode, p0_astro, p0_fancyLabels, chain, plotCorner, showPlot=False):
    # FIX - show plots if requested
    
    ndim = chain.shape[-1]
    samples = chain.reshape((-1, ndim))
    
    ind_a = len(p0_astro) # index where the astro params end
    labels = p0_fancyLabels[:ind_a]

    fname = savepath+'MCMC_'+mode+'_astroWalkers.pdf'
    make_plots.walk_style(ind_a, chain.shape[0], chain, 10, chain.shape[1], labels, fname, showPlot)

    if 'bliss' not in mode.lower() or r'$\sigma_F$' in p0_fancyLabels:
        labels = p0_fancyLabels[ind_a:]
        fname = savepath+'MCMC_'+mode+'_detecWalkers.pdf'
        make_plots.walk_style(len(p0_fancyLabels)-ind_a, chain.shape[0], chain[:,:,ind_a:], 10, chain.shape[1], labels, fname, showPlot)
    
    if plotCorner:
        fig = corner.corner(samples[:,:ind_a], labels=p0_fancyLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True, 
                            plot_datapoints=True, title_kwargs={"fontsize": 12})
        plotname = savepath + 'MCMC_'+mode+'_corner.pdf'
        fig.savefig(plotname, bbox_inches='tight')
        if showPlot:
            plt.show()
        plt.close()
        
    return


# FIX: Add a docstring for this function
def burnIn(p0, mode, p0_labels, p0_fancyLabels, dparams, gparams, astrofunc, detecfunc, signalfunc, lnpriorfunc,
           time, flux, astro_guess, resid, detec_inputs, signal_inputs, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd, ncpu, savepath, showPlot=False):
    
    if 'gp' in mode.lower():
        return burnIn_GP(p0, mode, p0_labels, p0_fancyLabels, dparams, gparams,
                         astrofunc, detecfunc, signalfunc, lnpriorfunc,
                         time, flux, astro_guess, resid, detec_inputs, signal_inputs, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd, ncpu, savepath, showPlot=False)
    
    p0_astro  = freeze.get_fitted_params(astro_models.ideal_lightcurve, dparams)
    p0_detec = freeze.get_fitted_params(detecfunc, dparams)
    p0_psfwi  = freeze.get_fitted_params(detec_models.detec_model_PSFW, dparams)
    p0_hside  = freeze.get_fitted_params(detec_models.hside, dparams)
    p0_tslope  = freeze.get_fitted_params(detec_models.tslope, dparams)

    
    #################
    # Run optimization on just detector parameters first
    #################
    
    if 'bliss' not in mode:
        # Should work for PLD and Poly
        
        spyFunc0 = lambda p0_temp, inputs: np.mean((resid-detecfunc(inputs, **dict([[p0_detec[i], p0_temp[i]] for i in range(len(p0_detec))])))**2)
        spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_detec))], detec_inputs, 'Nelder-Mead')

        # replace p0 with new detector coefficient values
        p0[np.where(np.in1d(p0_labels,p0_detec))] = spyResult0.x
        resid /= detecfunc(detec_inputs, **dict([[p0_detec[i], p0[np.where(np.in1d(p0_labels,p0_detec))][i]] for i in range(len(p0_detec))]))

        # 2) get initial guess for psfw model
        if 'psfw' in mode.lower():
            spyFunc0 = lambda p0_temp: np.mean((resid-psfwifunc([psfxw, psfyw], **dict([[p0_pfswi[i], p0_temp[i]] for i in range(len(p0_pfswi))])))**2)
            spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_psfwi))], method='Nelder-Mead')

            # replace p0 with new detector coefficient values
            if spyResult0.success:
                p0[np.where(np.in1d(p0_labels,p0_psfwi))] = spyResult0.x
                resid /= psfwifunc([psfxw, psfyw], **dict([[p0_detec[i], p0[np.where(np.in1d(p0_labels,p0_pfswi))][i]] for i in range(len(p0_pfswi))]))

        # 3) get initial guess for hside model
        if 'hside' in mode.lower():
            spyFunc0 = lambda p0_temp: np.mean((resid-hsidefunc(time, **dict([[p0_hside[i], p0_temp[i]] for i in range(len(p0_hside))])))**2)
            spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_hside))], method='Nelder-Mead')

            # replace p0 with new detector coefficient values
            if spyResult0.success:
                p0[np.where(np.in1d(p0_labels,p0_hside))] = spyResult0.x
                resid /= hsidefunc(time, **dict([[p0_detec[i], p0[np.where(np.in1d(p0_labels,p0_hside))][i]] for i in range(len(p0_hside))]))

        if 'tslope' in mode.lower():
            spyFunc0 = lambda p0_temp: np.mean((resid-tslopefunc(time, **dict([[p0_tslope[i], p0_temp[i]] for i in range(len(p0_tslope))])))**2)
            spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_tslope))], method='Nelder-Mead')

            # replace p0 with new detector coefficient values
            if spyResult0.success:
                p0[np.where(np.in1d(p0_labels,p0_tslope))] = spyResult0.x
                resid /= tslopefunc(time, **dict([[p0_detec[i], p0[np.where(np.in1d(p0_labels,p0_tslope))][i]] for i in range(len(p0_tslope))]))


    # initial guess
    signal_guess = signalfunc(signal_inputs, **dict([[p0_labels[i], p0[i]] for i in range(len(p0))]))
    #includes psfw and/or hside functions if they're being fit
    detec_full_guess = signal_guess/astro_guess
    
    if showPlot:
        make_plots.plot_init_guess(time, flux, astro_guess, detec_full_guess, showPlot=showPlot)
    
    #################
    # Run an initial MCMC burn-in and pick the best location found along the way
    #################
    
    ndim = len(p0)
    nwalkers = ndim*3
    nBurnInSteps1 = 25500 # Chosen to give 500 steps per walker for Poly2v1 and 250 steps per walker for Poly5v2
    
    # get scattered starting point in parameter space 
    # MUST HAVE THE INITIAL SPREAD SUCH THAT EVERY SINGLE WALKER PASSES lnpriorfunc AND lnprior_custom
    p0_rel_errs = 1e-4*np.ones_like(p0)
    gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
    p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
    pos0 = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(nwalkers)])

    checkPhasePhis = np.linspace(-np.pi,np.pi,1000)
    
    priorlnls = np.array([(lnpriorfunc(mode=mode, checkPhasePhis=checkPhasePhis, **dict([[p0_labels[i], p_tmp[i]] for i in range(len(p_tmp))])) != 0.0 or np.isinf(helpers.lnprior_custom(p_tmp, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd))) for p_tmp in pos0])
    iters = 10
    while np.any(priorlnls) and iters>0:
    #         print('Warning: Some of the initial values fail the lnprior!')
    #         print('Trying to re-draw positions...')
        p0_rel_errs /= 1.5
        pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
        priorlnls = np.array([(lnpriorfunc(mode=mode, checkPhasePhis=checkPhasePhis, **dict([[p0_labels[i], p_tmp[i]] for i in range(len(p_tmp))])) != 0.0 or np.isinf(helpers.lnprior_custom(p_tmp, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd))) for p_tmp in pos0])
        iters -= 1
    if iters==0 and np.any(priorlnls):
        print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!')

        
        
    #First burn-in
    tic = t.time()
    print('Running first burn-in')
    with threadpool_limits(limits=1, user_api='blas'):
#     if True:
        with Pool(ncpu) as pool:
            #sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, args=[p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd], a = 2, pool=pool)
            pos1, prob, state = sampler.run_mcmc(pos0, np.rint(nBurnInSteps1/nwalkers), progress=True)
    print('Mean burn-in acceptance fraction: {0:.3f}'
                .format(np.median(sampler.acceptance_fraction)))
    
    
    fname = savepath+'MCMC_'+mode+'_burnin1Walkers.pdf'
    fig = make_plots.walk_style(len(p0), nwalkers, sampler.chain, 10, int(np.rint(nBurnInSteps1/nwalkers)), p0_fancyLabels,
                                fname, showPlot)
    
    
    p0 = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
    astro_guess = astrofunc(time, **dict([[p0_astro[i], p0[np.where(np.in1d(p0_labels,p0_astro))][i]] for i in range(len(p0_astro))]))
    signal_guess = signalfunc(signal_inputs, **dict([[p0_labels[i], p0[i]] for i in range(len(p0))]))
    #includes psfw and/or hside functions if they're being fit
    detec_full_guess = signal_guess/astro_guess
    fig = make_plots.plot_init_guess(time, flux, astro_guess, detec_full_guess, savepath, showPlot)
    
    return p0






# FIX: Add a docstring for this function
def burnIn_GP(p0, mode, p0_labels, p0_fancyLabels, dparams, gparams, astrofunc, detecfunc, signalfunc, lnpriorfunc, 
              time, flux, astro_guess, resid, detec_inputs, signal_inputs, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd, ncpu, savepath, showPlot=False):
    
    p0_astro  = freeze.get_fitted_params(astro_models.ideal_lightcurve, dparams)
    p0_detec = freeze.get_fitted_params(detecfunc, dparams)
    p0_psfwi  = freeze.get_fitted_params(detec_models.detec_model_PSFW, dparams)
    p0_hside  = freeze.get_fitted_params(detec_models.hside, dparams)
    p0_tslope  = freeze.get_fitted_params(detec_models.tslope, dparams)
    
    ######################
    # Iteratively run scipy optimize
    ######################
    checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

    initial_lnprob = helpers.lnprob(p0, p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd)

    spyFunc_full = lambda p0_temp, inputs: -helpers.lnprob(p0_temp, *inputs)

    nIterScipy = 10
    
    final_lnprob = -np.inf
    p0_optimized = []
    p0_temps = []
    print('Running iterative scipy.optimize')
    for i in tqdm(range(nIterScipy)):
        p0_rel_errs = 1e-1*np.ones_like(p0)
        gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
        p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
        p0_temp = p0*(1+p0_rel_errs*np.random.randn(len(p0)))+p0_rel_errs/10.*np.abs(np.random.randn(len(p0)))

        p0_temp[p0_labels=='A'] = np.random.uniform(0.,0.3)
        p0_temp[p0_labels=='B'] = np.random.uniform(-0.2,0.2)
        # Assignment to non-existent indices is safe (safelt ignores it), so this is fine for all modes
        p0_temp[p0_labels=='C'] = np.random.uniform(-0.3,0.3)
        p0_temp[p0_labels=='D'] = np.random.uniform(-0.3,0.3)
        p0_temp[p0_labels=='gpAmp'] = np.random.uniform(-4,-6)
        p0_temp[p0_labels=='gpLx'] = np.random.uniform(-0.5,-1)
        p0_temp[p0_labels=='gpLy'] = np.random.uniform(-0.5,-1)

        spyResult_full = scipy.optimize.minimize(spyFunc_full, p0_temp, [p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd], 'Nelder-Mead')
        lnprob_temp = helpers.lnprob(spyResult_full.x, p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd)

        p0_temps.append(np.copy(spyResult_full.x))

        if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
            final_lnprob = lnprob_temp
            p0_optimized = np.copy(spyResult_full.x)

            if final_lnprob > initial_lnprob:
                print('Improved ln-likelihood!')
                print("ln-likelihood: {0:.2f}".format(final_lnprob))
                p0 = np.copy(p0_optimized)

    astro_guess = astrofunc(time, **dict([[p0_astro[i], p0[np.where(np.in1d(p0_labels,p0_astro))][i]] for i in range(len(p0_astro))]))
    signal_guess = signalfunc(signal_inputs, **dict([[p0_labels[i], p0[i]] for i in range(len(p0))]))
    #includes psfw and/or hside functions if they're being fit
    detec_full_guess = signal_guess/astro_guess

    
    if showPlot:
        # plot detector initial guess
        make_plots.plot_init_guess(time, flux, astro_guess, detec_full_guess)
        plt.show()
        plt.close()
    
    
    ######################
    # Iteratively run some MCMCs to break free of local minima
    ######################
    print('Running first burn-ins')
    p0_temps_mcmc = []
    for p0_temp in p0_temps:
        ndim = len(p0)
        nwalkers = ndim*3
        nBurnInSteps1 = 25500 # Chosen to give 500 steps per walker for Poly2v1 and 250 steps per walker for Poly5v2

        # get scattered starting point in parameter space 
        # MUST HAVE THE INITIAL SPREAD SUCH THAT EVERY SINGLE WALKER PASSES lnpriorfunc AND lnprior_custom
        p0_rel_errs = 1e-3*np.ones_like(p0_temp)
        gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
        p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
        pos0 = np.array([p0_temp*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(nwalkers)])

        checkPhasePhis = np.linspace(-np.pi,np.pi,1000)
       
        priorlnls = np.array([(lnpriorfunc(mode=mode, checkPhasePhis=checkPhasePhis, **dict([[p0_labels[i], p_tmp[i]] for i in range(len(p_tmp))])) != 0.0 or np.isinf(helpers.lnprior_custom(p_tmp, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd))) for p_tmp in pos0])
        iters = 10
        while np.any(priorlnls) and iters>0:
    #         print('Warning: Some of the initial values fail the lnprior!')
    #         print('Trying to re-draw positions...')
            p0_rel_errs /= 1.5
            pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
            priorlnls = np.array([(lnpriorfunc(mode=mode, checkPhasePhis=checkPhasePhis, **dict([[p0_labels[i], p_tmp[i]] for i in range(len(p_tmp))])) != 0.0 or np.isinf(helpers.lnprior_custom(p_tmp, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd))) for p_tmp in pos0])
            iters -= 1
        if iters==0 and np.any(priorlnls):
            print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!')

        #Second burn-in
        #Do quick burn-in to get walkers spread out
        tic = t.time()
#         with threadpool_limits(limits=1, user_api='blas'):
        if True:
            with Pool(ncpu) as pool:
                #sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, args=[p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd], a = 2, pool=pool)
                pos1, prob, state = sampler.run_mcmc(pos0, np.rint(nBurnInSteps1/nwalkers), progress=False)
        print('Mean burn-in acceptance fraction: {0:.3f}'
                        .format(np.median(sampler.acceptance_fraction)))
        # sampler.reset()
        toc = t.time()
        print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))

        p0_temps_mcmc.append(np.copy(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))
        
        
    ######################
    # Iteratively run some MCMCs to break free of local minima
    ######################
    
    checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

    initial_lnprob = helpers.lnprob(p0, p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd)

    spyFunc_full = lambda p0_temp, inputs: -helpers.lnprob(p0_temp, *inputs)

    final_lnprob = -np.inf
    p0_optimized = []
    p0_temps_final = []
    print('Running second iterative scipy.optimize')
    for p0_temp in tqdm(p0_temps_mcmc):

        spyResult_full = scipy.optimize.minimize(spyFunc_full, p0_temp, [p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd], 'Nelder-Mead')
        lnprob_temp = helpers.lnprob(spyResult_full.x, p0_labels, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, gpriorInds, priors, errs, upriorInds, uparams_limits, gammaInd)

        p0_temps_final.append(np.copy(spyResult_full.x))

        if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
            final_lnprob = lnprob_temp
            p0_optimized = np.copy(spyResult_full.x)

            if final_lnprob > initial_lnprob:
                print('Improved ln-likelihood!')
                print("ln-likelihood: {0:.2f}".format(final_lnprob))
                p0 = np.copy(p0_optimized)

    astro_guess = astrofunc(time, **dict([[p0_astro[i], p0[np.where(np.in1d(p0_labels,p0_astro))][i]] for i in range(len(p0_astro))]))
    signal_guess = signalfunc(signal_inputs, **dict([[p0_labels[i], p0[i]] for i in range(len(p0))]))
    #includes psfw and/or hside functions if they're being fit
    detec_full_guess = signal_guess/astro_guess
    
    fig = make_plots.plot_init_guess(time, flux, astro_guess, detec_full_guess, savepath, showPlot)
    
    return p0



# FIX: Add a docstring for this function
def look_for_residual_correlations(time, flux, xdata, ydata, psfxw, psfyw, residuals,
                                   p0_mcmc, p0_labels, p0_obj, mode, savepath=None, showPlot=False):
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
    make_plots.triangle_colors(data1, data2, data3, data4, plotname, showPlot)
    
    return

