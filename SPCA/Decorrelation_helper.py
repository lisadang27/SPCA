import scipy
import scipy.stats as sp
import scipy.optimize as spopt

import emcee
import corner

from astropy import constants as const
from astropy import units

import numpy as np
import time as t
import os, sys
import csv

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
from SPCA import helpers, astro_models, make_plots, make_plots_custom, detec_models, bliss


# FIX: Add a docstring for this function
def downloadExoplanetArchive():
    #Download the most recent masterfile of the best data on each target
    try:
        _ = urllib.request.urlretrieve('http://www.astro.umontreal.ca/~adb/masterfile.ecsv', '../masterfile.ecsv')
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
                                -data['pl_ratdorerr2'][nameIndex]])
    else:
        p0_obj['a'] = data['pl_orbsmax'][nameIndex]*const.au.value/data['st_rad'][nameIndex]/const.R_sun.value
        p0_obj['a_err'] = np.sqrt(
            (np.mean([data['pl_orbsmaxerr1'][nameIndex], -data['pl_orbsmaxerr2'][nameIndex]])*const.au.value
             /data['st_rad'][nameIndex]/const.R_sun.value)**2
            + (data['pl_orbsmax'][nameIndex]*const.au.value
               /data['st_rad'][nameIndex]**2/const.R_sun.value
               *np.mean([data['st_raderr1'][nameIndex], -data['st_raderr2'][nameIndex]]))**2
        )
    p0_obj['per'] = data['pl_orbper'][nameIndex]
    p0_obj['per_err'] = np.mean([data['pl_orbpererr1'][nameIndex],
                              -data['pl_orbpererr2'][nameIndex]])
    p0_obj['t0'] = data['pl_tranmid'][nameIndex]-2.4e6-0.5
    p0_obj['t0_err'] = np.mean([data['pl_tranmiderr1'][nameIndex],
                             -data['pl_tranmiderr2'][nameIndex]])
    p0_obj['inc'] = data['pl_orbincl'][nameIndex]
    p0_obj['inc_err'] = np.mean([data['pl_orbinclerr1'][nameIndex],
                              -data['pl_orbinclerr2'][nameIndex]])
    p0_obj['Tstar'] = data['st_teff'][nameIndex]
    p0_obj['Tstar_err'] = np.mean([data['st_tefferr1'][nameIndex],
                                -data['st_tefferr2'][nameIndex]])
    
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
    p0_obj['rp_err'] = rp_err
    p0_obj['a'] = a
    p0_obj['a_err'] = a_err
    p0_obj['per'] = per
    p0_obj['per_err'] = per_err
    p0_obj['t0'] = t0-2.4e6-0.5
    p0_obj['t0_err'] = t0_err
    p0_obj['inc'] = inc
    p0_obj['inc_err'] = inc_err
    p0_obj['Tstar'] = Tstar
    p0_obj['Tstar_err'] = Tstar_err
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
    temps = np.linspace(5500, 7000, 500)
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
def get_photon_limit(rootpath, datapath, planet, channel, mode, aors, nFrames, ignoreFrames):
    
    aor = aors[-1]
    rawfiles = np.sort(os.listdir(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'))
    rawfiles  = [rawfile for rawfile in rawfiles if '_bcd.fits' in rawfile]
    with fits.open(rootpath+planet+'/data/'+channel+'/'+aor+'/'+channel+'/bcd/'+rawfiles[0]) as rawImage:
        rawHeader = rawImage[0].header
    
    if 'pld' in mode.lower():
        if '3x3' in mode.lower():
            npix = 3
        elif '5x5' in mode.lower():
            npix = 5
        flux = np.loadtxt(datapath, usecols=list(np.arange(npix**2).astype(int)), skiprows=1)     # mJr/str
        flux = np.sum(flux, axis=1)
    else:
        # Calculate the photon noise limit
        flux = np.loadtxt(datapath, usecols=[0], skiprows=1)     # mJr/str
    
    # FIX: Check that I'm calculating this properly!
    flux *= rawHeader['GAIN']*rawHeader['EXPTIME']/rawHeader['FLUXCONV']
    
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

def print_MCMC_results(chain, lnprobchain, p0_labels, mode, channel, p0_obj, usebestfit, savepath, compFactor=0):
    #print the results

    ndim = chain.shape[-1]
    samples = chain.reshape((-1, ndim))
    
    MCMC_Results = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
    p0_mcmc = np.median(samples, axis=0)

    # taking max lnprob params instead of median bc degeneracy
    if usebestfit: 
        if runMCMC:
            maxk, maxiter = np.unravel_index((lnprobchain).argmax(), (lnprobchain).shape)
            p0_mcmc = chain[maxk, maxiter,:]
        else:
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
        
    return p0_mcmc, MCMC_Results

def plot_walkers(savepath, mode, p0_astro, p0_fancyLabels, chain, plotCorner):
    
    ndim = chain.shape[-1]
    samples = chain.reshape((-1, ndim))
    
    ind_a = len(p0_astro) # index where the astro params end
    labels = p0_fancyLabels[:ind_a]

    fname = savepath+'MCMC_'+mode+'_astroWalkers.pdf'
    make_plots.walk_style(ind_a, chain.shape[0], chain, 10, chain.shape[1], labels, fname)

    if 'bliss' not in mode.lower() or r'$\sigma_F$' in p0_fancyLabels:
        labels = p0_fancyLabels[ind_a:]
        fname = savepath+'MCMC_'+mode+'_detecWalkers.pdf'
        make_plots.walk_style(len(p0_fancyLabels)-ind_a, chain.shape[0], chain[:,:,ind_a:], 10, chain.shape[1], labels, fname)
    
    if plotCorner:
        fig = corner.corner(samples[:,:ind_a], labels=p0_fancyLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True, 
                            plot_datapoints=True, title_kwargs={"fontsize": 12})
        plotname = savepath + 'MCMC_'+mode+'_corner.pdf'
        fig.savefig(plotname, bbox_inches='tight')
        
    return