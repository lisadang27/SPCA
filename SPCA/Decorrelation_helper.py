import numpy as np
import matplotlib.pyplot as plt
import os, errno
from tqdm import tqdm

import scipy.optimize
import emcee
from astropy import constants as const

from multiprocessing import Pool
from threadpoolctl import threadpool_limits

import warnings
warnings.filterwarnings("ignore")

from astropy.table import Table
from astropy.io import fits

import urllib.request

import pickle

# SPCA libraries
from SPCA import helpers, astro_models, make_plots, bliss

def downloadExoplanetArchive():
    """Downloads the most recent masterfile of the best data on each target
    Args:
        None
    Returns:
        None
    """

    try:
        _ = urllib.request.urlretrieve('http://www.astro.umontreal.ca/~adb/masterfile.ecsv', './masterfile.ecsv')
    except:
        print('Unable to download the most recent Exoplanet Archive data - analyses will resort to a previously downloaded version if available.', flush=True)
    return

# FIX: Add a docstring for this function
def loadArchivalData(rootpath, planet, channel):
    """Loads system parameters and respective uncertainties from the NASA Exoplanets Archives into p0_obj.
    Args:
        rootpath (str): Path to data
        planet   (str): name of the planet
        channel  (str): Spitzer channel used to obtain the observation (ch1 or ch2) 
    Returns:
        p0_obj   (obj): object with initial parameters 
    """

    if os.path.exists('./masterfile.ecsv'):
        data = Table.to_pandas(Table.read('./masterfile.ecsv'))
    else:
        # Fix: throw a proper error
        print('ERROR: No previously downloaded Exoplanet Archive data - try again when you are connected to the internet.')
        print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), './masterfile.ecsv'), flush=True)
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
        print('ERROR: Planet', planet, 'not found in the Exoplanet Archive - please load your own prior values.', flush=True)
        return
    
    nameIndex = indices[0]

    if np.isfinite(data['pl_ratror'][nameIndex]):
        p0_obj['rp'] = data['pl_ratror'][nameIndex]
        p0_obj['rp_err'] = np.mean([np.abs(data['pl_ratrorerr1'][nameIndex]),
                                    np.abs(data['pl_ratrorerr2'][nameIndex])])
    else:
        p0_obj['rp'] = data['pl_rads'][nameIndex]/data['st_rad'][nameIndex]
        p0_obj['rp_err'] = np.mean([np.abs(data['pl_radserr1'][nameIndex]),
                                    np.abs(data['pl_radserr2'][nameIndex])])/data['st_rad'][nameIndex]
        
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

    if 'XO-3b' in planet:
        p0_obj['a'] = 7.052
        p0_obj['a_err'] = np.mean([0.076, 0.097])

    p0_obj['per'] = data['pl_orbper'][nameIndex]
    p0_obj['per_err'] = np.mean([data['pl_orbpererr1'][nameIndex],
                                 np.abs(data['pl_orbpererr2'][nameIndex])])
    p0_obj['t0'] = data['pl_tranmid'][nameIndex]-2.4e6-0.5
    p0_obj['t0_err'] = np.mean([data['pl_tranmiderr1'][nameIndex],
                                np.abs(data['pl_tranmiderr2'][nameIndex])])
    p0_obj['inc'] = data['pl_orbincl'][nameIndex]
    p0_obj['inc_err'] = np.mean([data['pl_orbinclerr1'][nameIndex],
                                 np.abs(data['pl_orbinclerr2'][nameIndex])])
    if planet=='HD189733b':
        p0_obj['inc'] = 85.710
        p0_obj['inc_err'] = 0.024
    elif 'XO-3b' in planet:
        p0_obj['inc'] = 84.11
        p0_obj['inc_err'] = 0.16

    p0_obj['Tstar'] = data['st_teff'][nameIndex]
    p0_obj['Tstar_err'] = np.mean([data['st_tefferr1'][nameIndex],
                                   np.abs(data['st_tefferr2'][nameIndex])])
    if planet=='KELT-1b':
        p0_obj['Tstar'], p0_obj['Tstar_err'] = (6518, 50)
    
    p0_obj['logg'] = data['st_logg'][nameIndex]
    p0_obj['feh'] = data['st_metfe'][nameIndex]

    if planet=='MASCARA-1b':
        p0_obj['logg'] = 4.1050000
        p0_obj['feh'] = 0
    
    e = data['pl_orbeccen'][nameIndex]
    e_err = np.mean([np.abs(data['pl_orbeccenerr1'][nameIndex]), np.abs(data['pl_orbeccenerr2'][nameIndex])])
    argp = data['pl_orblper'][nameIndex]
    argp_err = np.mean([np.abs(data['pl_orblpererr1'][nameIndex]), np.abs(data['pl_orblpererr2'][nameIndex])])
    
    if e != 0 and np.isfinite(e):

        if not np.isfinite(argp):
            print('Randomly generating an argument of periastron...', flush=True)
            argp = np.random.uniform(0.,360.,1)

        p0_obj['ecosw'] = e*np.cos(argp*np.pi/180.)
        if 90 < argp < 270:
            p0_obj['ecosw']*=-1
        # Using this weird method to make sure we get the sign correct
        p0_obj['esinw'] = np.tan(argp*np.pi/180.)*p0_obj['ecosw']
    
    
        p0_obj['ecosw_err'] = np.sqrt( (e_err*np.cos(argp*np.pi/180.)                    )**2
                                      +(e    *np.sin(argp*np.pi/180.)*argp_err*np.pi/180.)**2)

        p0_obj['esinw_err'] = np.sqrt( (e_err*np.sin(argp*np.pi/180.)                    )**2
                                      +(e    *np.cos(argp*np.pi/180.)*argp_err*np.pi/180.)**2)
    
    # Get the stellar brightness temperature to allow us to invert Plank equation later
    p0_obj['tstar_b'], p0_obj['tstar_b_err'] = getTstarBright(rootpath, planet, channel, p0_obj)
    
    return p0_obj

def loadCustomData(rootpath, planet, channel, rp, a, per, t0, inc, e, argp, Tstar, logg, feh, rp_err=np.inf, a_err=np.inf, t0_err=np.inf, per_err=np.inf, inc_err=np.inf, e_err=np.inf, argp_err=np.inf, Tstar_err=np.inf):
    """Load custom parameters into p0_obj.
    Args:
        rootpath (str): Path to data
        planet   (str): name of the planet
        channel  (str): Spitzer channel used to obtain the observation (ch1 or ch2) 
        rp     (float): radius of the planets in unit of stellar radius
        a      (float): semi-major axis of the orbit of the planet in unit of stellar radius
        per    (float): period of the orbit in days
        t0     (float): time of transit in BMJD
        inc    (float): inclination of the orbit in degrees (inc = 90 for edge-on orbit)
        e      (float): eccentricity of the orbit
        argp   (float): argument of periastron in degrees
        Tstar  (float): Stellar temperature in Kelvins
        logg   (float): Surface gravity in log_10(cm/s^2)
        feh    (float): Stellar metallicity ratio [Fe/H]
        rp_err   (float,optional): uncertainty on radius of the planets in unit of stellar radius
        a_err    (float,optional): uncertainty on semi-major axis of the orbit of the planet in unit of stellar radius
        per_err  (float,optional): uncertainty on period of the orbit in days
        t0_err   (float,optional): uncertainty on time of transit in BMJD
        inc_err  (float,optional): uncertainty on inclination of the orbit in degrees (inc = 90 for edge-on orbit)
        e_err    (float,optional): uncertainty on eccentricity of the orbit
        argp_err (float,optional): uncertainty on argument of periastron in degrees
        Tstar_err(float,optional): uncertainty on Stellar temperature in Kelvins

    Returns:
        p0_obj (obj): object with initial parameters 
    """

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
            print('Randomly generating an argument of periastron...', flush=True)
            argp = np.random.uniform(0.,360.,1)

        p0_obj['ecosw'] = e/np.sqrt(1+np.tan(argp*np.pi/180.)**2)
        if 90 < argp < 270:
            p0_obj['ecosw']*=-1
        p0_obj['esinw'] = np.tan(argp*np.pi/180.)*p0_obj['ecosw']
        
    # Get the stellar brightness temperature to allow us to invert Plank equation later
    p0_obj['tstar_b'], p0_obj['tstar_b_err'] = getTstarBright(rootpath, planet, channel, p0_obj)
    
    return p0_obj

# FIX: Add a docstring for this function (docstring not done)
def getTstarBright(rootpath, planet, channel, p0_obj):
    """Calculate the stellar brightness temperature based on phoenix spectrum.
    Args:
        rootpath (str): Path to data
        planet   (str): name of the planet
        channel  (str): Spitzer channel used to obtain the observation (ch1 or ch2) 
        p0_obj (obj): object with initial parameters 
    Returns:
        tstar_b (): ???
        p0_obj['Tstar_err'] (float): uncertainty on stellar temperature
    """

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
        print('Downloading PHOENIX wavelength data.', flush=True)
        try:
            _ = urllib.request.urlretrieve(webfolder+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', phoenixWavFile)
        except:
            # Fix: throw a proper error
            print('ERROR: No previously downloaded PHOENIX data - try again when you are connected to the internet.',
                  flush=True)
            exit()
        print('Done download.', flush=True)

    webfolder += 'PHOENIX-ACES-AGSS-COND-2011/Z'+("{0:+.01f}".format(feh) if feh!=0 else '-0.0')+'/'

    webfile = ('lte'+teffStr
             +("{0:+.02f}".format(logg) if logg!=0 else '-0.00')
             +("{0:+.01f}".format(feh) if feh!=0 else '-0.0')
             +'.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits')

    phoenixSpectraFile = phoenixPath+webfile

    if not os.path.exists(phoenixSpectraFile):
        print('Downloading relevant PHOENIX spectra...', flush=True)
        try:
            _ = urllib.request.urlretrieve(webfolder+webfile, phoenixSpectraFile)
        except:
            # Fix: throw a proper error
            print('ERROR: No previously downloaded PHOENIX data - try again when you are connected to the internet.',
                  flush=True)
            exit()
        print('Done download.', flush=True)
    
    
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
    """Calculate the stellar brightness temperature based on phoenix spectrum.
    Args:
        rootpath (str): Path to data
        planet   (str): name of the planet
        channel  (str): Spitzer channel used to obtain the observation (ch1 or ch2) 
        mode     (str): decorrelation mode chosen
        pldIgnoreFrames (bool, optional): True if you want PLD to also ignore the first AOR. Default is True.
        pldAddStack     (bool, optional): True is you want to apply (and already have) custom Spitzer correction to your data when using PLD. Default is False.
    Returns:
        foldername (str): path to folder containing 
        filename
        filename_full
        savepath
        path_params
        AOR_snip
        aors
        breaks
        ignoreFrames
        p0_obj['Tstar_err'] (float): uncertainty on stellar temperature
    """
    AOR_snip = ''
    with open(rootpath+planet+'/analysis/aorSnippet.txt') as f:
        AOR_snip = f.readline().strip()

    mainpath   = rootpath+planet+'/analysis/'+channel+'/'
    ignoreFrames = np.array([])
    if 'psfx'==mode.lower():
        if channel=='ch2':
            foldername = '4um'
        else:
            foldername = '3um'
        foldername += 'PSF/'
        
    elif 'pld' in mode.lower():
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
                ignoreFrames = np.array(lines[i].strip().split('=')[1].strip().replace(' ','').split(','))
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
                    ignoreFrames = np.array(lines[i].split('=')[1].strip().replace(' ','').split(','))
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
    aors = np.array([aor for aor in aors if AOR_snip==aor[:len(AOR_snip)]])
    firstTime = []
    for i in range(len(aors)):
        file = np.sort(os.listdir(rootpath+planet+'/data/'+channel+'/'+aors[i]+'/'+channel+'/bcd/'))[0]
        with fits.open(rootpath+planet+'/data/'+channel+'/'+aors[i]+'/'+channel+'/bcd/'+file) as img:
            firstTime.append(img[0].header['BMJD_OBS'])
    order = np.argsort(firstTime)
    aors = aors[order]
    AOR_snip = AOR_snip[1:]
    
    # Figure out where there are AOR breaks
    breakpath = rootpath+planet+'/analysis/'+channel+'/aorBreaks.txt'
    with open(breakpath, 'r') as file:
        breaks = file.readline().strip()
        if len(breaks)==0:
            breaks = np.array([])
        else:
            breaks = np.array(breaks.split(' ')).astype(float)
    
    # path to photometry outputs
    filename   = channel + '_datacube_binned_AORs'+AOR_snip+'.dat'
    filename_full  = channel + '_datacube_full_AORs'+AOR_snip+'.dat'
    # Path to previous mcmc results (optional)
    path_params = foldername + mode + '/ResultMCMC_'+mode+'_Params.npy'
    
    return foldername, filename, filename_full, savepath, path_params, AOR_snip, aors, breaks, ignoreFrames

# FIX: Add a docstring for this function
def get_photon_limit(path, mode, nFrames, ignoreFrames):
    
    if 'pld' in mode.lower() and 'pldaper' not in mode.lower():
        if '3x3' in mode.lower():
            npix = 3
        elif '5x5' in mode.lower():
            npix = 5
        else:
            # FIX, throw an actual error
            print('Error: only 3x3 and 5x5 boxes for PLD are supported.', flush=True)
            return np.nan
        stamp = np.loadtxt(path, usecols=np.arange(int(npix**2)), skiprows=1).T
        flux = np.nansum(stamp, axis=0)
    else:
        flux = np.loadtxt(path, usecols=[0], skiprows=1)
    
    return 1/np.sqrt(np.nanmedian(flux))/np.sqrt(nFrames-len(ignoreFrames))*1e6

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
def setup_gpriors(gparams, p0_obj):
    priors = []
    errs = []
    for par in gparams:
        priors.append(p0_obj[par])
        errs.append(p0_obj[par+'_err'])
    
    return priors, errs

# FIX: Add a docstring for this function
def reload_old_fit(path_params, p0_obj, dparams, mode):
    Table_par = np.load(path_params)                  # table of best-fit params from prev. run
    nparams   = p0_obj['params'][np.logical_not(np.in1d(p0_obj['params'], dparams))]   # get the name list of params to be fitted
    for name in nparams:
        try:
            p0_obj[name]  = Table_par[name][0]
        except Exception as e:
            # FIX: throw a more meaningful error message
            print("type error: " + str(e), flush=True)    # catch errors if you use values from fun with less params

    if 'bliss' in mode.lower():
        try:
            p0_obj['nBinX'] = Table_par['nBinX'][0]
            p0_obj['nBinY'] = Table_par['nBinY'][0]
        except:
            print('There are no previously saved BLISS dimensions.',
                  '\nYou will need to rerun the initialization to find the BLISS bin dimensions.')
            
    return p0_obj

# FIX: Add a docstring for this function
def print_MCMC_results(flux, flux_full, chain, lnprobchain, mode, channel,
                       signal_func, signal_inputs, signal_inputs_full, p0_labels, p0_obj,
                       astro_func, astro_inputs, astro_inputs_full, astro_labels,
                       usebestfit, savepath, sigF_photon_ppm, nFrames, secondOrderOffset, compFactor=1):
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
    if np.any(p0_labels == 'A') and np.any(p0_labels == 'B'):
        # getting and printing the phase offset
        As = samples[:,np.where(p0_labels == 'A')[0][0]][:,np.newaxis]
        Bs = samples[:,np.where(p0_labels == 'B')[0][0]][:,np.newaxis]
        phis = np.linspace(-np.pi,np.pi,1000)
        offsets = []
        # Doing this in steps to not overflow RAM
        stepSizeOffsets = int(1e2)
        amplitudes = np.sqrt(As**2 + Bs**2)
        amplitude = MCMC_Results[1][0]*np.percentile(np.array(amplitudes), [16, 50, 84])[[1,2,0]]
        amplitude[1] -= amplitude[0]
        amplitude[2] = amplitude[0]-amplitude[2]
        out += '{:>8} = {:>16}  +{:>16}  -{:>16} \n'.format('Phase Semi-amplitude', amplitude[0], amplitude[1], amplitude[2])
            
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
            out += '{:>8} = {:>16}  +{:>16}  -{:>16} degrees east\n'.format('Offset', offset[0], offset[1], offset[2])
    else : 
        #Had to make an else loop if A and B are not fitted for. 
        #Set A&B values to the ones found in p0_obj and calculate the offset
        #by generating a phase curve and finding the phi corresponding
        #to the maximum value.
        Aval = p0_obj['A']
        Bval = p0_obj['B']
        phis = np.linspace(-np.pi,np.pi,1000)
        PhaseMod = 1 + Aval*(np.cos(phis)-1) + Bval*np.sin(phis)
        offset = -phis[np.argmax(PhaseMod)]*180/np.pi
        out += '{:>8} = {:>16} degrees east\n'.format('Offset', offset)


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
        
    #Added an if loop for when A&B are not fitted for and 
    #so tday and tnight won't be calculated.     
    if np.any(p0_labels == 'A') and np.any(p0_labels == 'B'):
        tstar_bs = np.random.normal(p0_obj['tstar_b'], p0_obj['tstar_b_err'], As.shape[0])

        tday = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_bs))-1)/(fp_MCMC/rp_MCMC**2)))**-1
        tnight = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_bs))-1)/(fp_MCMC*(1-2*As[:,0])/rp_MCMC**2)))**-1

        out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Day: ', np.median(tday), np.percentile(tday, 84)-np.median(tday), np.median(tday)-np.percentile(tday, 16))
        out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Night: ', np.nanmedian(tnight), np.nanpercentile(tnight, 84)-np.nanmedian(tnight), np.nanmedian(tnight)-np.nanpercentile(tnight, 16))
        out += 'For T_{*,b} = '+str(p0_obj['tstar_b'])+'\n'

    print(out, flush=True)
    with open(savepath+'MCMC_RESULTS_'+mode+'.txt','w') as file:
        file.write(out) 
    
    
    mcmc_signal = signal_func(p0_mcmc, *signal_inputs)
    mcmc_lightcurve = astro_func(astro_inputs, **dict([[label, p0_mcmc[i]] for i, label in enumerate(p0_labels)
                                                       if label in astro_labels]))
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
        bliss_ind = np.where([partial_func.func.__name__=='detec_model_bliss'
                              for partial_func in signal_inputs[-3]])[0][0]
        nBinX, nBinY = signal_inputs[-1][bliss_ind][3:5]
        # index -2 from bliss.precomute is a boolean array where there is one True for each BLISS knot
        nKnotsUsed = np.sum(signal_inputs[-1][bliss_ind][-2])
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
        
        astro_full   = astro_func(astro_inputs_full, **dict([[label, p0_mcmc[i]] for i, label in enumerate(p0_labels)
                                                            if label in astro_labels]))
        
        signal_full = signal_func(p0_mcmc, *signal_inputs_full)
        detec_full = signal_full/astro_full
        data_full = flux_full/detec_full
        
        '''Get Fitted Uncertainty'''
        ferr_full = sigFMCMC*np.sqrt(nFrames)

        N = len(data_full)
        if 'bliss' in mode.lower():
            bliss_ind = np.where([partial_func.func.__name__=='detec_model_bliss'
                                  for partial_func in signal_inputs_full[-3]])[0][0]
            nKnotsUsed_full = np.sum(signal_inputs_full[-1][bliss_ind][-1])
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
    print(out, flush=True)
    
    ###########################
    # Save computed parameters to be used later in table generation
    ###########################
    
    ResultMCMC_Params = Table()

    for i in range(len(p0_labels)):
        ResultMCMC_Params[p0_labels[i]] = MCMC_Results[i]
        
    #Added an loop for when A and B are not fitted for
    if np.any(p0_labels == 'A') and np.any(p0_labels == 'B'):
        ResultMCMC_Params['offset'] = offset
        ResultMCMC_Params['tDay'] = [np.nanmedian(tday), np.nanpercentile(tday, 84)-np.nanmedian(tday), np.nanmedian(tday)-np.nanpercentile(tday, 16)]
        ResultMCMC_Params['tNight'] = [np.nanmedian(tnight), np.nanpercentile(tnight, 84)-np.nanmedian(tnight), np.nanmedian(tnight)-np.nanpercentile(tnight, 16)]
    else : 
        #if A&B are not fitted for output the offset calculated in lines 597-607
        # we don't need to output tday and tnight since they were not calculated
        ResultMCMC_Params['offset'] = offset
        
    ResultMCMC_Params['chi2B'] = [chisB]
    ResultMCMC_Params['chi2datum'] = [chisB/len(flux)]
    ResultMCMC_Params['logLB'] = [logLB]
    ResultMCMC_Params['evidenceB'] = [EB]
    ResultMCMC_Params['sigF_photon_ppm'] = [sigF_photon_ppm]

    if 'bliss' in mode.lower():
        ResultMCMC_Params['nBinX'] = [nBinX]
        ResultMCMC_Params['nBinY'] = [nBinY]
    
    if 'gp' not in mode.lower() and nFrames!=1:
        ResultMCMC_Params['chi2'] = [chis]
        ResultMCMC_Params['logL'] = [logL]
        ResultMCMC_Params['evidence'] = [E]

    pathres = savepath + 'ResultMCMC_'+mode+'_Params.npy'
    np.save(pathres, ResultMCMC_Params)
    
    return p0_mcmc, MCMC_Results, residuals

# FIX: Add a docstring for this function
def burnIn(p0, p0_labels, mode, astro_func, astro_labels, astro_inputs, astro_inputs_full, signal_func, signal_inputs, signal_inputs_full,
           lnprob_inputs, gparams, gpriorInds, priors, errs, time, flux, breaks, bestfitNbin,
           ncpu, savepath=None, showPlot=False, nIterScipy=10):
    
    if 'gp' not in mode.lower() and 'bliss' not in mode.lower():
        print('Optimizing detector parameters first...', flush=True)
        def minfunc(p0_detec, p0_astro, lnprob_inputs):
            # Add back in the astrophysical parameters
            p0_full = np.append(p0_astro, p0_detec)
            return -helpers.lnprob(p0_full, *lnprob_inputs)

        # Pull aside the astrophysical parameters
        p0_astro = p0[np.where(np.in1d(p0_labels,astro_labels))]
        p0_detec = p0[np.where(np.logical_not(np.in1d(p0_labels,astro_labels)))]

        spyResult = scipy.optimize.minimize(minfunc, p0_detec, (p0_astro, lnprob_inputs), 'Nelder-Mead')

        p0 = np.append(p0_astro, spyResult.x)
        
    #########################
    # Optimize BLISS Bin Size
    
    if 'bliss' in mode.lower():
        print('Running iterative scipy.optimize on all parameters', flush=True)
        def minfunc(p0, lnprob_inputs):
            return -helpers.lnprob(p0, *lnprob_inputs)

        initial_lnprob = helpers.lnprob(p0, *lnprob_inputs)
        final_lnprob = -np.inf
        p0_optimized = []
        p0_temps = []
        lnprob_temps = []
        for i in tqdm(range(nIterScipy)):
            p0_rel_errs = 1e-1*np.ones_like(p0)
            gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
            p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
            p0_temp = p0*(1+p0_rel_errs*np.random.randn(len(p0)))+p0_rel_errs/10.*np.abs(np.random.randn(len(p0)))

            amp = np.random.uniform(0.2,0.5)
            offset = np.random.uniform(-30,10)*np.pi/180

            p0_temp[p0_labels=='A'] = amp*np.cos(offset)
            p0_temp[p0_labels=='B'] = amp*np.sin(offset)
            # Assignment to non-existent indices is safely ignored, so this is fine for all modes
            p0_temp[p0_labels=='C'] = np.random.uniform(-0.1,0.1)
            p0_temp[p0_labels=='D'] = np.random.uniform(-0.1,0.1)

            spyResult = scipy.optimize.minimize(minfunc, p0_temp, lnprob_inputs, 'Nelder-Mead')
            lnprob_temp = helpers.lnprob(spyResult.x, *lnprob_inputs)

            p0_temps.append(np.copy(spyResult.x))
            lnprob_temps.append(lnprob_temp)

        for p0_temp, lnprob_temp in zip(p0_temps, lnprob_temps):
            if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
                final_lnprob = lnprob_temp
                p0_optimized = np.copy(p0_temp)

        if final_lnprob > initial_lnprob:
            print('Improved ln-likelihood!')
            print("ln-likelihood: {0:.2f}".format(final_lnprob), flush=True)
            p0 = np.copy(p0_optimized)
        
        print('Optimizing BLISS Bin Size', flush=True)
        blissInd = np.where([partial_func.func.__name__=='detec_model_bliss'
                             for partial_func in signal_inputs[-3]])[0][0]

        def minfunc(p0, lnprob_inputs):
            return -helpers.lnprob(p0, *lnprob_inputs)
        
        bliss_SDNRs = []
        bliss_fps = []
        nni_SDNRs = []
        nni_fps = []
        p0_temps = []
        flux = signal_inputs[-1][blissInd][0]
        xdata = signal_inputs[-1][blissInd][1]
        ydata = signal_inputs[-1][blissInd][2]
        
        xSpread = np.max(xdata)-np.min(xdata)
        ySpread = np.max(ydata)-np.min(ydata)
        spreads = np.array([xSpread, ySpread])
        spread = np.max(spreads)
        knotSizes = np.linspace(0.010, 0.060, int(np.round(spread/0.010-spread/0.060)/2))
        
        # Go in reverse order so initial tqdm ETA is a worst case estimate
        # Remove any options where there would only be a single bin
        nbins = np.unique([np.ceil(spreads/knotSize).astype(int) for knotSize in knotSizes if np.all(np.ceil(spreads/knotSize).astype(int) > 1)], axis=0)[::-1]
        
        for blissNBinX, blissNBinY in tqdm(nbins):
            for useNNI in [False, True]:
                if useNNI:
                    temp_inputs = list(bliss.precompute(flux, xdata, ydata, blissNBinX, blissNBinY))
                    temp_inputs[-2] = np.zeros_like(temp_inputs[0]).astype(bool)
                    temp_inputs[-1] = np.ones_like(temp_inputs[0]).astype(bool)
                    signal_inputs[-1][blissInd] = tuple(temp_inputs)
                    lnprob_inputs[4] = signal_inputs
                else:
                    temp_inputs = list(bliss.precompute(flux, xdata, ydata, blissNBinX, blissNBinY))
                    signal_inputs[-1][blissInd] = tuple(temp_inputs)
                    lnprob_inputs[4] = signal_inputs

                # Optimize parameters again
                initial_lnprob = helpers.lnprob(p0, *lnprob_inputs)
                final_lnprob = -np.inf
                p0_optimized = p0
                for i in range(int(nIterScipy)):
                    p0_rel_errs = 1e-1*np.ones_like(p0)
                    gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
                    p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
                    p0_temp = p0*(1+p0_rel_errs*np.random.randn(len(p0)))+p0_rel_errs/10.*np.abs(np.random.randn(len(p0)))

                    amp = np.random.uniform(0.2,0.5)
                    offset = np.random.uniform(-30,10)*np.pi/180

                    p0_temp[p0_labels=='A'] = amp*np.cos(offset)
                    p0_temp[p0_labels=='B'] = amp*np.sin(offset)
                    # Assignment to non-existent indices is safely ignored, so this is fine for all modes
                    p0_temp[p0_labels=='C'] = np.random.uniform(-0.1,0.1)
                    p0_temp[p0_labels=='D'] = np.random.uniform(-0.1,0.1)

                    p0_temp = scipy.optimize.minimize(minfunc, p0_temp, lnprob_inputs, 'Nelder-Mead').x
                    lnprob_temp = helpers.lnprob(p0_temp, *lnprob_inputs)

                    if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
                        final_lnprob = lnprob_temp
                        p0_optimized = np.copy(p0_temp)

                p0_temps.append(p0_optimized)
                SDNR = p0_optimized[-1]
                fp = p0_optimized[p0_labels=='fp']

                if useNNI:
                    nni_SDNRs.append(SDNR)
                    nni_fps.append(fp)
                else:
                    bliss_SDNRs.append(SDNR)
                    bliss_fps.append(fp)

        nbins = nbins[::-1]
        bliss_SDNRs = np.array(bliss_SDNRs)[::-1]
        bliss_fps = np.array(bliss_fps)[::-1]
        nni_SDNRs = np.array(nni_SDNRs)[::-1]
        nni_fps = np.array(nni_fps)[::-1]

        blissNBin = 0
        for i in range(len(nbins)):
            if bliss_SDNRs[i]<nni_SDNRs[i]:
                continue
            else:
                if i==0:
                    continue
                else:
                    blissNBin = nbins[i-1]
                    blissNBin = [84,64]
                    p0 = p0_temps[i-1]
                print('Using blissNBin =', blissNBin)
                break
        if np.all(blissNBin == 0):
            for i, nbin in enumerate(nbins[::-1]):
                temp_inputs = list(bliss.precompute(flux, xdata, ydata, *nbin))
                if np.sum(temp_inputs[-1]) > 50:
                    continue
                else:
                    blissNBin = nbin
                    p0 = p0_temps[i]
                    print('Defaulting to blissNBin =', blissNBin)
                    break
        if np.all(blissNBin == 0):
            blissNBin = nbins[0]
            blissNBin = [84,64]
            print('Hard defaulting to blissNBin =', blissNBin)
        
        # Make a diagnostic plot showing SDNR vs bin size for NNI and BLISS
        plt.plot(np.product(nbins, axis=1), nni_SDNRs*1e6, label=r'$\rm NNI$')
        plt.plot(np.product(nbins, axis=1), bliss_SDNRs*1e6, label=r'$\rm BLISS$')
        plt.xticks(np.product(nbins, axis=1))
        plt.axvline(np.product(blissNBin), c='k')
        plt.ylabel(r'$\rm SDNR~of~Residuals~(ppm)$')
        plt.xlabel(r'$\rm Total~number~of~knots$')
        plt.legend(loc=1)
        plt.xscale('log')
        if savepath!=None:
            plt.savefig(savepath+'BLISS_SDNR_vs_BinSize.pdf', bbox_inches='tight')
            # saving data used in the plot as pkl file
            header = 'HEADER: Number of bins per axis, blissNBin, BLISS SDNR of Residuals (ppm), NNI SDNR of Residuals (ppm)'
            data = [header, nbins, blissNBin, bliss_SDNRs*1e6, nni_SDNRs*1e6]
            pathdata = savepath + 'BLISS_SDNR_vs_BinSize.pkl'
            with open(pathdata, 'wb') as outfile:
                pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
            
        if showPlot:
            plt.show()
        plt.close()
        
        # Make a diagnostic plot showing eclipse depth vs bin size for NNI and BLISS
        plt.plot(np.product(nbins, axis=1), nni_fps*1e6, label=r'$\rm NNI$')
        plt.plot(np.product(nbins, axis=1), bliss_fps*1e6, label=r'$\rm BLISS$')
        plt.xticks(np.product(nbins, axis=1))
        plt.axvline(np.product(blissNBin), c='k')
        plt.ylabel(r'$\rm Eclipse~Depth~(ppm)$')
        plt.xlabel(r'$\rm Total~number~of~knots$')
        plt.legend(loc='best')
        plt.xscale('log')
        if savepath!=None:
            plt.savefig(savepath+'BLISS_EclipseDepth_vs_BinSize.pdf', bbox_inches='tight')
            # saving data used in the plot as pkl file
            header = 'HEADER: Number of bins per axis, blissNBin, BLISS Eclipse Depth (ppm), NNI Eclipse Depth (ppm)'
            data = [header, nbins, blissNBin, bliss_fps*1e6, nni_fps*1e6]
            pathdata = savepath + 'BLISS_EclipseDepth_vs_BinSize.pkl'
            with open(pathdata, 'wb') as outfile:
                pickle.dump(data, outfile, pickle.HIGHEST_PROTOCOL)
        if showPlot:
            plt.show()
        plt.close()
        
        # Update the bliss inputs to the best blissNBin
        astro_guess = astro_func(astro_inputs, **dict([[label, p0[i]] for i, label in enumerate(p0_labels)
                                                       if label in astro_labels]))
        temp_inputs = list(bliss.precompute(flux, xdata, ydata, *blissNBin, astro_guess, savepath, plot=True,
                                            showPlot=showPlot))
        signal_inputs[-1][blissInd] = tuple(temp_inputs)
        lnprob_inputs[4] = signal_inputs
        
        
        astro_guess_full = astro_func(astro_inputs_full, **dict([[label, p0[i]] for i, label in enumerate(p0_labels)
                                                                 if label in astro_labels]))
        flux_full = signal_inputs_full[-1][blissInd][0]
        xdata_full = signal_inputs_full[-1][blissInd][1]
        ydata_full = signal_inputs_full[-1][blissInd][2]
        
        temp_inputs_full = list(bliss.precompute(flux_full, xdata_full, ydata_full, *blissNBin, astro_guess_full, savepath, 
                                                 plot=False, showPlot=False))
        signal_inputs_full[-1][blissInd] = tuple(temp_inputs_full)
        
    #########################
    
    print('Running iterative scipy.optimize on all parameters', flush=True)
    def minfunc(p0, lnprob_inputs):
        return -helpers.lnprob(p0, *lnprob_inputs)

    initial_lnprob = helpers.lnprob(p0, *lnprob_inputs)
    final_lnprob = -np.inf
    p0_optimized = []
    p0_temps = []
    lnprob_temps = []
    for i in tqdm(range(nIterScipy)):
        p0_rel_errs = 1e-1*np.ones_like(p0)
        gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
        p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
        p0_temp = p0*(1+p0_rel_errs*np.random.randn(len(p0)))+p0_rel_errs/10.*np.abs(np.random.randn(len(p0)))

        amp = np.random.uniform(0.2,0.5)
        offset = np.random.uniform(-30,10)*np.pi/180

        p0_temp[p0_labels=='A'] = amp*np.cos(offset)
        p0_temp[p0_labels=='B'] = amp*np.sin(offset)
        # Assignment to non-existent indices is safely ignored, so this is fine for all modes
        p0_temp[p0_labels=='C'] = np.random.uniform(-0.1,0.1)
        p0_temp[p0_labels=='D'] = np.random.uniform(-0.1,0.1)
        p0_temp[p0_labels=='gpAmp'] = np.random.uniform(0.05,0.135)
        p0_temp[p0_labels=='gpLx'] = np.random.uniform(-0.5,-1)
        p0_temp[p0_labels=='gpLy'] = np.random.uniform(-0.5,-1)

        spyResult = scipy.optimize.minimize(minfunc, p0_temp, lnprob_inputs, 'Nelder-Mead')
        lnprob_temp = helpers.lnprob(spyResult.x, *lnprob_inputs)

        p0_temps.append(np.copy(spyResult.x))
        lnprob_temps.append(lnprob_temp)

    for p0_temp, lnprob_temp in zip(p0_temps, lnprob_temps):
        if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
            final_lnprob = lnprob_temp
            p0_optimized = np.copy(p0_temp)

    if final_lnprob > initial_lnprob:
        print('Improved ln-likelihood!')
        print("ln-likelihood: {0:.2f}".format(final_lnprob), flush=True)
        p0 = np.copy(p0_optimized)
        
    #########################
        
    print('Running first mini burn-ins', flush=True)
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
        priorlnls = np.array([np.isinf(helpers.lnprob(p_tmp, *lnprob_inputs)) for p_tmp in pos0])
        iters = 10
        while np.any(priorlnls) and iters>0:
            p0_rel_errs /= 1.5
            pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
            priorlnls[priorlnls] = np.array([np.isinf(helpers.lnprob(p_tmp, *lnprob_inputs)) for p_tmp in pos0[priorlnls]])
            iters -= 1
        if iters==0 and np.any(priorlnls):
            print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!',
                  flush=True)
        #Do quick burn-in to get walkers spread out
        with threadpool_limits(limits=1, user_api='blas'):
            with Pool(ncpu) as pool:
                #check input of both these functions
                sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, pool=pool, args=lnprob_inputs)
                pos1, prob, state = sampler.run_mcmc(pos0, int(np.rint(nBurnInSteps1/nwalkers)), progress=True)
        print('Mean burn-in acceptance fraction: {0:.3f}'.format(np.median(sampler.acceptance_fraction)), flush=True)

        p0_temps_mcmc.append(np.copy(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))
    
    #########################
    
    print('Running final iterative scipy.optimize on all parameters', flush=True)
    initial_lnprob = helpers.lnprob(p0, *lnprob_inputs)
    final_lnprob = -np.inf
    p0_optimized = []
    p0_temps_final = []
    lnprob_temps_final = []
    for p0_temp in tqdm(p0_temps_mcmc):
        spyResult = scipy.optimize.minimize(minfunc, p0_temp, lnprob_inputs, 'Nelder-Mead')
        lnprob_temp = helpers.lnprob(spyResult.x, *lnprob_inputs)

        p0_temps_final.append(np.copy(spyResult.x))
        lnprob_temps_final.append(lnprob_temp)

    for p0_temp, lnprob_temp in zip(p0_temps_final, lnprob_temps_final):
        if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
            final_lnprob = lnprob_temp
            p0_optimized = np.copy(p0_temp)

    if final_lnprob > initial_lnprob:
        print('Improved ln-likelihood!')
        print("ln-likelihood: {0:.2f}".format(final_lnprob), flush=True)
        p0 = np.copy(p0_optimized)
    
    astroModel = astro_func(astro_inputs, **dict([[label, p0[i]] for i, label in enumerate(p0_labels) if label in astro_labels]))
    signalModel = signal_func(p0, *signal_inputs)
    detecModel = signalModel/astroModel

    make_plots.plot_model(time, flux, astroModel, detecModel, breaks, savepath, 'Initial_Guess.pdf',
                          nbin=bestfitNbin, showPlot=showPlot, fontsize=24)
    
    return p0
