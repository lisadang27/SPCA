#!/usr/bin/env python
# coding: utf-8

'''Import Packages'''

import scipy
import scipy.stats as sp
import scipy.optimize as spopt

import emcee
import batman
import corner

from astropy import constants as const
from astropy import units

import numpy as np
import time as t
import timeit
import os, sys
import csv

import inspect

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

import astropy.time
from astropy.stats import sigma_clip
from astropy.table import Table, Column
from astropy.io import fits

import urllib.request

# SPCA libraries
from SPCA import helpers, astro_models, make_plots, make_plots_custom, detec_models, bliss




planets = ['MASCARA-1b', 'KELT-16b', 'WASP-121b', 'WASP-121b', 'CoRoT-2b', 'HAT-P-7b', 'HAT-P-7b', 'HD149026b', 'HD149026b', 'KELT-9b', 'WASP-14b', 'WASP-14b', 'WASP-18b', 'WASP-19b', 'WASP-33b', 'WASP-43b', 'WASP-43b_repeatCh1', 'WASP-43b_repeatCh2', 'WASP-43b_repeat2Ch2', 'Qatar1b', 'Qatar1b'][-1:]
channels = ['ch2', 'ch2', 'ch1', 'ch2', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch1', 'ch2', 'ch2', 'ch1', 'ch2'][-1:]

rootpath = '/homes/picaro/bellt/research/'
# rootpath = '/home/taylor/Documents/Research/spitzer/'

mode_appendix = '_autoRun'

# parameters you do not wish to fit
dparams_input = []#['ecosw','esinw']

# parameters you want to place a gaussian prior on
gparams = ['t0', 'per', 'a', 'inc']

# parameters you want to place a uniform prior on
uparams = ['gpLx', 'gpLy']
uparams_limits = [[0,-3],[0,-3]]



minPolys = 2*np.ones(len(planets)).astype(int)       # minimum polynomial order to consider
maxPolys = 5*np.ones(len(planets)).astype(int)       # maximum polynomial order to consider (set < minPoly to not use polynomial models)
tryBliss = True                          # whether to try BLISS detector model
tryGP = False                            # whether to try GP detector model
tryEllipse = False                       # Whether to try an ellipsoidal variation astrophysical model
tryPSFW = False

runMCMC = False                          # whether to run MCMC or just load-in past results
nBurnInSteps1 = 1e5                      # number of steps to use for the first mcmc burn-in (only used if not doing GP)
nBurnInSteps2 = 1e6                      # number of steps to use for the second mcmc burn-in
nProductionSteps = 2e5                   # number of steps to use with mcmc production run
usebestfit = False                       # used best-fit instead of most probable parameters 
blissNBin = 8                            # number of knots to allow in each direction
secondOrderOffset = False                # should you use the second order sinusoid terms when calculating offset
bestfitNbin = 50                         # the number of binned values to overplot on the bestfit 4-panel figure (use None if you don't want these overplotted)
nFrames  = 64                            # number of frames per binned data point
initializeWithOld = False                # initial with previous mcmc results using the same method


#non-unity multiplicative factors if you have dilution from a nearby companion
compFactors = np.ones(len(planets))

# non-zero if you want to remove some initial data points
cuts = np.zeros(len(planets)).astype(int)




######### FIX: REMOVE THIS LATER!!!! ###############
# compFactors[0] += 0.8858*0.1196
###############################################
######### FIX: REMOVE THIS LATER!!!! ###############
# if 'WASP-12' in planet:
#     if 'old' in planet.lower() and channel=='ch1':
#         compFactor += 0.9332*0.1149
#     elif 'old' in planet.lower() and channel=='ch2':
#         compFactor += 0.8382*0.1196
#     elif channel=='ch1':
#         compFactor += 0.8773*0.1149
#     elif channel=='ch2':
#         compFactor += 0.8858*0.1196







#####################################################################################################
# Everything below is automated


if rootpath[-1]!='/':
    rootpath += '/'


for iterationNumber in range(len(planets)):
    
    planet = planets[iterationNumber]
    channel = channels[iterationNumber]
    compFactor = compFactors[iterationNumber]
    cut_tmp = cuts[iterationNumber]
    minPoly = minPolys[iterationNumber]
    maxPoly = maxPolys[iterationNumber]

    with open(rootpath+planet+'/analysis/'+channel+'/cutFirstAOR.txt', 'r') as file:
        cutFirstAOR = file.readline().strip()=='True'

    #Download the most recent masterfile of the best data on each target
    try:
        _ = urllib.request.urlretrieve('http://www.astro.umontreal.ca/~adb/masterfile.ecsv', '../masterfile.ecsv')
    except:
        print('Unable to download the most recent Exoplanet Archive data - resorting to previously downloaded version.')

    if os.path.exists('../masterfile.ecsv'):
        data = Table.to_pandas(Table.read('../masterfile.ecsv'))
    else:
        print('ERROR: No previously downloaded Exoplanet Archive data - try again when you are connected to the internet.')
        print(FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), '../masterfile.ecsv'))
        exit()

    names = np.array(data['pl_hostname'])+np.array(data['pl_letter'])
    names = np.array([name.replace(' ','').replace('-', '').replace('_','') for name in names])


    # make params obj
    p0_obj  = helpers.signal_params() 

    # Personalize object default object values
    p0_obj.name = planet

    nameIndex = np.where(names==planet.replace(' ','').replace('-', '').split('_')[0])[0][0]

    if np.isfinite(data['pl_ratror'][nameIndex]):
        p0_obj.rp = data['pl_ratror'][nameIndex]
    else:
        p0_obj.rp = data['pl_rads'][nameIndex]/data['st_rad'][nameIndex]

    if np.isfinite(data['pl_ratdor'][nameIndex]):
        p0_obj.a = data['pl_ratdor'][nameIndex]
        p0_obj.a_err = np.mean([data['pl_ratdorerr1'][nameIndex],
                                -data['pl_ratdorerr2'][nameIndex]])
    else:
        p0_obj.a = data['pl_orbsmax'][nameIndex]*const.au.value/data['st_rad'][nameIndex]/const.R_sun.value
        p0_obj.a_err = np.sqrt(
            (np.mean([data['pl_orbsmaxerr1'][nameIndex], -data['pl_orbsmaxerr2'][nameIndex]])*const.au.value
             /data['st_rad'][nameIndex]/const.R_sun.value)**2
            + (data['pl_orbsmax'][nameIndex]*const.au.value
               /data['st_rad'][nameIndex]**2/const.R_sun.value
               *np.mean([data['st_raderr1'][nameIndex], -data['st_raderr2'][nameIndex]]))**2
        )
    p0_obj.per = data['pl_orbper'][nameIndex]
    p0_obj.per_err = np.mean([data['pl_orbpererr1'][nameIndex],
                              -data['pl_orbpererr2'][nameIndex]])
    p0_obj.t0 = data['pl_tranmid'][nameIndex]-2.4e6-0.5
    p0_obj.t0_err = np.mean([data['pl_tranmiderr1'][nameIndex],
                             -data['pl_tranmiderr2'][nameIndex]])
    p0_obj.inc = data['pl_orbincl'][nameIndex]
    p0_obj.inc_err = np.mean([data['pl_orbinclerr1'][nameIndex],
                              -data['pl_orbinclerr2'][nameIndex]])
    p0_obj.Tstar = data['st_teff'][nameIndex]
    p0_obj.Tstar_err = np.mean([data['st_tefferr1'][nameIndex],
                                -data['st_tefferr2'][nameIndex]])

    e = data['pl_orbeccen'][nameIndex]
    argp = data['pl_orblper'][nameIndex]

    if e != 0:

        if not np.isfinite(argp):
            print('Randomly generating an argument of periastron...')
            argp = np.random.uniform(0.,360.,1)

        p0_obj.ecosw = e/np.sqrt(1+np.tan(argp*np.pi/180.)**2)
        if 90 < argp < 270:
            p0_obj.ecosw*=-1
        p0_obj.esinw = np.tan(argp*np.pi/180.)*p0_obj.ecosw




    # Get the phoenix file ready to compute the stellar brightness temperature
    teffStr = p0_obj.Tstar
    if teffStr <= 7000:
        teffStr = teffStr - (teffStr%100) + np.rint((teffStr%100)/100)*100
    elif teffStr > 7000:
        teffStr = teffStr - (teffStr%200) + np.rint((teffStr%200)/200)*200
    elif teffStr > 12000:
        teffStr = 12000
    teffStr = str(int(teffStr)).zfill(5)

    logg = data['st_logg'][nameIndex]
    if np.isnan(logg):
        logg = 4.5
    logg = logg - (logg%0.5) + np.rint((logg%0.5)*2)/2.
    logg = -logg
    feh = data['st_metfe'][nameIndex]
    if np.isnan(feh):
        feh = 0.
    feh = (feh - (feh%0.5) + np.rint((feh%0.5)*2)/2.)
    if feh<-2.:
        feh = (feh - (feh%1) + np.rint((feh%1)))

    webfolder = 'ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/'
    phoenixPath = rootpath+planet+'/phoenix/'
    phoenixWavFile = phoenixPath+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits'
    if not os.path.exists(phoenixPath):
        print('Downloading relevant PHOENIX wavelengths file...')
        os.mkdir(phoenixPath)
        try:
            _ = urllib.request.urlretrieve(webfolder+'WAVE_PHOENIX-ACES-AGSS-COND-2011.fits', phoenixWavFile)
        except:
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
            print('ERROR: No previously downloaded PHOENIX data - try again when you are connected to the internet.')
            exit()
        print('Done download.')


    # set up Gaussian priors
    priors = []
    errs = []
    if 't0' in gparams:
        priors.append(p0_obj.t0)
        errs.append(p0_obj.t0_err)
    if 'per' in gparams:
        priors.append(p0_obj.per)
        errs.append(p0_obj.per_err)
    if 'a' in gparams:
        priors.append(p0_obj.a)
        errs.append(p0_obj.a_err)
    if 'inc' in gparams:
        priors.append(p0_obj.inc)
        errs.append(p0_obj.inc_err)







    # make all of the mode options
    modes = []                                # Detector model and Phase variation order
    for polyOrder in range(minPoly,maxPoly+1):
        modes.append('Poly'+str(polyOrder)+'_v1')
        modes.append('Poly'+str(polyOrder)+'_v2')
        if tryEllipse:
            modes.append('Poly'+str(polyOrder)+'_v1_ellipse')
            modes.append('Poly'+str(polyOrder)+'_v1_ellipseOffset')

    if tryBliss:
        modes.append('BLISS_v1')
        modes.append('BLISS_v2')

    if tryGP:
        modes.append('GP_v1')
        modes.append('GP_v2')

    #FIX: Make it so that it does an extend, not a replace
    if tryPSFW:
        modes.extend([mode+'_PSFW' for mode in modes])

    modes = [mode+mode_appendix for mode in modes]

    for mode in modes:

        print('Beginning', planet, channel, mode)
        
        p0_obj.mode = mode


        AOR_snip = ''
        with open(rootpath+planet+'/analysis/aorSnippet.txt') as f:
            AOR_snip = f.readline().strip()

        mainpath   = rootpath+planet+'/analysis/'+channel+'/'
        phoption = ''
        ignoreFrames = np.array([])
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




        # labels for all the possible fit parameters
        p0_names = np.array(['t0', 'per', 'rp', 'a', 'inc', 'ecosw', 'esinw', 'q1', 'q2', 'fp', 
                             'A', 'B', 'C', 'D', 'r2', 'r2off', 'c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7',
                             'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17',
                             'c18', 'c19', 'c20', 'c21', 'd1', 'd2', 'd3', 's1', 's2', 'm1',
                             'gpAmp', 'gpLx', 'gpLy', 'sigF'])

        # fancy labels for plot purposed  for all possible fit parameters
        p0_fancyNames = np.array([r'$t_0$', r'$P_{\rm orb}$', r'$R_p/R_*$', r'$a/R_*$', r'$i$', r'$e \cos(\omega)$',
                                  r'$e \sin(\omega)$', r'$q_1$', r'$q_2$', r'$f_p$', r'$A$', r'$B$',
                                  r'$C$', r'$D$', r'$R_{p,2}/R_*$', r'$R_{p,2}/R_*$ Offset', r'$C_1$', r'$C_2$', r'$C_3$',
                                  r'$C_4$', r'$C_5$', r'$C_6$', r'$C_7$', r'$C_8$', r'$C_9$',
                                  r'$C_{10}$', r'$C_{11}$', r'$C_{12}$', r'$C_{13}$', r'$C_{14}$',
                                  r'$C_{15}$', r'$C_{16}$', r'$C_{17}$', r'$C_{18}$', r'$C_{19}$',
                                  r'$C_{20}$', r'$C_{21}$',r'$D_1$', r'$D_2$', r'$D_3$', r'$S_1$', r'$S_2$', r'$M_1$',
                                  r'$GP_{amp}$', r'$GP_{Lx}$', r'$GP_{Ly}$', r'$\sigma_F$'])

        gparams_unsorted = np.copy(gparams)
        gparams = np.array([parm for parm in p0_names if parm in gparams])

        uparams_unsorted = np.copy(uparams)
        uparams = np.array([parm for parm in p0_names if parm in uparams])


        # path where outputs are saved
        savepath   = foldername + mode + '/'
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        
        aors = os.listdir(rootpath+planet+'/data/'+channel)
        aors = np.sort([aor for aor in aors if AOR_snip==aor[:len(AOR_snip)]])
        AOR_snip = AOR_snip[1:]
        
        # path to photometry outputs
        filename   = channel + '_datacube_binned_AORs'+AOR_snip+'.dat'
        filenamef  = channel + '_datacube_full_AORs'+AOR_snip+'.dat'
        # Path to previous mcmc results (optional)
        path_params = foldername + mode + '/ResultMCMC_'+mode+'_Params.npy'

        # For datasets where the first AOR is peak-up data
        if cutFirstAOR:
            rawfiles = np.sort(os.listdir(rootpath+planet+'/data/'+channel+'/'+aors[0]+'/'+channel+'/bcd/'))
            rawfiles  = [rawfile for rawfile in rawfiles if '_bcd.fits' in rawfile]
            cut = cut_tmp+len(rawfiles)
        else:
            cut = cut_tmp

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
        breaks = np.sort(breaks)[1:]

        # Calculate the photon noise limit
        flux = np.loadtxt(foldername+filename, usecols=[0], skiprows=1)     # mJr/str
        flux *= rawHeader['GAIN']*rawHeader['EXPTIME']/rawHeader['FLUXCONV']
        sigF_photon_ppm = 1/np.sqrt(np.median(flux))/np.sqrt(64-len(ignoreFrames))*1e6



        # # Everything below is now automated

        # In[8]:


        signalfunc = detec_models.signal

        if 'poly' in mode.lower():
            detecfunc = detec_models.detec_model_poly
        elif 'bliss' in mode.lower():
            detecfunc = detec_models.detec_model_bliss
        elif 'gp' in mode.lower():
            detecfunc = detec_models.detec_model_GP
        else:
            raise NotImplementedError('Only polynomial, BLISS, and GP models are currently implemented! \nmode=\''+mode+'\' does not include \'poly\', \'Poly\', \'bliss\', \'BLISS\', \'gp\', or \'GP\'.')



        # In[10]:


        # loading full data set for BIC calculation afterwards
        data_full = helpers.get_full_data(foldername, filenamef)

        # sigma clip the data
        flux_full, fluxerr_full, time_full, xdata_full, ydata_full, psfxw_full, psfyw_full = helpers.clip_full_data(*data_full, nFrames, cut, ignoreFrames)
        mid_x_full, mid_y_full = np.nanmean(xdata_full), np.nanmean(ydata_full)


        # In[11]:


        # Get Data
        data = helpers.get_data(foldername+filename)
        # Sort data
        flux0, flux_err0, time0, xdata0, ydata0, psfxw0, psfyw0 = helpers.time_sort_data(*data)
        # Trim AOR
        flux, flux_err, time, xdata, ydata, psfxw, psfyw = helpers.time_sort_data(*data, cut=cut)
        # pre-calculation
        mid_x, mid_y = np.mean(xdata), np.mean(ydata)


        ## FIX: peritime doesn't get made
        if True:#'ecosw' in dparams_input and 'esinw' in dparams_input:
            # make photometry plots
            make_plots.plot_photometry(time0, flux0, xdata0, ydata0, psfxw0, psfyw0, 
                            time, flux, xdata, ydata, psfxw, psfyw, breaks, savepath)
            plt.close()
        else:
            # plot raw data
            make_plots.plot_photometry(time0, flux0, xdata0, ydata0, psfxw0, psfyw0, 
                            time, flux, xdata, ydata, psfxw, psfyw, breaks, savepath, peritime)
            plt.close()


        # In[12]:


        # declare where the heaviside break occurs
        if 'hside' in mode.lower():
            p0_obj.s2 = timeaor1
            dparams = np.append(dparams, ['s2'])


        # redefining the zero centroid position
        if 'bliss' not in mode.lower():
            xdata -= mid_x
            ydata -= mid_y
            xdata_full -= mid_x_full
            ydata_full -= mid_y_full


        # In[14]:


        # True if user wants details about the lambda functions created
        debug = False

        # makes list of parameters that won't be fitted 
        dparams = helpers.expand_dparams(dparams_input, mode)  

        # if you want to use the best fit params from a previous MCMC run
        if initializeWithOld:
            Table_par = np.load(path_params)                  # table of best-fit params from prev. run
            index     = np.in1d(p0_names, dparams)            # get the index list of params to be fitted
            nparams   = p0_names[np.where(index==False)[0]]   # get the name list of params to be fitted
            for name in nparams:
                cmd = 'p0_obj.' + name + ' = ' + 'Table_par[\'' + name + '\'][0]'
                try:
                    exec(cmd)
                except Exception as e:
                    print("type error: " + str(e))            # catch errors if you use values from fun with less params


        # In[15]:


        debug = False
        # get p0
        obj = p0_obj
        p0, p0_labels, p0_fancyLabels = helpers.get_p0(p0_names, p0_fancyNames, dparams, p0_obj)
        #p0_signal  = p0
        #if 'sigF' in p0_labels:
        #    p0_signal = p0_signal[:-1]

        # make lambda function
        signalfunc = helpers.make_lambdafunc(signalfunc, dparams, p0_obj, debug=debug)
        if debug:
            print()

        # making lambda function for phasecurve and detector
        astrofunc = helpers.make_lambdafunc(astro_models.ideal_lightcurve, dparams, p0_obj, debug=debug)
        if debug:
            print()

        detecfunc = helpers.make_lambdafunc(detecfunc, dparams, p0_obj, debug=debug)
        if debug:
            print()

        psfwifunc = helpers.make_lambdafunc(detec_models.detec_model_PSFW, dparams, p0_obj, debug=debug)
        if debug:
            print()

        hsidefunc = helpers.make_lambdafunc(detec_models.hside, dparams, p0_obj, debug=debug)
        if debug:
            print()

        tslopefunc = helpers.make_lambdafunc(detec_models.tslope, dparams, p0_obj, debug=debug)
        if debug:
            print()

        # make a lnprior lambda function
        lnpriorfunc = helpers.make_lambdafunc(helpers.lnprior, dparams, obj=p0_obj, debug=debug)

        if gparams != [] or uparams != []:
            def lnprior_custom_gaussian_helper(p0, priorInds, priors, errs):
                prior = 0
                for i in range(len(priorInds)):
                    prior -= 0.5*(((p0[priorInds[i]] - priors[i])/errs[i])**2.)
                return prior

            def lnprior_custom_uniform_helper(p0, priorInds, limits):
                if priorInds == []:
                    return 0
                elif np.any(np.logical_or(np.array(uparams_limits)[:,0] < p0[priorInds],
                                        np.array(uparams_limits)[:,1] > p0[priorInds])):
                    return -np.inf
                else:
                    return 0

            def lnprior_custom_gamma_helper(p0, priorInd, shape, rate):
                if priorInd is not None:
                    x = np.exp(p0[priorInd])
                    alpha = shape
                    beta = rate
                    return np.log(beta**alpha * x**(alpha-1) * np.exp(-beta*x) / np.math.factorial(alpha-1))
                else:
                    return 0

            gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
            upriorInds = [np.where(p0_labels==upar)[0][0] for upar in uparams if upar in p0_labels]
            if 'gp' in mode.lower():
                gammaInd = np.where(p0_labels=='gpAmp')[0][0]
            else:
                gammaInd = None
            lnprior_custom = lambda p0: (lnprior_custom_gaussian_helper(p0, gpriorInds, priors, errs)+
                                         lnprior_custom_uniform_helper(p0, upriorInds, uparams_limits)+
                                         lnprior_custom_gamma_helper(p0, gammaInd, 1, 100))
        else:
            lnprior_custom = None

        # detemining which params in p0 is part of ideal_lightcurve, detec, psfw
        p0_astro  = inspect.getargspec(astro_models.ideal_lightcurve).args[1:]
        p0_asval, p0_astro, p0_astroFancy  = helpers.get_p0(p0_astro, p0_fancyNames, dparams,p0_obj)

        if 'bliss' not in mode.lower():
            p0_detec  = inspect.getargspec(detecfunc).args[1:]
            p0_deval, p0_detec, p0_detecFancy  = helpers.get_p0(p0_detec, p0_fancyNames, dparams,p0_obj)
        else:
            if 'sigF' in dparams:
                p0_detec = []
                p0_detecFancy = []
                p0_deval = p0_obj.sigF
            else:
                p0_detec = p0_labels[-1]
                p0_detecFancy = p0_fancyLabels[-1]
                p0_deval = p0[-1]

        p0_psfwi  = inspect.getargspec(detec_models.detec_model_PSFW).args[1:]
        p0_psval, p0_psfwi, p0_psfwiFancy  = helpers.get_p0(p0_psfwi, p0_fancyNames, dparams,p0_obj)

        p0_hside  = inspect.getargspec(detec_models.hside).args[1:]
        p0_hsval, p0_hside, p0_hsideFancy  = helpers.get_p0(p0_hside, p0_fancyNames, dparams,p0_obj)

        p0_tslope  = inspect.getargspec(detec_models.tslope).args[1:]
        p0_tsval, p0_tslope, p0_tslopeFancy  = helpers.get_p0(p0_tslope, p0_fancyNames, dparams,p0_obj)


        # initial astro model
        astro_guess = astrofunc(time, *p0_asval)
        resid       = flux/astro_guess

        if 'bliss' in mode.lower():
            make_plots.plot_centroids(xdata0, ydata0, xdata, ydata, savepath)

            signal_inputs = bliss.precompute(flux, time, xdata, ydata, psfxw, psfyw, mode,
                                             astro_guess, blissNBin, savepath)
        elif 'gp' in mode.lower():
            signal_inputs = [flux, time, xdata, ydata, psfxw, psfyw, mode]
            detec_inputs = [flux, xdata, ydata, time, True, astro_guess]
        elif 'pld' in mode.lower():
            #Something will need to go here
            print('PLD not yet implemented!')
        elif 'poly' in mode.lower():# and 'psfw' in mode.lower():
            signal_inputs = [flux, time, xdata, ydata, psfxw, psfyw, mode]
            detec_inputs = [xdata, ydata, mode]


            
            
            
            
        # Run a first fit on the detector parameters to get into the right ballpark
        if runMCMC:
            if not initializeWithOld and 'bliss' not in mode.lower() and 'gp' not in mode.lower():
                spyFunc0 = lambda p0_temp, inputs: np.mean((resid-detecfunc(inputs, *p0_temp))**2)
                spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_detec))], detec_inputs, 'Nelder-Mead')

                # replace p0 with new detector coefficient values
                if spyResult0.success:
                    p0[np.where(np.in1d(p0_labels,p0_detec))] = spyResult0.x
                    resid /= detecfunc(detec_inputs, *p0[np.where(np.in1d(p0_labels,p0_detec))])



                # 2) get initial guess for psfw model
                if 'psfw' in mode.lower():
                    spyFunc0 = lambda p0_temp: np.mean((resid-psfwifunc([psfxw, psfyw], *p0_temp))**2)
                    spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_psfwi))], method='Nelder-Mead')

                    # replace p0 with new detector coefficient values
                    if spyResult0.success:
                        p0[np.where(np.in1d(p0_labels,p0_psfwi))] = spyResult0.x
                        resid /= psfwifunc([psfxw, psfyw], *p0[np.where(np.in1d(p0_labels,p0_psfwi))])

                # 3) get initial guess for hside model
                if 'hside' in mode.lower():
                    spyFunc0 = lambda p0_temp: np.mean((resid-hsidefunc(time, *p0_temp))**2)
                    spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_hside))], method='Nelder-Mead')

                    # replace p0 with new detector coefficient values
                    if spyResult0.success:
                        p0[np.where(np.in1d(p0_labels,p0_hside))] = spyResult0.x
                        resid /= hsidefunc(time, *p0[np.where(np.in1d(p0_labels,p0_hside))])

                if 'tslope' in mode.lower():
                    spyFunc0 = lambda p0_temp: np.mean((resid-tslopefunc(time, *p0_temp))**2)
                    spyResult0 = scipy.optimize.minimize(spyFunc0, p0[np.where(np.in1d(p0_labels,p0_tslope))], method='Nelder-Mead')

                    # replace p0 with new detector coefficient values
                    if spyResult0.success:
                        p0[np.where(np.in1d(p0_labels,p0_tslope))] = spyResult0.x
                        resid /= tslopefunc(time, *p0[np.where(np.in1d(p0_labels,p0_tslope))])


                # initial guess
                signal_guess = signalfunc(signal_inputs, *p0)
                #includes psfw and/or hside functions if they're being fit
                detec_full_guess = signal_guess/astro_guess
            
            
            
            
            
            
       
            
            
        ## If GP, run initial full optimization to find best location
        if runMCMC and 'gp' in mode.lower():
            checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

            initial_lnprob = helpers.lnprob(p0, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom)

            spyFunc_full = lambda p0_temp, inputs: -helpers.lnprob(p0_temp, *inputs)

            nIterScipy = 10

            final_lnprob = -np.inf
            p0_optimized = []
            p0_temps = []
            print('Running iterative scipy.optimize')
            from tqdm import tqdm
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

                spyResult_full = scipy.optimize.minimize(spyFunc_full, p0_temp, [signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom], 'Nelder-Mead')
                lnprob_temp = helpers.lnprob(spyResult_full.x, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom)

                p0_temps.append(np.copy(spyResult_full.x))

                if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
                    final_lnprob = lnprob_temp
                    p0_optimized = np.copy(spyResult_full.x)

                    if final_lnprob > initial_lnprob:
                        print('Improved ln-likelihood!')
                        print("ln-likelihood: {0:.2f}".format(final_lnprob))
                        p0 = np.copy(p0_optimized)

            astro_guess = astrofunc(time, *p0[np.where(np.in1d(p0_labels,p0_astro))])
            signal_guess = signalfunc(signal_inputs, *p0)
            #includes psfw and/or hside functions if they're being fit
            detec_full_guess = signal_guess/astro_guess

        
            
            
            
        ## If GP, run an MCMC centred at the location of each optimization to break free of local minima
        if runMCMC and 'gp' in mode.lower():
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

                #sampler
                sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, a = 2,
                                                args=(signalfunc, lnpriorfunc, 
                                                      signal_inputs, checkPhasePhis, lnprior_custom))

                priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
                iters = 10
                while np.any(priorlnls) and iters>0:
            #         print('Warning: Some of the initial values fail the lnprior!')
            #         print('Trying to re-draw positions...')
                    p0_rel_errs /= 1.5
                    pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
                    priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
                    iters -= 1
                if iters==0 and np.any(priorlnls):
                    print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!')

                #Second burn-in
                #Do quick burn-in to get walkers spread out
                tic = t.time()
                pos1, prob, state = sampler.run_mcmc(pos0, np.rint(nBurnInSteps1/nwalkers), progress=False)
                print('Mean burn-in acceptance fraction: {0:.3f}'
                                .format(np.median(sampler.acceptance_fraction)))
                # sampler.reset()
                toc = t.time()
                print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))

                p0_temps_mcmc.append(np.copy(sampler.flatchain[np.argmax(sampler.flatlnprobability)]))
            
            
            
            
            
        # If GP, run a final optimization

        if runMCMC and 'gp' in mode.lower():
            checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

            initial_lnprob = helpers.lnprob(p0, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom)

            spyFunc_full = lambda p0_temp, inputs: -helpers.lnprob(p0_temp, *inputs)

            final_lnprob = -np.inf
            p0_optimized = []
            p0_temps_final = []
            print('Running second iterative scipy.optimize')
            from tqdm import tqdm
            for p0_temp in tqdm(p0_temps_mcmc):

                spyResult_full = scipy.optimize.minimize(spyFunc_full, p0_temp, [signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom], 'Nelder-Mead')
                lnprob_temp = helpers.lnprob(spyResult_full.x, signalfunc, lnpriorfunc, signal_inputs, checkPhasePhis, lnprior_custom)

                p0_temps_final.append(np.copy(spyResult_full.x))

                if np.isfinite(lnprob_temp) and lnprob_temp > final_lnprob:
                    final_lnprob = lnprob_temp
                    p0_optimized = np.copy(spyResult_full.x)

                    if final_lnprob > initial_lnprob:
                        print('Improved ln-likelihood!')
                        print("ln-likelihood: {0:.2f}".format(final_lnprob))
                        p0 = np.copy(p0_optimized)

            
            
            
            
            
        # If not a GP, run a first MCMC burn-in
        if runMCMC and 'gp' not in mode.lower():
            ndim, nwalkers = len(p0), 150

            # get scattered starting point in parameter space 
            # MUST HAVE THE INITIAL SPREAD SUCH THAT EVERY SINGLE WALKER PASSES lnpriorfunc AND lnprior_custom
            p0_rel_errs = 1e-4*np.ones_like(p0)
            gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
            p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
            pos0 = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(nwalkers)])

            checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

            #sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, a = 2,
                                            args=(signalfunc, lnpriorfunc, 
                                                  signal_inputs, checkPhasePhis, lnprior_custom))

            priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
            iters = 10
            while np.any(priorlnls) and iters>0:
            #         print('Warning: Some of the initial values fail the lnprior!')
            #         print('Trying to re-draw positions...')
                p0_rel_errs /= 1.5
                pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
                priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
                iters -= 1
            if iters==0 and np.any(priorlnls):
                print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!')

            #First burn-in
            tic = t.time()
            print('Running first burn-in')
            pos1, prob, state = sampler.run_mcmc(pos0, np.rint(nBurnInSteps1/nwalkers), progress=True)
            print('Mean burn-in acceptance fraction: {0:.3f}'
                        .format(np.median(sampler.acceptance_fraction)))


            fname = savepath+'MCMC_'+mode+'_burnin1Walkers.pdf'
            helpers.walk_style(len(p0), nwalkers, sampler.chain, 10, int(np.rint(nBurnInSteps1/nwalkers)), p0_fancyLabels, fname)
            plt.close()

            p0 = sampler.flatchain[np.argmax(sampler.flatlnprobability)]
        
        
        
        
        
        
        
        # plot detector initial guess
        astro_guess = astrofunc(time, *p0[np.where(np.in1d(p0_labels,p0_astro))])
        signal_guess = signalfunc(signal_inputs, *p0)
        #includes psfw and/or hside functions if they're being fit
        detec_full_guess = signal_guess/astro_guess
        make_plots.plot_init_guess(time, flux, astro_guess, detec_full_guess, savepath)
        
        
        # In[ ]:


        ndim, nwalkers = len(p0), 150

        if runMCMC:
            # get scattered starting point in parameter space 
            # MUST HAVE THE INITIAL SPREAD SUCH THAT EVERY SINGLE WALKER PASSES lnpriorfunc AND lnprior_custom
            p0_rel_errs = 1e-3*np.ones_like(p0)
            gpriorInds = [np.where(p0_labels==gpar)[0][0] for gpar in gparams]
            p0_rel_errs[gpriorInds] = np.array(errs)/np.array(priors)
            pos0 = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(nwalkers)])

            checkPhasePhis = np.linspace(-np.pi,np.pi,1000)

            #sampler
            sampler = emcee.EnsembleSampler(nwalkers, ndim, helpers.lnprob, a = 2,
                                            args=(signalfunc, lnpriorfunc, 
                                                  signal_inputs, checkPhasePhis, lnprior_custom))

            priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
            iters = 10
            while np.any(priorlnls) and iters>0:
            #         print('Warning: Some of the initial values fail the lnprior!')
            #         print('Trying to re-draw positions...')
                p0_rel_errs /= 1.5
                pos0[priorlnls] = np.array([p0*(1+p0_rel_errs*np.random.randn(ndim))+p0_rel_errs/10.*np.abs(np.random.randn(ndim)) for i in range(np.sum(priorlnls))])
                priorlnls = np.array([(lnpriorfunc(*p_tmp, mode, checkPhasePhis) != 0.0 or (lnprior_custom != 'none' and np.isinf(lnprior_custom(p_tmp)))) for p_tmp in pos0])
                iters -= 1
            if iters==0 and np.any(priorlnls):
                print('Warning: Some of the initial values still fail the lnprior and the following MCMC will likely not work!')

            #Second burn-in
            #Do quick burn-in to get walkers spread out
            tic = t.time()
            print('Running second burn-in')
            pos1, prob, state = sampler.run_mcmc(pos0, np.rint(nBurnInSteps2/nwalkers), progress=False)
            print('Mean burn-in acceptance fraction: {0:.3f}'
                            .format(np.median(sampler.acceptance_fraction)))
            fname = savepath+'MCMC_'+mode+'_burninWalkers.pdf'
            helpers.walk_style(len(p0), nwalkers, sampler.chain, 10, int(np.rint(nBurnInSteps2/nwalkers)), p0_fancyLabels, fname)
            sampler.reset()
            toc = t.time()
            print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))


            #Run production
            #Run that will be saved
            tic = t.time()
            # Continue from last positions and run production
            print('Running production')
            pos2, prob, state = sampler.run_mcmc(pos1, np.rint(nProductionSteps/nwalkers), progress=False)
            print("Mean acceptance fraction: {0:.3f}"
                            .format(np.mean(sampler.acceptance_fraction)))
            toc = t.time()
            print('MCMC runtime = %.2f min\n' % ((toc-tic)/60.))


            #Saving MCMC Results
            pathchain = savepath + 'samplerchain_'+mode+'.npy'
            pathposit = savepath + 'samplerposi_'+mode+'.npy'
            pathlnpro = savepath + 'samplerlnpr_'+mode+'.npy'
            np.save(pathchain, sampler.chain)
            np.save(pathposit, pos2)
            np.save(pathlnpro, prob)

            chain = sampler.chain

        else:

            pathchain = savepath + 'samplerchain_'+mode+'.npy'
            chain = np.load(pathchain)
            pathlnpro = savepath + 'samplerlnpr_'+mode+'.npy'
            if os.path.exists(pathlnpro):
                lnprobability = np.load(pathlnpro)

        samples = chain.reshape((-1, ndim))


        # ## Fold inclination back around since i>90 is meaningless

        # In[ ]:
        if 'inc' in p0_labels:
            pos_inc = np.where(p0_labels == 'inc')[0][0]
            samples[np.where(samples[:,pos_inc] > 90)[0],pos_inc] = 180 - samples[np.where(samples[:,pos_inc] > 90)[0],pos_inc]


        # In[ ]:


        #print the results

        (MCMC_Results) = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84],axis=0)))))
        p0_mcmc = np.median(samples, axis=0)

        # taking max lnprob params instead of median bc degeneracy
        if usebestfit == True: 
            if runMCMC == True:
                maxk, maxiter = np.unravel_index((sampler.lnprobability).argmax(), (sampler.lnprobability).shape)
                p0_mcmc = sampler.chain[maxk, maxiter,:]
            else:
                maxk, maxiter = np.unravel_index((lnprobability).argmax(), (lnprobability).shape)
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


        # In[ ]:


        # printing output from MCMC
        out = "MCMC result:\n\n"
        for i in range(len(p0)):
            out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format(p0_labels[i],MCMC_Results[i][0], MCMC_Results[i][1], MCMC_Results[i][2])

        # getting and printing the phase offset
        As = samples[:,np.where(p0_labels == 'A')[0][0]][:,np.newaxis]
        Bs = samples[:,np.where(p0_labels == 'B')[0][0]][:,np.newaxis]
        phis = np.linspace(-np.pi,np.pi,1000)
        offsets = []
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
            fp_MCMC = p0_obj.fp
        if 'rp' in p0_labels:
            rp_MCMC = samples[:,np.where(p0_labels == 'rp')[0][0]]*np.sqrt(compFactor)
        else:
            rp_MCMC = p0_obj.rp



        f = fits.open(phoenixSpectraFile)
        fStar = f[0].data*1e-1 # 'erg/s/cm^2/cm' to kg/s^3
        f.close()
        f = fits.open(phoenixWavFile)
        wavStar = f[0].data*1e-4 # angstrom to micron
        f.close()

        def planck(wav, T):
            intensity = (2.0*const.h.value*const.c.value**2) / ((wav**5) * (np.exp(const.h.value*const.c.value/(wav*const.k_B.value*T)) - 1.0))
            return intensity
        def fluxDiff(temp, fStarSummed, wavs):
            #factor of pi likely needed to account for emitting area (pi*rstar^2 where rstar=1)
            return (np.sum(planck(wavs, temp)*np.pi)-fStarSummed)**2
        temps = np.linspace(5500, 10000, 3536)
        if channel == 'ch1':
            incides = np.where(np.logical_and(wavStar < 4., wavStar > 3.))[0]
        else:
            incides = np.where(np.logical_and(wavStar < 5., wavStar > 4.))[0]
        diffs = [fluxDiff(temp, np.sum(fStar[incides]), wavStar[incides]*1e-6) for temp in temps]
        tstar_b = temps[np.argmin(diffs)]

        tday = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_b))-1)/(fp_MCMC/rp_MCMC**2)))**-1
        tnight = const.h.value*const.c.value/(const.k_B.value*wav)*(np.log(1+(np.exp(const.h.value*const.c.value/(const.k_B.value*wav*tstar_b))-1)/(fp_MCMC*(1-2*As[:,0])/rp_MCMC**2)))**-1

        out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Day: ', np.median(tday), np.percentile(tday, 84)-np.median(tday), np.median(tday)-np.percentile(tday, 16))
        out += '{:>8} = {:>16}  +{:>16}  -{:>16}\n'.format('T Night: ', np.nanmedian(tnight), np.nanpercentile(tnight, 84)-np.nanmedian(tnight), np.nanmedian(tnight)-np.nanpercentile(tnight, 16))
        out += 'For T_{*,b} = '+str(tstar_b)+'\n'

        print(out)
        with open(savepath+'MCMC_RESULTS_'+mode+'.txt','w') as file:
            file.write(out) 

        # In[ ]:


        ind_a = len(p0_astro) # index where the astro params end
        labels = p0_fancyLabels[:ind_a]

        fname = savepath+'MCMC_'+mode+'_astroWalkers.pdf'
        helpers.walk_style(ind_a, nwalkers, chain, 10, chain.shape[1], labels, fname)


        # In[ ]:


        if 'bliss' not in mode.lower() or 'sigF' not in dparams:
            labels = p0_fancyLabels[ind_a:]
            fname = savepath+'MCMC_'+mode+'_detecWalkers.pdf'
            helpers.walk_style(len(p0)-ind_a, nwalkers, chain[:,:,ind_a:], 10, chain.shape[1], labels, fname)


        # In[ ]:


        #save us some time when not running MCMC since this plot takes forever to make....
        if runMCMC:
            fig = corner.corner(samples[:,:ind_a], labels=p0_fancyLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True, 
                                plot_datapoints=True, title_kwargs={"fontsize": 12})
            plotname = savepath + 'MCMC_'+mode+'_corner.pdf'
            fig.savefig(plotname, bbox_inches='tight')
            plt.close()

        #     fig = corner.corner(samples, labels=p0_fancyLabels, quantiles=[0.16, 0.5, 0.84], show_titles=True, 
        #                         plot_datapoints=True, title_kwargs={"fontsize": 12})
        #     plotname = savepath + 'MCMC_'+mode+'_corner_complete.pdf'
        #     fig.savefig(plotname, bbox_inches='tight')
        #     plt.close()


        # In[ ]:


        if 'ecosw' in p0_labels and 'esinw' in p0_labels:
            '''Eccentricity and Longitude of Periastron Coefficient'''

            ind1 = np.where(p0_labels == 'ecosw')[0][0]
            ind2 = np.where(p0_labels == 'esinw')[0][0]
            e_chain = np.sqrt(samples[:,ind1]**2 + samples[:,ind2]**2)
            w_chain = np.arctan2(samples[:,ind2], samples[:,ind1]) #np.arctan(samples[:,ind2]/samples[:,ind1])
            binse = np.linspace(np.min(e_chain), np.max(e_chain), 20)
            binsw = np.linspace(np.min(w_chain), np.max(w_chain), 20)

            fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (8,6))
            axes[0,0].hist(samples[:,ind1], bins=np.linspace(np.min(samples[:,ind1]), np.max(samples[:,ind1]), 20), color='k', alpha=0.3)
            axes[0,1].hist(samples[:,ind2], bins=np.linspace(np.min(samples[:,ind2]), np.max(samples[:,ind2]), 20), color='k', alpha=0.3)
            axes[1,0].hist(e_chain, binse, color='k', alpha=0.3)
            axes[1,1].hist(w_chain, binsw, color='k', alpha=0.3)

            plt.setp(axes[0,0].get_yticklabels(), visible=False)
            plt.setp(axes[0,1].get_yticklabels(), visible=False)
            plt.setp(axes[1,0].get_yticklabels(), visible=False)
            plt.setp(axes[1,1].get_yticklabels(), visible=False)

            plt.setp(axes[0,0].get_xticklabels(), rotation = 45)
            plt.setp(axes[0,1].get_xticklabels(), rotation = 45)
            plt.setp(axes[1,0].get_xticklabels(), rotation = 45)
            plt.setp(axes[1,1].get_xticklabels(), rotation = 45)

            axes[0,0].set_title('$e \cos (\omega)$', fontsize=12)
            axes[0,1].set_title('$e \sin (\omega)$', fontsize=12)
            axes[1,0].set_title('$e$', fontsize=12)
            axes[1,1].set_title('$\omega$', fontsize=12)

            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.2)
            plotname = savepath + 'MCMC_'+mode+'_ecc-omega.pdf'
            fig.savefig(plotname, bbox_inches='tight')
            plt.close()


        # In[ ]:


        if 'q1' in p0_labels and 'q2' in p0_labels:
            '''Stellar Limb Darkening Parameters'''

            ind1 = np.where(p0_labels == 'q1')[0][0]
            ind2 = np.where(p0_labels == 'q2')[0][0]
            u1_chain = 2*np.sqrt(samples[:,ind1]**2)*samples[:,ind2]
            u2_chain = np.sqrt(samples[:,ind1]**2)*(1-2*samples[:,ind2])
            binsu1 = np.linspace(np.min(u1_chain), np.max(u1_chain), 20)
            binsu2 = np.linspace(np.min(u2_chain), np.max(u2_chain), 20)

            fig, axes = plt.subplots(ncols = 2, nrows = 2, figsize = (8,6))
            axes[0,0].hist(samples[:,ind1], bins=np.linspace(np.min(samples[:,ind1]), np.max(samples[:,ind1]), 20), color='k', alpha=0.3)
            axes[0,1].hist(samples[:,ind2], bins=np.linspace(np.min(samples[:,ind2]), np.max(samples[:,ind2]), 20), color='k', alpha=0.3)
            axes[1,0].hist(u1_chain, binsu1, color='k', alpha=0.3)
            axes[1,1].hist(u2_chain, binsu2, color='k', alpha=0.3)

            plt.setp(axes[0,0].get_yticklabels(), visible=False)
            plt.setp(axes[0,1].get_yticklabels(), visible=False)
            plt.setp(axes[1,0].get_yticklabels(), visible=False)
            plt.setp(axes[1,1].get_yticklabels(), visible=False)

            plt.setp(axes[0,0].get_xticklabels(), rotation = 45)
            plt.setp(axes[0,1].get_xticklabels(), rotation = 45)
            plt.setp(axes[1,0].get_xticklabels(), rotation = 45)
            plt.setp(axes[1,1].get_xticklabels(), rotation = 45)

            axes[0,0].set_title('$q_1$', fontsize=12)
            axes[0,1].set_title('$q_2$', fontsize=12)
            axes[1,0].set_title('$u_1$', fontsize=12)
            axes[1,1].set_title('$u_2$', fontsize=12)

            fig.subplots_adjust(hspace=0.5)
            fig.subplots_adjust(wspace=0.2)
            plotname = savepath + 'MCMC_'+mode+'_limbdark.pdf'
            fig.savefig(plotname, bbox_inches='tight')
            plt.close()


        # In[ ]:


        #Clean out the RAM
        samples = None
        sampler = None
        chain = None


        # In[ ]:


        # generate uniformly spaced time array for plot purposes
        time2 = np.linspace(np.min(time), np.max(time), 1000)

        # generate the models from best-fit parameters
        mcmc_signal = signalfunc(signal_inputs, *p0_mcmc)
        mcmc_lightcurve = astrofunc(time, *p0_mcmc[:ind_a])
        mcmc_detec = mcmc_signal/mcmc_lightcurve

        #for higher-rez red curve
        mcmc_lightplot  = astrofunc(time2, *p0_mcmc[:ind_a])


        # Differences from eccentricity
        # if 'ecosw' in dparams and 'esinw' in dparams:
        #     # converting time into orbital phases
        #     if 't0' in p0_labels:
        #         t0MCMC = p0_mcmc[np.where(p0_labels == 't0')[0][0]]
        #     else:
        #         t0MCMC = p0_obj.t0
        #     if 'per' in p0_labels:
        #         perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
        #     else:
        #         perMCMC = p0_obj.per
        #     x = (time-t0MCMC)/perMCMC
        #     orbNum = int(np.min(x))
        #     if np.min(x)>0:
        #         orbNum += 1
        #     x -= orbNum
        # 
        #     orb_breaks = np.empty(len(breaks))
        #     for j in range(len(breaks)):
        #         orb_breaks[j] = ((breaks[j]-t0MCMC)/perMCMC-orbNum)      
        # else:
        #     x       = time - peritime
        #     xbreaks = breaks - peritime

        # FIX: peritime isn't defined, so just using time for all plots for now
        orb_breaks = breaks
        if True:#'ecosw' in dparams and 'esinw' in dparams:
            make_plots.plot_bestfit(time, flux, mcmc_lightcurve, mcmc_detec, mode, orb_breaks, savepath, nbin=bestfitNbin, fontsize=24)
            plt.close()
        else:
            # FIX: make this default plotting option
            make_plots_custom.plot_bestfit(x, flux, mcmc_lightcurve, mcmc_detec, 
                                           mode, xbreaks, savepath, peritime=0, nbin=bestfitNbin)
            plt.close()


        # In[ ]:


        # if McCubed is installed
        try:
            from mc3.stats import time_avg
            intTime = (time[1]-time[0])
            minBins = 5
            residuals = flux/mcmc_detec - mcmc_lightcurve

            #WARNING: these durations assume circular orbits!!!
            ingrDuration = helpers.getIngressDuration(p0_mcmc, p0_labels, p0_obj, intTime)
            occDuration = helpers.getOccultationDuration(p0_mcmc, p0_labels, p0_obj, intTime)

            make_plots.plot_rednoise(residuals, minBins, ingrDuration, occDuration, intTime, mode, savepath, savetxt=True)
            plt.close()
        except ImportError:
            #Noise vs bin-size to look for red noise
            residuals = flux/mcmc_detec - mcmc_lightcurve

            sigmas = []
            for i in range(3,len(residuals)):
                sigmas.append(helpers.binnedNoise(x,residuals,i))
            sigmas = np.array(sigmas)

            n_binned = len(residuals)/np.arange(3,len(residuals))

            #In case there is a NaN or something while binning
            n_binned = n_binned[np.where(np.isfinite(sigmas))[0]]
            sigmas = sigmas[np.where(np.isfinite(sigmas))[0]]


            ax = plt.gca()
            ax.set_yscale('log')
            ax.set_xscale('log')

            if 'sigF' in p0_labels:
                sigFMCMC = p0_mcmc[np.where(p0_labels == 'sigF')[0][0]]
            else:
                sigFMCMC = p0_obj.sigF
            if 'rp' in p0_labels:
                rpMCMC = p0_mcmc[np.where(p0_labels == 'rp')[0][0]]
            else:
                rpMCMC = p0_obj.rp
            if 'a' in p0_labels:
                aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
            else:
                aMCMC = p0_obj.a
            if 'per' in p0_labels:
                perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
            else:
                perMCMC = p0_obj.per

            #FIX: WARNING: these durations assume circular orbits!!!
            eclDuration = (2*rpMCMC/(2*np.pi*aMCMC/perMCMC))/((time[1]-time[0])) #Eclipse/transit ingress time
            trDuration = (2/(2*np.pi*aMCMC/perMCMC))/((time[1]-time[0])) #Transit/eclipse duration

            ax.plot(n_binned, sigmas, c='black', label='Data')
            ax.plot([n_binned[-1],n_binned[0]], [sigFMCMC, sigFMCMC/np.sqrt(n_binned[0])], c='red', label='White Noise')
            ylim = ax.get_ylim()
            plt.plot([eclDuration,eclDuration],ylim, color='black', ls='--', alpha=0.6)
            plt.plot([trDuration,trDuration],ylim, color='black', ls='-.', alpha=0.6)
            ax.set_ylim(ylim)
            plt.ylabel(r'$\sigma$ (ppm)', fontsize='x-large')
            plt.xlabel(r'N$_{\rm binned}$', fontsize='x-large')
            plt.legend(loc='best', fontsize='large')
            plotname = savepath + 'MCMC_'+mode+'_RedNoise.pdf'
            plt.savefig(plotname, bbox_inches='tight')
            plt.close()


            #Figure out how much red noise we have

            #Eclipse Duration
            sreal = sigmas[np.where(n_binned<=eclDuration)[0][0]]*1e6
            s0 = sigFMCMC/np.sqrt(n_binned[np.where(n_binned<=eclDuration)[0][0]])*1e6
            outStr = 'Over Ingress ('+str(round(eclDuration*((time[1]-time[0]))*24*60, 1))+' min):\n'
            outStr += 'Expected Noise (ppm)\t'+'Observed Noise (ppm)\n'
            outStr += str(s0)+'\t'+str(sreal)+'\n'
            outStr += 'Observed/Expected\n'
            outStr += str(sreal/s0)+'\n\n'
            #Transit Duration
            sreal = sigmas[np.where(n_binned<=trDuration)[0][0]]*1e6
            s0 = sigFMCMC/np.sqrt(n_binned[np.where(n_binned<=trDuration)[0][0]])*1e6
            outStr += 'Over Transit/Eclipse ('+str(round(trDuration*((time[1]-time[0]))*24*60, 1))+' min):\n'
            outStr += 'Expected Noise (ppm)\t'+'Observed Noise (ppm)\n'
            outStr += str(s0)+'\t'+str(sreal)+'\n'
            outStr += 'Observed/Expected\n'
            outStr += str(sreal/s0)

            print(outStr)
            with open(plotname[:-3]+'txt','w') as file:
                file.write(outStr)


        # # Is $\chi ^2$ improving?

        # In[ ]:


        #Binned data
        data = flux/mcmc_detec
        astro  = mcmc_lightcurve
        if 'sigF' in p0_labels:
            sigFMCMC = p0_mcmc[np.where(p0_labels == 'sigF')[0][0]]
        else:
            sigFMCMC = p0_obj.sigF
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

        if 'gp' not in mode.lower():
            #Unbinned data
            '''Get model'''
            astro_full   = astrofunc(time_full, *p0_mcmc[:ind_a])
            if 'bliss' in mode.lower():
                signal_inputs_full = bliss.precompute(flux_full, time_full, xdata_full, ydata_full,
                                                      psfxw_full, psfyw_full, mode,
                                                      astro_full, blissNBin, savepath, False)
            elif 'pld' in mode.lower():
                #Something will need to go here
                print('PLD not yet implemented!')
            elif 'gp' in mode.lower():
                signal_inputs_full = [flux_full, time_full, xdata_full, ydata_full, psfxw_full, psfyw_full, mode]
            elif 'poly' in mode.lower():# and 'psfw' in mode.lower():
                signal_inputs_full = (flux_full, time_full, xdata_full, ydata_full, psfxw_full, psfyw_full, mode)

            signal_full = signalfunc(signal_inputs_full, *p0_mcmc)
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
            BIC = {4}""".format(chis, chis/len(xdata_full), logL, E, BIC)

        with open(savepath+'EVIDENCE_'+mode+'.txt','w') as file:
            file.write(out)
        print(out)


        # In[ ]:


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


        # In[ ]:


        # determining in-eclipse and in-transit index

        # generating transit model

        if 't0' in p0_labels:
            t0MCMC = p0_mcmc[np.where(p0_labels == 't0')[0][0]]
        else:
            t0MCMC = p0_obj.t0
        if 'per' in p0_labels:
            perMCMC = p0_mcmc[np.where(p0_labels == 'per')[0][0]]
        else:
            perMCMC = p0_obj.per
        if 'rp' in p0_labels:
            rpMCMC = p0_mcmc[np.where(p0_labels == 'rp')[0][0]]
        else:
            rpMCMC = p0_obj.rp
        if 'a' in p0_labels:
            aMCMC = p0_mcmc[np.where(p0_labels == 'a')[0][0]]
        else:
            aMCMC = p0_obj.a
        if 'inc' in p0_labels:
            incMCMC = p0_mcmc[np.where(p0_labels == 'inc')[0][0]]
        else:
            incMCMC = p0_obj.inc
        if 'ecosw' in p0_labels:
            ecoswMCMC = p0_mcmc[np.where(p0_labels == 'ecosw')[0][0]]
        else:
            ecoswMCMC = p0_obj.ecosw
        if 'esinw' in p0_labels:
            esinwMCMC = p0_mcmc[np.where(p0_labels == 'esinw')[0][0]]
        else:
            esinwMCMC = p0_obj.esinw
        if 'q1' in p0_labels:
            q1MCMC = p0_mcmc[np.where(p0_labels == 'q1')[0][0]]
        else:
            q1MCMC = p0_obj.q1
        if 'q2' in p0_labels:
            q2MCMC = p0_mcmc[np.where(p0_labels == 'q2')[0][0]]
        else:
            q2MCMC = p0_obj.q2
        if 'fp'in p0_labels:
            fpMCMC = p0_mcmc[np.where(p0_labels == 'fp')[0][0]]
        else:
            fpMCMC = p0_obj.fp

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


        # In[ ]:


        residual = flux/mcmc_detec - mcmc_lightcurve

        data1 = [xdata, ydata, psfxw, psfyw, flux, residual]
        data2 = [xdata[ind_ecli1], ydata[ind_ecli1], psfxw[ind_ecli1], psfyw[ind_ecli1], flux[ind_ecli1], residual[ind_ecli1]]
        data3 = [xdata[ind_trans], ydata[ind_trans], psfxw[ind_trans], psfyw[ind_trans], flux[ind_trans], residual[ind_trans]]
        data4 = [xdata[ind_ecli2], ydata[ind_ecli2], psfxw[ind_ecli2], psfyw[ind_ecli2], flux[ind_ecli2], residual[ind_ecli2]]
	
        plotname = savepath + 'MCMC_'+mode+'_7.pdf'
        make_plots.triangle_colors(data1, data2, data3, data4, plotname)

