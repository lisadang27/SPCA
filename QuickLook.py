import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

import os

from SPCA import make_plots, helpers


planets = ['CoRoT-2b', 'HAT-P-7b', 'HD189733b', 'HD209458b', 'KELT-1b', 'KELT-16b', 'KELT-9b', 'MASCARA-1b', 'Qatar-1b', 'WASP-103b', 'WASP-12b', 'WASP-12b_old', 'WASP-14b', 'WASP-18b', 'WASP-19b', 'WASP-33b', 'WASP-43b', 'HD149026b']
channels = ['ch2' for planet in planets]

rootpath = '/homes/picaro/bellt/research/'

for planet, channel in zip(planets, channels):
        
    AOR_snip = ''
    with open(rootpath+planet+'/analysis/aorSnippet.txt') as f:
        AOR_snip = f.readline().strip()[1:]

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
    
    breakpath = rootpath+planet+'/analysis/'+channel+'/aorBreaks.txt'
    with open(breakpath, 'r') as file:
        breaks = np.array(file.readline().strip().split(' ')).astype(float)

    filename   = channel + '_datacube_binned_AORs'+AOR_snip+'.dat'
    flux, time, xdata, ydata, psfxw, psfyw = helpers.get_data(foldername, filename, 'Poly2_v1')
    
    # Make the plots
    print(planet, channel)
    make_plots.plot_photometry(time, flux, xdata, ydata, psfxw, psfyw, 
                               time, flux, xdata, ydata, psfxw, psfyw, breaks, showPlot=True)
        
    response = input('Would you like to cut the first AOR (y/n)? ')

    if response=='y':
        cutFirstAOR = True
    else:
        cutFirstAOR = False

    with open(rootpath+planet+'/analysis/'+channel+'/cutFirstAOR.txt', 'w') as f:
        f.write(str(cutFirstAOR))
