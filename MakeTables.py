#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import decimal


mainpath   = '/home/taylor/Documents/Research/spitzer/'
planets = ['WASP12b', 'WASP121b', 'WASP121b', 'MASCARA1b', 'KELT16b']            # Name of the planet
channels = ['ch2', 'ch2', 'ch1', 'ch2', 'ch2']                     # Spitzer channel
twoLimits = False
bicThresh = 5


def roundToSigFigs(x, sigFigs=2):
    nDec = -int(np.floor(np.log10(np.abs(x))))+sigFigs-1
    rounded = np.round(x, nDec)
    if nDec <= 0:
        #remove decimals if this should be an integer
        rounded = int(rounded)
    output = str(rounded)
    if nDec > 1 and np.rint(rounded*10**nDec) % 10 == 0:
        #take care of the trailing zero issue
        output += '0'
    return nDec, output
def roundToDec(x, nDec=2):
    rounded = np.round(x, nDec)
    if nDec <= 0:
        #remove decimals if this should be an integer
        rounded = int(rounded)
    output = str(rounded)
    if nDec > 1 and np.rint(rounded*10**nDec) % 10 == 0:
        #take care of the trailing zero issue
        output += '0'
    return output


output1 = ''
output2 = ''
output3 = ''
output4 = ''
for iterNum in range(len(planets)):
    channel = channels[iterNum]
    planet = planets[iterNum]
    
    output1 += planet+' '+channel+'\n'
    output2 += planet+' '+channel+'\n'
    
    with open(mainpath+planet+'/analysis/'+channel+'/bestPhOption.txt') as f:
        line = f.readline().strip()
        foldername = mainpath+line[line.find(planet):]+'/'

    modes = np.sort([x[0].split('/')[-1] for x in os.walk(mainpath+line[line.find(planet):])][1:])

    #find the best modes
    BICs = []
    for mode in modes:
        savepath   = foldername + mode + '/'+ 'ResultMCMC_'+mode+'_Params.npy'
        if not os.path.exists(savepath):
            BICs.append(np.inf)
        else:
            ResultMCMC = np.load(savepath)
            BICs.append(-2*ResultMCMC['evidenceB'][0])
    BICs = np.array(BICs)

    bestBIC = np.min(BICs)

    ######################## MAKE TABLE2 #1 AND #2 ########################

    for i in range(len(modes)):

        mode = modes[i]

        savepath   = foldername + mode + '/' + 'ResultMCMC_'+mode+'_Params.npy'
        if not os.path.exists(savepath):
            continue

        if int(np.round(BICs[i]-bestBIC, 0)) <= bicThresh:
            output1 += '\\textbf{'
            output2 += '\\textbf{'
        else:
            output1 += '        '
            output2 += '        '

        ResultMCMC = np.load(savepath)

        if 'poly' in mode.lower():
            ind = mode.lower().find('poly')
            output1 += mode[ind:ind+5]
            output2 += mode[ind:ind+5]
        elif 'bliss' in mode.lower():
            output1 += 'BLISS'
            output2 += 'BLISS'
        elif 'pld' in mode.lower():
            output1 += 'PLD'
            output2 += 'PLD'
        elif 'gp' in mode.lower():
            output1 += 'GP'
            output2 += 'GP'

        if 'tslope' in mode.lower():
            output1 += '*$f$(t)'
            output2 += '*$f$(t)'

        if 'hside' in mode.lower():
            output1 += '*Step'
            output2 += '*Step'

        if 'psfw' in mode.lower():
            output1 += ', PSFW'
            output2 += ', PSFW'
            
        if int(np.round(BICs[i]-bestBIC, 0)) <= bicThresh:
            output1 += '}& '
            output2 += '}& '
        else:
            output1 += ' & '
            output2 += ' & '

        ind = mode.lower().find('_v')
        vorder = mode[ind+1:].split('_')[0][1:]

        if vorder == '1':
            output1 += '1st Order'
            output2 += '1st Order'
        elif vorder == '2':
            output1 += '2nd Order'
            output2 += '2nd Order'

        if 'ellipse' in mode.lower() and 'offset' in mode.lower():
            output1 += ' + Rotated Ellipse & '
            output2 += ' + Rotated Ellipse & '
        elif 'ellipse' in mode.lower():
            output1 += ' + Ellipse & '
            output2 += ' + Ellipse & '
        else:
            output1 += ' & '
            output2 += ' & '

        nData = ResultMCMC['chi2B'][0]/ResultMCMC['chi2datum'][0]
            
        val = ResultMCMC['A']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'\pm'+err+'$ & '


        val = ResultMCMC['B']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'\pm'+err+'$ & '

        if vorder == '1':
            output1 += '. & . & '
        else:
            val = ResultMCMC['C']
            if twoLimits:
                nDec1, err1 = roundToSigFigs(val[1])
                nDec2, err2 = roundToSigFigs(val[2])
                nDec = np.max([nDec1, nDec2])
                val = roundToDec(val[0], nDec)
                output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
            else:
                nDec, err = roundToSigFigs(np.mean(val[1:]))
                val = roundToDec(val[0], nDec)
                output1 += '$'+val+'\pm'+err+'$ & '

            val = ResultMCMC['D']
            if twoLimits:
                nDec1, err1 = roundToSigFigs(val[1])
                nDec2, err2 = roundToSigFigs(val[2])
                nDec = np.max([nDec1, nDec2])
                val = roundToDec(val[0], nDec)
                output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
            else:
                nDec, err = roundToSigFigs(np.mean(val[1:]))
                val = roundToDec(val[0], nDec)
                output1 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['fp']*1e6
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['sigF']*1e6
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output1 += '$'+val+'\pm'+err+'$ & '

        output1 += str(np.round(ResultMCMC['chi2B'][0], 2))+' & '







        val = ResultMCMC['rp']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'\pm'+err+'$ & '

#         if 'ellipse' in mode.lower():
#             val = ResultMCMC['r2']
#             if twoLimits:
#                 nDec1, err1 = roundToSigFigs(val[1])
#                 nDec2, err2 = roundToSigFigs(val[2])
#                 nDec = np.max([nDec1, nDec2])
#                 val = roundToDec(val[0], nDec)
#                 output2 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
#             else:
#                 nDec, err = roundToSigFigs(np.mean(val[1:]))
#                 val = roundToDec(val[0], nDec)
#                 output2 += '$'+val+'\pm'+err+'$ & '

#             val = [0,0,0]
#             r2 = ResultMCMC['r2'][0]
#             rp = ResultMCMC['rp'][0]
#             val[0] = r2/rp
#             r2err = np.max([ResultMCMC['r2'][1], ResultMCMC['r2'][2]])
#             rperr = np.max([ResultMCMC['rp'][1], ResultMCMC['rp'][2]])
#             val[1] = np.sqrt((r2err/rp)**2+(r2/rp**2*rperr)**2)
#             nDec, err = roundToSigFigs(val[1])
#             val = roundToDec(val[0], nDec)
#             output2 += '$'+val+'\pm'+err+'$ & '
#         else:
#             output2 += '. & . & '

        val = ResultMCMC['offset']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(-val[0], nDec)
            output2 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(-val[0], nDec)
            output2 += '$'+val+'\pm'+err+'$ & '

        if 'v2' in mode.lower():
            Cs =  ResultMCMC['C'][0]
            Cs_err = np.mean(ResultMCMC['C'][1:])
            Ds =  ResultMCMC['D'][0]
            Ds_err = np.mean(ResultMCMC['D'][1:])
            offset2 = 180-np.arctan2(Ds, Cs)*180/np.pi/2
            offset2_err = 1/(1+(Ds/Cs)**2)*np.sqrt((Ds_err/Cs)**2+(Ds/Cs**2*Cs_err)**2)*180/np.pi/2
            nDec, err = roundToSigFigs(offset2_err)
            val = roundToDec(offset2, nDec)
            output2 += '$'+val+'\pm'+err+'$ & '
        else:
            output2 += ' & '

        val = ResultMCMC['tDay']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['tNight']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output2 += '$'+val+'\pm'+err+'$ & '



        if int(np.round(BICs[i]-bestBIC, 0)) <= bicThresh:
            output1 += '\\textbf{'
            output2 += '\\textbf{'
        output1 += str(int(np.round(BICs[i]-bestBIC, 0)))
        output2 += str(int(np.round(BICs[i]-bestBIC, 0)))
        if int(np.round(BICs[i]-bestBIC, 0)) <= bicThresh:
            output1 += '}'
            output2 += '}'

        output1 += ' \\\\ % \n'
        output2 += ' \\\\ % \n'


        ######################## MAKE TABLE #3 and 4 ########################

        if BICs[i] != bestBIC:
            continue

        output3 += planet+' '+channel+' & '
        output4 += planet+' '+channel+' & '

        if 'poly' in mode.lower():
            ind = mode.lower().find('poly')
            output3 += mode[ind:ind+5]
            output4 += mode[ind:ind+5]
        elif 'bliss' in mode.lower():
            output3 += 'BLISS'
            output4 += 'BLISS'
        elif 'pld' in mode.lower():
            output3 += 'PLD'
            output4 += 'PLD'
        elif 'gp' in mode.lower():
            output3 += 'GP'
            output4 += 'GP'

        if 'tslope' in mode.lower():
            output3 += '*$f$(t)'
            output4 += '*$f$(t)'

        if 'hside' in mode.lower():
            output3 += '*Step'
            output4 += '*Step'

        if 'psfw' in mode.lower():
            output3 += ', PSFW'
            output4 += ', PSFW'
            
        output3 += ', '
        output4 += ', '

        ind = mode.lower().find('_v')
        vorder = mode[ind+1:].split('_')[0][1:]

        if vorder == '1':
            output3 += '1st Order'
            output4 += '1st Order'
        elif vorder == '2':
            output3 += '2nd Order'
            output4 += '2nd Order'

        if 'ellipse' in mode.lower() and 'offset' in mode.lower():
            output3 += ' + Rotated Ellipse & '
            output4 += ' + Rotated Ellipse & '
        elif 'ellipse' in mode.lower():
            output3 += ' + Ellipse & '
            output4 += ' + Ellipse & '
        else:
            output3 += ' & '
            output4 += ' & '

        val = ResultMCMC['A']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['B']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'\pm'+err+'$ & '

        if vorder == '1':
            output3 += '. & . & '
        else:
            val = ResultMCMC['C']
            if twoLimits:
                nDec1, err1 = roundToSigFigs(val[1])
                nDec2, err2 = roundToSigFigs(val[2])
                nDec = np.max([nDec1, nDec2])
                val = roundToDec(val[0], nDec)
                output3 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
            else:
                nDec, err = roundToSigFigs(np.mean(val[1:]))
                val = roundToDec(val[0], nDec)
                output3 += '$'+val+'\pm'+err+'$ & '

            val = ResultMCMC['D']
            if twoLimits:
                nDec1, err1 = roundToSigFigs(val[1])
                nDec2, err2 = roundToSigFigs(val[2])
                nDec = np.max([nDec1, nDec2])
                val = roundToDec(val[0], nDec)
                output3 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
            else:
                nDec, err = roundToSigFigs(np.mean(val[1:]))
                val = roundToDec(val[0], nDec)
                output3 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['fp']*1e6
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output3 += '$'+val+'\pm'+err+'$ & '






        val = ResultMCMC['rp']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['offset']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(-val[0], nDec)
            output4 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(-val[0], nDec)
            output4 += '$'+val+'\pm'+err+'$ & '


        if 'v2' in mode.lower():
            Cs =  ResultMCMC['C'][0]
            Cs_err = np.mean(ResultMCMC['C'][1:])
            Ds =  ResultMCMC['D'][0]
            Ds_err = np.mean(ResultMCMC['D'][1:])
            offset2 = 180-np.arctan2(Ds, Cs)*180/np.pi/2
            offset2_err = 1/(1+(Ds/Cs)**2)*np.sqrt((Ds_err/Cs)**2+(Ds/Cs**2*Cs_err)**2)*180/np.pi/2
            nDec, err = roundToSigFigs(offset2_err)
            val = roundToDec(offset2, nDec)
            output4 += '$'+val+'\pm'+err+'$ & '
        else:
            output4 += ' & '

        val = ResultMCMC['tDay']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'\pm'+err+'$ & '

        val = ResultMCMC['tNight']
        if twoLimits:
            nDec1, err1 = roundToSigFigs(val[1])
            nDec2, err2 = roundToSigFigs(val[2])
            nDec = np.max([nDec1, nDec2])
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'^{+'+err1+'}_{-'+err2+'}$ & '
        else:
            nDec, err = roundToSigFigs(np.mean(val[1:]))
            val = roundToDec(val[0], nDec)
            output4 += '$'+val+'\pm'+err+'$ & '


        output3 += str(np.round(ResultMCMC['logLB'][0]/nData, 2))
        output4 += str(np.round(ResultMCMC['logLB'][0]/nData, 2))

        output3 += ' \\\\ % \n'
        output4 += ' \\\\ % \n'

    output1 += '\n\n\n'
    output2 += '\n\n\n'
        

print(output1)

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

print(output2)

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

print(output3)

print('\n%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n\n')

print(output4)

