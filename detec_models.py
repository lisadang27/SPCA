import numpy as np

import astro_models

def detec_model_poly(detec_inputs, c1, c2, c3, c4, c5, c6, c7=0, c8=0, c9=0, c10=0, c11=0, 
                     c12=0, c13=0, c14=0, c15=0, c16=0, c17=0, c18=0, c19=0, c20=0, c21=0):
    
    x, y, mode = detec_inputs
    
    if   'poly2' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2))
        detec = np.array([c1, c2, c3, c4, c5, c6])
    elif 'poly3' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,])
    elif 'poly4' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3,
                        x**4, x**3*y, x**2*y**2, x**1*y**3,   y**4))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15,])
    elif 'poly5' in mode.lower():
        pos = np.vstack((np.ones_like(x),
                        x   ,      y,
                        x**2, x   *y,      y**2,
                        x**3, x**2*y,    x*y**2,      y**3,
                        x**4, x**3*y, x**2*y**2, x**1*y**3,   y**4,
                        x**5, x**4*y, x**3*y**2, x**2*y**3, x*y**4, y**5))
        detec = np.array([c1,  c2,  c3,  c4,  c5,  c6,
                          c7,  c8,  c9,  c10,
                          c11, c12, c13, c14, c15,
                          c16, c17, c18, c19, c20, c21])

    return np.dot(detec[np.newaxis,:], pos).reshape(-1)

def detec_model_PSFW(input_data, d1=1, d2=0, d3=0):
    px, py = input_data
    pw     = np.vstack((np.ones_like(px), px, py))
    syst   = np.array([d1, d2, d3])
    return np.dot(syst[np.newaxis,:], pw).reshape(-1)

def hside(time, s1, s2):
    x = time - s2
    return s1*np.heaviside(x, 0.0) + 1

def tslope(time, m1):
    return 1+(time-time[0])*m1

def detec_model_bliss(signal_input, astroModel): 
    '''
    Input:
        sensMap = sensitivity values at the knots
        nData   = number of measurements
        LL_dist = distance to lower-left  knot
        LR_dist = distance to lower-right knot
        UL_dist = distance to upper-left  knot
        UR_dist = distance to upper-right knot
        BLS     = BLISS points
        NNI     = NNI points
    '''
    
    (flux, time, psfxw, psfyw, nBin, nData, knotNdata, low_bnd_x, up_bnd_x, low_bnd_y, up_bnd_y,
     LL_dist, LR_dist, UL_dist, UR_dist, delta_xo, delta_yo, knot_nrst_x, knot_nrst_y,
     knot_nrst_lin, BLS, NNI, knots_x_mesh, knots_y_mesh, tmask_good_knotNdata, mode) = signal_input
    
    sensMap = np.bincount(knot_nrst_lin, weights=flux/astroModel, minlength=(nBin*nBin)).reshape((nBin,nBin))
    sensMap /= knotNdata
    
    detec = np.empty(nData)
    
    # weight knots values by distance to knots
    LL = sensMap[low_bnd_y, low_bnd_x]*LL_dist
    LR = sensMap[low_bnd_y, up_bnd_x]*LR_dist
    UL = sensMap[up_bnd_y, low_bnd_x]*UL_dist
    UR = sensMap[up_bnd_y, up_bnd_x]*UR_dist
    
    # BLISS points
    detec[BLS] = (LL[BLS] + LR[BLS] + UL[BLS] + UR[BLS])/(delta_xo*delta_yo)
    
    # Nearest Neighbor points
    detec[NNI] = sensMap[knot_nrst_y[NNI],knot_nrst_x[NNI]]  
    '''
    Output:
        detec  = BLISS/NNI sensitivity values at each knots
    '''
    return detec

def detec_model_PLD(input_data, p1_1, p2_1, p3_1, p4_1, p5_1, p6_1, p7_1, p8_1, p9_1, p10_1, 
                p11_1, p12_1, p13_1, p14_1, p15_1, p16_1, p17_1, p18_1, p19_1, p20_1, 
                p21_1, p22_1, p23_1, p24_1, p25_1,p1_2, p2_2, p3_2, p4_2, p5_2, p6_2, 
                p7_2, p8_2, p9_2, p10_2, p11_2, p12_2, p13_2, p14_2, p15_2, p16_2, p17_2, 
                p18_2, p19_2, p20_2, p21_2, p22_2, p23_2, p24_2, p25_2):
    
    Pgroup, mode = input_data 
    
    if '3x3' in mode:
        if 'PLD1' in mode.lower():
            detec = np.asarray(p1*P1+ p2_1*P2+ p3_1*P3+ p4_1*P4+ p5_1*P5+ p6_1*P6+ p7_1*P7+ p8_1*P8+ p9_1*P9)
        elif 'PLD2' in mode.lower():
            detec = np.asarray(p1*P1+ p2_1*P2+ p3_1*P3+ p4_1*P4+ p5_1*P5+ p6_1*P6+ p7_1*P7+ p8_1*P8+ p9_1*P9+ 
                               p2_2*P2+ p3_2*P3+ p4_2*P4+ p5_2*P5+ p6_2*P6+ p7_2*P7+ p8_2*P8+ p9_2*P9)
            
    elif '5x5' in mode:
        if 'PLD1' in mode.lower():
            detec = np.asarray(p1*P1+ p2_1*P2+ p3_1*P3+ p4_1*P4+ p5_1*P5+ p6_1*P6+ p7_1*P7+ p8_1*P8+ p9_1*P9+ 
                               p10_1*P10+ p11_1*P11+ p12_1*P12+ p13_1*P13+ p14_1*P14+ p15_1*P15+ p16_1*P16+ 
                               p17_1*P17+ p18_1*P18+ p19_1*P19+ p20_1*P20+ p21_1*P21+ p22_1*P22+ p23_1*P23+ 
                               p24_1*P24+ p25_1*P25)
        elif 'PLD2' in mode.lower():
            detec = np.asarray(p1*P1+ p2_1*P2+ p3_1*P3+ p4_1*P4+ p5_1*P5+ p6_1*P6+ p7_1*P7+ p8_1*P8+ p9_1*P9+ p10_1*P10+
                               p11_1*P11+ p12_1*P12+ p13_1*P13+ p14_1*P14+ p15_1*P15+ p16_1*P16+ p17_1*P17+ p18_1*P18+ 
                               p19_1*P19+ p20_1*P20+ p21_1*P21+ p22_1*P22+ p23_1*P23+ p24_1*P24+ p25_1*P25+ p2_2*P2+ 
                               p3_2*P3+ p4_2*P4+ p5_2*P5+ p6_2*P6+ p7_2*P7+ p8_2*P8+ p9_2*P9+ p10_2*P10+ p11_2*P11+ 
                               p12_2*P12+ p13_2*P13+ p14_2*P14+ p15_2*P15+ p16_2*P16+ p17_2*P17+ p18_2*P18+ p19_2*P19+ 
                               p20_2*P20+ p21_2*P21+ p22_2*P22+ p23_2*P23+ p24_2*P24+ p25_2*P25)
    
    return detec

######################################################################################
#THIS IS THE MAIN SIGNAL FUNCTION WHICH BRANCHES OUT TO THE CORRECT SIGNAL FUNCTION
def signal(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
                d1, d2, d3, s1, s2, m1, sigF):

    mode = signal_input[-1]
    if 'poly' in mode.lower():
        return signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                           c1,  c2,  c3,  c4,  c5,  c6,  c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17,
                           c18, c19, c20, c21, d1,  d2,  d3,  s1,  s2, m1, sigF)
    else:
        return signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                            d1, d2, d3, s1, s2, m1, sigF)
######################################################################################

def signal_poly(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off,
                c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, 
                d1, d2, d3, s1, s2, m1, sigF):
    
    #flux, time, xdata, ydata, psfwx, psfwy, mode = signal_input
    #psfw variables won't be there if you're not fitting against psf width
    
    time = signal_input[1]
    xdata = signal_input[2]
    ydata = signal_input[3]
    mode = signal_input[-1]
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[4:6]
        psfsys = detec_model_PSFW(psfwidths, d1, d2, d3)
    else:
        psfsys = 1
    
    if 'hside' in mode.lower():
        hstep  = hside(time, s1, s2)
    else:
        hstep = 1
        
    if 'tslope' in mode.lower():
        tcurve = tslope(time, m1)
    else:
        tcurve = 1
    
    astr   = astro_models.ideal_lightcurve(time, mode, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, 
                                           A, B, C, D, r2, r2off)
    detec  = detec_model_poly((xdata, ydata, mode), c1,  c2,  c3,  c4,  c5,  c6, c7,  c8,  c9, c10, 
                                           c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21)
    
    return astr*detec*psfsys*hstep*tcurve

def signal_bliss(signal_input, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off, 
                 d1, d2, d3, s1, s2, m1, sigF):
    
    time = signal_input[1]
    mode = signal_input[-1]
    
    if 'psfw' in mode.lower():
        psfwidths = signal_input[2:4]
        psfsys = detec_model_PSFW(psfwidths, d1, d2, d3)
    else:
        psfsys = 1
    
    if 'hside' in mode.lower():
        hstep  = hside(time, s1, s2)
    else:
        hstep = 1
    
    if 'tslope' in mode.lower():
        tcurve = tslope(time, m1)
    else:
        tcurve = 1
    
    astroModel = astro_models.ideal_lightcurve(time, mode, t0, per, rp, a, inc, ecosw, esinw, q1, q2,
                                               fp, A, B, C, D, r2, r2off)
    
    detecModel = detec_model_bliss(signal_input, astroModel)
    
    return astroModel*detecModel*psfsys*hstep*tcurve