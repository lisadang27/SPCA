import numpy as np
import batman


def transit_model(time, t0, per, rp, a, inc, ecc, w, u1, u2):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.rp = rp                                       #planet radius (in units of stellar radii)
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients

    m = batman.TransitModel(params, time)                #initializes model
    flux = m.light_curve(params)
    t_secondary = m.get_t_secondary(params)
    anom       = m.get_true_anomaly()
    return flux, t_secondary, anom

def transit_model_ellipse(time, t0, per, rp, r2, r2off, a, inc, ecc, w, u1, u2):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients
    
    flux = np.array([])
    rp_eff = np.sqrt(area(time, t_sec, per, rp, inc_raw, r2, r2off)*rp**2)
    for i in range(len(time)):
        params.rp = rp_eff[i]                            #planet radius (in units of stellar radii)
        m = batman.TransitModel(params, time)
        flux = np.append(flux, m.light_curve(params)[i])
    
    t_secondary = m.get_t_secondary(params)
    anom       = m.get_true_anomaly()
    return flux, t_secondary, anom

def eclipse(time, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.rp = rp                                       #planet radius (in units of stellar radii)
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients
    params.fp = fp                                       #planet/star brightnes
    params.t_secondary = t_sec
    
    m = batman.TransitModel(params, time, transittype="secondary")  #initializes model
    flux = m.light_curve(params)
    return flux

def eclipse_ellipse(time, t0, per, rp, r2, r2off, a, inc, ecc, w, u1, u2, fp, t_sec):
    params = batman.TransitParams()                      #object to store transit parameters
    params.t0 = t0                                       #time of inferior conjunction
    params.per = per                                     #orbital period
    params.a = a                                         #semi-major axis (in units of stellar radii)
    params.inc = inc                                     #orbital inclination (in degrees)
    params.ecc = ecc                                     #eccentricity
    params.w = w                                         #longitude of periastron (in degrees)
    params.limb_dark = "quadratic"                       #limb darkening model
    params.u = [u1, u2]                                  #limb darkening coefficients
    params.fp = fp                                       #planet/star brightnes
    params.t_secondary = t_sec
    
    flux = np.array([])
    rp_eff = np.sqrt(area(time, t_sec, per, rp, inc_raw, r2, r2off)*rp**2)
    for i in range(len(time)):
        params.rp = rp_eff[i]                            #planet radius (in units of stellar radii)
        m = batman.TransitModel(params, time, transittype="secondary")
        flux = np.append(flux, m.light_curve(params)[i])
    
    return flux

def area_noOffset(time, t_sec, per, rp, inc_raw, r2):
    t = time - t_sec
    w = 2*np.pi/per
    phi = (w*t-np.pi)%(2*np.pi)#+np.pi/2
    inc = inc_raw*np.pi/180 #converting inclination to radians for numpy
    return np.pi*np.sqrt(rp**2*np.sin(inc)**2*(r2**2*np.sin(phi)**2 + rp**2*np.cos(phi)**2) + rp**2*r2**2*np.cos(inc)**2)/(np.pi*rp**2)


def area(time, t_sec, per, rp, inc_raw, r2, r2off):
    t = time - t_sec
    w = 2*np.pi/per
    phi = (w*t-np.pi)%(2*np.pi)#+np.pi/2
    phi -= r2off*np.pi/180
    inc = inc_raw*np.pi/180 #converting inclination to radians for numpy
    return np.pi*np.sqrt(rp**2*np.sin(inc)**2*(r2**2*np.sin(phi)**2 + rp**2*np.cos(phi)**2) + rp**2*r2**2*np.cos(inc)**2)/(np.pi*rp**2)

def area_old(time, t_sec, per, rp, inc_raw, r2):
    t = time - t_sec
    w = 2*np.pi/per
    phi = (w*t-np.pi)%(2*np.pi)
    inc = inc_raw*np.pi/180 #converting inclination to radians for numpy
    #R = np.array([[np.sin(inc)*np.cos(phi),   np.sin(phi),  np.cos(inc)*np.cos(phi)],
    #              [-np.sin(inc)*np.sin(phi),  np.cos(phi),  -np.cos(inc)*np.sin(phi)],
    #              [-np.cos(inc),              0,            np.sin(inc)]])
    R = np.zeros((len(phi),3,3))
    R[:,0,0] = np.sin(inc)*np.cos(phi)
    R[:,0,1] = np.sin(phi)
    R[:,0,2] = np.cos(inc)*np.cos(phi)
    R[:,1,0] = -np.sin(inc)*np.sin(phi)
    R[:,1,1] = np.cos(phi)
    R[:,1,2] = -np.cos(inc)*np.sin(phi)
    R[:,2,0] = -np.cos(inc)
    R[:,2,2] = np.sin(inc)
    a_mat = np.array([[1/r2**2,  0,        0],
                      [0,        1/rp**2,  0],
                      [0,        0,        1/rp**2]])[np.newaxis,:,:]
    #[[a, d, f],
    # [_, b, e],
    # [_, _, c]] 
    arr = np.matmul(R.transpose(0,2,1), np.matmul(a_mat,R))
    a = arr[:,0,0]
    b = arr[:,1,1]
    c = arr[:,2,2]
    d = arr[:,0,1]
    e = arr[:,1,2]
    f = arr[:,0,2]
    return np.pi/np.sqrt(3*b*f**2/a + 3*c*d**2/a + -6*d*e*f/a + b*c - e**2)/(np.pi*rp**2)

def phase_variation(time, t_sec, per, anom, w, A, B, C, D, mode):
    if 'eccent' in mode:
        phi  = anom + w + np.pi/2
    else:
        t    = time - t_sec
        freq = 2*np.pi/per
        phi  = (freq*t)
    if 'v2' in mode:
        phase = 1 + A*(np.cos(phi)-1) + B*np.sin(phi) + C*(np.cos(2*phi)-1) + D*np.sin(2*phi)
    else:
        phase = 1 + A*(np.cos(phi)-1) + B*np.sin(phi)
    return phase

def fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C, D, r2, r2off, mode):
    phase = phase_variation(time, t_sec, per, anom, w, A, B, C, D, mode)
    eclip = eclipse_ellipse(time, t0, per, rp, r2, r2off, a, inc, ecc, w, u1, u2, fp, t_sec)
    if 'ellipse' in mode:
        return phase*(eclip - 1)*area(time, t_sec, per, rp, inc, r2, r2off)
    else:
        return phase*(eclip - 1)

def ideal_lightcurve(time, t0, per, rp, a, inc, ecosw, esinw, q1, q2, fp, A, B, C, D, r2, r2off, mode):
    
    ecc = np.sqrt(ecosw**2 + esinw**2)
    w   = np.arctan2(esinw, ecosw)
    u1  = 2*np.sqrt(q1)*q2
    u2  = np.sqrt(q1)*(1-2*q2)
    # create transit first and use orbital paramater to get time of superior conjunction
    transit, t_sec, anom = transit_model_ellipse(time, t0, per, rp, r2, r2off, a, inc, ecc, w, u1, u2)
    
    #ugly way of doing this as might pick up detector parameters, but thats alright - faster this way and still safe
    fplanet = fplanet_model(time, anom, t0, per, rp, a, inc, ecc, w, u1, u2, fp, t_sec, A, B, C, D, r2, r2off, mode)
    
    # add both light curves
    f_total = transit + fplanet
    return f_total

def check_phase(A, B, C, D, mode, phis):
    if 'v2' in mode:
        phase = 1 + A*(np.cos(phis)-1) + B*np.sin(phis) + C*(np.cos(2*phis)-1) + D*np.sin(2*phis)
    else: 
        phase = 1 + A*(np.cos(phis)-1) + B*np.sin(phis)
    return np.any(phase < 0)