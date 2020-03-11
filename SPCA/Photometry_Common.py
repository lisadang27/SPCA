import os, glob
import numpy as np
from astropy.stats import sigma_clip

def get_fnames(directory, AOR_snip, ch):
    """Find paths to all the fits files.

    Args:
        directory (string): Path to the directory containing all the Spitzer data.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        ch (string): Channel used for the observation eg. 'ch1' for channel 1.

    Returns:
        tuple: fname, lens (list, list).
            List of paths to all bcd.fits files, number of files for each AOR (needed for adding correction stacks).
    
    """
    
    lst      = os.listdir(directory)
    AOR_list = [k for k in lst if AOR_snip in k] 
    fnames   = []
    lens = []
    for i in range(len(AOR_list)):
        path = directory + '/' + AOR_list[i] + '/' + ch +'/bcd'	
        files = glob.glob(os.path.join(path, '*bcd.fits'))
        fnames.extend(files)
        lens.append(len(files))
    #fnames.sort()
    return fnames, lens

def get_stacks(calDir, dataDir, AOR_snip, ch):
    """Find paths to all the background subtraction correction stacks FITS files.

    Args:
        calDir (string): Path to the directory containing the correction stacks.
        dataDir (string): Path to the directory containing the Spitzer data to be corrected.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        ch (string): Channel used for the observation eg. 'ch1' for channel 1.

    Returns:
        list: List of paths to the relevant correction stacks
    
    """
    
    stacks = np.array(os.listdir(calDir))
    locs = np.array([stacks[i].find('SPITZER_I') for i in range(len(stacks))])
    good = np.where(locs!=-1)[0] #filter out all files that don't fit the correct naming convention for correction stacks
    offset = 11 #legth of the string "SPITZER_I#_"
    keys = np.array([stacks[i][locs[i]+offset:].split('_')[0] for i in good]) #pull out just the key that says what sdark this stack is for

    data_list = os.listdir(dataDir)
    AOR_list = [a for a in data_list if AOR_snip in a]
    calFiles = []
    for i in range(len(AOR_list)):
        path = dataDir + '/' + AOR_list[i] + '/' + ch +'/cal/'
        if not os.path.isdir(path):
            print('Error: Folder \''+path+'\' does not exist, so automatic correction stack selection cannot be performed')
            return []
        fname = glob.glob(path+'*sdark.fits')[0]
        loc = fname.find('SPITZER_I')+offset
        key = fname[loc:].split('_')[0]
        calFiles.append(os.path.join(calDir, stacks[list(good)][np.where(keys == key)[0][0]]))
    return calFiles

def get_time(hdu_list, time, ignoreFrames):
    """Gets the time stamp for each image.

    Args:
        hdu_list (list): content of fits file.
        time (ndarray): Array of existing time stamps.
        ignoreFrames (ndarray): Array of frames to ignore (consistently bad frames).

    Returns:
        ndarray: Updated time stamp array.
    
    """
    
    h, w, l = hdu_list[0].data.shape
    sec2day = 1.0/(3600.0*24.0)
    step    = hdu_list[0].header['FRAMTIME']*sec2day
    t       = np.linspace(hdu_list[0].header['BMJD_OBS'] + step/2, hdu_list[0].header['BMJD_OBS'] + (h-1)*step, h)
    if ignoreFrames != []:
        t = np.delete(t, ignoreFrames, axis=0)
    time.extend(t)
    return time

def sigma_clipping(image_data, filenb = 0 , fname = ['not provided'], tossed = 0, badframetable = None, bounds = (13, 18, 13, 18), sigma=4, maxiters=2):
    """Sigma clips bad pixels and mask entire frame if the sigma clipped pixel is too close to the target.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        filenb (int, optional): Index of current file in the 'fname' list (list of names of files) to keep track of the files that were tossed out. Default is 0.
        fname (list, optional): List of names of files to keep track of the files that were tossed out. 
        tossed (int, optional): Total number of image tossed out. Default is 0 if none provided.
        badframetable (list, optional): List of file names and frame number of images tossed out from 'fname'.
        bounds (tuple, optional): Bounds of box around the target. Default is (13, 18, 13, 18).

    Returns:
        tuple: sigma_clipped_data (3D array) - Data cube of sigma clipped images (2D arrays of pixel values).
            tossed (int) - Updated total number of image tossed out.
            badframetable (list) - Updated list of file names and frame number of images tossed out from 'fname'.
    
    """
    
    if badframetable is None:
        badframetable = []
    
    lbx, ubx, lby, uby = bounds
    h, w, l = image_data.shape
    # mask invalids
    image_data2 = np.ma.masked_invalid(image_data)
    # make mask to mask entire bad frame
    x = np.ones((w, l))
    mask = np.ma.make_mask(x)
    try:
        sig_clipped_data = sigma_clip(image_data2, sigma=sigma, maxiters=maxiters, 
                                      cenfunc=np.ma.median, axis = 0)
    except TypeError:
        sig_clipped_data = sigma_clip(image_data2, sigma=sigma, iters=maxiters, 
                                      cenfunc=np.ma.median, axis = 0)
    for i in range (h):
        if np.ma.is_masked(sig_clipped_data[i, lbx:ubx, lby:uby]):
            sig_clipped_data[i,:,:] = np.ma.masked_array(sig_clipped_data[i,:,:], mask = mask)
            badframetable.append([i,filenb,fname])
            tossed += 1
    return sig_clipped_data, tossed, badframetable

def bgsubtract(img_data, bg_flux=None, bg_err=None, bounds=(11, 19, 11, 19)):
    """Measure the background level and subtracts the background from each frame.

    Args:
        img_data (ndarray): Data cube of images (2D arrays of pixel values).
        bg_flux (ndarray, optional): Array of background measurements for previous images. Default is None.
        bg_err (ndarray, optional): Array of uncertainties on background measurements for previous images. Default is None.
        bounds (tuple, optional): Bounds of box around the target. Default is (11, 19, 11, 19).

    Returns:
        tuple: bgsub_data (3D array) Data cube of background subtracted images.
            bg_flux (1D array)  Updated array of background flux measurements for previous images.
            bg_err (1D array) Updated array of uncertainties on background measurements for previous images.
    
    """
    
    if bg_flux is None:
        bg_flux = []
    if bg_err is None:
        bg_err = []
    
    lbx, ubx, lby, uby = bounds
    image_data = np.ma.copy(img_data)
    h, w, l = image_data.shape
    x = np.zeros(image_data.shape)
    x[:, lbx:ubx,lby:uby] = 1
    mask   = np.ma.make_mask(x)
    masked = np.ma.masked_array(image_data, mask = mask)
    masked = np.reshape(masked, (h, w*l))
    bg_med = np.reshape(np.ma.median(masked, axis=1), (h, 1, 1))
    bgsub_data = image_data - bg_med
    bgsub_data = np.ma.masked_invalid(bgsub_data)
    bg_flux.extend(bg_med.ravel())
    bg_err.extend(np.ma.std(masked, axis=1))
    return bgsub_data, bg_flux, bg_err

def binning_data(data, size):
    """Median bin an array.

    Args:
        data (1D array): Array of data to be binned.
        size (int): Size of bins.

    Returns:
        tuple: binned_data (1D array) Array of binned data.
            binned_data_std (1D array) Array of standard deviation for each entry in binned_data.
    
    """
    
    data = np.ma.masked_invalid(data)
    reshaped_data   = data.reshape(int(len(data)/size), size)
    binned_data     = np.ma.median(reshaped_data, axis=1)
    binned_data_std = np.std(reshaped_data, axis=1)
    return binned_data, binned_data_std
