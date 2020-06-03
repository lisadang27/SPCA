import os, glob
import numpy as np
from astropy.stats import sigma_clip
import scipy.interpolate
from scipy.stats import binned_statistic

def create_folder(fullname, auto=False, overwrite=False):
    """Create a folder unless it exists.

    Args:
        fullname (string): Full path to the folder to be created.
        auto (bool, optional): If the folder already exists, should the folder just be skipped (True)
            or should the user be asked whether they want to overwrite the folder or change the folder name (False, Default).
        overwrite (bool, optional): Whether you want to overwrite the folder if it already exists

    Returns:
        string: The final name used for the folder.

    """
    
    solved = 'no'
    while(solved == 'no'):
        if not os.path.exists(fullname):
            # Folder doesn't exist yet and can be safely written to
            os.makedirs(fullname)
            solved = 'yes'
        elif len(os.listdir(fullname))==0:
            # Folder exists but is empty and can be safely written to
            solved = 'yes'
        else:
            if overwrite:
                solved = 'yes'
            elif auto:
                fullname = None
                solved = 'yes'
            else:
                folder = fullname.split('/')[-1]
                print('Warning:', folder, 'already exists! Are you sure you want to overwrite this folder? (y/n)')
                answer = input()
                if (answer=='y'):
                    solved = 'yes'
                else:
                    print('What would you like the new folder name to be?')
                    folder = input()
                    fullname = '/'.join(fullname.split('/')[0:-1])+'/'+folder
    return fullname

def get_fnames(directory, AOR_snip):
    """Find paths to all the fits files.

    Args:
        directory (string): Path to the directory containing all the Spitzer data.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        ch (string): Channel used for the observation eg. 'ch1' for channel 1.

    Returns:
        tuple: fname, lens (list, list).
            List of paths to all bcd.fits files, number of files for each AOR (needed for adding correction stacks).
    
    """
    
    while directory[-1]=='/':
        directory=directory[:-1]
    ch = directory.split('/')[-1]
    
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

def get_stacks(calDir, dataDir, AOR_snip):
    """Find paths to all the background subtraction correction stacks FITS files.

    Args:
        calDir (string): Path to the directory containing the correction stacks.
        dataDir (string): Path to the directory containing the Spitzer data to be corrected.
        AOR_snip (string): Common first characters of data directory eg. 'r579'.
        ch (string): Channel used for the observation eg. 'ch1' for channel 1.

    Returns:
        list: List of paths to the relevant correction stacks
    
    """
    
    while dataDir[-1]=='/':
        dataDir=dataDir[:-1]
    ch = dataDir.split('/')[-1]
    
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
    return np.array(calFiles)

def get_time(hdu_list, ignoreFrames):
    """Gets the time stamp for each image.

    Args:
        hdu_list (list): content of fits file.
        ignoreFrames (ndarray): Array of frames to ignore (consistently bad frames).

    Returns:
        ndarray: Updated time stamp array.
    
    """
    
    h, w, l = hdu_list[0].data.shape
    sec2day = 1.0/(3600.0*24.0)
    step    = hdu_list[0].header['FRAMTIME']*sec2day
    t       = np.linspace(hdu_list[0].header['BMJD_OBS'] + step/2, hdu_list[0].header['BMJD_OBS'] + (h-1)*step, h)
    if ignoreFrames != []:
        t[ignoreFrames] = np.nan
    return t

def oversampling(image_data, a = 2):
    """First, substitutes all invalid/sigma-clipped pixels by interpolating the value, then linearly oversamples the image.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        a (int, optional):  Sampling factor, e.g. if a = 2, there will be twice as much data points in the x and y axis.
            Default is 2. (Do not recommend larger than 2)

    Returns:
        ndarray: Data cube of oversampled images (2D arrays of pixel values).
    
    """
    
    l, h, w = image_data.shape
    gridy, gridx = np.mgrid[0:h:1/a, 0:w:1/a]
    image_over = np.ones((l, h*a, w*a))*np.nan
    for i in range(l):
        image_masked = np.ma.masked_invalid(image_data[i,:,:])
        if np.all(image_masked.mask):
            # Data will already be masked as nan is its default value
            continue
        points       = np.where(image_masked.mask == False)
        image_compre = np.ma.compressed(image_masked)
        image_over[i,:,:] = scipy.interpolate.griddata(points, image_compre,
                                                       (gridx, gridy), 
                                                       method = 'linear')
        
    # Mask any bad values
    image_masked = np.ma.masked_invalid(image_over)
    
    # conserve flux
    return image_masked/(a**2)

def sigma_clipping(image_stack, bounds = (13, 18, 13, 18), sigma=5, maxiters=2):
    """Sigma clips bad pixels and mask entire frame if the sigma clipped pixel is too close to the target.

    Args:
        image_stack (3D array): Data cube of images (2D arrays of pixel values).
        bounds (tuple, optional): Bounds of box around the target. Default is (13, 18, 13, 18).
        sigma (float, optional): How many sigma should something differ by to be clipped. Default is 5 which shouldn't trim any real data for Ndata=64*1000.
        maxiters (int, optional): How many iterations of sigma clipping should be done.

    Returns:
        3D array: sigma_clipped_data - Data cube of sigma clipped images (2D arrays of pixel values).
    
    """
    
    lbx, ubx, lby, uby = bounds
    h, w, l = image_stack.shape
    
    # mask invalids
    image_stack2 = np.ma.masked_invalid(image_stack)
    
    # make mask to mask entire bad frame
    x = np.ones((w, l))
    mask = np.ma.make_mask(x)
    
    try:
        sig_clipped_stack = sigma_clip(image_stack2, sigma=sigma,
                                       maxiters=maxiters, 
                                       cenfunc=np.ma.median, axis = 0)
    except TypeError:
        sig_clipped_stack = sigma_clip(image_stack2, sigma=sigma, iters=maxiters, 
                                       cenfunc=np.ma.median, axis = 0)
    for i in range(h):
        # If any pixels near the target star are bad, mask the entire frame
        if np.ma.is_masked(sig_clipped_stack[i, lbx:ubx, lby:uby]):
            sig_clipped_stack[i,:,:] = np.ma.masked_array(sig_clipped_stack[i,:,:],
                                                          mask = mask)
    return sig_clipped_stack

def bgsubtract(img_data, bounds=(11, 19, 11, 19)):
    """Measure the background level and subtracts the background from each frame.

    Args:
        img_data (ndarray): Data cube of images (2D arrays of pixel values).
        bounds (tuple, optional): Bounds of box around the target. Default is (11, 19, 11, 19).

    Returns:
        tuple: bgsub_data (3D array) Data cube of background subtracted images.
            bg_flux (1D array)  Updated array of background flux measurements for previous images.
            bg_err (1D array) Updated array of uncertainties on background measurements for previous images.
    
    """
    
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
    
    bg_flux = bg_med.ravel()
    bg_err = np.ma.std(masked, axis=1)
    return bgsub_data, bg_flux, bg_err

def noisepixparam(image_data, bounds=(13, 18, 13, 18)):
    """Compute the noise pixel parameter.

    Args:
        image_data (ndarray): FITS images stack.

    Returns:
        list: The noise pixel parameter for each image in the stack.

    """
    
    lbx, ubx, lby, uby = bounds
    
    #To find noise pixel parameter for each frame. For eqn, refer Knutson et al. 2012
    numer = np.ma.sum(image_data[:, lbx:ubx, lby:uby], axis=(1,2))**2
    denom = np.ma.sum(image_data[:, lbx:ubx, lby:uby]**2, axis=(1,2))
    npp = numer/denom
    
    return npp

def bin_array(data, size):
    """Median bin an array.

    Args:
        data (1D array): Array of data to be binned.
        size (int): Number of data points in each bin.

    Returns:
        tuple: binned_data (1D array) Array of binned data.
            binned_data_std (1D array) Array of standard deviation for each entry in binned_data.
    
    """
    
    data = np.ma.masked_invalid(data)
    x = np.arange(data.shape[0])
    bins = size*np.arange(np.ceil(len(x)/size)+1)-0.5
    
    binned_data = binned_statistic(x, data, statistic=np.nanmedian, bins=bins)[0]
    
    binned_data_std = binned_statistic(x, data, statistic=np.nanstd, bins=bins)[0]
    
    return binned_data, binned_data_std