import os, glob
import numpy as np
from multiprocessing import Pool
from functools import partial

from astropy.stats import sigma_clip
from astropy.convolution import convolve, Box1DKernel
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from astropy.coordinates import SkyCoord
from astropy.io import fits

import scipy.interpolate
from scipy.stats import binned_statistic

import warnings
warnings.filterwarnings('ignore')

image_stack = np.zeros((0,32,32))

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
    AOR_list = [folder for folder in lst if AOR_snip==folder[:len(AOR_snip)]]
    fnames   = []
    lens = []
    for i in range(len(AOR_list)):
        path = directory + '/' + AOR_list[i] + '/' + ch +'/bcd'
        files = np.sort(glob.glob(os.path.join(path, '*bcd.fits')))
        if len(files)!=0:
            fnames.extend(files)
            lens.append(len(files))
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
    
    lst      = os.listdir(dataDir)
    AOR_list = [folder for folder in lst if AOR_snip==folder[:len(AOR_snip)]]
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

def get_time(header, ignoreFrames):
    """Gets the time stamp for each image.

    Args:
        header (astropy.io.fits.header.Header): Header of fits file.
        ignoreFrames (ndarray): Array of frames to ignore (consistently bad frames).

    Returns:
        ndarray: Updated time stamp array.
    
    """
    
    if header['NAXIS']==2:
        h = 1
    else:
        h = header['NAXIS3']
    sec2day = 1.0/(3600.0*24.0)
    step    = header['FRAMTIME']*sec2day
    t       = np.linspace(header['BMJD_OBS'] + step/2, header['BMJD_OBS'] + (h-1)*step, h, endpoint=True)
    t = np.ma.masked_invalid(t)
    if ignoreFrames != []:
        t[ignoreFrames].mask = True
    return t

def oversampling(image_data, scale=2):
    """First, substitutes all invalid/sigma-clipped pixels by interpolating the value, then linearly oversamples the image.

    Args:
        image_data (ndarray): Data cube of images (2D arrays of pixel values).
        scale (int, optional):  Sampling factor, e.g. if scale = 2, there will be twice as many data points in the x
            and y axis. Default is 2. (Do not recommend larger than 2)

    Returns:
        ndarray: Data cube of oversampled images (2D arrays of pixel values).
    
    """
    
    l, h, w = image_data.shape
    gridy, gridx = np.mgrid[0:h:1/a, 0:w:1/a]
    image_over = np.zeros((l, h*a, w*a))
    for i in range(l):
        image_masked = np.ma.masked_invalid(image_data[i,:,:])
        if np.all(image_masked.mask):
            image_over[i,:,:].mask = True
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

def sigma_clipping(image_stack, bounds = (13, 18, 13, 18), sigma=5, maxiters=3):
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
    
    # Using a nested for-loop to significantly reduce RAM
    for i in range(image_stack.shape[1]):
        for j in range(image_stack.shape[2]):
            try:
                image_stack[:,i,j] = sigma_clip(image_stack[:,i,j].flatten(), sigma=sigma, maxiters=maxiters,
                                                cenfunc=np.ma.median, stdfunc=np.ma.std)
            except TypeError:
                image_stack[:,i,j] = sigma_clip(image_stack[:,i,j].flatten(), sigma=sigma, iters=maxiters,
                                                cenfunc=np.ma.median, stdfunc=np.ma.std)
    
    # If any pixels near the target star are bad, mask the entire frame
    image_stack[np.any(image_stack.mask[:,lbx:ubx,lby:uby], axis=(1,2))] = np.ma.masked
    
    return image_stack

def clip_data(arr, highpassWidth, sigma1=5, sigma2=5, maxiters=3):
    try:
        arr = sigma_clip(arr, sigma=sigma1, maxiters=maxiters, cenfunc=np.ma.median, stdfunc=np.ma.std)
    except TypeError:
        arr = sigma_clip(arr, sigma=sigma1, iters=maxiters, cenfunc=np.ma.median, stdfunc=np.ma.std)
    arr = replace_clipped(arr)
    arr = rolling_clip(arr, highpassWidth, sigma=sigma2, maxiters=maxiters)
    arr = replace_clipped(arr)
    
    return arr

def replace_clipped(arr):
    for i in np.where(np.ma.getmaskarray(arr))[0]:
        inds = np.array([i-2, i-1, i+1, i+2])
        inds = inds[inds<len(arr)]
        
        if not np.all(arr[inds].mask):
            arr[i] = np.ma.median(arr[inds])
        else:
            arr[i] = np.ma.median(arr)
    
    arr.mask = np.ma.nomask
    
    return arr

def rolling_clip(arr, highpassWidth, sigma=5, maxiters=3):
    smooth = highpassflist(arr, highpassWidth)
    smoothed = (arr - smooth)
    try:
        smoothed = sigma_clip(smoothed, sigma=sigma, maxiters=maxiters,
                              cenfunc=np.ma.median, stdfunc=np.ma.std)
    except:
        smoothed = sigma_clip(smoothed, sigma=sigma, iters=maxiters,
                              cenfunc=np.ma.median, stdfunc=np.ma.std)
    arr[np.ma.getmaskarray(smoothed)] = np.ma.masked
    return arr

def bgsubtract(bounds=(11, 19, 11, 19), i=0):
    """Measure the background level and subtracts the background from each frame.

    Args:
        bounds (tuple, optional): Bounds of box around the target. Default is (11, 19, 11, 19).

    Returns:
        tuple: bgsub_data (3D array) Data cube of background subtracted images.
            bg_flux (1D array)  Updated array of background flux measurements for previous images.
            bg_err (1D array) Updated array of uncertainties on background measurements for previous images.
    
    """
    
    # Access global variable
    global image_stack
    
    lbx, ubx, lby, uby = bounds
    
    # Get the initial mask
    image = np.ma.masked_invalid(image_stack[i])
    
    # Mask out the target star
    image[lbx:ubx,lby:uby] = np.ma.masked
    
    # Compute background and error
    bg = np.ma.median(image)
    bg_err = np.ma.std(image)
    
    # Subtract background
    image_stack[i] -= bg
    
    return np.array([bg, bg_err])

def bin_array(data, size):
    """Median bin an array.

    Args:
        data (1D array): Array of data to be binned.
        size (int): Number of data points in each bin.

    Returns:
        tuple: binned_data (1D array) Array of binned data.
            binned_data_std (1D array) Array of standard deviation for each entry in binned_data.
    
    """
    
    data = np.ma.masked_invalid(np.ma.copy(data))
    x = np.arange(data.shape[0])
    bins = size*np.arange(np.ceil(len(x)/size)+1)-0.5
    
    binned_data = binned_statistic(x, data, statistic=np.ma.median, bins=bins)[0]
    
    binned_data_std = binned_statistic(x, data, statistic=np.ma.std, bins=bins)[0]
    
    return binned_data, binned_data_std

def highpassflist(signal, highpassWidth):
    g = Box1DKernel(highpassWidth)
    return convolve(signal, g, boundary='extend')

def prepare_image(savepath, AOR_snip, fnames, lens, stacks=[], ignoreFrames=[],
                  oversamp=False, scale=2, reuse_oversamp=True, saveoversamp=True,
                  addStack=False, stackPath='', maskStars=[], i=0):
    
    # open fits file
    with fits.open(fnames[i]) as hdu_list:
        header = hdu_list[0].header
        time = get_time(header, ignoreFrames)

        if len(hdu_list[0].data.shape)==2:
            # Reshape fullframe data so that it can be used with our routines
            # Getting just the stamp around the sweet spot
            image = np.ma.masked_invalid(hdu_list[0].data[np.newaxis,217:249,9:41])
        else:
            image = hdu_list[0].data
            #ignore any consistently bad frames in datacubes
            image[ignoreFrames] = np.nan
            image = np.ma.masked_invalid(image)
    
    #add background correcting stack if requested
    if addStack:
        j=0 #counter to keep track of which correction stack we're using
        while i > np.sum(lens[:j+1]):
            j+=1 #if we've moved onto a new AOR, increment j
        stackHDU = fits.open(stacks[j])
        image += np.ma.masked_invalid(stackHDU[0].data)
    
    if image.shape[0]!=1:
        # Sigma clipping within datacubes
        image = sigma_clipping(image, sigma=4.)
    
    # convert MJy/str to electron count
    convfact = (header['GAIN']*header['EXPTIME']/header['FLUXCONV'])
    image = convfact*image
    
    # Mask any other stars in the frame to avoid them influencing the background subtraction
    if maskStars != []:
        header['CTYPE3'] = 'Time-SIP' #Just need to add a type so astropy doesn't complain
        w = WCS(header, naxis=[1,2])
        mask = np.ma.getmaskarray(image)
        for st in maskStars:
            coord = SkyCoord(st[0], st[1])
            x,y = np.rint(skycoord_to_pixel(coord, w)).astype(int)
            x = x+np.arange(-1,2)
            y = y+np.arange(-1,2)
            x,y = np.meshgrid(x,y)
            mask[:,x,y] = True
        image = np.ma.masked_array(image, mask=mask)
    
    # oversampling
    if oversamp:
        if reuse_oversamp:
            savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
            if os.path.isfile(savename):
                image = np.load(savename)
            else:
                print('Warning: Oversampled images were not previously saved! Making new ones now...')
                image = np.ma.masked_invalid(oversampling(image, scale))
                if (saveoversamp == True):
                    # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
                    image.dump(savename)
        else:
            image = np.ma.masked_invalid(oversampling(image))
        
        if saveoversamp:
            # THIS CHANGES FROM ONE SET OF DATA TO ANOTHER!!!
            savename = savepath + 'Oversampled/' + fnames[i].split('/')[-1].split('_')[-4] + '.pkl'
            image.dump(savename)
    
    data = image.reshape(-1,np.product(image.shape[1:]))
    data = np.append(data, time[:,np.newaxis], axis=1)
    
    return data

def prepare_images(basepath, planet, channel, AOR_snip, ignoreFrames=[],
                   oversamp=False, scale=2, reuse_oversamp=True, saveoversamp=True,
                   addStack=False, maskStars=[], ncpu=4):
    
    #folder containing properly named correction stacks (will be automatically selected)
    stackPath = basepath+'Calibration/'
    datapath   = basepath+planet+'/data/'+channel
    savepath = basepath+planet+'/analysis/'+channel+'/'
    if addStack:
        savepath += 'addedStack/'
    else:
        savepath += 'addedBlank/'
    
    if maskStars is None:
        maskStars = []
    if ignoreFrames is None:
        ignoreFrames = []
    
    print('\tGetting frames', end='', flush=True)
    if len(ignoreFrames)!=0:
        print(', masking bad frames', end='', flush=True)
    print('... ', end='')
    # get list of filenames and number of files
    fnames, lens = get_fnames(datapath, AOR_snip)
    
    # get path where the aor breaks will be saved
    breakpath = basepath+planet+'/analysis/'+channel+'/aorBreaks.txt'
    # get & write aor breaks
    index  = 0
    breaktimes = []
    for length in lens:
        with fits.open(fnames[index]) as rawImage:
            header = rawImage[0].header
            breaktimes.append(get_time(header, []).flatten()[0])
        index += length
    breaktimes = np.sort(breaktimes)[1:]
    with open(breakpath, 'w') as f:
        f.write(str(breaktimes)[1:-1])    
    
    # if need to add correction stack
    if addStack:
        stacks = get_stacks(stackPath, datapath, AOR_snip)
    else:
        stacks = []
    
    # Load all of the images using multiprocessing to speed things up
    with Pool(ncpu) as pool:
        func = partial(prepare_image, savepath, AOR_snip, fnames, lens, stacks, ignoreFrames,
                       oversamp, scale, reuse_oversamp, saveoversamp, addStack, stackPath, maskStars)
        inds = range(len(fnames))
        results = np.ma.masked_array(pool.map(func, inds)).reshape(-1,int(32**2+1))
    
    # Access global variable
    global image_stack
    
    time = results[:,-1].flatten()
    image_stack = results[:,:-1].reshape(-1, 32, 32)
    # Free up a bit of RAM
    results = None
    
    # Sort data into correct order
    order = np.argsort(time)
    time = time[order]
    image_stack = image_stack[order]
    # Free up a bit of RAM
    order = None
    
    print('Sigma clipping... ', end='', flush=True)
    # sigma clip bad pixels along full time axis
    image_stack = sigma_clipping(image_stack, sigma=5)
    
    print('Subtracting background... ', end='', flush=True)
    # background subtraction is done on global variable
    with Pool(ncpu) as pool:
        func = partial(bgsubtract, (11, 19, 11, 19))
        inds = range(image_stack.shape[0])
        bg, bg_err = np.array(pool.map(func, inds)).T
    
    print('Frames loaded!', flush=True)
    
    return image_stack, bg, bg_err, time
