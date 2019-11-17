import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
from skimage.feature import blob_log
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clipped_stats, SigmaClip
from astropy.visualization import SqrtStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy import wcs
from photutils import datasets, DAOStarFinder, SkyCircularAperture, aperture_photometry, CircularAperture, CircularAnnulus, Background2D, MedianBackground
from photutils.background import Background2D
from photutils.detection import find_peaks, detect_threshold
from photutils.utils import calc_total_error
import config

## python3 HSTphotometry.py --imgfolder=data/W0014+7915/ --RA=00:14:49.96 --DEC=+79:51:16.2
## python3 HSTphotometry.py --imgfolder=data/W0830-6323/ --RA=08:30:19.98 --DEC=-63:23:05.5
## python3 HSTphotometry.py --imgfolder=data/W0830+2837/ --RA=08:30:11.96 --DEC=+28:37:16.0
## python3 HSTphotometry.py --imgfolder=data/W1516+7217/ --RA=15:16:20.40 --DEC=+72:17:45.5
## python3 HSTphotometry.py --imgfolder=data/W1525+6053/ --RA=15:25:29.10 --DEC=+60:53:56.6

## description: python HSTphotometry.py --help

def load_images(imgfolder, ra, dec, instrument='HST_WFC3'):

    # read in image file names
    imgfiles = glob.glob(imgfolder+'*.fits', recursive=True)

    # add header info to dataframe for all images
    df = pd.DataFrame()
    images = {}
    aperture, pixscale, targname = None, None, ''

    for imgfile in tqdm(imgfiles):
        # get header info
        fits_file = fits.open(imgfile)
        hdr = fits_file[0].header
        targname = hdr['TARGNAME']

        # create row
        row = {hdr_var: hdr[hdr_var] for hdr_var in config.HEADER_VARS[instrument]}
        row['RA'] = ra
        row['DEC'] = dec
        c = SkyCoord(ra=row['RA_TARG']*u.deg,dec=row['DEC_TARG']*u.deg)
        row['RA_TARG_HMS'] = c.to_string('hmsdms',sep=':').split(' ')[0]
        row['DEC_TARG_DMS'] = c.to_string('hmsdms',sep=':').split(' ')[1]

        # add vega zero-point to row
        for key, val in config.VEGA_ZEROPTS.items():
            if row['FILTER'] == key:
                row['VEGAZPT'] = val

        # add drz to row
        if 'drz' in imgfile:
            pixscale = hdr[config.PIXSCALE[instrument]]
            aperture = 0.4 / pixscale  #0.4arcsecs / 0.128 arcsecs/pixel

        # append row to dataframe
        df = df.append(pd.Series(row), ignore_index=True)

        # store image
        images[row['FILENAME']] = fits_file[1].data

    # postprocessing
    if aperture is not None:
        df['APERTURE'] = aperture

    if pixscale is not None:
        df['PIXSCALE'] = pixscale

    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore')).sort_values('FILENAME').reset_index().drop('index',1)

    return df, images

def find_centroid(img, x, y):

    x = float(x)
    y = float(y)
    cutout = img[int(y - 25):int(y + 25),int(x - 25):int(x + 25)]

    z = mad_std(cutout[~np.isnan(cutout)])
    cutout[cutout < -20*z] = -20*z
    cutout[cutout > 100*z] = 100*z
    normcutout = 2*(cutout-cutout.min())/(cutout.max()-cutout.min())-1
    #blobs = blob_log(normcutout, max_sigma=5)
    #idx = np.argmin(np.sqrt((blobs[:,0] - len(normcutout) / 2)**2 + (blobs[:,1] - len(normcutout) / 2)**2))
    #xctrcut, yctrcut, rad = blobs[idx]
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*mad_std(normcutout))
    xysources = daofind(normcutout).to_pandas()
    xysources['xctr'] = xysources['xcentroid'] + x - 25
    xysources['yctr'] = xysources['ycentroid'] + y - 25
    print(xysources)

    return xysources, normcutout

def plot_centroids(cutout, filename, x, y, xysources):

    x = float(x)
    y = float(y)
    xctr = xysources['xctr'].astype(float)
    yctr = xysources['yctr'].astype(float)
    shortfile = filename.split('.fits')[0]

    plt.imshow(cutout, cmap='gray_r')
    plt.plot(25, 25, color='b', marker='+', markersize=30, linestyle='None')
    plt.plot(xctr - x + 25, yctr - y + 25, color='r', marker='+', markersize=30, linestyle='None')
    for i in range(len(xysources)):
        plt.annotate(xysources.loc[i,'id'],xy=(xysources.loc[i,'xcentroid'],xysources.loc[i,'ycentroid']),color='r')
    plt.title(shortfile)
    plt.xlim(0,50)
    plt.ylim(0,50)
    plt.legend(['image center','centroid'])
    plt.savefig('centroid_'+str(shortfile)+'.jpg')
    
    print('Look up centroid_'+str(shortfile)+'.jpg')
    plt.clf()


'''
def bkg_est(img, gain):
    
    bkgerror = Background2D(img, box_size=10, edge_method='pad')
    toterr = calc_total_error(img, bkg_error=bkgerror.background, effective_gain=gain)
    
    maderr = mad_std(err,ignore_nan=True)
    stderr = np.nanstd(err)
    
    return maderr,stderr
'''
    
def calculate_sky_background(img,images_info,return_table=False):

    niter = 100
    skymedians = np.zeros(niter)
    #maderr = mad_std(img,ignore_nan=True)
    xlen, ylen = img.shape
    nanmask = pd.isnull(img)
    
    for i in tqdm(range(niter)):
        xnpos = np.random.uniform(low=1,high=xlen,size=100)
        ynpos = np.random.uniform(low=1,high=ylen,size=100)
        xynpos = list(zip(xnpos, ynpos))
        ap = 0.4/images_info.loc[0,'PIXSCALE']
        napers = CircularAperture(xynpos, r=ap)
        noise_table = aperture_photometry(img, napers, mask=nanmask, method='exact')
        for col in noise_table.colnames:
            noise_table[col].info.format = '%.8g'  # for consistent table output
        noise_table = Table(noise_table).to_pandas()
        noise_table['aperture_area'] = napers.area
        skymedians[i] = np.nanmedian(noise_table['aperture_sum'])
        bkgflux = np.nanmedian(skymedians)
        bkgfluxerr = np.nanstd(skymedians)
        bkgmadstd = mad_std(skymedians)
    
    if return_table:
        return noise_table
    else:
        return [bkgflux, bkgfluxerr, bkgmadstd]

def single_star_photometry(img, xc, yc, bkg, filename, images_info, upper_limit=False):
    
    shortfile = filename.split('.fits')[0]
    #maderr = mad_std(img,ignore_nan=True)
    nanmask = pd.isnull(img)
    ap = 0.4/images_info.loc[0,'PIXSCALE']
    filt = images_info.loc[images_info['FILENAME'] == filename,'FILTER'].values[0]
    vegazpt = config.VEGA_ZEROPTS[filt]
    pos = [xc, yc]
    apertures = CircularAperture(pos,r=ap)
    
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkgimg = Background2D(img, (50, 50), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    error = calc_total_error(img, bkgimg.background, effective_gain=images_info.loc[images_info['FILENAME'] == filename,'CCDGAIN'].values[0])
    
    phot = aperture_photometry(img, apertures, method='exact', error=error, mask=nanmask)
    phot = Table(phot).to_pandas()
    
    phot = phot.rename(columns={'aperture_sum':'APERTURE_FLUX','aperture_sum_err':'APERTURE_FLUX_ERR'})
    phot['BKG_FLUX'] = bkg[0]
    phot['BKG_FLUX_ERR'] = bkg[1]
    phot = phot.apply(pd.to_numeric,errors='ignore')
    if upper_limit:
        phot['STAR_FLUX'] = 3*bkg[2]
        phot['STAR_FLUX_ERR'] = np.nan
    else:
        phot['STAR_FLUX'] = phot['APERTURE_FLUX'] - phot['BKG_FLUX']
        phot['STAR_FLUX_ERR'] = np.sqrt(phot['APERTURE_FLUX_ERR']**2 + phot['BKG_FLUX_ERR']**2)

    phot['VEGAMAG'] = vegazpt - 2.5 * np.log10(phot['STAR_FLUX'])
    phot['VEGAMAG_UNC'] = 1.0857 * phot['STAR_FLUX_ERR'] / phot['STAR_FLUX']
    phot.to_csv('phot_table_'+str(shortfile)+'.csv')
    
    return phot

def psf_photometry(img,filename):
    
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkgrms = MADStdBackgroundRMS()
    bkg_estimator = MedianBackground()
    bkgimg = Background2D(img, (10, 10), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    sigma_psf = 3
        
    daofind = DAOStarFinder(fwhm=3.0, threshold=5.*mad_std(img))
    daogroup = DAOGroup(2.0*sigma_psf*gaussian_sigma_to_fwhm)
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    photometry = IterativelySubtractedPSFPhotometry(finder=daofind,
                                                    group_maker=daogroup,
                                                    bkg_estimator=bkg_estimator,
                                                    psf_model=psf_model,
                                                    fitter=LevMarLSQFitter(),
                                                    niters=100, fitshape=(11,11))
    
    #photometry = DAOPhotPSFPhotometry
    
    
    psftab = photometry(image=img).to_pandas()
    psftab = psftab.rename(columns={'flux_fit':'PSF_FLUX','flux_unc':'PSF_FLUX_ERR'})
    psftab = psftab.apply(pd.to_numeric,errors='ignore')
    #psftab['STAR_FLUX'] = psftab['APERTURE_FLUX'] - psftab['BKG_FLUX']
    #psftab['STAR_FLUX_ERR'] = np.sqrt(phot['APERTURE_FLUX_ERR']**2 + phot['BKG_FLUX_ERR']**2)
    #psftab['VEGAMAG'] = vegazpt - 2.5 * np.log10(phot['STAR_FLUX'])
    #psftab['VEGAMAG_UNC'] = 1.0857 * psftab['STAR_FLUX_ERR'] / psftab['STAR_FLUX']
    psftab.to_csv('psf_table_'+str(shortfile)+'.csv')
    
    residual_image = photometry.get_residual_image()
    shortfile = filename.split('.fits')[0]
    plt.imshow(residual_image, cmap='gray_r')
    plt.savefig('resimg_'+str(shortfile)+'.jpg')
    
    return result_tab.reset_index()

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='calculate photometry for HST image')
    parser.add_argument('--imgfolder', dest='imgfolder', help='path to folder containing FITS images')
    parser.add_argument('--RA', dest='RA', help='RA in HMS separated by colons')
    parser.add_argument('--DEC', dest='DEC', help='DEC in DMS separated by colons')
    parser.add_argument('--instrument', dest='instrument', default='HST_WFC3', help='instrument name')

    args = parser.parse_args()

    # load_images
    images_info, images = load_images(args.imgfolder, args.RA, args.DEC, args.instrument)

    # loop through individual images
    for filename, image in images.items():

        stop = False

        while stop == False:
            # ask for x,y coordinate
            print(filename)
            x = input("Enter X value: ")
            y = input("Enter Y value: ")

            # save image info to csv
            images_info.loc[images_info['FILENAME']==filename,'x'] = x
            images_info.loc[images_info['FILENAME']==filename,'y'] = y

            # find centroid
            xysources, cutout = find_centroid(image, x, y)

            # plot centroids
            plot_centroids(cutout, filename, x, y, xysources)
            cid = int(input('Pick centroid '))
            is_good = input('Accept centroid? ')
            if (is_good == 'yes') or (is_good == 'y'):
                stop = True
                xctr = xysources.loc[xysources['id']==cid,'xctr'].values[0]
                yctr = xysources.loc[xysources['id']==cid,'yctr'].values[0]
            elif (is_good == 'force') or (is_good == 'f'):
                #force centroid
                xctr = x
                yctr = y
                stop = True
               
            #images_info.loc[images_info['FILENAME']==filename,'xctr'] = xctr
            #images_info.loc[images_info['FILENAME']==filename,'yctr'] = yctr
            
            bkg = calculate_sky_background(image,images_info)
            phottab = single_star_photometry(image, xctr, yctr, bkg, filename, images_info)
            #psftab = psf_photometry(image)
            print(phottab)
            print(filename)
            images_info.loc[images_info['FILENAME'] == filename,'VEGAMAG'] = phottab['VEGAMAG'].values
            images_info.loc[images_info['FILENAME'] == filename,'VEGAMAG_UNC'] = phottab['VEGAMAG_UNC'].values
            print(images_info)

    images_info.to_csv(args.imgfolder+images_info.loc[0, 'TARGNAME']+'_summary.csv',index=False)
