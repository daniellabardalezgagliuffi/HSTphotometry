import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from tqdm import tqdm
from astropy.io import fits
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std, sigma_clipped_stats, SigmaClip, sigma_clip
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
## python3 HSTphotometry.py --imgfolder=data/W0830+2837/ --RA=08:30:11.96 --DEC=+28:37:16.0 --upper_limit=True
## python3 HSTphotometry.py --imgfolder=data/W1516+7217/ --RA=15:16:20.40 --DEC=+72:17:45.5
## python3 HSTphotometry.py --imgfolder=data/W1525+6053/ --RA=15:25:29.10 --DEC=+60:53:56.6

## python3 HSTphotometry.py --imgfolder=data/Cushing_W0335+4310/ --RA=00:35:15.12 --DEC=+43:10:46.31

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
            ttexp = hdr[config.EXPTIME[instrument]]

        # append row to dataframe
        df = df.append(pd.Series(row), ignore_index=True)

        # store image
        images[row['FILENAME']] = fits_file[1].data

    # postprocessing
    if aperture is not None:
        df['APERTURE'] = aperture

    if pixscale is not None:
        df['PIXSCALE'] = pixscale
        
    if ttexp is not None:
        df['EXPTIME'] = ttexp

    df = df.apply(lambda x: pd.to_numeric(x,errors='ignore')).sort_values('FILENAME').reset_index().drop('index',1)

    return df, images

def find_centroid(img, x, y, upper_limit=False):

    x = float(x)
    y = float(y)
    cutout = img[int(y - 10):int(y + 10),int(x - 10):int(x + 10)]

    print(upper_limit == True)
    if upper_limit == False:
        daofind = DAOStarFinder(fwhm=3.0, threshold=3.*mad_std(cutout))
        xysources = daofind(cutout).to_pandas()
        xysources['xctr'] = xysources['xcentroid'] + x - 10
        xysources['yctr'] = xysources['ycentroid'] + y - 10
    else: 
        xysources = pd.DataFrame(columns=['xctr','yctr'])
        xysources['xctr'] = x
        xysources['yctr'] = y
    print(xysources)

    return xysources, cutout

def plot_centroids(cutout, filename, x, y, xysources):

    x = float(x)
    y = float(y)
    xctr = xysources['xctr'].astype(float)
    yctr = xysources['yctr'].astype(float)
    shortfile = filename.split('.fits')[0]

    plt.imshow(cutout, cmap='gray_r')
    plt.plot(10, 10, color='b', marker='+', markersize=30, linestyle='None')
    plt.plot(xctr - x + 10, yctr - y + 10, color='r', marker='+', markersize=30, linestyle='None')
    for i in range(len(xysources)):
        plt.annotate(xysources.loc[i,'id'],xy=(xysources.loc[i,'xcentroid'],xysources.loc[i,'ycentroid']),color='r')
    plt.title(shortfile)
    plt.xlim(0,20)
    plt.ylim(0,20)
    plt.xticks(np.arange(5) * 5, np.arange(5) * 5 - 10 + x)
    plt.yticks(np.arange(5) * 5, np.arange(5) * 5 - 10 + y)
    plt.legend(['image center','centroid'])
    plt.savefig('centroid_'+str(shortfile)+'.jpg')
    
    print('Look up centroid_'+str(shortfile)+'.jpg')
    plt.clf()

    
def calculate_sky_background(img,images_info,return_table=False):

    nskyap = 10000
    xlen, ylen = img.shape
    nanmask = pd.isnull(img)
    
    xnpos = np.random.uniform(low=1,high=xlen,size=nskyap)
    ynpos = np.random.uniform(low=1,high=ylen,size=nskyap)
    xynpos = list(zip(xnpos, ynpos))
    ap = 0.4/images_info.loc[0,'PIXSCALE']
    napers = CircularAperture(xynpos, r=ap)
    noise_table = aperture_photometry(img, napers, mask=nanmask, method='exact') 
    for col in noise_table.colnames:
        noise_table[col].info.format = '%.8g'  # for consistent table output
    noise_table = Table(noise_table).to_pandas()
    noise_table['aperture_area'] = napers.area
    
    #Sigma Clip image to avoid stars
    b1 = sigma_clip(noise_table['aperture_sum'])#, masked=False, axis=0) #default = 3 sigma
    a1 = b1.data[~b1.mask]
    a2 = b1.mask
    a2 = np.array([not x for x in a2])
    sigclip_skyflux = noise_table['aperture_sum']*a2
    bkgflux = np.average(a1) #rmean
    bkgfluxerr = np.nanstd(a1) #rstd
    
    print('background flux = ' + str(bkgflux))
    print('background flux error = ' + str(bkgfluxerr))
    
    if return_table:
        return noise_table
    else:
        return [bkgflux, bkgfluxerr]

def single_star_photometry(img, xc, yc, bkg, filename, images_info, upper_limit=False):
    
    shortfile = filename.split('.fits')[0]
    ttexp = images_info.loc[0,'EXPTIME']
    print('exptime = '+str(ttexp))
    nanmask = pd.isnull(img)
    ap = 0.4/images_info.loc[0,'PIXSCALE']
    filt = images_info.loc[images_info['FILENAME'] == filename,'FILTER'].values[0]
    vegazpt = config.VEGA_ZEROPTS[filt]
    pos = [float(xc), float(yc)]
    apertures = CircularAperture(pos,r=ap)

    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkgimg = Background2D(img, (20, 20), filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    error = calc_total_error(img, bkgimg.background, effective_gain=images_info.loc[images_info['FILENAME'] == filename,'CCDGAIN'].values[0])
  #  error = calc_total_error(img, bkgimg.background, effective_gain=config.GAIN[instrument])
     
    #For sky annuli background
    #sky_inner = 0.8/images_info.loc[0,'PIXSCALE']
    #sky_outer = 0.9/images_info.loc[0,'PIXSCALE']
    #skyannuli = CircularAnnulus(pos, r_in=sky_inner, r_out=sky_outer)
    #phot_apers = [apertures, skyannuli]
    #phot_table2 = Table(aperture_photometry(img, phot_apers, method='exact', error=error, mask=nanmask)).to_pandas()
    
    #For multiapertures background
    phot = aperture_photometry(img, apertures, method='exact', mask=nanmask)#error=error, 
    phot = Table(phot).to_pandas()
    
    phot = phot.rename(columns={'aperture_sum':'APERTURE_FLUX'})#,'aperture_sum_err':'APERTURE_FLUX_ERR'
    phot['BKG_FLUX'] = bkg[0] 
    phot['BKG_FLUX_ERR'] = bkg[1] 
    phot['APERTURE_FLUX_ERR'] = np.sqrt(phot['APERTURE_FLUX']/ttexp + 2 * phot['BKG_FLUX_ERR']**2)
    phot = phot.apply(pd.to_numeric,errors='ignore')
    if upper_limit == True:
        print('this is true')
        print('background flux error = ' + str(bkg[1]))
        phot['STAR_FLUX'] = 3 * bkg[1] 
        #3 sigma = 3 * 1.4826 * MAD since distribution is Gaussian - No more MAD, just STD
        phot['STAR_FLUX_ERR'] = np.nan
    else:
        #no need to divide over area because aperture areas for both star flux and backgroun flux are the same
        phot['STAR_FLUX'] = phot['APERTURE_FLUX'] - phot['BKG_FLUX']
        phot['STAR_FLUX_ERR'] = np.sqrt(phot['APERTURE_FLUX_ERR']**2 + phot['BKG_FLUX_ERR']**2)
     

        #bkg_mean = phot_table2['aperture_sum_1'] / skyannuli.area
        #bkg_starap_sum = bkg_mean * apertures.area
        #final_sum = phot_table2['aperture_sum_0']-bkg_starap_sum
        #phot_table2['bg_subtracted_star_counts'] = final_sum
        #bkg_mean_err = phot_table2['aperture_sum_err_1'] / skyannuli.area
        #bkg_sum_err = bkg_mean_err * apertures.area
        #phot_table2['bg_sub_star_cts_err'] = np.sqrt((phot_table2['aperture_sum_err_0']**2)+(bkg_sum_err**2)) 
        
        #phot['STAR_FLUX_ANNULI'] = final_sum
        #phot['STAR_FLUX_ERR_ANNULI'] = phot_table2['bg_sub_star_cts_err']

    phot['VEGAMAG'] = vegazpt - 2.5 * np.log10(phot['STAR_FLUX'])
    phot['VEGAMAG_UNC'] = 1.0857 * phot['STAR_FLUX_ERR'] / phot['STAR_FLUX']
    #phot['VEGAMAG_UNC_APPHOT_ONLY'] = 1.0857 * phot['APERTURE_FLUX_ERR'] / phot['STAR_FLUX']
    #phot['VEGAMAG_ANNULI'] = vegazpt - 2.5 * np.log10(phot['STAR_FLUX_ANNULI'])
    #phot['VEGAMAG_UNC_ANNULI'] = 1.0857 * phot['STAR_FLUX_ERR_ANNULI'] / phot['STAR_FLUX_ANNULI']
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
    parser.add_argument('--upper_limit', dest='upper_limit', default='False')
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
            if args.upper_limit == 'False':
                xysources, cutout = find_centroid(image, x, y)

                # plot centroids
                plot_centroids(cutout, filename, x, y, xysources)
                cid = int(input('Pick centroid id '))
                print(xysources.loc[xysources['id']==cid,['xctr','yctr']])
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
            else:
                print('this is true')
                xysources, cutout = find_centroid(image, x, y, upper_limit=True)
                xctr = x
                yctr = y
                stop = True
               
            images_info.loc[images_info['FILENAME']==filename,'xctr'] = xctr
            images_info.loc[images_info['FILENAME']==filename,'yctr'] = yctr
            
            bkg = calculate_sky_background(image,images_info)
            if args.upper_limit == 'False':
                phottab = single_star_photometry(image, xctr, yctr, bkg, filename, images_info, upper_limit=False)
            #psftab = psf_photometry(image)
                print(phottab)
                print(filename)
            else:
                phottab = single_star_photometry(image, xctr, yctr, bkg, filename, images_info, upper_limit=True)
            images_info.loc[images_info['FILENAME'] == filename,'APERTURE_FLUX'] = phottab['APERTURE_FLUX'].values
            images_info.loc[images_info['FILENAME'] == filename,'BKG_FLUX'] = phottab['BKG_FLUX'].values
            images_info.loc[images_info['FILENAME'] == filename,'STAR_FLUX'] = phottab['STAR_FLUX'].values
            images_info.loc[images_info['FILENAME'] == filename,'APERTURE_FLUX_ERR'] = phottab['APERTURE_FLUX_ERR'].values
            images_info.loc[images_info['FILENAME'] == filename,'BKG_FLUX_ERR'] = phottab['BKG_FLUX_ERR'].values
            images_info.loc[images_info['FILENAME'] == filename,'STAR_FLUX_ERR'] = phottab['STAR_FLUX_ERR'].values
            images_info.loc[images_info['FILENAME'] == filename,'VEGAMAG'] = phottab['VEGAMAG'].values
            images_info.loc[images_info['FILENAME'] == filename,'VEGAMAG_UNC'] = phottab['VEGAMAG_UNC'].values
#            images_info.loc[images_info['FILENAME'] == filename,'VEGAMAG_UNC_APPHOT_ONLY'] = phottab['VEGAMAG_UNC_APPHOT_ONLY'].values
            print(images_info)

    images_info.to_csv(args.imgfolder+images_info.loc[0, 'TARGNAME']+'_summary.csv',index=False)
    print('open ' + args.imgfolder+images_info.loc[0, 'TARGNAME']+'_summary.csv')