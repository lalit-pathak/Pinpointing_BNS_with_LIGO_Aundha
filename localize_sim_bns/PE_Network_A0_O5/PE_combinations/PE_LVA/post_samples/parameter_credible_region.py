#-- This python script is used to calculate and store 90% credible regions of [GW parameters + delta_omega**2 (sky area)] 
#-- in a hdf file (quantities_of_interest.hdf). The 90% credible regions for each parameter is stored as an array dataset with 
#-- 500 elements (each element referring to an injection). These arrays are stored in a hdf file group structure for ease of 
#-- access. 



# import libraries and modules

import time
import numpy as np
import h5py
import math
import astropy_healpix as ah
from astropy import units as u
import healpy as hp
import ligo.skymap.plot           
from ligo.skymap.tool import ArgumentParser, FileType
from ligo.skymap.io import fits
from ligo.skymap.tool import ligo_skymap_contour
from ligo.skymap.postprocess import contour
from ligo.skymap import postprocess

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.time import Time

import scipy.stats as st
from scipy.optimize import bisect

from ligo.skymap import io
from ligo.skymap.bayestar import rasterize
from ligo.skymap import version
from astropy.table import Table
from astropy.time import Time
import numpy as np
import os
import sys
import pickle
from ligo.skymap.kde import Clustered2Plus1DSkyKDE, Clustered2DSkyKDE
import logging
from textwrap import wrap



#-- function to calculate 90% credible area (ra,dec) --

def skymap_credible_area(count, ra_samps, dec_samps, ci=90):
    
    """Function to get desired level credible region skymap area. By default, calculates 90% credible sky area.
    It also creates and stores '.fits' files according to the injection number in a separate folder('fits files')
    
    Parameters
    ----------
    count    : injection number
    ra_samps : posterior samples for right ascension
    dec_samps: posterior samples for declination
    ci       : credible interval (default = 90%)
    
    Returns
    -------
    area : 90% credible region in delta_omega**2 """
    
    #-- samples for calculating KDE --
    pts = np.column_stack((ra_samps, dec_samps)) 
    
    #-- KDE function params
    trials = 1
    jobs = 1
    
    #-- Evaluating ClusteredKDE using a k-means clustering algorithm
    
    skypost = Clustered2DSkyKDE(pts, trials=trials, jobs=trials)

    hpmap = skypost.as_healpix(top_nside=16)
    
    #-- writing and saving the fits files --
    io.write_sky_map(os.getcwd() + '/fits_files/injection_{}_skymap.fits.gz'.format(count), hpmap, nest=True)

    #-- Calculating the skymap area from Healpix map --

    skymap, metadata = fits.read_sky_map(os.getcwd() + '/fits_files/injection_{}_skymap.fits.gz'.format(count), nest=None)
    nside = ah.npix_to_nside(len(skymap))

    #-- Convert sky map from probability to probability per square degree.

    deg2perpix = ah.nside_to_pixel_area(nside).to_value(u.deg**2)
    probperdeg = skymap / deg2perpix

    levels = [ci]   # this should be provided as a list
                    #(see: https://github.com/lpsinger/ligo.skymap/blob/main/ligo/skymap/tool/ligo_skymap_plot.py)

    #-- for contours 
    vals = 100 * postprocess.find_greedy_credible_levels(skymap)

    #-- for area of 90% credible region 

    pp = np.round([ci]).astype(int)               
    area = (np.searchsorted(np.sort(vals), [ci]) * deg2perpix)[0]
    
    return area


#-- main program --

ifos = ['L1', 'V1', 'A0']

name = "".join(ifos)

total_injections = 500

#-- arrays to stores 90% credible bound values of parameters --

param_list = ['Mc', 'mass_ratio', 'chi_eff', 'ra', 'dec', 'iota', 'dL', 'z', 'pol', 'tc']

param_ci_dict = {}    #-- dictionary to store the 90% credible bound for each parameter

for p in param_list:
    
    if p=='ra':
        pass
            
    elif p=='dec':
        pass
    
    else:   
        param_ci_dict[p] = np.zeros(total_injections)   #-- to store 90% ci of GW parameters
        
#-- for skymap area (delta_omega**2)--

delta_omega = np.zeros(total_injections)



#-- calculation of 90% credible region for parameters starts --

st = time.time()

for count in range(total_injections):
     
    with h5py.File(os.getcwd() + '/post_samples_interp_SNR_{}.hdf'.format(name), 'a') as file :
        
        group = file['injection_{}'.format(count)]
        
        #-- unravel the posterior samples --
        
        params = {}
        
        for p in param_list:
            
            params[p] = np.array(group[p])
        
        #-- 90% credible region in sky_area --
        
        area = skymap_credible_area(count=count, ra_samps= params['ra'], dec_samps= params['dec'], ci=90)
        
        #-- 90% credible region for other parameters --
        
        for p in param_list:
            
            if p=='ra':
                pass
            
            elif p=='dec':
                pass
            
            else:
                
                param_ci_dict[p][count] = np.quantile(params[p], 0.95) - np.quantile(params[p], 0.05)
     
    #-- storing the area in the delta_omega array --

    delta_omega[count] = area

    print('injection {}, area: {}'.format(count, area)) 
    
et = time.time()
    
print('time for 500 events: {} hours'.format((et-st)/3600))
    
     
#-- creating a new hdf file 'quantities_of_interest.hdf' with group name wrt the detector combination --
#-- each group (detector combination) would contain 'datasets' of 90% credible regions of GW parameters --

with h5py.File(os.getcwd() + '/quantities_of_interest.hdf', 'a') as new_file :
    
    group = new_file.create_group(name)   #-- (Here, name = H1V1K1A0)
    
    group.create_dataset('delta_omega', data = delta_omega)
    
    for p in param_list:
        
        if p=='ra':    
            pass
            
        elif p=='dec':
            pass
        
        else:
            group.create_dataset('delta_{}'.format(p), data=param_ci_dict[p])