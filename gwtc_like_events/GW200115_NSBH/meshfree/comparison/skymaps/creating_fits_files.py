#-- import modules --


import numpy as np
import matplotlib.pyplot as plt
import h5py
import time
import math
from tqdm import tqdm
import astropy_healpix as ah
from astropy import units as u
import healpy as hp
import ligo.skymap.plot           
from ligo.skymap.tool import ArgumentParser, FileType
from ligo.skymap.io import fits
from ligo.skymap.tool import ligo_skymap_contour
from ligo.skymap.postprocess import contour
from ligo.skymap import postprocess
from astropy.coordinates import SkyCoord
from astropy.time import Time
import scipy.stats as st
from scipy.optimize import bisect

from ligo.skymap import io
from ligo.skymap.bayestar import rasterize
from ligo.skymap import version
from astropy.table import Table
from astropy.time import Time
import os
import sys
import pickle
from ligo.skymap.kde import Clustered2Plus1DSkyKDE, Clustered2DSkyKDE
import logging
from textwrap import wrap
from pycbc.detector import Detector


#-- creating fits files ----

def create_fits_file(event, noise_case, network, ra_samps, dec_samps, ci=90):
    
    #-- samples for calculating KDE --
    pts = np.column_stack((ra_samps, dec_samps)) 
    
    #-- KDE function params
    trials = 5
    jobs = 10
    
    #-- Evaluating ClusteredKDE using a k-means clustering algorithm
    
    skypost = Clustered2DSkyKDE(pts, trials=trials, jobs=jobs)

    hpmap = skypost.as_healpix(top_nside=16)
    
    #-- writing and saving the fits files --
    io.write_sky_map(os.getcwd() + '/fits_files/noise_{}_{}_{}_skymap.fits.gz'.format(noise_case, event, network), hpmap, nest=True)

    return None


#-- reading PE samples --

def pos_samples(filename):
    
    file = h5py.File(filename, 'r')

    mchirp, q, chi_eff = np.array(file['Mc']), np.array(file['mass_ratio']), np.array(file['chi_eff'])    
    iota, ra, dec = np.array(file['iota']), np.array(file['ra']), np.array(file['dec'])
    z =  np.array(file['z'])
    dL =  np.array(file['dL'])
        
    return np.column_stack((mchirp, chi_eff, iota, ra, dec, dL, z)), np.column_stack((mchirp, q, chi_eff, iota, ra, dec, dL, z))


#-- main code --

event = 'GW200115'

base_path = os.getcwd()  

k_array = np.array([300, 700, 1300, 1600, 1800, 1900, 2000])

for k in k_array:

    noise_case = k
    
    samps_LHV,_ = pos_samples(base_path+'/../../LHV/post_samples/post_samples_interp_L1H1V1_{}.hdf'.format(event,noise_case))
 
    samps_LHVA,_ = pos_samples(base_path+'/../../LHVA/post_samples/post_samples_interp_L1H1V1A0_{}.hdf'.format(event,noise_case)) 
    
    
    #-- ra & dec samps --

    LHV_ra_samps,LHV_dec_samps = samps_LHV[:,3], samps_LHV[:,4]

    LHVA_ra_samps,LHVA_dec_samps = samps_LHVA[:,3], samps_LHVA[:,4]
    
    networks = {'LHV': [LHV_ra_samps, LHV_dec_samps] , 'LHVA': [LHVA_ra_samps, LHVA_dec_samps]}
    
    for key, vals in networks.items():
    
        network = key
        ra_samps = vals[0]
        dec_samps = vals[1]
        create_fits_file(event=event, noise_case=noise_case, network=network, ra_samps=ra_samps, dec_samps=dec_samps, ci=90)
    
    
    
    

