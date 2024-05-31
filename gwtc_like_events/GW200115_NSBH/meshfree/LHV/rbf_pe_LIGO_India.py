import warnings
  
# Settings the warnings to be ignored
warnings.filterwarnings('ignore')

import os
import sys
import time
import glob
import shutil
import h5py
import psutil
import pickle
import random
import dynesty
import cProfile
import matplotlib
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from scipy import special
from scipy.linalg import svd
from scipy.special import hyp2f1
from scipy.interpolate import interp1d
from scipy.interpolate import CubicSpline, PchipInterpolator

import multiprocessing as mp
from numpy.random import Generator, PCG64

from rbf.poly import mvmonos
from rbf.interpolate import RBFInterpolant

import pycbc
from pycbc.fft import ifft
from pycbc.types import zeros
from pycbc.filter import sigmasq
from pycbc.catalog import Merger
from pycbc.detector import Detector
from gwosc.datasets import event_gps
from pycbc.frame.frame import read_frame
from pycbc.pnutils import get_final_freq
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.conversions import mass1_from_mchirp_q, mass2_from_mchirp_q, \
                                    tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2
from pycbc.conversions import mass1_from_tau0_tau3, mass2_from_tau0_tau3, \
                                    mass1_from_mchirp_eta, mass2_from_mchirp_eta, eta_from_mass1_mass2, q_from_mass1_mass2

from rbf_pe_utils_LIGO_India import cdfinv_q, start_up, prior_transform, \
                                                            RBFInterpolatedLikelihood, calc_mode, pycbc_samps_mode_vals, hdf_to_txt
from pe_sampler_LIGO_India import dynesty_sampler, generate_postSamps
from rbf_pe_utils_LIGO_India import generate_sim_data
from extract_noise_psds import extract_noise_psds

start_total_time = time.time()

for k in np.arange(1200, 2200, 100):
        
    #-- Extracting the real noise from interferometers and evaluating their corresponding PSDs --

    event = 'GW200115'
    ifos = ['L1','H1','V1']
    initial_gap = k
    raw_duration = 76     # secs [segLen=64 (power of 2) because duration function shows min length ~ 51 secs at f_lower=20]
    A0_gap = 100           # secs
    A0_noise_det = 'V1'
    highpass_cutoff = 18   # Hz
    filter_order = 4       
    crop_len = 6           # secs
    psd_segLen=2
    psd_estimate_method='median-mean'
    low_frequency_cutoff= 20  # secs


    noise, noise_tilde, psd = extract_noise_psds(event, ifos, initial_gap, raw_duration, A0_gap,\
                                                 A0_noise_det, highpass_cutoff, filter_order, crop_len,\
                                                 psd_segLen, psd_estimate_method, low_frequency_cutoff, save_to_hdf=True)





    #-- Generating a GW200115 like data with signal embedded in real noise --

    # tc is taken such that it is 3 secs before end_time for L1 noise
    trigTime = float(noise["L1"].end_time) - 3   # since waveform duration is ~51 secs (64-3=61 secs is for the entire waveform)


    inject_params = {'m1_det': 7.409,  #detector frame mass_1, taken from MAP values of (mass_1) GW200115 posterior samples
                     'm2_det': 1.351,  #detector frame mass_2,taken from MAP values (mass_2) of GW200115 posterior samples 
                     'spin_1z': -0.0034, #spin_1z, taken from MAP values (spin_1z) of GW200115 posterior samples
                     'spin_2z': -0.0002, #spin_2z, taken from MAP values (spin_1z) of GW200115 posterior samples
                     'luminosity_distance': 269,   #taken from MAP values (luminosity_distance) of GW200115 posterior samples
                     'iota': 0.606,       # taken from MAP value (iota) of GW200115 posterior samples
                     'ra': 0.721,         # taken from MAP value (ra) of GW200115 posterior samples
                     'dec': -0.021,       # taken from MAP value (dec) of GW200115 posterior samples
                     'pol':0,             # taken from MAP value (psi) of GW200115 posterior samples
                     'tc': trigTime}   # tc is taken such that it is 3 secs before end_time for L1 noise

    #-- injection_approximant --

    inj_approx = 'IMRPhenomD'

    #-- injection_params array --

    inj_params = np.array([inject_params['m1_det'], inject_params['m2_det'],\
                           inject_params['spin_1z'], inject_params['spin_2z'],\
                           inject_params['iota'], inject_params['luminosity_distance'],\
                           inject_params['ra'], inject_params['dec'],\
                           inject_params['pol'], inject_params['tc']])


    fLow, fHigh = 20, 1600

    low_frequency_cutoff, high_frequency_cutoff = 20, 1600   #(need to remove to optimize the code)

    sampling_frequency = 4096       # Hz
    segLen = 64                     # secs   (will also be used in start_up function)

    #-- Derived Params from injections (set as centers for the interpolation box) --
    #-- These should be detector frame bcoz we get these from template search --

    mchirp_center = mchirp_from_mass1_mass2(inject_params['m1_det'], inject_params['m2_det'])  

    eta_center = eta_from_mass1_mass2(inject_params['m1_det'], inject_params['m2_det'])

    mass_ratio_center = q_from_mass1_mass2(inject_params['m1_det'], inject_params['m2_det'])


    #-- calling the 'generate_sim_data' function from 'rbf_pe_utils' --

    data = generate_sim_data(inj_approx=inj_approx, fLow=fLow, segLen=segLen, ifos=ifos, inj_params=inj_params,\
                             noise_tilde=noise_tilde, psd=psd)

    data_dict = {'psd': psd, 'data': data}




    #-- START UP STAGE --#   

    st_start_up = time.time()

    # center from the trigger and calculate the covariance matrix at the center

    params_list = ['Mc', 'eta', 'chi1z', 'chi2z', 'iota', 'phi', 'theta', 'dL']  #-- not being used in the analysis

    del(params_list) 

    #-- DEFINING FIDUCIAL PARAMS --

    fiducial_params = {}

    fiducial_params['Mc'] = mchirp_center
    fiducial_params['eta'] = eta_center
    fiducial_params['chi1z'] = inject_params['spin_1z']
    fiducial_params['chi2z'] = inject_params['spin_2z']
    fiducial_params['iota'] = inject_params['iota']
    fiducial_params['phi'] = inject_params['ra']
    fiducial_params['theta'] = inject_params['dec']
    fiducial_params['dL'] = inject_params['luminosity_distance']
    fiducial_params.update(dict(psi=inject_params['pol'], tGPS=trigTime)) 

    #-- Boundary for rbf interpolation --- (Check for the boundary width of Mchirp and boundary limits of mass_ratio)

    mchirp_min, mchirp_max = mchirp_center - 0.0011, mchirp_center + 0.0011  

    if(mass_ratio_center - 0.07 < 1): # (Not valid here)

        mass_ratio_min, mass_ratio_max = 2, 8

    else:

        mass_ratio_min, mass_ratio_max = mass_ratio_center - 0.1, mass_ratio_center + 0.1  # (previous: +_0.07)

    chi1z_min, chi1z_max = inject_params['spin_1z'] - 0.0025, inject_params['spin_1z'] + 0.0025
    chi2z_min, chi2z_max = inject_params['spin_2z'] - 0.0025, inject_params['spin_2z'] + 0.0025

    #-- Boundary dict --
    boundary = {'Mc': np.array([mchirp_min, mchirp_max]), 'mass_ratio': np.array([mass_ratio_min, mass_ratio_max]), \
                'chi1z': np.array([chi1z_min, chi1z_max]), 'chi2z': np.array([chi2z_min, chi2z_max])}



    #-- Section: start_up analysis --------------

    #-- waveform params ----
    approximant = 'IMRPhenomD'
    waveform_params = {'approximant': approximant, 'fLow': low_frequency_cutoff, 'fHigh': high_frequency_cutoff,\
                       'sampling_frequency': sampling_frequency}  # to remove redundancy of low_frequency_cutoff


    #-- rbf params---
    tau = 0.15
    Nbasis = 20
    phi = 'ga'
    order = 10 #7
    eps = 30 
    rbf_params = {'Nbasis': Nbasis, 'phi': phi, 'order': order, 'eps': eps, 'tau': tau}


    #-- Nodes params ---
    Nnodes = 1100
    nodes_gauss_num = 0.1
    nodes_seed = 9
    nodes_params = {'boundary': boundary, 'Nnodes': Nnodes, 'nodes_gauss_num': nodes_gauss_num, 'nodes_seed': nodes_seed}


    print('start-up stage starts........')

    #-- startup stage ---

    nProcs = int(sys.argv[1])

    filename = '{}_noise_strain.hdf'.format(event)    
    hdf_to_txt(event, filename, ifos) # function to convert 

    interp_data = start_up(fiducial_params, data_dict, waveform_params, rbf_params, nodes_params, \
                           ifos, nProcs, segLen, event, savetrainingdata=False, saveinterpolants=False)  #-- (changed from  True to False)

    interpolants = interp_data['interpolants']
    basis_vectors = interp_data['basis_vectors']   
    times = interp_data['times']

    interp_data.update(dict(nodes_params=nodes_params))

    et_start_up = time.time()

    print('start-up stage is completed......')
    print('start-up stage took %.2f seconds....'%(et_start_up-st_start_up))


    #-- Sampling using dynesty ---
    bVecs = 20
    det = {}

    for ifo in ifos:

        det[ifo] = Detector(ifo)



    #-- defining prior tranform ---
    
    #(source:https://github.com/gwastro/4-ogc/blob/master/inference_configuration/inference-GW200115_042309.ini)       
    Vcom_min, Vcom_max = 1e6, 3e9       #2.6e7, 3e9 
    
    tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
    boundary_Vcom_tc = {'Vcom': np.array([Vcom_min, Vcom_max]), 'tc': np.array([tc_min, tc_max])}

    nLive = 500 # 2500
    nWalks = 100 # 350
    nDims = 10    #-- verify --
    dlogz = 0.1
    sample = 'rwalk'
    seed_PE = 0

    sampler_params = dict(nLive = nLive, nWalks=nWalks, nDims=nDims, dlogz=dlogz, \
                                              sample=sample, seed_PE=seed_PE, nProcs=nProcs)

    params = {'det': det, 'low_frequency_cutoff': low_frequency_cutoff, \
                              'sampling_frequency': sampling_frequency, 'ifos': ifos, 'bVecs': bVecs}

    st_PE = time.time()

    raw_samples = dynesty_sampler(interp_data, boundary, boundary_Vcom_tc, sampler_params=sampler_params, save_to_hdf=False,\
                                  **params)

    eff_samps_dict = generate_postSamps(raw_samples, ifos, k, save_to_hdf=True)

    et_PE = time.time()

    name = "".join(ifos)

    print('trigTime = {}'.format(trigTime))
    
    with open('noise_details.txt', 'a') as fn:
        
        fn.write('\n' + 'initial_gap: {} secs, trigTime = {} \n'.format(k, trigTime))
        
end_total_time = time.time()  

print("Total time for analysis of %s is : %.3f mins"%(name, (end_total_time-start_total_time)/60))
