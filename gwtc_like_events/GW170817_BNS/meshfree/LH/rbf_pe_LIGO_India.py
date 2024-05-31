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

for k in [1500]:
        
    #-- Extracting the real noise from interferometers and evaluating their corresponding PSDs --

    event = 'GW170817'
    ifos = ['L1','H1']
    initial_gap = k
    raw_duration = 372
    A0_gap = 100
    A0_noise_det = 'V1'
    highpass_cutoff = 18
    filter_order = 4
    crop_len = 6
    psd_segLen=2
    psd_estimate_method='median-mean'
    low_frequency_cutoff= 20


    noise, noise_tilde, psd = extract_noise_psds(event, ifos, initial_gap, raw_duration, A0_gap,\
                                                 A0_noise_det, highpass_cutoff, filter_order, crop_len,\
                                                 psd_segLen, psd_estimate_method, low_frequency_cutoff, save_to_hdf=True)





    #-- Generating a GW170817 like data with signal embedded in real noise --

    # tc is taken such that it is 30 secs before end_time for L1 noise
    trigTime = float(noise["L1"].end_time) - 30   


    inject_params = {'m1_det': 1.399, #1.399899220066443,   #detector frame mass_1, taken from MAP values of (mass_1) GW170817 bilby samples
                     'm2_det': 1.351, #1.3506923499131296,  #detector frame mass_2,taken from MAP values (mass_2) of GW170817 bilby samples 
                     'spin_1z': 0.00013, #0.0001294116725339975, #spin_1z, taken from MAP values (spin_1z) of GW170817 bilby samples
                     'spin_2z': 3.54e-05, #3.5461352589076355e-05, #spin_2z, taken from MAP values (spin_1z) of GW170817 bilby samples
                     'luminosity_distance': 40.4,   #taken from EM measurement of NGC 4993 (src:Hjorth et al. (2017))
                     'iota': 2.497,       #2.4975664366099606,   # taken from MAP value (iota) of GW170817 bilby samples
                     'ra': 3.446, #3.44616, 
                     'dec': -0.408, #-0.408084, #location of host galaxy NGC-4993
                     'pol':0,
                     'tc': trigTime}   # tc is taken such that it is 30 secs before end_time for L1 noise

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
    segLen = 360                    # secs   (will also be used in start_up function)

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

    mchirp_min, mchirp_max = mchirp_center - 0.0002, mchirp_center + 0.0002   

    if(mass_ratio_center - 0.07 < 1):

        mass_ratio_min, mass_ratio_max = 1, 1.14

    else:

        mass_ratio_min, mass_ratio_max = mass_ratio_center - 0.07, mass_ratio_center + 0.07

    chi1z_min, chi1z_max = inject_params['spin_1z'] - 0.0025, inject_params['spin_1z'] + 0.0025
    chi2z_min, chi2z_max = inject_params['spin_2z'] - 0.0025, inject_params['spin_2z'] + 0.0025

    #-- Boundary dict --
    boundary = {'Mc': np.array([mchirp_min, mchirp_max]), 'mass_ratio': np.array([mass_ratio_min, mass_ratio_max]), \
                'chi1z': np.array([chi1z_min, chi1z_max]), 'chi2z': np.array([chi2z_min, chi2z_max])}





    #-- Section: start_up analysis --------------

    #-- waveform params ----
    approximant = 'TaylorF2'
    waveform_params = {'approximant': approximant, 'fLow': low_frequency_cutoff, 'fHigh': high_frequency_cutoff,\
                       'sampling_frequency': sampling_frequency}  # to remove redundancy of low_frequency_cutoff


    #-- rbf params---
    tau = 0.15
    Nbasis = 20
    phi = 'ga'
    order = 7 
    eps = 10 
    rbf_params = {'Nbasis': Nbasis, 'phi': phi, 'order': order, 'eps': eps, 'tau': tau}


    #-- Nodes params ---
    Nnodes = 800
    nodes_gauss_num = 0.2
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
    Vcom_min, Vcom_max = 5e3, 1e8       #7e3, 6e5      
    tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
    boundary_Vcom_tc = {'Vcom': np.array([Vcom_min, Vcom_max]), 'tc': np.array([tc_min, tc_max])}

    nLive = 500
    nWalks = 100
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

    #et_inj = time.time()

    # with open('seeds_and_timings_rbf.txt', 'a') as f:

    #     f.write('\n' + 'injection: %d, rbf_PE_time: %.2f minutes, sim_time: %.2f minutes'%(count, (et_PE-st_PE)/60,\
    #                                                                                        (et_inj-st_inj)/60))



    name = "".join(ifos)

    print('trigTime = {}'.format(trigTime))
    
    with open('noise_details.txt', 'a') as fn:
        
        fn.write('\n' + 'initial_gap: {} secs, trigTime = {} \n'.format(k, trigTime))
        
end_total_time = time.time()  

print("Total time for analysis of %s is : %.3f mins"%(name, (end_total_time-start_total_time)/60))
