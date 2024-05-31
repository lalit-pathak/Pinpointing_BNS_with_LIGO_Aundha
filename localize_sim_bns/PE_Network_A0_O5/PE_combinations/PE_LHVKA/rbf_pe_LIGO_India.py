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

from pycbc.detector import add_detector_on_earth

from rbf_pe_utils_LIGO_India import cdfinv_q, start_up, get_cleaned_data_psd, prior_transform, \
                                                            RBFInterpolatedLikelihood, calc_mode, pycbc_samps_mode_vals, hdf_to_txt
from pe_sampler_LIGO_India import dynesty_sampler, generate_postSamps
from rbf_pe_utils_LIGO_India import generate_sim_data



start_total_time = time.time()

#----------------------------------------------------------------------------------------------------------------------#
#-- adding LIGO India to PyCBC detectors --

add_detector_on_earth(name='A0', longitude=1.34444215058, latitude=0.34231676739,\
                      yangle=4.23039066080, xangle=5.80120119264, height=440) 


#-- CHECK FOR THE HEIGHT PARAMETER VALUE --
#----------------------------------------------------------------------------------------------------------------------#

#-- COMBINATION CONSIDERED --

combination = 'L1H1V1K1A0'  #-- Along with this always change the 'ifos' defined inside the main loop below --


#-- loading the injection data from .txt file --

injection_data = np.loadtxt('injections_Net_SNR_20_to_25_{}.txt'.format(combination))

#-- defining a variable to store the total no. of injections --

total_injections = len(injection_data)

#-- releasing 'injection_data' from memory
del(injection_data)


#-- intrinsic source parameters -- 
    
m1_src = 1.387
m2_src = 1.326
s1z = 0.0001294116725339975
s2z = 3.5461352589076355e-05
    
#-- fixed extrinsic paramters -- 

iota, pol = np.pi/6, 0

#-- injection_approximant --
inj_approx = 'IMRPhenomD'

noise_param = 1    #-- this value will change and be saved for every combination (say LHVK or LHV or HLV)
    
for count in range(total_injections):
    
    st_inj = time.time()
    #-- reading the variable extrinsic parameters (ra, dec, distance, redshift, tc) --
    
    ra, dec, dL, z, trigTime = np.loadtxt('injections_Net_SNR_20_to_25_{}.txt'.format(combination))[count, :5]  # (Here '5' means it will go  till 4th indexed column)
    
    #-- storing trigTime as 'tc' for future reference
    tc = trigTime
    
    #-- Detector Frame masses --

    m1 = m1_src * (1+z)
    m2 = m2_src * (1+z)
    
    #-- injection_params array --

    inj_params = np.array([m1, m2, s1z, s2z, iota, pol, ra, dec, dL, z, tc])
    
    #-- Detectors taken for the analysis --
    
    ifos = ['L1', 'H1', 'V1', 'K1', 'A0']       # list of available detectors for the given event

    fLow, fHigh = 10, 1600

    low_frequency_cutoff, high_frequency_cutoff = 10, 1600   #(need to remove to optimize the code)

    sampling_frequency = 4096       # Hz
    segLen = 1500                   # secs
    
    #-- Derived Params from injections (set as centers for the interpolation box) --

    mchirp_center = mchirp_from_mass1_mass2(m1, m2)

    eta_center = eta_from_mass1_mass2(m1, m2)

    mass_ratio_center = q_from_mass1_mass2(m1, m2)
    
    #-- Generating the simulated data --
    
    noise_seed = count + noise_param     #-- initializing a noise 

    data, psd = generate_sim_data(inj_approx=inj_approx, fLow=fLow, segLen=segLen, ifos=ifos, inj_params=inj_params, noise_seed=noise_seed)

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
    fiducial_params['chi1z'] = s1z
    fiducial_params['chi2z'] = s2z
    fiducial_params['iota'] = iota
    fiducial_params['phi'] = ra
    fiducial_params['theta'] = dec
    fiducial_params['dL'] = dL
    fiducial_params.update(dict(psi=pol, tGPS=tc))  
    
    #-- Boundary for rbf interpolation --- (Check for the boundary width of Mchirp and boundary limits of mass_ratio)
    
    mchirp_min, mchirp_max = mchirp_center - 0.0001, mchirp_center + 0.0001   

    if(mass_ratio_center - 0.07 < 1):
    
        mass_ratio_min, mass_ratio_max = 1, 1.14
    
    else:
    
        mass_ratio_min, mass_ratio_max = mass_ratio_center - 0.07, mass_ratio_center + 0.07

    chi1z_min, chi1z_max = s1z - 0.0025, s1z + 0.0025
    chi2z_min, chi2z_max = s2z - 0.0025, s2z + 0.0025
    
    #-- Boundary dict --
    boundary = {'Mc': np.array([mchirp_min, mchirp_max]), 'mass_ratio': np.array([mass_ratio_min, mass_ratio_max]), \
                'chi1z': np.array([chi1z_min, chi1z_max]), 'chi2z': np.array([chi2z_min, chi2z_max])}
    
    
    #-- Section: start_up analysis --------------

    #-- waveform params ----
    approximant = 'TaylorF2'
    waveform_params = {'approximant': approximant, 'fLow': low_frequency_cutoff, 'fHigh': high_frequency_cutoff,\
                       'sampling_frequency': sampling_frequency}
    

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
    
    #filename = '{}_strain.hdf'.format(event)    
    # hdf_to_txt(event, filename, ifos) # function to convert 

    interp_data = start_up(fiducial_params, data_dict, waveform_params, rbf_params, nodes_params, \
                           ifos, nProcs, savetrainingdata=False, saveinterpolants=False)  #-- (changed from  True to False)

    interpolants = interp_data['interpolants']
    basis_vectors = interp_data['basis_vectors']   
    times = interp_data['times']
    
    interp_data.update(dict(nodes_params=nodes_params))

    et_start_up = time.time()

    print('start-up stage is completed......')
    print('start-up stage took %.2f seconds....'%(et_start_up-st_start_up))

#     # writing startup time and nodes seed in .txt file for future references
#     with open('seeds_and_timings_rbf.txt', 'a') as f:

#         f.write('nodes_seed: %d, start_up_time: %.2f minutes'%(nodes_seed, (et_start_up-st_start_up)/60))

        
    #-- Sampling using dynesty ---
    bVecs = 20
    det = {}

    for ifo in ifos:

        det[ifo] = Detector(ifo)

    #-- defining prior tranform ---
    Vcom_min, Vcom_max = 16973.036781764156, 890267808.9822887    # 16 Mpc, 680 Mpc
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
    
    eff_samps_dict = generate_postSamps(raw_samples, ifos, count, save_to_hdf=True)
    
    et_PE = time.time()
    
    et_inj = time.time()
    

    with open('seeds_and_timings_rbf.txt', 'a') as f:

        f.write('\n' + 'injection: %d, rbf_PE_time: %.2f minutes, sim_time: %.2f minutes'%(count, (et_PE-st_PE)/60,\
                                                                                           (et_inj-st_inj)/60))
        
end_total_time = time.time()

name = "".join(ifos)

print("Total time for %d injections for %s is : %.3f hours"%(total_injections, name, (end_total_time-start_total_time)/3600))
