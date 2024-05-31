# importing modules/ libraries

import os
import h5py   
import numpy as np
from tqdm import tqdm     
import matplotlib.pyplot as plt

from scipy import special
from scipy.interpolate import interp1d
from scipy.special import hyp2f1       # Gauss hypergeometric funciton 2F1(a,b;c;z)
from scipy.linalg import svd   
from scipy.interpolate import CubicSpline, PchipInterpolator

from multiprocessing import Process
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

import pycbc
from pycbc.fft import ifft
from pycbc.catalog import Merger
from gwosc.datasets import event_gps     
from pycbc.filter import highpass, matched_filter
from pycbc.detector import Detector
from pycbc.frame.frame import read_frame   
from pycbc.pnutils import get_final_freq
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.types.array import complex_same_precision_as
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise, GaussianNoise
from pycbc.pnutils import f_SchwarzISCO
from pycbc.psd import interpolate, welch               # pycbc.psd.estimate module
from pycbc.types import FrequencySeries,TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation





def extract_noise_psds(event, ifos, initial_gap, raw_duration, A0_gap, A0_noise_det, highpass_cutoff, filter_order, crop_len, 
                       psd_segLen, psd_estimate_method, low_frequency_cutoff, save_to_hdf=False):
    
    """
    Function to give noise (in both domains) and psds from .hdf files containing detector strain as PyCBC timeseries
    
    Parameters
    ----------
    
    event : GW Event Name (eg: GW170817)
    
    ifos : list of interferometers for which we need noise time_series
       
    initial_gap : Initial gap from the 'start_time' of LIGO frame file (secs) [Float value] 
       
    raw_duration : The raw segLen(duration) of the noise(in secs) to be extracted from the frame file (secs)
                 (This is before highpass_filter & cropping)
       
    A0_gap : The time difference of the start time for the noise segment for LIGO India from the end_time of the noise for
            L,H,V dets
       
    A0_noise_det : The detector (abbrev) whose noise data would be used for LIGO India [As per PyCBC]
       
    highpass_cutoff : The highpass cutoff frequency [for each detector] used in PSD calculation   (Hz)
       
    filter_order : float-type [by default use 4]
       
    crop_len : The length of noise to be cropped from both ends (in seconds) after highpass_filter [By default use 6]
       
    psd_segLen : The segment length used for psd interpolation (secs)
                float (by default use 2, but 4, 6, 8, 16 can also be used. THe smaller value ensure better calculation of psds)
                
    psd_estimate_method : method to estimate the psd of the time segment(welch method) [by default use: 'median-mean']
    
    low_frequency_cutoff : The lower cutoff frequency used in 'inverse_spectrum_truncation()'  (Hz)
       
    Return: 
    
    noise : Noise Timeseries                         [PyCBC Timeseries]
    noise_tilde : Noise FrequencySeries              [PyCBC FrequencySeries]
    psds : PSD FrequencySeries                       [PyCBC FrequencySeries]
    
    """
   
   #-- dictionaries --
    psds = {}
    noise = {}
    noise_tilde = {}
    strain = {}
    
    #-- load the timeseries from hdf files --
    
    for ifo in ifos:
        
        if (ifo !='A0') :
                               
            strain[ifo] = load_timeseries(os.getcwd()+'/../../{}_post_merger_only_noise_1240215703-5000.hdf'.format(event),\
                                          group='/strain_{}'.format(ifo))
            
        else:
            
            strain[ifo] = load_timeseries(os.getcwd()+'/../../{}_post_merger_only_noise_1240215703-5000.hdf'.format(event),\
                                          group='/strain_{}'.format(A0_noise_det))
            
            
    #-- We shall take the raw-noise data-duration to be 372 secs (then effective duration would be cropped to 360 secs)

    st_L1 = 1240215703 + initial_gap              # start time for L, H and V dets
    et_L1 = st_L1 + raw_duration                  # end time for L, H and V dets

    st_I1 = et_L1 + A0_gap                        # start time for A0 detector (LIGO India)
    et_I1 = st_I1 + raw_duration                  # end time for A0 detector (LIGO India)

    #-- reading the noise data ---
    
    for ifo in ifos:

        if (ifo !='A0') :
            
            # required noise strain for L1,H1,V1
            ts = strain[ifo].time_slice(st_L1, et_L1)

            # Read the detector data and remove low frequency content
            noise[ifo] = highpass(ts, highpass_cutoff, filter_order=filter_order)  # frequency below highpass_cutoff is suppressed

            # Remove time corrupted by the high pass filter
            noise[ifo] = noise[ifo].crop(crop_len, crop_len)  

            # Also create a frequency domain version of the data
            noise_tilde[ifo] = noise[ifo].to_frequencyseries()

        else :
            
            # required noise strain for A0
            ts = strain[ifo].time_slice(st_I1, et_I1)

            # Read the detector data and remove low frequency content
            noise[ifo] = highpass(ts, highpass_cutoff, filter_order=filter_order)  # frequency below highpass_cutoff is suppressed

            # Remove time corrupted by the high pass filter
            noise[ifo] = noise[ifo].crop(crop_len, crop_len)           

            # Also create a frequency domain version of the data
            noise_tilde[ifo] = noise[ifo].to_frequencyseries()

            #-- relabelling the epoch for noise_tilde["A0"] --(Relabelled according to the start_time of noise["L1"] or any det)

            noise_tilde[ifo] = FrequencySeries(noise_tilde[ifo], delta_f = noise_tilde[ifo].delta_f,\
                                                epoch=noise_tilde["L1"].epoch)
            
            noise[ifo] = noise_tilde[ifo].to_timeseries()
    
    # Remove the 'Event_noise_strain.hdf' if it exists 
    
    if os.path.exists("{}_noise_strain.hdf".format(event)):    
        os.remove("{}_noise_strain.hdf".format(event))

    #-- PSD generation ---
    for ifo in ifos:
        
        # We then interpolate the PSD to the desired frequency step. 
        psds[ifo] = interpolate(noise[ifo].psd(psd_segLen, avg_method=psd_estimate_method), noise_tilde[ifo].delta_f)      

        # We explicitly control how much data will be corrupted by overwhitening the data later on
        psds[ifo] = inverse_spectrum_truncation(psds[ifo], int(2 * noise[ifo].sample_rate),\
                                                low_frequency_cutoff=low_frequency_cutoff, trunc_method='hann')
        
        if(save_to_hdf):

                noise_tilde[ifo].save('{}_noise_strain.hdf'.format(event), group='/data/noise_tilde_{}'.format(ifo))
                psds[ifo].save('{}_noise_strain.hdf'.format(event), group='/data/psd_{}'.format(ifo))

    return noise, noise_tilde, psds




