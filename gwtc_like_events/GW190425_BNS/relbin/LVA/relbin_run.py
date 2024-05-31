#-- NOTE: Here we sample in 'comoving volume' but I think relative binning posses some issues when we do so. Instead for a better comparison the sampling should be done in 'luminosity distance'

# import modules 

# Settings the warnings to be ignored
import warnings
warnings.filterwarnings('ignore')

import os
import time  
import h5py   
import pickle  

import random   
import dynesty  
import matplotlib
import numpy as np
from tqdm import tqdm   
import cProfile     
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
from pycbc.types import zeros
from pycbc.filter import sigmasq
from pycbc.catalog import Merger
from gwosc.datasets import event_gps     
from pycbc.filter import highpass, matched_filter
from pycbc.detector import Detector
from pycbc.frame.frame import read_frame   
from pycbc.pnutils import get_final_freq
from pycbc.waveform import get_fd_waveform, get_td_waveform
from pycbc.types.timeseries import TimeSeries
from pycbc.types.timeseries import load_timeseries
from pycbc.types.array import complex_same_precision_as
from pycbc.filter.matchedfilter import get_cutoff_indices
from pycbc.types.frequencyseries import load_frequencyseries
from pycbc.waveform.generator import FDomainDetFrameGenerator, FDomainCBCGenerator
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise, GaussianNoise
from pycbc.inference.models.relbin import Relative  
from pycbc.conversions import mass1_from_mchirp_eta, mass2_from_mchirp_eta,\
tau0_from_mass1_mass2, tau3_from_mass1_mass2, mchirp_from_mass1_mass2, eta_from_mass1_mass2,\
mass1_from_tau0_tau3, mass2_from_tau0_tau3, mass1_from_mchirp_q, mass2_from_mchirp_q, q_from_mass1_mass2


#--
from pycbc.pnutils import f_SchwarzISCO
from pycbc.psd import interpolate, welch               # pycbc.psd.estimate module
from pycbc.types import FrequencySeries,TimeSeries
from pycbc.psd import interpolate, inverse_spectrum_truncation
from numpy.random import Generator, PCG64
from pycbc.cosmology import redshift_from_comoving_volume, distance_from_comoving_volume

#-- self defined function --
from extract_noise_psds import extract_noise_psds

st = time.time()

#-- adding LIGO India to PyCBC detectors --

add_detector_on_earth(name='A0', longitude=1.34444215058, latitude=0.34231676739,\
                      yangle=4.23039066080, xangle=5.80120119264, height=440) 


#-- Extracting the real noise from interferometers and evaluating their corresponding PSDs --

event = 'GW190425'
ifos = ['L1','V1','A0']
initial_gap = 100
raw_duration = 372
A0_gap = 100
A0_noise_det = 'V1'
highpass_cutoff = 18
filter_order = 4
crop_len = 6
psd_segLen=2
psd_estimate_method='median-mean'
low_frequency_cutoff= 20


noise, noise_tilde, psds = extract_noise_psds(event, ifos, initial_gap, raw_duration, A0_gap,\
                                             A0_noise_det, highpass_cutoff, filter_order, crop_len,\
                                             psd_segLen, psd_estimate_method, low_frequency_cutoff, save_to_hdf=True)


#-- Generating a GW170817 like data with signal embedded in real noise --

# tc is taken such that it is 30 secs before end_time for L1 noise
trigTime = float(noise["L1"].end_time) - 30   

  
inject_params = {'m1_det': 1.759,  #detector frame mass_1, taken from MAP values of (mass_1) GW190425 posterior samples
                 'm2_det': 1.654,  #detector frame mass_2,taken from MAP values (mass_2) of GW190425 posterior samples 
                 'spin_1z': 0.0011, #spin_1z, taken from MAP values (spin_1z) of GW190425 posterior samples
                 'spin_2z': 0.0019, #spin_2z, taken from MAP values (spin_1z) of GW190425 posterior samples
                 'luminosity_distance': 170.99,   #taken from MAP values (luminosity_distance) of GW190425 posterior samples
                 'iota': 0.583,       # taken from MAP value (iota) of GW190425 posterior samples
                 'ra': 4.338,         # taken from MAP value (ra) of GW190425 posterior samples
                 'dec': -0.254,       # taken from MAP value (dec) of GW190425 posterior samples
                 'pol':0,             # taken from MAP value (psi) of GW190425 posterior samples
                 'tc': trigTime}   # tc is taken such that it is 30 secs before end_time for L1 noise


# GW signal generation:
inj_approx = 'IMRPhenomD'
segLen = 360                     # secs
sampling_frequency = 4096        # Hz   # data cannot capture frequency above 2048 Hz
fLow = 20                        # Hz
fHigh = 1600                     # Hz

static_params_gen = {'approximant':inj_approx, 'f_lower': fLow, 'mass1': inject_params['m1_det'],\
                     'mass2': inject_params['m2_det'], 'spin1z': inject_params['spin_1z'], 'spin2z': inject_params['spin_2z'],\
                     'ra': inject_params['ra'], 'dec': inject_params['dec'], 'polarization': inject_params['pol'],\
                     'distance': inject_params['luminosity_distance'], 'inclination': inject_params['iota']}

variable_args = ['tc']

generator = FDomainDetFrameGenerator(FDomainCBCGenerator, epoch= inject_params['tc'] - 310, detectors=ifos,\
                                     variable_args=variable_args, delta_f = 1/segLen,  **static_params_gen)

signal = generator.generate(tc=inject_params['tc']) 


#-- data containers 

low_frequency_cutoff = {}
high_frequency_cutoff = {}

htilde = {}
data = {}

for ifo in ifos:
 
    low_frequency_cutoff[ifo] = fLow
    high_frequency_cutoff[ifo] = fHigh


#-- SIMULATED DATA GENERATION --

for ifo in ifos:
    
    htilde[ifo] = signal[ifo]
    htilde[ifo].resize(len(psds[ifo]))      # length of psds[ifo] is same as that of lenght of noise_tilde[ifo]
    
    data[ifo] = htilde[ifo] + noise_tilde[ifo]  
    
#-- setting up the loglikelihood model --

#-- params --
recover_approx = 'TaylorF2'                # for likelihoood model

#-- setting fixed parameters and factors ---
static_params = {'approximant': recover_approx, 'f_lower': fLow} 

#-- Here we use "comoving_volume" as a variable parameter instead of "distance" in relative binning --
variable_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'ra', 'dec', 'inclination', 'comoving_volume', 'tc', 'polarization']

trigTime = trigTime                    #-- to be used in defining prior transform

#-- fiducial params (for model)

epsilon = 0.00098
mass1_ref = inject_params['m1_det']
mass2_ref = inject_params['m2_det']
tc_ref = trigTime
ra_ref = inject_params['ra']
dec_ref = inject_params['dec']
pol_ref = 0
iota_ref = inject_params['iota']

fiducial_params = {'mass1': mass1_ref, 'mass2': mass2_ref, 'tc': tc_ref, 'ra': ra_ref, 'dec': dec_ref, 'polarization': pol_ref, 'inclination': iota_ref}

kwargs = {'psds': psds, 'static_params':static_params, 'high_frequency_cutoff': high_frequency_cutoff}



model = Relative(variable_params=variable_params, data=data, low_frequency_cutoff=low_frequency_cutoff, fiducial_params=fiducial_params, gammas=None, epsilon=epsilon, earth_rotation=False, marginalize_phase=True, **kwargs)


#-- Relbin Likelihood ---
def relbin_log_likelihood(q):
        
    """
    Function to calculate relbin log-likelihood
    
    Parameters
    -----------
    q: parameters
    model: pycbc object containg the information regarding noise and waveform model
    PSD: power spectral density
    data: strain data
    
    Returns
    --------
    model.loglr: marginalized phase likelihood 
    
    For more details visit: https://pycbc.org/pycbc/latest/html/_modules/pycbc/inference/models/
    marginalized_gaussian_noise.html#MarginalizedPhaseGaussianNoise
        
    """
    mchirp, mass_ratio, s1z, s2z, ra, dec, iota, pol, comoving_volume, tc = q 
    m1 = mass1_from_mchirp_q(mchirp, mass_ratio)
    m2 = mass2_from_mchirp_q(mchirp, mass_ratio)
    
    model.update(mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, tc=tc, comoving_volume=comoving_volume, ra=ra, dec=dec, inclination=iota, polarization=pol)

    return model.loglr     

#-- for setting same prior limits as in meshfree --

mchirp_center = mchirp_from_mass1_mass2(inject_params['m1_det'], inject_params['m2_det'])  
mass_ratio_center = q_from_mass1_mass2(inject_params['m1_det'], inject_params['m2_det'])


#-- defining prior tranform ---
mchirp_min, mchirp_max = mchirp_center - 0.0002, mchirp_center + 0.0002              # if Wide Priors were taken: 1.18, 1.21

if(mass_ratio_center - 0.07 < 1):
    q_min, q_max = 1, 1.20
else:
    q_min, q_max = mass_ratio_center - 0.07, mass_ratio_center + 0.07
    
s1z_min, s1z_max = inject_params['spin_1z'] - 0.0025, inject_params['spin_1z'] + 0.0025
s2z_min, s2z_max = inject_params['spin_2z'] - 0.0025, inject_params['spin_2z'] + 0.0025

tc_min, tc_max = trigTime - 0.12, trigTime + 0.12
#distance_min, distance_max = 1, 75
Vcom_min, Vcom_max = 5e3, 4e8


def cdf_param(mass_ratio):

    return -5. * mass_ratio**(-1./5) * hyp2f1(-2./5, -1./5, 4./5, -mass_ratio)
        
def cdfinv_q(mass_ratio_min, mass_ratio_max, value):

    mass_ratio_array = np.linspace(mass_ratio_min, mass_ratio_max, num=1000, endpoint=True)
    mass_ratio_invcdf_interp = interp1d(cdf_param(mass_ratio_array),
                               mass_ratio_array, kind='cubic',
                               bounds_error=True)
    
    return mass_ratio_invcdf_interp((cdf_param(mass_ratio_max) - cdf_param(mass_ratio_min)) * value + cdf_param(mass_ratio_min))


#-- priors (messages) --
# print('mchirp: A distrbution Uniform in m1 and m2 and constraint by mchirp\n**********************')
# print('q: A distrbution Uniform in m1 and m2 and constraint by q\n**********************')
# print('s1z: Uniform in btw. {}----{}\n**********************'.format(s1z_min, s1z_max))
# print('s2z: Uniform in btw. {}----{}\n**********************'.format(s2z_min, s2z_max))
# print('tc: Uniform in btw. {}----{}\n**********************'.format(tc_min, tc_max))
# print('distance: Uniform volume with min: {} and max: {} distance (Mpc)\n**********************'.format(distance_min, distance_max))
# print('ra-dec: Uniform sky\n**********************')
# print('iota: Uniform in cos(iota)\n**********************')
# print('pol: Uniform angle}\n**********************')


def prior_transform(cube):
        
    cube[0] = np.power((mchirp_max**2-mchirp_min**2)*cube[0]+mchirp_min**2,1./2)       # chirpmass: power law mc**1
    cube[1] = cdfinv_q(q_min, q_max, cube[1])                                          # eta: uniform prior
    cube[2] = s1z_min + (s1z_max - s1z_min) * cube[2]                                  # s1z: uniform prior
    cube[3] = s2z_min + (s2z_max - s2z_min) * cube[3]                                  # s2z: uniform prior
    cube[4] = 2*np.pi*cube[4]                                                          # ra: uniform sky
    cube[5] = np.arcsin(2*cube[5] - 1)                                                 # dec: uniform sky
    cube[6] = np.arccos(2*cube[6] - 1)                                                 # iota: sin angle
    cube[7] = 2*np.pi*cube[7]                                                          # pol: uniform angle
   # cube[8] = np.power((distance_max**3-distance_min**3)*cube[8]+distance_min**3,1./3) # distance: unifrom prior in dL**3
    cube[8] = Vcom_min + (Vcom_max - Vcom_min) * cube[8] 
    cube[9] = tc_min + (tc_max - tc_min) * cube[9]                                     # tc: uniform prior

    return cube



print('********** Sampling starts *********\n')
st = time.time()

nProcs = 64
nLive = 500
nWalks = 100
nDims = 10
seed_PE = 0
dlogz=0.1

with mp.Pool(nProcs) as pool:
    
    sampler = dynesty.NestedSampler(relbin_log_likelihood,
                                               prior_transform,
                                               ndim=nDims,
                                               sample='rwalk',
                                               pool=pool, 
                                               nlive=nLive,
                                               walks=nWalks, queue_size=nProcs, rstate= np.random.Generator(PCG64(seed=seed_PE)))
    sampler.run_nested(dlogz=dlogz)
    
    
#-- saving pe samples ---
result = sampler.results
print('Evidence:{}'.format(result['logz'][-1]))

sample_keys = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'ra', 'dec', 'iota', 'pol', 'Vcom', 'tc']

raw_samples = {}
i = 0
for key in sample_keys:
    
    raw_samples[key] = result['samples'][:,i]
    i = i + 1
    
raw_samples.update(dict(logwt=result['logwt'], logz=result['logz'], logl=result['logl'], seeds=dict(seed_PE=seed_PE)))


def generate_postSamps(raw_samples, ifos, save_to_hdf=False):
    
    """Function to generate posterior samples using the raw samples obtained from dynesty sampler
    
    Parameters
    ----------
    raw_samples: a dictionary containing raw samples 
    
    Returns
    -------
    eff_samps_dict: a dictionary containing the posterior samples
    """
    
    mchirp, q, chi1z = raw_samples['Mc'], raw_samples['mass_ratio'], raw_samples['chi1z']    
    chi2z, iota, ra, dec = raw_samples['chi2z'], raw_samples['iota'], raw_samples['ra'], raw_samples['dec']
    Vcom, pol, tc =  raw_samples['Vcom'], raw_samples['pol'], raw_samples['tc']
    seed_PE = raw_samples['seeds']['seed_PE']
    
        
    logwt = raw_samples['logwt']
    logz = raw_samples['logz']
    logl = raw_samples['logl']
        
    wts =  np.exp(logwt - logz[-1])

    samples = np.zeros((len(mchirp), 10))
    samples[:,0], samples[:,1], samples[:,2], samples[:,3], samples[:,4] = mchirp, q, chi1z, chi2z, iota
    samples[:,5], samples[:,6], samples[:,7], samples[:,8], samples[:,9] = ra, dec, Vcom, pol, tc
    effective_size = int(len(wts)/(1+(wts/wts.mean() - 1)**2).mean())
    
    np.random.seed(0)
    eff_samps_index = np.random.choice(len(mchirp), effective_size, p=wts, replace=False)
    eff_samps = samples[eff_samps_index]
    eff_samps_llr = logl[eff_samps_index]

    m1, m2 = mass1_from_mchirp_q(eff_samps[:,0], eff_samps[:,1]), mass2_from_mchirp_q(eff_samps[:,0], eff_samps[:,1])
    eta = eta_from_mass1_mass2(m1, m2)
    
    #-- from comoving distance to redshift and luminosity_distance
    z = redshift_from_comoving_volume(eff_samps[:,7])
    dL = distance_from_comoving_volume(eff_samps[:,7])
    
    chi_eff = (m1*eff_samps[:,2] + m2*eff_samps[:,3])/(m1 + m2)
    
    eff_samps_dict = {}
    
    params = ['Mc', 'mass_ratio', 'chi_eff', 'iota', 'ra', 'dec', 'z', 'dL']
    
    i = 0
    for p in params: 
        
        if(p=='chi_eff'):
            eff_samps_dict[p] = chi_eff
            i = i + 2
        elif(p=='z'):
            eff_samps_dict[p] = z
            i = i + 1
        elif(p=='dL'):
            eff_samps_dict[p] = dL
            i = i + 1
        else:
            eff_samps_dict[p] = eff_samps[:,i]
            i = i + 1
     
    eff_samps_dict.update(dict(pol=eff_samps[:,8], Vcom=eff_samps[:,7], tc=eff_samps[:,9], eta=eta, chi1z=eff_samps[:,2], chi2z=eff_samps[:,3], seed_PE=seed_PE))
            
    if(save_to_hdf):
        
        params = ['Mc', 'mass_ratio', 'chi1z', 'chi2z', 'chi_eff', 'iota', 'ra', 'dec', 'z', 'dL', 'Vcom', 'pol', 'tc']
        
        name = "".join(ifos)
        
        with h5py.File(os.getcwd() + '/post_samples/post_samples_interp_{}.hdf'.format(name), 'a') as file:
            
            for key in params:

                file.create_dataset(key, data=eff_samps_dict[key])
             
    return eff_samps_dict

eff_samps_dict = generate_postSamps(raw_samples, ifos, save_to_hdf=True)

et = time.time()

print('tc = {}'.format(trigTime))
print('Done!!!') 
print('Time taken: {} mins'.format((et-st)/60))
