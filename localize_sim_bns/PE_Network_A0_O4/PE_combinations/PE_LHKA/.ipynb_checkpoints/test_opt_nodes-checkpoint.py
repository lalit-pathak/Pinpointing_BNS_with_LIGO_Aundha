import os
import time
import glob
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
from scipy.optimize import minimize, basinhopping

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
from pycbc.inference.models.marginalized_gaussian_noise import MarginalizedPhaseGaussianNoise
from rbf_pe_utils import pycbc_log_likelihood, RBFInterpolatedLikelihood, prior_transform
from pycbc.conversions import mchirp_from_mass1_mass2, q_from_mass1_mass2, mass1_from_mchirp_q, mass2_from_mchirp_q

def net_snr(q):
    
    net_snr = 0
    
    snrs = {}
    
    m1, m2, s1z, s2z = q
            
    hp, hc = pycbc.waveform.get_fd_waveform(approximant=approximant, mass1=m1, mass2=m2, spin1z=s1z, spin2z=s2z, \
                                                            f_lower=fLow, delta_f = psd[list(data.keys())[0]].delta_f)
    for ifo in list(data.keys()):
    
        hp.resize(len(psd[ifo]))  # signal generated in frequency domain
        snrs[ifo] = pycbc.filter.matched_filter(hp, data[ifo], psd=psd[ifo],
                                                          low_frequency_cutoff=fLow, high_frequency_cutoff=fHigh)
        net_snr += abs(snrs[ifo]).max()**2
    
    net_snr = np.sqrt(net_snr)
            
    with open('points.txt', 'a') as f:

        f.write('\n' + '%.8f %.8f %.8f %.8f %.8f'%(m1, m2, s1z, s2z, net_snr))
        
    return -net_snr
                             
    
#-- PyCBC likelihood parameters ---
approximant = 'TaylorF2'
fLow = 20
fHigh = 1600
sampling_frequency = 4096 

psd = {}
data = {}
ifos = ['L1', 'H1', 'V1']
trigTime = 1187008882.4
low_frequency_cutoff = {}
high_frequency_cutoff = {}

for ifo in ifos:
    
    data[ifo] = load_frequencyseries('GW170817_strain.hdf', group='/data/strain_{}'.format(ifo))
    psd[ifo] = load_frequencyseries('GW170817_strain.hdf', group='/data/psd_{}'.format(ifo))
    low_frequency_cutoff[ifo] = fLow
    high_frequency_cutoff[ifo] = fHigh
    
#--best matched template information------
#--link: https://gracedb.ligo.org/events/view/G298107----
m1_bm = 1.4574227
m2_bm = 1.2993017
mchirp_bm = mchirp_from_mass1_mass2(m1_bm, m2_bm)
q_bm = m1_bm/m2_bm

s1z_bm = -0.018811436
s2z_bm = 0.011776949

x0 = (m1_bm, m2_bm, s1z_bm, s2z_bm)

mass1_min, mass1_max = 1.1, 2.35
mass2_min, mass2_max = 1.1, 2.35
s1z_min, s1z_max = -0.05, 0.05
s2z_min, s2z_max = -0.05, 0.05

bnds = ((mass1_min, mass1_max), (mass2_min, mass2_max), (s1z_min, s1z_max), (s2z_min, s2z_max))

with open('points.txt', 'w') as fp:
    pass

st = time.time()
val = minimize(net_snr, x0, bounds=bnds, options=dict(disp=True))
et = time.time()

print('Fiducial point found in {} seconds !! x_fid = {}'.format((et-st), val.x))

