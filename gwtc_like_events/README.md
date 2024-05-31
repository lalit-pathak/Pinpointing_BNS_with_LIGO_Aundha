## Overview :

This section describes the second part of the paper:https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.044051.

* In this section we simulate GW170817-like, GW190814-like, and GW200115-like events from the Gravitational Wave Transient Catalog. 

* We compare the reduction in the localization uncertainty with the addition of LIGO-Aundha (A1) detector into the GW networks which were in observation mode during the real events. 

* The signals are put into real noise and the noise from the detector which recorded the lowest SNR during the real GW event is taken as a surrogate for the A1 detector noise. 

* Note that in each of the directories corresponding to the GWTC-like events we have deleted two types of files due to limited storage permissions:
  1) The LIGO Analysis posterior files (for getting the injection parameters from the MAP values of the posteriors).
  2) The hdf file containing real noise and psds. 
  
  The first file can be downloaded following the instructions presented in ```Generating_params_map_vals_for_injections.ipynb``` and the   second file can be created using the ```extracting_detector_noise_for_LIGO_India.ipynb``` files respectively.

 

