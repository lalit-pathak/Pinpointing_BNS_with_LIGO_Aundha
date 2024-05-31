
# Pinpointing coalescing binary neutron star sources with the IGWN, including LIGO-Aundha


Welcome to the repository. This repository contains all the relevant code that underpins the results presented in the research paper: [PhysRevD.109.044051](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.044051), authored by Sachin Shukla, Lalit Pathak, and Anand Sengupta.

![BNS Collision](https://media4.giphy.com/media/v1.Y2lkPTc5MGI3NjExdHp5OHl1c3c4enJ3bHExYW51aDRkeWRwMHlvOGF3eTN5NzlzbXp5NCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/3ohc1f8hcZ7LBe2Zzy/giphy.webp)
Source: [MIT](https://youtu.be/sgkDoSbHHVU)

LIGO-Aundha (A1), the Indian gravitational wave detector, is expected to join the International Gravitational-Wave Observatory Network (IGWN) and commence operations in the early 2030s. The primary objective of this work is to examine the impact of this additional detector on the accuracy of determining the direction of incoming transient signals from coalescing binary neutron star (BNS) sources.

## Installations
To run the scripts and notebooks, ensure that you have the following:

* Required Installations: Python (version > 3.6)
* Required packages: PyCBC, GWFAST, RBF

Install PyCBC:

    $ git clone https://github.com/gwastro/pycbc.git
    $ cd pycbc
    $ pip install -r requirements.txt
    $ pip install -r companion.txt
    $ pip install .
    
Install GWfast: 

    $ pip install --upgrade pip
    $ pip install gwfast
    
Install RBF package: 

    $ git clone https://github.com/treverhines/RBF.git
    $ python3 setup.py install
    
## Contents

The repository contains the following main directories:

1)```psds```: The directory contains the code and the references used to generate the 'Noise Power Spectral Density (PSD)' employed for different detectors. The generated PSDs are used in the ```injections``` and ```localize_sim_bns``` directories discussed below.

2)```injections```: The directory contains the code to generate the simulated binary neutron star (BNS) sources which are used for the Gravitational Wave (GW) analysis and are used in the ```localize_sim_bns``` directory mentioned below.

3)```localize_sim_bns```: The directory contains the code where we perform Bayesian Parameter Estimation of the GWs from the coalescence of the simulated BNS sources, and estimate the improvements in localization uncertainties of the sources in the presence of LIGO-Aundha.

4)```gwtc_like_events```: The directory contains the code and the references where we simulate GW170817-like, GW190814-like, and GW200115-like events from the Gravitational Wave Transient Catalog. The impact of the presence of LIGO-Aundha is observed in the localization of these events.

## Authors

The authors involved in the development of the codes for this project:
- [Sachin Shukla](https://github.com/sachin-shukla-1402)
- [Lalit Pathak](https://github.com/lalit-pathak)
- Dr. Anand Sengupta

