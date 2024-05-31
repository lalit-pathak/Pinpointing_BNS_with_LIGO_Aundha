## Overview :

This section describes the first part of the paper:https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.044051.

Here we analyse the simulated binary neutron star sources in the presence of L1H1V1K1 and L1H1V1K1A1 networks respectively. Two scenarios with the LIGO-Aundha (A1) detector configured at O4 and A+ Design Sensitivity (O5) respectively are taken into consideration. Duty cycles of the detectors are also taken into account for a more comprehensive estimate. 

* Note that the files containing the 'posterior samples' and 'quantities_of_interest' in the ```post_samples``` sub-directories of each subnetwork configuration are not present owing to the limited storage permits. Although these files can be generated using the code as per the instructions provided in each subnetwork directory.

Required packages: PyCBC, GWfast, RBF

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
    
    