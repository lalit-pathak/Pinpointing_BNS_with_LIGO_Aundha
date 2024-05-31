#-- import modules --

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
import os
import scipy
from itertools import combinations
from scipy.stats import gaussian_kde




#---------------- FUNCTION TO CALCULATE THE PROBABILITY OF EACH COMBINATION FOR A GIVEN DUTY RATE OF DETECTORS ---------------

def combination_prob_calculator(network, duty_rate, det_num_allowed):
    
    """
    Function to calculate the probability of a 'detector combination' from a 'Network Combination' to be active 
    during an observation run.
    
    Parameter
    ---------
    network: List of Detectors in a Network for a Science Observation Run
    
    duty_rate: Dictionary object containing the detector abbreviations as keys and Detector Duty Cycle Rate as value
    
    det_num_allowed: List object specifying "No. of Detector combinations allowed"
                     eg: 1) [3, 4] for a Four Detector Network
                         2) [3, 4, 5] for a Five Detector Network
    
    Returns
    -------
    
    combination_prob: A dictionary with the probability of a detector combination to be active during an observation
    run
    
    """
    
    #-- List to store all detector combinations in a Detector "NETWORK"
    
    possible_combinations = []
    
    
    #-- creating the detector combinations and storing the combinations in 'possible_combinations'
    for det_num in det_num_allowed:
        
        for comb in combinations(network, det_num):
            
            possible_combinations.append(list(comb))
    
    #print('possible_combinations: ', possible_combinations)
    
    #-- Dictionary containing the probability of a particular detector combination to be active --
    
    combination_prob = {}
    
    #-- Calculating the probability of each combination in 'possible_combinations' --
    
    for comb in possible_combinations:
    
        #-- name of the combination whose probability is to be evaluated
        name = "".join(comb)

        #-- initializing the probability with temporary variable 'prob'
        prob = 1

        #-- Calculating the probability for the combination as prob = (det_1)_mode * (det_2)_mode * (det_3)_mode * (det_4)_mode
        
        for i in comb:

            prob *= duty_rate[i]
       
        for j in network:

            if (j not in comb):
              #  print("{} not in {}".format(j,name))
                prob = prob * (1 - duty_rate[j])
     
    
        #-- Here we round the probability obtained upto 4 floating decimals, because of the binary representation limitations in python
        #-- while performing multiple arithmetic operations of 'multiplication'
        
        combination_prob[name] = np.round(prob, 4)
        
    print("\n")  
    print("Probabilities: ", combination_prob)
    
    total_prob = 0
    
    for key,val in combination_prob.items():
        
        total_prob += val
    
    print("\n")
    print('Total Probability:', total_prob)   
    
    return combination_prob, total_prob




#-------------------FUNCTION FOR GETTING EVENTS IN EACH DETECTOR COMBINATION a/c TO PROBABILITY FROM DUTY CYCLE -----------


##-- Code for counting total number of events distributed among each detector combination --

def network_events_calculator(net_comb_prob_list, total_events):
    
    """
    Function to calculate the number of events a 'detector combination' from a 'Network Combination' will observe 
    during an observation run. This is a special function which should be used only for a specific purpose. Here in
    our case, we have a limited number of simulated events, so in order to get a good statistic, we have defined 
    this function to calculate the number of events for a combination. Here, the combination with the highest probability
    is taken as a 'reference' (and assumes all the simulated events) and other combinations with lower probabilities 
    are distributed the events by considering the 'reference' as base value.
    
    Parameter
    ---------
    net_comb_prob_list : List of Detector Network Probabilities for a Science Observation Run
    
    total_events : Total number of simulated events

    
    Returns
    -------
    
    network_events: A Nested dictionary giving the number of events observed by a detector combination of a particular Network
    
    """
    
    
    #-- list to store all the probability values for the each 'combination' from each 'Network'
    
    prob_full_list = []
    
    #-- Going through each 'Network' -- (eg: LHVK or LHVKA )
    for net in net_comb_prob_list:
        
        #-- going through each combination of a particular network --
        for keys,vals in net.items():
    
            prob_full_list.append(vals)
    
    #-- converting the probability list into a numpy array --
    temp_prob = np.array(prob_full_list)
    
    #-- Highest Probability (among all the combinations across all network (As Required for our Case)) --
    
    highest_prob = temp_prob.max()      #-- To be Taken as a 'Reference' and would contain all simulated events
    
    
    #-- Nested Dictionary to be created to store the number of events for each combination for a network, 
    #-- where 'NETWORK' name would be a keyword for a dictionary containing the events for the corresponding 'NETWORK'
    
    network_events = {}

    #-- Variable to be used in defining the 'PRIMARY KEY's representing Networks in the Nested Dictionary 'network_events' --
    
    name_size = 0

    
    #-- Going through each 'Network' -- (eg: LHVK or LHVKA )
    for net in net_comb_prob_list:
        
        #-- going through each combination of a particular network --
        for key,val in net.items():

            if (len(key) > name_size):

                name_size = len(key)
                temp_name = key
        
        #-- Nested Dictionary Initializes its element 'dictionary' --
        network_events[temp_name] = {}
        
        #-- Going through combinations in a particular Detector Network
        for key,val in net.items():

            name = "".join(key)
            
            #-- Nested Dictionary [Primary_Dictionary][Secondary Dictionary]
            network_events[temp_name][name] = (val/highest_prob) * total_events
            
            #-- where, Primary_Dictionary stores the Network Name.
            #-- eg: LHVK (4-Detector-Network), LHVKA (5-Detector-Network)
            
            #-- Secondary_Dictionary stores the number of events of a particular 'Combination' in the 'Network'
            #-- eg: network_events['LHVKA']['LHV']
            
            
    return network_events



#------------------------------------- FUNCTION TO CALCULATE EMPERICAL CDF --------------------------------------------------

def ecdf(samples):
    
    x = np.sort(samples)
    
    cdf = np.arange(1, len(samples)+1)/len(samples)
    
    return x, cdf

#-- This code makes more sense to me. Since CDF represents fraction as the data corresponding to 'quantity' of interest increases
#-- This means that as we have the first element of dataset our CDF increase from zero and should have a non-zero value 
#-- The increase in the value should be in 'fraction = 1/(total_length_of_dataset). Then as we take more values from the dataset, the fraction increases.
#-- ecdf_v1 does this job very well but the issue is with the rounding off for 'unique' values and also "decreases the no. of datapoints"


#------------------------------ I dont know why importing this function is causing error --------------------------------------

def new_cdf(samples):
    
    #sort data
    x = np.sort(samples)

    #calculate CDF values
    y = np.arange(len(samples)) / (len(samples) - 1)
    
    return x, y
    

