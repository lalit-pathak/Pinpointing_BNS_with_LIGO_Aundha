# Run the simulated binary neutron star events using meshfree PE


## Here are the instructions to run the PE:

1. Set the number of CPUs to be used in `run.py`.
2. Start PE by running `python3 run.py` on terminal.
3. Posterior samples are stored in the `post_samples` directory. 
4. Enter `plots` directory and generate the corner plots in `corner_plots_TaylorF2.ipynb` notebook. 



# Side Note: 

1) In 'rbf_pe_LIGO_India.py' file:

   1a) Change the combination to the required case. Here 'H1K1A0'
   
   2b) The 'noise_param' variable is set 14 here (so that we have a changing noise for every injection)
   
   3c) Change the 'ifos' list in the main loop. Here ['H1', 'K1', 'A0']
   
   4d) We do not save the 'rbf interpolants'
   
   
2) The injection file 'injections_Net_SNR_20_to_25_H1K1A0.txt' has the same injections as in 'injections_Net_SNR_20_to_25_L1H1V1K1.txt'. It should not be taken as a wrong idea that the injections having a network SNR from 20 to 25 in L1H1V1K1 network also have the network SNR values in the same range when observed in H1K1A0 network. This injection files have the same injections in all the network with or without A0 detector, the files are named just for the sake convenience in running the programs. 

3) Here we use the 'A0' detector abbreviation for LIGO-Aundha (India) detector. This is used only in the code while defining a new detector to avoid any potential confusion (Since A1 was assigned to Australia in the LAL code at the time of simulations). But all the results and final plots in the paper: https://journals.aps.org/prd/abstract/10.1103/PhysRevD.109.044051 , have LIGO-Aundha denoted by 'A1'.