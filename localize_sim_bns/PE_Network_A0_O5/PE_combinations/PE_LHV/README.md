# Run the simulated binary neutron star events using meshfree PE


## Here are the instructions to run the PE:

1. Set the number of CPUs to be used in `run.py`.
2. Start PE by running `python3 run.py` on terminal.
3. Posterior samples are stored in the `post_samples` directory. 
4. Enter `plots` directory and generate the corner plots in `corner_plots_TaylorF2.ipynb` notebook. 



# Side Note: 

1) In 'rbf_pe_LIGO_India.py' file:

   1a) Change the combination to the required case. Here 'L1H1V1'
   
   2b) The 'noise_param' variable is set 2 here (so that we have a changing noise for every injection)
   
   3c) Change the 'ifos' list in the main loop. Here ['L1', 'H1', 'V1']
   
   4d) We do not save the 'rbf interpolants' 
   
   
2) The files with no 'A0' detector shall have the same results as in PE_Network_A0_O4, so no need to re-run.
   