## Regarding posterior samples file containing related to all the injections:

1) After running the Parameter Estimation code, the posterior samples file is generated in this directory with the name: 'post_samples_interp_SNR_20to25_{}.hdf', where {} is the detector network (H1V1K1 in this case).

2) Then 'parameter_credible_region.py' is run to evaluate the 90% credible region of various parameters. eg: sky-area. chirp-mass, etc. This file first generates the '.fits' file corresponding to each injection and then uses it to calculate the sky-localization area within the 90% credible region. On successful evaluation of all these quantities, the values are stored in 'quantities_of_interest.hdf' file.

3) The 'quantities_of_interest.hdf' file generated from the first combination (L1H1V1K1) was taken and then the samples obtained for the H1V1K1 combination were added to the same file. This is done for all the combinations consecutively. 