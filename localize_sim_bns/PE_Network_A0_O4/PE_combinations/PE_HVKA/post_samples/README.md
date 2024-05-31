## Regarding posterior samples file containing related to all the injections:

1) After running the Parameter Estimation code, the posterior samples file is generated in this directory with the name: 'post_samples_interp_SNR_{}.hdf', where {} is the detector network (H1V1K1A0 in this case).

2) Then 'parameter_credible_region.py' is run to evaluate the 90% credible region of various parameters. eg: sky-area. chirp-mass, etc. This file first generates the '.fits' file corresponding to each injection and then uses it to calculate the sky-localization area within the 90% credible region. On successful evaluation of all these quantities, the values are stored in 'quantities_of_interest.hdf' file.

3) Here 'post_samples_interp_SNR_{}.hdf' does not have the 'SNR 20 to 25' in its nomenclature because the observed events in 'H1V1K1A0' network may have higher/lower network SNR as compared to the ones observed in 'L1H1V1K1' network. This is just to avoid the confusion. (Although this has not been done for combinations without A0 detector, but the same argument is valid there too. This confusion should be avoided)