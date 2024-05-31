## NOTE:

1) The results of PE runs shall depend on the noise realizations.

2) Since GW190425 was a low SNR events, we take the 'ngauss=0' (Gaussian_Nodes using fisher matrix are 0) 

3) The Prior Boundaries (Interpolation boundary for chirp mass and mass ratio is increased)

mchirp_min, mchirp_max = mchirp_center - 0.0005, mchirp_center + 0.0005  

mass_ratio_min, mass_ratio_max = 1, 1.28 (since mass_ratio_center - 0.07 < 1)

4) To get more accuracy we take (LivePoints = 1500 and nwalks=500) 

5) Prior on Comoving Volume is increased (mentioned in the PE scripts)

6) k=400 noise case seems to be the best case 