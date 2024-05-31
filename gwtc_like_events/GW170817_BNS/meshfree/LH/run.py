import os
# nProcs = 64 # number of CPUs
# cmd = 'taskset -c 32-{} python3 rbf_pe_LIGO_India.py {}'.format(nProcs-1, nProcs-32)
# os.system(cmd)
nProcs = 64 # number of CPUs
cmd = 'taskset -c 0-{} python3 rbf_pe_LIGO_India.py {}'.format(nProcs-1, nProcs)
os.system(cmd)