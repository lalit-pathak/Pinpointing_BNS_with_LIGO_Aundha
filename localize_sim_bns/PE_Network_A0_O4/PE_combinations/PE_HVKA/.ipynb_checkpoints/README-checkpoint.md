# Run GW170817 event using meshfree PE


## Here are the instructions to run the PE:

1. Run `python3 get_optimized_center.py` on terminal to find the center for placing nodes in the intrinsic parameters space.
2. Set the number of CPUs to be used in `run.py`.
3. Start PE by running `python3 run.py` on terminal.
4. Posterior samples are stored in the `post_samples` directory.
5. Generate the corner plots in `corner_plots_TaylorF2.ipynb` notebook.