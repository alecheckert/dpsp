#!/usr/bin/env python
"""
example_plots.py -- a short demo on how to use some of the 
plotting utilities included with dpsp

"""
import os
import sys
from glob import glob
import numpy as np
import pandas as pd

# Plotting functions
from dpsp import (
    plot_spatial_likelihood,
    plot_likelihood_by_file
)

def make_example_plots():
    """
    A short demo on making some plots. This uses some trajectories
    acquired from U2OS cells containing NPM1-HaloTag-3xFLAG, which
    have been labeled with the dye PA-JFX549 and tracked at 7.48 ms
    frame intervals with 1 ms stroboscopic illumination pulses. 

    First, we'll make a plot of the likelihood for each of a set of
    diffusion coefficients by file. This can be useful to see how
    much variability there is between individual cells.

    Second, for a single cell, we'll plot the likelihood for each 
    of a smaller set of diffusion coefficients as a function of the
    spatial position of the corresponding trajectories in the cell.

    """
    # Specify some example files
    example_dir = os.path.dirname(os.path.abspath(__file__))
    example_track_csvs = [os.path.join(example_dir, f) for f in [
        "npm1_ht_tracks_cell_1.csv",
        "npm1_ht_tracks_cell_2.csv",
        "npm1_ht_tracks_cell_3.csv"
    ]]

    # Make a plot that shows the prior likelihoods of each 
    # of a set of diffusion coefficients
    plot_likelihood_by_file(
        example_track_csvs,
        diff_coefs=np.logspace(-2.0, 2.0, 301),
        posterior=None,
        frame_interval=0.00748,
        start_frame=8000,
        pixel_size_um=0.16,
        loc_error=0.035,
        label_by_file=True,
        out_png="example_plot_likelihood_by_file.png"
    )

    # Show the likelihood of each of a set of diffusion
    # coefficients as a function of the spatial position
    # of the trajectory
    diff_coefs = np.array([0.01, 0.7, 2.0, 10.0])
    plot_spatial_likelihood(
        example_track_csvs[0],
        diff_coefs,
        posterior=None,
        filter_kernel_um=0.1,
        frame_interval=0.00748,
        start_frame=8000,
        pixel_size_um=0.16,
        loc_error=0.035,
        out_png="example_plot_spatial_likelihood.png"
    )

if __name__ == "__main__":
    make_example_plots()
