#!/usr/bin/env python
"""
example_script.py -- sample usage of dpsp to analyze
some trajectories

"""
import pandas as pd
from dpsp import dpsp, load_tracks

def run_analysis():
    """
    Concatenate some sets of trajectories from different CSVs
    and run dpsp on the result.

    """
    # Target files
    track_csvs = ["example_tracks.csv", "more_example_tracks.csv"]

    # Load trajectories into one dataframe
    tracks = load_tracks(
        *track_csvs,
        start_frame=0,       # Don't include trajectories before this
        drop_singlets=True   # Drop single-point trajectories
    )

    # Run dpsp
    occs, diff_coefs = dpsp(
        tracks,
        branch_prob=0.05,
        frame_interval=0.00748,
        pixel_size_um=1.0,
        loc_error=0.03,
        n_iter=400,
        n_threads=1,
        pos_cols=["y", "x"],
        dz=0.7,
        plot=True,
        out_png="example_plots.png"
    )

if __name__ == "__main__":
    run_analysis()
