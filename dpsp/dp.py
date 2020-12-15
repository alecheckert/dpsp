#!/usr/bin/env python
"""
dp.py

"""
import os
import warnings
import time 
import numpy as np
import pandas as pd
import dask
from .defoc import f_remain 
from .plot import plot_diff_coef_spectrum
from .utils import (
    assert_gs_dp_diff_exists,
    format_cl_args,
    sum_squared_jumps
)

# Default binning scheme
DEFAULT_DIFF_COEFS = np.logspace(-2.0, 2.0, 301)

def dpsp(tracks, diff_coefs=None, alpha=10.0, branch_prob=0.1, m=10,
    m0=30, n_iter=200, burnin=20, frame_interval=0.00748, splitsize=12,
    pixel_size_um=0.16, loc_error=0.03, B=20000, max_jumps_per_track=None,
    metropolis_sigma=0.1, n_threads=1, max_occ_weight=100, dz=None,
    start_frame=None, pos_cols=["y", "x"], use_defoc_likelihoods=False,
    plot=False, out_png="default.png"):
    """
    Estimate the spectrum of diffusion coefficients present in a 
    set of trajectories using a Dirichlet process mixture model.

    args
    ----
        tracks          :   pandas.DataFrame, trajectories. Must
                            contain the "trajectory" and "frame"
                            columns, along with the contents of 
                            *pos_cols*

        diff_coefs      :   1D ndarray of shape (n_bins+1,), the 
                            edges of the bins to use for dicretizing
                            the posterior density. If *None*, a 
                            default binning scheme is used.

        alpha           :   float, the concentration parameter for 
                            the Dirichlet process 

        branch_prob     :   float, alternative way to specify *alpha*.
                            The fraction of the total number of trajectories
                            in the dataset to use for *alpha*. If set,
                            this overrides *alpha*.

        m               :   int, the number of samples to use when 
                            drawing from the prior 

        m0              :   int, the initial number of samples to use

        n_iter          :   int, the number of iterations to do

        burnin          :   int, the number of iterations to do before
                            recording anything

        frame_interval  :   float, the time between frames in seconds

        splitsize       :   int, # jumps. If trajectories are larger than this,
                            split them into subtrajectories.

        pixel_size_um   :   float, the size of camera pixels in microns

        loc_error       :   float, root localization variance in microns

        B               :   int, the buffer size defining the maximum 
                            number of components active at any given 
                            time

        max_jumps_per_track :   int, the maximum number of jumps to 
                            use from any single trajectory

        metropolis_sigma:   float, the standard deviations of the 
                            steps for the Metropolis-Hastings steps

        n_threads       :   int, the number of independent Gibbs 
                            samplers to run in parallel

        max_occ_weight  :   int, the maximum jumps to use when weighting
                            the likelihood of any individual component

        dz              :   float, the depth of field in microns

        start_frame     :   int, ignore jumps before this frame

        pos_cols        :   list of str, the columns in *tracks* with
                            the spatial coordinates of each detection
                            ~in pixels~

        use_defoc_likelihoods   :   bool, incorporate defocalization
                            likelihoods into the likelihood used during
                            Gibbs sampling. Otherwise, if *dz* is 
                            specified, we do a single defocalization
                            correction at the very end, after
                            aggregating the posterior density across
                            threads.

        plot            :   bool, make some summary plots

        out_png         :   str, path to use for saving summary plots

    returns
    -------
        (
            1D ndarray of shape (n_bins,), the normalized posterior
                probability of the diffusion coefficient;
            1D ndarray of shape (n_bins+1,), the edges of each 
                diffusion coefficient in squared microns per second
        )

    """
    # Check that the Gibbs samplers exist
    assert_gs_dp_diff_exists(use_defoc_likelihoods)

    # Use a defocalization correction
    corr_defoc = (not dz is None) and (not dz is np.inf)   
    if use_defoc_likelihoods: assert corr_defoc 

    # If not given a binning scheme for the posterior distribution,
    # use the default binning scheme
    if diff_coefs is None:
        diff_coefs = DEFAULT_DIFF_COEFS
    n_bins = diff_coefs.shape[0] - 1

    # Minimum and maximum log diffusion coefficient
    min_log_D = np.log(diff_coefs.min())
    max_log_D = np.log(diff_coefs.max())


    ## PREPROCESSING

    # Calculate the sum of squared jumps for every trajectory
    jumps = sum_squared_jumps(tracks, n_frames=1, start_frame=start_frame,
        splitsize=splitsize, pixel_size_um=pixel_size_um, pos_cols=pos_cols,
        max_jumps_per_track=max_jumps_per_track)

    # If there are no jumps detected, bail
    if jumps.empty:
        warnings.warn("no jumps detected in dataset")
        return np.full(n_bins, np.nan, dtype=np.float64), diff_coefs

    # Integer jump count
    jumps["n_jumps"] = jumps["n_jumps"].astype(np.int64)

    # Save the sum of squared jumps and the number of jumps for
    # each trajectory as a CSV, to be read by the Gibbs samplers
    track_csv = "_TEMP.csv"
    jumps[["sum_sq_jump", "n_jumps"]].to_csv(track_csv, index=False, header=None)

    # If the user wants to set the concentration parameter in terms
    # of the branch probability, calculate the corresponding alpha 
    if not branch_prob is None:
        n_tracks = jumps["trajectory"].nunique()
        alpha = branch_prob * n_tracks / (1.0 - branch_prob)
        print("Calculated concentration parameter: %.3f" % alpha)

    # Calculate a discretized defocalization function, if explicitly 
    # considering defocalization to calculate state likelihoods at 
    # each iteration
    bias_csv = "_TEMP_CORR.csv"
    if use_defoc_likelihoods or corr_defoc:

        # Evaluate the fraction of particles that are expected to remain 
        # inside the focal volume after one frame interval
        corr = np.zeros(n_bins, dtype=np.float64)

        # Use the geometric mean of each diffusion coefficient bin to 
        # estimate the defocalization fraction for the whole bin
        Dgm = np.sqrt(diff_coefs[1:] * diff_coefs[:-1])
        for i, D in enumerate(Dgm):
            corr[i] = f_remain(D, 1, frame_interval, dz)[0]

        # Save to a CSV that can subsequently be passed to the Gibbs
        # sampler
        _out = pd.DataFrame(index=range(len(Dgm)), columns=["D", "f_remain"])
        _out["D"] = Dgm 
        _out["f_remain"] = corr
        _out.to_csv(bias_csv, index=False, header=None)
        del _out 

    # Format the call to the Gibbs sampler
    kwargs = {
        "alpha": alpha,
        "frame_interval": frame_interval,
        "m": m,
        "metropolis_sigma": metropolis_sigma,
        "B": B,
        "n_iter": n_iter,
        "burnin": burnin,
        "min_log_D": min_log_D,
        "max_log_D": max_log_D,
        "max_occ_weight": max_occ_weight,
        "loc_error": loc_error,
        "use_defoc_likelihoods": use_defoc_likelihoods,
    }
    if use_defoc_likelihoods:
        kwargs["bias_csv"] = bias_csv 

    commands = []
    for i in range(n_threads):
        commands.append(format_cl_args(
            track_csv,
            "_TEMP_OUT_{}.csv".format(i),
            verbose=(i==0),
            seed=int((time.perf_counter()+i)*1777)%373,
            **kwargs
        ))


    ## CORE GIBBS SAMPLING ROUTINE

    @dask.delayed 
    def gb(i):
        """
        i   :   int, the thread index

        """
        # Execute Gibbs sampling
        os.system(commands[i])

        # Read output and discretize the posterior density
        # of the Markov chain into a histogram
        df = pd.read_csv("_TEMP_OUT_{}.csv".format(i), header=None)
        df["D"] = (np.exp(df[0]) - 4 * (loc_error**2)) / (4 * frame_interval)
        H = np.histogram(
            df["D"],
            bins=diff_coefs,
            weights=df[1]
        )[0].astype(np.float64)

        return H 

    # Run threads in parallel
    scheduler = "processes" if n_threads > 1 else "single-threaded"
    threads = [gb(i) for i in range(n_threads)]
    posterior = dask.compute(
        *threads,
        scheduler=scheduler, 
        num_workers=n_threads
    )


    ## POST-PROCESSING

    # Aggregate densities across threads 
    posterior = np.asarray(posterior).sum(axis=0)

    # Correct for defocalization bias
    if corr_defoc:
        nonzero = posterior > 0
        posterior[nonzero] = posterior[nonzero] / corr[nonzero]

    # Normalize
    if posterior.sum() > 0:
        posterior /= posterior.sum()

    # Clean up
    for fn in (
        [track_csv, bias_csv] + 
        ["_TEMP_OUT_{}.csv".format(i) for i in range(n_threads)]
    ):
        if os.path.isfile(fn):
            os.remove(fn)

    # Show some output plots
    if plot:
        plot_diff_coef_spectrum(posterior, diff_coefs, 
            log_scale=True, out_png=out_png, ylim=None, 
            d_err=(loc_error**2 / frame_interval), 
            truncate_immobile_frac=True)

    # Return the posterior sum of Markov chain densities across
    # all threads, along with the binning scheme
    return posterior, diff_coefs 
