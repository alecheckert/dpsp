#!/usr/bin/env python
"""
lik.py -- likelihood functions

"""
import sys
import os
import numpy as np
import pandas as pd
from scipy.special import gammainc, expi
from .utils import (
    squared_jumps,
    sum_squared_jumps
)

def likelihood_matrix(tracks, diff_coefs, posterior=None, 
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.03, 
    start_frame=None, pos_cols=["y", "x"], max_jumps_per_track=None,
    likelihood_mode="binned", by_jump=False):
    """
    For each of a set of trajectories, calculate the likelihood of 
    each of a set of diffusion coefficients.

    args
    ----
        tracks          :   pandas.DataFrame
        diff_coefs      :   1D np.ndarray, diffusion coefficients in 
                            squared microns per second
        posterior       :   1D np.ndarray, occupations of each diffusion
                            coefficient bin (for instance, the output of
                            dpsp)
        frame_interval  :   float, seconds
        pixel_size_um   :   float, microns
        loc_error       :   float, microns (root variance)
        start_frame     :   int, ignore jumps before this frame
        pos_cols        :   list of str, columns in *tracks* with the 
                            coordinates of each detections in pixels
        max_jumps_per_track :   int, the maximum number of jumps to 
                            consider from each trajectory
        likelihood_mode :   str, either "binned" or "point", the 
                            type of likelihood to calculate
        by_jump         :   bool, calculate likelihood on a jump-by-jump
                            basis, which does not make the assumption that
                            trajectories stay in the same state
    
    returns
    -------
        (
            2D ndarray of shape (n_tracks, n_bins), the likelihood
                of each diffusion coefficient bin for each trajectory;
            1D ndarray of shape (n_tracks,), the number of jumps per
                trajectory;
            1D ndarray of shape (n_tracks,), the indices of each 
                trajectory
        )

    """
    le2 = loc_error ** 2
    m = len(pos_cols)
    diff_coefs = np.asarray(diff_coefs)
    K = diff_coefs.shape[0]

    # Split each trajectory into separate jumps, which are treated
    # separately
    if by_jump:

        # Calculate all of the jumps in the dataset
        jumps = squared_jumps(tracks, n_frames=1, start_frame=start_frame, 
            pixel_size_um=pixel_size_um, pos_cols=pos_cols)

        # Format as a dataframe
        S = pd.DataFrame(jumps[:,:4],
            columns=["track_length", "trajectory", "frame", \
                "sum_sq_jump"])
        S["n_jumps"] = 1.0

        # Limit the number of jumps to consider per trajectory,
        # if desired
        if (not max_jumps_per_track is None) and (not max_jumps_per_track is np.inf):
            S = assign_index_in_track(S)
            S = S[S["index_in_track"] <= max_jumps_per_track]

        n_tracks = len(S)

    # Compute the sum of squared jumps for each trajectory
    else:
        S = sum_squared_jumps(tracks, n_frames=1, pixel_size_um=pixel_size_um,
            pos_cols=pos_cols, max_jumps_per_track=max_jumps_per_track,
            start_frame=start_frame)
        n_tracks = S["trajectory"].nunique()

    # Alpha parameter governing the gamma distribution over the 
    # sum of squared jumps
    S["deg_free"] = S["n_jumps"] * m / 2.0

    # Integrate likelihood across each diffusion coefficient bin
    if likelihood_mode == "binned":

        # Likelihood of each of the diffusion coefficients
        lik = np.zeros((n_tracks, K-1), dtype=np.float64)

        # Divide the trajectories into doublets and non-doublets
        doublets = np.asarray(S["deg_free"] == 1)
        S_doublets = np.asarray(S.loc[doublets, "sum_sq_jump"])
        S_nondoublets = np.asarray(S.loc[~doublets, "sum_sq_jump"])
        L_nondoublets = np.asarray(S.loc[~doublets, "deg_free"])
        
        for j in range(K-1):

            # Spatial variance
            V0 = 4 * (diff_coefs[j] * frame_interval + le2)
            V1 = 4 * (diff_coefs[j+1] * frame_interval + le2)

            # Deal with doublets
            lik[doublets, j] = expi(-S_doublets / V0) - expi(-S_doublets / V1)

            # Deal with everything else
            lik[~doublets, j] = (gammainc(L_nondoublets - 1, S_nondoublets / V0) - \
                gammainc(L_nondoublets - 1, S_nondoublets / V1)) / (L_nondoublets - 1)

        # Scale by state occupations
        if not posterior is None:
            posterior = np.asarray(posterior)
            lik = lik * posterior

    # Evaluate the likelihood in a pointwise manner
    elif likelihood_mode == "point":

        lik = np.zeros((n_tracks, K), dtype=np.float64)

        # Gamma degrees of freedom
        L = np.asarray(S["deg_free"])

        # Sum of squared jumps in each trajectory
        sum_r2 = np.asarray(S["sum_sq_jump"])

        # Calculate the log likelihood of each state
        for j in range(K-1):
            phi = 4 * (diff_coefs[j] * frame_interval + le2)
            lik[:,j] = -(sum_r2 / phi) - L * np.log(phi)

        # Scale by the state occupations, if desired
        if not posterior is None:
            posterior = np.asarray(posterior)
            nonzero = posterior > 0
            log_occs = np.full(posterior.shape, -np.inf)
            log_occs[nonzero] = np.log(posterior[nonzero])
            lik = lik + log_occs 

        # Convert to likelihood
        lik = (lik.T - lik.max(axis=1)).T
        lik = np.exp(lik)

    # Normalize
    lik = (lik.T / lik.sum(axis=1)).T 

    return lik, np.asarray(S["n_jumps"]), np.asarray(S["trajectory"])
