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
    sum_squared_jumps
)

def likelihood_matrix(tracks, diff_coefs, occupations=None, 
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.03, 
    pos_cols=["y", "x"], max_jumps_per_track=None,
    likelihood_mode="binned"):
    """
    For each of a set of trajectories, calculate the likelihood of 
    each of a set of diffusion coefficients.

    args
    ----
        tracks          :   pandas.DataFrame
        diff_coefs      :   1D np.ndarray, diffusion coefficients in 
                            squared microns per second
        occupations     :   1D np.ndarray, occupations of each diffusion
                            coefficient bin
        frame_interval  :   float, seconds
        pixel_size_um   :   float, microns
        loc_error       :   float, microns (root variance)
        pos_cols        :   list of str, columns in *tracks* with the 
                            coordinates of each detections in pixels
        max_jumps_per_track :   int, the maximum number of jumps to 
                            consider from each trajectory
        likelihood_mode :   str, either "binned" or "point", the 
                            type of likelihood to calculate
    
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

    # Compute the sum of squared jumps for each trajectory
    S = sum_squared_jumps(tracks, n_frames=1, pixel_size_um=pixel_size_um,
        pos_cols=pos_cols, max_jumps_per_track=max_jumps_per_track)
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
        if not occupations is None:
            occupations = np.asarray(occupations)
            lik = lik * occupations

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
        if not occupations is None:
            occupations = np.asarray(occupations)
            nonzero = occupations > 0
            log_occs = np.full(occupations.shape, -np.inf)
            log_occs[nonzero] = np.log(occupations[nonzero])
            lik = lik + log_occs 

        # Convert to likelihood
        lik = (lik.T - lik.max(axis=1)).T
        lik = np.exp(lik)

    # Normalize
    lik = (lik.T / lik.sum(axis=1)).T 

    return lik, np.asarray(S["n_jumps"]), np.asarray(S["trajectory"])
