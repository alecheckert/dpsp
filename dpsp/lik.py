#!/usr/bin/env python
"""
lik.py -- likelihood functions

"""
import sys
import os
from tqdm import tqdm 
import numpy as np
import pandas as pd
from scipy.special import gammainc, expi
from .utils import (
    squared_jumps,
    sum_squared_jumps,
    track_length,
    assign_index_in_track
)

def likelihood_matrix(tracks, diff_coefs, posterior=None, 
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.03, 
    start_frame=None, pos_cols=["y", "x"], max_jumps_per_track=None,
    likelihood_mode="point", by_jump=False):
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
                gammainc(L_nondoublets - 1, S_nondoublets / V1)) / gamma(L_nondoublets - 1)

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

def likelihood_matrix_fbm(tracks, hurst_pars, diff_coefs, posterior=None,
    frame_interval=0.00748, pixel_size_um=0.16, loc_error=0.03,
    start_frame=None, pos_cols=["y", "x"], max_jumps_per_track=None, 
    min_jumps_per_track=1):
    """
    Evaluate the likelihood of different combinations of parameters
    for fractional Brownian motion on each of a set of trajectories.
    The result is a 3D ndarray that gives the likelihood of each 
    2-tuple (Hurst parameter, diffusion coefficient) for each 
    trajectory.

    note
    ----
        Each state has an associated Hurst parameter and diffusion
        coefficient, which are layed out in a 2D ndarray. The total
        set of 2-tuples (Hurst parameter, diffusion coefficient)
        is formed by the Cartesian product of *hurst_pars* and 
        *diff_coefs*.

    args
    ----
        tracks          :   pandas.DataFrame
        hurst_pars      :   1D ndarray, the Hurst parameters 
                            corresponding to each state
        diff_coefs      :   1D ndarray, the diffusion coefficients
                            corresponding to each state
        posterior       :   1D ndarray of shape (n_hurst, n_diff_coefs),
                            the posterior occupation of each state.
                            If *None*, all states are given equal prior
                            occupancy.
        frame_interval  :   float, seconds
        pixel_size_um   :   float, microns
        loc_error       :   float, root variance in microns
        start_frame     :   int, discard jumps before this frame
        pos_cols        :   list of str, the y- and x- column names 
                            in *tracks*
        max_jumps_per_track :   int, the maximum number of jumps to 
                            consider per trajectory
        min_jumps_per_track :   int, the minimum number of jumps to 
                            consider per trajectory

    returns
    -------
        (
            3D ndarray of shape (n_tracks, n_hurst, n_diff_coefs),
                the likelihoods of each state for each trajectory.
                These likelihoods are normalized to sum to 1 across
                all states for that trajectory;
            1D ndarray of shape (n_tracks), the number of jumps per
                trajectory;
            1D ndarray of shape (n_tracks), the indices of each 
                trajectory in the original dataframe
        )

    """
    # Convenience
    le2 = loc_error ** 2
    m = len(pos_cols)

    # Coerce into ndarray
    diff_coefs = np.asarray(diff_coefs)
    hurst_pars = np.asarray(hurst_pars)
    nD = diff_coefs.shape[0]
    nH = hurst_pars.shape[0]

    def bail():
        return np.zeros((0, nH, nD), dtype=np.float64), np.zeros(0), np.zeros(0)

    # Avoid modifying the original dataframe
    tracks = tracks.copy()

    # Disregard jumps before the start frame
    if (not start_frame is None):
        tracks = tracks[tracks["frame"] >= start_frame]

    # Purge trajectories that are too short
    tracks = track_length(tracks)
    tracks = tracks[tracks["track_length"] >= (min_jumps_per_track+1)]

    # Truncate trajectories that are too long
    if (not max_jumps_per_track is None) and (not max_jumps_per_track is np.inf):
        tracks = assign_index_in_track(tracks)
        tracks = tracks[tracks["index_in_track"] <= max_jumps_per_track]
        tracks = track_length(tracks)

    # If no trajectories remain, bail
    if tracks.empty: return bail()

    # Convert from pixels to microns
    tracks[pos_cols] = tracks[pos_cols] * pixel_size_um 

    # The number of points in each trajectory
    track_lengths = np.asarray(tracks.groupby("trajectory").size())
    max_track_len = max(track_lengths)

    # The original index of each trajectory
    track_indices = np.asarray(tracks.groupby("trajectory").apply(
        lambda i: i.name)).astype(np.int64)

    # The total number of trajectories
    n_tracks = tracks["trajectory"].nunique()
    print("Number of trajectories: %d" % n_tracks)

    # The log likelihood matrix for tuple (diffusion coefficient, Hurst parameter)
    log_L = np.zeros((n_tracks, nH, nD), dtype=np.float64)

    # Evaluate the log likelihoods for each state
    for i, H in tqdm(enumerate(hurst_pars)):
        for j, D in enumerate(diff_coefs):

            # Modified diffusion coefficient
            D_mod = D / np.power(frame_interval, 2 * H - 1)

            # Determine the likelihood for each group of trajectories
            # with the same length. Here, *l* is the number of jumps
            # in each trajectory, which is one less than the trajectory
            # length (assuming we track without gaps)
            for l in range(min_jumps_per_track, max_track_len):

                # Evaluate the covariance matrix for an FBM with this 
                # diffusion coefficient and Hurst parameter
                T, S = (np.indices((l, l)) + 1) * frame_interval 
                C = D_mod * (
                    np.power(T, 2*H) + np.power(S, 2*H) - \
                        np.power(np.abs(T-S), 2*H)
                )

                # Account for localization error
                C += np.diag(np.ones(l) * le2)
                C += le2 

                # Invert the covariance matrix
                C_inv = np.linalg.inv(C)

                # Normalization factor (slogdet is the log determinant)
                norm_fac = l * np.log(2 * np.pi) + np.linalg.slogdet(C)[1]

                # Get all trajectories matching this length
                subtracks = np.asarray(
                    tracks.loc[tracks["track_length"] == l+1,
                    ["y", "x"]]
                )

                # The number of trajectories in this set
                n_match = subtracks.shape[0] // (l+1)

                # The y- and x-coordinates of each trajectory
                y_coord = subtracks[:,0].reshape((n_match, l+1)).T 
                x_coord = subtracks[:,1].reshape((n_match, l+1)).T 

                # Subtract the starting points 
                y_coord = y_coord[1:,:] - y_coord[0,:]
                x_coord = x_coord[1:,:] - x_coord[0,:]

                # Evaluate the log likelihood for the y- and x-components
                y_ll = (y_coord * (C_inv @ y_coord)).sum(axis=0)
                x_ll = (x_coord * (C_inv @ x_coord)).sum(axis=0)

                # Combine the two components
                log_L[track_lengths==(l+1), i, j] = -0.5 * (y_ll + x_ll) - norm_fac 

    # Scale by the posterior occupations, if desired
    if (not posterior is None):
        assert posterior.shape == log_L.shape 
        nonzero = posterior > 1.0e-8
        log_posterior = np.zeros(posterior.shape)
        log_posterior[nonzero] = np.log(posterior[nonzero])
        log_posterior[~nonzero] = -np.inf
        log_L += log_posterior 

    # Normalize over all states for each trajectory
    L = np.zeros(log_L.shape, dtype=np.float64)
    for t in range(n_tracks):
        log_L[t,:,:] -= log_L[t,:,:].max()
        L[t,:,:] = np.exp(log_L[t,:,:])
        L[t,:,:] /= L[t,:,:].sum()

    n_jumps = track_lengths - 1
    return L, n_jumps, track_indices

