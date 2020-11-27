#!/usr/bin/env python
"""
utils.py

"""
import sys
import os
import numpy as np
import pandas as pd

def squared_jumps(tracks, n_frames=1, start_frame=None, pixel_size_um=0.16,
    pos_cols=["y", "x"]):
    """
    Given a set of trajectories, return all of the the squared jumps 
    as an ndarray.

    args
    ----
        tracks      :   pandas.DataFrame. Must contain the "trajectory"
                        and "frame" columns, along with whatever columns
                        are specified in *pos_cols*
        n_frames    :   int, the number of frame intervals over which 
                        to compute the jump. For instance, if *n_frames*
                        is 1, only compute jumps between consecutive
                        frames.
        start_frame :   int, exclude all jumps before this frame
        pixel_size_um:  float, size of pixels in um
        pos_cols    :   list of str, the columns with the spatial
                        coordinates of each detection *in pixels*

    returns
    -------
        *jumps*, a 2D ndarray of shape (n_jumps, 5+). Each row corresponds
            to a single jump from the dataset.

        The columns of *vecs* have the following meaning:
            jumps[:,0] -> length of the origin trajectory in frames
            jumps[:,1] -> index of the origin trajectory
            jumps[:,2] -> frame corresponding to the first point in 
                          the jump
            jumps[:,3] -> sum of squared jumps across all spatial dimensions
                          in squared microns
            jumps[:,4] -> Euclidean jumps in each dimension in microns

    """
    def bail():
        return np.zeros((0, 6), dtype=np.float64)

    # If passed an empty dataframe, bail
    if tracks.empty: bail()

    # Do not modify the original dataframe
    tracks = tracks.copy()

    # Calculate the original trajectory length and exclude
    # singlets and negative trajectory indices
    tracks = track_length(tracks)
    tracks = tracks[np.logical_and(
        tracks["trajectory"] >= 0,
        tracks["track_length"] > 1
    )]

    # Only consider trajectories after some start frame
    if not start_frame is None:
        tracks = tracks[tracks["frame"] >= start_frame]

    # If no trajectories remain, bail
    if tracks.empty: bail()

    # Convert from pixels to um
    tracks[pos_cols] *= pixel_size_um 

    # Work with an ndarray, for speed
    tracks = tracks.sort_values(by=["trajectory", "frame"])
    T = np.asarray(tracks[["track_length", "trajectory", "frame", pos_cols[0]] + pos_cols])

    # Allowing for gaps, consider every possible comparison that 
    # leads to the correct frame interval
    target_jumps = []
    for j in range(1, n_frames+1):
        
        # Compute jumps
        jumps = T[j:,:] - T[:-j,:]

        # Only consider vectors between points originating 
        # from the same trajectory and from the target frame
        # interval
        same_track = jumps[:,1] == 0
        target_interval = jumps[:,2] == n_frames 
        take = np.logical_and(same_track, target_interval)

        # Map the corresponding track lengths, track indices,
        # and frame indices back to each jump
        vecs[:,:3] = T[:-j,:3]
        vecs = vecs[take, :]

        # Calculate the corresponding 2D squared jump and accumulate
        if vecs.shape[0] > 0:
            vecs[:,3] = (vecs[:,4:]**2).sum(axis=1)
            target_jumps.append(vecs)

    # Concatenate
    if len(target_jumps) > 0:
        return np.concatenate(target_jumps, axis=0)
    else:
        bail()

def sum_squared_jumps(tracks, n_frames=1, start_frame=None, pixel_size_um=0.16,
    pos_cols=["y", "x"], max_jumps_per_track=None):
    """
    For each trajectory in a dataset, calculate the sum of its squared
    jumps across all spatial dimensions.

    args
    ----
        tracks          :   pandas.DataFrame. Must contain "trajectory",
                            "frame", and the contents of *pos_cols*.
        n_frames        :   int, the number of frame intervals over
                            which to compute the jumps
        start_frame     :   int, exclude jumps before this frame
        pixel_size_um   :   float, the size of pixels in microns
        pos_cols        :   list of str, the columns in *tracks* that
                            have the spatial coordinates of each 
                            detection ~in pixels~
        max_jumps_per_track :   int, the maximum number of jumps 
                            to consider from any single trajectory

    returns
    -------
        pandas.DataFrame. Each row corresponds to a trajectory, with
            the following columns:

            "sum_sq_jump": the summed squared jumps of that trajectory
                           in microns
            "trajectory" : the index of the origin trajectory
            "frame"      : the first frame of the first jumps in the 
                           origin trajectory
            "n_jumps"    : the number of jumps used in *sum_sq_jump*

    """
    out_cols = ["sum_sq_jump", "trajectory", "frame", "n_jumps"]

    # Calculate squared displacements
    jumps = squared_jumps(tracks, n_frames=n_frames, start_frame=start_frame,
        pixel_size_um=pixel_size_um, pos_cols=pos_cols)

    # If there are no jumps in this set of trajectories, bail
    if jumps.shape[0] == 0:
        return pd.DataFrame(index=[], columns=out_cols)

    # Format as a dataframe, indexed by jump
    cols = ["track_length", "trajectory", "frame", "sq_jump"] + list(pos_cols)
    jumps = pd.DataFrame(jumps, columns=cols)
    n_tracks = jump["trajectory"].nunique()

    # Output dataframe, indexed by trajectory
    sum_jumps = pd.DataFrame(index=np.arange(n_tracks), columns=out_cols)

    # Calculate the sum of squared jumps for each trajectory
    sum_jumps["sum_sq_jump"] = np.asarray(jumps.groupby("trajectory")["sq_jump"].sum())

    # Calculate the number of jumps in each trajectory
    sum_jumps["n_jumps"] = np.asarray(jumps.groupby("trajectory").size())

    # Map back the indices of the origin trajectories
    sum_jumps["trajectory"] = np.asarray(jumps.groupby("trajectory").apply(lambda i: i.name))

    # Map back the frame indices
    sum_jumps["frame"] = np.asarray(jumps.groupby("frame").first())

    return sum_jumps

def track_length(tracks):
    """
    Add a new column to a trajectory dataframe with the trajectory
    length in frames.

    args
    ----
        tracks      :   pandas.DataFrame. Must have the "trajectory"
                        column.

    returns
    -------
        pandas.DataFrame, with the "track_length" column. Overwritten
            if it already exists.

    """
    if "track_length" in tracks.columns:
        tracks = tracks.drop("track_length", axis=1)
    return tracks.join(
        tracks.groupby("trajectory").size().rename("track_length"),
        on="trajectory"
    )

def assert_gs_dp_diff_exists(defoc=True):
    """
    Raise an RuntimeError if the *gs_dp_diff* and
    *gs_dp_diff_defoc* executables do not exist.
    These are the raw Gibbs samplers.

    args
    ----
        defoc       :   bool. If *True*, look for   
                        *gs_dp_diff_defoc*. Otherwise
                        look for *gs_dp_diff*.

    """
    if defoc:
        if (not os.path.isfile("gs_dp_diff_defoc")) and \
            (os.access("gs_dp_diff_defoc", os.X_OK)):
            raise RuntimeError("could not find working " \
                "executable for gs_dp_diff_defoc")
    else:
        if (not os.path.isfile("gs_dp_diff")) and \
            (os.access("gs_dp_diff", os.X_OK)):
            raise RuntimeError("could not find working " \
                "executable for gs_dp_diff")

def format_cl_args(in_csv, out_csv, verbose=False, **kwargs):
    """
    Format a set of arguments for a CL call to *gs_dp_diff*
    or *gs_dp_diff_defoc*.

    args
    ----
        in_csv      :   str, path to a CSV with the summed
                        squared jumps and the number of jumps
                        for each trajectory
        out_csv     :   str, path to the output CSV with 
                        the posterior model
        verbose     :   str, tell the Gibbs samplers to 
                        be verbose
        kwargs      :   keyword arguments passed to the Gibbs
                        samplers

    returns
    -------
        str, a CL call to *gs_dp_diff* or *gs_dp_diff_defoc*

    """
    keymap = {
        "alpha": "a",
        "frame_interval": "t",
        "m": "m",
        "metropolis_sigma": "s",
        "B": "z",
        "n_iter": "n",
        "burnin": "b",
        "min_log_D": "c",
        "max_log_D": "d",
        "seed": "e",
        "max_occ_weight": "x",
        "loc_error": "l",
        "bias_csv": "i"
    }
    executable = "gs_dp_diff_defoc" if kwargs.get( \
        "use_defoc_likelihoods", False) else "gs_dp_diff"
    optstr = " ".join(["-{} {}".format(str(keymap.get(k)), str(kwargs.get(k))) \
        for k in kwargs.keys() if k in keymap.keys()])
    if verbose:
        return "{} {} -v {} {}".format(executable, optstr, in_csv, out_csv)
    else:
        return "{} {} {} {}".format(executable, optstr, in_csv, out_csv)

