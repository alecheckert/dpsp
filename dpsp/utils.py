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
        jumps[:,:3] = T[:-j,:3]
        jumps = jumps[take, :]

        # Calculate the corresponding 2D squared jump and accumulate
        if jumps.shape[0] > 0:
            jumps[:,3] = (jumps[:,4:]**2).sum(axis=1)
            target_jumps.append(jumps)

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
    n_tracks = jumps["trajectory"].nunique()

    # Limit the number of jumps to consider per trajectory, if desired
    if not max_jumps_per_track is None:
        jumps = assign_index_in_track(jumps)
        tracks = jumps[jumps["index_in_track"] >= max_jumps_per_track]

    # Output dataframe, indexed by trajectory
    sum_jumps = pd.DataFrame(index=np.arange(n_tracks), columns=out_cols)

    # Calculate the sum of squared jumps for each trajectory
    sum_jumps["sum_sq_jump"] = np.asarray(jumps.groupby("trajectory")["sq_jump"].sum())

    # Calculate the number of jumps in each trajectory
    sum_jumps["n_jumps"] = np.asarray(jumps.groupby("trajectory").size())

    # Map back the indices of the origin trajectories
    sum_jumps["trajectory"] = np.asarray(jumps.groupby("trajectory").apply(lambda i: i.name))

    # Map back the frame indices
    sum_jumps["frame"] = np.asarray(jumps.groupby("trajectory")["frame"].first())

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

def assign_index_in_track(tracks):
    """
    Given a set of trajectories, determine the index of each localization in the
    context of its respective trajectory.

    args
    ----
        tracks      :   pandas.DataFrame, containing the "trajectory" and "frame"
                        columns

    returns
    -------
        pandas.DataFrame, the same dataframe with a new column, "index_in_track"

    """
    tracks["one"] =  1
    tracks["index_in_track"] = tracks.groupby("trajectory")["one"].cumsum() - 1
    tracks = tracks.drop("one", axis=1)
    return tracks 

def concat_tracks(*tracks):
    """
    Join some trajectory dataframes together into a larger dataframe,
    while preserving uniqe trajectory indices.

    args
    ----
        tracks      :   pandas.DataFrame with the "trajectory" column

    returns
    -------
        pandas.DataFrame, the concatenated trajectories

    """
    n = len(tracks)

    # Sort the tracks dataframes by their size. The only important thing
    # here is that if at least one of the tracks dataframes is nonempty,
    # we need to put that one first.
    df_lens = [len(t) for t in tracks]
    try:
        tracks = [t for _, t in sorted(zip(df_lens, tracks))][::-1]
    except ValueError:
        pass

    # Iteratively concatenate each dataframe to the first while 
    # incrementing the trajectory index as necessary
    out = tracks[0].assign(dataframe_index=0)
    c_idx = out["trajectory"].max() + 1

    for t in range(1, n):

        # Get the next set of trajectories and keep track of the origin
        # dataframe
        new = tracks[t].assign(dataframe_index=t)

        # Ignore negative trajectory indices (facilitating a user filter)
        new.loc[new["trajectory"]>=0, "trajectory"] += c_idx 

        # Increment the total number of trajectories
        c_idx = new["trajectory"].max() + 1

        # Concatenate
        out = pd.concat([out, new], ignore_index=True, sort=False)

    return out

def concat_tracks_files(*csv_paths, out_csv=None, start_frame=0,
    drop_singlets=False):
    """
    Given a set of trajectories stored as CSVs, concatenate all
    of them, storing the paths to the original CSVs in the resulting
    dataframe, and optionally save the result to another CSV.

    args
    ----
        csv_paths       :   list of str, a set of trajectory CSVs.
                            Each must contain the "y", "x", "trajectory",
                            and "frame" columns
        out_csv         :   str, path to save to 
        start_frame     :   int, exclude any trajectories that begin before
                            this frame
        drop_singlets   :   bool, drop singlet localizations before
                            concatenating

    returns
    -------
        pandas.DataFrame, the concatenated result

    """
    n = len(csv_paths)

    def drop_before_start_frame(tracks, start_frame):
        """
        Drop all trajectories that start before a specific frame.

        """
        tracks = tracks.join(
            (tracks.groupby("trajectory")["frame"].first() >= start_frame).rename("_take"),
            on="trajectory"
        )
        tracks = tracks[tracks["_take"]]
        tracks = tracks.drop("_take", axis=1)
        return tracks

    def drop_singlets_dataframe(tracks):
        """
        Drop all singlets and unassigned localizations from a 
        pandas.DataFrame with trajectory information.

        """
        if start_frame != 0:
            tracks = drop_before_start_frame(tracks, start_frame)

        tracks = track_length(tracks)
        tracks = tracks[np.logical_and(tracks["track_length"]>1,
            tracks["trajectory"]>=0)]
        return tracks 

    # Load the trajectories into memory
    tracks = []
    for path in csv_paths:
        if drop_singlets:
            tracks.append(drop_singlets_dataframe(pd.read_csv(path)))
        else:
            tracks.append(pd.read_csv(path))

    # Concatenate 
    tracks = concat_tracks(*tracks)

    # Map the original path back to each file
    for i, path in enumerate(csv_paths):
        tracks.loc[tracks["dataframe_index"]==i, "source_file"] = \
            os.path.abspath(path)

    # Optionally save concatenated trajectories to a new CSV
    if not out_csv is None:
        tracks.to_csv(out_csv, index=False)

    return tracks 

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

