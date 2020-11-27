#!/usr/bin/env python
"""
plot.py

"""
import sys
import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 

# Set the default font to Arial
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'Arial'

def save_png(out_png, dpi=600):
    """
    Save a matplotlib.figure.Figure to a PNG.

    args
    ----
        out_png         :   str, out path
        dpi             :   int, resolution

    """
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()
    if sys.platform == "darwin":
        os.system("open {}".format(out_png))

def diff_coef_spectrum(posterior, diff_coefs, log_scale=True,
    out_png=None, d_err=None, ylim=None, axes=None):
    """
    Make a plot of the posterior distribution over the 
    diffusion coefficient. 

    args
    ----
        posterior       :   1D ndarray of shape (n_bins,),
                            the posterior density
        diff_coefs      :   1D ndarray of shape (n_bins+1,),
                            the edges of each diffusion
                            coefficient bin in squared microns
                            per second
        log_scale       :   bool, use a logarithmic scale for 
                            the diffusion coefficient
        out_png         :   str, save path for figure
        d_err           :   float, the apparent speed of immobile
                            objects due to localization error
        ylim            :   (float, float), the y-axis limits
        axes            :   matplotlib.axes.Axes, the axis to 
                            use for the plot. If not given, a 
                            new axis is generated.

    returns
    -------
        If *out_png* is set, saves directly to a PNG.

        Otherwise returns
        (
            matplotlib.figure.Figure,
            matplotlib.axes.Axes
        )

    """
    fontsize = 12

    # Figure layout
    if axes is None:
        fig, axes = plt.subplots(figsize=(4, 2))

    # Use either the midpoint or the geometric mean of each 
    # bin, depending on whether we're working in linear or 
    # log space
    if log_scale:
        d = np.sqrt(diff_coefs[1:] * diff_coefs[:-1])
    else:
        d = 0.5 * (diff_coefs[1:] * diff_coefs[:-1])

    # Plot the posterior density
    axes.plot(d, posterior, color="k", linestyle="-", label=None)

    # Make a vertical dotted line at a user-defined location
    if not d_err is None:
        axes.plot([d_err, d_err], [0, posterior.max()], color="k",
            linestyle="--", label="$\\sigma_{loc}^{2}$ s$^{-1}$")
        axes.legend(frameon=False, loc="upper right", 
            prop={"size": 8})

    # Log scale
    if log_scale:
        axes.set_xscale("log")

    # Axis labels
    axes.set_xlabel("Diffusion coefficient ($\mu$m$^{2}$ s$^{-1}$)",
        fontsize=fontsize)
    axes.set_ylabel("Posterior density", fontsize=fontsize)

    # Limit the y-axis 
    if not ylim is None:
        axes.set_ylim(ylim)
    else:
        axes.set_ylim((0, axes.get_ylim()[1]))

    # Save, if desired
    if not out_png is None:
        save_png(out_png, dpi=800)
    else:
        return fig, axes 
