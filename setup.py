#!/usr/bin/env python
"""
setup.py

"""
import os
import setuptools

# This repository's directory
REPODIR = os.path.abspath(os.path.dirname(__file__))

# User's home directory
HOMEDIR = os.path.expanduser("~")

# Directory for binaries
BINDIR = os.path.join(REPODIR, "bin")

# Source code for Dirichlet process Gibbs samplers
SRCDIR = os.path.join(REPODIR, "dpsp", "cpp")

# Target executables
EXECS = ["gsdpdiff", "gsdpdiffdefoc"]

# Potential user .bashrc files
BASHRCS = [os.path.join(HOMEDIR, f) for f in [".bash_profile", ".bashrc"]]
PATHLINE = '\nexport PATH=$PATH:"{}"'.format(BINDIR)

# Try to compile the Gibbs sampler binaries
GSDPDIFF_INSTALLED = False
try:

    # Make directory for binaries if it doesn't already exist
    if not os.path.isdir(BINDIR):
        os.mkdir(BINDIR)

    # Compile
    os.system("make -f {}".format(os.path.join(SRCDIR, "makefile")))
    for _exec in EXECS:
        os.rename(
            os.path.join(SRCDIR, _exec),
            os.path.join(BINDIR, _exec)
        )

    # Add the binary directory to PATH
    for bashrc in BASHRCS:
        if os.path.isfile(bashrc):
            with open(bashrc, "r") as f:
                flines = f.read()
            if not PATHLINE.replace("\n", "") in flines:
                with open(bashrc, "a") as f:
                    f.write(PATHLINE)
    GSDPDIFF_INSTALLED = True
except:
    print("WARNING: gsdpdiff not installed")

# Install Python library
setuptools.setup(
    name="dpsp", 
    version="1.0",
    author="Alec Heckert",
    author_email="aheckert@berkeley.edu",
    description="Dirichlet process mixture models for single particle tracking",
    packages=setuptools.find_packages()
)

# User message
if GSDPDIFF_INSTALLED:
    print("\nIMPORTANT:\n" \
        "Successfully installed gsdpdiff and gsdpdiffdefoc.\n" \
        "If you want to run Dirichlet process mixture models, check\n" \
        "that these executables exist by doing the following:\n" \
        "\n\t1. Open a new terminal.\n" \
        "\t2. Enter 'gsdpdiff' or 'gsdpdiffdefoc'. If configured\n" \
        "\tcorrectly, a docstring should print to the terminal.\n" \
        "\nIf you get an 'executable not found', add the executables\n" \
        "at dpsp/bin to your $PATH.\n")
else:
    print("\n\nWARNING:\nNecessary binaries gsdpdiff and gspddiffdefoc\n" \
        "NOT installed. Both source code files at dpsp/dpsp/cpp will\n" \
        "need to be compiled (with -std>=c++11) and placed in the PATH\n" \
        "before running any Dirichlet processes.\n")
