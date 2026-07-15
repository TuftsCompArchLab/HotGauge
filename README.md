# HotGauge
A framework for characterizing hotspots in next-generation processors

[[_TOC_]]

# Local Machine Setup
This codebase requires python 3 and was developed using python 3.9

# Installing System Dependencies
Follow the instructions below to install the system dependencies. You can run HotGauge without installing these dependencies system-wide by following the ddockerization instructions

1) dnf -y install dnf-plugins-core and epel-release
1) dnf config-manager --set-enabled crb
1) dnf config-manager --add-repo https://build.openmodelica.org/linux/rpm/el9/omc.repo
1) Install the packages from the next section
1) chmod -R a+rX /opt/openmodelica-nightly/share/omc/runtime/c/fmi/buildproject/

# Dockerization Instructions
Make sure Docker is installed and running.

From the HotGauge directory, build the Docker image:

docker build -t my-project .

Then start a container from the image:

docker run --rm -it HotGauge_docker bash

Contnue with the build instructions in the Initial Setup portion of the README

# System Dependencies (RHEL 9)

The following packages must be installed before using HotGauge.

| Package            | Required Version       | Notes |
|--------------------|------------------------|------|
| Dnf-plugins-core   | 4.3.0                  | Required for downloading other packages |
| Epel-release       | 9-10.el9               | Required for downloading other packages |
| python3            | ≥ 3.9                  | Newer Scripts developed using Python 3.9, older ones with 3.4 |
| python3-devel      | Matching Python3       | Needed for building Sniper |
| gcc                | ≥ 7.4 (tested 11.5)    | Required for Sniper + 3D-ICE |
| gcc-c++            | ≥ 7.4                  | Required for Sniper + 3D-ICE |
| make and cmake     | Any recent             | Build tools |
| bison              | ≥ 3.0 (tested 3.7.4)   | Required for 3D-ICE |
| flex               | ≥ 2.6                  | Required for 3D-ICE |
| pkg-config         | Any recent             | Used in builds |
| boost-devel        | Any recent             | Sniper dependency |
| sqlite-devel       | Any recent             | Sniper dependency |
| zlib-devel         | Any recent             | Sniper dependency |
| openblas-devel     | ≥ 0.3 expected to work | BLAS backend |
| openblas-openmp    | Any compatible         | Parallel BLAS |
| bzip2-devel        | Any recent             | Build dependency |
| xz                 | Any recent             | Needed for archive extraction |
| wget and unzip and zip | Any recent             | Utility tools |
| csh                | Any recent             | Required by Sniper scripts |
| parallel           | Any recent             | Used in workflows |
| pugixml            | ≥ 1.8                  | Required for 3D-ICE plugin |
| pugixml-devel      | ≥ 1.8                  | Headers |
| openmodelica-nightly       | ≥ 1.16                 | Required for heat-sink model |
| ffmpeg-free-devel.x86_64             | Any Recent             | Required for analyis script |
| glibc-devel.i686   | Any Recent             | Required for McPAT |
| libstdc++.i686     | Any Recent             | Required for McPAT |
| libgcc.i686        | Any Recent             | Required for McPAT |


# Version Compatibility Notes

HotGauge integrates several external tools (Sniper, McPAT, 3D-ICE) that were originally developed with older dependencies.

- 3D-ICE was originally built with:
  - gcc 7.4.0
  - bison 3.0.4
  - flex 2.6.4
  - BLAS 3.7.1

- This repository has been successfully tested on:
  - gcc 11.5.0
  - bison 3.7.4
  - flex 2.6.4
  - OpenBLAS 0.3.26

In general:
- Newer versions of these dependencies work correctly
- No strict version pinning is required
- If build issues occur, they are most likely due to missing packages rather than version incompatibility

## Initial Setup
1. Clone this repository
2. Set up a virtual-environment
   1. Create the virtual-environment: `python3 -m venv env`
   2. Activate the virtual-environment
      * `source env/bin/activate` if using `bash`
      * `source env/bin/activate.csh` if using `csh`
      * `source env/bin/activate.fish` if using `fish`
   3. Update pip `python -m pip install --upgrade pip`
   4. Install required modules: `pip install -r requirements.txt`
3. Set up Sniper (the performance simulator)
   1. Clone snipersim: `https://github.com/snipersim/snipersim.git`
   2. Enter the new directory
   3. Check out the most recent commit before February 28, 2026: `git checkout $(git rev-list -n 1 --before="2026-02-28" HEAD)`
   4. Create a named branch from that historical version before applying patches: `git switch -c snipersim-feb-2026-patched`
   5. Set the `SNIPER_ROOT` environment variable using: `export SNIPER_ROOT="$PWD"`
   6. Apply the patch: `patch -p1 < ~/HotGauge/RHEL9_patches/sniper_rhel9_delta.patch`
   7. Run: `make`
4. Set up McPAT (the power simulator)
   1. Re-enter the HotGauge root directory and download and patch McPAT using: `get_and_patch_McPAT.sh`
   2. Change into the HotGauge/McPAT directory and run: `make`
   3. Make the McPAT Binary executable: `chmod u+x ~/HotGauge/McPAT/mcpat`
      * If you make clean and then make again it will likely not work. In the event that you run make clean, just delete the McPAT directory entirely and start over at the beginning of this step
5. Set up 3D-ICE (the thermal simulator)
   1. Return to the HotGauge root directory and download and patch 3d-ice using: `get_and_patch_3DICE.sh`
   2. Enter the 3d-ice directory and run: `./install-superlu.sh`
      * At the end of the installation there will be a bunch of small tests that superlu runs and they may fail due to segmentation faults. Do not worry about this, the build was successful.
   3. Enter into the `SuperLU_4.3` directory and apply the patch: `patch -p1 < ~/HotGauge/RHEL9_patches/supLU_rhel9_delta.patch`
   4. Compile the heatsink plugin using `make` in `/3d-ice/heatsink_plugin/`
      * This can take 30 minutes to an hour
   5. Compile 3D-ICE executables using `make` in `./3d-ice/`
   6. Test by navigating to `~/HotGauge/examples` and running: `Python custom_simulation_with_warmup.py`
      * You may need to go into the python file and then uncomment and adjust the sys path insert line
      * Note: This script can take up to 2 days to run. You can also test the system by following the instructions in `use_instructions.pdf`
6. Utilize **HotGauge** as you see fit!
    
## Subsequent Use
1) Activate the environment: `source env/bin/activate`
2) Refer to `use_instructions.pdf` in this directory for a description of how to use HotGauge


# Repository Contents
The following directories contain the code developed for use with this project. Each directory has a
different purpose, as described below.

## Examples
This directory holds HotGauge outputs, scripts to run the 3D-ICE and analysis parts of HotGauge, 
and sample sniper configuration and 3D-ICE warmup files.

## Scripts
This is where the scripts that feed Sniper Output into McPAT live.

## End to End
This directory holds a script that runs HotGauge from sniper all the way through 3D-ice. It also holds
a template configuration file that the end to end script takes as input.

## HotGauge
The python package for HotGauge. This includes scripts to run simulations and process the outputs.

## RHEL9 patches
The patch files that modify the tools in HotGauge such that they can run on RHEL9.

## ThermalSideChannelAnalysisTools
Python scripts that help with visualizing hotspots
