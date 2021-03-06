# HotGauge
A framework for characterizing hotspots in next-generation processors

[[_TOC_]]

# Docker Setup
**HotGauge** can be tested and run inside a Docker container. Simply run `./docker_build.sh`
followed by `./docker_run.sh` on a machine with Docker installed. Then follow the instructions below
for "Subsequent Use" below.

# Local Machine Setup
This codebase requires python 3 and was developed using python 3.4

## Initial Setup
1) Clone this repository
1) Set up a virtual-environment
   1) Create the virtual-environment: `python -m venv env`
   1) Activate the virtual-environment
      * `source env/bin/activate` if using `bash`
      * `source env/bin/activate.csh` if using `csh`
      * `source env/bin/activate.fish` if using `fish`
   1) Update pip `pip install --update pip` 
   1) Install required modules: `pip install -r requirements.txt`
1) Set up Sniper (the performance simulator)
    1) Clone the Sniper (the performance simulator) git repository by following the instructions on https://snipersim.org/w/Getting_Started 
    1) Use `cd sniper && git log` to checkout the commit for version 6.1
    1) Apply the patch using `patch_Sniper.sh`
1) Set up McPAT (the power simulator)
    1) Download and patch McPAT (the power simulator) using `get_and_patch_McPAT.sh`
1) Set up 3d-ice (the thermal simulator)
    1) Download and patch 3d-ice (the thermal simulator) using `get_and_patch_3DICE.sh`
    1) Compile and test 3d-ice
        1) make sure compatible version of bison, flex, and superLU are installed. See compilation
        hints below for more details
        1) Compile the heatsink_plugin using `make` in *./3d-ice/heatsink_plugin/*
        1) Compile 3D-ICE executables using `make` in *./3d-ice/*
        1) Test 3D-ICE executables using `make test`
1) Try out the scripts in the *./examples/* directory. Running `python simulation_with_warmup.py` in
the `examples/` directory is a good way to test HotGauge. (hint: make sure you have your python
virtual-environment activated!)
1) Utilize **HotGauge** as you see fit!

## Subsequent Use
1) Activate the environment: `source env/bin/activate`

# Sniper Compilation Hints

## Requirments

* Python 2

# 3D-ICE Compilation Hints

## Requirments
As per the documentation, 3D-ICE was developed using the following versions of its dependancies

* gcc 7.4.0
* bison 3.0.4
* flex 2.6.4
* blas 3.7.1
* SuperLU 4.3

The pluggable heat-sink interface---which is used by HotGauge---has its own set of  dependancies:

* OpenModelica 1.16.0
* Pugixml 1.8.1
* Python 3 header files
* pkg-config


## HotGauge Development Environment (RHEL 6.10)
HotGauge was developed and tested on a server running Red Hat Enterprise Linux Server release 6.10
(RHEL 6).  The development machine had the following versions of 3D-ICE's dependancies:

* gcc 4.8.0
* bison 2.4.1
* flex 2.5.35
* blas 3.2.1
* SuperLU 4.3
* pkg-config 0.23

### HotGauge Example Patch
This repository also comes with a [second patch file](HotGauge.3D-ICE.ThermalInit.RHEL_6.patch) that
enables compilation on the development server. This patch must be applied *after* the [base HotGauge
patch file](HotGauge.3D-ICE.ThermalInit.patch) While this patch file may not be *exactly* the
changes you will need to make to get 3D-ICE installed on your system, it is included with the hope
that it will substantially help you overcome the compilation issues we encountered, should you also
encounter them. The details of the changes made by this patch file are given below.

#### Local copy of OpenModelica
A script was created to download and compile a local copy of OpenModelica (omc). In order to get omc
to compile, a few changes needed to be made, including the removal of calls to asciidoc, which was
an incompatible version.

#### Download source code for pugixml
A script was created to download the source files for pugixml and add their compilation and
inclusion to the required Makefile.

#### Disable installation of SuperLU
The development machine already has a compatible version of SuperLU, so the patch changes the
SuperLU installation file to disable installation.

#### Disable heatsink-plugin python loader
The python loader for the heatsink-plugin also had to be disabled due to a python version issue.

# Repository Contents
The following directories contain the code developed for use with this project. Each directory has a
different purpose, as described below.

## examples
This directory contains some example scripts that do various tasks such as generating floorplans as
well as running thermal-simulations and doing subsequent analyis.

## 3d-ice
This is where the thermal simulator, 3d-ice, is compiled. HotGauge uses a custom version of 3d-ice,
which is generated by apply the supplied [patch file](HotGauge.3D-ICE.ThermalInit.patch).
## HotGauge
The python package for HotGauge. This includes scripts to run simulations and process the outputs.
