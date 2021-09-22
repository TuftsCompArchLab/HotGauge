#!/bin/bash
# This script creates a transient visualization of the power map that was supplied to 3D-ICE.

set -e # exit if a command fails

# Define the plot command
PLT_CMD='python -m HotGauge.visualization.ICE_plt power_map'

# Configure the directories
SIM_DIR=outputs/sim/
PLT_DIR=plots

# Navigate to the simulation directory
cd $SIM_DIR
mkdir -p $PLT_DIR

################################################################################
########################## Make the power map images ###########################
################################################################################
$PLT_CMD die_grid.pows IC.flp -o $PLT_DIR/ptrace_{step:04}.png

################################################################################
################### Convert the images into a video (mp4) ####################
################################################################################
ffmpeg -y -i plots/ptrace_%04d.png -q:v 3 plots/ptrace.mp4
