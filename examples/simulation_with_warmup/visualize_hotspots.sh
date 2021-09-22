#!/bin/bash
# This script creates thermal images from the output of a 3D-ICE thermal simulation and uses the
# outputs of compute_local_maxima_stats.sh to identify and label hotspot locations.

set -e # exit if a command fails

# Define the plot command
PLT_CMD='python -m HotGauge.visualization.ICE_plt hotspot_locations'

# Configure the directories
SIM_DIR=outputs/sim/
PLT_DIR=plots

# Navigate to the simulation directory
cd $SIM_DIR
mkdir -p $PLT_DIR

################################################################################
######################## Make the thermal image frames #########################
################################################################################
$PLT_CMD die_grid.temps IC.flp 1.0 -o $PLT_DIR/ttrace_{step:04}.png -l die_grid.temps.2dmaxima

################################################################################
################### Convert the images into a video (mp4) ####################
################################################################################
ffmpeg -y -i plots/ttrace_%04d.png -q:v 3 plots/ttrace.mp4
