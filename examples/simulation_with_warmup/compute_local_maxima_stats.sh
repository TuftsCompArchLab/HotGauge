#!/bin/bash
# This script does analysis on the grid output of 3D-ICE simulations.  It finds local maxima and
# records stats for each (e.g. temperature, MLTD).  These local maxima are 'hotspot candidates'.

set -e # exit if a command fails

# Define the plot command
PLT_CMD='python -m HotGauge.thermal.analysis local_max_stats'

# Print the help message to see command doumentation and options
$PLT_CMD --help

# Configure the directories
SIM_DIR=outputs/sim/

################################################################################
########################### Do the hotspot analysis ############################
################################################################################
# Navigate to the simulation directory
cd $SIM_DIR

echo "Starting analysis..."

# Get the stats for local maxima, save as a text file, pickle(pandas.DataFrame), and csv
$PLT_CMD die_grid.temps 20 -o die_grid.temps.2dmaxima -o die_grid.temps.2dmaxima.pkl -o die_grid.temps.2dmaxima.csv
# Get the stats for maxima in either dimension
$PLT_CMD die_grid.temps 20 -o die_grid.temps.1dmaxima --in_either_dimension

echo "Analysis complete!"
