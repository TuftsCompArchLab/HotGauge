#!/bin/bash
# This script plots temperature and power vs time. The first plot type shows the min/mean/max value
# of each vs time as a line and the second is the distribution of each vs time.

set -e # exit if a command fails

# Define the plot command
PLT_CMD='python -m HotGauge.visualization.ICE_plt grid_transient'

# Configure the directories
SIM_DIR=outputs/sim/
PLT_DIR=plots

# Navigate to the simulation directory
cd $SIM_DIR
mkdir -p $PLT_DIR

################################################################################
###################### Plot the temperature data vs time #######################
################################################################################
temp_kwargs=" --data_type temperature --min_val 25 --max_val 135 -t 110"
$PLT_CMD die_grid.temps --plot_type stats $temp_kwargs --output $PLT_DIR/temperature_stats.png
$PLT_CMD die_grid.temps --plot_type dist  $temp_kwargs --output $PLT_DIR/temperature_dist.png

################################################################################
######################### Plot the power data vs time ##########################
################################################################################
power_kwargs=" --data_type power"
$PLT_CMD die_grid.pows --plot_type stats $power_kwargs --output $PLT_DIR/power_stats.png
$PLT_CMD die_grid.pows --plot_type dist  $power_kwargs --output $PLT_DIR/power_dist.png
