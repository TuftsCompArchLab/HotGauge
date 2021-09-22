#!/bin/bash

################################################################################
################################# Script setup #################################
################################################################################
# exit when any command fails
set -e
# keep track of the last executed command
trap 'last_command=$current_command; current_command=$BASH_COMMAND' DEBUG
# echo an error message before exiting
trap 'echo "\"${last_command}\" command exited with exit code $?."' EXIT


################################################################################
########################## Download and Patch 3d-ice ###########################
################################################################################

[ -d "sniper" ] && \
echo "Directory for Sniper already exists! Remove it if you'd like to download a fresh copy" \
&& exit 1

mv sniper/ Sniper/

# Go into the copy of Sniper and apply the feature patch
pushd Sniper
patch -sf -p1 < ../HotGauge.Sniper.patch
popd

echo "Update complete! Navigate to the Sniper directory and run make"

