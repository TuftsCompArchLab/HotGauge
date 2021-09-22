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

[ -d "McPAT" ] && \
echo "Directory for McPAT already exists! Remove it if you'd like to download a fresh copy" \
&& exit 1

# Get a clean copy of McPAT (v1.2)
wget https://github.com/HewlettPackard/mcpat/archive/refs/tags/v1.2.0.zip
unzip v1.2.0.zip
mv mcpat-1.2.0/ McPAT/

# Go into the copy of McPAT and apply the feature patch
pushd McPAT
patch -sf -p1 < ../HotGauge.McPAT.patch
popd

# Cleanup
rm v1.2.0.zip
echo "Update complete! Navigate to the McPAT directory and run make"

