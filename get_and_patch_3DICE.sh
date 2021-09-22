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

[ -d "3d-ice" ] && \
echo "Directory for 3d-ice already exists! Remove it if you'd like to download a fresh copy" \
&& exit 1

# Get a clean copy of 3d-ice (specific commit)
wget https://github.com/esl-epfl/3d-ice/archive/1ce8c0ad00b96acae52fc83508c61e3b4ad612df.zip
unzip 1ce8c0ad00b96acae52fc83508c61e3b4ad612df.zip
mv 3d-ice-1ce8c0ad00b96acae52fc83508c61e3b4ad612df/ 3d-ice/

# Go into the copy of 3d-ice and apply the feature patch
pushd 3d-ice
patch -p1 < ../HotGauge.3D-ICE.ThermalInit.patch
popd

# Cleanup
rm 1ce8c0ad00b96acae52fc83508c61e3b4ad612df.zip
echo "Update complete! Navigate to the 3d-ice directory and follow the compilation instructions"

# TODO: assist with possible compilation issues, with RHEL_6 patch as example
