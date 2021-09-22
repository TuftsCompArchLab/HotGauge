#!/bin/bash
# Sniper sims are all 14nm; 10nm and 7nm stats are copies of the 14nm stats files
# MCPAT is run for each node

# Make sure one agrument is passed: root directory where the 14nm sims directory can be found
[ $# -ne 1 ] && { echo "Usage $0 sim_dir" ; exit 1 ;}

# Make sure the directory exists
[ ! -d $1 ] && { echo "$1 doesn\'t exist" ; exit 1 ;}

# The 14nm sims should be in a 14nm directory within the root directory
BASE_SIM_DIR_14nm="$1/14nm"
SIM_DIRS_14nm=`ls -1d ${BASE_SIM_DIR_14nm}/*/`
[ -z "$SIM_DIRS_14nm" ] && {  echo "$BASE_SIM_DIR_14nm has no sub-directories.. sims missing..."; exit 1;}

for sim_dir_14nm in $SIM_DIRS_14nm; do
    echo $sim_dir_14nm
    for node in 7 10; do
        sim_dir_other_nm=${sim_dir_14nm/14nm/${node}nm}
        mkdir -p ${sim_dir_other_nm}
        cp ${sim_dir_14nm}/energystats-temp-* ${sim_dir_other_nm}
        sed -i "s/\"core_tech_node\" value=\"14\"/\"core_tech_node\" value=\"${node}\"/" ${sim_dir_other_nm}/energystats-temp-*
    done
done

SIM_DIRS_ALL=`ls -1d $1/*nm/*`
for node in 7 10 14; do
    echo "Running ${node}nm sims..."
    SIM_DIRS=`ls -1d $1/${node}nm/*`
    for sim_dir in $SIM_DIRS; do
        python run_mcpat.py ${sim_dir} # Don't parallelize this one... it's already paralelized
    done
    echo "MCPAT done"
    
    for sim_dir in $SIM_DIRS; do
        python mcpat_txt_to_json.py ${sim_dir} &
    done
    wait
    echo "MCPAT txt to json done"

    for sim_dir in $SIM_DIRS; do
        python mcpat_to_blk_lvl_power_dict.py ${sim_dir} # Don't parallelize this one... it's already paralelized
    done
    echo "MCPAT json blk_lvl_power_dict done"
done

exit 0
