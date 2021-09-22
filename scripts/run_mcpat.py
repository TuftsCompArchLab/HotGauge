#!/usr/bin/env python
import os
import shutil
import sys
import glob

##
# Hotspot Project
# run_mcpat.py: script to run Harvard-updated McPAT on .xml files from different time steps
##


rundir = sys.argv[1]

# added by maz
if rundir[-1] != '/':
    rundir += '/'

# Collect unnecessary files
txtfiles = glob.glob(rundir + 'energystats-temp*.txt')
pyfiles = glob.glob(rundir + 'energystats-temp*.py')
cfgfiles =  glob.glob(rundir + 'energystats-temp*.cfg')
files_to_delete = txtfiles + pyfiles + cfgfiles

# Delete the unnecessary files
for filePath in files_to_delete:
    try:
        os.remove(filePath)
    except:
        print("Error while deleting file : ", filePath)

# Move old mcpat_commands file
fp_commands_file = os.path.join(rundir, "mcpat_commands.txt")
if os.path.exists(fp_commands_file):
    fp_commands_file_bak = os.path.join(rundir, "mcpat_commands_old.txt")
    shutil.move(fp_commands_file, fp_commands_file_bak)

with open(fp_commands_file, "a") as f:

    # Loop over all .xml files (McPAT inputs)
    for fname in os.listdir(rundir):
        if fname.endswith(".xml"):
            # Extract time step from file name
            time_step = fname.split('-')[2].split('.')[0]
            # Form McPAT command for the file
            # Example syntax: /r/tcal/archgroup/archtools/sims/Harvard_McPAT/hpca2015_release/v1.2/mcpat -infile <file.xml> -print_level 5 > mcpat_output_<time_step>.txt
            command = "../McPAT/mcpat -infile " + rundir + fname + " -print_level 5 > " + rundir + "mcpat_output_" + time_step + ".txt\n"
            f.write(command)

# Use GNU parallel to run McPAT on all .xml files in parallel
os.system("parallel < " + rundir + "mcpat_commands.txt")
