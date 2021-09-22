#!/usr/bin/env python
import sys
import os
import glob
import multiprocessing
import click
import tqdm
import csv
from collections import defaultdict
import json
import pprint


def single_mcpat_to_blk_lvl_power_dict(mcpat_json_file, write=True):
    # Alex Hankin

    # Objective: Parse power model output (json format) and create input format for thermal simulator (python dictionary)

    # Setup pretty printing for easier debugging/visualization
    pp = pprint.PrettyPrinter(indent=4)

    # Read in json file
    with open(mcpat_json_file, 'r') as f:
        mcpat_output_dict = json.load(f)
    #pp.pprint(mcpat_output_dict)

    # Initialize number of cores
    num_cores = 8

    # Create list of FUs/Memories to parse for ... PER CORE
    # Note: Execution Unit (just parent unit) has been taken out due to it missing gate leakage data... need to
    # investigate. Same for Memory Management Unit (just parent unit). Should we sum up children gate
    # leakages?
    per_core_units = ['Execution Unit/Complex ALUs', 'Execution Unit/Floating Point Units', 'Execution Unit/Instruction Scheduler/FP Instruction Window', 'Execution Unit/Instruction Scheduler/Instruction Window', 'Execution Unit/Instruction Scheduler/ROB', 'Execution Unit/Instruction Scheduler', 'Execution Unit/Integer ALUs', 'Execution Unit/Register Files/Floating Point RF', 'Execution Unit/Register Files', 'Execution Unit/Register Files/Integer RF', 'Execution Unit/Results Broadcast Bus', 'Instruction Fetch Unit/Branch Predictor/Chooser', 'Instruction Fetch Unit/Branch Predictor/Global Predictor', 'Instruction Fetch Unit/Branch Predictor', 'Instruction Fetch Unit/Branch Predictor/L1_Local Predictor', 'Instruction Fetch Unit/Branch Predictor/L2_Local Predictor', 'Instruction Fetch Unit/Branch Predictor/RAS', 'Instruction Fetch Unit/Branch Target Buffer', 'Instruction Fetch Unit', 'Instruction Fetch Unit/Instruction Buffer', 'Instruction Fetch Unit/Instruction Cache', 'Instruction Fetch Unit/Instruction Decoder', 'Instruction Fetch Unit', 'L2', 'Load Store Unit/Data Cache', 'Load Store Unit/LoadQ', 'Load Store Unit', 'Load Store Unit/StoreQ', 'Memory Management Unit/Dtlb', 'Memory Management Unit/Itlb', 'Renaming Unit', 'Renaming Unit/FP Front End RAT', 'Renaming Unit/Free List', 'Renaming Unit/Int Front End RAT']

    # Create list of units for parse for ... PER PROCESSOR (1 processor in this case)
    per_processor_units = ['Total Cores', 'Total L3s', 'Total NoCs']

    # Initialize dict which will be input to thermal simulator
    thermal_input_dict = defaultdict(lambda: 'N/A')

    # Gets total power for specified unit within core
    def get_per_core_total_power(mcpat_output_dict, unit_name, core_num):
        runtime_dynamic = mcpat_output_dict['Core'][core_num][unit_name + '/Runtime Dynamic']
        subthreshold_leakage = mcpat_output_dict['Core'][core_num][unit_name + '/Subthreshold Leakage']
        gate_leakage = mcpat_output_dict['Core'][core_num][unit_name + '/Gate Leakage']
        total_dynamic_power = 2*runtime_dynamic
        total_leakage_power = subthreshold_leakage + gate_leakage
        thermal_input_dict['Core' + str(core_num) + '/' + unit_name] = [total_dynamic_power, total_leakage_power]

    # Gets total power for specified unit within processor
    def get_per_processor_total_power(mcpat_output_dict, unit_name):
        runtime_dynamic = mcpat_output_dict['Processor'][unit_name + '/Runtime Dynamic']
        subthreshold_leakage = mcpat_output_dict['Processor'][unit_name + '/Subthreshold Leakage']
        gate_leakage = mcpat_output_dict['Processor'][unit_name + '/Gate Leakage']
        total_power = runtime_dynamic + subthreshold_leakage + gate_leakage
        total_leakage_power = subthreshold_leakage + gate_leakage
        thermal_input_dict['Processor' + '/' + unit_name] = [runtime_dynamic, total_leakage_power]

    # Gets total power for core (parent unit) and children
    for core_num in range(num_cores):
        runtime_dynamic = mcpat_output_dict['Core'][core_num]['Runtime Dynamic']
        subthreshold_leakage = mcpat_output_dict['Core'][core_num]['Subthreshold Leakage']
        gate_leakage = mcpat_output_dict['Core'][core_num]['Gate Leakage']
        total_leakage_power = subthreshold_leakage + gate_leakage
        thermal_input_dict['Core' + str(core_num)] = [runtime_dynamic, total_leakage_power]

        for unit in per_core_units:
            get_per_core_total_power(mcpat_output_dict, unit, core_num)

    # Gets total power for BUSES
    runtime_dynamic = mcpat_output_dict['BUSES']['Runtime Dynamic']
    subthreshold_leakage = mcpat_output_dict['BUSES']['Subthreshold Leakage']
    gate_leakage = mcpat_output_dict['BUSES']['Gate Leakage']
    total_leakage_power = subthreshold_leakage + gate_leakage
    thermal_input_dict['BUSES'] = [runtime_dynamic, total_leakage_power]
    runtime_dynamic = mcpat_output_dict['BUSES']['Bus/Runtime Dynamic']
    subthreshold_leakage = mcpat_output_dict['BUSES']['Bus/Subthreshold Leakage']
    gate_leakage = mcpat_output_dict['BUSES']['Bus/Gate Leakage']
    total_leakage_power = subthreshold_leakage + gate_leakage
    thermal_input_dict['BUSES/Bus'] = [runtime_dynamic, total_leakage_power]

    # Gets total power for NUCA
    runtime_dynamic = mcpat_output_dict['NUCA'][0]['Runtime Dynamic']
    subthreshold_leakage = mcpat_output_dict['NUCA'][0]['Subthreshold Leakage']
    gate_leakage = mcpat_output_dict['NUCA'][0]['Gate Leakage']
    total_leakage_power = subthreshold_leakage + gate_leakage
    thermal_input_dict['NUCA'] = [runtime_dynamic, total_leakage_power]

    # Gets total power for Processor (parent unit) and children
    runtime_dynamic = mcpat_output_dict['Processor']['Runtime Dynamic']
    subthreshold_leakage = mcpat_output_dict['Processor']['Subthreshold Leakage']
    gate_leakage = mcpat_output_dict['Processor']['Gate Leakage']
    total_leakage_power = runtime_dynamic + subthreshold_leakage + gate_leakage
    thermal_input_dict['Processor'] = [runtime_dynamic, total_leakage_power]
    for unit in per_processor_units:
        get_per_processor_total_power(mcpat_output_dict, unit)

    # Gets total power for IMC
    # total power (W) = (num DRAM accesses)*[[(0.1 W/GB/s)*(Line Size in GB)]/(time step size in s)]
    # TODO: Don't hardcode in time step size, this may change
    outputfile_dirname = os.path.dirname(mcpat_json_file)
    outputfile_basename = os.path.basename(mcpat_json_file).split('.')[0]
    timestep_ID = outputfile_basename.split('_')[-1]
    inputfile_name = os.path.join(outputfile_dirname, "energystats-temp-" + timestep_ID + ".xml")

    # Open .xml input file associated with current timestep and grab num memory accesses
    # Line from example .xml: <stat name="memory_accesses" value="501"/>
    mcpat_input_file = open(inputfile_name, 'r') 
    Lines = mcpat_input_file.readlines() 
    line_count = 0
    for line in Lines: 
        if '<stat name="memory_accesses"' in line:
            num_mem_accesses = line.strip().split('"')[-2]
        line_count = line_count + 1

    IMC_total_power = int(num_mem_accesses)*((0.1*0.000000064)/0.0002)
    thermal_input_dict['IMC'] = [IMC_total_power, 0] # This is already the sum of dynamic and leakage

    # Pretty print thermal input dictionary for validation/visualization
    #pp.pprint(thermal_input_dict)

    # Create list of all parent units. This will be used to deal with hierarchy. Parent unit power will
    # just be the power of the stitch logic for that unit
    # TODO: what to do about Execution Unit and Memeory Management Unit
    all_per_core_parent_units = ['Execution Unit/Instruction Scheduler', 'Execution Unit/Register Files', 'Instruction Fetch Unit/Branch Predictor', 'Instruction Fetch Unit', 'Load Store Unit', 'Renaming Unit']

    # For each unit in all_parent_units, subtract childrens' power (dynamic and leakage separate still)
    for core_num in range(num_cores):
        for parent_unit in all_per_core_parent_units:
            childrens_dynamic_power_vals = [power[0] for unit_name, power in thermal_input_dict.items() if ('Core' + str(core_num) + '/' + parent_unit + '/') in unit_name]
            childrens_leakage_power_vals = [power[1] for unit_name, power in thermal_input_dict.items() if ('Core' + str(core_num) + '/' + parent_unit + '/') in unit_name]

            total_childrens_dynamic_power = sum(childrens_dynamic_power_vals)
            total_childrens_leakage_power = sum(childrens_leakage_power_vals)

            # Update parent dynamic power
            thermal_input_dict['Core' + str(core_num) + '/' + parent_unit][0] = (thermal_input_dict['Core' + str(core_num) + '/' + parent_unit][0]) - total_childrens_dynamic_power
            # Update parent leakage power
            thermal_input_dict['Core' + str(core_num) + '/' + parent_unit][1] = (thermal_input_dict['Core' + str(core_num) + '/' + parent_unit][1]) - total_childrens_leakage_power


    # Compute AVX512 Accelerator Power
    outputfile_dirname = os.path.dirname(mcpat_json_file)
    outputfile_basename = os.path.basename(mcpat_json_file).split('.')[0]
    timestep_ID = outputfile_basename.split('_')[-1]
    inputfile_name = os.path.join(outputfile_dirname, "energystats-temp-" + timestep_ID + ".xml")

    # Open .xml input file associated with current timestep and grab num memory accesses
    # Line from example .xml:
    # <stat name="fp_instructions" value="513696"/>
    # <stat name="AVX512_instructions" value="8396"/>
    mcpat_input_file = open(inputfile_name, 'r') 
    Lines = mcpat_input_file.readlines() 
    line_count = 0
    num_fp_instructions = []
    num_avx512_instructions = []
    
    # Grab number of fp instructions and avx512 instructions per core
    for line in Lines: 
        if '<stat name="fp_instructions"' in line:
            num_fp_instructions.append(line.strip().split('"')[-2])
        if '<stat name="AVX512_instructions"' in line:
            num_avx512_instructions.append(line.strip().split('"')[-2])
        line_count = line_count + 1

    avx512_to_fp_ratios = [] 

    # Compute ratio (power scale factor)
    for num_fp, num_avx512 in zip(num_fp_instructions, num_avx512_instructions):
        avx512_to_fp_ratios.append(float(num_avx512)/float(num_fp))

    # Scale int ALU power vals per core by scale factor
    for core_num in range(num_cores):
        runtime_dynamic = mcpat_output_dict['Core'][core_num]['Execution Unit/Floating Point Units/Runtime Dynamic']
        subthreshold_leakage = mcpat_output_dict['Core'][core_num]['Execution Unit/Floating Point Units/Subthreshold Leakage']
        gate_leakage = mcpat_output_dict['Core'][core_num]['Execution Unit/Floating Point Units/Gate Leakage']
        total_dynamic_power = 2*runtime_dynamic
        total_leakage_power = subthreshold_leakage + gate_leakage

        # commented by Maz
        #scaled_dynamic_power = (1024/64)*avx512_to_fp_ratios[core_num]*total_dynamic_power 
        #scaled_leakage_power = (1024/64)*avx512_to_fp_ratios[core_num]*total_leakage_power
        
        # commented by Maz
        # thermal_input_dict['Core' + str(core_num) + '/Execution Unit/AVX512 Accelerator'] = [scaled_dynamic_power, scaled_leakage_power]


    # Sum dynamic and leakage power here
    for unit_name in thermal_input_dict:
        # TODO: This is where we can also scale leakage (right before we combine them)
        # Look-up unit_name in temperature dictionary to get leakage scale factor (comes from 3D ICE)
        thermal_input_dict[unit_name] = sum(thermal_input_dict[unit_name])

    # Pretty print thermal input dictionary for validation/visualization
    #pp.pprint(thermal_input_dict)

    if write == True:
        # Write thermal input dictionary to csv file
        dirname, fname = os.path.split(mcpat_json_file)
        fname = fname.replace('mcpat_output_', 'block_powers_')
        out_file = os.path.join(dirname, fname)
        with open(out_file, 'w') as json_file:
            json.dump(thermal_input_dict, json_file)

    return dict(thermal_input_dict)

@click.command()
@click.argument('mcpat_run_dir')
def mcpat_to_blk_lvl_power_dict(mcpat_run_dir):
    NUM_PROCESSES = int(0.75*multiprocessing.cpu_count())
    pool = multiprocessing.Pool(NUM_PROCESSES)

    input_files = glob.glob(os.path.join(mcpat_run_dir, 'mcpat_output_*.json'))
    for _ in tqdm.tqdm(pool.imap_unordered(single_mcpat_to_blk_lvl_power_dict, input_files), total=len(input_files)):
        pass

if __name__ == '__main__':
    mcpat_to_blk_lvl_power_dict()
