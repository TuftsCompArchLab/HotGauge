#!/usr/bin/env python
import os

import click

from HotGauge.thermal.utils import C_to_K
from HotGauge.thermal import ICETransientSim, get_stk_template
from HotGauge.configuration import load_block_powers

HEATSINK_MODEL = 'HS483'
HEATSINK_ARGS = '6000' # rpm for HS483
@click.command()
@click.option('-1/-N','--single-thread/--multi-thread', default=False)
def main(single_thread):
    from config import LOGGER, EXP_BASE_DIR, FLP_BASE_DIR, OUTPUT_DIR
    time_slot_ms = 0.2
    time_slot = time_slot_ms / 1000 # seconds

    tech_node = 7 # example workload is 7nm
    stk_template = get_stk_template('skylake_{}'.format(HEATSINK_MODEL))
    flp_template = os.path.join(FLP_BASE_DIR,
                                'skylake{}nm_7core_3_3D-ICE_template.flp'.format(tech_node))

    workload_name = 'example_workload'
    ptrace_dir = os.path.join(EXP_BASE_DIR, 'traces', 'example_workload', '{}nm'.format(tech_node))
    ptraces = load_block_powers(ptrace_dir)

    ################################################################################
    ############################## Warmup Simulation ###############################
    ################################################################################
    # Create an example warmup: the start of the workload for 10 time-slots starting at 40C
    warmup_ptraces = [ptraces[0]] * 1
    warmup_initial_t = C_to_K(40)
    warmup_dir = os.path.join(OUTPUT_DIR, 'warmup')
    warmup_outputs = [ICETransientSim.OUTPUT_TSTACK_FINAL, ICETransientSim.DIE_TMAP_OUTPUT]
    # Configure the Simulation
    warmup_sim = ICETransientSim(time_slot, stk_template, flp_template, warmup_ptraces,
                                 tech_node, warmup_dir, initial_temp=warmup_initial_t,
                                 plugin_args=HEATSINK_ARGS, output_list=warmup_outputs)

    if single_thread:
        ICETransientSim.run([warmup_sim])
    else:
        ICETransientSim.run_with_parallels([warmup_sim])

    ################################################################################
    ########################### Main Workload Simulation ###########################
    ################################################################################
    sim_dir = os.path.join(OUTPUT_DIR, 'sim')
    warmup_stack = os.path.join(warmup_sim.run_path,
                                ICETransientSim.get_file_name(warmup_outputs[0]))
    initial_t = warmup_initial_t
    sim_outputs = [ICETransientSim.OUTPUT_TSTACK_FINAL,
                   ICETransientSim.DIE_TMAP_OUTPUT,
                   ICETransientSim.DIE_PMAP_OUTPUT,
                   ICETransientSim.DIE_TFLP_OUTPUT]

    # Swap cores 0 and 1 (mcpat 0 runs on flp 1, and mcpat 1 runs on flp 0)
    core_mapping = {1:0, 0:1}

    sim = ICETransientSim(time_slot, stk_template, flp_template, ptraces,
                          tech_node, sim_dir,
                          core_mapping=core_mapping,
                          initial_temp=initial_t, initial_temp_file=warmup_stack,
                          plugin_args=HEATSINK_ARGS, output_list=sim_outputs)

    if single_thread:
        ICETransientSim.run([sim])
    else:
        ICETransientSim.run_with_parallels([sim])

    LOGGER.log_end()

if __name__ == '__main__':
    main()
