import os
import re
import glob
from collections import namedtuple

########################### MCPAT Configuration #########################################

################################## IC Area #########################################

# Constants used for scaling MCPAT areas from 14nm to other nodes
NODE_AREA_FACTORS = {
                      '14nm' : 1.0,
                      '10nm' : 0.5,
                      '7nm'  : 0.25,
                    }
NODE_LENGTH_FACTORS = {node: factor ** 0.5 for node,factor in NODE_AREA_FACTORS.items()}

################## Mapping beteen MCPAT units and floorplan names ########################

# Name mapping from MCPAT to floorplan (hotspot/3D-ICE)
MCPAT_UNIT_NAME_MAP = {
    'Core' : 'core_other',
    'Instruction Fetch Unit' : 'IF',
    'Execution Unit' : 'EX',
    'Load Store Unit' : 'LS',
    'L2' : 'L2',
    'L3' : 'L3',
    'Renaming Unit' : 'Rename',
    'Memory Management Unit' : 'MMU',
    'Int Front End RAT' : 'iRAT',
    'FP Front End RAT' : 'fpRAT',
    'Free List' : 'FreeList',
    'Data Cache' : 'DCache',
    'LoadQ' : 'LoadQ',
    'StoreQ' : 'StoreQ',
    'Itlb' : 'iTLB',
    'Dtlb' : 'dTLB',
    'Complex ALUs' : 'cALU',
    'Floating Point Units' : 'FPUs',
    'Instruction Scheduler' : 'iSched',
    'Integer ALUs' : 'iALU',
    'Register Files' : 'regs',
    'Results Broadcast Bus' : 'RBB',
    'Branch Predictor' : 'brPred',
    'Branch Target Buffer' : 'BTB',
    'Instruction Buffer' : 'iBuf',
    'Instruction Cache' : 'iCache',
    'Instruction Decoder' : 'iDec',
    'Floating Point RF' : 'fpRF',
    'Integer RF' : 'iRF',
    'Global Predictor' : 'gPred',
    'L2_Local Predictor' : 'L2Pred',
    'L1_Local Predictor' : 'L1Pred',
    'Chooser' : 'Chooser',
    'RAS' : 'RAS',
    'Instruction Window' : 'iWin',
    'FP Instruction Window' : 'fpiWin',
    'ROB' : 'ROB',
    'AVX512 Accelerator' : 'AVXs', # The AVXs unit is for both AVX units
    'AVX_FPU' : 'AVX_FPU' # The AVX_FPU unit contains both AVXs and FPUs

}

# Regex for components that reside within a core
CORE_COMPONENT_RGX = re.compile('^(Core)(\d+)(.*)')

def mcpat_to_flp_name(label, include_core_idx=False, core_idx_lookup=None):
    """Translate mcpat name to floorplan name
       label            : unit name in mcpat
       include_core_idx : specifies whether to append core number at end of name, e.g. ALU_0
       core_idx_lookup  : Map (e.g. dict) 

       Returns: floorplan label (optionally with core number appended)
    """

    # See if the unit label includes the core number as in "Core0/L2"
    label_has_core_match = CORE_COMPONENT_RGX.match(label)
    if label_has_core_match:
        # Extract the parts from the label
        c, cnum, rest = label_has_core_match.groups() # c == 'Core'
    else:
        # There's no core information included
        # make sure it was not requested
        assert include_core_idx==False, 'Cannot find core idx for {}'.format(label)
        rest = label
        c='' # no core information

    # Reassemble the label (without core num) and split it into the hierarchy
    hierarchy = (c+rest).split('/')
    # The deepest (end) of the hierarchy determines the floorplan unit name
    unit_label = MCPAT_UNIT_NAME_MAP[hierarchy[-1]]

    # Optionally append the core index
    if include_core_idx:
        # If no core mapping is provided, don't swap any cores (i.e. keep cnum unchanged)
        if core_idx_lookup is not None:
            # Default to cnum if the specific core isn't in the mapping (it wasn't swapped)
            default_cnum = cnum
            # Get the appropriate core from the lookup table
            cnum = core_idx_lookup.get(cnum, default_cnum)

        # Actualy append the core number
        unit_label = '{}_{}'.format(unit_label, cnum)

    return unit_label


######################## AVX and FPU Model Adjustments #######################
_FPU_NAME = 'Execution Unit/Floating Point Units'

# Stats of the FPU to help calculate the size of the AVX
FPU_UNITS = 2
FPU_BITWIDTH = 64 * 2 # Two inputs double precision
FPU_INTERFACE_WIDTH = FPU_BITWIDTH * FPU_UNITS

# Stats of the AVX to help calculate the size of the AVX
AVX_UNITS = 2
AVX_BITWIDTH = 512 * 2 # Two 512 bit inputs
AVX_INTERFACE_WIDTH = AVX_UNITS * AVX_BITWIDTH
AVX_512_AREA_VS_FPU = AVX_INTERFACE_WIDTH / FPU_INTERFACE_WIDTH # AVX_area = FPU_area * ratio

# HACK: Fudge the AVX ratio to match die photos
ORIGINAL_RATIO = 0.30972859241877726 # AVX_SIZE / Total Die Size
DESIRED_RATIO = 0.1 # Based on IC photos
AVX_FUDGE_FACTOR = ((1-ORIGINAL_RATIO) / ORIGINAL_RATIO)/ ((1-DESIRED_RATIO) / DESIRED_RATIO)
AVX_512_AREA_VS_FPU *= AVX_FUDGE_FACTOR

SourceComponent = namedtuple('SourceComponent', ['base', 'ratio'])

# Units that don't come directly from MCPAT
DERIVED_UNITS = {
                 # Move the FPU in the component heirarchy to make floorplanning easier
                 'AVX_FPU/Floating Point Units' : [SourceComponent(_FPU_NAME, 1.0)],
                 # Add a single AVX515 accelerator, with area derived from the FPU model
                 'AVX_FPU/AVX512 Accelerator' : [SourceComponent(_FPU_NAME, AVX_512_AREA_VS_FPU)]
                }
# Units that neee to be removed (e.g. FPU was moved somewhere else)
REMOVED_UNITS = [_FPU_NAME]

############################# Clock Frequency ############################
CLK_FREQ = 5e9
