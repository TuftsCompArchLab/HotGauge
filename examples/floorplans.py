#!/usr/bin/env python
import sys
import os
import json
import copy
from collections import defaultdict
import itertools
import math

from HotGauge.utils import Floorplan, FloorplanElement
from HotGauge.configuration import mcpat_to_flp_name, MISSING_POWER_INFO, DERIVED_UNITS, \
                                   REMOVED_UNITS, NODE_LENGTH_FACTORS

# TODO: These values are hard coded due to sim configuration issues
L3_CORRECTION_FACTOR = 4

# These values are used for scaling MCPAT's output
L2_AREA_MULTIPLIER = 2.908127478180562

# These values are used for misc constants
PADDING_SIZE = 500 # um

def get_parents(unit):
    heir_indices = [i for i,ch in enumerate(unit) if ch=='/']
    for index in heir_indices:
        yield unit[:index]

def load_14nm_stats(stats_file, num_cores=8):
    # Loaded units are in mm**2 and converted to um**2
    stats = json.load(open(stats_file, 'r'))
    stats = {unit: {k: (1000**2)*v for k,v in areas.items()} for unit, areas in stats.items()}

    # Make L3 part of each core (and quadruple its size??)
    L3_area = stats['Processor']['Total L3s/Area'] * L3_CORRECTION_FACTOR
    stats['Core']['L3/Area'] = L3_area / num_cores
    stats['Core']['Area'] += stats['Core']['L3/Area']

    # Make RBB Area instead of Area Overhead
    unit_name = 'Execution Unit/Results Broadcast Bus/Area'
    stats['Core'][unit_name] = stats['Core'][unit_name + ' Overhead']
    del(stats['Core'][unit_name + ' Overhead'])

    # Make L2 roughly 1/3 of size of core
    stats['Core']['Area'] -= stats['Core']['L2/Area']
    stats['Core']['L2/Area'] *= L2_AREA_MULTIPLIER
    stats['Core']['Area'] += stats['Core']['L2/Area']


    # add derived units (e.g. AVX512)
    for unit_name, components in DERIVED_UNITS.items():
        unit_area = 0.0
        for component in components:
            component_area=  stats['Core']['{}/Area'.format(component.base)]
            unit_area += component_area * component.ratio
        stats['Core']['{}/Area'.format(unit_name)] = unit_area
        stats['Core']['Area'] += unit_area

        # Also propogate the area up to the parents!
        for parent_name in get_parents(unit_name):
            parent_area_label = '{}/Area'.format(parent_name)
            if parent_area_label in stats['Core']:
                stats['Core'][parent_area_label] += unit_area
            else:
                stats['Core'][parent_area_label] = unit_area

    # Remove units that are relocated/removed
    for unit_name in REMOVED_UNITS:
        unit_label = '{}/Area'.format(unit_name)
        unit_area = stats['Core'][unit_label]
        # Also propogate the removed area up to the parents!
        for parent_name in get_parents(unit_name):
            parent_area_label = '{}/Area'.format(parent_name)
            stats['Core'][parent_area_label] -= unit_area
        # TODO: add this line back in! It's a bug from original HPCA submission
        # stats['Core']['Area'] -= unit_area
        del stats['Core'][unit_label]

    # Delete unused units
    for unit in ['Processor', 'NUCA', 'BUSES']:
        del(stats[unit])

    # Return the stats for a single core
    assert len(stats) == 1, 'Only expected "Core" to be left'
    return stats['Core']

def split_levels(stats):
    stats_per_level = defaultdict(dict)
    for k,v in stats.items():
        tokens = k.split('/')
        hierarchy, area = tokens[:-1], tokens[-1]
        assert area == 'Area', 'Expected only Area stats but got {}'.format(area)
        hierarchy_level = len(hierarchy)

        # If top level, call it core
        if hierarchy_level == 0:
            hierarchy = ['Core']

        # Make L2 and L3 part of the top level
        if hierarchy_level == 1:
            if hierarchy[0] in ['L2', 'L3']:
                hierarchy_level = 0
        unit_name = '/'.join(hierarchy)
        stats_per_level[hierarchy_level][unit_name] = v

    # Readjust the top level to account for adding L2 and L3
    stats_per_level[0]['Core'] -= stats_per_level[0]['L2']
    stats_per_level[0]['Core'] -= stats_per_level[0]['L3']

    return stats_per_level

def get_base_floorplan(split_level_stats):
    # The total of Core, L2, L3, and possibly more
    total_area = sum(split_level_stats[0].values())

    cache_area = split_level_stats[0]['L2'] + split_level_stats[0]['L3']
    core_area = total_area - cache_area

    # ratio of non-cache portion of die from IC photos
    core_aspect_ratio = 4.0 / 6.0

    core_width = core_area ** 0.5 / (core_aspect_ratio ** 0.5)
    core_height = core_area / core_width

    core = FloorplanElement('Core', core_width, core_height, 0.0, 0.0)

    flp = Floorplan([core], frmt='3D-ICE')
    flp.auto_place_element('L2', split_level_stats[0]['L2'], where='right')
    flp.auto_place_element('L3', split_level_stats[0]['L3'], where='above')
    return flp

def add_pipeline(flp, pipeline_stats):
    total_area = sum(pipeline_stats.values())
    EX_area = pipeline_stats['Execution Unit']
    AVX_FPU_area = pipeline_stats['AVX_FPU']
    not_EX_area = total_area - EX_area - AVX_FPU_area

    # Place EX on the end, double wide
    flp = replace(flp, 'Core', [('Execution Unit', EX_area), ('not_EX', not_EX_area)], vertical=False)

    cols = [['Renaming Unit', 'Instruction Fetch Unit'],['Load Store Unit', 'Memory Management Unit']]
    col_sizes = []
    for col_els in cols:
        col_sizes.append(sum(pipeline_stats[el] for el in col_els))
    col_flp_els = [('not_EX{}'.format(idx), col_sizes[idx]) for idx in range(len(cols))]
    flp = replace(flp, 'not_EX', col_flp_els, extra='none')

    for col_idx, col_els in enumerate(cols):
        new_els = [(el, pipeline_stats[el]) for el in col_els]
        flp = replace(flp, 'not_EX{}'.format(col_idx), new_els, extra='none', vertical=False)

    flp = replace(flp, 'Core', [('AVX_FPU', AVX_FPU_area)], vertical=False)
    return flp

def replace(flp, unit, subunit_sizes, vertical=True, extra='before'):
    unit_idx = ([e.name for e in flp.elements]).index(unit)
    unit = flp.elements[unit_idx]
    del flp.elements[unit_idx]
    total_size = sum(subunit[1] for subunit in subunit_sizes)
    x,y = unit.minx, unit.miny
    new_els = []
    extra_area = unit.area - total_size
    # Make sure the children are no larger than 0.1um**2 larger than parent
    # Branch Predictor children are slightly larger than Branch Predictor
    assert total_size <= unit.area + 1e-1

    # If there is more than 0.1um missing, occupy the extra space
    if extra_area > 5e-1:
        if extra == 'pad': # Pad between each element
            padding = extra_area / (len(subunit_sizes) + 1)
            all_units = subunit_sizes
        elif extra.lower() == 'none': # Don't pad
            padding = 0.0
            all_units = subunit_sizes

            if extra == 'NONE':
                msg = 'Parent unit, {}, has {} extra area, but is being removed'.format(unit.name, extra_area)
                LOGGER.warn(msg)
            else:
                assert extra_area < 1e-8
        elif extra == 'before': # Place extra (parent) element first
            padding = 0.0
            all_units = [(unit.name, extra_area)] + subunit_sizes
        elif extra == 'after': # Place extra (parent) element last
            padding = 0.0
            all_units = subunit_sizes + [(unit.name, extra_area)]
    else: # No need to pad
        padding = 0.0
        all_units = subunit_sizes

    for subunit, area in all_units:
        if vertical:
            el_w = unit.width
            el_h = area / el_w
            y += padding / el_w
            new_els.append(FloorplanElement(subunit, el_w, el_h, x, y))
            y += el_h
        else:
            el_h = unit.height
            el_w = area / el_h
            x += padding / el_h
            new_els.append(FloorplanElement(subunit, el_w, el_h, x, y))
            x += el_w
    flp.elements.extend(new_els)
    return flp

def add_level2(flp, split_level_stats2):
    avx_order = ['Floating Point Units', 'AVX512 Accelerator']
    r_order = ['Int Front End RAT', 'FP Front End RAT', 'Free List']
    ls_order = ['LoadQ', 'StoreQ', 'Data Cache']
    mmu_order = ['Itlb', 'Dtlb']
    # TODO: After adding AVX, should the IALU be moved too?
    ex_order = ['Instruction Scheduler', 'Register Files',
               'Results Broadcast Bus', 
               'Complex ALUs', 'Integer ALUs']
    if_order = ['Branch Target Buffer', 'Branch Predictor',
                'Instruction Decoder',
                'Instruction Buffer', 'Instruction Cache']
    ordering = {
                'Renaming Unit' : r_order,
                'Load Store Unit': ls_order,
                'Memory Management Unit': mmu_order,
                'Execution Unit': ex_order,
                'Instruction Fetch Unit': if_order,
                'AVX_FPU': avx_order
               }
    for el, sub_els in ordering.items():
        # Force removal of missing elements
        kwargs = {'extra' : 'NONE'} if el in MISSING_POWER_INFO else {}
        if el == 'Instruction Fetch Unit':
            kwargs['vertical'] = False
        flp = replace(flp, el.split('/')[-1],
                      [(sub_el, split_level_stats2['{}/{}'.format(el, sub_el)])
                      for sub_el in sub_els], **kwargs)
    return flp

# TODO: After adding AVX, should the RF be moved?
def add_level3(flp, split_level_stats3):
    rf_order = ['Integer RF', 'Floating Point RF']
    rf_els = [(sub_el, split_level_stats3['Execution Unit/Register Files/{}'.format(sub_el)]) for sub_el in rf_order]
    flp = replace(flp, 'Register Files', rf_els, extra='none')

    bp_order = ['Global Predictor', 'L2_Local Predictor',
                'L1_Local Predictor', 'Chooser', 'RAS']
    bp_els = [(sub_el, split_level_stats3['Instruction Fetch Unit/Branch Predictor/{}'.format(sub_el)]) for sub_el in bp_order]
    flp = replace(flp, 'Branch Predictor', bp_els)

    is_order = ['Instruction Window', 'FP Instruction Window', 'ROB']
    is_els = [(sub_el, split_level_stats3['Execution Unit/Instruction Scheduler/{}'.format(sub_el)]) for sub_el in is_order]
    flp = replace(flp, 'Instruction Scheduler', is_els)
    return flp

def CORE_SUBSITUTE(core_flp, name):
    width = core_flp.width
    height = core_flp.height
    return Floorplan([FloorplanElement(name, width, height, 0, 0)])

def make_7_core_processor(core_flp):
    return make_processor(core_flp, 3, 3, {(1,0):'IMC',(1,2):'SoC'})

def make_processor(core_flp, width, length, core_substitutes):
    processor = Floorplan([], frmt=core_flp.frmt)
    core_idx=0
    for y,x  in itertools.product(range(width),range(length)):
        if (x,y) in core_substitutes:
           substitute_name = core_substitutes[(x,y)]
           loc_n = CORE_SUBSITUTE(core_flp, substitute_name)
        else:
           loc_n = core_flp.create_numbered_instance(core_idx)
           if x%2==1:
              loc_n.mirror_horizontal()
           core_idx+=1
        loc_n.minx = x*core_flp.width
        loc_n.miny = y*core_flp.height
        processor += loc_n
    return processor

def add_padding(flp, padding_width):
    width = flp.width
    flp.auto_place_element('N', width*padding_width, where='above')
    flp.auto_place_element('S', width*padding_width, where='below')
    height = flp.height
    flp.auto_place_element('E', height*padding_width, where='right')
    flp.auto_place_element('W', height*padding_width, where='left')

def add_IO(flp, padding_width):
    width = flp.width
    flp.auto_place_element('IO_N', width*padding_width, where='above')
    flp.auto_place_element('IO_S', width*padding_width, where='below')

def generate_floorplans(output_dir, fname_frmt, flp, padding_width=PADDING_SIZE):
    """Saves floorplan for all tech nodes with all formats with added padding"""
    old_frmt = flp.frmt
    flp.frmt = '3D-ICE'
    for node in ['14nm', '10nm', '7nm']:
        core_flp = flp * NODE_LENGTH_FACTORS[node]
        for el in core_flp.elements:
            el.name = mcpat_to_flp_name(el.name)

        # The core is now complete. Generate single core and 7 core flps, with padding
        processor_flp = make_7_core_processor(core_flp)
        if padding_width:
            add_padding(core_flp, padding_width)
            add_IO(processor_flp, padding_width)
        core_flp.reset_to_origin()
        processor_flp.reset_to_origin()
        for frmt in ['3D-ICE', 'hotspot']:
            core_flp.frmt = frmt
            core_flp.to_file(os.path.join(output_dir, fname_frmt.format(frmt=frmt, node=node, suffix='core')))
            processor_flp.frmt = frmt
            processor_flp.to_file(os.path.join(output_dir, fname_frmt.format(frmt=frmt, node=node, suffix='7core')))

        # Also save them in 3D-ICE format with power strs
        core_flp.frmt = '3D-ICE'
        core_flp.to_file(os.path.join(output_dir,
                                      fname_frmt.format(frmt=core_flp.frmt+'_template',
                                                        node=node, suffix='core')
                                     ), element_powers=True)
        processor_flp.frmt = '3D-ICE'
        processor_flp.to_file(os.path.join(output_dir,
                                           fname_frmt.format(frmt=processor_flp.frmt+'_template',
                                                             node=node, suffix='7core')
                                          ), element_powers=True)

    flp.frmt = old_frmt

def scale_units(configs):
    def scale_fn(stats):
        new_stats = dict(stats)
        for name, factor in configs:
            area_delta = stats['{}/Area'.format(name)] * (factor-1.0)
            new_stats['{}/Area'.format(name)] += area_delta
            # Also propogate the area up to the parents!
            for parent_name in get_parents(name):
                new_stats['{}/Area'.format(parent_name)] += area_delta
            new_stats['Area'] += area_delta
        return new_stats
    return scale_fn

iRF = 'Execution Unit/Register Files/Integer RF'
fpRF = 'Execution Unit/Register Files/Floating Point RF'
RBB = 'Execution Unit/Results Broadcast Bus'
fpIWin = 'Execution Unit/Instruction Scheduler/FP Instruction Window'
fpRAT = 'Renaming Unit/FP Front End RAT'
iRAT = 'Renaming Unit/Int Front End RAT'
freeList = 'Renaming Unit/Free List'
PROBLEM_UNITS = [('RF', [iRF, fpRF]),
                 ('RBB',[RBB]),
                 ('fpIWin', [fpIWin]),
                 ('freeList', [freeList]),
                 ('RATS', [fpRAT,iRAT])]
PROBLEM_FLP_UNITS = []
for group, units in PROBLEM_UNITS:
    flp_units = list(map(mcpat_to_flp_name, units))
    PROBLEM_FLP_UNITS.append((group, flp_units))

def main():
    global LOGGER
    from config import LOGGER, EXP_BASE_DIR, OUTPUT_DIR

    # Load base stats
    raw_stats = load_14nm_stats(os.path.join(EXP_BASE_DIR,'adjusted_14nm-area.json'))

    no_change = lambda s: dict(s)
    generation_configs = [(no_change, '')]

    PROBLEM_UNIT_SCALE_FACTORS = [1.1, 1.2, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    for (label, units), factor in itertools.product(PROBLEM_UNITS, PROBLEM_UNIT_SCALE_FACTORS):
        scale_fn = scale_units([(unit, factor) for unit in units])
        conf = (scale_fn, '{}_{}'.format(label, factor))
        generation_configs.append(conf)

    for stat_change_fn, label in generation_configs:
        updated_stats = stat_change_fn(raw_stats)
        split_level_stats = split_levels(updated_stats)

        # Base floorplans
        flp = get_base_floorplan(split_level_stats)
        if label != '':
            label += '_'
        flp_frmt = 'skylake{{node}}_{{suffix}}_{detail}_' + label + '{{frmt}}.flp'
        generate_floorplans(OUTPUT_DIR, flp_frmt.format(detail='0'), flp)

        # Next level of detail
        flp = add_pipeline(flp, split_level_stats[1])
        generate_floorplans(OUTPUT_DIR, flp_frmt.format(detail='1'), flp)

        # Next level of detail
        flp = add_level2(flp, split_level_stats[2])
        generate_floorplans(OUTPUT_DIR, flp_frmt.format(detail='2'), flp)

        # Final level of detail
        flp = add_level3(flp, split_level_stats[3])
        generate_floorplans(OUTPUT_DIR, flp_frmt.format(detail='3'), flp)

        # Scale the overall size of the base floorplans only
        if label == '':
            for area_scale_factor in [1.05, 1.1, 1.2, 1.35, 1.5, 2.0, 2.5, 3.0]:
                scaled_flp = flp * math.sqrt(area_scale_factor)
                generate_floorplans(OUTPUT_DIR, flp_frmt.format(detail='3_{:g}x'.format(area_scale_factor)), scaled_flp)

    LOGGER.log_end()

if __name__ == '__main__':
    main()
else:
    import logging
    #LOGGER = logging.getLogger()
