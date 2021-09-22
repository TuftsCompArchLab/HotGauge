import os
import re
import glob
import math
import json
from collections import defaultdict
import pathlib
import copy
import logging
LOGGER = logging.getLogger(__name__)

import numpy as np

from HotGauge import HOT_GAUGE_ROOT_DIR
from HotGauge.script_runner import ExecutableJob
from HotGauge.script_runner.tool_scripts import write_or_update_file, populate_template
from HotGauge.utils import Floorplan
from HotGauge.configuration import mcpat_to_flp_name, NO_POWER_UNITS, DRAM_IMC_UNITS, DRAM_IMC_POWER, DRAM_IO_UNITS, DRAM_IO_POWER, IMC_REGEX, get_IO_SoC_powers
from HotGauge.thermal.utils import K_to_C

def load_raw_ptraces(ptraces):
    verbose_power_trace_dict = defaultdict(list)
    for mcpat_output in ptraces:
        power_dict = json.load(open(mcpat_output, 'r'))
        for k,v in power_dict.items():
            verbose_power_trace_dict[k].append(v)
    return verbose_power_trace_dict

def load_ptraces(ptraces, core_mapping, num_cores=8):
    if isinstance(core_mapping, dict):
        # TODO: Should this be the keys of the dict? Could remove num_cores as well
        # TODO: Core mapping would need to be fully populated, even when core_x -> core_x
        selected_cores = ['Core{}'.format(k) for k in range(num_cores)]
        multicore=True
    elif isinstance(core_mapping, int):
        selected_cores = ['Core{}'.format(core_mapping)]
        multicore=False
    else:
        raise RuntimeError('core mapping invalid: {}'.format(core_mapping))
    verbose_power_trace_dict = load_raw_ptraces(ptraces)
    power_dict = {}
    for label, vals in verbose_power_trace_dict.items():
        # Skip everything but cores and IMC
        if label.split('/')[0] not in selected_cores and label != 'IMC':
            continue
        # Let IMC slip through untouched
        if label == 'IMC':
            flp_label = label
        # Otherwise check if the sim is multicore
        elif multicore:
            flp_label = mcpat_to_flp_name(label, include_core_idx=True, core_idx_lookup=core_mapping)
        else:
            flp_label = mcpat_to_flp_name(label, include_core_idx=False)
        power_dict[flp_label] = np.array(vals)

    # Split the L3 power among all cores
    L3_powers = np.array(verbose_power_trace_dict['Processor/Total L3s']) / num_cores
    if multicore:
        for core_idx in range(num_cores):
            power_dict['L3_{}'.format(core_idx)] = L3_powers
    else:
        power_dict['L3'] = L3_powers
    return power_dict

def load_3DICE_grid_file(fname, convert_K_to_C=True):
    def load_step(raw_data):
        rows = raw_data.split('\n')
        data = []
        for row in rows:
            vals = list(float(val) for val in row.split())
            if len(vals) > 0:
                data.append(vals)
        return data

    f = open(fname, 'r')
    l = f.readline() # skip first line
    assert l[0] == '%'
    content = f.read()
    steps_data = content.split('\n\n')
    grids = []
    for step_data in steps_data:
        vals = load_step(step_data)
        if len(vals) > 0:
            grids.append(vals)

    ttrace = np.flip(np.asarray(grids, dtype=float), 1)
    if convert_K_to_C:
        ttrace = K_to_C(ttrace)
    return ttrace

def load_3DICE_block_file(fname, convert_K_to_C=False, to_dict=True):
    # TODO: make this more readable so that convert_K_to_C can be implemented
    if convert_K_to_C:
        raise NotImplementedError()
    f = open(fname, 'r')
    l1 = f.readline() # skip first line
    assert l1[0] == '%'
    l2 = f.readline() # The second line is the header
    assert l2[0] == '%'

    header = [t[:-4].strip() for t in l2[1:].split('\t')]
    data = [[]]*len(header)
    content = f.read()
    lines = content.split('\n')
    values = [list(map(float, line.split('\t')[:-1])) for line in lines][:-1]
    if to_dict:
        return dict(zip(header,map(np.array, zip(*values))))
    return header, values

def float_to_string(number, precision=20):
    return '{0:.{prec}f}'.format(
            number, prec=precision,
            ).rstrip('0').rstrip('.') or '0'

ICE_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '3d-ice'))
ICE_STK_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), 'stack_templates'))
class ICESim(ExecutableJob):
    CELL_SIZE_REGEX = re.compile(r'cell\s+length\s+(\S+)\s*,\s+width\s+(\S+)\s*;')
    OUTPUT_START_REGEX = re.compile('^\s*output\s*:\s*$')
    OUTPUT_REGEX = re.compile(r'^(\w*)\s*\((\s*\w*\s*,)?\s*"(.*)"\s*(,\s*\w*\s*)+\)\s*;$')
    COMMENT_REGEX = r'("[^\n]*"(?!\\))|(//[^\n]*$|/(?!\\)\*[\s\S]*?\*(?!\\)/)'
    PLUGIN_REGEX = re.compile(r'\s*plugin\s*".*"\s*,\s*"(\S+)\s.*"\s*;\s*')
    EMULATOR_EXECUTABLE = os.path.join(ICE_DIR, 'bin', '3D-ICE-Emulator')

    OUTPUT_TSTACK_FINAL = 'Tstack ("final.tstack", final ) ;'

    parallel_options = copy.deepcopy(ExecutableJob.parallel_options)

    def __init__(self, stk_template, flp_template, ptraces, tech_node, *args, **kwargs):
        """
        kwargs:
            core_mapping:
                (dict) : multi-core sims {str -> str} key:sim_core_number | value:floorplan_core_label
                (int) : single-cre sims
            num_cores: (int) number of cores in power traces
        """
        self.stk_template = stk_template
        self.flp_template = flp_template
        self.ptraces = ptraces
        self.tech_node = tech_node
        self.core_mapping = kwargs.pop('core_mapping', {})
        self.num_cores = kwargs.pop('num_cores', 8)
        self.initial_temp = kwargs.pop('initial_temp', 300)
        self.initial_temp_file = kwargs.pop('initial_temp_file', None)
        self.plugin_args = kwargs.pop('plugin_args', None)
        self.output_list = kwargs.pop('output_list', '')
        self._output_files_cache = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def get_file_name(output_str):
        return ICESim.OUTPUT_REGEX.match(output_str).group(3)

    @classmethod
    def get_cell_size(cls, stk_file):
        cell_size = None
        try:
            lines = cls.strip_stk_file(open(stk_file, 'r').read())
            for line in lines:
                match = cls.CELL_SIZE_REGEX.match(line)
                if match:
                    assert cell_size == None
                    length, width = match.groups((0,1))
                    cell_size = (float(length), float(width))
            return cell_size
        except:
            pass
        return (None, None)

    @classmethod
    def get_pluggable_heatsink(cls, contents):
        for line in contents.split('\n'):
            match = cls.PLUGIN_REGEX.match(line)
            if match:
                return match.group(1)
        return None


    @classmethod
    def strip_stk_file(cls, contents):
        # Remove C style comments
        for x in re.findall(ICESim.COMMENT_REGEX,contents,8):
            contents = contents.replace(x[1],'')
        # Remove blank lines and strip others
        return [l.strip() for l in contents.split('\n') if len(l.strip())>0]

    @property
    def flp_file(self):
        return os.path.join(self.run_path, 'IC.flp')

    @property
    def stk_file(self):
        return os.path.join(self.run_path, 'IC.stk')

    @property
    def solver_config(self):
        lines = [self.sim_type_str]
        if self.initial_temp_file is None:
            if isinstance(self.initial_temp, tuple):
                lines.append('initial temperature {} ;'.format(self.initial_temp[0]))
                lines.append('initial sink temperature {} ;'.format(self.initial_temp[1]))
            else:
                lines.append('initial temperature {} ;'.format(self.initial_temp))
        else:
            if isinstance(self.initial_temp, tuple):
                raise ValueError('initial_temperature must be scalar (not tuple) when using initial_temp_file')
            else:
                lines.append('initial temperature "{}" ;'.format(self.initial_temp_file))
                lines.append('initial sink temperature {} ;'.format(self.initial_temp))
        return '\n'.join('   '+ l for l in lines)

    def output_files(self):
        if self._output_files_cache is not None:
            return self._output_files_cache
        with open(self.stk_file, 'r') as f:
            contents = f.read()
        trimmed_lines = iter(self.__class__.strip_stk_file(contents))
        for line in trimmed_lines:
            if ICESim.OUTPUT_START_REGEX.match(line):
                break
        files = [ICESim.OUTPUT_REGEX.match(l).group(3) for l in trimmed_lines]
        fullpath_files = list(map(lambda f: os.path.join(self.run_path, f), files))
        self._output_files_cache = fullpath_files
        return self._output_files_cache

    def input_files(self):
        files = [self.stk_file, self.flp_file, self.__class__.EMULATOR_EXECUTABLE]
        files.extend(self.ptraces)
        if self.initial_temp_file != None:
            files.append(self.initial_temp_file)
        return files

    def job_args(self):
        # runs {3D-ICE_Emulator} {stk_file}
        return '{} {} {}'.format(self.run_path,
                                 self.__class__.EMULATOR_EXECUTABLE,
                                 self.stk_file)

    @classmethod
    def job_cmd(cls):
        # TODO: make this it's own *template.sh script
        """The command to run with job_args from each instance"""
        return os.path.join(HOT_GAUGE_ROOT_DIR, 'script_runner', 'run_executable_instance')

    def prep_for_run(self):
        self.fill_flp_template()
        self.fill_stk_template()

    def load_power_dict(self):
        power_dict = load_ptraces(self.ptraces, self.core_mapping)

        # Add power values for other units
        empty = np.abs(0*list(power_dict.values())[0])
        ones = empty+1

        # Tally up areas for IMC/IO
        flp = Floorplan.from_file(self.flp_template)
        total_IMC_area = 0.0
        total_IO_area = 0.0
        for el in flp.elements:
            if el.name in DRAM_IMC_UNITS:
                total_IMC_area += el.area
            elif el.name in DRAM_IO_UNITS:
                total_IO_area += el.area
            # Skip no power units
            elif el.name in NO_POWER_UNITS:
                continue
            # Make sure the unit isn't an IMC
            elif IMC_REGEX.match(el.name):
                LOGGER.warn('{} encountered but not included in IMC area'.format(el.name))
            # Let other units pass through
        try:
            total_IMC_IO_power = power_dict['IMC']
        except KeyError as e:
            LOGGER.error('{} in {} is missin IMC data'.format(self.__class__, self.run_path))
            raise
        total_IMC_power = DRAM_IMC_POWER * total_IMC_IO_power
        total_IO_power = DRAM_IO_POWER * total_IMC_IO_power
        IO_SoC_powers = get_IO_SoC_powers(self.tech_node)
        for el in flp.elements:
            if el.name in DRAM_IMC_UNITS:
                power_dict[el.name] = total_IMC_power * el.area / total_IMC_area * ones
            elif el.name in DRAM_IO_UNITS:
                power_dict[el.name] = total_IO_power * el.area / total_IO_area * ones
            elif el.name in NO_POWER_UNITS:
                 power_dict[el.name] = empty.copy()
            elif el.name in IO_SoC_powers:
                power_dict[el.name] = ones * IO_SoC_powers[el.name]
            # Let other units pass through
        return power_dict

    def fill_flp_template(self):
        power_dict = self.load_power_dict()
        # Convert raw numbers to 3DICE format
        for k,v in power_dict.items():
            power_dict[k] = ', '.join(map(str, v))
        contents = populate_template(self.flp_template, powers=power_dict)
        return write_or_update_file(self.flp_file, contents)


    def fill_stk_template(self):
        flp = Floorplan.from_file(self.flp_file)
        stk_width, stk_height = flp.width+1e-3, flp.height+1e-3

        # Make sure there are an integer number of pixels
        #   (actualy make it even too, so heatspreader border is even too)
        #   Cell is 50 um, so 2 cells are 100um. Spreader is 30,000um
        stk_width = math.ceil(stk_width/100)*100
        stk_height = math.ceil(stk_height/100)*100

        if isinstance(self.output_list, list):
            output_list_str = '\n'.join('   ' + o for o in self.output_list)
        elif isinstance(self.output_list, type(None)):
            output_list_str = ''
        elif isinstance(self.output_list, str):
            output_list_str = self.output_list


        rel_flp_file = os.path.relpath(self.flp_file, self.run_path)
        contents = populate_template(self.stk_template, flp_file=rel_flp_file,
                                     flp_width=stk_width,
                                     flp_height=stk_height,
                                     solver_config=self.solver_config,
                                     initial_temp=self.initial_temp,
                                     ICE_DIR=ICE_DIR,
                                     plugin_args=self.plugin_args,
                                     output_list=output_list_str)

        pluggable_heatsink = self.__class__.get_pluggable_heatsink(contents)
        if pluggable_heatsink:
            base_heatsink_dir = os.path.join(ICE_DIR, 'heatsink_plugin', 'heatsinks')
            basename = pluggable_heatsink.split('.')[0]
            basename = {'Cuplex':'cuplex_kryos_21606'}.get(basename, basename)
            heatsink_model_dir = os.path.join(base_heatsink_dir, basename, pluggable_heatsink)
            dst_link = os.path.join(self.run_path, pluggable_heatsink)
            tmp_link = os.path.join(self.run_path, "_"+pluggable_heatsink)
            os.symlink(heatsink_model_dir,tmp_link)
            os.rename(tmp_link, dst_link)
        return write_or_update_file(self.stk_file, contents)

    def output_status(self):
        num_outputs = self.output_length
        msgs = []
        for output in self.output_files():
            if os.path.splitext(output)[1] == '.tstack':
                # TODO: Check number of lines in this files based on flp size, resolution and length?
                pass
            else: # Other files have number of outputs equal to simulation steps
                lines = open(output, 'r').readlines()
                regexes = [
                           re.compile('^$'), # Grid files are seperated by empty lines
                           re.compile('^\s*[^%]') # Element files have 1 entry per line (plus comments)
                          ]
                counts = []
                for regex in regexes:
                    count = sum(regex.match(line)!=None for line in lines)
                    if count == num_outputs:
                        break
                    counts.append(count)
                else:
                    # None of the known file formats have been validated
                    lens = ', '.join(['{1} for {0}'.format(r,c) for r,c in zip(regexes, counts)])
                    msgs.append('Output length using python for {} not correct. Expected {} but got {}'.format(os.path.basename(output), num_outputs, lens))
        if len(msgs)==0:
            return True, None
        return False, "\n".join(msgs)


class ICETransientSim(ICESim):
    TRANSIENT_STEP_REGEX = re.compile(r'transient\s*step\s*(\S+)\s*,\s*slot\s*(\S+)\s*;')
    DIE_TMAP_OUTPUT = 'Tmap (PROCESSOR_DIE, "die_grid.temps", slot ) ;'
    DIE_PMAP_OUTPUT = 'Pmap (PROCESSOR_DIE, "die_grid.pows", slot ) ;'
    DIE_TFLP_OUTPUT = 'Tflp (PROCESSOR_DIE, "die_elements.temps", average, slot) ;'
    def __init__(self, time_slot, *args, time_step=None, **kwargs):
        self.time_slot = time_slot
        if time_step is None:
            self.time_step = self.time_slot / 10.0
        else:
            assert time_step <= self.time_slot, 'Time step must be <= time slot'
            self.time_step = time_step
        super().__init__(*args, **kwargs)

    @classmethod
    def get_step_slot(cls, stk_file):
        step_slot = None
        try:
            lines = cls.strip_stk_file(open(stk_file, 'r').read())
            for line in lines:
                match = cls.TRANSIENT_STEP_REGEX.match(line)
                if match:
                    assert step_slot == None
                    step, slot = match.groups((0,1))
                    step_slot = (float(step), float(slot))
            return step_slot
        except:
            pass
        return (None, None)

    @property
    def sim_type_str(self):
        step =  float_to_string(self.time_step, 10)
        slot =  float_to_string(self.time_slot, 10)
        return 'transient step {}, slot {} ;'.format(step, slot)

    @property
    def output_length(self):
        return len(self.ptraces)

class ICETransientSimRAW(ICETransientSim):
    def load_power_dict(self):
        return load_raw_ptraces(self.ptraces)


class ICESteadySim(ICESim):
    DIE_TMAP_OUTPUT = 'Tmap (PROCESSOR_DIE, "die_grid.temps", final ) ;'
    DIE_PMAP_OUTPUT = 'Pmap (PROCESSOR_DIE, "die_grid.pows", final ) ;'
    DIE_TFLP_OUTPUT = 'Tflp (PROCESSOR_DIE, "die_elements.temps", average, final ) ;'
    @property
    def sim_type_str(self):
        return 'steady ;'

    @property
    def output_length(self):
      return 1

def get_stk_template(template_name):
    template_fname = template_name.replace('.stk','') + '.stk'
    stk_fp = os.path.join(ICE_STK_DIR, template_fname)
    return stk_fp

class ICESteadySimRAW(ICESteadySim):
    def load_power_dict(self):
        return load_raw_ptraces(self.ptraces)
