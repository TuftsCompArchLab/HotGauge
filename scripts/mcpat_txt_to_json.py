#!/usr/bin/env python
import glob
import re
import json
import sys
import os
import tqdm

# See sniper/tools/mcpat.py
nuca_at_level=3

def dram_power(results, config):
  return compute_dram_power(
    sum(results['dram.reads']),
    sum(results['dram.writes']),
    results['global.time'] * 1e-15,
    config
  )

# Parse McPAT output and display it in a way that's easier to parse
def parse_output(outputfile):

  # Parse output
  power_txt = open(outputfile)
  power_dat = {}

  components = power_txt.read().split('*'*89)[2:-1]
  for component in components:
    lines = component.strip().split('\n')
    componentname = lines[0].strip().strip(':')
    values = {}
    prefix = []; spaces = []
    for line in lines[1:]:
      if not line.strip():
        continue
      elif '=' in line:
        res = re.match(' *([^=]+)= *([-+0-9.e]+)(nan)?', line)
        if res:
          name = ('/'.join(prefix + [res.group(1)])).strip()
          if res.groups()[-1] == 'nan':
            # Result is -nan. Happens for instance with 'Subthreshold Leakage with power gating'
            # on components with 0 area, such as the Instruction Scheduler for in-order cores
            value = 0.
          else:
            try:
              value = float(res.group(2))
            except:
              print >> sys.stderr, 'Invalid float:', line, res.groups()
              raise
          values[name] = value
      else:
        res = re.match('^( *)([^:(]*)', line)
        if res:
          j = len(res.group(1))
          while(spaces and j <= spaces[-1]):
            spaces = spaces[:-1]
            prefix = prefix[:-1]
          spaces.append(j)
          name = res.group(2).strip()
          prefix.append(name)
    if componentname in ('Core', 'L2', 'L3'):
      # Translate whatever level we used for NUCA back into NUCA
      if componentname == 'L%d' % nuca_at_level:
        outputname = 'NUCA'
      else:
        outputname = componentname
      if outputname not in power_dat:
        power_dat[outputname] = []
      power_dat[outputname].append(values)
    else:
      assert componentname not in power_dat
      power_dat[componentname] = values

  if not power_dat:
    raise ValueError('No valid McPAT output found')

  # Write back
  outputfile_name = os.path.basename(outputfile).split('.')[0]
  outfile = os.path.join(sys.argv[1], outputfile_name + '.json')
  json.dump(power_dat, open(outfile, 'w'), sort_keys=True,  indent=2, separators=(',', ':'))

def main():
    # Loop over all McPAT outputs (.txt)
    for out_file in tqdm.tqdm(glob.glob(os.path.join(sys.argv[1], "mcpat_output_*.txt"))):
        parse_output(out_file)

if __name__=='__main__':
    main()
