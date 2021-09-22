#!/usr/bin/env python
import os
from collections import defaultdict

import pandas as pd
import click
import numpy as np
from scipy.signal import argrelmax

from HotGauge.thermal.ICE import load_3DICE_grid_file
from HotGauge.utils.io import open_file_or_stdout

################################################################################
############################## Analysis Functions ##############################
################################################################################

def compute_MLTDs(thermal_grid, xy_location, region_offsets):
    x_idx, y_idx = xy_location
    t_center = thermal_grid[x_idx, y_idx]
    region = ((x_idx + x_off, y_idx + y_off) for x_off, y_off in region_offsets)
    tmin_in_radius, tmax_in_radius = _range_within_region(thermal_grid, region)
    pos_MLTD = tmax_in_radius - t_center
    neg_MLTD = t_center - tmin_in_radius
    return neg_MLTD, pos_MLTD

# TODO: add gradient calculations back in?
# Possibly optionally (since they are computationally expensive and not currently used)

def characterize_maxima(thermal_grid, pixel_radius, in_both_dimensions=True, as_df=True):
    # First filter candidates in either both dimensions or in either dimension
    if in_both_dimensions == True:
        candidates = _local_max_indices_2D(thermal_grid)
    else:
        candidates = _local_max_indices_1D(thermal_grid)

    circle_offsets = list(_circle_region_offsets(pixel_radius))

    data = defaultdict(list)
    for xy_location in candidates:
        neg_MLTD, pos_MLTD = compute_MLTDs(thermal_grid, xy_location, circle_offsets)
        x_idx, y_idx = xy_location
        data['x_idx'].append(x_idx)
        data['y_idx'].append(y_idx)
        data['temp_xy'].append(thermal_grid[x_idx, y_idx])
        data['neg_MLTD'].append(neg_MLTD)
        data['pos_MLTD'].append(pos_MLTD)
    if as_df:
        return _local_max_stats_dict_to_df(data)
    return data

def characterize_maxima_from_trace(thermal_trace, pixel_radius, in_both_dimensions=True, as_df=True):
    all_data = defaultdict(list)
    for time_step, thermal_grid in enumerate(thermal_trace):
        data = characterize_maxima(thermal_grid, pixel_radius, in_both_dimensions, as_df=False)
        data['time_step'] = [time_step] * len(data['x_idx'])
        for k, v in data.items():
            all_data[k].extend(v)
    if as_df:
        return _local_max_stats_dict_to_df(all_data)
    return all_data

def local_max_stats_df(ice_grid_output, mltd_radius_px, in_both_dimensions=True):
    return _local_max_stats_fn(ice_grid_output, mltd_radius_px, True, in_both_dimensions=True)

def local_max_stats_dict(ice_grid_output, mltd_radius_px, in_both_dimensions=True):
    return _local_max_stats_fn(ice_grid_output, mltd_radius_px, False, in_both_dimensions=True)

def _local_max_stats_fn(ice_grid_output, mltd_radius_px, as_df, in_both_dimensions=True):
    t_trace = load_3DICE_grid_file(ice_grid_output)
    maxima_data = characterize_maxima_from_trace(t_trace, mltd_radius_px,
                                                 in_both_dimensions=in_both_dimensions, as_df=False)
    if as_df:
        return _local_max_stats_dict_to_df(maxima_data)
    return maxima_data

def _local_max_stats_dict_to_df(maxima_data):
    df = pd.DataFrame(maxima_data)
    df.x_idx = df.x_idx.astype(int)
    df.y_idx = df.y_idx.astype(int)
    df.time_step = df.time_step.astype(int)
    return df

def local_max_stats_to_file(local_max_stats_df, output_file=None):
    with open_file_or_stdout(output_file) as f:
        columns = ['time_step', 'x_idx', 'y_idx', 'temp_xy', 'pos_MLTD', 'neg_MLTD']
        line_frmt = '\t'.join(['{}'] * len(columns)) + '\n'
        f.write(line_frmt.format(*columns))
        for _, row in local_max_stats_df.astype('O').iterrows():
            values = [row[col] for col in columns]
            f.write(line_frmt.format(*values))

def local_max_stats_from_file(local_max_stats_file):
    def _load_pkl():
        return pd.read_pickle(local_max_stats_file)
    def _load_csv():
        return pd.read_csv(local_max_stats_file)
    def _load_txt():
        return pd.read_csv(local_max_stats_file, sep='\t')
    for load_fn in _load_pkl, _load_csv, _load_txt:
        try:
            df = load_fn()
            df.x_idx = df.x_idx.astype(int)
            df.y_idx = df.y_idx.astype(int)
            df.time_step = df.time_step.astype(int)
            df['MLTD'] = df[['pos_MLTD', 'neg_MLTD']].values.max(1)
            return df
        except:
            pass
    raise ValueError('Cannot load stats file...')

################################################################################
########################### Interal Helper Functions ###########################
################################################################################

def _local_max_indices_2D(data):
    axis_0_maxs = set(zip(*argrelmax(data, axis=0)))
    axis_1_maxs = set(zip(*argrelmax(data, axis=1)))
    return list(axis_0_maxs.intersection(axis_1_maxs))

def _local_max_indices_1D(data):
    axis_0_maxs = set(zip(*argrelmax(data, axis=0)))
    axis_1_maxs = set(zip(*argrelmax(data, axis=1)))
    return list(axis_0_maxs.union(axis_1_maxs))

def _circle_region_offsets(radius):
    a = np.arange(radius+1)
    for x, y in zip(*np.where(a[:, np.newaxis]**2 + a**2 <= radius**2)):
        yield from set(((x, y), (x, -y),
                        (-x, y), (-x, -y),))

def _clip_valid_region(data, region):
    return [(x, y) for x, y in region
            if x >= 0 and y >= 0 and
            x < data.shape[0] and y < data.shape[1]
           ]

def _get_ring_offsets(rmin, rmax):
    rmax_offsets = _circle_region_offsets(rmax)
    rmin_offsets = _circle_region_offsets(rmin)
    ring_offsets = set(rmax_offsets).difference(rmin_offsets)
    return ring_offsets

def _range_within_region(grid, region):
    valid_region = _clip_valid_region(grid, region)
    region_grid = grid[tuple(zip(*valid_region))]
    return region_grid.min(), region_grid.max()

################################################################################
########################## Command Line Functionality ##########################
################################################################################

@click.group()
def main():
    pass

@main.command()
@click.argument('ice_grid_output', type=click.Path(exists=True))
@click.argument('mltd_radius_px', type=int)
@click.option('--in_both_dimensions/--in_either_dimension', default=True,
              help='Either find true local maxima, or local max in either dimension')
@click.option('-o', '--output_file', multiple=True, type=click.Path(),
              help='Output file(s)')
def local_max_stats(ice_grid_output, mltd_radius_px, in_both_dimensions=True, output_file=None):
    """Compute the MLTD and temperature at the local maxima of the ICE_GRID_OUTPUT file

    MLTD_RADIUS_PX : the number of pixels over which MLTD should be computed

    \b
    Output file formats :
                default : print human readable format to stdout
                  *.csv : save as comma seperated values format
                  *.pkl : save pickle file of pandas.DataFrame
                      * : human-readable format otherwise
    """
    df = local_max_stats_df(ice_grid_output, mltd_radius_px, in_both_dimensions=in_both_dimensions)

    if len(output_file) == 0:
        outputs = [None]
    else:
        outputs = output_file

    for output_file in outputs:
        # Determine the output type
        if output_file is None:
            # The output is stdout, no file extension available
            ext = None
        else:
            # The output file is a path; get the extension
            _, ext = os.path.splitext(output_file)

        if ext in ['.pkl']:
            df.to_pickle(output_file)
        elif ext in ['.csv']:
            df.to_csv(output_file, index=False)
        elif ext in [None]:
            # Indicates use of stdout; print in human readable format
            local_max_stats_to_file(df, output_file)
        else:
            # Unknown extension, default to human readable format
            local_max_stats_to_file(df, output_file)

if __name__ == '__main__':
    main()
