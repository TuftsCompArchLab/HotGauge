#!/usr/bin/env python
# This script creates plots of hottest/coldest temps vs. time
import os
import glob
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import joypy
import click
import itertools
from tqdm import tqdm

import pint
ureg = pint.UnitRegistry()

from HotGauge.thermal import ICETransientSim, severity_metric
from HotGauge.thermal.ICE import load_3DICE_grid_file, load_3DICE_block_file
from HotGauge.thermal.analysis import local_max_stats_from_file

from HotGauge.utils import Floorplan

def find_stk_file(dir_name):
    candidates = glob.glob(os.path.join(dir_name, '*.stk'))
    if len(candidates) == 1:
        return candidates[0]
    return None

def get_stats(data):
    min_ = data.min(axis=(1,2))
    max_ = data.max(axis=(1,2))
    mean = np.mean(data, axis=(1,2))
    return {'min':min_, 'max': max_ , 'mean': mean}

def plot_stats(data, time_slot=None, thresholds=None):
    thresholds = [] if thresholds is None else thresholds
    stats = get_stats(data)
    x = np.arange(0, stats['min'].shape[0])
    if time_slot is not None:
        x = x*time_slot
        plt.xlabel('Seconds')
    else:
        plt.xlabel('Sim Iterations')
    for k,v in sorted(stats.items()):
        plt.plot(x, v, label=k)
    for t in thresholds:
        plt.plot([x[0],x[-1]], [t,t], '--', label=str(t))
    plt.legend()

def get_first_gte(values, threshold):
    for val in values:
        if val >= threshold:
            return val
    return val # default last entry

def plot_dist(data, data_label, time_slot=None, min_val=None, max_val=None, thresholds=None):
    thresholds = [] if thresholds is None else thresholds
    t, w, h = data.shape
    dflat = data.reshape(t, w*h)
    steps = np.array([1,2,5,10,20,25,50,100,200])
    if time_slot is None:
        iter_mul = 1
        iter_unit = 'Sim Iterations'
    else:
        iter_mul = time_slot * 1000
        iter_unit = 'Time (ms)'
        steps = steps * iter_mul
    step = int(get_first_gte(steps, t * iter_mul / 10) / iter_mul)
    labels = [round(val*iter_mul, 4) if val%step==0 else None for val in range(t)]
    dfs = [pd.DataFrame({data_label:list(dflat[i]), iter_unit:i*iter_mul, 'iter':i})
           for i in range(len(dflat))]
    df = pd.concat(dfs, ignore_index=True)

    # Make t=0 be at bottom
    # Reverse the order of the time column
    df[iter_unit] *= -1
    # Reverse the order of the labels
    labels.reverse()

    min_val = min_val if min_val is not None else df[data_label].min()
    max_val = max_val if max_val is not None else df[data_label].max()
    plt.rc("font", size=16)
    fig, axes = joypy.joyplot(df, column=data_label, by=iter_unit, labels=labels,
                  # aesthetics...
                  grid=False, # dont show x/y grids
                  ylim='own', # make height of each slice the same
                  linewidth=1, # thinner lines
                  overlap=1.75, # '3D' angle... higher = flatter
                  fade=True, # make distributions in read faded
                  range_style='own', # only plot to tmax for given time_slot
                  colormap=matplotlib.cm.autumn_r, # red -> yellow color scheme
                  figsize=(6.4,6.4), # square figure seems to look better
                  x_range=[min_val, max_val]
                  )
    plt.xlabel(data_label)
    # The first n axes are each row/slice within the data
    ax = axes[-1] # meta axes for the axes that make up each sub-plot
    ax.yaxis.set_label_position("left")
    ax.set_ylabel(iter_unit, labelpad=50)
    ax.yaxis.set_ticks([]) # no need to enumerate the other axes
    ax.yaxis.set_visible(True)

    for thresh,color in zip(thresholds, itertools.cycle('kbrgy')):
        ax.plot([thresh,thresh],[0,1.0], '{}--'.format(color))

def plot_temps(ttrace, axes=None, tmin=None, tmax=None):
    if axes is None:
        axes = [plt.subplots()[1] for _ in range(len(ttrace))]
    assert len(axes) == len(ttrace), \
           '{} axes provided for {} timesteps'.format(len(axes), len(ttrace))

    tmin = tmin if tmin is not None else ttrace.min()
    tmax = tmax if tmax is not None else ttrace.max()
    for i, (t_i, ax) in tqdm(enumerate(zip(ttrace, axes)), total=len(axes),
                             desc='ttrace timestep thermal image'):
        im = ax.imshow(t_i, interpolation=None, vmin=tmin, vmax=tmax, cmap='plasma')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Temperature (K)', rotation=-90, va="bottom")
    return axes

def plot_power_density(pd_trace, axes=None, pd_min=None, pd_max=None):
    if axes is None:
        axes = [plt.subplots()[1] for _ in range(len(pd_trace))]
    assert len(axes) == len(pd_trace), \
           '{} axes provided for {} timesteps'.format(len(axes), len(pd_trace))

    pd_min = pd_min if pd_min is not None else pd_trace.min()
    pd_max = pd_max if pd_max is not None else pd_trace.max()
    for i, (t_i, ax) in tqdm(enumerate(zip(pd_trace, axes)), total=len(axes),
                             desc='pd_trace timestep power image'):
        im = ax.imshow(t_i, interpolation=None, vmin=pd_min, vmax=pd_max, cmap='plasma')
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel('Power Density (W/cm^2)', rotation=-90, va="bottom")
    return axes

def label_hotspot(ax, xc, yc, circle_radius):
    ax.text(yc,xc, '+', ha="center", va="center")
    c = plt.Circle((yc ,xc), circle_radius, color='r', fill=False)
    ax.add_artist(c)
        
@click.group()
def cli():
    pass

@cli.command()
@click.argument('input_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--output', type=click.File(mode='wb'), default=None)
@click.option('--plot_type', type=click.Choice(['stats', 'dist', 'delta']), default='dist')
@click.option('--data_type', type=click.Choice(['power', 'temperature']), default='temperature')
@click.option('--min_val', type=float, default=None)
@click.option('--max_val', type=float, default=None)
@click.option('--threshold', '-t', type=float, multiple=True)
def grid_transient(input_file, output=None, plot_type='dist', data_type='power', min_val=None, max_val=None, threshold=None):
    data = load_3DICE_grid_file(input_file, convert_K_to_C=data_type=='temperature')

    stk_file = find_stk_file(os.path.dirname(input_file))
    if stk_file is not None:
        _, slot = ICETransientSim.get_step_slot(stk_file)
        cell_size = ICETransientSim.get_cell_size(stk_file)
    else:
        slot = None
        cell_size = None
    if data_type=='temperature':
        data_label = 'Temperature'
        unit = ureg['C'].units
        factor = 1
    else:
        data_label = 'Power Density'
        if cell_size is not None:
            length, width = cell_size
            max_p = data.max()
            cell_size = length * ureg['um'] * width * ureg['um']
            max_value = (max_p * (ureg['W'] / cell_size)).to('W/um^2')
            unit = max_value.units
            factor = max_p / max_value.magnitude
    data_label = '{}({:~P})'.format(data_label, unit)
    if plot_type == 'stats':
        plot_stats(data*factor, time_slot=slot, thresholds=threshold)
        dmin, dmax = plt.ylim()
        min_val = dmin if min_val is None else min_val
        max_val = dmax if max_val is None else max_val
        plt.ylim(min_val, max_val)
    elif plot_type == 'dist':
        plot_dist(data*factor,
                  data_label=data_label, thresholds=threshold,
                  time_slot=slot, min_val=min_val, max_val=max_val)
    elif plot_type == 'delta':
        data_label = 'Change in {}'.format(data_label)
        data = np.diff(data, axis=0)
        plot_delta_dist(data*factor,
                  data_label=data_label, thresholds=threshold,
                  time_slot=slot, min_val=min_val, max_val=max_val)
    else:
        raise ValueError('Invalid plot_type: {}'.format(plot_type))
    if output is None:
        plt.show()
    else:
        plt.savefig(output)

def _get_px_size_mm_from_flp_ttrace(flp, ttrace):
    old_frmt = flp.frmt
    flp.frmt = '3D-ICE' # make sure units in um
    px_width = flp.width / ttrace[0].shape[1]
    px_height = flp.height / ttrace[0].shape[0]
    assert abs(px_height-px_width) < 1, 'Pixels must be square for this analysis'
    px_height_mm = px_height / 1000.0
    flp.frmt = old_frmt
    return px_height_mm

@cli.command()
@click.argument('ice_grid_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('floorplan_file', type=click.Path(exists=True))
@click.argument('mltd_radius', type=float)
@click.option('-o', '--output-format', type=str, default='thermal_trace_{step:04}.png',
              help='Output file name format (see help for detailed options)')
@click.option('-l', '--local_max_stats', type=click.Path(exists=True, dir_okay=False),
              help='Local max stats file')
@click.option('-s', '--severity_threshold', type=click.FLOAT,  default=0.75,
              help='Severity threshold for labling hotspots')
@click.option('--celsius/--kelvin', default=True)
@click.option('--tmin', type=float, default=None)
@click.option('--tmax', type=float, default=None)
def hotspot_locations(ice_grid_file, floorplan_file, mltd_radius, output_format, local_max_stats,
                      severity_threshold, celsius, tmin, tmax):
    """Plot the thermal trace with hotspot locations

    \b
    Output file name format options:
        {step}               : time step of thermal simulation
        {severity_threshold} : threshold for severity used to filter hotspots
    """
    ttrace = load_3DICE_grid_file(ice_grid_file, convert_K_to_C=celsius)
    
    flp = Floorplan.from_file(floorplan_file)
    px_height_mm = _get_px_size_mm_from_flp_ttrace(flp, ttrace)
    mltd_px = mltd_radius / px_height_mm
    
    axes = plot_temps(ttrace, tmin=tmin, tmax=tmax)
    if local_max_stats is not None:
        stats_df = local_max_stats_from_file(local_max_stats)
        stats_df['hotspot_severity'] = severity_metric(stats_df['MLTD'], stats_df['temp_xy'])
        stats_df = stats_df[stats_df.hotspot_severity >= severity_threshold]
        for step, ax in enumerate(axes):
            candidates_df = stats_df[stats_df.time_step == step]
            candidates_df.apply(lambda r: label_hotspot(ax, r['x_idx'], r['y_idx'], mltd_px), axis=1)
            
    for step, ax in enumerate(axes):
        fig = ax.get_figure()
        figname =  output_format.format(step=step, severity_threshold=severity_threshold)
        fig.savefig(figname)

@cli.command()
@click.argument('ice_power_grid_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('floorplan_file', type=click.Path(exists=True))
@click.option('-o', '--output-format', type=str, default='power_trace_{step:04}.png',
              help='Output file name format (see help for detailed options)')
@click.option('--pd_min', type=float, default=None)
@click.option('--pd_max', type=float, default=None)
def power_map(ice_power_grid_file, floorplan_file, output_format, pd_min, pd_max):
    """Plot the power trace in W/cm^2

    \b
    Output file name format options:
        {step}               : time step of thermal simulation
    """
    ptrace = load_3DICE_grid_file(ice_power_grid_file, convert_K_to_C=False)
    
    flp = Floorplan.from_file(floorplan_file)
    px_height_mm = _get_px_size_mm_from_flp_ttrace(flp, ptrace)
    px_height_cm = 10 * px_height_mm
    px_area_cm_sq = px_height_cm * px_height_cm
    
    axes = plot_power_density(ptrace / px_area_cm_sq, pd_min=pd_min, pd_max=pd_max)
            
    for step, ax in enumerate(axes):
        fig = ax.get_figure()
        figname =  output_format.format(step=step)
        fig.savefig(figname)

if __name__ == '__main__':
    cli()
