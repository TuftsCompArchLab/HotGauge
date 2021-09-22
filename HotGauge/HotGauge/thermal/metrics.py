import os
import tempfile
import logging

from mpl_toolkits import mplot3d
from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import numpy as np
import click
from tqdm import tqdm

def _sigmoid_fn(x_offset, y_shift, slope, amplitude=None):
    """
    x_offset: center of sigmoid in x direction
    y_shift: minimum value in y dimension
    amplitude: None -> Auto scale up to 1.0
               float -> use specific value
    """
    if amplitude is None:
        amplitude = 1-y_shift
    def sigmoid(x):
        """ General sigmoid function
        c adjusts x offset
        d adjusts slope """
        y = amplitude / (1 + np.exp((x-x_offset)*(-1*slope))) + y_shift
        return y
    return sigmoid

T_metric = _sigmoid_fn(60, 0.35, 1.0/20)
MLTD_metric = _sigmoid_fn(15, -0.25, 1.0/5)

T_BREAKDOWN = 115
T_metric_breakdown = _sigmoid_fn(T_BREAKDOWN, 0, 1.0/12, 2)

def severity_metric(MLTD, T):
    A = MLTD_metric(MLTD)
    B0 = T_metric(T)
    B1 = T_metric_breakdown(T)
    product = A * B0
    output = product + B1
    return np.clip(output, 0, 1)

def _whitespace_bbox(fname):
    im = Image.open(fname)
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    diff = ImageChops.difference(im, bg)
    diff = ImageChops.add(diff, diff, 2.0, -1)
    im.close()
    return diff.getbbox()

def _bbox_union(bb1, bb2):
    return (min(bb1[0], bb2[0]), min(bb1[1], bb2[1]), max(bb1[2], bb2[2]), max(bb1[3], bb2[3]))

def _cropped_img_from_file(image_name, crop_box):
    img = Image.open(image_name)
    img = img.crop(crop_box)
    return img

def create_severity_plot(severity_fn=severity_metric, **kwargs):
    TMIN = kwargs.pop('TMIN', 25)
    TMAX = kwargs.pop('TMAX', 125)
    MLTD_MAX = kwargs.pop('MLTD_MAX', 40)
    color_scheme = kwargs.pop('color_scheme', 0)

    if isinstance(color_scheme, str):
        color_scheme = COLOR_SCHEMES.index(color_scheme)

    T_GRID, MLTD_GRID = np.meshgrid(range(TMIN, TMAX), range(0, MLTD_MAX))
    METRIC_GRID = severity_fn(MLTD_GRID, T_GRID)

    f = plt.figure()
    ax = plt.axes(projection='3d')
    if color_scheme == 0:
        ax.plot_surface(T_GRID, MLTD_GRID, METRIC_GRID, cmap='gist_earth')
    elif color_scheme == 1:
        ax.contour3D(T_GRID, MLTD_GRID, METRIC_GRID, 55, cmap='jet')
    ax.set_xlabel('\nTemperature ($^{\circ}C$)', linespacing=1.5)
    ax.set_ylabel('\nMLTD$_{1mm}$ ($\Delta^{\circ}C$)', linespacing=1.5)
    ax.set_zlabel('\nHotspot Severity', linespacing=1.5)

    return f, ax

@click.group()
def cli():
    pass

COLOR_SCHEMES = ['gist_earth', 'jet']
@cli.command()
@click.option('-t', '--temp_range', nargs=2, type=click.INT, default=(25, 125),
              help='Range of temperature axis ( e.g. -t 25 125 )')
@click.option('-m', '--mltd_max', type=click.INT, default=45,
              help='Max value on MLTD axis (minimum is 0)')
@click.option('-o', '--output-format', type=str, default='severity_metric_{index:03}.png',
              help='Output file name format (see help for detailed options)')
@click.option('-a', '--angle', type=click.FLOAT, default=0,
              help='Angle by which to rotate 3D-plot in degrees [-44:44]')
@click.option('-f', '--font-size', type=click.FLOAT, default=16,
              help='Font size to use for plot')
@click.option('-d', '--dpi', type=click.FLOAT, default=300,
              help='DPI to save figure at')
@click.option('-c', '--color_scheme', type=click.Choice(COLOR_SCHEMES), default=COLOR_SCHEMES[0],
              help='Color scheme to use, see detail below')
@click.option('--crop/--no-crop', default=True)
def graph_metric(temp_range, mltd_max, output_format, angle, font_size, dpi, color_scheme, crop):
    """Graph the severity metric

    \b
    Output file name format options:
        {angle}        : angle of the rotation (0 degrees looks directly at corner of axes)
        {font_size}    : specified font size
        {index}        : index of figure being produced
        {color_scheme} : selected color scheme
        {crop}         : True or False
        {crop_str}     : '_cropped' or ''
    """
    plt.rc('font', size=font_size)

    output_file = output_format.format(angle=angle, font_size=font_size, index=0,
                                       total_frames=1, dpi=dpi,
                                       color_scheme=color_scheme, crop=crop,
                                       crop_str='_cropped' if crop else '')

    tmin, tmax = temp_range
    f, ax = create_severity_plot(TMIN=tmin, TMAX=tmax, MLTD_max=mltd_max, color_scheme=color_scheme)

    ax.view_init(30, angle + 225)

    plt.draw()

    if crop == False:
        plt.savefig(output_file, dpi=dpi)
        return

    _, ext = os.path.splitext(output_file)
    # Otherwise save to a temporary file and then save it to the output file location
    with tempfile.NamedTemporaryFile(suffix=ext) as fp:
        plt.savefig(fp.name, dpi=dpi)
        try:
            crop_box = _whitespace_bbox(fp.name)
        except OSError as e:
            msg = 'Failed to determine crop box with extension {}... Saving directly to {}'
            logging.warn(msg.format(ext, output_file))
            plt.savefig(output_file, dpi=dpi)
            return

        try:
            img = _cropped_img_from_file(fp.name, crop_box)
            img.save(output_file)
        except OSError as e:
            msg = 'Failed to crop image with extension {} to box {}... Saving directly to {}'
            logging.warn(msg.format(ext, crop_box, output_file))
            plt.savefig(output_file, dpi=dpi)
            return

@cli.command()
@click.option('-t', '--temp_range', nargs=2, type=click.INT, default=(25, 125),
              help='Range of temperature axis ( e.g. -t 25 125 )')
@click.option('-m', '--mltd_max', type=click.INT, default=45,
              help='Max value on MLTD axis (minimum is 0)')
@click.option('-o', '--output-format', type=str, default='metric_',
              help='Output file name format (see help for detailed options)')
@click.option('-a', '--angle_range', nargs=2, type=click.FLOAT, default=(-44,44),
              help='Angles by which to rotate 3D-plot in degrees [-44:44], e.g -a -44 44')
@click.option('-f', '--font-size', type=click.FLOAT, default=16,
              help='Font size to use for plot')
@click.option('-d', '--dpi', type=click.FLOAT, default=300,
              help='DPI to save figure at')
@click.option('-c', '--color_scheme', type=click.Choice(COLOR_SCHEMES), default=COLOR_SCHEMES[0],
              help='Color scheme to use')
@click.option('--crop/--no-crop', default=True)
@click.option('--smooth/--linear', default=True,
              help='Smooth uses a sin wave; good for animations. Linear equally spaces angles.')
@click.option('-t', '--total_frames', type=click.INT, default=120,
              help='Total number of images to produce')
def graph_metric_range(temp_range, mltd_max, output_format, angle_range, font_size, dpi,
                       color_scheme, crop, smooth, total_frames):
    """ Graph the severity metric over a range of angles

    \b
    Output file name format options:
        {angle}        : angle of the rotation (0 degrees looks directly at corner of axes)
        {font_size}    : specified font size
        {index}        : index of figure being produced
        {total_frames} : total number of frames produced
        {color_scheme} : selected color scheme
        {crop}         : True or False
        {crop_str}     : '_cropped' or ''

    color_scheme : see options below
    """
    plt.rc('font', size=font_size)

    tmin, tmax = temp_range
    f, ax = create_severity_plot(TMIN=tmin, TMAX=tmax, MLTD_max=mltd_max, color_scheme=color_scheme)

    images, crop_box = [], None

    min_angle, max_angle = angle_range
    if smooth:
        x = np.linspace(0, 2*np.pi, total_frames)
        angles = np.cos(x)*((min_angle-max_angle)/2) + (min_angle + max_angle) / 2
    else:
        angles = np.append(np.linspace(min_angle, max_angle, int(total_frames/2)),
                           np.linspace(max_angle, min_angle, int((total_frames+0.5)/2)))
    for index, angle in tqdm(enumerate(angles), total=total_frames, desc='Generating images'):
        output_file = output_format.format(angle=angle, font_size=font_size, index=index,
                                           total_frames=total_frames, dpi=dpi,
                                           color_scheme=color_scheme, crop=crop,
                                           crop_str='_cropped' if crop else '')

        _, ext = os.path.splitext(output_file)
        ax.view_init(30, angle + 225)
        plt.draw()
        tmpf = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        plt.savefig(tmpf.name, dpi=dpi)
        try:
            this_crop_box = _whitespace_bbox(tmpf.name)
        except OSError as e:
            msg = 'Failed to determine crop box with extension {}... Saving directly to {}'
            logging.warn(msg.format(ext, output_file))
            plt.savefig(output_file, dpi=dpi)
            tmpf.close()
            os.unlink(tmpf.name)
            continue

        if crop_box is None:
            crop_box = this_crop_box
        else:
            crop_box = _bbox_union(crop_box, this_crop_box)
        images.append((tmpf, output_file))

    for tmpf, final in tqdm(images, desc='Cropping temp files and cleaning up'):
        try:
            img = _cropped_img_from_file(tmpf.name, crop_box)
            img.save(final)
        except OSError as e:
            msg = 'Failed to crop image with extension {} to box {}... Saving directly to {}'
            logging.warn(msg.format(ext, crop_box, final))
            plt.savefig(final, dpi=dpi)
            return
        finally:
            tmpf.close()
            os.unlink(tmpf.name)

if __name__ == '__main__':
    cli()
