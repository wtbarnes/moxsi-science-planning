import pathlib

import asdf
import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits
import astropy.table
import astropy.units as u
import astropy.wcs

from astropy.coordinates import SkyCoord
from astropy.visualization import ImageNormalize, LogStretch
from astropy.wcs.utils import wcs_to_celestial_frame
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from mocksipipeline.visualization import plot_labeled_spectrum
from overlappy.io import read_overlappogram
from overlappy.util import color_lat_lon_axes

# OBS_TYPE = 'ar'
OBS_TYPE = 'flare'

IMAGE_LINE_LABELS = False
MAKE_IMAGE_PLOT = False
MAKE_PINHOLE_LINE_PLOT = False
MAKE_SLOT_LINE_PLOT = True

rcparams = {
   'axes.titlesize': 18,
   'axes.labelsize': 18,
   'legend.fontsize': 14,
   'legend.frameon': 'false',
   'xtick.labelsize': 14,
   'ytick.labelsize': 14,
   'xtick.major.pad': 3,
   'xtick.minor.pad': 3,
   'ytick.major.pad': 3,
   'ytick.minor.pad': 3,
   'xtick.direction': 'in',
   'ytick.direction': 'in',
   'savefig.bbox': 'tight',
   'lines.linewidth': 2,
   'patch.linewidth': 2,
}
for k,v in rcparams.items():
    plt.rcParams[k] = v


def add_arrow_from_coords(ax, tail, head, **arrow_kwargs):
    if tail.unit == u.pix:
        transform = 'pixel'
        end_unit = 'pixel'
    else:
        transform = 'world'
        end_unit = 'deg'
    arrow = matplotlib.patches.FancyArrowPatch(tail.to_value(end_unit),
                                               head.to_value(end_unit),
                                               transform=ax.get_transform(transform),
                                               **arrow_kwargs)
    ax.add_patch(arrow)


with asdf.open(f'../../data/line_lists/{OBS_TYPE}-line-table.asdf') as af:
    line_list = af.tree['data']

results_root = pathlib.Path('/Users/wtbarnes/Documents/codes/mocksipipeline/pipeline/results/tess_amir_talk/')

sort_func = lambda x: int(x.name.split('.')[0].split('_')[-1])
spectrogram_pinhole = [
    read_overlappogram(f) for f in 
    sorted((results_root / OBS_TYPE / 'detector_images').glob('spectrogram_pinhole_*.fits'), key=sort_func)
]
spectrogram_slot = [
    read_overlappogram(f) for f in 
    sorted((results_root / OBS_TYPE / 'detector_images').glob('spectrogram_slot_*.fits'), key=sort_func)
]
sort_func = lambda x: int(x.name.split('.')[0].split('_')[-2])
filtergrams = [
    read_overlappogram(f) for f in 
    sorted((results_root / OBS_TYPE / 'detector_images').glob('filtergram_*.fits'), key=sort_func)
]

full_overlap = read_overlappogram(results_root / OBS_TYPE / 'detector_images' / 'all_components.fits')

annot_pt = filtergrams[0][0,...].wcs.array_index_to_world(
    *np.unravel_index(filtergrams[0].data[0].argmax(),
                      filtergrams[0].data[0].shape))

if OBS_TYPE == 'ar':
    plot_title = 'Active Region, 1 h integration'
    colname = 'active\_region'
    thresh = 0.05 * u.ph / (u.h * u.pix)
elif OBS_TYPE == 'flare':
    plot_title = 'Flare, 1 minute integration'
    colname = 'flare\_ext'
    thresh = 0.25 * u.ph / (u.s * u.pix)
else:
    raise ValueError(f'Unrecognized obs type {OBS_TYPE}')

if MAKE_IMAGE_PLOT:
    ## Image Plot
    fig = plt.figure(figsize=(20,20*(1504/2000)))
    ax = fig.add_subplot(projection=full_overlap[0,...].wcs)
    full_overlap[0,...].plot(axes=ax,
                            cmap='hinodexrt',
                            #interpolation='none',
                            norm=ImageNormalize(vmax=1e4,stretch=LogStretch()))

    # Ticks and direction annotationes
    color_lat_lon_axes(ax, lon_color='w', lat_color='w')
    ax.coords[0].set_ticklabel(rotation=90, color='k')
    ax.coords[1].set_ticklabel(color='k')
    ax.coords[0].grid(ls=':', color='w')
    ax.coords[1].grid(ls=':', color='w')
    ax.coords[1].set_axislabel('HPC Longitude', color='k')
    ax.coords[0].set_axislabel('HPC Latitude', color='k')
    for c in ax.coords:
        c.set_ticks(([-1000,0,1000]*u.arcsec).to('deg'))
        c.set_major_formatter('s')
    # Add directional arrow
    add_arrow_from_coords(ax, (1500, -400)*u.arcsec, (1500, 400)*u.arcsec, color='C4', mutation_scale=25,)
    # Add labels to filtergrams
    for fg,label in zip(filtergrams,['Be-thin', 'Be-med', 'Be-thick', 'Al-poly']):
        coord = SkyCoord(Tx=-1200*u.arcsec, Ty=0*u.arcsec, frame=wcs_to_celestial_frame(fg.wcs))
        pix_coord = fg[0].wcs.world_to_pixel(coord)
        ax.annotate(label, pix_coord, ha='center', va='bottom', color='w', fontsize=16)
    # Add wavelength annotations
    annotate_kw = {
        'textcoords': 'offset points',
        'color': 'w',
        'arrowprops': dict(color='w', arrowstyle='-|>', lw=1),
        'horizontalalignment':'center',
        'verticalalignment':'center',
        'rotation':90,
        'fontsize': plt.rcParams['xtick.labelsize']*0.8,
        'weight': 'bold',
    }
    if IMAGE_LINE_LABELS:
        ytext_nom = 40
        for _wcs in [spectrogram_pinhole[10].wcs, spectrogram_pinhole[12].wcs]:
            ytext = ytext_nom
            pos_previous = 0
            for group in line_list.group_by('MOXSI pixel').groups:
                i_sort = np.argsort(group[colname])
                row = group[i_sort[-1]]
                if row[colname] < thresh:
                    continue
                if np.fabs(row['MOXSI pixel'] - pos_previous) < 11:
                    ytext *= -1.1
                else:
                    ytext = ytext_nom
                _annotation = ax.annotate(
                    f'{row["ion name"]}',
                    xy=_wcs.world_to_pixel(annot_pt, row['wavelength'])[:2],
                    xytext=(0, ytext),
                    **annotate_kw
                )
                _annotation.draggable(True)
                pos_previous = row['MOXSI pixel']
    # colorbar
    cax = make_axes_locatable(ax).append_axes("top", size="4%", pad="2%", axes_class=matplotlib.axes.Axes)
    fig.colorbar(
        ax.get_images()[0],
        cax=cax,
        orientation='horizontal',
        location='top',
        extend='max',
        extendfrac=0.02,
        format=matplotlib.ticker.LogFormatterMathtext(base=10.0,),
        ticks=[0, 100, 1000, 1e4],
        label=f'{plot_title} [DN]'
    )
    # fig.savefig(f'./figures/{OBS_TYPE}_image{"_line_labels" if IMAGE_LINE_LABELS else ""}.png')
    plt.show()

if MAKE_PINHOLE_LINE_PLOT or MAKE_SLOT_LINE_PLOT:
    # Pare down line list
    rows = []
    for group in line_list.group_by('MOXSI pixel').groups:
        max_row = group[np.argsort(group[colname])[-1]]
        if max_row[colname] >= thresh:
            rows.append(max_row)
    reduced_line_table = astropy.table.QTable(astropy.table.vstack(rows))
    reduced_line_table.sort('wavelength')

## Spectral plot
if MAKE_PINHOLE_LINE_PLOT:
    irow, _ = spectrogram_pinhole[11][0,...].wcs.world_to_array_index(annot_pt)
    pixel_width = 30  # This is approximate
    cutout_slice = np.s_[:,(irow-pixel_width//2):(irow+pixel_width//2),:]
    bins_to_sum = cutout_slice[1].stop - cutout_slice[1].start
    summed_overlap = full_overlap[cutout_slice].rebin((1,bins_to_sum,1),operation=np.sum)
    bins_to_sum = cutout_slice[1].stop - cutout_slice[1].start
    summed_overlap = full_overlap[cutout_slice].rebin((1,bins_to_sum,1),operation=np.sum)
    components = []
    orders = np.arange(-11, 12,1)
    for order,_component in zip(orders, spectrogram_pinhole):
        components.append(
            _component[cutout_slice].rebin((1, bins_to_sum, 1), operation=np.sum)
        )

    fig = plt.figure(figsize=(20,10))
    ax = components[10][0,0,:].plot(color='C0', ls='--', label='order=-1')
    components[11][0,0,:].plot(axes=ax, color='k', ls=':', label='order=0')
    plot_labeled_spectrum(
        components[12:16],
        summed_overlap,
        reduced_line_table,
        annot_pt,
        threshold=0.1,
        labels=[f'order={order}' for order in orders[12:16]],
        x_lim=(900,2000),
        y_lim=(100,2e5),
        figure=fig,
        axes=ax,
        draggable=True,
    )
    ax.legend(loc=1)
    ax.set_title(f'{plot_title} pinhole')
    # fig.savefig(...)
    plt.show()


if MAKE_SLOT_LINE_PLOT:
    irow, _ = spectrogram_slot[11][0,...].wcs.world_to_array_index(annot_pt)
    slot_pixel_width = 110
    cutout_slice = np.s_[:,(irow-slot_pixel_width//2):(irow+slot_pixel_width//2),:]
    bins_to_sum = cutout_slice[1].stop - cutout_slice[1].start
    summed_overlap = full_overlap[cutout_slice].rebin((1,bins_to_sum,1),operation=np.sum)
    bins_to_sum = cutout_slice[1].stop - cutout_slice[1].start
    summed_overlap = full_overlap[cutout_slice].rebin((1,bins_to_sum,1),operation=np.sum)
    components = []
    orders = np.arange(-11,12,1)
    for order,_component in zip(orders,spectrogram_slot):
        components.append(
            _component[cutout_slice].rebin((1, bins_to_sum, 1),operation=np.sum)
        )

    fig = plt.figure(figsize=(20,10))
    ax = components[10][0,0,:].plot(color='C0', ls='--', label='order=-1')
    components[11][0,0,:].plot(axes=ax, color='k', ls=':', label='order=0')
    plot_labeled_spectrum(
        components[12:16],
        summed_overlap,
        reduced_line_table,
        annot_pt,
        threshold=0.1,
        labels=[f'order={order}' for order in orders[12:16]],
        x_lim=(900,2000),
        y_lim=(1000,3e6),
        figure=fig,
        axes=ax,
        draggable=True,
    )
    ax.legend(loc=1)
    ax.set_title(f'{plot_title} slot')
    # fig.savefig(...)
    plt.show()
