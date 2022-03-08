import numpy as np
import astropy.units as u
from scipy.io import readsav
import sys
import matplotlib.pyplot as plt
from mpl_animators import ArrayAnimatorWCS
from astropy.visualization import ImageNormalize,LogStretch
import sunpy.map
import ndcube
from sunpy.coordinates import get_earth

from util import (make_moxsi_ndcube, 
                  construct_overlappogram,
                  construct_dispersion_matrix,
                  construct_rot_matrix, 
                  overlappogram_wcs,
                  color_lat_lon_axes)


if __name__ == '__main__':
    observer = get_earth()
    norm=ImageNormalize(vmin=0, vmax=10, stretch=LogStretch())
    cmap='hinodexrt'
    plot_props = {'cmap':cmap, 'norm':norm}

    savdata = readsav("../data/forDan_MOXSI_DATA_09112020_0440_feldman.sav")

    moxsi_cube = make_moxsi_ndcube(
        savdata["moxsi1_img"], savdata["cubixss_wave"] * u.angstrom
    )
    iw_rank = moxsi_cube.data.mean(axis=(1,2)).argsort()[::-1]
    new_data = np.zeros(moxsi_cube.data.shape)
    for i in iw_rank[:50:2]:
        new_data[i, :, :] = moxsi_cube.data[i, :, :]
    moxsi_cube = make_moxsi_ndcube(new_data, savdata['cubixss_wave']*u.angstrom)

    moxsi_overlap = construct_overlappogram(
        moxsi_cube, 
        roll_angle=0*u.deg,
        dispersion_angle=10*u.deg,
        observer=observer, 
        order=1,
        correlate_p12_with_wave=True,
    )

    coord_params = {
        'em.wl':{
            'grid': {
                'grid_type': 'contours',
                'color': 'C1',
            },
            'format_unit': 'Angstrom',
            'major_formatter': 'x.x',
            'ticks': {
                'color': 'C1',
            }
        },
        'custom:pos.helioprojective.lon': {
            'grid': {
                'grid_type': 'contours',
                'color': 'C0',
            },
            'ticks': {
                'color': 'C0',
            }
        },
        'custom:pos.helioprojective.lat': {
            'grid': {
                'grid_type': 'contours',
                'color': 'C2',
            },
            'ticks': {
                'color': 'C2',
            }
        },
    }
#    moxsi_overlap_cutout = moxsi_overlap[400:600,400:600, :]
#
#    ani = moxsi_overlap_cutout.plot(**plot_props)
#    plt.show()

#    ani = ArrayAnimatorWCS(
#        moxsi_overlap[:300, :, :].data,
#        moxsi_overlap[:300, :, :].wcs.low_level_wcs,
#        ("x", "y", 0),
#        **plot_props,
#        coord_params=coord_params
#    )
#    ani.get_animation(interval=25).save("moxsi-cutout.mp4", dpi=100)
