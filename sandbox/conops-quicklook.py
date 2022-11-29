"""
Testing rotational symmetry
---------------------------

This is an experiment to understand whether we can account for 180 flips of the spacecraft by doing transposes 
of the final array in order to do coalignment.
We'll run the following the experiments

- original orientation of -90, flip 180
- original orientation of -75, flip 180
- original orientation of -90 with a grating roll of 25, flip 180

This means we need to build overlappograms with the following roll angles and dispersion angles $(\alpha, \gamma)$:

1. (-90, 0)
2. (90, 0)
3. (-75, 0)
4. (105, 0)
5. (-90, 25)
6. (90, 25)

If we can do this succesfully, 1 and 2 will be the same (within tolerance), 3 and 4, 5 and 6, in the case where we
are applying the correct transpose operation to the first overlappogram in the pair.
"""
import sys
import os

import astropy.wcs
import astropy.units as u
import distributed

from overlappy.reproject import reproject_to_overlappogram
from overlappy.io import write_overlappogram

sys.path.append('../../')
from mocksipipeline.util import read_data_cube
from mocksipipeline.detector.response import SpectrogramChannel, convolve_with_response

EA_FILE = '../../mocksipipeline/data/MOXSI_effarea.genx'
SPECTRAL_CUBE_FILE = 'moxsi-spectral-cube-ar.fits'
DASK_CLUSTER = 'tcp://127.0.0.1:51848'  # start your cluster and get this address


def make_overlappogram(roll_angle, grating_angle, root_dir):
    # Set up directory structure
    dir_name = f'roll{a.value:.0f}_grating{g.value:.0f}'
    top_dir = os.path.join(root_dir, dir_name)
    os.makedirs(top_dir, exist_ok=True)
    # Build overlappogram for each spectral order
    spectral_orders = [-3, -1, 0, 1, 3]
    for order in spectral_orders:
        print(f'Computing overlap for order={order}')
        chan = SpectrogramChannel(order, EA_FILE)
        instr_cube = convolve_with_response(spectral_cube, chan, include_gain=True)
        overlap = reproject_to_overlappogram(
            instr_cube,
            chan.detector_shape,
            observer=observer,
            reference_pixel=(
                (chan.detector_shape[1] + 1)/2,
                (chan.detector_shape[0] + 1)/2,
                1,
            ) * u.pix,
            reference_coord=(
                0 * u.arcsec,
                0 * u.arcsec,
                instr_cube.axis_world_coords(0)[0].to('angstrom')[0],
            ),
            scale=(
                chan.resolution[0],
                chan.resolution[1],
                chan.spectral_resolution,
            ),
            roll_angle=roll_angle,
            dispersion_angle=grating_angle,
            dispersion_axis=0,
            order=chan.spectral_order,
            meta_keys=['CHANNAME'],
            use_dask=True,
            sum_over_lambda=True,
            algorithm='interpolation',
        )
        filename = f'overlappogram_ar_{dir_name}_order{order}.fits'
        write_overlappogram(overlap, os.path.join(top_dir, filename))


if __name__ == '__main__':
    client = distributed.Client(address=DASK_CLUSTER)
    spectral_cube = read_data_cube(SPECTRAL_CUBE_FILE, hdu=1)
    observer = astropy.wcs.utils.wcs_to_celestial_frame(spectral_cube.wcs).observer
    orientations = [
        (-90, 0) * u.deg,
        (90, 0) * u.deg,
        (-75, 0) * u.deg,
        (105, 0) * u.deg,
        (-90, 25) * u.deg,
        (90, 25) * u.deg,
    ]
    for a, g in orientations:
        print(f'Computing overlappogram for roll={a}, dispersion angle={g}')
        make_overlappogram(a, g, 'conops_testing')
