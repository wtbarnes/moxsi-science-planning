"""
Dispersed Filter Trade Study
---------------------------

This is an experiment to understand the throughput for several 
different configurations of the dispersed transmission filter.
This iterates over both different materials as well as different
thicknesses.

The different configurations are:

0. Al, 100 nm w/o Au-Cr layer
1. Al, 100 nm
2. Al oxide, 100 nm
3. Al oxide, 200 nm
4. Al oxide with polymide, 200 nm
5. Al oxide with polymide, 300 nm


"""
import os

import astropy.units as u
import distributed
from overlappy.io import write_overlappogram

from mocksipipeline.detector.response import SpectrogramChannel
from mocksipipeline.detector.filter import ThinFilmFilter
from mocksipipeline.util import read_data_cube
from mocksipipeline.detector.component import DetectorComponent

SPECTRAL_CUBE_FILE = 'data/moxsi-spectral-cube-ar.fits'
DASK_CLUSTER = "tcp://127.0.0.1:36281"  # start your cluster and get this address
ROOT_DIR = '/home/ubuntu/pipeline-runs/dispersed-filter-trade-study'


if __name__ == '__main__':
    # Setup client
    client = distributed.Client(address=DASK_CLUSTER)
    # Read in spectral cube file
    spectral_cube = read_data_cube(SPECTRAL_CUBE_FILE, hdu=1)
    # Setup filters
    ## Case 1
    al_thin = ThinFilmFilter(elements='Al', thickness=100*u.nm)
    ## Case 2
    al_thin_oxide = [
        ThinFilmFilter(elements='Al', thickness=93*u.nm),
        ThinFilmFilter(elements=['Al','O'], quantities=[2,3], thickness=7*u.nm),
    ]
    ## Case 3
    al_thick_oxide = [
        ThinFilmFilter(elements='Al', thickness=192*u.nm),
        ThinFilmFilter(elements=['Al','O'], quantities=[2,3], thickness=8*u.nm),
    ]
    ## Case 4
    polymide = ThinFilmFilter(elements=['C','H','N','O'],
                              quantities=[22,10,2,5],
                              density=1.43*u.g/u.cm**3,
                              thickness=100*u.nm)
    al_poly_thin_oxide = al_thin_oxide + [polymide]
    ## Case 5
    al_poly_thick_oxide = al_thick_oxide + [polymide]
    filter_list = [
        (al_thin, False),
        (al_thin, True),
        (al_thin_oxide, True),
        (al_thick_oxide, True),
        (al_poly_thin_oxide, True),
        (al_poly_thick_oxide, True),
    ]
    for tff,au_cr in filter_list:
        # Build overlappogram for each spectral order
        spectral_orders = [-3, -1, 0, 1, 3]
        for order in spectral_orders:
            # Set up channel
            chan = SpectrogramChannel(order, tff)
            # Setup directory structure and filenames
            dir_name = ''.join(chan.filter_label.split(' '))
            dir_name = '_'.join(dir_name.split(','))
            if not au_cr:
                dir_name += '_no_au_cr'
            top_dir = os.path.join(ROOT_DIR, dir_name)
            os.makedirs(top_dir, exist_ok=True)
            filename = f'overlappogram_ar_{dir_name}_order{order}.fits'
            # Compute overlappogram
            print(f'Computing overlap for {chan.filter_label}, order={order}')
            dc = DetectorComponent(chan)
            overlap = dc.compute(spectral_cube, include_gain=False)
            # Save file
            filename = os.path.join(top_dir, filename)
            write_overlappogram(overlap, filename)
