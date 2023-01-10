"""
AR-flare Comparison Study
---------------------------

This is an experiment to compare the dispersed images for the case of
an active region and an X1 flaring case.
This will use the thick Al-poly (with an oxide layer) filter 
"""
import os

import aiapy
import astropy.time
import astropy.units as u
import distributed
import numpy as np
from overlappy.io import write_overlappogram

from mocksipipeline.physics.dem.data_prep import DataQuery
from mocksipipeline.physics.spectral import SpectralModel
from mocksipipeline.detector.filter import ThinFilmFilter
from mocksipipeline.detector.component import DispersedComponent

DASK_CLUSTER = ""  # start your cluster and get this address
ROOT_DIR = '/home/ubuntu/pipeline-runs/'


if __name__ == '__main__':
    # Set up client
    client = distributed.Client(address=DASK_CLUSTER)
    # Get needed AIA data for calibration
    correction_table = aiapy.calibrate.util.get_correction_table() 
    error_table = aiapy.calibrate.util.get_error_table(error_table=os.path.join(ROOT_DIR, 'aia_V3_error_table.txt'))
    # Set up temperature bin edges for DEM
    temperature_bin_edges = 10**np.arange(5.5, 7.6, 0.1) * u.K
    # Set up filter
    al_thick_oxide = [
        ThinFilmFilter(elements='Al', thickness=192*u.nm),
        ThinFilmFilter(elements=['Al','O'], quantities=[2,3], thickness=8*u.nm),
    ]
    polymide = ThinFilmFilter(elements=['C','H','N','O'],
                              quantities=[22,10,2,5],
                              density=1.43*u.g/u.cm**3,
                              thickness=100*u.nm)
    al_poly_thick_oxide = al_thick_oxide + [polymide]
    ##################
    #### AR Case #####
    ##################
    print('Running active region case')
    ar_time = astropy.time.Time('2020-11-09 18:00:00')
    dir_name = f'ar_{ar_time.isot}'
    top_dir = os.path.join(ROOT_DIR, dir_name)
    os.makedirs(top_dir, exist_ok=True)
    # DEM
    print('Computing DEM')
    pointing_table = aiapy.calibrate.util.get_pointing_table(ar_time-6*u.h, ar_time+6*u.h)
    dq = DataQuery(top_dir,
                   ar_time,
                   aia_error_table=error_table,
                   aia_correction_table=correction_table,
                   aia_pointing_table=pointing_table,
                   temperature_bin_edges=temperature_bin_edges)
    dem_cube = dq.run()
    # Spectral cube
    print('Computing spectral cube')
    spec_model = SpectralModel()
    spec_cube = spec_model.run(dem_cube['em'], dq.celestial_wcs)
    # Overlappogram
    print('Computing dispersed components for all orders')
    dc = DispersedComponent(al_poly_thick_oxide)
    components = dc.compute(spec_cube)
    for k in components:
        filename = os.path.join(top_dir, f'overlappogram_{dir_name}_order{k}.fits')
        write_overlappogram(components[k], filename)
    ##################
    ### Flare Case ###
    ##################
    flare_time = astropy.time.Time('2022-03-30T17:55')
    dir_name = f'flare_{flare_time.isot}'
    top_dir = os.path.join(ROOT_DIR, dir_name)
    os.makedirs(top_dir, exist_ok=True)
    # DEM
    print('Computing DEM')
    pointing_table = aiapy.calibrate.util.get_pointing_table(flare_time-6*u.h, flare_time+6*u.h)
    dq = DataQuery(top_dir,
                   flare_time,
                   aia_error_table=error_table,
                   aia_correction_table=correction_table,
                   aia_pointing_table=pointing_table,
                   temperature_bin_edges=temperature_bin_edges)
    dem_cube = dq.run()
    # Spectral cube
    print('Computing spectral cube')
    spec_model = SpectralModel()
    spec_cube = spec_model.run(dem_cube['em'], dq.celestial_wcs)
    # Overlappogram
    print('Computing dispersed components for all orders')
    dc = DispersedComponent(al_poly_thick_oxide)
    components = dc.compute(spec_cube)
    for k in components:
        filename = os.path.join(top_dir, f'overlappogram_{dir_name}_order{k}.fits')
        write_overlappogram(components[k], filename)
