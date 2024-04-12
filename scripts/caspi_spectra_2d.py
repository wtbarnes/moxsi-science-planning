"""
Build two-dimensional versions of 1D spectra that include HXR component
"""
import pathlib

import asdf
import astropy.units as u
import ndcube
import numpy as np
import sunpy.io._fits as sunpy_fits
import sunpy.map
import astropy.wcs

from astropy.coordinates import SkyCoord
from scipy.interpolate import PchipInterpolator
from sunpy.coordinates import get_earth, Helioprojective

from synthesizAR.instruments.util import add_wave_keys_to_header
from mocksipipeline.instrument.configuration import moxsi_slot

CASPI_FLARE_SPECTRA_FILE = '../data/reference_spectra/caspi_flare_spectra.asdf'
OUTPUT_DIR = '/Users/wtbarnes/Documents/codes/mocksipipeline/pipeline/results/'


if __name__ == '__main__':
    # Read in 1D spectra
    goes_class = ['a1', 'b1', 'b7', 'm1', 'm5', 'x5']
    flare_spectra = []
    with asdf.open('../data/reference_spectra/caspi_flare_spectra.asdf', copy_arrays=True) as af:
        for gc in goes_class:
            flare_spectra.append((gc, ndcube.NDCube(af.tree[gc]['data'], wcs=af.tree[gc]['wcs'])))
    flare_spectra = ndcube.NDCollection(flare_spectra, aligned_axes=(0,))

    # Interpolate each spectra to an appropriate wavelength grid
    wave_min = 0.1 * u.AA
    wave_max = 80 * u.AA
    flare_spectra_interp = {}
    for k in flare_spectra:
        energy_axis = flare_spectra[k].axis_world_coords(0)[0]
        wavelength_axis = energy_axis.to('AA', equivalencies=u.spectral())
        delta_wave = np.min(np.fabs(np.diff(wavelength_axis))) * 50  # Don't need the full resolution
        wavelength_axis_norm = u.Quantity(np.arange(wave_min.to_value('AA'), wave_max.to_value('AA'), delta_wave.to_value('AA')), 'AA')
        f_interp = PchipInterpolator(energy_axis.to_value('keV'), flare_spectra[k].data, extrapolate=True)
        data_interp = f_interp(wavelength_axis_norm.to_value('keV', equivalencies=u.spectral())) * flare_spectra[k].unit
        flare_spectra_interp[k] = (wavelength_axis_norm, data_interp)

    # Embed 1D spectra at origin in a 2-by-2 kernel
    earth_observer = get_earth('2020-01-01')
    ref_coord = SkyCoord(0, 0, unit='arcsec', frame=Helioprojective(observer=earth_observer))
    array_size = (30, 130)  # this should always be even and be >> bigger than the PSF size
    flare_spectra_interp_cubes = {}
    for k in flare_spectra_interp:
        wavelength, data = flare_spectra_interp[k]
        data_array = np.zeros(wavelength.shape+array_size) * data.unit
        data_array[:, array_size[0]//2-1:array_size[0]//2+1, array_size[1]//2-1:array_size[1]//2+1] = data[:, None, None]
        header = sunpy.map.make_fitswcs_header(
            array_size,
            ref_coord,
            scale=moxsi_slot.optical_design.spatial_plate_scale,
            unit=data.unit,
        )
        header = add_wave_keys_to_header(wavelength, header)
        wcs = astropy.wcs.WCS(header=header)
        flare_spectra_interp_cubes[k] = ndcube.NDCube(data_array, wcs=wcs, meta=header)

    # Write out spectra to files
    results_root = pathlib.Path(OUTPUT_DIR)
    for k in flare_spectra_interp_cubes:
        results_dir = results_root / f'caspi_spectra_{k}'
        results_dir.mkdir(exist_ok=True)
        sunpy_fits.write(results_dir / 'spectral_cube.fits',
                         flare_spectra_interp_cubes[k].data,
                         flare_spectra_interp_cubes[k].meta,
                         overwrite=True)