"""
Build two-dimensional versions of 1D spectra that include HXR component
"""
import pathlib

import asdf
import astropy.units as u
import ndcube
import sunpy.io._fits as sunpy_fits

from astropy.coordinates import SkyCoord
from astropy.wcs.utils import wcs_to_celestial_frame
from scipy.interpolate import PchipInterpolator

from mocksipipeline.util import read_data_cube


CASPI_FLARE_SPECTRA_FILE = '../data/reference_spectra/caspi_flare_spectra.asdf'
FULL_DISK_SPECTRAL_CUBE_FILE = '/Users/wtbarnes/Documents/codes/mocksipipeline/pipeline/results/hic-ar-1h/spectral_cube.fits'
OUTPUT_DIR = '/Users/wtbarnes/Documents/codes/mocksipipeline/pipeline/results/'


if __name__ == '__main__':
    # Read in 1D spectra
    goes_class = ['x5'] #['a1', 'm1']
    flare_spectra = []
    with asdf.open('../data/reference_spectra/caspi_flare_spectra.asdf', copy_arrays=True) as af:
        for gc in goes_class:
            flare_spectra.append((gc, ndcube.NDCube(af.tree[gc]['data'], wcs=af.tree[gc]['wcs'])))
    flare_spectra = ndcube.NDCollection(flare_spectra, aligned_axes=(0,))

    # Load relevant spectral cube for AR observation
    spec_cube = read_data_cube(FULL_DISK_SPECTRAL_CUBE_FILE)
    # Extract wavelength axis
    wavelength_axis_norm = spec_cube.axis_world_coords(0)[0]

    # Interpolate each spectra to an appropriate wavelength grid
    flare_spectra_interp = {}
    for k in flare_spectra:
        energy_axis = flare_spectra[k].axis_world_coords(0)[0]
        wavelength_axis = energy_axis.to('AA', equivalencies=u.spectral())
        f_interp = PchipInterpolator(energy_axis.to_value('keV'), flare_spectra[k].data, extrapolate=True)
        data_interp = f_interp(wavelength_axis_norm.to_value('keV', equivalencies=u.spectral())) * flare_spectra[k].unit
        flare_spectra_interp[k] = (wavelength_axis_norm, data_interp)

    # Embed 1D spectra at origin in a 3-by-3 kernel
    frame = wcs_to_celestial_frame(spec_cube.wcs)  # get this from the cube
    flare_loc = SkyCoord(0, 0, unit='arcsec', frame=frame)  # Should correspond to where flare occurs on disk
    pixel_idx = spec_cube[0].wcs.world_to_array_index(flare_loc)
    flare_spectra_interp_cubes = {}
    for k in flare_spectra_interp:
        wavelength, data = flare_spectra_interp[k]
        data_array = spec_cube.data.copy()
        data_array[:, pixel_idx[0]-1:pixel_idx[0]+2, pixel_idx[1]-1:pixel_idx[1]+2] = data[:, None, None]
        flare_spectra_interp_cubes[k] = ndcube.NDCube(data_array, wcs=spec_cube.wcs, meta=spec_cube.meta)

    # Write out spectra to files
    results_root = pathlib.Path(OUTPUT_DIR)
    for k in flare_spectra_interp_cubes:
        results_dir = results_root / 'cdr' / f'flare_{k}_embedded'
        results_dir.mkdir(exist_ok=True, parents=True)
        sunpy_fits.write(results_dir / 'spectral_cube.fits',
                         flare_spectra_interp_cubes[k].data,
                         flare_spectra_interp_cubes[k].meta,
                         overwrite=True)
