"""
Utility functions for constucting overlappograms
"""
import astropy.units as u
import astropy.wcs
import ndcube

from overlappy.util import hgs_observer_to_keys


# The following numbers are from Jake and Albert:
CDELT_SPACE = 5.66 * u.arcsec / u.pix
CDELT_WAVE = 55 * u.milliangstrom / u.pix
# The units are from Athiray (and the proposal figure)
BUNIT = 'ph / (pix s)'
# This comes from the proposal:
DETECTOR_SHAPE_DISPERSED = (750, 2000)
DETECTOR_SHAPE_PINHOLE = (750, 475) 


def spectral_cube_wcs(shape, wavelength, cdelt, observer):
    wcs_keys = {
        'CRVAL1': 0, # Assume for now that the sun is at the center of the image.
        'CRVAL2': 0, # Assume for now that the sun is at the center of the image.
        'CRVAL3': ((wavelength[0] + wavelength[-1])/2).to('angstrom').value,
        'CRPIX1': (shape[2] + 1) / 2,
        'CRPIX2': (shape[1] + 1) / 2,
        'CRPIX3': (shape[0] + 1) / 2,
        'CDELT1': cdelt[0].to('arcsec / pix').value,
        'CDELT2': cdelt[1].to('arcsec / pix').value,
        'CDELT3': cdelt[2].to('angstrom / pix').value,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CUNIT3': 'Angstrom',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CTYPE3': 'WAVE',
    }
    if observer is not None:
        wcs_keys = {**wcs_keys, **hgs_observer_to_keys(observer)}
    return astropy.wcs.WCS(wcs_keys,)


def make_moxsi_ndcube(data,
                      wavelength,
                      cdelt1=CDELT_SPACE,
                      cdelt2=CDELT_SPACE,
                      cdelt3=CDELT_WAVE,
                      observer=None):
    cdelt3 = (wavelength[-1] - wavelength[0]) / (wavelength.shape[0] * u.pix)
    # Here, assume that the data cube has three dimensions and that
    # the first dimension corresponds to wavelength, then latitude,
    # then longitude.
    wcs = spectral_cube_wcs(data.shape, wavelength, (cdelt1, cdelt2, cdelt3), observer)
    return ndcube.NDCube(data, wcs=wcs, unit=BUNIT)
