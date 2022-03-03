"""
Utility functions for constucting overlappograms
"""
import numpy as np
import astropy.constants
import astropy.units as u
import astropy.wcs
import ndcube
from sunpy.coordinates import get_earth

from sunpy.image.transform import affine_transform

# The following numbers are from Jake and Albert:
CDELT_SPACE = 5.66 * u.arcsec / u.pix
CDELT_WAVE = 55 * u.milliangstrom / u.pix
# The units are from Athiray (and the proposal figure)
BUNIT = 'ph / (pix s)'


def hgs_observer_to_keys(observer):
    return {
        'DATE-OBS': observer.obstime.isot,
        'HGLN_OBS': observer.lon.to('deg').value,
        'HGLT_OBS': observer.lat.to('deg').value,
        'DSUN_OBS': observer.radius.to('m').value,
        'RSUN_REF': astropy.constants.R_sun.to('m').value,
    }


@u.quantity_input
def construct_rot_matrix(angle:u.deg, order=1):
    """
    Parameters
    ----------
    angle: 
        Angle between the second pixel axis and the y-like
        world axis.
    order:
        Order of the dispersion. Default is 1.
    """
    return np.array([[np.cos(angle), -np.sin(angle), order*np.sin(angle)],
                     [np.sin(angle), np.cos(angle), -order*np.cos(angle)],
                     [0, 0, 1]])


def rmatrix_to_pcij(rmatrix):
    pcij_keys = {}
    for i in range(rmatrix.shape[0]):
        for j in range(rmatrix.shape[1]):
            pcij_keys[f'PC{i+1}_{j+1}'] = rmatrix[i,j]
    return pcij_keys


@u.quantity_input
def rotate_image(data, rmatrix, order=4, missing=0.0,):
    """
    This is a stripped-down version of `~sunpy.map.GenericMap.rotate`
    This should not be used for anything and should be replaced at
    some point.
    """
    if order not in range(6):
        raise ValueError("Order must be between 0 and 5.")

    # Calculate the shape in pixels to contain all of the image data
    extent = np.max(np.abs(np.vstack((data.shape @ rmatrix,
                                      data.shape @ rmatrix.T))), axis=0)

    # Calculate the needed padding or unpadding
    diff = np.asarray(np.ceil((extent - data.shape) / 2), dtype=int).ravel()
    # Pad the image array
    pad_x = int(np.max((diff[1], 0)))
    pad_y = int(np.max((diff[0], 0)))

    new_data = np.pad(data,
                      ((pad_y, pad_y), (pad_x, pad_x)),
                      mode='constant',
                      constant_values=(missing, missing))

    # All of the following pixel calculations use a pixel origin of 0

    pixel_array_center = (np.flipud(new_data.shape) - 1) / 2.0

    # Apply the rotation to the image data
    new_data = affine_transform(new_data.T,
                                np.asarray(rmatrix),
                                order=order,
                                scale=1.0,
                                image_center=np.flipud(pixel_array_center),
                                recenter=False,
                                missing=missing,
                                use_scipy=True).T

    # Unpad the array if necessary
    unpad_x = -np.min((diff[1], 0))
    if unpad_x > 0:
        new_data = new_data[:, unpad_x:-unpad_x]
    unpad_y = -np.min((diff[0], 0))
    if unpad_y > 0:
        new_data = new_data[unpad_y:-unpad_y, :]

    return new_data


def overlap_arrays(cube, dispersion_angle=0*u.deg, clip=True, order=0):
    """
    Flatten intensity cube into an overlappogram such that the first array index direction
    and wavelength directions overlap. 
    
    The array is clipped such that it starts halfway
    (in first array index direction) through the first wavelength slice and ends halfway
    through the last wavelength slice. This ensures the dispersion direction has the same
    dimension as wavelength.

    TODO: generalize such that the dispersion direction can be along any axis or even any
    non-orthogonal path through lon,lat space

    Parameters
    ----------
    dispersion_angle : `float`, optional
        Angle between the dispersion axis and the latitude world coordinate. The dispersion axis
        is always aligned with the y-like (row-indexing) pixel axis. A dispersion angle of 0
        means that the dispersion is completely latitudinal.
    clip : `bool`, optional
        If true, ensure that the dispersion direction has the same shape as the wavelength
        dimension.
    """
    if dispersion_angle % (360*u.deg) == 0:
        rot_data = cube.data
    else:
        rmatrix = construct_rot_matrix(dispersion_angle)
        # apply the necessary rotation to every slice in the cube.
        # this is an overly complicated way to do the rotation, but
        # will suffice for now
        layers = []
        for d in cube.data:
            layers.append(rotate_image(d, rmatrix[:2,:2], order=order))
        rot_data = np.array(layers)
    shape = rot_data.shape
    n_y = int(shape[0] + shape[1])
    n_x = int(shape[2])
    overlappogram = np.zeros((n_y, n_x))
    for i in range(shape[0]):
        overlappogram[i:(i+shape[1]), :] += rot_data[i, :, :]
    if clip:
        # Clip to desired range
        # When shape[1] is even, clip_1 == clip_2
        # When shape[1] is odd, we arbitrarily shift down by a half index at the short wavelength end
        # and up a half at the long wavelength end such that the latitude/wavelength dimension is 
        # cropped appropriately.
        clip_1 = int(np.floor(shape[1]/2))
        clip_2 = int(np.ceil(shape[1]/2))
        overlappogram = overlappogram[clip_1:(overlappogram.shape[0]-clip_2),:]

    return overlappogram  * cube.unit


def strided_overlappogram(overlappogram):
    """
    Return a "strided" version of the overlappogram array.
    
    For an overlappogram image of shape (N_lam, N_pix), this
    function creates an array of dimension (N_lam, N_lam, N_pix)
    where each layer is a view of the original array. In other
    words, the values at (k,i,j) and (k+1,i,j) point to the
    same place in memory.
    """
    return np.lib.stride_tricks.as_strided(
        overlappogram,
        shape=overlappogram.shape[:1]+overlappogram.shape,
        strides=(0,)+overlappogram.strides,
        writeable=False
    )


def make_moxsi_ndcube(data, wavelength, cdelt1=CDELT_SPACE, cdelt2=CDELT_SPACE):
    cdelt3 = (wavelength[1] - wavelength[0]) / u.pix
    # Here, assume that the data cube has three dimensions and that
    # the first dimension corresponds to wavelength, then latitude,
    # then longitude.
    moxsi_wcs = {
        'CRVAL1': 0, # Assume for now that the sun is at the center of the image.
        'CRVAL2': 0, # Assume for now that the sun is at the center of the image.
        'CRVAL3': wavelength[0].to('Angstrom').value,
        'CRPIX1': (data.shape[2] + 1) / 2,
        'CRPIX2': (data.shape[1] + 1) / 2,
        'CRPIX3': 1,
        'CDELT1': cdelt1.to('arcsec / pix').value,
        'CDELT2': cdelt2.to('arcsec / pix').value,
        'CDELT3': cdelt3.to('angstrom / pix').value,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CUNIT3': 'Angstrom',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CTYPE3': 'WAVE',
    }
    wcs = astropy.wcs.WCS(moxsi_wcs,)
    return ndcube.NDCube(data, wcs=wcs, unit=BUNIT)


@u.quantity_input
def construct_overlappogram(cube, angle=0*u.deg, order=1, observer=None):
    """
    TODO: generalize to any direction for the dispersion axis. This still
          assumes that the dispersion axis is exactly aligned with one
          of the pixel axes (in particular, p2).
    """
    rmatrix = construct_rot_matrix(angle, order=order)
    # Flatten to overlappogram
    moxsi_overlap = overlap_arrays(cube, dispersion_angle=angle)
    # Make strided 3D array
    wave = cube.axis_world_coords(0)[0].to('angstrom')
    moxsi_strided_overlap = strided_overlappogram(moxsi_overlap)
    wcs_keys = {
        'WCSAXES': 3,
        'NAXIS1': moxsi_strided_overlap.shape[2],
        'NAXIS2': moxsi_strided_overlap.shape[1],
        'NAXIS3': moxsi_strided_overlap.shape[0],
        'CDELT1': u.Quantity(cube.wcs.wcs.cdelt[0], f'{cube.wcs.wcs.cunit[0]} / pix').to('arcsec / pix').value,
        'CDELT2': u.Quantity(cube.wcs.wcs.cdelt[1], f'{cube.wcs.wcs.cunit[1]} / pix').to('arcsec / pix').value,
        'CDELT3': u.Quantity(cube.wcs.wcs.cdelt[2], f'{cube.wcs.wcs.cunit[2]} / pix').to('Angstrom / pix').value,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CUNIT3': 'Angstrom',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CTYPE3': 'WAVE',
        'CRPIX1': (moxsi_strided_overlap.shape[2] + 1)/2,
        'CRPIX2': (moxsi_strided_overlap.shape[1] + 1)/2,
        'CRPIX3': (moxsi_strided_overlap.shape[0] + 1)/2,
        'CRVAL1': 0,
        'CRVAL2': 0,
        'CRVAL3': ((wave[0] + wave[-1])/2).to('angstrom').value,
    }
    wcs_keys = {**wcs_keys, **rmatrix_to_pcij(rmatrix)}
    if observer is not None:
        wcs_keys = {**wcs_keys, **hgs_observer_to_keys(observer)}
    wcs = astropy.wcs.WCS(wcs_keys)
    overlap_cube = ndcube.NDCube(moxsi_strided_overlap, wcs=wcs)
    return overlap_cube
