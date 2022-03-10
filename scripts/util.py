"""
Utility functions for constucting overlappograms
"""
import numpy as np
import scipy.ndimage
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


def color_lat_lon_axes(ax,
                       lon_color='C0',
                       lat_color='C3',
                       wvl_color='C4',
                       lat_tick_ops=None,
                       lon_tick_ops=None,
                       wvl_tick_ops=None):
    lat_tick_ops = {} if lat_tick_ops is None else lat_tick_ops
    lon_tick_ops = {} if lon_tick_ops is None else lon_tick_ops
    lat_tick_ops['color'] = lat_color
    lon_tick_ops['color'] = lon_color
    lon = ax.coords[0]
    lat = ax.coords[1]
    # Ticks-lon
    lon.set_ticklabel_position('lb')
    lon.set_axislabel(ax.get_xlabel(),color=lon_color)
    lon.set_ticklabel(color=lon_color)
    lon.set_ticks(**lon_tick_ops)
    # Ticks-lat
    lat.set_ticklabel_position('lb')
    lat.set_axislabel(ax.get_ylabel(),color=lat_color)
    lat.set_ticklabel(color=lat_color)
    lat.set_ticks(**lat_tick_ops)
    # Grid
    lon.grid(color=lon_color,grid_type='contours')
    lat.grid(color=lat_color,grid_type='contours')
    # If wavelength axis is set, do some styling there too
    if len(ax.coords.get_coord_range()) > 2:
        wvl = ax.coords[2]
        wvl_tick_ops = {} if wvl_tick_ops is None else wvl_tick_ops
        wvl_tick_ops['color'] = wvl_color
        wvl.set_ticklabel_position('rt')
        wvl.set_format_unit(u.angstrom)
        wvl.set_major_formatter('x.x')
        wvl.set_ticklabel(color=wvl_color)
        wvl.set_ticks(color=wvl_color)
        wvl.set_axislabel('Wavelength [Angstrom]', color=wvl_color)
        wvl.grid(color=wvl_color, grid_type='contours')
        return lon, lat, wvl
    return lon,lat


def hgs_observer_to_keys(observer):
    return {
        'DATE-OBS': observer.obstime.isot,
        'HGLN_OBS': observer.lon.to('deg').value,
        'HGLT_OBS': observer.lat.to('deg').value,
        'DSUN_OBS': observer.radius.to('m').value,
        'RSUN_REF': astropy.constants.R_sun.to('m').value,
    }


@u.quantity_input
def rotation_matrix(angle:u.deg):
    return np.array([
        [np.cos(angle), np.sin(angle), 0],
        [-np.sin(angle), np.cos(angle), 0],
        [0, 0, 1]
    ])


@u.quantity_input
def dispersion_matrix(order):
    return np.array([
        [1, 0, 0],
        [0, 1, -order],
        [0, 0, 1],
    ])


@u.quantity_input
def construct_pcij(roll_angle:u.deg, dispersion_angle:u.deg, order=1,
                   correlate_p12_with_wave=False):
    """
    Parameters
    ----------
    roll_angle: 
        Angle between the second pixel axis and the y-like
        world axis.
    dispersion_angle:
        Angle between the wavelength (dispersion) axis and the second pixel axis.
    order:
        Order of the dispersion. Default is 1.
    """
    R_2 = rotation_matrix(roll_angle - dispersion_angle)
    D = dispersion_matrix(order)
    if correlate_p12_with_wave:
        # This aligns the dispersion axis with the wavelength axis
        # and decorrelates wavelength with the the third "fake"
        # pixel axis. This means that wavelength *does not* vary
        # as you increment that third axis.
        # This is not strictly correct as the world grid at each p3
        # should only correspond to a particular wavelength, not
        # all of them.
        D[2,:] = [0, 1, 0]
    R_1 = rotation_matrix(dispersion_angle)
    return R_2 @ D @ R_1


def get_pcij_keys(pcij_matrix):
    pcij_keys = {}
    for i in range(pcij_matrix.shape[0]):
        for j in range(pcij_matrix.shape[1]):
            pcij_keys[f'PC{i+1}_{j+1}'] = pcij_matrix[i,j]
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
    # escape hatch for skipping unneeded rotation
    if not np.all(new_data == 0.0):
        new_data = affine_transform(
            new_data.T,
            np.asarray(rmatrix),
            order=order,
            scale=1.0,
            image_center=np.flipud(pixel_array_center),
            recenter=False,
            missing=missing,
            use_scipy=True
        ).T

    # Unpad the array if necessary
    unpad_x = -np.min((diff[1], 0))
    if unpad_x > 0:
        new_data = new_data[:, unpad_x:-unpad_x]
    unpad_y = -np.min((diff[0], 0))
    if unpad_y > 0:
        new_data = new_data[unpad_y:-unpad_y, :]

    return new_data


def overlap_arrays(cube, roll_angle=0*u.deg, dispersion_angle=0*u.deg, clip=True):
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
    roll_angle : `float`, optional
        Angle between the y-like pixel axis and the latitude world coordinate. This is often
        referred to as the "roll angle" of the satellite.
    dispersion_angle : `float`, optional
        Angle between the y-like pixel axis and the dispersion or wavelength. Ideally this should
        be small as non-zero values mean that information is lost off the edges of the detector.
        A dispersion angle of 0 means that the y-like pixel axis and wavelength are aligned such
        that all dispersion occurs along that pixel axis.
    clip : `bool`, optional
        If true, ensure that the dispersion direction has the same shape as the wavelength
        dimension.
    """
    if roll_angle % (360*u.deg) == 0:
        rot_data = cube.data
    else:
        # The order here is not important
        rmatrix = rotation_matrix(roll_angle)[:2,:2]
        # apply the necessary rotation to every slice in the cube.
        # this is an overly complicated way to do the rotation, but
        # will suffice for now
        layers = []
        for d in cube.data:
            layers.append(rotate_image(d, rmatrix.T, order=0, missing=0.0))
        rot_data = np.array(layers)
        # FIXME: do we need to cut this array down to the detector size?
    shape = rot_data.shape
    n_y = int(shape[0] + shape[1])
    n_x = int(shape[2])
    overlappogram = np.zeros((n_y, n_x))
    for i in range(shape[0]):
        layer = apply_dispersion_shift(
            dispersion_angle,
            rot_data[i, :, :],
            overlappogram.shape,
            i,
            i+shape[1],
        )
        overlappogram += layer
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


def apply_dispersion_shift(gamma, array, overlap_shape, row_start, row_end):
    # calculate center (in pix coordinates, relative to center)
    center = np.array([(row_start + row_end)/2 - overlap_shape[0]/2, 0])
    # calculate new center
    new_center = rotation_matrix(gamma)[:2,:2] @ center
    # calculate shift
    shift = new_center - center
    # Create dummy
    dummy = np.zeros(overlap_shape)
    dummy[row_start:row_end, :] = array
    # If aligned with the pixel axis, don't waste time applying 0 shift
    if gamma % (360*u.deg) == 0:
        return dummy
    # If the array is all zeros anyway, don't waste time shifting nothing
    if np.all(dummy == 0.0):
        return dummy
    return scipy.ndimage.shift(dummy, shift)


def strided_overlappogram(overlappogram, wave):
    """
    Return a "strided" version of the overlappogram array.
    
    For an overlappogram image of shape (N_pix2, N_pix1), this
    function creates an array of dimension (N_lam, N_pix2, N_pix1)
    where each layer is a view of the original array. In other
    words, the values at (k,i,j) and (k+1,i,j) point to the
    same place in memory.
    """
    return np.lib.stride_tricks.as_strided(
        overlappogram,
        shape=wave.shape+overlappogram.shape,
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


def overlappogram_wcs(shape, wave, cdelt, cunit, pc_matrix, observer):
    wcs_keys = {
        'WCSAXES': 3,
        'NAXIS1': shape[2],
        'NAXIS2': shape[1],
        'NAXIS3': shape[0],
        'CDELT1': u.Quantity(cdelt[0], f'{cunit[0]} / pix').to('arcsec / pix').value,
        'CDELT2': u.Quantity(cdelt[1], f'{cunit[1]} / pix').to('arcsec / pix').value,
        'CDELT3': u.Quantity(cdelt[2], f'{cunit[2]} / pix').to('Angstrom / pix').value,
        'CUNIT1': 'arcsec',
        'CUNIT2': 'arcsec',
        'CUNIT3': 'Angstrom',
        'CTYPE1': 'HPLN-TAN',
        'CTYPE2': 'HPLT-TAN',
        'CTYPE3': 'WAVE',
        'CRPIX1': (shape[2] + 1)/2,
        'CRPIX2': (shape[1] + 1)/2,
        'CRPIX3': (shape[0] + 1)/2,
        'CRVAL1': 0,
        'CRVAL2': 0,
        'CRVAL3': ((wave[0] + wave[-1])/2).to('angstrom').value,
    }
    wcs_keys = {**wcs_keys, **get_pcij_keys(pc_matrix)}
    if observer is not None:
        wcs_keys = {**wcs_keys, **hgs_observer_to_keys(observer)}
    wcs = astropy.wcs.WCS(wcs_keys)
    return wcs


@u.quantity_input
def construct_overlappogram(cube,
                            roll_angle=0*u.deg,
                            dispersion_angle=0*u.deg,
                            order=1,
                            observer=None,
                            correlate_p12_with_wave=False):
    """
    Given a spectral cube, build an overlappogram "cube"

    Parameters
    ----------
    cube
    roll_angle
    dispersion_angle
    observer
    correlate_p12_with_wave
    """
    pc_matrix = construct_pcij(roll_angle, dispersion_angle, order=order,
                               correlate_p12_with_wave=correlate_p12_with_wave)
    # Flatten to overlappogram
    moxsi_array = overlap_arrays(
        cube,
        roll_angle=roll_angle,
        dispersion_angle=dispersion_angle
    )
    # Make strided 3D array
    wave = cube.axis_world_coords(0)[0].to('angstrom')
    moxsi_strided_array = strided_overlappogram(moxsi_array, wave)
    wcs = overlappogram_wcs(
        moxsi_strided_array.shape,
        wave,
        cube.wcs.wcs.cdelt,
        cube.wcs.wcs.cunit,
        pc_matrix,
        observer,
    )
    overlap_cube = ndcube.NDCube(moxsi_strided_array, wcs=wcs)
    return overlap_cube
