"""
Overlappogram modelling. 

NOTE: This is a very brute force approach. It is better to just do
a reprojection.
"""
import numpy as np
import astropy.units as u
import scipy.ndimage
import ndcube
from sunpy.image import affine_transform

from overlappy.wcs import rotation_matrix, pcij_matrix, overlappogram_fits_wcs
from overlappy.util import strided_array


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


def overlap_arrays(cube,
                   roll_angle=0*u.deg,
                   dispersion_angle=0*u.deg,
                   detector_shape=None):
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
    detector_shape : `tuple`, optional
        Shape of the detector specified in number of pixels per row and
        per column. Defaults to 
    """
    if roll_angle % (360*u.deg) == 0:
        rot_data = cube.data
    else:
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
    pc_matrix = pcij_matrix(roll_angle,
                            dispersion_angle,
                            order=order,
                            align_p2_wave=correlate_p12_with_wave)
    # Flatten to overlappogram
    moxsi_array = overlap_arrays(
        cube,
        roll_angle=roll_angle,
        dispersion_angle=dispersion_angle
    )
    # Make strided 3D array
    wave = cube.axis_world_coords(0)[0].to('angstrom')
    moxsi_strided_array = strided_array(moxsi_array, wave.shape[0])
    scale = [u.Quantity(cd, f'{cu} / pix') for cd,cu in
                 zip(cube.wcs.wcs.cdelt, cube.wcs.wcs.cunit)]
    wcs = overlappogram_fits_wcs(
        moxsi_strided_array.shape[1:],
        wave,
        scale,
        pc_matrix=pc_matrix,
        observer=observer,
    )
    overlap_cube = ndcube.NDCube(moxsi_strided_array, wcs=wcs)
    return overlap_cube