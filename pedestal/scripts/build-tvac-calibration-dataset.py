"""
Build the a full dataset from the files that were taken during TVAC testing.
"""
import sys
import pathlib

import astropy.io.fits
import dask
import distributed
import dask.array
import numpy as np
import pandas
import tqdm
import xarray

#sys.path.append(pathlib.Path(__file__).parent.parent.as_posix())
#from build_zarr_dataset import list_of_files_to_ds

CLIENT_ADDRESS = 'tcp://127.0.0.1:38557'
MOXSI_DETECTOR_SHAPE = (2000, 1504)


def _steinharthart(coeff, scale=False):
    # scale=True is for "det0", which is the detector temperature
    def func(dn):
        r = dn / (4096 - dn)
        if scale:
            r *= 10000
        result = 1 / np.poly1d(coeff[::-1])(np.log(r)) - 273.15
        return result
    return func


# These are the Steinhart-Hart functions, which we will use
det0temp_sh = _steinharthart([1.1292E-03, 2.3411E-04, 0.0000E+00, 8.7755E-08], scale=True)
det1temp_sh = _steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
fpgatemp_sh = _steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
thermadc_sh = _steinharthart([3.3540E-03, 3.0013E-04, 5.0852E-06, 2.1877E-07])


def _read_fits_image(filename):
    with astropy.io.fits.open(filename, memmap=False) as hdul:
        image = hdul[0].data[:MOXSI_DETECTOR_SHAPE[0], :MOXSI_DETECTOR_SHAPE[1]]
        # Copy so the returned array remains valid after the FITS file closes.
        return np.array(image, copy=True)


def list_of_files_to_ds(filenames):
    """
    Transform a list of MOXSI FITS files to a single xarray DataSet

    Parameters
    ----------
    filenames
    """
    # Build dataset
    meta_keys = [
        'DET0TEMP', 'DET1TEMP', 'THERMADC', 'FPGATEMP', 'EXPTIME', 'DATE', 'FRAME_ID'
    ]
    meta_arrays = {k: [] for k in meta_keys}
    data = np.empty((len(filenames),)+MOXSI_DETECTOR_SHAPE)
    #frames = []
    for i, filename in enumerate(tqdm.tqdm(filenames)):
        with astropy.io.fits.open(filename, memmap=False) as hdul:
            _header = hdul[0].header
            #image_dtype = hdul[0].data.dtype
            data[i,...] = hdul[0].data[:MOXSI_DETECTOR_SHAPE[0], :MOXSI_DETECTOR_SHAPE[1]]
        # NOTE: This avoids having to read in the whole array into memory
        # frames.append(dask.array.from_delayed(
        #     dask.delayed(_read_fits_image)(filename),
        #     shape=MOXSI_DETECTOR_SHAPE,
        #     dtype=image_dtype,
        # ))
        for k in meta_arrays.keys():
            meta_arrays[k].append(_header[k])
    #data = dask.array.stack(frames, axis=0)
    data = dask.array.from_array(
        data,
        chunks=(None, 100, 100),
    )
    time_coord = pandas.to_datetime(meta_arrays['DATE'])
    ds = xarray.Dataset(
        {'data': (["sample", "row", "column"], data),},
        coords={
            "time": (["sample"], time_coord),
            "frame_id": (["sample"], np.array(meta_arrays['FRAME_ID'])),
            'temperature_detector_0': (["sample"], np.array(meta_arrays['DET0TEMP'])),
            'temperature_detector_1': (["sample"], np.array(meta_arrays['DET1TEMP'])),
            'temperature_adc': (["sample"], np.array(meta_arrays['THERMADC'])),
            'temperature_fpga': (["sample"], np.array(meta_arrays['FPGATEMP'])),
            'exposure_time': (["sample"], np.array(meta_arrays['EXPTIME'])),
        }
    )
    # Apply conversion functions for temperature
    ds['temperature_detector_0'].data = det0temp_sh(ds['temperature_detector_0'])
    ds['temperature_detector_1'].data = det1temp_sh(ds['temperature_detector_1'])
    ds['temperature_adc'].data = thermadc_sh(ds['temperature_adc'])
    ds['temperature_fpga'].data = fpgatemp_sh(ds['temperature_fpga'])
    # Add unit information
    ds['data'].attrs['unit'] = 'DN'
    for k in ['detector_0', 'detector_1', 'adc', 'fpga']:
        ds[f'temperature_{k}'].attrs['unit'] = 'deg C'
    ds['exposure_time'].attrs['unit'] = 'ms'
    return ds


if __name__ == '__main__':
    # client = distributed.Client(address=CLIENT_ADDRESS)
    # This is the root directory for the MOXSI calibration data
    moxsi_data_dir = pathlib.Path('/data1/moxsi')
    # Read the list of images from the TVAC dataset that will be used for model fitting
    with (moxsi_data_dir / 'notebooks' / 'clean_TVAC_images.txt').open() as f:
        clean_tvac_images = f.read().splitlines()
    clean_tvac_images = [pathlib.Path(f) for f in clean_tvac_images]
    # NOTE: The slice of images here is specifically selected to to include a full temperature cycle
    # from the TVAC data. Ideally, we'd select the whole dataset, but that seems to cause problems...
    clean_tvac_images = clean_tvac_images[13000:22500]
    # Build the dataset
    ds = list_of_files_to_ds(clean_tvac_images)
    # Build ideal chunk shape
    # NOTE: These chunk sizes are somewhat arbitrary, but seem to work fairly well for this detector shape
    # With the current size of this dataset, this works out to about 2 GB per chunk
    #chunk_shape = ds['data'].shape[:1] + (150, 150)
    # Save out the dataset
    ds.to_zarr(moxsi_data_dir / 'notebooks' / 'clean_tvac_images.zarr', mode='w',)
               #encoding={'data': {'chunks': chunk_shape}},
               #align_chunks=True)
