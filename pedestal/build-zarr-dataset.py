"""
Convert many FITS files to Zarr dataset
"""
import pathlib

import astropy.io.fits
import dask.array
import numpy as np
import pandas
import tqdm
import xarray


def steinharthart(coeff, scale=False):
    # scale=True is for "det0", which is the detector temperature
    def func(dn):
        r = dn / (4096 - dn)
        if scale:
            r *= 10000
        result = 1 / np.poly1d(coeff[::-1])(np.log(r)) - 273.15
        return result
    return func


# These are the Steinhart-Hart functions, which we will use
det0temp_sh = steinharthart([1.1292E-03, 2.3411E-04, 0.0000E+00, 8.7755E-08], scale=True)
det1temp_sh = steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
fpgatemp_sh = steinharthart([3.3540E-03, 2.5708E-04, 1.8939E-06, 1.8973E-07])
thermadc_sh = steinharthart([3.3540E-03, 3.0013E-04, 5.0852E-06, 2.1877E-07])


if __name__ == '__main__':
    # Get all filenames
    data_dir = pathlib.Path('../data/moxsi_gsfc_calibration_images/')
    all_fits_files = sorted(data_dir.glob('csie_image_*.bin.fits'))
    # Skip the first file due to uncertainty in the true exposure time
    all_fits_files = all_fits_files[1:]
    # Build dataset
    data = np.empty((len(all_fits_files),2000,1504))
    temp_det_1 = []
    temp_det_0 = []
    temp_adc = []
    exposure_time = []
    time = []
    meta_keys = [
        'DET0TEMP', 'DET1TEMP', 'THERMADC', 'FPGATEMP', 'EXPTIME', 'DATE', 'FRAME_ID'
    ]
    meta_arrays = {k: [] for k in meta_keys}
    for i,filename in enumerate(tqdm.tqdm(all_fits_files)):
        with astropy.io.fits.open(filename, memmap=False) as hdul:
            _header = hdul[0].header
            data[i,...] = hdul[0].data[:2000,:]  #disregard extra row
        for k in meta_arrays.keys():
            meta_arrays[k].append(_header[k])
    time_coord = pandas.to_datetime(meta_arrays['DATE'])
    data = dask.array.from_array(
        data,
        chunks=(None, data.shape[1]//10, data.shape[2]//10)
    )
    ds = xarray.Dataset(
        {
            'data': (["sample", "row", "column"], data),

        },
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
    # Correct exposure time
    exptime_fixed = np.empty_like(ds['exposure_time'].data)
    exptime_fixed[:414] = np.where(ds['exposure_time'].data[:414] == 1250, 5000, 1250)
    exptime_fixed[414:] = np.where(
        ds['exposure_time'].data[414:] == 2500,
        1250,
        np.where(ds['exposure_time'].data[414:] == 5000, 2500, 5000)
    )
    exptime_fixed[614] = 5000  # this single image needs to be separately fixed
    ds['exposure_time'].data = exptime_fixed
    # Add unit information
    ds['data'].attrs['unit'] = 'DN'
    for k in ['detector_0', 'detector_1', 'adc', 'fpga']:
        ds[f'temperature_{k}'].attrs['unit'] = 'deg C'
    ds['exposure_time'].attrs['unit'] = 'ms'
    # Write out the Zarr dataset
    ds.to_zarr(data_dir / 'images.zarr')
