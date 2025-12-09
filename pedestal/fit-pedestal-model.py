"""
Fit pedestal model to every pixel on the detector
"""
import pathlib

import distributed
import numpy as np
import xarray

CLIENT_ADDRESS = 'tcp://127.0.0.1:45429'


def fitting_function(X, a, b, c, d, e, f):
    t_det0, t_adc, exposure = X
    return a + b * exposure * np.exp(t_det0 / c) + d * t_det0 + e * t_det0**2 + f * t_adc


if __name__ == '__main__':
    # Connect to Dask cluster
    client = distributed.Client(address=CLIENT_ADDRESS)
    # Load data
    data_dir = pathlib.Path('../data/moxsi_gsfc_calibration_images/')
    ds = xarray.open_zarr(data_dir / 'images.zarr')
    # Rechunk?
    # Fit data
    array_slice = np.s_[:,:,:]  # Make this smaller for easy testing
    data_to_fit = ds['data'][array_slice]
    #data_to_fit = data_to_fit.chunk({
    #    'sample': data_to_fit.sample.size,
    #    'row': data_to_fit.row.size//10,
    #    'column': data_to_fit.column.size//10,
    #})
    da_fit = data_to_fit.curvefit(
        (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
        fitting_function,
        reduce_dims=['sample'],
        errors='ignore',
    )
    coeff_arrays = da_fit.curvefit_coefficients.compute()
    # Save out fit coefficients
    coeff_arrays.to_netcdf(data_dir / 'fit_coefficients.nc')
