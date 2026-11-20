"""
Fit pedestal model to every pixel on the detector
"""
import pathlib

import distributed
import numpy as np
import xarray

CLIENT_ADDRESS = 'tcp://127.0.0.1:37149'

OUTPUT_FILENAME = 'fit_coefficients_tvac.nc'

# This is the full fitting function. Adapt it below as needed.
def fitting_function(X, a, b, c, d, e, f):
    t_det0, t_adc, exposure = X
    return a + b * exposure * np.exp(t_det0 / c) + d * t_det0 + e * t_det0**2 + f * t_adc


if __name__ == '__main__':
    client = distributed.Client(address=CLIENT_ADDRESS)
    data_dir = pathlib.Path('/data1/moxsi/notebooks/')
    ds = xarray.open_zarr(data_dir / 'clean_tvac_images.zarr')
    #data_to_fit = ds['data'].chunk({
    #    'sample': ds['data'].sample.size,
    #    'row': 150,
    #    'column': 150,
    #})
    data_to_fit = ds['data']
    coeff_arrays = []
    da_fit = data_to_fit.curvefit(
        (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
        fitting_function,
        reduce_dims=['sample'],
        errors='ignore',
    )
    coeff_arrays = da_fit.curvefit_coefficients.compute()
    coeff_arrays.to_netcdf(data_dir / OUTPUT_FILENAME)
