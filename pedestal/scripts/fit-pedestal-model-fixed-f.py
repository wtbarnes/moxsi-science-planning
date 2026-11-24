"""
Fit pedestal model to every pixel on the detector, fixing the f parameter (ADC term)
"""
import pathlib

import astropy.modeling.fitting
import distributed
import numpy as np
import xarray

from astropy.modeling.models import Gaussian1D

CLIENT_ADDRESS = 'tcp://127.0.0.1:45459'

OUTPUT_FILENAME = 'fit_coefficients_tvac_fixed_f.nc'


if __name__ == '__main__':
    # Connect to Dask cluster
    client = distributed.Client(address=CLIENT_ADDRESS)
    # Load data
    data_dir = pathlib.Path('/data1/moxsi/notebooks/')
    ds = xarray.open_zarr(data_dir / 'clean_tvac_images.zarr')
    # Load the full fit parameter dataset
    da_fit = xarray.open_dataarray(data_dir / 'pedestal_fit_parameters' / 'fit_coefficients_tvac.nc')
    # Fit a Gaussian to the f matrix for each of the four "taps"
    # These boundaries were determined manually.
    tap_boundaries = [
        [0, 375],
        [375, 750],
        [750, 1125],
        [1125, 1504],
    ]
    f_mean = []
    _, bins, _ = da_fit.sel(param='f').plot.hist(bins='scott', range=(0.5,3));
    for tb in tap_boundaries:
        y, xedge, _ = da_fit.sel(param='f')[:, tb[0]:tb[1]].plot.hist(bins=bins)
        xmid = (xedge[1:] + xedge[:-1])/2
        fitter = astropy.modeling.fitting.TRFLSQFitter()
        ginit = Gaussian1D(amplitude=y.max(), mean=xmid[y.argmax()], stddev=0.5)
        gfit = fitter(ginit, xmid, y)
        f_mean.append(gfit.mean)
    print(f'The fixed values of f are {f_mean}')
    # This loop fits the individual segments of the detector individually because the value of the 
    # f parameter is roughly constant over these segments. This reduces the number of full arrays we
    # need to store in order to determine the pedestal.
    coeff_arrays = []
    for tb, fe in zip(tap_boundaries, f_mean):
        print(f'Tap: {tb}')

        # Redefine fitting function for fixed value of f
        def fitting_function_fixed_f(X, a, b, c, d, e):
            t_det0, t_adc, exposure = X
            return a + b * exposure * np.exp(t_det0 / c) + d* t_det0 + e * t_det0**2 + fe * t_adc
        
        da_fit = ds['data'][..., tb[0]:tb[1]].curvefit(
            (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
            fitting_function_fixed_f,
            reduce_dims=['sample'],
            errors='ignore',
        )
        _coeff_array = da_fit.curvefit_coefficients.compute()
        # NOTE: This saves the constant f value per segment in the same array for convenience
        _const_f = xarray.ones_like(_coeff_array[...,:1]).assign_coords(param=['f'])*fe
        coeff_arrays.append(xarray.concat([_coeff_array,]+[_const_f], dim='param'))
    # Save out fit coefficients
    coeff_arrays = xarray.concat(coeff_arrays, dim='column')
    coeff_arrays.to_netcdf(data_dir / 'pedestal_fit_parameters' / OUTPUT_FILENAME)
