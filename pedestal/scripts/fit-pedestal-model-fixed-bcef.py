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

OUTPUT_FILENAME = 'fit_coefficients_tvac_fixed_bcef.nc'


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
    sel_f_coeffs = da_fit.sel(param='f').data
    _, xedge = np.histogram(sel_f_coeffs, bins='scott', range=np.quantile(sel_f_coeffs, [.001, .999]));
    xmid = (xedge[1:] + xedge[:-1])/2
    for tb in tap_boundaries:
        sel_ftb_coeffs = da_fit.sel(param='f')[:, tb[0]:tb[1]].data
        y, _ = np.histogram(sel_ftb_coeffs, bins=xedge)
        fitter = astropy.modeling.fitting.TRFLSQFitter()
        q1, q3 = np.quantile(sel_ftb_coeffs, [.25, .75])
        ginit = Gaussian1D(amplitude=y.max(), mean=xmid[y.argmax()], stddev=q3-q1)
        gfit = fitter(ginit, xmid, y)
        f_mean.append(gfit.mean)
    print(f'The fixed values of f are {f_mean}')
    # Fit Gaussians to the other selected parameters. Effectively, we are pinning these fitting parameters to
    # their means when we fit over the whole detector. This is done separately from f because we do not need
    # to fit these separately on each tap
    pinned_params = ['b', 'c', 'e']
    pinned_params_mean = {}
    for param in pinned_params:
        sel_param_coeffs = da_fit.sel(param=param)
        # Make an estimate of the domain and IQR for estimating a STDDEV guess
        qa, q1, q3, qb = np.quantile(sel_param_coeffs, [.01, .25, .75, .99])
        y, xedge = np.histogram(sel_param_coeffs.data, bins='scott', range=(qa,qb))
        xmid = (xedge[1:] + xedge[:-1])/2
        fitter = astropy.modeling.fitting.TRFLSQFitter()
        ginit = Gaussian1D(amplitude=y.max(), mean=xmid[y.argmax()], stddev=q3-q1)
        gfit = fitter(ginit, xmid, y)
        pinned_params_mean[param] = gfit.mean
    print(f'The fixed parameter means are {pinned_params_mean}')
    # This loop fits the individual segments of the detector individually because the value of the 
    # f parameter is roughly constant over these segments. This reduces the number of full arrays we
    # need to store in order to determine the pedestal.
    coeff_arrays = []
    for tb, fe in zip(tap_boundaries, f_mean):
        pinned_params_mean['f'] = fe
        print(f'Tap: {tb}')
        print(f'Using fixed parameters {pinned_params_mean}')

        # Redefine fitting function for fixed value of f (per tap)
        def fitting_function_fixed_f(X, a, d):
            t_det0, t_adc, exposure = X
            return (a + 
                    pinned_params_mean['b'] * exposure * np.exp(t_det0 / pinned_params_mean['c']) + 
                    d* t_det0 +
                    pinned_params_mean['e'] * t_det0**2 +
                    pinned_params_mean['f'] * t_adc)
        
        da_fit = ds['data'][..., tb[0]:tb[1]].curvefit(
            (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
            fitting_function_fixed_f,
            reduce_dims=['sample'],
            errors='ignore',
        )
        _coeff_array = da_fit.curvefit_coefficients.compute()
        # NOTE: This saves the constant parameter value per segment in the same array for convenience
        _const_coeffs = []
        for param in pinned_params+['f',]:
            _const_coeffs.append(xarray.ones_like(_coeff_array[...,:1]).assign_coords(param=[param])*pinned_params_mean[param])
        coeff_arrays.append(xarray.concat([_coeff_array,]+_const_coeffs, dim='param'))
    # Save out fit coefficients
    coeff_arrays = xarray.concat(coeff_arrays, dim='column')
    coeff_arrays.to_netcdf(data_dir / 'pedestal_fit_parameters' / OUTPUT_FILENAME)
