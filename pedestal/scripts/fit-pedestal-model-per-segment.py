"""
Fit pedestal model to every pixel on the detector
"""
import pathlib

import distributed
import numpy as np
import xarray

CLIENT_ADDRESS = 'tcp://127.0.0.1:40259'

OUTPUT_FILENAME = 'fit_coefficients_full_abovezero.nc'

# This is the full fitting function. Adapt it below as needed.
#def fitting_function(X, a, b, c, d, e, f):
#    t_det0, t_adc, exposure = X
#    return a + b * exposure * np.exp(t_det0 / c) + d * t_det0 + e * t_det0**2 + f * t_adc


if __name__ == '__main__':
    # Connect to Dask cluster
    client = distributed.Client(address=CLIENT_ADDRESS)
    # Load data
    data_dir = pathlib.Path('../data/moxsi_gsfc_calibration_images/')
    ds = xarray.open_zarr(data_dir / 'images.zarr')
    # Rechunk?
    # Select out only the frames we consider to be "dark"
    # I did this by taking a cutout around one of the alignment pinholes and then choosing a threshold
    # that seemed to exclude most of the illuminated frames.
    # pinhole_1_slice = np.s_[:,82:158,900:975]
    # ts_alignment_ph1 = ds['data'][pinhole_1_slice].sum(dim=['row','column']).compute()
    # dark_thresh = 3.17e7
    # time_lims = np.array(['2025-09-30T19:00:00','2025-09-30T22:00:00'], dtype='datetime64[ns]')
    # dark_frames = ds.isel(sample=np.where(ts_alignment_ph1 < dark_thresh)[0])
    # dark_frames = dark_frames.isel(
    #     sample=np.where(np.logical_and(dark_frames.time>time_lims[0],
    #                                    dark_frames.time<time_lims[-1]))[0]
    # )
    # Select only the frames corresponding to a detector temperature > 0
    idx = np.where(np.logical_and(ds.temperature_detector_0 > 0, ds.temperature_detector_1 > 0))
    #ds = ds.isel(sample=idx[0])
    # Fit data
    #array_slice = np.s_[:,:,:]  # Make this smaller for easy testing
    data_to_fit = ds['data'] #[array_slice]
    #data_to_fit = dark_frames['data']
    #data_to_fit = data_to_fit.chunk({
    #    'sample': data_to_fit.sample.size,
    #    'row': data_to_fit.row.size//10,
    #    'column': data_to_fit.column.size//10,
    #})
    # This loop fits the individual segments of the detector individually because the value of the 
    # f parameter is roughly constant over these segments. The locations of the segments were determined
    # manually and then gaussian distributions were fit to f in each segment to determine the approximate
    # value of f
    f_divisions = [
        [0, 375],
        [375, 750],
        [750, 1125],
        [1125, 1504],
    ]
    f_fixed_estimate = [
        1.9662495342645396,
        2.08083032721758,
        2.2667223450191125,
        0.9512635133522603,
    ]
    # I found this by fititng a Gaussian to the distribution of d values for the case in which I 
    # allowed a and d (only) to vary and fixed f across the four taps.
    # d_fixed = 2.288023013652479
    # I found these values by fitting a Gaussian to the distribution of parameter values for the
    # case in which I allowed all of the parameters to vary for the full range of temperatures.
    params_fixed_estimate = {
    #    'b': np.float64(0.0048736182432977115),
    #    'c': np.float64(7.047611800096444),
    #    'd': np.float64(2.4490954061290844),
    #    'e': np.float64(0.01927927292177515),
    }
    coeff_arrays = []
    for fd, fe in zip(f_divisions, f_fixed_estimate):
        print(fd)

        def fitting_function_fixed_f(X, a, b, c, d, e):
            t_det0, t_adc, exposure = X
            return (a + 
                    b * exposure * np.exp(t_det0 / c) +
                    d* t_det0 +
                    e * t_det0**2 +
                    fe * t_adc)
        
        da_fit = data_to_fit[:, :, fd[0]:fd[1]].curvefit(
            (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
            fitting_function_fixed_f,
            reduce_dims=['sample'],
            errors='ignore',
        )
        _coeff_array = da_fit.curvefit_coefficients.compute()
        # NOTE: This saves the constant f value per segment in the same array for convenience
        _const_f = xarray.ones_like(_coeff_array[...,:1]).assign_coords(param=['f'])*fe
        # NOTE: This saves the constant parameter value per segment in the same array for convenience
        _const_coeffs = []
        for p in params_fixed_estimate:
            _const_coeffs.append(xarray.ones_like(_coeff_array[...,:1]).assign_coords(param=[p])*params_fixed_estimate[p])
        _const_coeffs += [_const_f]
        coeff_arrays.append(xarray.concat([_coeff_array,]+_const_coeffs, dim='param'))
    # Save out fit coefficients
    coeff_arrays = xarray.concat(coeff_arrays, dim='column')
    coeff_arrays.to_netcdf(data_dir / OUTPUT_FILENAME)
