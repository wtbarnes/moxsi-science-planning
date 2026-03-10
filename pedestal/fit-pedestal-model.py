"""
Fit pedestal model to every pixel on the detector
"""
import pathlib

import distributed
import numpy as np
import xarray

CLIENT_ADDRESS = 'tcp://127.0.0.1:33767'

OUTPUT_FILENAME = 'fit_coefficients_ade_fixed_f_subzero.nc'

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
    # Select only the frames corresponding to a detector temperature <0
    idx = np.where(np.logical_and(ds.temperature_detector_0 < 0, ds.temperature_detector_1 < 0))
    ds = ds.isel(sample=idx[0])
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
    coeff_arrays = []
    for fd, fe in zip(f_divisions, f_fixed_estimate):
        print(fd)

        def fitting_function_fixed_f(X, a, d, e):
            t_det0, t_adc, exposure = X
            return a + d * t_det0 + e * t_det0**2 + fe * t_adc
        
        da_fit = data_to_fit[:, :, fd[0]:fd[1]].curvefit(
            (ds.temperature_detector_0, ds.temperature_adc, ds.exposure_time),
            fitting_function_fixed_f,
            reduce_dims=['sample'],
            errors='ignore',
        )
        _coeff_array = da_fit.curvefit_coefficients.compute()
        # NOTE: This saves the constant f value per segment in the same array for convenience
        _const_f = xarray.ones_like(_coeff_array[...,:1]).assign_coords(param=['f'])*fe
        coeff_arrays.append(xarray.concat([_coeff_array, _const_f], dim='param'))
    # Save out fit coefficients
    coeff_arrays = xarray.concat(coeff_arrays, dim='column')
    coeff_arrays.to_netcdf(data_dir / OUTPUT_FILENAME)
