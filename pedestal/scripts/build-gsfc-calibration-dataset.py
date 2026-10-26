"""
Build the dataset from the files corresponding to early calibration efforts at GSFC in October 2025.
"""
import pathlib

import numpy as np

from ..build_zarr_dataset import list_of_files_to_ds


if __name__ == '__main__':
    # Get all filenames
    data_dir = pathlib.Path('../data/moxsi_gsfc_calibration_images/')
    all_fits_files = sorted(data_dir.glob('csie_image_*.bin.fits'))
    # Skip the first file due to uncertainty in the true exposure time
    all_fits_files = all_fits_files[1:]
    ds = list_of_files_to_ds(all_fits_files)
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
    # Write out the Zarr dataset
    ds.to_zarr(data_dir / 'images.zarr', mode='w')