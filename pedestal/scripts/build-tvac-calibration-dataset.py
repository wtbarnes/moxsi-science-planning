"""
Build the a full dataset from the files that were taken during TVAC testing.
"""
import pathlib

from ..build_zarr_dataset import list_of_files_to_ds


if __name__ == '__main__':
    # This is the root directory for the MOXSI calibration data
    moxsi_data_dir = pathlib.Path('/data1/moxsi')
    # Read the list of images from the TVAC dataset that will be used for model fitting
    with (moxsi_data_dir / 'notebooks' / 'clean_TVAC_images.txt').open() as f:
        clean_tvac_images = f.read().splitlines()
    clean_tvac_images = [moxsi_data_dir / pathlib.Path(f).relative_to('..') for f in clean_tvac_images]
    # Build the dataset
    ds = list_of_files_to_ds(clean_tvac_images)
    # Save out the dataset
    ds.to_zarr(moxsi_data_dir / 'clean_tvac_images.zarr')
