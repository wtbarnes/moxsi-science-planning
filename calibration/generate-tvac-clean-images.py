"""
Generate a list of images appropriate for fitting a pedestal model.
"""
import glob
import re

import numpy as np

from astropy.io import fits
from astropy.time import Time


if __name__ == '__main__':
    folders = ["csie_2026082*", "csie_2026083*", "csie_20260901*", "csie_20260902*", "csie_20260903*"]
    paths = [f"/data1/moxsi/moxsi_images_from_tvac/{folder}/*.fits" for folder in folders]
    files = glob.glob(paths[0])
    for path in paths[1:]:
        files += glob.glob(path)
    files.sort(key=lambda file: int(re.search(r"frame(\d+).bin.fits", file).group(1)))
    nfiles = len(files)
    hdu_time = np.empty(nfiles)
    csie_time = np.empty(nfiles)
    csie_frame_id = np.empty(nfiles)
    exptime = np.empty(nfiles)
    median = np.empty(nfiles)
    for i, file in enumerate(files):
        with fits.open(file) as f:
            hdu_time[i] = (Time(f[0].header["date"]) - Time('2026-08-20 01:59')).to('s').value
            csie_time[i] = f[0].header["SHCOARSE"] + f[0].header["SHFINE"] / 1000
            csie_frame_id[i] = f[0].header["frame_id"]
            exptime[i] = f[0].header["exptime"]
            median[i] = np.median(f[0].data[:, -376:])  # fourth stripe
    # Frames that are not overexposed and do not obviously have the LED on
    good = ((exptime <= 10000) & (median < 6200)) | ((exptime > 10000) & (median < 12000))
    # Good frames that are no more than 5-s exposures and are preceded by a good frame
    clean = (exptime <= 5000) & (csie_frame_id >= 86392) & (csie_time > np.roll(csie_time, 1)) & good & np.roll(good, 1)
    with open('/data1/moxsi/notebooks/clean_TVAC_images.txt', 'w') as f:
        for item in np.array(files)[clean]:
            f.write(f'{item}\n')
