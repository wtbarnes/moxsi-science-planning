import numpy as np
import astropy.units as u
from scipy.io import readsav
import sys
import matplotlib.pyplot as plt
from astropy.visualization import ImageNormalize, LogStretch
import sunpy.map  # this loads the colormaps
from sunpy.coordinates import get_earth

from util import make_moxsi_ndcube, construct_overlappogram


if __name__ == '__main__':
    observer = get_earth()
    norm=ImageNormalize(stretch=LogStretch())
    cmap='hinodexrt'
    plot_props = {'cmap':cmap, 'norm':norm}

    savdata = readsav("../data/forDan_MOXSI_DATA_09112020_0440_feldman.sav")

    moxsi_cube = make_moxsi_ndcube(
        savdata["moxsi1_img"], savdata["cubixss_wave"] * u.angstrom
    )

    moxsi_overlap = construct_overlappogram(moxsi_cube, angle=10 * u.deg, order=0, observer=observer)

    moxsi_overlap_cutout = moxsi_overlap[400:600,400:600, :]

    ani = moxsi_overlap_cutout.plot(**plot_props)
    plt.show()
