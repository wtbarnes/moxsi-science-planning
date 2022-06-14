"""
Classes for computing wavelength response functions for MOXSI
"""
import numpy as np
import astropy.units as u
import astropy.constants as const

from sunpy.util import MetaDict
from sunpy.io.special import read_genx


class Channel:
    
    def __init__(self, name, instrument_file):
        # Switch this to accept a filter type or an order and then construct name
        # based on that.
        self._name = name
        self._instrument_data = self._get_instrument_data(instrument_file)
        
    def _get_instrument_data(self, instrument_file):
        return read_genx(instrument_file)
        
    @property
    def _data(self):
        index_mapping = {}
        for i,c in enumerate(self._instrument_data['SAVEGEN0']):
            index_mapping[c['CHANNEL']] = i
        return MetaDict(self._instrument_data['SAVEGEN0'][index_mapping[self._name]])
        
    @property
    def name(self):
        return self._name
        
    @property
    @u.quantity_input
    def wavelength(self) -> u.angstrom:
        return u.Quantity(self._data['wave'], 'angstrom')
        
    @property
    @u.quantity_input
    def geometrical_collecting_area(self) -> u.cm**2:
        return u.Quantity(self._data['geo_area'], 'cm^2')
        
    @property
    @u.quantity_input
    def filter_transmission(self) -> u.dimensionless_unscaled:
        return u.Quantity(self._data['filter'])
        
    @property
    @u.quantity_input
    def grating_efficiency(self) -> u.dimensionless_unscaled:
        # NOTE: this is just 1 for the filtergrams
        return u.Quantity(self._data['grating'])
        
    @property
    @u.quantity_input
    def detector_efficiency(self) -> u.dimensionless_unscaled:
        return u.Quantity(self._data['det'])
    
    @property
    @u.quantity_input
    def effective_area(self) -> u.cm**2:
        return (self.geometrical_collecting_area * 
                self.filter_transmission *
                self.grating_efficiency *
                self.detector_efficiency)
    
    @property
    @u.quantity_input
    def plate_scale(self) -> u.steradian / u.pixel:
        return u.Quantity(self._data['sr_per_pix'], 'steradian / pixel')
    
    @property
    @u.quantity_input
    def gain(self) -> u.ct / u.photon:
        # TODO: double check the units on this
        camera_gain = u.Quantity(self._data['gain'], 'ct / electron')
        # This is approximately the average energy to free an electron
        # in silicon
        energy_per_electron = 3.65 * u.Unit('eV / electron')
        energy_per_photon = const.h * const.c / self.wavelength / u.photon
        electron_per_photon = energy_per_photon / energy_per_electron
        # Cannot discharge less than one electron per photon
        discharge_floor = 1 * u.Unit('electron / photon')
        electron_per_photon[electron_per_photon<discharge_floor] = discharge_floor
        return electron_per_photon * camera_gain

    @property
    def wavelength_response(self) -> u.Unit('cm^2 ct / photon'):
        return self.effective_area * self.gain
    

class SpectrogramChannel(Channel):
    
    def __init__(self, order, instrument_file):
        name = f'MOXSI_S{int(np.fabs(order))}'
        super().__init__(name, instrument_file)
