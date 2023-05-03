"""
Functions that implement analytical DEM paramterization from Guennou et al. (2014)
"""
import astropy.units as u
import numpy as np


@u.quantity_input
def guennou_dem(temperature: u.K, T_P: u.K, EM_total: u.cm**(-5), alpha, sigma, sigma_fw=0.15):
    T_P = T_P.to(temperature.unit)
    T_0 = _calculate_tangent_point(T_P, alpha, sigma_fw)
    dem_low = _guennou_dem_low(temperature, T_0, T_P, alpha, sigma_fw)
    dem_high = _guennou_dem_high(temperature, T_P, sigma, sigma_fw)
    dem = _guennou_dem_connection(temperature, T_P, sigma_fw)
    dem[temperature < T_0] = dem_low[temperature < T_0]
    dem[temperature > T_P] = dem_high[temperature > T_P]
    dem = dem * EM_total
    return dem


def _calculate_tangent_point(T_P, alpha, sigma):
    # Point of tangency between Gaussian and power-law 
    # in log-log space
    return T_P * 10**(-alpha * (sigma**2) * np.log(10))


def _gaussian(x, sigma):
    return np.exp(-((x/sigma)**2)/2)/sigma/np.sqrt(2*np.pi)


def _guennou_dem_low(temperature, T_0, T_P, alpha, sigma):
    x = np.log10(T_0.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma) * (temperature / T_0)**alpha


def _guennou_dem_high(temperature, T_P, sigma, sigma_fw):
    x = np.log10(temperature.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma) * sigma / sigma_fw
    
    
def _guennou_dem_connection(temperature, T_P, sigma):
    x = np.log10(temperature.to_value('K')) - np.log10(T_P.to_value('K'))
    return _gaussian(x, sigma)