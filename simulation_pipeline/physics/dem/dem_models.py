"""
DEM models and utilities
"""
import astropy.units as u
import asdf
from scipy.interpolate import splrep, splev
import numpy as np
from sunkit_dem import GenericModel

from dem_algorithms import simple_reg_dem, sparse_em_init, sparse_em_solve
try:
    from dem_algorithms_fast import simple_reg_dem_gpu, simple_reg_dem_numba, simple_reg_dem_jax
except ImportError:
    pass


@u.quantity_input
def get_aia_temperature_response(channels, temperature_bin_centers: u.K):
    with asdf.open('../data/aia_temperature_response.asdf', 'r') as af:
        _TEMPERATURE_RESPONSE = af.tree
    response = {}
    T = _TEMPERATURE_RESPONSE['temperature']
    for c in channels:
        K = _TEMPERATURE_RESPONSE[f'{c.to_value("Angstrom"):.0f}']
        nots = splrep(T.value, K.value)
        response[str(c)] = u.Quantity(splev(temperature_bin_centers.value, nots), K.unit)
    return response


@u.quantity_input
def get_xrt_temperature_response(channels, temperature_bin_centers: u.K, correction_factor=1):
    with asdf.open('../data/xrt_temperature_response.asdf', 'r') as af:
        _TEMPERATURE_RESPONSE = af.tree
    response = {}
    T = _TEMPERATURE_RESPONSE['temperature']
    for c in channels:
        K = _TEMPERATURE_RESPONSE[c]['response']
        nots = splrep(T.value, K.value)
        fw1 = ' '.join(_TEMPERATURE_RESPONSE[c]['filter_wheel_1'].capitalize().split('_'))
        fw2 = ' '.join(_TEMPERATURE_RESPONSE[c]['filter_wheel_2'].capitalize().split('_'))
        key = f'{fw1}-{fw2}'
        response[key] = u.Quantity(
            splev(temperature_bin_centers.value, nots), K.unit) * correction_factor
    return response


class PlowmanModel(GenericModel):

    def _model(self, **kwargs):
        # Reshape some of the data
        data_array = self.data_matrix.to_value('ct pix-1').T
        uncertainty_array = np.array([self.data[k].uncertainty.array for k in self._keys]).T
        tresp_array = self.kernel_matrix.to_value('cm^5 ct pix-1 s-1').T
        logt = self.temperature_bin_centers.to_value('K')
        # Assume exposure times
        exp_unit = u.Unit('s')
        exp_times = np.array([self.data[k].meta['exptime'] for k in self._keys])

        # Solve
        method = kwargs.pop('method', None)
        if method == 'gpu':
            dem, chi2 = simple_reg_dem_gpu(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        elif method == 'numba':
            dem, chi2 = simple_reg_dem_numba(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        elif method == 'jax':
            dem, chi2 = simple_reg_dem_jax(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )
        else:
            dem, chi2 = simple_reg_dem(
                data_array,
                uncertainty_array,
                exp_times,
                logt,
                tresp_array,
                **kwargs
            )

        # Reshape outputs
        data_units = self.data_matrix.unit / exp_unit
        dem_unit = data_units / self.kernel_matrix.unit / self.temperature_bin_edges.unit
        em = (dem * np.diff(self.temperature_bin_edges)).T * dem_unit
        dem = dem.T * dem_unit

        return {'dem': dem,
                'em': em,
                'chi_squared': np.atleast_1d(chi2).T,
                'uncertainty': np.zeros(dem.shape)}

    @classmethod
    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'plowman'


class CheungModel(GenericModel):

    def _model(self, init_kwargs=None, solve_kwargs=None):
        # Extract needed keyword arguments
        init_kwargs = {} if init_kwargs is None else init_kwargs
        solve_kwargs = {} if solve_kwargs is None else solve_kwargs

        # Reshape some of the data
        exp_times = np.array([self.data[k].meta['exptime'] for k in self._keys])
        tr_list = self.kernel_matrix.to_value('cm^5 ct / (pix s)')
        logt_list = len(tr_list) * [np.log10(self.temperature_bin_centers.to_value('K'))]
        data_array = self.data_matrix.to_value().T
        uncertainty_array = np.array([self.data[k].uncertainty.array for k in self._keys]).T
        
        # Call model initializer
        k_basis_int, _, basis_funcs, _ = sparse_em_init(logt_list, tr_list, **init_kwargs)
        # Solve
        coeffs, _, _ = sparse_em_solve(data_array,
                                       uncertainty_array,
                                       exp_times,
                                       k_basis_int,
                                       **solve_kwargs)

        # Compute product between coefficients and basis functions
        # NOTE: I am assuming that all basis functions are computed on the same temperature
        # array defined by temperature_bin_centers
        dem = np.tensordot(coeffs, basis_funcs, axes=(2, 0))

        # Reshape outputs
        dem_unit = self.data_matrix.unit / self.kernel_matrix.unit / self.temperature_bin_edges.unit
        dem = dem.T * dem_unit

        return {'dem': dem,
                'uncertainty': np.zeros(dem.shape)}

    def defines_model_for(self, *args, **kwargs):
        return kwargs.get('model') == 'cheung'
