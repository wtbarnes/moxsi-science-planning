"""
Utilities for building spectral table
"""
import asdf
import astropy.units as u
import hissw
import ndcube
from ndcube.extra_coords import QuantityTableCoordinate


_chianti_script = '''
ioneq_name = '{{ [ ssw_home, 'packages/chianti/dbase/ioneq', ioneq_file ] | join('/') }}'
abund_name = '{{ [ ssw_home, 'packages/chianti/dbase/abundance', abundance_file ] | join('/') }}'
wavelength = {{ wavelength | to_unit('Angstrom') | list }}
wave_min = wavelength[0]
wave_max = wavelength[-1]
log_temperature = {{ temperature | to_unit('K') | log10 }}
log_em = {{ emission_measure | to_unit('cm-5') | log10 }}
density = {{ density | to_unit('cm-3') }}

;generate transition structure for selected wavelength and temperature range
ch_synthetic, wave_min, wave_max, output=transitions,$
              ioneq_name=ioneq_name,$
              logt_isothermal=log_temperature,$
              logem_isothermal=log_em,$
              density=density[0]

;compute the spectra as a function of lambda and T
make_chianti_spec, transitions, wavelength, spectrum,$
                   abund_name=abund_name,$
                   /continuum,/lookup,/photons
'''


def compute_spectral_table(temperature,
                           density,
                           wavelength,
                           ioneq_filename,
                           abundance_filename,
                           emission_measure=1*u.Unit('cm-5')):
    # setup SSW environment and inputs
    input_args = {
        'wavelength': wavelength,
        'emission_measure': emission_measure,
        'ioneq_file': ioneq_filename,
        'abundance_file': abundance_filename,
    }
    env = hissw.Environment(ssw_packages=['chianti'])

    # Iterate over T and n values
    all_spectra = []
    for T, n in zip(temperature, density):
        input_args['temperature'] = T
        input_args['density'] = n
        spec = _get_isothermal_spectra(env, input_args)
        all_spectra.append(spec)

    return u.Quantity(all_spectra)


def _get_isothermal_spectra(env, input_args):
    """
    Compute isothermal spectra for a specified temperature and density.

    Parameters
    ----------
    temperature (_type_): _description_
    density (_type_): _description_

    Returns
    --------
    _type_: _description_
    """
    output = env.run(_chianti_script, args=input_args, save_vars=['spectrum'])
    spectrum = output['spectrum']['spectrum'][0]
    # The unit string from CHIANTI uses representations astropy
    # does not like so we fake those units
    u.add_enabled_units([
        u.def_unit('photons', represents=u.photon),
        u.def_unit('Angstroms', represents=u.Angstrom)
    ])
    spectrum_unit = u.Unit(output['spectrum']['units'][0][1].decode('utf-8'))
    spectrum = u.Quantity(spectrum, spectrum_unit)
    # Originally, the spectrum was computed assuming unit EM.
    # Divide through to get the units right
    spectrum = spectrum / input_args['emission_measure']

    return spectrum.to('cm3 ph Angstrom-1 s-1 sr-1')


def write_spectral_table(filename,
                         spectrum,
                         temperature,
                         density,
                         wavelength,
                         ioneq_filename,
                         abundance_filename):
    tree = {}
    tree['temperature'] = temperature
    tree['density'] = density
    tree['wavelength'] = wavelength
    tree['ioneq_filename'] = ioneq_filename
    tree['abundance_filename'] = abundance_filename
    tree['spectrum'] = spectrum
    with asdf.AsdfFile(tree) as asdf_file:
        asdf_file.write_to(filename)


def read_spectral_table(filename):
    """
    Read a spectral table file and return an NDCube

    Parameters
    ----------
    filename : `str` or path-like
        Path to ASDF file containing 
    """
    # Read file
    with asdf.open(filename, mode='r', copy_arrays=True) as af:
        temperature = af['temperature']
        density = af['density']
        wavelength = af['wavelength']
        ioneq_filename = af['ioneq_filename']
        abundance_filename = af['abundance_filename']
        spectrum = af['spectrum']

    # Build cube
    gwcs = (
        QuantityTableCoordinate(wavelength, physical_types='em.wl') &
        QuantityTableCoordinate(temperature, physical_types='phys.temperature')
    ).wcs
    meta = {
        'ioneq_filename': ioneq_filename,
        'abundance_filename': abundance_filename,
    }
    spec_cube = ndcube.NDCube(spectrum, wcs=gwcs, meta=meta)
    spec_cube.extra_coords.add('density', (0,), density, physical_types='phys.density')

    return spec_cube
