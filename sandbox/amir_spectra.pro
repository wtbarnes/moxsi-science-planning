{% if energy_bin_edges is defined %}
energy_bin_edges = {{ energy_bin_edges | to_unit('keV') | list }}
{% else %}
e_min = {{ energy_min | to_unit('keV') }}
e_max = {{ energy_max | to_unit('keV') }}
n_e = {{ number_energy_bins }}
wid_ee = (e_max - e_min)/n_e
energy_bin_edges = findgen(n_e+1)*wid_ee + e_min
{% endif %}
tmp = get_edges(energy_bin_edges, edges_2=eee, mean=eee_mean, wid=eee_wid)
area = 1
eee_wid = 1
; Calculate various incident photon flux spectra
 ; spec = (phot / cm^2 / s / keV) * keV * cm^2 = phot / s
 ; For 2002 Jul 23, interval 9 (peak HXR) based on Caspi & Lin (2010)
 spec_x5 = (f_vth(eee, [1.27277, 3.63241, 1.0]) + f_vth(eee, [5.40545, 2.02360, 1.0]) + f_vth(eee, [58, 0.52, 1.0]) + f_3pow(eee, [19.9396, 1.50000, 48.1353, 2.65948, 400.000, 2.00000])) * eee_wid * area
; if not keyword_set(filter) then spec_x5[where(eee_mean ge 27.)] = interpol(spec_x5[where(eee_mean ge 20. and eee_mean lt 27.)], eee_mean[where(eee_mean ge 20. and eee_mean lt 27.)], eee_mean[where(eee_mean ge 27.)])
 ; With temps based on Caspi, Krucker, & Lin 2014, EMs adjusted to match GOES flux, and PL adjusted arbitrarily
 spec_m5 = (f_vth(eee, [0.175, 2.75755, 1.0]) + f_vth(eee, [0.6, 1.55112, 1.0]) + f_vth(eee, [5.8, 0.52, 1.0]) + f_3pow(eee, [1.99396, 1.50000, 35, 3.5, 400.000, 2.00000])) * eee_wid * area
 spec_m1 = (f_vth(eee, [0.044, 1.80964, 1.0]) + f_vth(eee, [0.135, 1.29260, 1.0]) + f_vth(eee, [1.2, 0.52, 1.0]) + f_3pow(eee, [0.39879, 1.50000, 20, 4, 400.000, 2.00000])) * eee_wid * area
 ; With temps based on Caspi, Krucker, & Lin 2014 for GOES, guesstimate for RHESSI, EMs adjusted to match GOES flux, and PL adjusted arbitrarily
 spec_c1 = (f_vth(eee, [0.01, 1.2, 1.0]) + f_vth(eee, [0.015, 0.896204, 1.0]) + f_vth(eee, [.08, 0.52, 1.0]) + f_vth(eee, [3.5, 0.2, 0.41]) + f_3pow(eee, [0.04, 1.50000, 15, 6, 400.000, 2.00000])) * eee_wid * area
 ; For strong and weak ARs, based on X123 rocket results of Caspi et al. (2015) -- B7 and B1.6 levels
 spec_b7 = (f_vth(eee, [0.031242997, 0.74194414, 0.41]) + f_vth(eee, [3.5, 0.23129428, 0.41])) * eee_wid * area
 spec_b1 = (f_vth(eee, [0.0014166682, 0.75919482, 1.0]) + f_vth(eee, [0.4, 0.25346233, 1.0])) * eee_wid * area
 spec_a1 = (f_vth(eee, [0.0003, 0.6, 1.0]) + f_vth(eee, [0.07, 0.22, 1.0])) * eee_wid * area
 ; For deep minimum, based on Sylwester et al. (2012)
 spec_min = (f_vth(eee, [0.0978000, 0.147357, 1.0])) * eee_wid * area
 ; Add B7 active-region background to the BIG flares, B1 to the small flare
 spec_x5 += spec_b7 & spec_m5 += spec_b7 & spec_m1 += spec_b7 & spec_c1 += spec_b1
 spec_b1 += spec_min & spec_a1 += spec_min