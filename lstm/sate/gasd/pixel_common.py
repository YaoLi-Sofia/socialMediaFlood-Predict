import numpy as np
from numba import i4, f4
from numba.typed import Dict

solar_rtm_tau_h2o_coef = Dict.empty(i4, f4[:])
solar_rtm_tau_h2o_coef.update({
    1: np.array([4.5112e-06, 5.1395e-05, -1.9105e-07], 'f4'),
    2: np.array([3.5560e-05, 6.7475e-04, -2.1035e-06], 'f4'),
    3: np.array([8.7089e-05, 2.9454e-03, -7.7288e-05], 'f4'),
    4: np.array([1.0741e-04, 3.6017e-03, -1.0748e-04], 'f4'),
    5: np.array([4.5871e-06, 1.0012e-03, 3.0030e-06], 'f4'),
    6: np.array([-1.3006e-04, 1.3990e-03, -4.6335e-05], 'f4'),
    7: np.array([1.2103e-03, 2.6850e-02, -1.3311e-03], 'f4'),
})

solar_rtm_tau_ray = Dict.empty(i4, f4)
solar_rtm_tau_ray.update({
    1: 2.0146e-01,
    2: 1.4393e-01,
    3: 5.8655e-02,
    4: 1.7662e-02,
    5: 1.4074e-03,
    6: 3.5377e-04,
    7: 2.5785e-05
})

solar_rtm_tau_o2 = Dict.empty(i4, f4)
solar_rtm_tau_o2.update({
    1: 0.00000,
    2: 0.00000,
    3: 0.00000,
    4: 0.00000,
    5: 0.00000,
    6: 0.00000,
    7: 0.00000
})

solar_rtm_tau_o3 = Dict.empty(i4, f4)
solar_rtm_tau_o3.update({
    1: 0.0000e+00,
    2: 3.8917e-06,
    3: 4.6808e-05,
    4: 0.0000e+00,
    5: 0.0000e+00,
    6: 0.0000e+00,
    7: 0.0000e+00
})

solar_rtm_tau_ch4 = Dict.empty(i4, f4)
solar_rtm_tau_ch4.update({
    1: 0.0000e+00,
    2: 0.0000e+00,
    3: 0.0000e+00,
    4: 0.0000e+00,
    5: 0.0000e+00,
    6: 0.0000e+00,
    7: 0.0000e+00
})

solar_rtm_tau_co2 = Dict.empty(i4, f4)
solar_rtm_tau_co2.update({
    1: 0.0000e+00,
    2: 0.0000e+00,
    3: 0.0000e+00,
    4: 0.0000e+00,
    5: 2.2504e-02,
    6: 3.3139e-06,
    7: 2.9989e-03
})

solar_rtm_tau_aer = Dict.empty(i4, f4)
solar_rtm_tau_aer.update({
    1: 0.0,
    2: 0.0,
    3: 0.0,
    4: 0.0,
    5: 0.0,
    6: 0.0,
    7: 0.0
})

solar_rtm_wo_aer = Dict.empty(i4, f4)
solar_rtm_wo_aer.update({
    1: 0.8,
    2: 0.8,
    3: 0.8,
    4: 0.8,
    5: 0.8,
    6: 0.8,
    7: 0.8
})

solar_rtm_g_aer = Dict.empty(i4, f4)
solar_rtm_g_aer.update({
    1: 0.6,
    2: 0.6,
    3: 0.6,
    4: 0.6,
    5: 0.6,
    6: 0.6,
    7: 0.6
})
