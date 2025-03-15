import numpy as np
from numba import i1, f4
from numba.typed import Dict

n_k = 5
n_l = 101
n_m = n_l - 1

n_xd = 8
n_cd = n_xd + 1
len_cd = n_cd * n_m
len_cdb = len_cd * 4

n_xo = 9
n_co = n_xo + 1
len_co = n_co * n_m
len_cob = len_co * 4

n_xc = 4
n_cc = n_xc + 1
len_cc = n_cc * n_m
len_ccb = len_cc * 4

n_xl = 2
n_cl = n_xl + 1
len_cl = n_cl * n_m
len_clb = len_cl * 4

n_xs = 11
n_cs = n_xs + 1
len_cs = n_cs * n_m
len_csb = len_cs * 4

n_xw = n_xl + n_xs

comp = ('dry', 'ozo', 'wco', 'wtl', 'wts')

len_cf = (len_cdb, len_cob, len_ccb, len_clb, len_csb)

n_d = 10
n_channels = 10

coef_sensor = 'ahi'

coef_dry = Dict.empty(i1, f4[:, :])
coef_ozon = Dict.empty(i1, f4[:, :])
coef_wvp_cont = Dict.empty(i1, f4[:, :])
coef_wvp_solid = Dict.empty(i1, f4[:, :])
coef_wvp_liquid = Dict.empty(i1, f4[:, :])

for c in np.arange(7, 17, dtype='i1'):
    coef_dry[c] = np.fromfile(
        'static/ahidry101.dat', dtype='f4',
        offset=(c - 7) * 4 * n_m * n_cd, count=n_m * n_cd
    ).reshape((n_m, n_cd))
    coef_ozon[c] = np.fromfile(
        'static/ahiozo101.dat', dtype='f4',
        offset=(c - 7) * 4 * n_m * n_co, count=n_m * n_co
    ).reshape((n_m, n_co))
    coef_wvp_cont[c] = np.fromfile(
        'static/ahiwco101.dat', dtype='f4',
        offset=(c - 7) * 4 * n_m * n_cc, count=n_m * n_cc
    ).reshape((n_m, n_cc))
    coef_wvp_solid[c] = np.fromfile(
        'static/ahiwts101.dat', dtype='f4',
        offset=(c - 7) * 4 * n_m * n_cs, count=n_m * n_cs
    ).reshape((n_m, n_cs))
    coef_wvp_liquid[c] = np.fromfile(
        'static/ahiwtl101.dat', dtype='f4',
        offset=(c - 7) * 4 * n_m * n_cl, count=n_m * n_cl
    ).reshape((n_m, n_cl))
