from math import nan

import numpy as np
from numba import njit

from .dcomp_lut_mod import lut_thick_cloud_rfl


# 波段 7
@njit(nogil=True, error_model='numpy', boundscheck=True)
def thick_cloud_cps(
        rfl_nir_obs, channel_nir, pixel_is_water_phase,
        pos_sol, pos_sat, pos_azi,
        planck_rad, rad_to_refl,
        ch_phase_cld_refl, phase_cld_ems,
        dcomp_mode
):
    cps_vec_lut = np.arange(4, 22, 2) / 10.0

    if pixel_is_water_phase:
        phase_num = 0
    else:
        phase_num = 1

    if dcomp_mode == 3:  # todo 把4也加上?
        rfl_lut, ems_lut = lut_thick_cloud_rfl(
            pos_sol, pos_sat, pos_azi,
            channel_nir, phase_num,
            ch_phase_cld_refl, phase_cld_ems, has_ems=True,
        )
        rfl = rfl_lut
        rad = ems_lut * planck_rad
        rfl_terr = rad * rad_to_refl
        rfl += rfl_terr
    else:
        rfl, ems = lut_thick_cloud_rfl(
            pos_sol, pos_sat, pos_azi,
            channel_nir, phase_num,
            ch_phase_cld_refl, phase_cld_ems, has_ems=False,
        )

    cps_pos = -999
    cps = nan

    for ii in range(8):
        if rfl[ii] > rfl_nir_obs > rfl[ii + 1]:
            cps_pos = ii

    if cps_pos != -999:
        cps_wgt = (rfl[cps_pos] - rfl_nir_obs) / (rfl[cps_pos] - rfl[cps_pos + 1])
        cps = cps_vec_lut[cps_pos] + (cps_wgt * 0.2)

    return cps
