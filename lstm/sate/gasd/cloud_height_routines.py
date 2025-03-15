from math import log, nan, isnan

import numpy as np
from numba import njit, prange

from constants import (
    sym_yes,

)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from rtm_common import rtm_p_std
from utils import show_time


# -----------------------------------------------------------------------------
# opaque cloud height
# -----------------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def opaque_cloud_height(
        bad_pixel_mask,
        rtm_tropo_level, rtm_sfc_level, rtm_z_prof, rtm_t_prof,
        rtm_ch_rad_bb_cloud_profile, ch_rad_toa, ch_bt_toa
):
    dt_dz_strato = -0.0065  # k / m
    dp_dz_strato = -0.0150  # hpa / m

    pc_opaque_cloud = np.full(image_shape, nan, 'f4')
    zc_opaque_cloud = np.full(image_shape, nan, 'f4')
    tc_opaque_cloud = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            # --- skip bad pixels
            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue

            tropo_level_idx = rtm_tropo_level[line_idx, elem_idx]
            sfc_level_idx = rtm_sfc_level[line_idx, elem_idx]

            # --- restrict levels to consider
            level_idx_start = tropo_level_idx
            level_idx_end = sfc_level_idx

            rad_toa = ch_rad_toa[14][line_idx, elem_idx]
            bt_toa = ch_bt_toa[14][line_idx, elem_idx]

            rad_bb_cloud_profile = rtm_ch_rad_bb_cloud_profile[14][line_idx, elem_idx]
            t_prof = rtm_t_prof[line_idx, elem_idx]
            z_prof = rtm_z_prof[line_idx, elem_idx]

            if rad_toa < rad_bb_cloud_profile[tropo_level_idx]:
                tc_opaque_cloud[line_idx, elem_idx] = bt_toa
                zc_opaque_cloud[line_idx, elem_idx] = (
                        z_prof[tropo_level_idx] +
                        (tc_opaque_cloud[line_idx, elem_idx] - t_prof[level_idx_end]) / dt_dz_strato
                )
                pc_opaque_cloud[line_idx, elem_idx] = (
                        rtm_p_std[tropo_level_idx] +
                        (zc_opaque_cloud[line_idx, elem_idx] - z_prof[tropo_level_idx]) * dp_dz_strato
                )
                continue

            solution_found = False
            for level_idx in range(level_idx_end, level_idx_start, -1):

                if rad_bb_cloud_profile[level_idx - 1] > rad_toa > rad_bb_cloud_profile[level_idx]:
                    solution_found = True
                if rad_bb_cloud_profile[level_idx - 1] < rad_toa < rad_bb_cloud_profile[level_idx]:
                    solution_found = True

                if solution_found:
                    d_rad = rad_bb_cloud_profile[level_idx] - rad_bb_cloud_profile[level_idx - 1]
                    if d_rad != 0.00:
                        slope = (rtm_p_std[level_idx] - rtm_p_std[level_idx - 1]) / d_rad
                        pc_opaque_cloud[line_idx, elem_idx] = (
                                rtm_p_std[level_idx - 1] +
                                slope * (rad_toa - rad_bb_cloud_profile[level_idx - 1])
                        )
                        slope = (z_prof[level_idx] - z_prof[level_idx - 1]) / d_rad
                        zc_opaque_cloud[line_idx, elem_idx] = (
                                z_prof[level_idx - 1] +
                                slope * (rad_toa - rad_bb_cloud_profile[level_idx - 1])
                        )
                        slope = (t_prof[level_idx] - t_prof[level_idx - 1]) / d_rad
                        tc_opaque_cloud[line_idx, elem_idx] = (
                                t_prof[level_idx - 1] +
                                slope * (rad_toa - rad_bb_cloud_profile[level_idx - 1])
                        )
                    else:
                        pc_opaque_cloud[line_idx, elem_idx] = rtm_p_std[level_idx - 1]
                        zc_opaque_cloud[line_idx, elem_idx] = z_prof[level_idx - 1]
                        tc_opaque_cloud[line_idx, elem_idx] = t_prof[level_idx - 1]
                    break
            else:
                if rad_toa > rad_bb_cloud_profile[sfc_level_idx]:
                    pc_opaque_cloud[line_idx, elem_idx] = rtm_p_std[level_idx_end]
                    zc_opaque_cloud[line_idx, elem_idx] = rtm_z_prof[line_idx, elem_idx][level_idx_end]
                    tc_opaque_cloud[line_idx, elem_idx] = rtm_t_prof[line_idx, elem_idx][level_idx_end]
                else:
                    tc_opaque_cloud[line_idx, elem_idx] = nan
                    zc_opaque_cloud[line_idx, elem_idx] = nan
                    pc_opaque_cloud[line_idx, elem_idx] = nan

    return pc_opaque_cloud, zc_opaque_cloud, tc_opaque_cloud


# ----------------------------------------------------------------------
# routine to interpolate pressure to flight level altitude.
# ----------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_altitude_from_pressure(bad_pixel_mask, pc_in):
    # --- based on sarah monette calculations for hs3.  her calculation is based on:
    # --- http://psas.pdx.edu/rocketscience/pressurealtitude_derived.pdf.

    # --- constants from sarah monette
    pw1 = 227.9  # hpa
    pw2 = 56.89  # hpa
    pw3 = 11.01  # hpa
    p0 = 1013.25  # hpa
    lr_over_g = 0.190263
    z0 = 145422.16  # feet
    ln_1 = -20859.0
    ln_2 = 149255.0
    pn_4 = 0.000470034
    pn_3 = -0.364267
    pn_2 = 47.5627
    pn_1 = -2647.45
    pn_0 = 123842.0

    # --- initialize
    alt_out = np.full(image_shape, nan, 'f4')

    # ----------------------------------------------------------
    # loop through segment
    # ----------------------------------------------------------
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- initialize temporary value each time.
            pc_temp = nan
            alt_temp = nan

            # --- save indices
            # nwp_lon_idx = i_nwp[line_idx, elem_idx]
            # nwp_lat_idx = j_nwp[line_idx, elem_idx]

            # --- skip bad pixels
            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue

            # --- check if indices are valid
            # --- stw may not need this.
            # if nwp_lon_idx < 0 or nwp_lat_idx < 0:
            #     continue

            # # todo ???
            # if mask[line_idx, elem_idx]:
            #     continue

            # --- check if cld-top pressure is valid
            if isnan(pc_in[line_idx, elem_idx]):
                continue

            # --- place valid pressure in temp variable for readability in the
            # --- calculations below.
            pc_temp = pc_in[line_idx, elem_idx]

            # --- calculated altitude, in feet, from pressure.
            # --- 1st pivot point is directly from the pressure to
            # --- altitude from above reference.
            if pc_temp > pw1:
                alt_temp = (1.0 - (pc_temp / p0) ** lr_over_g) * z0

            # --- 2nd pivot point was modeled best with a natural log
            # --- fit.  from sarah monette.
            if pw1 >= pc_temp >= pw2:
                alt_temp = ln_1 * log(pc_temp) + ln_2

            # --- 3rd pivot point. modeled best with a polynomial
            # --- fit from sarah monette.
            if pw2 > pc_temp >= pw3:
                alt_temp = (pn_4 * pc_temp ** 4) + (pn_3 * pc_temp ** 3) + (pn_2 * pc_temp ** 2) + (
                        pn_1 * pc_temp) + pn_0

            if pc_temp < pw3:
                alt_temp = nan

            # --- assign final altitude, in feet, to the level2 array.
            alt_out[line_idx, elem_idx] = alt_temp

    return alt_out
