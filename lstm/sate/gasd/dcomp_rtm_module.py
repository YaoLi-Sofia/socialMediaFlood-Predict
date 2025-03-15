from math import cos, radians

import numpy as np
from numba import njit, prange

from numerical_routines import locate
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def perform_rtm_dcomp(
        bad_pixel_mask,
        acha_tc,
        nwp_pix_ozone, nwp_pix_p_sfc,
        nwp_pix_t_prof, nwp_pix_z_prof, nwp_p_std, nwp_pix_tpw_prof,
        nwp_pix_sfc_level, nwp_pix_tropo_level, nwp_pix_inversion_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof, rtm_sfc_level, rtm_tropo_level, rtm_inversion_level,
        rtm_ch_trans_atm_profile, rtm_ch_rad_atm_profile,
        ch_rad_toa_clear,
        geo_sat_zen,
        rtm_n_levels,
        nwp_n_levels
):
    # -----------------------------------------
    rtm_trans_ir_ac = np.empty(image_shape, 'f4')
    rtm_trans_ir_ac_nadir = np.empty(image_shape, 'f4')
    rtm_tpw_ac = np.empty(image_shape, 'f4')
    rtm_sfc_nwp = np.empty(image_shape, 'f4')
    rtm_rad_clear_sky_toc_ch07 = np.empty(image_shape, 'f4')
    rtm_rad_clear_sky_toa_ch07 = np.empty(image_shape, 'f4')
    rtm_ozone_path = np.empty(image_shape, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            if bad_pixel_mask[line_idx, elem_idx]:
                continue

            # - alias local variables
            # cld_height_loc = acha_zc[line_idx, elem_idx]
            cld_temp_loc = acha_tc[line_idx, elem_idx]
            # cld_press_loc = acha_pc[line_idx, elem_idx]

            # - for convenience, save nwp indices to local variables
            rtm_ozone_path[line_idx, elem_idx] = nwp_pix_ozone[line_idx, elem_idx]

            # - compute cloud level (idx_lev and prof_wgt) in nwp profiles
            # nwp_temp_prof = nwp_pix_t_prof[line_idx, elem_idx, :]  # - temperature profile
            # nwp_hgt_prof = nwp_pix_z_prof[line_idx, elem_idx, :]  # - height profile

            # - level indicies and weights in nwp and rtm profiles
            # placeholder_cld = cld_height_loc
            placeholder_cld, cld_height_loc, nwp_idx_lev, nwp_prof_wgt = t_to_pz_from_profile(
                cld_temp_loc,
                # nwp_temp_prof,
                nwp_pix_t_prof[line_idx, elem_idx],
                nwp_p_std,
                # nwp_hgt_prof,
                nwp_pix_z_prof[line_idx, elem_idx],
                nwp_pix_sfc_level[line_idx, elem_idx],
                nwp_pix_tropo_level[line_idx, elem_idx],
                nwp_pix_inversion_level[line_idx, elem_idx],
                rtm_n_levels,
            )

            placeholder_cld, cld_height_loc, rtm_idx_lev, rtm_prof_wgt = t_to_pz_from_profile(
                cld_temp_loc,
                rtm_t_prof[line_idx, elem_idx],
                rtm_p_std,
                rtm_z_prof[line_idx, elem_idx],
                rtm_sfc_level[line_idx, elem_idx],
                rtm_tropo_level[line_idx, elem_idx],
                rtm_inversion_level[line_idx, elem_idx],
                rtm_n_levels,
            )

            nwp_wv_prof = nwp_pix_tpw_prof[line_idx, elem_idx, :]  # - total water path profile

            if nwp_idx_lev == nwp_n_levels - 1:
                rtm_tpw_ac[line_idx, elem_idx] = nwp_wv_prof[nwp_idx_lev]  # - catch invalid data
            else:
                rtm_tpw_ac[line_idx, elem_idx] = (
                        nwp_wv_prof[nwp_idx_lev] +
                        nwp_prof_wgt * (nwp_wv_prof[nwp_idx_lev + 1] - nwp_wv_prof[nwp_idx_lev])
                )

            rtm_trans_ir_ac[line_idx, elem_idx] = (
                    rtm_ch_trans_atm_profile[7][line_idx, elem_idx][rtm_idx_lev] +
                    rtm_prof_wgt * (rtm_ch_trans_atm_profile[7][line_idx, elem_idx][rtm_idx_lev + 1] -
                                    rtm_ch_trans_atm_profile[7][line_idx, elem_idx][rtm_idx_lev])
            )

            rtm_trans_ir_ac_nadir[line_idx, elem_idx] = rtm_trans_ir_ac[line_idx, elem_idx] ** (
                cos(radians(geo_sat_zen[line_idx, elem_idx])))

            rtm_sfc_nwp[line_idx, elem_idx] = nwp_pix_p_sfc[line_idx, elem_idx]

            # clear_trans_prof_rtm = rtm_ch_trans_atm_profile[7][line_idx, elem_idx]

            clear_rad_prof_rtm = rtm_ch_rad_atm_profile[7][line_idx, elem_idx]

            rtm_rad_clear_sky_toc_ch07[line_idx, elem_idx] = clear_rad_prof_rtm[rtm_idx_lev]
            rtm_rad_clear_sky_toa_ch07[line_idx, elem_idx] = ch_rad_toa_clear[7][line_idx, elem_idx]

    return (
        rtm_trans_ir_ac,
        rtm_trans_ir_ac_nadir,
        rtm_tpw_ac,
        rtm_sfc_nwp,  # 不需要算 只是换了个名字
        rtm_rad_clear_sky_toc_ch07,
        rtm_rad_clear_sky_toa_ch07,  # 不需要算 只是换了个名字
        rtm_ozone_path  # 不需要算 只是换了个名字
    )


# -----------------------------------------------------------------
# computes pressure and height from temperature from given profiles
# considers possible inversion tropopause
@njit(nogil=True, error_model='numpy', boundscheck=True)
def t_to_pz_from_profile(
        temp,
        t_prof, p_prof, z_prof,
        sfc_idx, trp_idx, inv_idx,
        n_prof
):
    # -- temperature warmer than surface?
    if temp > np.amax(t_prof[trp_idx: sfc_idx + 1]):
        press = p_prof[sfc_idx]
        height = z_prof[sfc_idx]
        lev_idx = sfc_idx
        lev_wgt = 0.0
        return press, height, lev_idx, lev_wgt

    if temp < np.amin(t_prof[trp_idx: sfc_idx + 1]):
        press = p_prof[trp_idx]
        height = z_prof[trp_idx]
        lev_idx = trp_idx
        lev_wgt = 0.0
        return press, height, lev_idx, lev_wgt

    if inv_idx >= 0:
        n_levels_temp = sfc_idx - inv_idx + 1
        lev_idx = locate(t_prof[inv_idx: sfc_idx + 1], n_levels_temp, temp)
        if 0 <= lev_idx < n_levels_temp - 2:
            lev_idx += inv_idx
        # --- if no solution within an inversion, look above
        else:
            n_levels_temp = sfc_idx - trp_idx + 1
            lev_idx = locate(t_prof[trp_idx: sfc_idx + 1], n_levels_temp, temp)
            lev_idx = lev_idx + trp_idx
            lev_idx = max(0, min(n_prof - 2, lev_idx))
    # --- if no solution within an inversion, look above
    else:
        n_levels_temp = sfc_idx - trp_idx + 1
        lev_idx = locate(t_prof[trp_idx: sfc_idx + 1], n_levels_temp, temp)
        lev_idx = lev_idx + trp_idx
        lev_idx = max(0, min(n_prof - 2, lev_idx))

    # -- if solution is above tropo, set to tropo values
    if lev_idx < trp_idx:
        press = p_prof[trp_idx]
        height = z_prof[trp_idx]
        lev_idx = trp_idx
        lev_wgt = 0.0
        return press, height, lev_idx, lev_wgt

    # --- determine derivatives
    d_press = p_prof[lev_idx + 1] - p_prof[lev_idx]
    d_height = z_prof[lev_idx + 1] - z_prof[lev_idx]
    d_temp = t_prof[lev_idx + 1] - t_prof[lev_idx]

    if d_temp != 0.0:
        lev_wgt = (temp - t_prof[lev_idx]) / d_temp
        press = p_prof[lev_idx] + d_press * lev_wgt
        height = z_prof[lev_idx] + d_height * lev_wgt
    else:
        press = p_prof[lev_idx]
        height = z_prof[lev_idx]
        lev_wgt = 0.0

    return press, height, lev_idx, lev_wgt
