import numpy as np
from numba import njit, prange

from cloud_type_algo_module import (
    # et_cloudiness_class_missing,
    et_cloudiness_class_prob_clear,
    et_cloudiness_class_clear,

    # et_cloud_type_first,
    et_cloud_type_clear,
    et_cloud_type_prob_clear,
    et_cloud_type_first_water,
    et_cloud_type_fog,
    et_cloud_type_water,
    et_cloud_type_supercooled,
    et_cloud_type_last_water,
    et_cloud_type_opaque_ice,
    et_cloud_type_cirrus,
    et_cloud_type_overlap,
    et_cloud_type_unknown,
    et_cloud_type_dust,
    et_cloud_type_smoke,
    et_cloud_type_fire,
    # et_cloud_type_last,
    et_cloud_type_missing,

    # et_cloud_phase_first,
    # et_cloud_phase_last,
    determine_type_water, determine_type_ice,
    get_ice_probability
)
from public import image_shape, image_number_of_lines, image_number_of_elements
from utils import show_time


# -----------------------------------------------------------------------------
# - this computes cloud type without lrc correction
# - lrc correction can be done with the optional force_ice keyword:
# - 
# -  lrc core is water and pixel is ice  ==>   correct the pixel to water
# -  lrc core is ice and pixel is water  ==> run this with force_ice =  True 
# 
# - akh, when would not be in a 'force_ice' scenario.  this is how the algo
#        works,  why is there no corresponding 'force_water' flag
# ----------------------------------------------------------------------------- 
@njit(nogil=True, error_model='numpy', boundscheck=True)
def cloud_type_pixel(
        inp_sat_rad_ch14,
        inp_rtm_rad_ch14_bb_prof,
        inp_rtm_rad_ch14_atm_sfc,
        inp_sat_rad_ch09,
        inp_rtm_rad_ch09_bb_prof,
        inp_rtm_rad_ch09_atm_sfc,
        inp_rtm_covar_ch09_ch14_5x5,
        inp_rtm_tropo_lev,
        inp_rtm_sfc_lev,
        inp_rtm_t_prof,
        inp_rtm_z_prof,
        inp_sat_bt_ch14,
        inp_geo_sol_zen,
        inp_rtm_ref_ch05_clear,
        inp_sat_ref_ch05,
        inp_sfc_ems_ch07,
        inp_sat_ref_ch07,
        inp_sat_bt_ch11,
        inp_rtm_bt_ch14_3x3_std,
        inp_rtm_bt_ch14_atm_sfc,
        inp_rtm_bt_ch15_atm_sfc,
        inp_sat_bt_ch15,
        inp_rtm_bt_ch09_3x3_max,
        inp_rtm_ems_tropo_ch14,
        inp_rtm_beta_110um_120um_tropo,
        inp_rtm_beta_110um_133um_tropo,
        force_ice=False,
        force_water=False
):
    if force_ice:
        force_ice_phase = True
    else:
        force_ice_phase = False

    if force_water:
        force_water_phase = True
    else:
        force_water_phase = False

    ice_prob, is_cirrus, is_water, t_cld, z_cld = get_ice_probability(
        inp_sat_rad_ch14,
        inp_rtm_rad_ch14_bb_prof,
        inp_rtm_rad_ch14_atm_sfc,
        inp_sat_rad_ch09,
        inp_rtm_rad_ch09_bb_prof,
        inp_rtm_rad_ch09_atm_sfc,
        inp_rtm_covar_ch09_ch14_5x5,
        inp_rtm_tropo_lev,
        inp_rtm_sfc_lev,
        inp_rtm_t_prof,
        inp_rtm_z_prof,
        inp_sat_bt_ch14,
        inp_geo_sol_zen,
        inp_rtm_ref_ch05_clear,
        inp_sat_ref_ch05,
        inp_sfc_ems_ch07,
        inp_sat_ref_ch07,
        inp_sat_bt_ch11,
        inp_rtm_bt_ch14_3x3_std,
        inp_rtm_bt_ch14_atm_sfc,
        inp_rtm_bt_ch15_atm_sfc,
        inp_sat_bt_ch15,
        inp_rtm_bt_ch09_3x3_max
    )

    # - compute type from ice probability phase discrimination
    if (ice_prob > 0.5 or force_ice_phase) and not force_water_phase:
        c_type = determine_type_ice(
            inp_rtm_ems_tropo_ch14,
            inp_sat_bt_ch14,
            inp_rtm_beta_110um_120um_tropo,
            inp_rtm_beta_110um_133um_tropo,
            is_water, is_cirrus
        )
    else:
        c_type = determine_type_water(z_cld, t_cld)

    # - optional output of ice probability
    ice_prob_out = ice_prob

    return c_type, ice_prob_out


# ====================================================================
# universal cloud type bridge
# ====================================================================
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_cloud_type(
        geo_sol_zen,
        rtm_sfc_level, rtm_tropo_level, rtm_t_prof, rtm_z_prof, rtm_ch_rad_bb_cloud_profile,
        ch_rad_toa, ch_ref_toa, ch_bt_toa, ch_rad_toa_clear, ch_ref_toa_clear, ch_bt_toa_clear,
        ch_ems_tropo, ch_sfc_ems,
        bt_ch14_std_3x3, covar_ch09_ch14_5x5, bt_ch09_max_3x3,
        beta_110um_120um_tropo_rtm, beta_110um_133um_tropo_rtm,
        bad_pixel_mask, cld_mask_cld_mask, dust_mask, smoke_mask, fire_mask,
        i_lrc, j_lrc,
):
    # ice_prob = -999.0

    cld_type = np.empty(image_shape, 'i1')
    # -----------    loop over lrc core pixels to get ice probability -----
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if bad_pixel_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_missing
                continue

            # cld_mask_cld_mask与dust_mask、smoke_mask、fire_mask是重合的 先赋值clear标记就会跳过后面三种赋值
            if cld_mask_cld_mask[i, j] == et_cloudiness_class_clear:
                cld_type[i, j] = et_cloud_type_clear
                continue

            if cld_mask_cld_mask[i, j] == et_cloudiness_class_prob_clear:
                cld_type[i, j] = et_cloud_type_prob_clear
                continue

            if dust_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_dust
                continue

            if smoke_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_smoke
                continue

            if fire_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_fire
                continue

            # - take only lrc cores
            if i != i_lrc[i, j] or j != j_lrc[i, j]:
                continue

            # populate_input(i, j, type_inp)

            # -----------------------------------------------------------------------------------
            # - sat
            # -----------------------------------------------------------------------------------

            type_inp_sat_rad_ch14 = ch_rad_toa[14][i, j]
            type_inp_sat_bt_ch14 = ch_bt_toa[14][i, j]
            type_inp_sat_bt_ch15 = ch_bt_toa[15][i, j]
            type_inp_sat_ref_ch05 = ch_ref_toa[5][i, j]
            type_inp_sat_ref_ch07 = ch_ref_toa[7][i, j]
            type_inp_sat_rad_ch09 = ch_rad_toa[9][i, j]
            type_inp_sat_bt_ch11 = ch_bt_toa[11][i, j]

            # -----------------------------------------------------------------------------------
            # - rtm
            # -----------------------------------------------------------------------------------

            type_inp_rtm_t_prof = rtm_t_prof[i, j]
            type_inp_rtm_z_prof = rtm_z_prof[i, j]
            type_inp_rtm_tropo_lev = rtm_tropo_level[i, j]
            type_inp_rtm_sfc_lev = rtm_sfc_level[i, j]

            type_inp_rtm_ref_ch05_clear = ch_ref_toa_clear[5][i, j]

            type_inp_rtm_rad_ch14_bb_prof = rtm_ch_rad_bb_cloud_profile[14][i, j]

            type_inp_rtm_bt_ch14_3x3_std = bt_ch14_std_3x3[i, j]
            type_inp_rtm_rad_ch14_atm_sfc = ch_rad_toa_clear[14][i, j]
            type_inp_rtm_bt_ch14_atm_sfc = ch_bt_toa_clear[14][i, j]
            type_inp_rtm_ems_tropo_ch14 = ch_ems_tropo[14][i, j]

            type_inp_rtm_covar_ch09_ch14_5x5 = covar_ch09_ch14_5x5[i, j]

            type_inp_rtm_beta_110um_120um_tropo = beta_110um_120um_tropo_rtm[i, j]
            type_inp_rtm_bt_ch15_atm_sfc = ch_bt_toa_clear[15][i, j]

            type_inp_rtm_beta_110um_133um_tropo = beta_110um_133um_tropo_rtm[i, j]

            type_inp_rtm_bt_ch09_3x3_max = bt_ch09_max_3x3[i, j]
            type_inp_rtm_rad_ch09_atm_sfc = ch_rad_toa_clear[9][i, j]
            type_inp_rtm_rad_ch09_bb_prof = rtm_ch_rad_bb_cloud_profile[9][i, j]
            # type_inp_rtm_rad_ch09_bb_prof = rtm_ch_rad_bb_cloud_profile[9][i, j]

            # -----------------------------------------------------------------------------------
            # - geo
            # -----------------------------------------------------------------------------------
            type_inp_geo_sol_zen = geo_sol_zen[i, j]

            # -----------------------------------------------------------------------------------
            # - sfc
            # -----------------------------------------------------------------------------------
            type_inp_sfc_ems_ch07 = ch_sfc_ems[7][i, j]

            c_type, ice_prob = cloud_type_pixel(
                type_inp_sat_rad_ch14,
                type_inp_rtm_rad_ch14_bb_prof,
                type_inp_rtm_rad_ch14_atm_sfc,
                type_inp_sat_rad_ch09,
                type_inp_rtm_rad_ch09_bb_prof,
                type_inp_rtm_rad_ch09_atm_sfc,
                type_inp_rtm_covar_ch09_ch14_5x5,
                type_inp_rtm_tropo_lev,
                type_inp_rtm_sfc_lev,
                type_inp_rtm_t_prof,
                type_inp_rtm_z_prof,
                type_inp_sat_bt_ch14,
                type_inp_geo_sol_zen,
                type_inp_rtm_ref_ch05_clear,
                type_inp_sat_ref_ch05,
                type_inp_sfc_ems_ch07,
                type_inp_sat_ref_ch07,
                type_inp_sat_bt_ch11,
                type_inp_rtm_bt_ch14_3x3_std,
                type_inp_rtm_bt_ch14_atm_sfc,
                type_inp_rtm_bt_ch15_atm_sfc,
                type_inp_sat_bt_ch15,
                type_inp_rtm_bt_ch09_3x3_max,
                type_inp_rtm_ems_tropo_ch14,
                type_inp_rtm_beta_110um_120um_tropo,
                type_inp_rtm_beta_110um_133um_tropo,
            )
            cld_type[i, j] = c_type

    # - now loop over all non lrc-cores
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):

            if bad_pixel_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_missing
                continue

            if cld_mask_cld_mask[i, j] == et_cloudiness_class_clear:
                cld_type[i, j] = et_cloud_type_clear
                continue

            if cld_mask_cld_mask[i, j] == et_cloudiness_class_prob_clear:
                cld_type[i, j] = et_cloud_type_prob_clear
                continue

            if dust_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_dust
                continue

            if smoke_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_smoke
                continue

            if fire_mask[i, j] == 1:
                cld_type[i, j] = et_cloud_type_fire
                continue

            ii = i_lrc[i, j]
            jj = j_lrc[i, j]

            # - we don't need the lrc cores again
            if i == ii and j == jj:
                continue

            # populate_input(i, j, type_inp)

            # -----------------------------------------------------------------------------------
            # - sat
            # -----------------------------------------------------------------------------------

            type_inp_sat_rad_ch14 = ch_rad_toa[14][i, j]
            type_inp_sat_bt_ch14 = ch_bt_toa[14][i, j]
            type_inp_sat_bt_ch15 = ch_bt_toa[15][i, j]
            type_inp_sat_ref_ch05 = ch_ref_toa[5][i, j]
            type_inp_sat_ref_ch07 = ch_ref_toa[7][i, j]
            type_inp_sat_rad_ch09 = ch_rad_toa[9][i, j]
            type_inp_sat_bt_ch11 = ch_bt_toa[11][i, j]

            # -----------------------------------------------------------------------------------
            # - rtm
            # -----------------------------------------------------------------------------------

            type_inp_rtm_t_prof = rtm_t_prof[i, j]
            type_inp_rtm_z_prof = rtm_z_prof[i, j]
            type_inp_rtm_tropo_lev = rtm_tropo_level[i, j]
            type_inp_rtm_sfc_lev = rtm_sfc_level[i, j]

            type_inp_rtm_ref_ch05_clear = ch_ref_toa_clear[5][i, j]

            type_inp_rtm_rad_ch14_bb_prof = rtm_ch_rad_bb_cloud_profile[14][i, j]
            # type_inp_rtm_rad_ch14_bb_prof = rtm_ch_rad_bb_cloud_profile[14][i, j]

            type_inp_rtm_bt_ch14_3x3_std = bt_ch14_std_3x3[i, j]
            type_inp_rtm_rad_ch14_atm_sfc = ch_rad_toa_clear[14][i, j]
            type_inp_rtm_bt_ch14_atm_sfc = ch_bt_toa_clear[14][i, j]
            type_inp_rtm_ems_tropo_ch14 = ch_ems_tropo[14][i, j]

            type_inp_rtm_covar_ch09_ch14_5x5 = covar_ch09_ch14_5x5[i, j]

            type_inp_rtm_beta_110um_120um_tropo = beta_110um_120um_tropo_rtm[i, j]
            type_inp_rtm_bt_ch15_atm_sfc = ch_bt_toa_clear[15][i, j]

            type_inp_rtm_beta_110um_133um_tropo = beta_110um_133um_tropo_rtm[i, j]

            type_inp_rtm_bt_ch09_3x3_max = bt_ch09_max_3x3[i, j]
            type_inp_rtm_rad_ch09_atm_sfc = ch_rad_toa_clear[9][i, j]
            type_inp_rtm_rad_ch09_bb_prof = rtm_ch_rad_bb_cloud_profile[9][i, j]
            # type_inp_rtm_rad_ch09_bb_prof = rtm_ch_rad_bb_cloud_profile[9][i, j]

            # -----------------------------------------------------------------------------------
            # - geo
            # -----------------------------------------------------------------------------------
            type_inp_geo_sol_zen = geo_sol_zen[i, j]

            # -----------------------------------------------------------------------------------
            # - sfc
            # -----------------------------------------------------------------------------------
            type_inp_sfc_ems_ch07 = ch_sfc_ems[7][i, j]

            c_type, ice_prob = cloud_type_pixel(
                type_inp_sat_rad_ch14,
                type_inp_rtm_rad_ch14_bb_prof,
                type_inp_rtm_rad_ch14_atm_sfc,
                type_inp_sat_rad_ch09,
                type_inp_rtm_rad_ch09_bb_prof,
                type_inp_rtm_rad_ch09_atm_sfc,
                type_inp_rtm_covar_ch09_ch14_5x5,
                type_inp_rtm_tropo_lev,
                type_inp_rtm_sfc_lev,
                type_inp_rtm_t_prof,
                type_inp_rtm_z_prof,
                type_inp_sat_bt_ch14,
                type_inp_geo_sol_zen,
                type_inp_rtm_ref_ch05_clear,
                type_inp_sat_ref_ch05,
                type_inp_sfc_ems_ch07,
                type_inp_sat_ref_ch07,
                type_inp_sat_bt_ch11,
                type_inp_rtm_bt_ch14_3x3_std,
                type_inp_rtm_bt_ch14_atm_sfc,
                type_inp_rtm_bt_ch15_atm_sfc,
                type_inp_sat_bt_ch15,
                type_inp_rtm_bt_ch09_3x3_max,
                type_inp_rtm_ems_tropo_ch14,
                type_inp_rtm_beta_110um_120um_tropo,
                type_inp_rtm_beta_110um_133um_tropo,
            )
            cld_type[i, j] = c_type

            # --- set lrc value
            cld_type_lrc = et_cloud_type_unknown
            # todo ii,jj
            if ii > -1 and jj > -1:
                cld_type_lrc = cld_type[ii, jj]

            # - compare this c_type with lrc

            #  - identical or lrc is not valid => take the current
            # todo ii,jj
            if c_type == cld_type_lrc or ii < 0 or jj < 0 or cld_type_lrc == et_cloud_type_unknown:
                cld_type[i, j] = c_type
            else:
                # - if lrc core is water phase ==> use lrc
                if et_cloud_type_first_water <= cld_type_lrc <= et_cloud_type_last_water:

                    # akh says not to overwrite water,  fog or supercooled
                    if et_cloud_type_first_water <= c_type <= et_cloud_type_last_water:

                        cld_type[i, j] = c_type
                    else:
                        # - the original ice pixels should also be checking on supercooled, fog or water.
                        c_type, ice_prob = cloud_type_pixel(
                            type_inp_sat_rad_ch14,
                            type_inp_rtm_rad_ch14_bb_prof,
                            type_inp_rtm_rad_ch14_atm_sfc,
                            type_inp_sat_rad_ch09,
                            type_inp_rtm_rad_ch09_bb_prof,
                            type_inp_rtm_rad_ch09_atm_sfc,
                            type_inp_rtm_covar_ch09_ch14_5x5,
                            type_inp_rtm_tropo_lev,
                            type_inp_rtm_sfc_lev,
                            type_inp_rtm_t_prof,
                            type_inp_rtm_z_prof,
                            type_inp_sat_bt_ch14,
                            type_inp_geo_sol_zen,
                            type_inp_rtm_ref_ch05_clear,
                            type_inp_sat_ref_ch05,
                            type_inp_sfc_ems_ch07,
                            type_inp_sat_ref_ch07,
                            type_inp_sat_bt_ch11,
                            type_inp_rtm_bt_ch14_3x3_std,
                            type_inp_rtm_bt_ch14_atm_sfc,
                            type_inp_rtm_bt_ch15_atm_sfc,
                            type_inp_sat_bt_ch15,
                            type_inp_rtm_bt_ch09_3x3_max,
                            type_inp_rtm_ems_tropo_ch14,
                            type_inp_rtm_beta_110um_120um_tropo,
                            type_inp_rtm_beta_110um_133um_tropo,
                            force_water=True
                        )
                        cld_type[i, j] = c_type

                # - lrc core is ice phase
                elif ((c_type == et_cloud_type_fog or c_type == et_cloud_type_water)
                      and (cld_type_lrc == et_cloud_type_cirrus
                           or cld_type_lrc == et_cloud_type_overlap
                           or cld_type_lrc == et_cloud_type_opaque_ice)):

                    cld_type[i, j] = et_cloud_type_cirrus

                # - lrc core is ice phase and current is supercooled => switch to ice
                elif ((cld_type_lrc == et_cloud_type_cirrus
                       or cld_type_lrc == et_cloud_type_opaque_ice
                       or cld_type_lrc == et_cloud_type_overlap)
                      and c_type == et_cloud_type_supercooled):

                    c_type, ice_prob = cloud_type_pixel(
                        type_inp_sat_rad_ch14,
                        type_inp_rtm_rad_ch14_bb_prof,
                        type_inp_rtm_rad_ch14_atm_sfc,
                        type_inp_sat_rad_ch09,
                        type_inp_rtm_rad_ch09_bb_prof,
                        type_inp_rtm_rad_ch09_atm_sfc,
                        type_inp_rtm_covar_ch09_ch14_5x5,
                        type_inp_rtm_tropo_lev,
                        type_inp_rtm_sfc_lev,
                        type_inp_rtm_t_prof,
                        type_inp_rtm_z_prof,
                        type_inp_sat_bt_ch14,
                        type_inp_geo_sol_zen,
                        type_inp_rtm_ref_ch05_clear,
                        type_inp_sat_ref_ch05,
                        type_inp_sfc_ems_ch07,
                        type_inp_sat_ref_ch07,
                        type_inp_sat_bt_ch11,
                        type_inp_rtm_bt_ch14_3x3_std,
                        type_inp_rtm_bt_ch14_atm_sfc,
                        type_inp_rtm_bt_ch15_atm_sfc,
                        type_inp_sat_bt_ch15,
                        type_inp_rtm_bt_ch09_3x3_max,
                        type_inp_rtm_ems_tropo_ch14,
                        type_inp_rtm_beta_110um_120um_tropo,
                        type_inp_rtm_beta_110um_133um_tropo,
                        force_ice=True
                    )
                    cld_type[i, j] = c_type

                # -- this is mainly cirrus / opaque ice => keep current
                else:
                    cld_type[i, j] = c_type

    return cld_type
