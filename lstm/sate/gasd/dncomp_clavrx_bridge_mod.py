from math import nan, pi, cos, radians

import numpy as np
from numba import i1, f4, prange, njit
from numba.typed import Dict

from cx_dncomp.dcomp_array_loop_sub import (
    dcomp_array_loop,
    em_cloud_mask_cloudy, em_cloud_mask_prob_cloudy
)
from cx_dncomp.dncomp_trans_atmos_mod import trans_atm_above_cloud
from dcomp_rtm_module import perform_rtm_dcomp
from planck import planck_rad_fast
from public import (
    image_shape,
    image_number_of_lines,
    image_number_of_elements,
)
from utils import show_time

f42d_array = f4[:, :]


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def awg_cloud_dncomp_algorithm(
        bad_pixel_mask,
        # acha_zc,
        acha_tc, acha_pc,
        nwp_pix_ozone, nwp_pix_p_sfc,
        nwp_pix_t_prof, nwp_pix_z_prof, nwp_p_std, nwp_pix_tpw_prof,
        nwp_pix_sfc_level, nwp_pix_tropo_level, nwp_pix_inversion_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof, rtm_sfc_level, rtm_tropo_level, rtm_inversion_level,
        rtm_ch_trans_atm_profile, rtm_ch_rad_atm_profile,
        ch_rad_toa_clear,
        geo_sol_zen, geo_sat_zen,
        ch_ref_toa, ch_sfc_ref_white_sky, ch_rad_toa, ch_sfc_ems,
        solar_rtm_tau_h2o_coef,
        sfc_snow, sfc_land_mask, cld_type, cld_mask_cld_mask, geo_rel_azi,
        sun_earth_distance, solar_ch07_nu,
        rtm_n_levels, nwp_n_levels,
        ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
        phase_cld_ems, phase_cld_trn_ems,
):
    tau_dcomp = np.full(image_shape, nan, 'f4')
    reff_dcomp = np.full(image_shape, nan, 'f4')

    if np.sum((geo_sol_zen < 75.0) & (geo_sol_zen >= 0.0) & (geo_sat_zen < 75.0)) < 1:
        return tau_dcomp, reff_dcomp

    # - compute dcomp related rtm
    (
        rtm_trans_ir_ac,
        rtm_trans_ir_ac_nadir,
        rtm_tpw_ac,
        rtm_sfc_nwp,  # 压强
        rtm_rad_clear_sky_toc_ch07,
        rtm_rad_clear_sky_toa_ch07,
        rtm_ozone_path
    ) = perform_rtm_dcomp(
        bad_pixel_mask,
        acha_tc,
        nwp_pix_ozone, nwp_pix_p_sfc,
        nwp_pix_t_prof, nwp_pix_z_prof, nwp_p_std, nwp_pix_tpw_prof,
        nwp_pix_sfc_level, nwp_pix_tropo_level, nwp_pix_inversion_level,
        rtm_p_std, rtm_t_prof,
        rtm_z_prof, rtm_sfc_level, rtm_tropo_level, rtm_inversion_level,
        rtm_ch_trans_atm_profile, rtm_ch_rad_atm_profile,
        ch_rad_toa_clear,
        geo_sat_zen,
        rtm_n_levels,
        nwp_n_levels
    )

    # 移到外面
    ch_sfc_ref_white_sky[7] = np.empty(image_shape, 'f4')
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            ch_sfc_ref_white_sky[7][line_idx, elem_idx] = 100.0 * (1.0 - ch_sfc_ems[7][line_idx, elem_idx])

    input_sat = geo_sat_zen
    input_sol = geo_sol_zen
    input_azi = geo_rel_azi

    # - cloud products
    input_cloud_press = acha_pc
    input_cloud_temp = acha_tc
    input_cloud_mask = cld_mask_cld_mask

    input_cloud_type = cld_type

    # - flags
    input_is_land = sfc_land_mask == 1
    input_is_valid = bad_pixel_mask != 1

    input_press_sfc = rtm_sfc_nwp
    input_snow_class = sfc_snow

    # - atmospheric contents
    # ozone column in dobson
    input_nwp_ozone = rtm_ozone_path
    # total water vapour above the cloud
    input_tpw_ac = rtm_tpw_ac

    # planck_rad = get_planck_radiance_39um(input_cloud_temp)
    # rad_to_refl = get_rad_refl_factor(input_sol)

    # 移到外面
    planck_rad = np.full(image_shape, nan, 'f4')
    rad_to_refl = np.full(image_shape, nan, 'f4')

    factor = (pi * sun_earth_distance ** 2) / solar_ch07_nu
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            planck_rad[line_idx, elem_idx] = planck_rad_fast(7, input_cloud_temp[line_idx, elem_idx])[0]
            rad_to_refl[line_idx, elem_idx] = factor / cos(radians(input_sol[line_idx, elem_idx]))

    sat_zen_max = 75.0
    sol_zen_max = 82.0

    albedo_ocean = Dict.empty(i1, f4)
    albedo_ocean[3] = 0.03
    albedo_ocean[5] = 0.03
    albedo_ocean[6] = 0.03
    albedo_ocean[7] = 0.03

    air_mass_array = 1.0 / np.cos(np.radians(input_sat)) + 1.0 / np.cos(np.radians(input_sol))

    refl_toc = Dict.empty(i1, f42d_array)  # 7
    alb_sfc = Dict.empty(i1, f42d_array)  # 3,5,6,7
    alb_unc_sfc = Dict.empty(i1, f42d_array)

    trans_total = Dict.empty(i1, f42d_array)  # 3,5,6,7
    trans_unc_total = Dict.empty(i1, f42d_array)

    refl_toc[7] = np.full(image_shape, nan, 'f4')
    for chn_idx in (3, 5, 7):
        alb_sfc[chn_idx] = np.full(image_shape, nan, 'f4')
        alb_unc_sfc[chn_idx] = np.full(image_shape, nan, 'f4')
        trans_total[chn_idx] = np.full(image_shape, nan, 'f4')
        trans_unc_total[chn_idx] = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if not input_is_valid[line_idx, elem_idx]:
                continue
            if input_sat[line_idx, elem_idx] > sat_zen_max:
                continue
            if input_sol[line_idx, elem_idx] > sol_zen_max:
                continue
            if air_mass_array[line_idx, elem_idx] < 2.0:
                continue
            if input_cloud_temp[line_idx, elem_idx] <= 10:
                continue
            if input_cloud_mask[line_idx, elem_idx] not in (em_cloud_mask_cloudy, em_cloud_mask_prob_cloudy):
                continue

            for chn_idx in (3, 5, 7):
                if ch_ref_toa[chn_idx][line_idx, elem_idx] < 0.0:
                    continue

                # - compute transmission
                if chn_idx == 3:
                    ozone_coeff = np.array([-0.000606266, 9.77984e-05, -1.67962e-08], 'f4')
                else:
                    ozone_coeff = np.zeros(3, 'f4')

                (
                    trans_total[chn_idx][line_idx, elem_idx],
                    trans_unc_total[chn_idx][line_idx, elem_idx]
                ) = trans_atm_above_cloud(
                    input_tpw_ac[line_idx, elem_idx],
                    input_nwp_ozone[line_idx, elem_idx],
                    input_press_sfc[line_idx, elem_idx],
                    input_cloud_press[line_idx, elem_idx],
                    air_mass_array[line_idx, elem_idx],
                    solar_rtm_tau_h2o_coef[chn_idx],
                    ozone_coeff,
                    0.044,
                )

                # refl_toc[chn_idx] = refl_toa * trans_total[chn_idx]
                alb_sfc[chn_idx][line_idx, elem_idx] = (ch_sfc_ref_white_sky[chn_idx][line_idx, elem_idx]) / 100.
                alb_sfc[chn_idx][line_idx, elem_idx] = max(alb_sfc[chn_idx][line_idx, elem_idx], albedo_ocean[chn_idx])
                alb_unc_sfc[chn_idx][line_idx, elem_idx] = 0.05

                # if chn_idx == 7:

            trans_total[7][line_idx, elem_idx] = rtm_trans_ir_ac_nadir[line_idx, elem_idx]
            # rad_to_refl_factor = (
            #         (pi * sun_earth_distance ** 2) /
            #         (cos(radians(input_sol[line_idx, elem_idx])) * solar_ch07_nu)
            # )
            refl_toc[7][line_idx, elem_idx] = ch_rad_toa[7][line_idx, elem_idx] * rad_to_refl[line_idx, elem_idx]

            # rad_clear_sky_toc_ch07[line_idx, elem_idx] = input_rad_clear_sky_toc[7][line_idx, elem_idx]
            # rad_clear_sky_toa_ch07[line_idx, elem_idx] = input_rad_clear_sky_toa[7][line_idx, elem_idx]

    dcomp_array_loop(
        input_sat, input_sol, input_is_valid, ch_ref_toa,
        3,
        input_cloud_temp, input_cloud_mask,
        input_cloud_type,
        input_azi,
        input_is_land, input_snow_class,
        # input_chn_alb_sfc_dark_sky,  # todo

        air_mass_array,
        refl_toc,
        alb_sfc,
        alb_unc_sfc,
        trans_total,
        trans_unc_total,
        # input_rad_clear_sky_toc[7],
        # input_rad_clear_sky_toa[7],
        rtm_rad_clear_sky_toc_ch07,
        rtm_rad_clear_sky_toa_ch07,

        planck_rad,
        rad_to_refl,

        ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
        phase_cld_ems, phase_cld_trn_ems,

        tau_dcomp,
        reff_dcomp,
    )

    return tau_dcomp, reff_dcomp
