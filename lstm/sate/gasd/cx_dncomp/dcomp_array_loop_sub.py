from math import log10, nan

import numpy as np
from numba import i1, f4
from numba import njit, prange
from numba.typed import Dict

from public import (
    image_number_of_lines,
    image_number_of_elements,
)
from .dcomp_retrieval_mod import dcomp_algorithm

em_cloud_type_first = 0
em_cloud_type_clear = 0
em_cloud_type_prob_clear = 1
em_cloud_type_fog = 2
em_cloud_type_water = 3
em_cloud_type_supercooled = 4
em_cloud_type_mixed = 5
em_cloud_type_opaque_ice = 6
em_cloud_type_tice = 6
em_cloud_type_cirrus = 7
em_cloud_type_overlap = 8
em_cloud_type_overshooting = 9
em_cloud_type_unknown = 10
em_cloud_type_dust = 11
em_cloud_type_smoke = 12
em_cloud_type_fire = 13
em_cloud_type_last = 13

em_cloud_mask_last = 3
em_cloud_mask_cloudy = 3
em_cloud_mask_prob_cloudy = 2
em_cloud_mask_prob_clear = 1
em_cloud_mask_clear = 0
em_cloud_mask_first = 0


# 波段7
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def dcomp_array_loop(
        input_sat, input_sol, input_is_valid, input_refl,
        input_mode,
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
        rad_clear_sky_toc_ch07,
        rad_clear_sky_toa_ch07,

        planck_rad,
        rad_to_refl,

        ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
        phase_cld_ems, phase_cld_trn_ems,

        output_cod,
        output_cps,
):
    chn_vis_default = 3

    sat_zen_max = 75.0
    sol_zen_max = 82.0

    calib_err = Dict.empty(i1, f4)
    calib_err[3] = 0.03
    calib_err[5] = 0.03
    calib_err[7] = 0.03

    obs_array = (
            input_is_valid &
            (input_sat <= sat_zen_max) &
            (input_sol <= sol_zen_max) &
            (input_refl[3] >= 0.) &
            (air_mass_array >= 2.)
    )

    if input_mode == 3:
        chn_nir_default = 7
        obs_array = obs_array & (input_refl[7] >= 0.)
    else:
        raise ValueError

    obs_and_acha_array = obs_array & (input_cloud_temp > 10)

    cloud_array = (
            obs_and_acha_array &
            ((input_cloud_mask == em_cloud_mask_cloudy) |
             (input_cloud_mask == em_cloud_mask_prob_cloudy))
    )

    water_phase_array = (
            (input_cloud_type == em_cloud_type_fog) |
            (input_cloud_type == em_cloud_type_water) |
            (input_cloud_type == em_cloud_type_supercooled) |
            (input_cloud_type == em_cloud_type_mixed)
    )

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            obs_vec = np.full(2, nan, 'f4')
            obs_unc = np.full(2, nan, 'f4')
            alb_vec = np.full(3, nan, 'f4')
            alb_unc = np.full(2, nan, 'f4')
            trans_vec = np.full(2, nan, 'f4')

            state_apriori = np.empty(2, 'f4')

            if not cloud_array[line_idx, elem_idx]:
                continue

            # - set aliases
            sol_zen = input_sol[line_idx, elem_idx]
            sat_zen = input_sat[line_idx, elem_idx]
            rel_azi = input_azi[line_idx, elem_idx]

            if input_snow_class[line_idx, elem_idx] == 3 and input_refl[5][line_idx, elem_idx] > 0.0:
                chn_vis = 5
                chn_nir = 7
                dcomp_mode = 4
            else:
                chn_vis = chn_vis_default
                chn_nir = chn_nir_default
                dcomp_mode = input_mode

            # - nir
            obs_vec[0] = input_refl[chn_vis][line_idx, elem_idx] / 100.
            obs_unc[0] = trans_unc_total[chn_vis][line_idx, elem_idx] + calib_err[chn_vis]
            alb_vec[0] = alb_sfc[chn_vis][line_idx, elem_idx]
            # todo 虽然填了值却没有用到
            # alb_vec[2] = input_chn_alb_sfc_dark_sky[3][line_idx, elem_idx] / 100.
            # alb_vec[0] = alb_sfc[3]
            alb_unc[0] = 0.05
            trans_vec[0] = trans_total[chn_vis][line_idx, elem_idx]

            if chn_nir == 7:
                obs_vec[1] = refl_toc[chn_nir][line_idx, elem_idx]
                obs_unc[1] = obs_vec[1] * 0.1
            else:
                obs_vec[1] = input_refl[chn_nir][line_idx, elem_idx] / 100.
                obs_unc[1] = max(trans_unc_total[chn_nir][line_idx, elem_idx], 0.01) + calib_err[chn_nir]

            alb_vec[1] = alb_sfc[chn_nir][line_idx, elem_idx]
            alb_unc[1] = 0.05

            trans_vec[1] = trans_total[chn_nir][line_idx, elem_idx]

            # - apriori
            state_apriori[0] = 0.7 * (100. * obs_vec[0]) ** 0.9
            state_apriori[0] = log10(max(0.1, state_apriori[0]))
            if water_phase_array[line_idx, elem_idx]:
                state_apriori[1] = 1.0
            else:
                state_apriori[1] = 1.3

            dcomp_out_cod, dcomp_out_cps = dcomp_algorithm(
                obs_vec,
                obs_unc,
                alb_vec,
                alb_unc,
                state_apriori,
                trans_vec,
                sol_zen,
                sat_zen,
                rel_azi,
                water_phase_array[line_idx, elem_idx],
                rad_clear_sky_toc_ch07[line_idx, elem_idx],
                rad_clear_sky_toa_ch07[line_idx, elem_idx],
                dcomp_mode,
                planck_rad[line_idx, elem_idx],
                rad_to_refl[line_idx, elem_idx],
                ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
                phase_cld_ems, phase_cld_trn_ems,
            )

            output_cod[line_idx, elem_idx] = dcomp_out_cod
            output_cps[line_idx, elem_idx] = dcomp_out_cps

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if obs_array[line_idx, elem_idx] and not cloud_array[line_idx, elem_idx]:
                output_cod[line_idx, elem_idx] = 0.0
                output_cps[line_idx, elem_idx] = nan

    # # compute success_rate
    # output_nr_obs = np.sum(obs_array)
    # output_nr_clouds = np.sum(cloud_array)
    # output_nr_success_cod = np.sum(quality_flag & 0b00000010)
    # output_nr_success_cps = np.sum(quality_flag & 0b00000100)
    #
    # tried = np.sum(quality_flag & 0b00000001)
    #
    # output_success_rate = 0.0
    # if tried > 0:
    #     success = np.sum((~(quality_flag & 0b00000010)) & (~(quality_flag & 0b00000100)))
    #     output_success_rate = success / tried
