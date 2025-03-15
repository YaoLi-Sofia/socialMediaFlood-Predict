from math import sqrt, cos, radians, nan

import numpy as np
from numba import njit

from .dcomp_forward_mod import thick_cloud_cps
from .dcomp_lut_mod import (
    dims_sat_zen,
    dims_sol_zen,
    dims_rel_azi,
    dims_cod,
    dims_cps,
    dims_n_sat_zen,
    dims_n_sol_zen,
    dims_n_rel_azi,
    dims_n_cod,
    dims_n_cps,
)
from .dcomp_lut_mod import (
    lut_get_cld_refl,
    lut_get_cld_trn,
    lut_get_cld_sph_alb,
    lut_get_cld_ems,
    lut_get_cld_trn_ems,
)
from .dcomp_math_tools_mod import (
    dcomp_interpolation_weight, dcomp_interpolation_weight2
)


@njit(nogil=True, error_model='numpy', boundscheck=True)
def dcomp_algorithm(
        obs_vec,
        obs_u,
        alb_sfc,
        alb_sfc_u,
        state_apr,
        air_trans_ac,
        sol_zen,
        sat_zen,
        rel_azi,
        cld_phase,
        rad_abv_cld,
        rad_clear_toc,
        dcomp_mode,
        planck_rad,
        rad_to_refl,
        ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
        phase_cld_ems, phase_cld_trn_ems,
):
    output_str_cod = nan
    output_str_cps = nan

    # -- observation
    pxl_sol_zen = sol_zen
    pxl_sat_zen = sat_zen
    pxl_rel_azi = rel_azi
    pxl_is_water_phase = cld_phase

    if dcomp_mode == 1:
        channels = (3, 5)
    elif dcomp_mode == 2:
        channels = (3, 6)
    elif dcomp_mode == 3:
        channels = (3, 7)
    elif dcomp_mode == 4:
        channels = (5, 7)
    else:
        raise ValueError('this mode is not set stop')

    s_a = np.zeros((2, 2), 'f4')
    # s_a[0, 0] = state_apr[0] ** 2
    s_a[0, 0] = 0.8 ** 2
    s_a[1, 1] = 0.65 ** 2

    try:
        s_a_inv = np.linalg.inv(s_a)
    except Exception:  # np.linalg.LinAlgErrorException:
        # print("s_a matrix is non-invertible")
        # print(s_a)
        s_a_inv = np.zeros_like(s_a)

    # - observation error cov
    obs_crl = 0.7
    s_m = np.zeros((2, 2), 'f4')
    s_m[0, 0] = (max(obs_u[0] * obs_vec[0], 0.01)) ** 2
    s_m[1, 1] = (max(obs_u[1] * obs_vec[1], 0.01)) ** 2
    s_m[0, 1] = s_m[1, 0] = (obs_u[1] * obs_vec[1]) * (obs_u[0] * obs_vec[0]) * obs_crl

    # = forward model components vector
    s_b = np.zeros((5, 5), 'f4')

    s_b[0, 0] = (alb_sfc_u[0]) ** 2
    s_b[1, 1] = (alb_sfc_u[1]) ** 2
    s_b[2, 2] = 1.0
    s_b[3, 3] = 1.0
    s_b[4, 4] = 1.0

    state_vec = state_apr.copy()

    iteration_idx = 0
    conv_test_criteria = 0.08

    pos_sat = dcomp_interpolation_weight2(dims_n_sat_zen, pxl_sat_zen, dims_sat_zen)
    pos_sol = dcomp_interpolation_weight2(dims_n_sol_zen, pxl_sol_zen, dims_sol_zen)
    pos_azi = dcomp_interpolation_weight2(dims_n_rel_azi, pxl_rel_azi, dims_rel_azi)

    if pxl_is_water_phase:
        phase_num = 0
    else:
        phase_num = 1

    air_mass_two_way = (1.0 / cos(radians(pxl_sol_zen)) + 1.0 / cos(radians(pxl_sat_zen)))

    while True:
        iteration_idx += 1

        if iteration_idx > 10:
            conv_test_criteria = 0.5
        if dcomp_mode == 4 and iteration_idx > 10:
            conv_test_criteria = 1.0

        fm_vec = np.empty(2, 'f4')
        kernel = np.empty((2, 2), 'f4')
        cld_trn_sol = np.empty(2, 'f4')
        cld_trn_sat = np.empty(2, 'f4')
        cld_sph_alb = np.empty(2, 'f4')

        for i_channel in range(2):

            idx_chn = channels[i_channel]
            idx_phase = phase_num

            cod_log10 = state_vec[0]
            cps_log10 = state_vec[1]

            wgt_cod, pos_cod = dcomp_interpolation_weight(dims_n_cod, cod_log10, dims_cod)
            wgt_cps, pos_cps = dcomp_interpolation_weight(dims_n_cps, cps_log10, dims_cps)

            refl, d_refl_d_cps, d_refl_d_cod = lut_get_cld_refl(
                idx_chn, idx_phase, pos_sol, pos_sat, pos_azi,
                wgt_cod, pos_cod, wgt_cps, pos_cps,
                ch_phase_cld_refl,
            )

            (
                trn_sol, d_trans_sol_d_cps, d_trans_sol_d_cod,
                trn_sat, d_trans_sat_d_cod, d_trans_sat_d_cps,
            ) = lut_get_cld_trn(
                idx_chn, idx_phase, pos_sol, pos_sat,
                wgt_cod, pos_cod, wgt_cps, pos_cps,
                ch_phase_cld_trn,
            )

            alb_sph, d_sph_alb_d_cod, d_sph_alb_d_cps = lut_get_cld_sph_alb(
                idx_chn, idx_phase,
                wgt_cod, pos_cod, wgt_cps, pos_cps,
                ch_phase_cld_sph_alb,
            )

            cld_trn_sol[i_channel] = trn_sol
            cld_trn_sat[i_channel] = trn_sat
            cld_sph_alb[i_channel] = alb_sph

            alb_sfc_term = max(0.0, alb_sfc[i_channel] / (1.0 - alb_sfc[i_channel] * alb_sph))
            fm_vec[i_channel] = refl + alb_sfc_term * trn_sol * trn_sat

            kernel[i_channel, 0] = (
                    d_refl_d_cod +
                    alb_sfc_term * trn_sol * d_trans_sat_d_cod +
                    alb_sfc_term * trn_sat * d_trans_sol_d_cod +
                    ((trn_sol * trn_sat * alb_sfc[i_channel] * alb_sfc[i_channel] * d_sph_alb_d_cod) /
                     ((1 - alb_sfc[i_channel] * alb_sph) ** 2))
            )
            kernel[i_channel, 1] = (
                    d_refl_d_cps +
                    alb_sfc_term * trn_sol * d_trans_sat_d_cps +
                    alb_sfc_term * trn_sat * d_trans_sol_d_cps +
                    ((trn_sol * trn_sat * alb_sfc[i_channel] * alb_sfc[i_channel] * d_sph_alb_d_cps) /
                     ((1 - alb_sfc[i_channel] * alb_sph) ** 2))
            )
            trans_two_way = air_trans_ac[i_channel] ** air_mass_two_way
            fm_vec[i_channel] *= trans_two_way
            kernel[i_channel, 0] *= trans_two_way
            kernel[i_channel, 1] *= trans_two_way

            # - the only ems channel is channel 07

            if channels[i_channel] == 7:
                ems, d_ems_d_cps, d_ems_d_cod = lut_get_cld_ems(
                    idx_phase, pos_sat,
                    wgt_cod, pos_cod, wgt_cps, pos_cps,
                    phase_cld_ems,
                )

                trn_ems, d_trn_ems_d_cps, d_trn_ems_d_cod = lut_get_cld_trn_ems(
                    idx_phase, pos_sat,
                    wgt_cod, pos_cod, wgt_cps, pos_cps,
                    phase_cld_trn_ems,
                )

                air_mass_sat = 1.0 / cos(radians(pxl_sat_zen))
                trans_abv_cld = air_trans_ac[i_channel] ** air_mass_sat

                fm_nir_terr = rad_to_refl * (
                        rad_abv_cld * ems +
                        trans_abv_cld * ems * planck_rad +
                        trn_ems * rad_clear_toc
                )

                kernel_nir_terr_cod = rad_to_refl * (
                        rad_abv_cld * d_ems_d_cod +
                        trans_abv_cld * planck_rad * d_ems_d_cod +
                        rad_clear_toc * d_trn_ems_d_cod
                )

                kernel_nir_terr_cps = rad_to_refl * (
                        rad_abv_cld * d_ems_d_cps +
                        trans_abv_cld * planck_rad * d_ems_d_cps +
                        rad_clear_toc * d_trn_ems_d_cps
                )

                fm_vec[i_channel] = fm_vec[i_channel] + fm_nir_terr
                kernel[i_channel, 0] = kernel[i_channel, 0] + kernel_nir_terr_cod
                kernel[i_channel, 1] = kernel[i_channel, 1] + kernel_nir_terr_cps

        obs_fwd = fm_vec
        cld_trans_sol = cld_trn_sol
        cld_trans_sat = cld_trn_sat

        # - define forward model vector
        # - first dimension : the two channels
        # - 1 sfc albedo vis ; 2 -  sfc albedo ir ; 3- rtm error in vis  4 - rtm error in nir
        # - 5 - terrestrial part

        kernel_b = np.zeros((2, 5), 'f4')
        kernel_b[0, 0] = (cld_trans_sol[0] * cld_trans_sat[0]) / ((1 - cld_sph_alb[0] * alb_sfc[0]) ** 2.)
        kernel_b[0, 1] = 0.0
        kernel_b[0, 2] = 0.04
        kernel_b[0, 3] = 0.0
        kernel_b[0, 4] = 0.0

        kernel_b[1, 0] = 0.0
        kernel_b[1, 1] = (cld_trans_sol[1] * cld_trans_sat[1]) / ((1 - cld_sph_alb[1] * alb_sfc[1]) ** 2.)
        kernel_b[1, 2] = 0.0
        kernel_b[1, 3] = 0.02
        kernel_b[1, 4] = 0.05 * obs_vec[1]

        # - calculate observation error covariance
        s_y = s_m + (kernel_b @ (s_b @ kernel_b.T))
        # s_y_inv, error_flag = find_inv(s_y, 2)
        try:
            s_y_inv = np.linalg.inv(s_y)
        except Exception:  # np.linalg.LinAlgErrorException:
            # print("s_y matrix is non-invertible")
            # print(kernel_b, cld_trans_sol, cld_trans_sat, cld_sph_alb, alb_sfc)
            s_y_inv = np.zeros_like(s_y)

        # --compute sx error covariance of solution x
        s_x_inv = s_a_inv + (kernel.T @ (s_y_inv @ kernel))
        # s_x, error_flag = find_inv(s_x_inv, 2)
        try:
            s_x = np.linalg.inv(s_x_inv)
        except Exception:  # np.linalg.LinAlgErrorException:
            # print("s_x_inv matrix is non-invertible")
            # print(s_x_inv)
            s_x = np.zeros_like(s_x_inv)

        delta_x = s_x @ ((kernel.T @ (s_y_inv @ (obs_vec - obs_fwd))) + (s_a_inv @ (state_apr - state_vec)))

        # - check for convergence
        conv_test = abs(np.sum(delta_x * (s_x_inv @ delta_x)))

        # - control step size
        max_step_size = 0.5
        delta_dstnc = sqrt(np.sum(delta_x ** 2))
        if np.amax(np.abs(delta_x)) > max_step_size:
            delta_x *= max_step_size / delta_dstnc

        state_vec += delta_x

        if conv_test < conv_test_criteria:

            output_str_cod = 10 ** state_vec[0]
            state_vec[1] = min(state_vec[1], 2.2)
            output_str_cps = 10 ** state_vec[1]

            if state_vec[0] > 2.2:
                state_vec[1] = thick_cloud_cps(
                    obs_vec[1], channels[1], pxl_is_water_phase,
                    pos_sol, pos_sat, pos_azi,
                    planck_rad, rad_to_refl,
                    ch_phase_cld_refl, phase_cld_ems,
                    dcomp_mode
                )

                state_vec[1] = min(state_vec[1], 2.2)
                output_str_cod = 10 ** 2.2
                output_str_cps = 10 ** state_vec[1]

            break

        if state_vec[0] > 2.0 and iteration_idx > 6:
            state_vec[1] = thick_cloud_cps(
                obs_vec[1], channels[1], pxl_is_water_phase,
                pos_sol, pos_sat, pos_azi,
                planck_rad, rad_to_refl,
                ch_phase_cld_refl, phase_cld_ems,
                dcomp_mode
            )
            state_vec[1] = min(state_vec[1], 2.2)
            output_str_cod = 10 ** 2.2
            output_str_cps = 10 ** state_vec[1]
            break

        if iteration_idx > 20:
            break

    return output_str_cod, output_str_cps
