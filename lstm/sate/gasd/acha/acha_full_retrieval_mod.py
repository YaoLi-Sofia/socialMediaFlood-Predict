from math import log, nan

import numpy as np
from numba import njit

from constants import (
    sym_yes, sym_opaque_ice_type, sym_cirrus_type, sym_overlap_type
)
from .acha_num_mod import knowing_t_compute_p_z_bottom_up, knowing_z_compute_t_p, optimal_estimation
from .acha_parameters import iter_idx_max, delta_x_max, min_allowable_tc, ice_extinction_tuning_factor
from .acha_rtm_mod import (
    bt_fm, btd_fm,
    determine_acha_extinction, determine_acha_ice_extinction,
    compute_sy_based_on_clear_sky_covariance, compute_clear_sky_terms
)

beta_110um_120um_coef_water = np.array([
    1.0, 1.0, 0.0, 0.0
], 'f4')
beta_110um_120um_coef_ice = np.array([
    1.0, 1.0, 0.0, 0.0
], 'f4')


# ----------------------------------------------------------------------------------
# the pixel-level acha retrieval subroutine
# ----------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def full_acha_retrieval(
        y, y_variance, x_ap, sa_inv,
        convergence_criteria,
        t_sfc_est, t_tropo, z_tropo, p_tropo, cloud_type, cos_zen,
        beta_110um_133um_coef_water,
        beta_110um_133um_coef_ice,
        ems_sfc_110um,
        ems_sfc_120um,
        ems_sfc_133um,
        sfc_type_forward_model,
        acha_rtm_nwp_tropo_level, acha_rtm_nwp_sfc_level,
        acha_rtm_nwp_p_prof, acha_rtm_nwp_z_prof, acha_rtm_nwp_t_prof,
        acha_rtm_nwp_atm_rad_prof_110um, acha_rtm_nwp_atm_trans_prof_110um,
        acha_rtm_nwp_atm_rad_prof_120um, acha_rtm_nwp_atm_trans_prof_120um,
        acha_rtm_nwp_atm_rad_prof_133um, acha_rtm_nwp_atm_trans_prof_133um,
        num_param,
):
    # ----------------------------------------------------------
    iter_idx = 0
    delta_x_prev = np.full(num_param, nan, 'f4')

    # ---- assign x to the first guess
    x = x_ap.copy()

    while True:
        iter_idx += 1
        # ---------------------------------------------------------------------
        # estimate clear-sky radiative transfer terms used in forward model
        # ---------------------------------------------------------------------
        tc_temp = x[0]
        ec_temp = x[1]
        beta_temp = x[2]
        ts_temp = t_sfc_est

        pc_temp, zc_temp = knowing_t_compute_p_z_bottom_up(
            cloud_type, tc_temp, t_tropo, z_tropo, p_tropo,
            acha_rtm_nwp_tropo_level, acha_rtm_nwp_sfc_level,
            acha_rtm_nwp_p_prof, acha_rtm_nwp_z_prof, acha_rtm_nwp_t_prof
        )

        ps_temp, zs_temp = knowing_t_compute_p_z_bottom_up(
            cloud_type, ts_temp, t_tropo, z_tropo, p_tropo,
            acha_rtm_nwp_tropo_level, acha_rtm_nwp_sfc_level,
            acha_rtm_nwp_p_prof, acha_rtm_nwp_z_prof, acha_rtm_nwp_t_prof
        )

        (
            rad_ac_110um, trans_ac_110um, rad_clear_110um,
            rad_ac_120um, trans_ac_120um, rad_clear_120um,
            rad_ac_133um, trans_ac_133um, rad_clear_133um
        ) = compute_clear_sky_terms(
            zc_temp, zs_temp, ts_temp, acha_rtm_nwp_z_prof,  # hgt_prof,
            acha_rtm_nwp_atm_rad_prof_110um, acha_rtm_nwp_atm_trans_prof_110um,
            acha_rtm_nwp_atm_rad_prof_120um, acha_rtm_nwp_atm_trans_prof_120um,
            acha_rtm_nwp_atm_rad_prof_133um, acha_rtm_nwp_atm_trans_prof_133um,
            ems_sfc_110um, ems_sfc_120um, ems_sfc_133um,
        )

        # --------------------------------------------------
        # determine slope of planck emission through cloud
        # --------------------------------------------------

        cloud_opd = -1.0 * log(1.0 - ec_temp)
        cloud_opd = max(0.01, min(10.0, cloud_opd))
        cloud_opd *= cos_zen

        # --- andy heidinger's routine replaces ice values but not water
        if cloud_type in (sym_cirrus_type, sym_overlap_type, sym_opaque_ice_type):
            cloud_extinction = determine_acha_ice_extinction(tc_temp, ec_temp, beta_temp)
            cloud_extinction *= ice_extinction_tuning_factor
        else:
            # --- yue li's routine does ice as f(t) and water
            cloud_extinction = determine_acha_extinction(cloud_type, tc_temp)

        zc_thick = 1000.0 * cloud_opd / cloud_extinction
        zc_base = zc_temp - zc_thick
        zc_base = max(zc_base, zs_temp)  # constrain to be above surface
        zc_base = max(zc_base, 100.0)  # constrain to be positive (greater than 100 m)

        r4_dummy, tc_base = knowing_z_compute_t_p(
            acha_rtm_nwp_z_prof, acha_rtm_nwp_p_prof, acha_rtm_nwp_t_prof, zc_base
        )
        # --------------------------------------------------
        # call forward models
        # --------------------------------------------------

        # --- at this point, if goes-17 mitigation is using 10.4 um data
        # --- the following are switched from 11 um to 104 um data:
        # --- input_chan_idx_104um,

        f, k, ems_vector = compute_forward_model_and_kernel(
            x,
            rad_clear_110um, rad_ac_110um, trans_ac_110um,
            rad_clear_120um, rad_ac_120um, trans_ac_120um,
            rad_clear_133um, rad_ac_133um, trans_ac_133um,
            beta_110um_133um_coef_water,
            beta_110um_133um_coef_ice,
            tc_base,
            num_param,
        )

        # --------------------------------------------------
        # compute the sy covariance matrix
        # --------------------------------------------------
        sy = compute_sy_based_on_clear_sky_covariance(
            sfc_type_forward_model,
            ems_vector,
            y_variance,
        )

        # --------------------------------------------------
        # call oe routine to advance the iteration
        # --------------------------------------------------
        delta_x, converged_flag, fail_flag = optimal_estimation(
            iter_idx, iter_idx_max,
            convergence_criteria, delta_x_max,
            y, f, x, x_ap, k, sy, sa_inv,
            delta_x_prev,
            num_param,
        )
        # todo 需不需要copy
        delta_x_prev = delta_x

        # --- check for a failed iteration
        if fail_flag == sym_yes:
            break

        # --------------------------------------------------------
        # break retrieval loop if converged
        # --------------------------------------------------------
        if converged_flag == sym_yes:
            break

        # ---------------------------------------------------------
        # update retrieved output_vector
        # ---------------------------------------------------------
        x += delta_x

        # -------------------------------------------------------
        # constrain to reasonable values
        # -------------------------------------------------------
        x[0] = max(min_allowable_tc, min(t_sfc_est + 5, x[0]))
        x[1] = max(0.0, min(x[1], 1.0))
        x[2] = max(0.8, min(x[2], 1.8))

    return x, fail_flag


# ---------------------------------------------------------------------
# --- compute the forward model estimate (f) and its kernel (df/dx)
# ---------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_forward_model_and_kernel(
        x,
        rad_clear_110um, rad_ac_110um, trans_ac_110um,
        rad_clear_120um, rad_ac_120um, trans_ac_120um,
        rad_clear_133um, rad_ac_133um, trans_ac_133um,
        beta_110um_133um_coef_water,
        beta_110um_133um_coef_ice,
        tc_base,
        num_param,
):
    f = np.empty(3, 'f4')
    k = np.empty((3, num_param), 'f4')
    ems_vector = np.empty(3, 'f4')
    # ---  for notational convenience, rename elements of x to local variables

    tc = x[0]
    ems_110um = min(x[1], 0.999999)  # values must be below unity
    beta_110um_120um = x[2]

    alpha = 1.0

    # ----------------------------------------------------------------------------------------------
    # make terms for the kernel matrix
    # ----------------------------------------------------------------------------------------------

    # --- 11 um
    f_t_110, dt_110_dtc, dt_110_dec, dt_110_d_beta = bt_fm(
        14, tc, ems_110um, tc_base,
        rad_ac_110um, trans_ac_110um, rad_clear_110um,
    )

    # --- 11 - 12 um
    f_btd_110_120, d_btd_110_120_dtc, d_btd_110_120_dec, d_btd_110_120_d_beta, ems_120um = btd_fm(
        15,
        beta_110um_120um_coef_water,
        beta_110um_120um_coef_ice,
        # np.array([1.0, 1.0, 0.0, 0.0], 'f4'),
        # np.array([1.0, 1.0, 0.0, 0.0], 'f4'),
        tc, ems_110um, beta_110um_120um, tc_base, alpha,
        f_t_110, dt_110_dtc, dt_110_dec,
        rad_ac_120um, trans_ac_120um, rad_clear_120um
    )

    # --- 11 - 133 um
    f_btd_110_133, d_btd_110_133_dtc, d_btd_110_133_dec, d_btd_110_133_d_beta, ems_133um = btd_fm(
        16,
        beta_110um_133um_coef_water,
        beta_110um_133um_coef_ice,
        tc, ems_110um, beta_110um_120um, tc_base, alpha,
        f_t_110, dt_110_dtc, dt_110_dec,
        rad_ac_133um, trans_ac_133um, rad_clear_133um,
    )

    # ----------------------------------------------------------------------------------------------
    # fill in the kernel matrix
    # ----------------------------------------------------------------------------------------------
    f[0] = f_t_110
    k[0, 0] = dt_110_dtc
    k[0, 1] = dt_110_dec
    k[0, 2] = dt_110_d_beta

    f[1] = f_btd_110_120
    f[2] = f_btd_110_133
    k[1, 0] = d_btd_110_120_dtc
    k[1, 1] = d_btd_110_120_dec
    k[1, 2] = d_btd_110_120_d_beta

    k[2, 0] = d_btd_110_133_dtc
    k[2, 1] = d_btd_110_133_dec
    k[2, 2] = d_btd_110_133_d_beta

    ems_vector[0] = ems_110um
    ems_vector[1] = ems_120um
    ems_vector[2] = ems_133um

    # todo
    assert not np.any(np.isnan(ems_vector))

    return f, k, ems_vector
