from math import exp, log, isnan

import numpy as np
from numba import njit

from constants import (
    sym_opaque_ice_type,
    sym_cirrus_type,
    sym_overlap_type,
    sym_overshooting_type
)
from numerical_routines import locate
from planck import planck_rad_fast, planck_temp_fast
from .acha_ice_cloud_microphysical_model_ahi_110um import (
    re_beta_110um_coef_ice,
    qe_006um_coef_ice,
    qe_110um_coef_ice,
)
from .acha_microphysical_module import (
    beta_degree_ice,
    beta_degree_water,
)
from .acha_parameters import (
    t110um_cal_uncer,
    t110um_120um_cal_uncer,
    t110um_133um_cal_uncer,

    water_extinction,

    ice_extinction1,
    ice_extinction2,
    ice_extinction3,
    ice_extinction4,
    ice_extinction5,

    cirrus_extinction1,
    cirrus_extinction2,
    cirrus_extinction3,
    cirrus_extinction4,
    cirrus_extinction5,
)

# ------------------------------------------------------
# --- surface = all
# ------------------------------------------------------
bt_110um_bt_110um_covar_all = 21.901
bt_110um_btd_110um_120um_covar_all = 0.501
bt_110um_btd_110um_133um_covar_all = 12.691
btd_110um_120um_btd_110um_120um_covar_all = 0.210
btd_110um_120um_btd_110um_133um_covar_all = 0.544
btd_110um_133um_btd_110um_120um_covar_all = 0.544
btd_110um_133um_btd_110um_133um_covar_all = 8.371
# ------------------------------------------------------
# --- surface = antarctic
# ------------------------------------------------------
bt_110um_bt_110um_covar_antarctic = 37.244
bt_110um_btd_110um_120um_covar_antarctic = 0.737
bt_110um_btd_110um_133um_covar_antarctic = 15.142
btd_110um_120um_btd_110um_120um_covar_antarctic = 0.135
btd_110um_120um_btd_110um_133um_covar_antarctic = 0.465
btd_110um_133um_btd_110um_120um_covar_antarctic = 0.465
btd_110um_133um_btd_110um_133um_covar_antarctic = 6.975
# ------------------------------------------------------
# --- surface = arctic
# ------------------------------------------------------
bt_110um_bt_110um_covar_arctic = 10.524
bt_110um_btd_110um_120um_covar_arctic = 0.115
bt_110um_btd_110um_133um_covar_arctic = 6.166
btd_110um_120um_btd_110um_120um_covar_arctic = 0.037
btd_110um_120um_btd_110um_133um_covar_arctic = 0.082
btd_110um_133um_btd_110um_120um_covar_arctic = 0.082
btd_110um_133um_btd_110um_133um_covar_arctic = 3.840
# ------------------------------------------------------
# --- surface = desert
# ------------------------------------------------------
bt_110um_bt_110um_covar_desert = 40.916
bt_110um_btd_110um_120um_covar_desert = 1.479
bt_110um_btd_110um_133um_covar_desert = 28.101
btd_110um_120um_btd_110um_120um_covar_desert = 0.524
btd_110um_120um_btd_110um_133um_covar_desert = 1.486
btd_110um_133um_btd_110um_120um_covar_desert = 1.486
btd_110um_133um_btd_110um_133um_covar_desert = 20.838
# ------------------------------------------------------
# --- surface = land
# ------------------------------------------------------
bt_110um_bt_110um_covar_land = 27.168
bt_110um_btd_110um_120um_covar_land = 1.103
bt_110um_btd_110um_133um_covar_land = 17.087
btd_110um_120um_btd_110um_120um_covar_land = 0.333
btd_110um_120um_btd_110um_133um_covar_land = 1.132
btd_110um_133um_btd_110um_120um_covar_land = 1.132
btd_110um_133um_btd_110um_133um_covar_land = 12.152
# ------------------------------------------------------
# --- surface = snow
# ------------------------------------------------------
bt_110um_bt_110um_covar_snow = 18.151
bt_110um_btd_110um_120um_covar_snow = 0.050
bt_110um_btd_110um_133um_covar_snow = 10.189
btd_110um_120um_btd_110um_120um_covar_snow = 0.086
btd_110um_120um_btd_110um_133um_covar_snow = 0.085
btd_110um_133um_btd_110um_120um_covar_snow = 0.085
btd_110um_133um_btd_110um_133um_covar_snow = 6.174
# ------------------------------------------------------
# --- surface = water
# ------------------------------------------------------
bt_110um_bt_110um_covar_water = 0.873
bt_110um_btd_110um_120um_covar_water = -0.027
bt_110um_btd_110um_133um_covar_water = 0.421
btd_110um_120um_btd_110um_120um_covar_water = 0.066
btd_110um_120um_btd_110um_133um_covar_water = 0.057
btd_110um_133um_btd_110um_120um_covar_water = 0.057
btd_110um_133um_btd_110um_133um_covar_water = 0.402

# -------------------------------------------------------------------------------
# ice extinction
# -------------------------------------------------------------------------------
n_ec_ext = 10
ec_ext_min = 0.0
ec_ext_bin = 0.1
m_ext = 4
tc_ext_offset = 182.5
ice_ext_coef = np.array([
    [1.2316e-02, 2.4031e-03, -4.7896e-05, 4.5502e-07],
    [3.3809e-02, 9.7992e-03, -2.7210e-04, 2.9489e-06],
    [9.6130e-02, 8.8489e-03, -1.6911e-04, 1.8791e-06],
    [1.0245e-01, 1.9004e-02, -4.9367e-04, 4.8926e-06],
    [1.6542e-01, 2.4050e-02, -7.0344e-04, 7.7430e-06],
    [2.5281e-01, 2.0355e-02, -4.5441e-04, 5.0307e-06],
    [2.7558e-01, 4.3940e-02, -1.3112e-03, 1.3254e-05],
    [5.1518e-01, 2.5070e-02, -5.8033e-04, 7.6746e-06],
    [5.4931e-01, 7.4288e-02, -1.9975e-03, 1.8927e-05],
    [2.6006e+00, 4.7639e-02, 7.5674e-04, -6.2741e-06],
])


# -------------------------------------------------------
#  linear in optical depth emission routine
# -------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def linear_in_opd_emission(ems, b_base, b_top):
    opd = -1.0 * log(1.0 - ems)
    opd = max(0.01, min(10.0, opd))
    bd = (b_base - b_top) / opd
    linear_term = exp(-opd) * (1 + opd) - 1
    cloud_emission = ems * b_top - bd * linear_term
    return cloud_emission


# -------------------------------------------------------------------------------------
# forward model for a brightness temperature
#
# channel x refers to the 11 micron channel
# input
#  chan_x = channel number of channel x
#  f_x = forward model estimate of the 11 micron brightness temperature
#  ec_x = cloud emissivity at 11 micron
#
# --------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def bt_fm(
        chan_x, tc, ec_x, tc_base,
        rad_ac_x, trans_ac_x, rad_clear_x,
):
    # --- planck function terms
    bc_x, db_dtc_x = planck_rad_fast(chan_x, tc)
    bc_base_x = planck_rad_fast(chan_x, tc_base)[0]

    trans_x = 1.0 - ec_x

    cloud_emission = linear_in_opd_emission(ec_x, bc_base_x, bc_x)

    rad_x = ec_x * rad_ac_x + trans_ac_x * cloud_emission + trans_x * rad_clear_x

    f, db_dt_x = planck_temp_fast(chan_x, rad_x)

    # --- kernel matrix terms
    alpha_x = rad_ac_x + trans_ac_x * bc_x - rad_clear_x
    df_dtc = (trans_ac_x * ec_x * db_dtc_x) / db_dt_x
    df_dec = alpha_x / db_dt_x
    df_d_beta = 0.0

    return f, df_dtc, df_dec, df_d_beta


# -------------------------------------------------------------------------------------
# forward model for a brightness temperature difference
#
# channel x refers to the 11 micron channel
# input
#  chan_x = channel number of channel x
#  f_x = forward model estimate of the 11 micron brightness temperature
#  ec_x = cloud emissivity at 11 micron
#  beta_x_12 = beta of 11 micron and 12 micron (the reference value)
#
# output
#  f = the forward model estimate of the btd of x - y
#  df_dtc = the derivative of f wrt to cloud temperature
#  df_dec = the derivative of f wrt to cloud emissivity
#  df_d_beta = the derivative of f wrt to cloud beta 
#  df_dts = the derivative of f wrt to surface temperature
#  df_d_alpha = the derivative of f wrt to ice_fraction (alpha)
# -------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def btd_fm(
        chan_y,
        beta_xy_coef_water,
        beta_xy_coef_ice,
        tc, ec_x, beta_x_12, tc_base, alpha,
        f_x, df_x_dtc, df_x_dec,
        rad_ac_y, trans_ac_y, rad_clear_y,
):
    # --- planck function terms
    bc_y, db_dtc_y = planck_rad_fast(chan_y, tc)
    bc_base_y = planck_rad_fast(chan_y, tc_base)[0]

    # --- intermediate terms
    beta_xy, d_beta_xy_d_beta_x_12 = compute_beta_and_derivative(
        beta_degree_water, beta_xy_coef_water, beta_degree_ice, beta_xy_coef_ice,
        alpha, beta_x_12,
    )

    dec_y_dec_x = beta_xy * (1.0 - ec_x) ** (beta_xy - 1.0)
    ec_y = 1.0 - (1.0 - ec_x) ** beta_xy
    trans_y = 1.0 - ec_y
    cloud_emission = linear_in_opd_emission(ec_y, bc_base_y, bc_y)
    rad_y = ec_y * rad_ac_y + trans_ac_y * cloud_emission + trans_y * rad_clear_y

    # --- forward model term
    t_y, db_dt_y = planck_temp_fast(chan_y, rad_y)
    f = f_x - t_y

    # --- kernel matrix terms
    alpha_y = rad_ac_y + trans_ac_y * bc_y - rad_clear_y

    df_dtc = df_x_dtc - (trans_ac_y * ec_y * db_dtc_y) / db_dt_y
    df_dec = df_x_dec - (alpha_y * dec_y_dec_x) / db_dt_y
    df_d_beta = alpha_y * log(1.0 - ec_x) * (1.0 - ec_y) * d_beta_xy_d_beta_x_12 / db_dt_y

    return f, df_dtc, df_dec, df_d_beta, ec_y


# ------------------------------------------------------------------------------
# compute a channel pairs channel beta and derivative
#
# input: beta_xy_coef_water - beta coefficients for water clouds
#        beta_xy_coef_ice - beta coefficients for ice clouds
#        beta_degree_water - degree of the polynomial phase for water
#        beta_degree_ice - degree of the polynomial phase for ice
#        alpha - ice cloud fraction
#        beta_x_12 - the beta value for 11 and 12 micron
#
# output:
#        beta_xy - the beta value for this channel pair
#        d_beta_xy_d_beta_x_12 - the derivative of beta value for this channel 
#                              pair to the beta_x_12
# ------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_beta_and_derivative(
        beta_degree_water,
        beta_xy_coef_water,
        beta_degree_ice,
        beta_xy_coef_ice,
        alpha, beta_x_12,
):
    # ----------------------------------------------------------------------
    # water
    # ----------------------------------------------------------------------
    beta_xy_water = beta_xy_coef_water[0]
    d_beta_xy_d_beta_x_12_water = 0.0

    for i in range(1, beta_degree_water + 1):
        beta_xy_water += beta_xy_coef_water[i] * (beta_x_12 - 1.0) ** i
        d_beta_xy_d_beta_x_12_water += beta_xy_coef_water[i] * i * (beta_x_12 - 1.0) ** (i - 1)

    # ----------------------------------------------------------------------
    # ice
    # ----------------------------------------------------------------------
    beta_xy_ice = beta_xy_coef_ice[0]
    d_beta_xy_d_beta_x_12_ice = 0.0

    for i in range(1, beta_degree_ice + 1):
        beta_xy_ice += beta_xy_coef_ice[i] * (beta_x_12 - 1.0) ** i
        d_beta_xy_d_beta_x_12_ice += beta_xy_coef_ice[i] * i * (beta_x_12 - 1.0) ** (i - 1)

    # ----------------------------------------------------------------------
    # combine
    # ----------------------------------------------------------------------
    beta_xy = (1.0 - alpha) * beta_xy_water + alpha * beta_xy_ice

    d_beta_xy_d_beta_x_12 = (1.0 - alpha) * d_beta_xy_d_beta_x_12_water + alpha * d_beta_xy_d_beta_x_12_ice

    return beta_xy, d_beta_xy_d_beta_x_12


# ---------------------------------------------------------------------------------------------
# compute clear-sky terms needed in forward model
# ---------------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_clear_sky_terms(
        zc, zs, ts, hgt_prof,
        atm_rad_prof_110um, atm_trans_prof_110um,
        atm_rad_prof_120um, atm_trans_prof_120um,
        atm_rad_prof_133um, atm_trans_prof_133um,
        ems_sfc_110um, ems_sfc_120um, ems_sfc_133um,
):
    rad_ac_110um, trans_ac_110um, rad_clear_110um = clear_sky_internal_routine(
        zc, zs, ts, hgt_prof, 14,
        atm_rad_prof_110um, atm_trans_prof_110um,
        ems_sfc_110um
    )

    # --- compute 120um radiative transfer terms
    rad_ac_120um, trans_ac_120um, rad_clear_120um = clear_sky_internal_routine(
        zc, zs, ts, hgt_prof, 15,
        atm_rad_prof_120um, atm_trans_prof_120um,
        ems_sfc_120um
    )

    # --- 13.3um clear radiative transfer terms
    rad_ac_133um, trans_ac_133um, rad_clear_133um = clear_sky_internal_routine(
        zc, zs, ts, hgt_prof, 16,
        atm_rad_prof_133um, atm_trans_prof_133um,
        ems_sfc_133um
    )
    return (
        rad_ac_110um, trans_ac_110um, rad_clear_110um,
        rad_ac_120um, trans_ac_120um, rad_clear_120um,
        rad_ac_133um, trans_ac_133um, rad_clear_133um,
    )


# ------------------------------------------------------------------------------------------
# routine to compute the terms needed in compute_clear_sky_terms
# ------------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def clear_sky_internal_routine(
        zc, zs, ts, hgt_prof, chan_idx,
        atm_rad_prof, atm_trans_prof, ems_sfc,
):
    rad_ac = generic_profile_interpolation(zc, hgt_prof, atm_rad_prof)
    trans_ac = generic_profile_interpolation(zc, hgt_prof, atm_trans_prof)
    rad_atm = generic_profile_interpolation(zs, hgt_prof, atm_rad_prof)
    trans_atm = generic_profile_interpolation(zs, hgt_prof, atm_trans_prof)
    bs = planck_rad_fast(chan_idx, ts)[0]
    rad_clear = rad_atm + trans_atm * ems_sfc * bs

    return rad_ac, trans_ac, rad_clear


# -----------------------------------------------------------------
# interpolate within profiles knowing z to determine above cloud
# radiative terms used in forward model
# -----------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def generic_profile_interpolation(x_value, x_profile, y_profile):
    n_levels = x_profile.size

    # --- interpolate pressure profile
    lev_idx = locate(x_profile, n_levels, x_value)
    lev_idx = max(0, min(n_levels - 2, lev_idx))

    dx = x_profile[lev_idx + 1] - x_profile[lev_idx]

    # --- perform interpolation
    if dx != 0.0:
        y_value = y_profile[lev_idx] + (x_value - x_profile[lev_idx]) * (
                y_profile[lev_idx + 1] - y_profile[lev_idx]) / dx
    else:
        y_value = y_profile[lev_idx]

    return y_value


# ----------------------------------------------------------------------
#  andy heidinger's extinction routine  (km^-1)
# input:
#        tc = cloud temperature
#        ec = cloud emissivity
# output:
#        ice_cloud_extinction in 1/km units
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def determine_acha_ice_extinction(tc, ec, beta):
    log10_reff = 1.0
    # todo
    assert not isnan(beta)
    log10_reff = (
            re_beta_110um_coef_ice[0] +
            re_beta_110um_coef_ice[1] * (beta - 1.0) +
            re_beta_110um_coef_ice[2] * (beta - 1.0) ** 2 +
            re_beta_110um_coef_ice[3] * (beta - 1.0) ** 3
    )
    log10_reff = 1.0 / log10_reff  # fit is in 1/log10_reff
    log10_reff = min(2.0, max(log10_reff, 0.6))  # constrain to 4 to 100 microns
    qe_006um = (
            qe_006um_coef_ice[0] +
            qe_006um_coef_ice[1] * log10_reff +
            qe_006um_coef_ice[2] * log10_reff ** 2
    )
    qe_110um = (
            qe_110um_coef_ice[0] +
            qe_110um_coef_ice[1] * log10_reff +
            qe_110um_coef_ice[2] * log10_reff ** 2
    )

    # --- determine which emissivity to use
    iec = int((ec - ec_ext_min) / ec_ext_bin)
    iec = max(0, iec)
    iec = min(n_ec_ext - 1, iec)

    # --- compute ice cloud extinction (1/km)
    ice_cloud_extinction = 0.0
    xtc = max(0.0, tc - tc_ext_offset)
    xtc = min(xtc, 90.0)
    for i in range(m_ext):
        ice_cloud_extinction += ice_ext_coef[iec, i] * (xtc ** i)

    # --- convert from 532nm (0.65 um) to 11 um
    ice_cloud_extinction *= qe_110um / qe_006um

    # --- limit
    ice_cloud_extinction = min(10.0, max(0.01, ice_cloud_extinction))

    return ice_cloud_extinction


# ----------------------------------------------------------------------
#  yue li's extinction routine
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def determine_acha_extinction(cloud_type, tc):
    if cloud_type in (sym_opaque_ice_type, sym_overshooting_type):
        if tc < 200:
            cloud_extinction = ice_extinction1
        elif tc < 220:
            cloud_extinction = ice_extinction2
        elif tc < 240:
            cloud_extinction = ice_extinction3
        elif tc < 260:
            cloud_extinction = ice_extinction4
        else:
            cloud_extinction = ice_extinction5
    elif cloud_type in (sym_cirrus_type, sym_overlap_type):
        if tc < 200:
            cloud_extinction = cirrus_extinction1
        elif tc < 220:
            cloud_extinction = cirrus_extinction2
        elif tc < 240:
            cloud_extinction = cirrus_extinction3
        elif tc < 260:
            cloud_extinction = cirrus_extinction4
        else:
            cloud_extinction = cirrus_extinction5
    else:
        cloud_extinction = water_extinction

    return cloud_extinction


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_sy_based_on_clear_sky_covariance(
        sfc_type_forward_model,
        ems_vector,
        y_variance,
):
    sub_pixel_uncer = np.empty(3, 'f4')

    ems_110um = min(1.0, max(0.0, ems_vector[0]))

    trans2 = (1.0 - ems_110um) ** 2  # cloud transmission squared
    trans2 = max(trans2, 0.25)  # do not let this go to zero

    # ----------------------------------------------------------------
    # --- modify y_variance to represent a sub-pixel uncertainty
    # --- assume that all of standard deviation is due to sub-pixel
    # --- heterogeneity and that this is a good estimate of the
    # --- forward model error due to sub-pixel heterogeneity
    # ----------------------------------------------------------------
    for i in range(3):
        sub_pixel_uncer[i] = y_variance[i]

    # # ---- compute the sy matrix
    # for i in range(3):
    #     for j in range(3):
    #         sy[i, j] = trans2 * btd_covar(chan_idx_y[i], chan_idx_y[j])
    #         if i == 0:
    #             sy[i, j] = trans2 * bt_covar(chan_idx_y[j])
    #         if j == 0:
    #             sy[i, j] = trans2 * bt_covar(chan_idx_y[i])

    # --- water
    if sfc_type_forward_model == 0:
        sy = np.array([
            [bt_110um_bt_110um_covar_water, bt_110um_btd_110um_120um_covar_water,
             bt_110um_btd_110um_133um_covar_water],
            [bt_110um_btd_110um_120um_covar_water, btd_110um_120um_btd_110um_120um_covar_water,
             btd_110um_120um_btd_110um_133um_covar_water],
            [bt_110um_btd_110um_133um_covar_water, btd_110um_133um_btd_110um_120um_covar_water,
             btd_110um_133um_btd_110um_133um_covar_water],
        ], 'f4')
    # --- land
    elif sfc_type_forward_model == 1:
        sy = np.array([
            [bt_110um_bt_110um_covar_land, bt_110um_btd_110um_120um_covar_land,
             bt_110um_btd_110um_133um_covar_land],
            [bt_110um_btd_110um_120um_covar_land, btd_110um_120um_btd_110um_120um_covar_land,
             btd_110um_120um_btd_110um_133um_covar_land],
            [bt_110um_btd_110um_133um_covar_land, btd_110um_133um_btd_110um_120um_covar_land,
             btd_110um_133um_btd_110um_133um_covar_land],
        ], 'f4')
    # --- snow
    elif sfc_type_forward_model == 2:
        sy = np.array([
            [bt_110um_bt_110um_covar_snow, bt_110um_btd_110um_120um_covar_snow,
             bt_110um_btd_110um_133um_covar_snow],
            [bt_110um_btd_110um_120um_covar_snow, btd_110um_120um_btd_110um_120um_covar_snow,
             btd_110um_120um_btd_110um_133um_covar_snow],
            [bt_110um_btd_110um_133um_covar_snow, btd_110um_133um_btd_110um_120um_covar_snow,
             btd_110um_133um_btd_110um_133um_covar_snow],
        ], 'f4')
    # --- desert
    elif sfc_type_forward_model == 3:
        sy = np.array([
            [bt_110um_bt_110um_covar_desert, bt_110um_btd_110um_120um_covar_desert,
             bt_110um_btd_110um_133um_covar_desert],
            [bt_110um_btd_110um_120um_covar_desert, btd_110um_120um_btd_110um_120um_covar_desert,
             btd_110um_120um_btd_110um_133um_covar_desert],
            [bt_110um_btd_110um_133um_covar_desert, btd_110um_133um_btd_110um_120um_covar_desert,
             btd_110um_133um_btd_110um_133um_covar_desert],
        ], 'f4')
    # --- arctic
    elif sfc_type_forward_model == 4:
        sy = np.array([
            [bt_110um_bt_110um_covar_arctic, bt_110um_btd_110um_120um_covar_arctic,
             bt_110um_btd_110um_133um_covar_arctic],
            [bt_110um_btd_110um_120um_covar_arctic, btd_110um_120um_btd_110um_120um_covar_arctic,
             btd_110um_120um_btd_110um_133um_covar_arctic],
            [bt_110um_btd_110um_133um_covar_arctic, btd_110um_133um_btd_110um_120um_covar_arctic,
             btd_110um_133um_btd_110um_133um_covar_arctic],
        ], 'f4')
    # --- antarctic
    elif sfc_type_forward_model == 5:
        sy = np.array([
            [bt_110um_bt_110um_covar_antarctic, bt_110um_btd_110um_120um_covar_antarctic,
             bt_110um_btd_110um_133um_covar_antarctic],
            [bt_110um_btd_110um_120um_covar_antarctic, btd_110um_120um_btd_110um_120um_covar_antarctic,
             btd_110um_120um_btd_110um_133um_covar_antarctic],
            [bt_110um_btd_110um_133um_covar_antarctic, btd_110um_133um_btd_110um_120um_covar_antarctic,
             btd_110um_133um_btd_110um_133um_covar_antarctic],
        ], 'f4')
    else:
        sy = np.array([
            [bt_110um_bt_110um_covar_all, bt_110um_btd_110um_120um_covar_all,
             bt_110um_btd_110um_133um_covar_all],
            [bt_110um_btd_110um_120um_covar_all, btd_110um_120um_btd_110um_120um_covar_all,
             btd_110um_120um_btd_110um_133um_covar_all],
            [bt_110um_btd_110um_133um_covar_all, btd_110um_133um_btd_110um_120um_covar_all,
             btd_110um_133um_btd_110um_133um_covar_all],
        ], 'f4')
    sy *= trans2

    # bt_covar = 0.00
    # btd_covar = 0.00
    #
    # cal_uncer[31] = t110um_cal_uncer  # note, not a btd
    # cal_uncer[32] = t110um_120um_cal_uncer
    # cal_uncer[33] = t110um_133um_cal_uncer
    #
    # # --- additional terms to sy for the cloud error [bt and btd]
    cloud_btd_uncer = 1.0  # 2.0
    cloud_bt_uncer = 4.0  # 5.0

    # # -- add in terms for diagonal elements to sy
    # for i in range(3):
    #     sy[i, i] = sy[i, i] + cal_uncer[chan_idx_y[i]] ** 2 + sub_pixel_uncer[i]

    sy[0, 0] += t110um_cal_uncer ** 2 + sub_pixel_uncer[0]
    sy[1, 1] += t110um_120um_cal_uncer ** 2 + sub_pixel_uncer[1]
    sy[2, 2] += t110um_133um_cal_uncer ** 2 + sub_pixel_uncer[2]

    # # -- add in terms for diagonal elements for cloud btd error
    # sy[0, 0] += (ems_vector[0] * cloud_bt_uncer) ** 2
    # for i in range(1, 3):
    #     sy[i, i] = sy[i, i] + (ems_vector[i] * cloud_btd_uncer[chan_idx_y[i]]) ** 2

    sy[0, 0] += (ems_vector[0] * cloud_bt_uncer) ** 2
    sy[1, 1] += (ems_vector[1] * cloud_btd_uncer) ** 2
    sy[2, 2] += (ems_vector[2] * cloud_btd_uncer) ** 2

    return sy
