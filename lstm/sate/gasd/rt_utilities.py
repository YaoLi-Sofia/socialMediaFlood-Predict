from math import log, pi, nan, isnan

import numpy as np
from numba import njit, prange, i1, f4
from numba.typed import Dict

from calibration_constants import solar_ch07_nu
from constants import sym_yes, sym_land, g, missing_value_int1
from interp import VerticalLinearInterpolator
from numerical_routines import vapor, locate
from pfaast.cx_pfaast_coef import (
    n_l, n_xd, n_xo, n_xc, n_xw,
)
from pfaast.cx_pfaast_constants import n_l, t_std, w_std, o_std
from pfaast.cx_pfaast_tools import conpir, calpir, tau_doc, tau_wtr
from planck import planck_rad_fast, planck_temp_fast
from public import (
    image_shape, image_number_of_lines, image_number_of_elements, thermal_channels,
    p_inversion_min, delta_t_inversion
)
from rtm_common import rtm_n_levels, rtm_p_std, rtm_t_std, rtm_ozmr_std, rtm_wvmr_std
from utils import show_time

f4_3d_array = f4[:, :, :]


@njit(nogil=True, error_model='numpy', boundscheck=True)
def emissivity(radiance_toa, radiance_clear_toa, radiance_cloud_bb_toa):
    ems = nan
    if radiance_cloud_bb_toa != radiance_clear_toa:
        ems = (radiance_toa - radiance_clear_toa) / (radiance_cloud_bb_toa - radiance_clear_toa)
    return ems


@njit(nogil=True, error_model='numpy', boundscheck=True)
def beta_ratio(ems_top, ems_bot):
    beta = nan
    if 0.0 < ems_top < 1.0 and 0.0 < ems_bot < 1.0:
        beta = log(1.0 - ems_top) / log(1.0 - ems_bot)
    return beta


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_nwp_wvmr_profile(press_profile, temp_profile, rh_profile):
    """
    :param press_profile: pressure profile (hPa)
    :param temp_profile: temperature profile [k]
    :param rh_profile: relative humidity profile (%)
    :return: wvmr_profile: water vapor mixing ratio profile (g/kg)
    """
    n_levels = press_profile.size
    wvmr_profile = np.empty(n_levels, 'f4')
    for lev_idx in prange(n_levels):
        es = vapor(temp_profile[lev_idx])
        e = rh_profile[lev_idx] * es / 100.0
        wvmr_profile[lev_idx] = 1000.0 * 0.622 * (e / (press_profile[lev_idx] - e))
    return wvmr_profile


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_tpw_profile_nwp(press_profile, wvmr_profile):
    """
    :param press_profile: pressure profile (hPa)
    :param wvmr_profile: water vapor mixing ratio profile (g/kg)
    :return: tpw_profile: profile of Tpw from level to space (g/m^2)
    """
    n_levels = press_profile.size
    tpw_profile = np.empty(n_levels, 'f4')
    tpw_profile[0] = 0.0
    for lay_idx in range(n_levels - 1):
        w_mean = 0.5 * (wvmr_profile[lay_idx + 1] + wvmr_profile[lay_idx]) / 1000.0
        u_layer = (10.0 / g) * (press_profile[lay_idx + 1] - press_profile[lay_idx]) * w_mean
        tpw_profile[lay_idx + 1] = tpw_profile[lay_idx] + u_layer
    return tpw_profile


# subroutine Name: CONVERT_ATMOS_PROF_NWP_RTM
#
# Description:
# This routine interpolate the NWP profiles to profiles with the
# vertical spacing defined by the RTM model.  It operates on profiles
# stored in this module
#
# INPUTS:
#
# Highest_Level_Rtm_Nwp - highest Rtm Level that is below the highest nwp Level
# Lowest_Level_Rtm_Nwp - lowest Rtm Level that is above the lowest nwp Level
# Sfc_Level_Rtm - lowest Rtm Level above the surface
# P_near_Sfc_Nwp - lowest standard nwp Level above surface pressure
@njit(nogil=True, error_model='numpy', boundscheck=True)
def convert_atmos_prof_nwp_rtm(
        num_nwp_profile_levels,
        nwp_surface_level,
        nwp_surface_elevation,
        nwp_air_temperature,
        nwp_surface_rh,
        nwp_surface_pressure,
        nwp_press_profile,
        nwp_t_profile,
        nwp_z_profile,
        nwp_wvmr_profile,
        nwp_ozmr_profile,
        num_rtm_profile_levels,
        press_profile_rtm,
        t_std_profile_rtm,
        wvmr_std_profile_rtm,
        ozmr_std_profile_rtm
):
    rtm_t_profile = np.empty(num_rtm_profile_levels, 'f4')
    rtm_z_profile = np.empty(num_rtm_profile_levels, 'f4')
    rtm_wvmr_profile = np.empty(num_rtm_profile_levels, 'f4')
    rtm_ozmr_profile = np.empty(num_rtm_profile_levels, 'f4')
    # initialize indices
    # lowest Rtm Level that is above the lowest nwp Level
    lowest_level_rtm_nwp = num_rtm_profile_levels - 1
    # highest Rtm Level that is below the highest nwp Level
    highest_level_rtm_nwp = 0
    # lowest Rtm Level above the surface
    rtm_sfc_level = num_rtm_profile_levels - 1
    # lowest standard nwp Level above surface pressure
    nwp_p_near_sfc = nwp_press_profile[nwp_surface_level]
    # make wvmr at sfc
    es = vapor(nwp_air_temperature)
    e = nwp_surface_rh * es / 100.0
    wvmr_sfc = 1000.0 * 0.622 * (e / (nwp_surface_pressure - e))  # (g/kg)
    # determine some critical levels in the rtm profile
    for k in range(num_rtm_profile_levels):
        if press_profile_rtm[k] > nwp_press_profile[0]:
            highest_level_rtm_nwp = k
            break
    for k in range(num_rtm_profile_levels - 1, -1, -1):
        if press_profile_rtm[k] < nwp_surface_pressure:
            rtm_sfc_level = k
            break
    for k in range(num_rtm_profile_levels - 1, -1, -1):
        if press_profile_rtm[k] < nwp_p_near_sfc:
            lowest_level_rtm_nwp = k
            break

    # todo 有些疑问
    # compute t and wvmr lapse rate near the surface
    if nwp_surface_pressure != nwp_press_profile[num_nwp_profile_levels - 1]:
        d_t_d_p_near_sfc = (
                (nwp_t_profile[nwp_surface_level] - nwp_air_temperature) /
                (nwp_press_profile[nwp_surface_level] - nwp_surface_pressure)
        )
        d_wvmr_d_p_near_sfc = (
                (nwp_wvmr_profile[nwp_surface_level] - wvmr_sfc) /
                (nwp_press_profile[nwp_surface_level] - nwp_surface_pressure)
        )
    else:
        d_t_d_p_near_sfc = (
                (nwp_t_profile[nwp_surface_level - 1] - nwp_air_temperature) /
                (nwp_press_profile[nwp_surface_level - 1] - nwp_surface_pressure)
        )
        d_wvmr_d_p_near_sfc = (
                (nwp_wvmr_profile[nwp_surface_level - 1] - wvmr_sfc) /
                (nwp_press_profile[nwp_surface_level - 1] - nwp_surface_pressure)
        )

    if nwp_press_profile[nwp_surface_level - 1] != nwp_press_profile[nwp_surface_level]:
        d_z_d_p_near_sfc = (
                (nwp_z_profile[nwp_surface_level - 1] - nwp_z_profile[nwp_surface_level]) /
                (nwp_press_profile[nwp_surface_level - 1] - nwp_press_profile[nwp_surface_level])
        )
    else:
        d_z_d_p_near_sfc = 0.0

    # compute temperature offset between standard and nwp profiles at top
    # this will be added to the standard profiles
    t_offset = nwp_t_profile[0] - t_std_profile_rtm[highest_level_rtm_nwp]
    # for rtm levels above the highest nwp level, use rtm standard values
    for k in prange(highest_level_rtm_nwp):
        rtm_z_profile[k] = nwp_z_profile[0]
        rtm_t_profile[k] = t_std_profile_rtm[k] + t_offset
        rtm_wvmr_profile[k] = wvmr_std_profile_rtm[k]
        rtm_ozmr_profile[k] = ozmr_std_profile_rtm[k]
    # rtm levels within standard nwp levels above the surface

    interpolator = VerticalLinearInterpolator(
        nwp_press_profile, press_profile_rtm[highest_level_rtm_nwp: lowest_level_rtm_nwp + 1]
    )
    rtm_t_profile[highest_level_rtm_nwp: lowest_level_rtm_nwp + 1] = interpolator.interp(nwp_t_profile)
    rtm_z_profile[highest_level_rtm_nwp: lowest_level_rtm_nwp + 1] = interpolator.interp(nwp_z_profile)
    rtm_wvmr_profile[highest_level_rtm_nwp: lowest_level_rtm_nwp + 1] = interpolator.interp(nwp_wvmr_profile)
    rtm_ozmr_profile[highest_level_rtm_nwp: lowest_level_rtm_nwp + 1] = interpolator.interp(nwp_ozmr_profile)

    # for k in range(highest_level_rtm_nwp, lowest_level_rtm_nwp + 1):
    #     # rtm_t_profile[k] = (nwp_t_profile(k_rtm_nwp[k]) + x_rtm_nwp[k] *
    #     # (nwp_t_profile(k_rtm_nwp[k] + 1) - nwp_t_profile(k_rtm_nwp[k])))
    #     rtm_t_profile[k] = np.interp(press_profile_rtm[k], nwp_press_profile, nwp_t_profile)
    #     # rtm_z_profile[k] = (nwp_z_profile(k_rtm_nwp[k]) + x_rtm_nwp[k] *
    #     # (nwp_z_profile(k_rtm_nwp[k] + 1) - nwp_z_profile(k_rtm_nwp[k])))
    #     rtm_z_profile[k] = np.interp(press_profile_rtm[k], nwp_press_profile, nwp_z_profile)
    #     # rtm_wvmr_profile[k] = (nwp_wvmr_profile(k_rtm_nwp[k]) + x_rtm_nwp[k] *
    #     # (nwp_wvmr_profile(k_rtm_nwp[k] + 1) - nwp_wvmr_profile(k_rtm_nwp[k])))
    #     rtm_wvmr_profile[k] = np.interp(press_profile_rtm[k], nwp_press_profile, nwp_wvmr_profile)
    #     # rtm_ozmr_profile[k] = (nwp_ozmr_profile(k_rtm_nwp[k]) + x_rtm_nwp[k] *
    #     # (nwp_ozmr_profile(k_rtm_nwp[k] + 1) - nwp_ozmr_profile(k_rtm_nwp[k])))
    #     rtm_ozmr_profile[k] = np.interp(press_profile_rtm[k], nwp_press_profile, nwp_ozmr_profile)

    # rtm levels that are below the lowest nwp level but above the surface
    for k in prange(lowest_level_rtm_nwp + 1, rtm_sfc_level + 1):
        rtm_t_profile[k] = nwp_air_temperature + d_t_d_p_near_sfc * (press_profile_rtm[k] - nwp_surface_pressure)
        rtm_wvmr_profile[k] = wvmr_sfc + d_wvmr_d_p_near_sfc * (press_profile_rtm[k] - nwp_surface_pressure)
        rtm_z_profile[k] = nwp_surface_elevation + d_z_d_p_near_sfc * (press_profile_rtm[k] - nwp_surface_pressure)
        rtm_ozmr_profile[k] = nwp_ozmr_profile[num_nwp_profile_levels - 1]
    # rtm levels below the surface
    for k in prange(rtm_sfc_level + 1, num_rtm_profile_levels):
        rtm_t_profile[k] = nwp_air_temperature
        rtm_z_profile[k] = nwp_surface_elevation + d_z_d_p_near_sfc * (press_profile_rtm[k] - nwp_surface_pressure)
        rtm_wvmr_profile[k] = wvmr_sfc
        rtm_ozmr_profile[k] = ozmr_std_profile_rtm[k]
    # if using ncep reanalysis which has no ozone profile, use default
    # if nwp_opt == 2:
    # rtm_ozmr_profile = ozmr_std_profile_rtm
    return rtm_t_profile, rtm_z_profile, rtm_wvmr_profile, rtm_ozmr_profile


@njit(nogil=True, error_model='numpy', boundscheck=True)
def find_rtm_levels(rtm_n_levels, rtm_p_std, rtm_t_prof, nwp_p_sfc, nwp_p_tropo):
    rtm_sfc_level = rtm_n_levels - 1
    for k in range(rtm_n_levels - 1, -1, -1):
        if rtm_p_std[k] < nwp_p_sfc:
            rtm_sfc_level = k
            break
    # find tropopause level based on tropopause pressure
    # tropopause is between tropopause_level and tropopause_level + 1
    rtm_tropo_level = -1  # todo
    for k in range(rtm_sfc_level):
        if rtm_p_std[k] <= nwp_p_tropo < rtm_p_std[k + 1]:
            rtm_tropo_level = k
            break

    rtm_inversion_level = -1
    if rtm_tropo_level >= 0 and rtm_sfc_level >= 0:
        for k in range(rtm_tropo_level, rtm_sfc_level):
            if (rtm_t_prof[k] - rtm_t_prof[k + 1]) > delta_t_inversion and rtm_p_std[k] >= p_inversion_min:
                rtm_inversion_level = k
                break
    return rtm_sfc_level, rtm_tropo_level, rtm_inversion_level


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_channel_atm_sfc_rad_bt(
        chan_idx,
        sfc_idx,
        profile_weight,
        sfc_ems,
        sfc_temp,
        rad_atm_profile,
        trans_atm_profile,
):
    sfc_rad = sfc_ems * planck_rad_fast(chan_idx, sfc_temp)[0]
    rad_atm = rad_atm_profile[sfc_idx] + (rad_atm_profile[sfc_idx + 1] - rad_atm_profile[sfc_idx]) * profile_weight
    trans_atm = trans_atm_profile[sfc_idx] + (
            trans_atm_profile[sfc_idx + 1] - trans_atm_profile[sfc_idx]) * profile_weight
    rad_atm_sfc = rad_atm + trans_atm * sfc_rad
    bt_atm_sfc = planck_temp_fast(chan_idx, rad_atm_sfc)[0]
    return rad_atm_sfc, bt_atm_sfc


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def get_pixel_nwp_rtm(
        bad_pixel_mask,
        space_mask,

        nwp_n_levels,
        nwp_p_std,
        nwp_pix_t_prof,
        nwp_pix_rh_prof,
        nwp_pix_sfc_level,
        nwp_pix_z_sfc,
        nwp_pix_t_air,
        nwp_pix_rh_sfc,
        nwp_pix_p_sfc,
        nwp_pix_z_prof,
        nwp_pix_ozone_prof,
        nwp_pix_p_tropo,

        p_std,

        coef_dry, coef_ozon, coef_wvp_cont, coef_wvp_solid, coef_wvp_liquid,

        geo_cos_zen, geo_cos_sol_zen,
        sfc_land, sfc_z_sfc, nwp_pix_t_sfc,
        ch_sfc_ems, ch_rad_toa,

):
    rtm_sfc_level = np.empty(image_shape, 'i1')
    rtm_tropo_level = np.empty(image_shape, 'i1')
    rtm_inversion_level = np.empty(image_shape, 'i1')
    rtm_t_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_z_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_wvmr_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_ozmr_prof = np.empty((*image_shape, rtm_n_levels), 'f4')

    nwp_pix_tpw_prof = np.empty((*image_shape, nwp_n_levels), 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                rtm_sfc_level[line_idx, elem_idx] = missing_value_int1
                rtm_tropo_level[line_idx, elem_idx] = missing_value_int1
                rtm_inversion_level[line_idx, elem_idx] = missing_value_int1
                rtm_t_prof[line_idx, elem_idx] = nan
                rtm_z_prof[line_idx, elem_idx] = nan
                rtm_wvmr_prof[line_idx, elem_idx] = nan
                rtm_ozmr_prof[line_idx, elem_idx] = nan
                # rtm_tpw_prof[line_idx, elem_idx] = nan
                nwp_pix_tpw_prof[line_idx, elem_idx] = nan
            else:
                nwp_pix_wvmr = compute_nwp_wvmr_profile(
                    nwp_p_std,
                    nwp_pix_t_prof[line_idx, elem_idx],
                    nwp_pix_rh_prof[line_idx, elem_idx],
                )

                nwp_pix_tpw_prof[line_idx, elem_idx] = compute_tpw_profile_nwp(
                    nwp_p_std, nwp_pix_wvmr
                )

                (
                    rtm_t_prof[line_idx, elem_idx],
                    rtm_z_prof[line_idx, elem_idx],
                    rtm_wvmr_prof[line_idx, elem_idx],
                    rtm_ozmr_prof[line_idx, elem_idx]
                ) = convert_atmos_prof_nwp_rtm(
                    nwp_n_levels,
                    nwp_pix_sfc_level[line_idx, elem_idx],
                    nwp_pix_z_sfc[line_idx, elem_idx],
                    nwp_pix_t_air[line_idx, elem_idx],
                    nwp_pix_rh_sfc[line_idx, elem_idx],
                    nwp_pix_p_sfc[line_idx, elem_idx],
                    nwp_p_std,
                    nwp_pix_t_prof[line_idx, elem_idx],
                    nwp_pix_z_prof[line_idx, elem_idx],
                    nwp_pix_wvmr,
                    nwp_pix_ozone_prof[line_idx, elem_idx],
                    rtm_n_levels,
                    rtm_p_std,
                    rtm_t_std,
                    rtm_wvmr_std,
                    rtm_ozmr_std,
                )

                (
                    rtm_sfc_level[line_idx, elem_idx],
                    rtm_tropo_level[line_idx, elem_idx],
                    rtm_inversion_level[line_idx, elem_idx]
                ) = find_rtm_levels(
                    rtm_n_levels,
                    rtm_p_std,
                    rtm_t_prof[line_idx, elem_idx],
                    nwp_pix_p_sfc[line_idx, elem_idx],
                    nwp_pix_p_tropo[line_idx, elem_idx],
                )

    rtm_ch_trans_atm_profile = Dict.empty(i1, f4_3d_array)
    rtm_ch_rad_atm_profile = Dict.empty(i1, f4_3d_array)
    rtm_ch_rad_bb_cloud_profile = Dict.empty(i1, f4_3d_array)
    for c in thermal_channels:
        rtm_ch_trans_atm_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')
        rtm_ch_rad_atm_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')
        rtm_ch_rad_bb_cloud_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')

    n_m = n_l - 1

    p_ref, t_ref, w_ref, o_ref = conpir(p_std, t_std, w_std, o_std, n_l)

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                for c in thermal_channels:
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx] = nan
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx] = nan
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx] = nan
            else:
                p_avg, t_avg, w_amt, o_amt = conpir(
                    p_std,
                    rtm_t_prof[line_idx, elem_idx],
                    rtm_wvmr_prof[line_idx, elem_idx],
                    rtm_ozmr_prof[line_idx, elem_idx],
                    n_l
                )

                sec_z = np.full(n_m, 1.0 / geo_cos_zen[line_idx, elem_idx])

                x_dry, x_wet, x_ozo, x_con = calpir(
                    t_ref, w_ref, o_ref,
                    t_avg, w_amt, o_amt, p_avg,
                    sec_z,
                    n_m, n_xd, n_xw, n_xo, n_xc
                )
                # matrix multiplication coef x predictors
                for c in thermal_channels:
                    tau_d = tau_doc(coef_dry[c][:, :], x_dry)
                    # ozone
                    tau_o = tau_doc(coef_ozon[c][:, :], x_ozo)
                    # wvp cont
                    tau_c = tau_doc(coef_wvp_cont[c][:, :], x_con)
                    # avp solid+liquid
                    tau_w = tau_wtr(coef_wvp_solid[c][:, :], coef_wvp_liquid[c][:, :], x_wet)
                    # build total transmission
                    tau_t = tau_d * tau_o * tau_c * tau_w
                    # done
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx] = tau_t

                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx] = np.empty(rtm_n_levels, 'f4')
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx][0] = 0.0
                    for lev_idx in range(1, rtm_n_levels):
                        t_mean = 0.5 * (
                                rtm_t_prof[line_idx, elem_idx][lev_idx - 1] +
                                rtm_t_prof[line_idx, elem_idx][lev_idx]
                        )
                        b_mean = planck_rad_fast(c, t_mean)[0]
                        rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx] = (
                                rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx - 1] +
                                (rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx - 1] -
                                 rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx]) *
                                b_mean
                        )

                    # 红外通道
                    # --- upwelling profiles
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx] = np.empty(rtm_n_levels, 'f4')
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][0] = 0.0

                    for lev_idx in range(1, rtm_n_levels):
                        b_level = planck_rad_fast(c, rtm_t_prof[line_idx, elem_idx][lev_idx])[0]
                        rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][lev_idx] = (
                                rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx] +
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx] * b_level
                        )

                for c in np.array([14], 'i1'):
                    trans_total = 1.0

                    for lev_idx in range(1, rtm_n_levels):
                        trans_layer = (
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx] /
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx - 1]
                        )
                        trans_total = trans_total * trans_layer

    ch_rad_toa_clear = {
        7: np.full(image_shape, nan, 'f4'),
        # 8: np.full(image_shape, nan, 'f4'),
        9: np.full(image_shape, nan, 'f4'),
        # 10: np.full(image_shape, nan, 'f4'),
        # 11: np.full(image_shape, nan, 'f4'),
        # 12: np.full(image_shape, nan, 'f4'),
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }
    ch_bt_toa_clear = {
        7: np.full(image_shape, nan, 'f4'),
        # 8: np.full(image_shape, nan, 'f4'),
        9: np.full(image_shape, nan, 'f4'),
        # 10: np.full(image_shape, nan, 'f4'),
        # 11: np.full(image_shape, nan, 'f4'),
        # 12: np.full(image_shape, nan, 'f4'),
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }
    # ch_rad_atm_dwn_sfc = {
    #     13: np.full(image_shape, nan, 'f4'),
    #     14: np.full(image_shape, nan, 'f4')
    # }
    ch_ems_tropo = {
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }

    beta_110um_120um_tropo_rtm = np.full(image_shape, nan, 'f4')
    # beta_104um_120um_tropo_rtm = np.full(image_shape, nan, 'f4')
    beta_110um_133um_tropo_rtm = np.full(image_shape, nan, 'f4')
    # beta_104um_133um_tropo_rtm = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                continue
            if space_mask[line_idx, elem_idx]:
                continue

            if sfc_land[line_idx, elem_idx] == sym_land and not isnan(sfc_z_sfc[line_idx, elem_idx]):
                sfc_level_idx = locate(rtm_z_prof[line_idx, elem_idx], rtm_n_levels, sfc_z_sfc[line_idx, elem_idx])
            else:
                sfc_level_idx = rtm_sfc_level[line_idx, elem_idx]
            sfc_level_idx = max(0, min(rtm_n_levels - 2, sfc_level_idx))

            # # todo
            # if z_prof[lat_idx, lon_idx, sfc_level_idx + 1] != z_prof[lat_idx, lon_idx, sfc_level_idx]:
            prof_weight = (
                    (sfc_z_sfc[line_idx, elem_idx] - rtm_z_prof[line_idx, elem_idx][sfc_level_idx]) /
                    (rtm_z_prof[line_idx, elem_idx][sfc_level_idx + 1] - rtm_z_prof[line_idx, elem_idx][
                        sfc_level_idx])
            )
            prof_weight = max(0.0, min(1.0, prof_weight))
            # else:
            #     prof_weight = 1.0

            # 红外波段
            for c in thermal_channels:
                (
                    ch_rad_toa_clear[c][line_idx, elem_idx],
                    ch_bt_toa_clear[c][line_idx, elem_idx]
                ) = compute_channel_atm_sfc_rad_bt(
                    c,
                    sfc_level_idx,
                    prof_weight,
                    ch_sfc_ems[c][line_idx, elem_idx],
                    nwp_pix_t_sfc[line_idx, elem_idx],
                    # 各层对toa的透射率，以及各层对toa的累计的rad
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx],
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx],
                )

            # todo 自己做了修改
            if geo_cos_sol_zen[line_idx, elem_idx] > 0.0:
                trans_atm_total_profile = (
                        rtm_ch_trans_atm_profile[7][line_idx, elem_idx] **
                        ((geo_cos_zen[line_idx, elem_idx] + geo_cos_sol_zen[line_idx, elem_idx]) /
                         geo_cos_sol_zen[line_idx, elem_idx])
                )
                trans_atm_ch07_solar_total_rtm = (
                        trans_atm_total_profile[sfc_level_idx] +
                        (trans_atm_total_profile[sfc_level_idx + 1] - trans_atm_total_profile[sfc_level_idx]) *
                        prof_weight
                )

                sfc_ref = 1.0 - ch_sfc_ems[7][line_idx, elem_idx]

                ch_rad_toa_clear[7][line_idx, elem_idx] += (
                        trans_atm_ch07_solar_total_rtm * sfc_ref * geo_cos_sol_zen[line_idx, elem_idx] *
                        solar_ch07_nu / pi
                )
                ch_bt_toa_clear[7][line_idx, elem_idx] = planck_temp_fast(7, ch_rad_toa_clear[7][line_idx, elem_idx])[0]

            # compute_tropopause_emissivities()
            lev_bnd = rtm_tropo_level[line_idx, elem_idx]
            # --- check for missing tropopause level
            # todo
            if rtm_tropo_level[line_idx, elem_idx] == -1:
                bad_pixel_mask[line_idx, elem_idx] = sym_yes
            else:
                for c in np.array([14, 15, 16], 'i1'):
                    ch_ems_tropo[c][line_idx, elem_idx] = emissivity(
                        ch_rad_toa[c][line_idx, elem_idx],
                        ch_rad_toa_clear[c][line_idx, elem_idx],
                        rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][lev_bnd]
                    )

            # compute_beta_ratioes()
            # --- compute 11 and 12 beta ratio at tropopause
            beta_110um_120um_tropo_rtm[line_idx, elem_idx] = beta_ratio(
                ch_ems_tropo[15][line_idx, elem_idx], ch_ems_tropo[14][line_idx, elem_idx]
            )
            # --- compute 11 and 13.3 beta ratio at tropopause
            beta_110um_133um_tropo_rtm[line_idx, elem_idx] = beta_ratio(
                ch_ems_tropo[16][line_idx, elem_idx], ch_ems_tropo[14][line_idx, elem_idx]
            )

    return (
        rtm_sfc_level,
        rtm_tropo_level,
        rtm_inversion_level,
        rtm_t_prof,
        rtm_z_prof,
        rtm_wvmr_prof,
        rtm_ozmr_prof,
        nwp_pix_tpw_prof,
        rtm_ch_trans_atm_profile,
        rtm_ch_rad_atm_profile,
        rtm_ch_rad_bb_cloud_profile,
        ch_rad_toa_clear, ch_bt_toa_clear,
        ch_ems_tropo,
        beta_110um_120um_tropo_rtm,
        beta_110um_133um_tropo_rtm,
    )


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def test0(
        nwp_n_levels,
        nwp_p_std,
        nwp_pix_t_prof,
        nwp_pix_rh_prof,
        nwp_pix_sfc_level,
        nwp_pix_z_sfc,
        nwp_pix_t_air,
        nwp_pix_rh_sfc,
        nwp_pix_p_sfc,
        nwp_pix_z_prof,
        nwp_pix_ozone_prof,
        nwp_pix_p_tropo,
        bad_pixel_mask,
):
    rtm_sfc_level = np.empty(image_shape, 'i1')
    rtm_tropo_level = np.empty(image_shape, 'i1')
    rtm_inversion_level = np.empty(image_shape, 'i1')
    rtm_t_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_z_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_wvmr_prof = np.empty((*image_shape, rtm_n_levels), 'f4')
    rtm_ozmr_prof = np.empty((*image_shape, rtm_n_levels), 'f4')

    nwp_pix_tpw_prof = np.empty((*image_shape, nwp_n_levels), 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:

                rtm_sfc_level[line_idx, elem_idx] = missing_value_int1
                rtm_tropo_level[line_idx, elem_idx] = missing_value_int1
                rtm_inversion_level[line_idx, elem_idx] = missing_value_int1
                rtm_t_prof[line_idx, elem_idx] = nan
                rtm_z_prof[line_idx, elem_idx] = nan
                rtm_wvmr_prof[line_idx, elem_idx] = nan
                rtm_ozmr_prof[line_idx, elem_idx] = nan
                # rtm_tpw_prof[line_idx, elem_idx] = nan
                nwp_pix_tpw_prof[line_idx, elem_idx] = nan
            else:
                nwp_pix_wvmr = compute_nwp_wvmr_profile(
                    nwp_p_std,
                    nwp_pix_t_prof[line_idx, elem_idx],
                    nwp_pix_rh_prof[line_idx, elem_idx],
                )

                nwp_pix_tpw_prof[line_idx, elem_idx] = compute_tpw_profile_nwp(
                    nwp_p_std, nwp_pix_wvmr
                )

                (
                    rtm_t_prof[line_idx, elem_idx],
                    rtm_z_prof[line_idx, elem_idx],
                    rtm_wvmr_prof[line_idx, elem_idx],
                    rtm_ozmr_prof[line_idx, elem_idx]
                ) = convert_atmos_prof_nwp_rtm(
                    nwp_n_levels,
                    nwp_pix_sfc_level[line_idx, elem_idx],
                    nwp_pix_z_sfc[line_idx, elem_idx],
                    nwp_pix_t_air[line_idx, elem_idx],
                    nwp_pix_rh_sfc[line_idx, elem_idx],
                    nwp_pix_p_sfc[line_idx, elem_idx],
                    nwp_p_std,
                    nwp_pix_t_prof[line_idx, elem_idx],
                    nwp_pix_z_prof[line_idx, elem_idx],
                    nwp_pix_wvmr,
                    nwp_pix_ozone_prof[line_idx, elem_idx],
                    rtm_n_levels,
                    rtm_p_std,
                    rtm_t_std,
                    rtm_wvmr_std,
                    rtm_ozmr_std,
                )

                (
                    rtm_sfc_level[line_idx, elem_idx],
                    rtm_tropo_level[line_idx, elem_idx],
                    rtm_inversion_level[line_idx, elem_idx]
                ) = find_rtm_levels(
                    rtm_n_levels,
                    rtm_p_std,
                    rtm_t_prof[line_idx, elem_idx],
                    nwp_pix_p_sfc[line_idx, elem_idx],
                    nwp_pix_p_tropo[line_idx, elem_idx],
                )

    return (
        rtm_sfc_level,
        rtm_tropo_level,
        rtm_inversion_level,
        rtm_t_prof,
        rtm_z_prof,
        rtm_wvmr_prof,
        rtm_ozmr_prof,
        nwp_pix_tpw_prof,
    )


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def test1(
        p_std,
        rtm_t_prof,
        rtm_wvmr_prof,
        rtm_ozmr_prof,
        geo_cos_zen,
        bad_pixel_mask,

        coef_dry, coef_ozon, coef_wvp_cont, coef_wvp_solid, coef_wvp_liquid,
):
    rtm_ch_trans_atm_profile = Dict.empty(i1, f4_3d_array)
    for c in thermal_channels:
        rtm_ch_trans_atm_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')

    n_m = n_l - 1

    p_ref, t_ref, w_ref, o_ref = conpir(p_std, t_std, w_std, o_std, n_l)

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                for c in thermal_channels:
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx] = nan
            else:

                p_avg, t_avg, w_amt, o_amt = conpir(
                    p_std,
                    rtm_t_prof[line_idx, elem_idx],
                    rtm_wvmr_prof[line_idx, elem_idx],
                    rtm_ozmr_prof[line_idx, elem_idx],
                    n_l
                )

                sec_z = np.full(n_m, 1.0 / geo_cos_zen[line_idx, elem_idx])

                x_dry, x_wet, x_ozo, x_con = calpir(
                    t_ref, w_ref, o_ref,
                    t_avg, w_amt, o_amt, p_avg,
                    sec_z,
                    n_m, n_xd, n_xw, n_xo, n_xc
                )
                # matrix multiplication coef x predictors
                for c in thermal_channels:
                    tau_d = tau_doc(coef_dry[c][:, :], x_dry)
                    # ozone
                    tau_o = tau_doc(coef_ozon[c][:, :], x_ozo)
                    # wvp cont
                    tau_c = tau_doc(coef_wvp_cont[c][:, :], x_con)
                    # avp solid+liquid
                    tau_w = tau_wtr(coef_wvp_solid[c][:, :], coef_wvp_liquid[c][:, :], x_wet)
                    # build total transmission
                    tau_t = tau_d * tau_o * tau_c * tau_w
                    # done
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx] = tau_t

    return rtm_ch_trans_atm_profile


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def test2(
        bad_pixel_mask,
        rtm_t_prof,
        rtm_ch_trans_atm_profile,
):
    rtm_ch_rad_atm_profile = Dict.empty(i1, f4_3d_array)
    rtm_ch_rad_bb_cloud_profile = Dict.empty(i1, f4_3d_array)

    for c in thermal_channels:
        rtm_ch_rad_atm_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')
        rtm_ch_rad_bb_cloud_profile[c] = np.full((*image_shape, rtm_n_levels), nan, 'f4')

    # todo
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                for c in thermal_channels:
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx] = nan
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx] = nan

            else:
                for c in thermal_channels:
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx] = np.empty(rtm_n_levels, 'f4')
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx][0] = 0.0
                    for lev_idx in range(1, rtm_n_levels):
                        t_mean = 0.5 * (
                                rtm_t_prof[line_idx, elem_idx][lev_idx - 1] +
                                rtm_t_prof[line_idx, elem_idx][lev_idx]
                        )
                        b_mean = planck_rad_fast(c, t_mean)[0]
                        rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx] = (
                                rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx - 1] +
                                (rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx - 1] -
                                 rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx]) *
                                b_mean
                        )

                    # 红外通道
                    # --- upwelling profiles
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx] = np.empty(rtm_n_levels, 'f4')
                    rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][0] = 0.0

                    for lev_idx in range(1, rtm_n_levels):
                        b_level = planck_rad_fast(c, rtm_t_prof[line_idx, elem_idx][lev_idx])[0]
                        rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][lev_idx] = (
                                rtm_ch_rad_atm_profile[c][line_idx, elem_idx][lev_idx] +
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx] * b_level
                        )

                for c in np.array([14], 'i1'):
                    trans_total = 1.0

                    for lev_idx in range(1, rtm_n_levels):
                        trans_layer = (
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx] /
                                rtm_ch_trans_atm_profile[c][line_idx, elem_idx][lev_idx - 1]
                        )
                        trans_total = trans_total * trans_layer

    return rtm_ch_rad_atm_profile, rtm_ch_rad_bb_cloud_profile


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def test3(
        bad_pixel_mask, space_mask,
        geo_cos_zen, geo_cos_sol_zen,
        sfc_land, sfc_z_sfc, nwp_pix_t_sfc,
        ch_sfc_ems, ch_rad_toa,

        rtm_sfc_level,
        rtm_tropo_level,
        rtm_z_prof,

        # rtm_ch_trans_atm_total_profile,
        rtm_ch_trans_atm_profile,
        rtm_ch_rad_atm_profile,
        rtm_ch_rad_bb_cloud_profile,
):
    ch_rad_toa_clear = {
        7: np.full(image_shape, nan, 'f4'),
        # 8: np.full(image_shape, nan, 'f4'),
        9: np.full(image_shape, nan, 'f4'),
        # 10: np.full(image_shape, nan, 'f4'),
        # 11: np.full(image_shape, nan, 'f4'),
        # 12: np.full(image_shape, nan, 'f4'),
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }
    ch_bt_toa_clear = {
        7: np.full(image_shape, nan, 'f4'),
        # 8: np.full(image_shape, nan, 'f4'),
        9: np.full(image_shape, nan, 'f4'),
        # 10: np.full(image_shape, nan, 'f4'),
        # 11: np.full(image_shape, nan, 'f4'),
        # 12: np.full(image_shape, nan, 'f4'),
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }
    # ch_rad_atm_dwn_sfc = {
    #     13: np.full(image_shape, nan, 'f4'),
    #     14: np.full(image_shape, nan, 'f4')
    # }
    ch_ems_tropo = {
        # 13: np.full(image_shape, nan, 'f4'),
        14: np.full(image_shape, nan, 'f4'),
        15: np.full(image_shape, nan, 'f4'),
        16: np.full(image_shape, nan, 'f4')
    }

    beta_110um_120um_tropo_rtm = np.full(image_shape, nan, 'f4')
    # beta_104um_120um_tropo_rtm = np.full(image_shape, nan, 'f4')
    beta_110um_133um_tropo_rtm = np.full(image_shape, nan, 'f4')
    # beta_104um_133um_tropo_rtm = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx]:
                continue
            if space_mask[line_idx, elem_idx]:
                continue

            if sfc_land[line_idx, elem_idx] == sym_land and not isnan(sfc_z_sfc[line_idx, elem_idx]):
                sfc_level_idx = locate(rtm_z_prof[line_idx, elem_idx], rtm_n_levels, sfc_z_sfc[line_idx, elem_idx])
            else:
                sfc_level_idx = rtm_sfc_level[line_idx, elem_idx]
            sfc_level_idx = max(0, min(rtm_n_levels - 2, sfc_level_idx))

            # # todo
            # if z_prof[lat_idx, lon_idx, sfc_level_idx + 1] != z_prof[lat_idx, lon_idx, sfc_level_idx]:
            prof_weight = (
                    (sfc_z_sfc[line_idx, elem_idx] - rtm_z_prof[line_idx, elem_idx][sfc_level_idx]) /
                    (rtm_z_prof[line_idx, elem_idx][sfc_level_idx + 1] - rtm_z_prof[line_idx, elem_idx][
                        sfc_level_idx])
            )
            prof_weight = max(0.0, min(1.0, prof_weight))
            # else:
            #     prof_weight = 1.0

            # 红外波段
            for c in thermal_channels:
                (
                    ch_rad_toa_clear[c][line_idx, elem_idx],
                    ch_bt_toa_clear[c][line_idx, elem_idx]
                ) = compute_channel_atm_sfc_rad_bt(
                    c,
                    sfc_level_idx,
                    prof_weight,
                    ch_sfc_ems[c][line_idx, elem_idx],
                    nwp_pix_t_sfc[line_idx, elem_idx],
                    # 各层对toa的透射率，以及各层对toa的累计的rad
                    rtm_ch_rad_atm_profile[c][line_idx, elem_idx],
                    rtm_ch_trans_atm_profile[c][line_idx, elem_idx],
                )

                # if ch_rad_toa_clear[chan_idx][line_idx, elem_idx] <= 0:
                #     print(
                #         chan_idx,
                #         sfc_level_idx,
                #         prof_weight,
                #         ch_sfc_ems[chan_idx][line_idx, elem_idx],
                #         nwp_pix_t_sfc[line_idx, elem_idx],
                #         # 各层对toa的透射率，以及各层对toa的累计的rad
                #         rtm_ch_rad_atm_profile[chan_idx][line_idx, elem_idx],
                #         rtm_ch_trans_atm_profile[chan_idx][line_idx, elem_idx],
                #     )
                #     assert ch_rad_toa_clear[chan_idx][line_idx, elem_idx] > 0

            # for c in np.array([13, 14], 'i1'):
            #     ch_rad_atm_dwn_sfc[c][line_idx, elem_idx] = compute_channel_atm_dwn_sfc_rad(
            #         sfc_level_idx,
            #         prof_weight,
            #         rtm_ch_rad_atm_dwn_profile[c][line_idx, elem_idx]
            #     )

            # trans_atm_ch07_solar_total_rtm = (
            #         rtm_ch_trans_atm_total_profile[7][line_idx, elem_idx, sfc_level_idx] +
            #         (rtm_ch_trans_atm_total_profile[7][line_idx, elem_idx, sfc_level_idx + 1] -
            #          rtm_ch_trans_atm_total_profile[7][line_idx, elem_idx, sfc_level_idx]) *
            #         prof_weight
            # )
            #
            # if geo_cos_sol_zen[line_idx, elem_idx] > 0.0:
            #     sfc_ref = 1.0 - ch_sfc_ems[7][line_idx, elem_idx]
            #
            #     ch_rad_toa_clear[7][line_idx, elem_idx] += (
            #             trans_atm_ch07_solar_total_rtm *
            #             sfc_ref * geo_cos_sol_zen[line_idx, elem_idx] * solar_ch07_nu / pi
            #     )
            #     ch_bt_toa_clear[7][line_idx, elem_idx] = planck_temp_fast(7, ch_rad_toa_clear[7][line_idx, elem_idx])[0]

            # todo 自己做了修改
            if geo_cos_sol_zen[line_idx, elem_idx] > 0.0:
                trans_atm_total_profile = (
                        rtm_ch_trans_atm_profile[7][line_idx, elem_idx] **
                        ((geo_cos_zen[line_idx, elem_idx] + geo_cos_sol_zen[line_idx, elem_idx]) /
                         geo_cos_sol_zen[line_idx, elem_idx])
                )
                trans_atm_ch07_solar_total_rtm = (
                        trans_atm_total_profile[sfc_level_idx] +
                        (trans_atm_total_profile[sfc_level_idx + 1] - trans_atm_total_profile[sfc_level_idx]) *
                        prof_weight
                )

                sfc_ref = 1.0 - ch_sfc_ems[7][line_idx, elem_idx]

                ch_rad_toa_clear[7][line_idx, elem_idx] += (
                        trans_atm_ch07_solar_total_rtm * sfc_ref * geo_cos_sol_zen[line_idx, elem_idx] *
                        solar_ch07_nu / pi
                )
                ch_bt_toa_clear[7][line_idx, elem_idx] = planck_temp_fast(7, ch_rad_toa_clear[7][line_idx, elem_idx])[0]

                # if ch_rad_toa_clear[chan_idx][line_idx, elem_idx] <= 0:
                #     print(
                #         trans_atm_ch07_solar_total_rtm,
                #         sfc_ref,
                #         geo_cos_sol_zen[line_idx, elem_idx],
                #     )
                #     assert ch_rad_toa_clear[chan_idx][line_idx, elem_idx] > 0

            # compute_tropopause_emissivities()
            lev_bnd = rtm_tropo_level[line_idx, elem_idx]
            # --- check for missing tropopause level
            # todo
            if rtm_tropo_level[line_idx, elem_idx] == -1:
                bad_pixel_mask[line_idx, elem_idx] = sym_yes
            else:
                for c in np.array([14, 15, 16], 'i1'):
                    ch_ems_tropo[c][line_idx, elem_idx] = emissivity(
                        ch_rad_toa[c][line_idx, elem_idx],
                        ch_rad_toa_clear[c][line_idx, elem_idx],
                        rtm_ch_rad_bb_cloud_profile[c][line_idx, elem_idx][lev_bnd]
                    )

            # compute_beta_ratioes()
            # --- compute 11 and 12 beta ratio at tropopause
            beta_110um_120um_tropo_rtm[line_idx, elem_idx] = beta_ratio(
                ch_ems_tropo[15][line_idx, elem_idx], ch_ems_tropo[14][line_idx, elem_idx]
            )
            # --- compute 11 and 13.3 beta ratio at tropopause
            beta_110um_133um_tropo_rtm[line_idx, elem_idx] = beta_ratio(
                ch_ems_tropo[16][line_idx, elem_idx], ch_ems_tropo[14][line_idx, elem_idx]
            )

    return (
        ch_rad_toa_clear, ch_bt_toa_clear,
        ch_ems_tropo,
        beta_110um_120um_tropo_rtm,  # beta_104um_120um_tropo_rtm,
        beta_110um_133um_tropo_rtm,  # beta_104um_133um_tropo_rtm
    )
