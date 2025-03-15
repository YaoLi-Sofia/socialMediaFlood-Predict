from math import pi, nan, inf, isnan, sqrt, radians, cos, exp

import numpy as np
from numba import njit, vectorize, prange

from constants import (
    missing_value_int1,
    sym_no,
    sym_yes,
    sym_land,
    sym_shallow_inland_water,
    sym_no_snow,
    sym_sea_ice,
    sym_snow,
)
from numerical_routines import covariance
from planck import planck_rad_fast
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
    solar_channels,
)
from utils import show_time


@vectorize(nopython=True, forceobj=False)
def compute_snow_class_nwp(nwp_wat_eqv_snow_depth, nwp_sea_ice_frac):
    snow_class_nwp = missing_value_int1
    if nwp_wat_eqv_snow_depth >= 0.0 or nwp_sea_ice_frac >= 0.0:
        snow_class_nwp = sym_no_snow
    if nwp_sea_ice_frac > 0.5:
        snow_class_nwp = sym_sea_ice
    if nwp_wat_eqv_snow_depth > 0.1:
        snow_class_nwp = sym_snow
    return snow_class_nwp


# ----------------------------------------------------------------------
# compute spatial metrics for 3x3 elements in a 2d array
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_nxn_metrics(n, bad_mask, z):
    z_min = np.full(image_shape, nan, 'f4')
    z_max = np.full(image_shape, nan, 'f4')
    z_mean = np.full(image_shape, nan, 'f4')
    z_std = np.full(image_shape, nan, 'f4')

    for i in prange(image_number_of_lines):
        i1 = max(0, i - n)  # left index of local array
        i2 = min(image_number_of_lines, i + n + 1)  # right index of local array
        for j in prange(image_number_of_elements):
            # --- set limits of nxn array in the j-direction
            j1 = max(0, j - n)  # top index of local array
            j2 = min(image_number_of_elements, j + n + 1)  # bottom index of local array

            if bad_mask[i, j] == sym_yes or isnan(z[i, j]):
                continue

            # --- initialize
            sum_tmp = 0.0
            sum_tmp2 = 0.0
            count_tmp = 0
            min_tmp = inf
            max_tmp = -inf
            n_good = 0

            # --- go through each element in nxn array

            for ii in range(i1, i2):
                for jj in range(j1, j2):

                    if bad_mask[ii, jj] == sym_yes:
                        continue

                    if isnan(z[ii, jj]):
                        continue

                    n_good += 1
                    sum_tmp = sum_tmp + z[ii, jj]
                    sum_tmp2 = sum_tmp2 + z[ii, jj] ** 2

                    if z[ii, jj] < min_tmp:
                        min_tmp = z[ii, jj]

                    if z[ii, jj] > max_tmp:
                        max_tmp = z[ii, jj]

            # --- if any good pixels found, compute mean and standard deviation
            if n_good > 0:
                z_mean[i, j] = sum_tmp / n_good
                z_std[i, j] = sqrt(max(0.0, (sum_tmp2 / n_good - z_mean[i, j] ** 2)))
                z_min[i, j] = min_tmp
                z_max[i, j] = max_tmp

            if isnan(max_tmp):
                print("missing max = ", n_good, max_tmp)

    return z_min, z_max, z_mean, z_std


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_min_max_mean_std_metrics(
        bad_pixel_mask,
        ch_ref_toa, ch_sfc_ref_white_sky, ch_bt_toa,
):
    ch03_ref_toa_min_3x3, ch03_ref_toa_max_3x3, ch03_ref_toa_mean_3x3, ch03_ref_toa_std_3x3 = compute_nxn_metrics(
        1, bad_pixel_mask, ch_ref_toa[3],
    )

    temp_pix_array_1, temp_pix_array_1, ch03_sfc_ref_white_sky_mean_3x3, temp_pix_array_1 = compute_nxn_metrics(
        1, bad_pixel_mask, ch_sfc_ref_white_sky[3],
    )

    temp_pix_array_1, temp_pix_array_1, temp_pix_array_1, ch07_bt_toa_std_3x3 = compute_nxn_metrics(
        1, bad_pixel_mask, ch_bt_toa[7],
    )

    temp_pix_array_1, ch09_bt_toa_max_3x3, temp_pix_array_1, ch09_bt_toa_std_3x3 = compute_nxn_metrics(
        1, bad_pixel_mask, ch_bt_toa[9],
    )

    ch14_bt_toa_min_3x3, ch14_bt_toa_max_3x3, ch14_bt_toa_mean_3x3, ch14_bt_toa_std_3x3 = compute_nxn_metrics(
        1, bad_pixel_mask, ch_bt_toa[14],
    )

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if isnan(ch_sfc_ref_white_sky[3][i, j]) and not isnan(ch03_sfc_ref_white_sky_mean_3x3[i, j]):
                ch_sfc_ref_white_sky[3][i, j] = ch03_sfc_ref_white_sky_mean_3x3[i, j]

    return (
        ch03_ref_toa_min_3x3,
        ch03_ref_toa_std_3x3,
        ch07_bt_toa_std_3x3,
        ch14_bt_toa_std_3x3,
        ch09_bt_toa_max_3x3,
        ch14_bt_toa_max_3x3,
        ch14_bt_toa_min_3x3
    )


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_spatial_correlation_arrays(bad_pixel_mask, ch_bt_toa):
    covar_ch09_ch14_5x5 = np.empty(image_shape, 'f4')

    for line_idx in prange(image_number_of_lines):
        if line_idx < 2:
            line_idx_min = 0
        else:
            line_idx_min = line_idx - 2
        if line_idx > image_number_of_lines - 3:
            line_idx_max = image_number_of_lines
        else:
            line_idx_max = line_idx + 3
        for elem_idx in prange(image_number_of_elements):
            if elem_idx < 2:
                elem_idx_min = 0
            else:
                elem_idx_min = elem_idx - 2
            if elem_idx > image_number_of_elements - 3:
                elem_idx_max = image_number_of_elements
            else:
                elem_idx_max = elem_idx + 3

                # --- compute 5x5 arrays

            covar_ch09_ch14_5x5[line_idx, elem_idx] = covariance(
                ch_bt_toa[14][line_idx_min:line_idx_max, elem_idx_min:elem_idx_max],
                ch_bt_toa[9][line_idx_min:line_idx_max, elem_idx_min:elem_idx_max],
                bad_pixel_mask[line_idx_min:line_idx_max, elem_idx_min:elem_idx_max]
            )
    return covar_ch09_ch14_5x5


@vectorize(nopython=True, forceobj=False)
def term_refl_norm(cos_sol_zen, reflectance):
    reflectance_normalized = reflectance * cos_sol_zen
    norm_param = 24.35 / (2 * cos_sol_zen + sqrt(498.5225 * (cos_sol_zen ** 2) + 1))
    reflectance_normalized = reflectance_normalized * norm_param
    return reflectance_normalized


# todo
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def ch07_pseudo_reflectance(
        solar_ch07_nu, cos_sol_zen, rad_ch07, bt_ch14, sun_earth_distance, bad_pixel_mask
):
    ref_ch07 = np.full(image_shape, nan, 'f4')
    ems_ch07 = np.full(image_shape, nan, 'f4')

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if bad_pixel_mask[i][j]:
                continue
            if bt_ch14[i][j] > 180.0:
                rad_ch07_ems = planck_rad_fast(7, bt_ch14[i][j])[0]
                ems_ch07[i][j] = rad_ch07[i][j] / rad_ch07_ems
            else:
                rad_ch07_ems = nan
                ems_ch07[i][j] = nan
            if rad_ch07_ems > 0.0 and rad_ch07[i][j] > 0.0:
                solar_irradiance = max(0.0, (solar_ch07_nu * cos_sol_zen[i][j]) / (sun_earth_distance ** 2))
                ref_ch07[i][j] = 100.0 * pi * (rad_ch07[i][j] - rad_ch07_ems) / (solar_irradiance - pi * rad_ch07_ems)
            if not isnan(ref_ch07[i][j]):
                ref_ch07[i][j] = max(-50.0, min(100.0, ref_ch07[i][j]))
    return ref_ch07, ems_ch07


# ------------------------------------------------------------
# compute the single scatter and aerosol reflectance
# this assumes that the gas is mixed in with scattering
# -----------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_clear_sky_scatter(
        tau_aer,
        wo_aer,
        g_aer,
        tau_ray,
        tau_gas,
        scattering_angle,
        cos_zen,
        cos_sol_zen,
        cloud_albedo_view,
        cloud_albedo_sun,
):
    # --- compute cosine of scattering angle
    mu = cos(radians(scattering_angle))

    # -- compute rayleigh phase function
    airmass = 1.0 / cos_zen + 1.0 / cos_sol_zen
    p_ray = 0.75 * (1.0 + mu ** 2)

    # --- compute total transmission
    tau_total = tau_aer + tau_ray + tau_gas
    trans_total = exp(-tau_total * airmass)

    tau_iso_total = (1.0 - g_aer) * tau_aer + tau_ray + tau_gas
    trans_iso_total_view = exp(-tau_iso_total / cos_zen)
    trans_iso_total_sun = exp(-tau_iso_total / cos_sol_zen)

    # --- compute total scattering optical depth
    tau_scat_total = wo_aer * tau_aer + tau_ray
    tau_iso_scat_total = wo_aer * (1.0 - g_aer) * tau_aer + tau_ray

    # --- single scatter albedo
    wo = (wo_aer * tau_aer + tau_ray) / tau_total

    # aerosol phase function (henyey-greenstein)
    p_aer = (1.0 - g_aer ** 2) / (1.0 + g_aer ** 2 - 2.0 * g_aer * mu) ** 1.5

    # --- compute effective phase function
    pf = p_aer
    if tau_scat_total > 0.0:
        pf = (wo_aer * tau_aer * p_aer + tau_ray * p_ray) / tau_scat_total

    # --- compute single scatter reflectance (0-100%)
    ref_ss_a = wo * pf / (4.0 * airmass * cos_zen * cos_sol_zen) * (1.0 - trans_total)
    ref_ss_b = tau_iso_scat_total / (2.0 * cos_sol_zen) * trans_iso_total_view * cloud_albedo_view
    ref_ss_c = tau_iso_scat_total / (2.0 * cos_zen) * trans_iso_total_sun * cloud_albedo_sun
    ref_ss = 100.0 * (ref_ss_a + ref_ss_b + ref_ss_c)

    return ref_ss


# todo 有更新
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True)
def atmos_corr(
        bad_pixel_mask,
        geo_sol_zen,
        geo_cos_zen,
        geo_scatter_zen,
        nwp_pix_tpw,
        ch_ref_toa,
        ch_sfc_ref_white_sky,
        sfc_sfc_type,
        sfc_snow,
        zc_opaque_cloud,
        pc_opaque_cloud,

        solar_rtm_tau_h2o_coef,
        solar_rtm_tau_ray,
        solar_rtm_tau_o3,
        solar_rtm_tau_o2,
        solar_rtm_tau_ch4,
        solar_rtm_tau_co2,
        solar_rtm_tau_aer,
        solar_rtm_wo_aer,
        solar_rtm_g_aer,

        ch03_sfc_alb_umd,
        ch04_sfc_alb_umd,
        ch05_sfc_alb_umd,
        ch07_sfc_alb_umd,
        ch03_snow_sfc_alb_umd,
        ch04_snow_sfc_alb_umd,
        ch05_snow_sfc_alb_umd,
        ch06_snow_sfc_alb_umd,

):
    ch_ref_toa_clear = {
        # 1: np.full(image_shape, nan, 'f4'),
        # 2: np.full(image_shape, nan, 'f4'),
        3: np.full(image_shape, nan, 'f4'),
        4: np.full(image_shape, nan, 'f4'),
        5: np.full(image_shape, nan, 'f4'),
        # 6: np.full(image_shape, nan, 'f4'),
        # 7: np.full(shape, nan, 'f4')
    }

    h_h2o = 2000  # m
    p_sfc = 1013.25
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue
            for c in solar_channels:
                if isnan(ch_ref_toa[c][line_idx, elem_idx]):
                    continue
                # --- set source angle
                source_zen = geo_sol_zen[line_idx, elem_idx]
                scattering_angle = geo_scatter_zen[line_idx, elem_idx]
                # --- check for appropriate illumination
                if source_zen >= 90.0:
                    continue
                # tau_h2o = (
                #         solar_rtm_tau_h2o_coef[chan_idx][0] +
                #         solar_rtm_tau_h2o_coef[chan_idx][1] * nwp_pix_tpw[line_idx, elem_idx] +
                #         solar_rtm_tau_h2o_coef[chan_idx][2] * nwp_pix_tpw[line_idx, elem_idx] ** 2
                # )
                # tau_gas = (
                #         max(0.0, tau_h2o) +
                #         solar_rtm_tau_o3[chan_idx] +
                #         solar_rtm_tau_o2[chan_idx] +
                #         solar_rtm_tau_co2[chan_idx] +
                #         solar_rtm_tau_ch4[chan_idx]
                # )
                # tau_aer = solar_rtm_tau_aer[chan_idx]
                # wo_aer = solar_rtm_wo_aer[chan_idx]
                # g_aer = solar_rtm_g_aer[chan_idx]
                # tau_ray = solar_rtm_tau_ray[chan_idx]

                # --- compute gas terms

                tpw_ac = nwp_pix_tpw[line_idx, elem_idx]
                if not isnan(zc_opaque_cloud[line_idx, elem_idx]):
                    zc = max(0.0, zc_opaque_cloud[line_idx, elem_idx])
                    tpw_ac *= exp(-zc / h_h2o)

                tau_h2o = (
                        solar_rtm_tau_h2o_coef[c][0] +
                        solar_rtm_tau_h2o_coef[c][1] * tpw_ac +
                        solar_rtm_tau_h2o_coef[c][2] * (tpw_ac ** 2)
                )
                tau_gas = (
                        max(0.0, tau_h2o) +
                        solar_rtm_tau_o3[c] +
                        solar_rtm_tau_o2[c] +
                        solar_rtm_tau_co2[c] +
                        solar_rtm_tau_ch4[c]
                )
                tau_ray = solar_rtm_tau_ray[c]
                tau_aer = solar_rtm_tau_aer[c]
                if not isnan(pc_opaque_cloud[line_idx, elem_idx]):
                    pc = min(1000.0, max(0.0, pc_opaque_cloud[line_idx, elem_idx]))
                    tau_ray *= pc_opaque_cloud[line_idx, elem_idx] / p_sfc
                    tau_aer *= (pc_opaque_cloud[line_idx, elem_idx] / p_sfc) ** 2

                wo_aer = solar_rtm_wo_aer[c]
                g_aer = solar_rtm_g_aer[c]

                # ------------------------------------------------------------------------
                # select gas and surface reflectance parameters
                # ------------------------------------------------------------------------
                if c == 1:
                    if not isnan(ch_sfc_ref_white_sky[3][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[3][line_idx, elem_idx] / 100.0
                    else:
                        albedo_view = ch03_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch03_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                elif c == 2:
                    if not isnan(ch_sfc_ref_white_sky[3][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[3][line_idx, elem_idx] / 100.0
                    else:
                        albedo_view = ch03_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch03_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                elif c == 3:
                    if not isnan(ch_sfc_ref_white_sky[3][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[3][line_idx, elem_idx] / 100.0
                    else:
                        albedo_view = ch03_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch03_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                elif c == 4:
                    if not isnan(ch_sfc_ref_white_sky[4][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[4][line_idx, elem_idx] / 100.0
                    else:
                        albedo_view = ch04_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch04_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                elif c == 5:
                    if not isnan(ch_sfc_ref_white_sky[5][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[5][line_idx, elem_idx] / 100.0
                    else:
                        albedo_view = ch05_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch05_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                else:  # elif chan_idx == 6:
                    if not isnan(ch_sfc_ref_white_sky[6][line_idx, elem_idx]):
                        albedo_view = ch_sfc_ref_white_sky[6][line_idx, elem_idx] / 100.0
                    else:
                        # note there is no ch06_sfc_alb_umd
                        albedo_view = ch05_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                    if sfc_snow[line_idx, elem_idx] != sym_no_snow:
                        albedo_view = ch06_snow_sfc_alb_umd[sfc_sfc_type[line_idx, elem_idx]] / 100.0
                albedo_sun = albedo_view

                if 0.0 <= source_zen < 90.0:
                    cos_source_zen = cos(radians(source_zen))
                else:
                    cos_source_zen = nan

                airmass_factor = 1.0 / geo_cos_zen[line_idx, elem_idx] + 1.0 / cos_source_zen

                # --- compute atmospheric scattering
                ref_ss = compute_clear_sky_scatter(
                    tau_aer,
                    wo_aer,
                    g_aer,
                    tau_ray,
                    tau_gas,
                    scattering_angle,
                    geo_cos_zen[line_idx, elem_idx],
                    cos_source_zen,
                    albedo_view,
                    albedo_sun
                )

                # --- compute total transmission for combining terms
                tau_total = tau_aer + tau_ray + tau_gas
                trans_total = exp(-tau_total * airmass_factor)

                # --- compute top of clear-sky atmosphere reflectance
                ch_ref_toa_clear[c][line_idx, elem_idx] = ref_ss + trans_total * 100.0 * albedo_view

    return ch_ref_toa_clear


# ----------------------------------------------------------------------
# --- compute a mask identifying presence of oceanic glint
# ---
# --- input and output passed through global arrays
# ----------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_glint(
        source_glint_zen, source_ref_toa, source_ref_std_3x3,
        bad_pixel_mask,
        sfc_land_mask, sfc_snow,
        ch_bt_toa, ch_bt_toa_clear,
        geo_sat_zen, bt_ch14_std_3x3
):
    glint_zen_thresh = 40.0

    # --- alias some global sizes into local values
    source_glint_mask = np.full(image_shape, missing_value_int1, 'i1')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- skip bad pixels
            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue

            # --- initialize valid pixel to no
            source_glint_mask[line_idx, elem_idx] = sym_no
            # --- skip land pixels
            if sfc_land_mask[line_idx, elem_idx] == sym_no and sfc_snow[line_idx, elem_idx] == sym_no_snow:
                # --- turn on in geometric glint cone and sufficient ref_ch1
                if source_glint_zen[line_idx, elem_idx] < glint_zen_thresh:
                    # --- assume to be glint if in geometric zone
                    source_glint_mask[line_idx, elem_idx] = sym_yes
                    # --- exclude pixels colder than the freezing temperature
                    if ch_bt_toa[14][line_idx, elem_idx] < 273.15:
                        source_glint_mask[line_idx, elem_idx] = sym_no
                        continue
                    # --- exclude pixels colder than the surface
                    if ch_bt_toa[14][line_idx, elem_idx] < ch_bt_toa_clear[14][line_idx, elem_idx] - 5.0:
                        source_glint_mask[line_idx, elem_idx] = sym_no
                        continue
                    # -turn off if non-uniform - but not near limb
                    if geo_sat_zen[line_idx, elem_idx] < 45.0:
                        if bt_ch14_std_3x3[line_idx, elem_idx] > 1.0:
                            source_glint_mask[line_idx, elem_idx] = sym_no
                            continue
                        if source_ref_std_3x3[line_idx, elem_idx] > 2.0:
                            source_glint_mask[line_idx, elem_idx] = sym_no
                            continue
                        # -checks on the value of ch1
                        # -turn off if dark
                        if source_ref_toa[line_idx, elem_idx] < 5.0:
                            source_glint_mask[line_idx, elem_idx] = sym_no
                            continue
                        # -turn off if bright
                        if 10.0 < source_glint_zen[line_idx, elem_idx] < 40.0:
                            refl_thresh = 25.0 - source_glint_zen[line_idx, elem_idx] / 3.0
                            if source_ref_toa[line_idx, elem_idx] > refl_thresh:
                                source_glint_mask[line_idx, elem_idx] = sym_no
                                continue

    return source_glint_mask


# ====================================================================
#
# attempt to fix the land classification based on observed ndvi
#
# if the ndvi is high and the land class is not land, this pixel should be land
# if the ndvi is low and the land class is land, this pixel should be water
#
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def modify_land_class_with_ndvi(bad_pixel_mask, geo_sol_zen, ch_ref_toa, sfc_land):
    ndvi_land_threshold = 0.25
    ndvi_water_threshold = -0.25
    sol_zen_threshold = 60.0

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue
            if geo_sol_zen[line_idx, elem_idx] > sol_zen_threshold:
                continue

            ndvi_temp = ((ch_ref_toa[4][line_idx, elem_idx] - ch_ref_toa[3][line_idx, elem_idx]) /
                         (ch_ref_toa[4][line_idx, elem_idx] + ch_ref_toa[3][line_idx, elem_idx]))

            if ndvi_temp > ndvi_land_threshold:
                sfc_land[line_idx, elem_idx] = sym_land
            if ndvi_temp < ndvi_water_threshold and sfc_land[line_idx, elem_idx] == sym_land:
                sfc_land[line_idx, elem_idx] = sym_shallow_inland_water
