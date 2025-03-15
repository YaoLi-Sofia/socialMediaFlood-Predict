import glob
import os
import time
from datetime import datetime, timedelta
from math import cos, radians, nan, floor

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from numba import njit, prange, i1, f4, vectorize
from numba.typed import Dict
from scipy.interpolate import NearestNDInterpolator

from acha import awg_cloud_height_bridge
from calibration_constants import solar_ch07_nu
from ccl import ccl_bridge
from cloud_base import cloud_base_bridge
from cloud_height_routines import (
    opaque_cloud_height,
    compute_altitude_from_pressure
)
from cloud_mask import compute_posterior_cld_probability, compute_cld_mask
from cloud_mask.ecm1.nb_cloud_mask_addons_module import (
    clavrx_dust_test, clavrx_smoke_over_water_test, clavrx_smoke_over_land_test, eumetsat_fire_test
)
from cloud_mask.ecm1.nb_cloud_mask_module import compute_bayes_sfc_type
from cloud_type_bridge import compute_cloud_type
from constants import (
    missing_value_int1,
    missing_value_int4,

    sym_no,
    sym_yes,

    sym_water_sfc,
    sym_open_shrubs_sfc,
    sym_bare_sfc,

    sym_land,
    sym_moderate_ocean,
    sym_deep_ocean,

    sym_no_snow,
    sym_sea_ice,
    sym_snow,
)
from cx_dncomp.dcomp_lut_mod import (
    ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
    phase_cld_ems, phase_cld_trn_ems
)
from cx_sfc_emissivity_mod import cx_sfc_ems_correct_for_sfc_type
from dcomp_derived_products_module import compute_cloud_water_path
from dncomp_clavrx_bridge_mod import awg_cloud_dncomp_algorithm
from download import GFSData
from interp import GlobalNearestInterpolator, GlobalLinearInterpolator
from numerical_routines import vapor, vapor_ice, wind_speed, compute_median_segment
from nwp_common import (
    qc_nwp, compute_nwp_levels_segment, compute_nwp_level_height, compute_segment_nwp_cloud_parameters,
    modify_nwp_pix_t_sfc
)
from pfaast.cx_pfaast_coef import coef_dry, coef_ozon, coef_wvp_cont, coef_wvp_solid, coef_wvp_liquid
from pfaast.cx_pfaast_constants import p_std
from pixel_common import (
    solar_rtm_tau_h2o_coef, solar_rtm_tau_ch4, solar_rtm_tau_co2, solar_rtm_tau_o2, solar_rtm_tau_o3,
    solar_rtm_tau_aer, solar_rtm_tau_ray,
    solar_rtm_wo_aer, solar_rtm_g_aer,
)
from pixel_rouines import (
    ch07_pseudo_reflectance, term_refl_norm, modify_land_class_with_ndvi,
    compute_snow_class_nwp,
    compute_glint, compute_spatial_correlation_arrays,
    compute_min_max_mean_std_metrics,
    atmos_corr
)
from public import (
    sensor_spatial_resolution_meters,
    solar_channels, thermal_channels,
    image_shape, image_number_of_lines, image_number_of_elements,
    geo_sat_zen_max_limit, geo_sat_zen_min_limit, geo_sol_zen_max_limit, geo_sol_zen_min_limit,
    nav_lat_max_limit, nav_lat_min_limit, nav_lon_max_limit, nav_lon_min_limit,
)
from reader import AgriL1
from reader import pix_lin2lon_lat
from rt_utilities import get_pixel_nwp_rtm
from rtm_common import rtm_p_std, rtm_n_levels
from snow_routines_mod import compute_snow_class
from surface_properties import (
    ch03_sfc_alb_umd, ch04_sfc_alb_umd, ch05_sfc_alb_umd, ch07_sfc_alb_umd,
    ch03_snow_sfc_alb_umd, ch04_snow_sfc_alb_umd, ch05_snow_sfc_alb_umd, ch06_snow_sfc_alb_umd,
    compute_binary_land_coast_masks,
)
from utils import show0, show_time
from viewing_geometry import (
    calculate_solar_zenith, calculate_solar_azimuth, calculate_sensor_zenith, calculate_sensor_azimuth,
    calculate_glint_angle, calculate_scattering_angle, calculate_relative_azimuth
)

# from numba.core.types import UniTuple, DictType
# from dcomp.dcomp_lut_mod import (
#     ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
#     phase_cld_ems, phase_cld_trn_ems
# )

# warnings.filterwarnings(
#     'ignore',
#     message='\nEncountered the use of a type that is scheduled for '
#             'deprecation: type 'reflected .*' found for argument '
#             ''.*' of function '.*'.\n\nFor more information visit '
#             '.*',
#     category=NumbaPendingDeprecationWarning,
# )

lrc_meander_flag = 1
max_lrc_distance = 10
min_lrc_jump = 0.0  # 0.5
max_lrc_jump = 100.0  # 10.0

grad_flag_lrc = -1
min_bt_110um_lrc = 220.0
max_bt_110um_lrc = 300.0


# Fix GFS RH scaling
#
# In the current GFS output, the definition of RH varies between
# 253 and 273 K. At 273 it is with respect to water.
# At 253 it is defined with respect to ice.
# At temperatures in between, it varies linearly.
# This routine attempts to define RH with respect to water for all temps
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def fix_gfs_rh(nwp_rh_prof, nwp_t_prof, nwp_shape):
    nwp_n_lat, nwp_n_lon, nwp_n_levels = nwp_shape
    fixed = np.empty((nwp_n_lat, nwp_n_lon, nwp_n_levels), 'f4')
    for i in prange(nwp_n_lat):
        for j in prange(nwp_n_lon):
            for k in prange(nwp_n_levels):
                if nwp_rh_prof[i, j, k] > 0.0:
                    t = nwp_t_prof[i, j, k]
                    # compute saturation vapor pressures
                    es_water = vapor(t)
                    es_ice = vapor_ice(t)
                    # derive the ice/water weight used in gfs
                    ice_weight = (273.16 - t) / (273.16 - 253.16)
                    ice_weight = min(1.0, max(0.0, ice_weight))
                    # derive es used in original rh definition
                    es = ice_weight * es_ice + (1.0 - ice_weight) * es_water
                    # compute actual e
                    e = nwp_rh_prof[i, j, k] * es / 100.0
                    # compute actual rh with respect to water
                    fixed[i, j, k] = 100.0 * e / es_water
                else:
                    fixed[i, j, k] = nwp_rh_prof[i, j, k]
    return fixed


# ----------------------------------------------------------------------
#  local linear radiative center
# ----------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def local_linear_radiative_center(
        grid_data,  # ch_bt_toa[14]
        max_grad_distance,  # max_lrc_distance = 10
        min_grad_value,  # min_lrc_jump = 0.0
        max_grad_value,  # max_lrc_jump = 100.0
        skip_lrc_mask,  # bad_pixel_mask
        min_grid_data_valid,  # min_bt_110um_lrc = 220.0
        max_grid_data_valid  # max_bt_110um_lrc = 300.0
):
    # --- initialize
    line_idx_lrc = np.full(image_shape, missing_value_int4, 'i4')
    elem_idx_lrc = np.full(image_shape, missing_value_int4, 'i4')

    # ----------------------------------------------------------------------
    # loop through pixels in segment
    # ----------------------------------------------------------------------

    for line_idx in prange(1, image_number_of_lines - 1):
        for elem_idx in prange(1, image_number_of_elements - 1):

            # --- skip data due to mask
            if skip_lrc_mask[line_idx, elem_idx] == sym_yes:
                continue
            # -- check for out of bounds data
            if grid_data[line_idx, elem_idx] > max_grid_data_valid:
                continue

            # -- check for data that already meets lrc criteria
            if grid_data[line_idx, elem_idx] < min_grid_data_valid:
                line_idx_lrc[line_idx, elem_idx] = line_idx
                elem_idx_lrc[line_idx, elem_idx] = elem_idx
                continue

            # --- initialize previous variables
            line_idx_previous = line_idx
            elem_idx_previous = elem_idx

            # --- construct 3x3 array for analysis
            grad_array = (
                    grid_data[line_idx - 1:line_idx + 2, elem_idx - 1:elem_idx + 2] -
                    grid_data[line_idx, elem_idx]
            )

            # --- look for bad data
            # if np.amin(grad_array) == missing_value_real4:
            # if np.isnan(np.amin(grad_array)):
            if np.any(np.isnan(grad_array)):
                continue

            # --- compute local gradients, find strongest gradient
            grad_indices = divmod(np.argmin(grad_array), grad_array.shape[1])

            # --- compute direction
            line_idx_dir = grad_indices[0] - 1
            elem_idx_dir = grad_indices[1] - 1

            # --- check for pixels that are located at  minima/maxima
            if elem_idx_dir == 0 and line_idx_dir == 0:
                line_idx_lrc[line_idx, elem_idx] = line_idx_previous
                elem_idx_lrc[line_idx, elem_idx] = elem_idx_previous
                continue

            # --- on first step, only proceed if gradient magnitude exceeds a threshold
            if abs(grad_array[grad_indices[0], grad_indices[1]]) < min_grad_value:
                continue

            # --- check for going up to steep of a gradient
            if abs(grad_array[grad_indices[0], grad_indices[1]]) > max_grad_value:
                continue

            # ---- go long gradient and check for a reversal or saturation
            for i_point in range(max_grad_distance):

                # # --- compute local gradient, find strongest gradient in 3x3 array and compute direction
                # if i_point == 0:
                #
                #     # --- construct 3x3 array for analysis
                #     grad_array = (
                #             grid_data[line_idx_previous - 1:line_idx_previous + 2,
                #             elem_idx_previous - 1:elem_idx_previous + 2] -
                #             grid_data[line_idx_previous, elem_idx_previous]
                #     )
                #
                #     # --- look for bad data
                #     if np.isnan(np.amin(grad_array)):
                #         break
                #
                #     # --- compute local gradients, find strongest gradient
                #     grad_indices = divmod(np.argmin(grad_array), grad_array.shape[1])
                #
                #     # --- compute direction
                #     line_idx_dir = grad_indices[0] - 1
                #     elem_idx_dir = grad_indices[1] - 1
                #
                #     # --- check for pixels that are located at  minima/maxima
                #     if elem_idx_dir == -1 and line_idx_dir == -1:
                #         line_idx_lrc[line_idx, elem_idx] = line_idx_previous
                #         elem_idx_lrc[line_idx, elem_idx] = elem_idx_previous
                #         break
                #
                #     # --- on first step, only proceed if gradient magnitude exceeds a threshold
                #     if abs(grad_array[grad_indices[0], grad_indices[1]]) < min_grad_value:
                #         break
                #
                #     # --- check for going up to steep of a gradient
                #     if abs(grad_array[grad_indices[0], grad_indices[1]]) > max_grad_value:
                #         break

                # -- select next point on the path
                line_idx_next = line_idx_previous + line_idx_dir
                elem_idx_next = elem_idx_previous + elem_idx_dir

                # --- check for hitting segment boundaries
                if (line_idx_next == 0 or line_idx_next == image_number_of_lines - 1 or
                        elem_idx_next == 0 or elem_idx_next == image_number_of_elements - 1):
                    line_idx_lrc[line_idx, elem_idx] = line_idx_previous
                    elem_idx_lrc[line_idx, elem_idx] = elem_idx_previous
                    break

                # --- check for hitting bad data
                if skip_lrc_mask[line_idx_next, elem_idx_next] == sym_yes:
                    line_idx_lrc[line_idx, elem_idx] = line_idx_previous
                    elem_idx_lrc[line_idx, elem_idx] = elem_idx_previous
                    break

                if grid_data[line_idx_next, elem_idx_next] < min_grid_data_valid:
                    line_idx_lrc[line_idx, elem_idx] = line_idx_next
                    elem_idx_lrc[line_idx, elem_idx] = elem_idx_next
                    break

                # --- store position
                elem_idx_previous = elem_idx_next
                line_idx_previous = line_idx_next

    return line_idx_lrc, elem_idx_lrc


# -------------------------------------------------------------------------------
# --- populate the snow_class array based on all available sources of snow data
# --
# --- input:
# ---  nwp_wat_eqv_snow_depth - water equivalent snow depth from nwp
# ---  nwp_sea_ice_frac - sea ice fraction from nwp
# ---  sst_sea_ice_frac - sea ice fraction from sst data source
# ---  snow_class_ims - high resolution snow class field (highest priority)
# ---  snow_class_global - esa globsnow products (lower priority)
# ---
# --- output:
# ---  snow_class_final - final classification
# ---
# --- symbology:
# ---  1 = sym%no_snow
# ---  2 = sym%sea_ice
# ---  3 = sym%snow
# -------------------------------------------------------------------------------
# def compute_snow_class(snow_class_nwp, snow_class_oisst, snow_class_ims, snow_class_glob, land_class):
#     snow_class_final = missing_value_int1
#
#     finished_flag = 0
#
#     while (finished_flag == 0):
#
#         # --- high res
#         if (read_snow_mask == sym_read_snow_hires and failed_ims_snow_mask_flag == sym_no):
#             snow_class_final = snow_class_ims
#             finished_flag = 1
#
#         # -- globsnow - does not work
#         if (read_snow_mask == sym_read_snow_glob and failed_glob_snow_mask_flag == sym_no):
#             snow_class_final = snow_class_glob
#             finished_flag = 1
#
#         snow_class_final = snow_class_nwp
#
#         # --- overwrite with oisst
#
#         snow_class_final[snow_class_oisst == sym_sea_ice] = snow_class_oisst
#
#         finished_flag = 1
#
#     # -- check for consistnecy of land and snow masks
#
#     snow_class_final[snow_class_final == sym_snow and land_class != sym_land] = sym_sea_ice
#     snow_class_final[snow_class_final == sym_sea_ice and land_class == sym_land] = sym_snow
#
#     # ---- remove snow under certain conditions
#     # -- can't be snow if warm
#     snow_class_final[ch14_bt_toa > 277.0] = sym_no_snow
#     # --- some day-specific tests
#     snow_class_final[ch03_ref_toa < 10.0 and geo_sol_zen < 75.0] = sym_no_snow


# @vectorize(nopython=True, forceobj=False)
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_bayes_mask_sfc_type(
        bad_pixel_mask,
        sfc_land,
        sfc_coast_mask,
        sfc_snow,
        sfc_sfc_type,
        nav_lat,
        nav_lon,
        sst_anal_uni,
        ch_sfc_ems,
):
    bayes_mask_sfc_type = np.empty(image_shape, 'i1')

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if bad_pixel_mask[i, j]:
                bayes_mask_sfc_type[i, j] = missing_value_int1
            else:
                bayes_mask_sfc_type[i, j] = compute_bayes_sfc_type(
                    sfc_land[i, j],
                    sfc_coast_mask[i, j],
                    sfc_snow[i, j],
                    sfc_sfc_type[i, j],
                    nav_lat[i, j],
                    nav_lon[i, j],
                    sst_anal_uni[i, j],
                    ch_sfc_ems[7][i, j],
                )

    return bayes_mask_sfc_type


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_dust_mask(
        bad_pixel_mask,
        sfc_snow, sfc_land,
        ch_bt_toa, ch_bt_toa_clear, ch_ems_tropo,
        bt_ch14_std_3x3,
):
    n_med = 2

    dust_mask_temp = np.full(image_shape, missing_value_int1, 'i1')
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if bad_pixel_mask[i, j] == sym_no:
                # -------------------------------------------------------------------------
                # -- ir dust algorithm
                # -------------------------------------------------------------------------
                if sfc_snow[i, j] == sym_no_snow and sfc_land[i, j] in (sym_deep_ocean, sym_moderate_ocean):
                    dust_mask_temp[i, j] = clavrx_dust_test(
                        ch_bt_toa[11][i, j],
                        ch_bt_toa[14][i, j],
                        ch_bt_toa[15][i, j],
                        ch_bt_toa_clear[14][i, j],
                        ch_bt_toa_clear[15][i, j],
                        bt_ch14_std_3x3[i, j],
                        ch_ems_tropo[14][i, j]
                    )

    dust_mask = np.empty(image_shape, 'i1')
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            i1 = max(0, i - n_med)
            i2 = min(image_number_of_lines, i + n_med + 1)
            j1 = max(0, j - n_med)
            j2 = min(image_number_of_elements, j + n_med + 1)
            if np.any(dust_mask_temp[i1:i2, j1:j2] != missing_value_int1):
                dust_mask[i, j] = floor(np.mean(dust_mask_temp[i1:i2, j1:j2] == sym_yes) + 0.5)
            else:
                dust_mask[i, j] = missing_value_int1

    return dust_mask


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_smoke_mask(
        bad_pixel_mask,
        sfc_snow, sfc_land,
        geo_sol_zen,
        ch_ref_toa, ch_ref_toa_clear, ch_bt_toa, ch_bt_toa_clear, ch_ems_tropo,
        ref_ch03_std_3x3, bt_ch14_std_3x3,
):
    n_med = 2

    smoke_mask_temp = np.full(image_shape, missing_value_int1, 'i1')
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if bad_pixel_mask[i, j] == sym_no:
                # ----------------------------------------------------------------------------
                # clavr-x smoke test
                #
                # day and ice-free ocean only
                #
                # -------------------------------------------------------------------------
                if sfc_snow[i, j] == sym_no_snow:
                    if sfc_land[i, j] in (sym_deep_ocean, sym_moderate_ocean):
                        smoke_mask_temp[i, j] = clavrx_smoke_over_water_test(
                            ch_ref_toa[3][i, j], ch_ref_toa_clear[3][i, j],
                            ch_ref_toa[5][i, j],
                            ch_ref_toa[7][i, j], ch_ref_toa_clear[7][i, j],
                            ch_bt_toa[7][i, j],
                            ch_bt_toa[14][i, j], ch_bt_toa_clear[14][i, j],
                            ch_bt_toa[15][i, j], ch_bt_toa_clear[15][i, j],
                            ch_ems_tropo[14][i, j],
                            ref_ch03_std_3x3[i, j], bt_ch14_std_3x3[i, j],
                            geo_sol_zen[i, j]
                        )
                    elif sfc_land[i, j] == sym_land:
                        smoke_mask_temp[i, j] = clavrx_smoke_over_land_test(
                            ch_ref_toa[3][i, j], ch_ref_toa_clear[3][i, j],
                            ch_ref_toa[4][i, j],
                            ch_ref_toa[26][i, j],
                            ch_ref_toa[5][i, j],
                            ch_ref_toa[7][i, j], ch_ref_toa_clear[7][i, j],
                            ch_bt_toa[7][i, j],
                            ch_bt_toa[14][i, j], ch_bt_toa_clear[14][i, j],
                            ch_bt_toa[15][i, j], ch_bt_toa_clear[15][i, j],
                            ch_ems_tropo[14][i, j],
                            ref_ch03_std_3x3[i, j], bt_ch14_std_3x3[i, j],
                            geo_sol_zen[i, j]
                        )

    smoke_mask = np.empty(image_shape, 'i1')
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            i1 = max(0, i - n_med)
            i2 = min(image_number_of_lines, i + n_med + 1)
            j1 = max(0, j - n_med)
            j2 = min(image_number_of_elements, j + n_med + 1)
            if np.any(smoke_mask_temp[i1:i2, j1:j2] != missing_value_int1):
                smoke_mask[i, j] = floor(np.mean(smoke_mask_temp[i1:i2, j1:j2] == sym_yes) + 0.5)
            else:
                smoke_mask[i, j] = missing_value_int1

    return smoke_mask


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_fire_mask(
        bad_pixel_mask,
        sfc_land, sfc_sfc_type,
        geo_sol_zen,
        ch_bt_toa,
        bt_ch07_std_3x3, bt_ch14_std_3x3,
):
    fire_mask = np.full(image_shape, missing_value_int1, 'i1')

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            # --- check for valid data
            if bad_pixel_mask[i, j] == sym_no:
                # -----------------------------------------------------------------------------
                # eumetsat fire algorithm
                # -----------------------------------------------------------------------------
                if sfc_land[i, j] == sym_land and (sfc_sfc_type[i, j] not in (sym_bare_sfc, sym_open_shrubs_sfc)):
                    fire_mask[i, j] = eumetsat_fire_test(
                        ch_bt_toa[14][i, j],
                        ch_bt_toa[7][i, j],
                        bt_ch14_std_3x3[i, j],
                        bt_ch07_std_3x3[i, j],
                        geo_sol_zen[i, j]
                    )

    return fire_mask


@vectorize(nopython=True, forceobj=False)
def determine_sfc_type_forward_model(
        bad_pixel_mask,
        surface_type,
        snow_class,
        latitude,
        ch07_surface_emissivity
):
    if bad_pixel_mask:
        return missing_value_int1

    if surface_type == sym_water_sfc:
        sfc_type_forward_model = 0
    else:
        sfc_type_forward_model = 1  # land

    if snow_class == sym_snow and latitude > -60.0:
        sfc_type_forward_model = 2  # snow

    if (surface_type != sym_water_sfc and snow_class == sym_no_snow and
            ch07_surface_emissivity > 0.90 and abs(latitude) < 60.0):
        sfc_type_forward_model = 3  # desert

    if snow_class == sym_sea_ice and latitude > -60.0:
        sfc_type_forward_model = 4  # arctic

    if snow_class != sym_no_snow and latitude < -60.0:
        sfc_type_forward_model = 5  # antarctic

    return sfc_type_forward_model


def compute(
        time_utc,
        nav_lon, nav_lat,
        ch_rad_toa, ch_ref_toa, ch_bt_toa,
        nwp_n_lat, nwp_n_lon, nwp_n_levels,
        nwp_p_std,
        nwp_sea_ice_frac,
        nwp_p_sfc,
        nwp_t_sfc,
        nwp_z_sfc,
        nwp_t_air,
        nwp_rh_sfc,
        nwp_weasd,
        nwp_tpw,
        nwp_t_tropo,
        nwp_p_tropo,
        nwp_u_wnd_10m,
        nwp_v_wnd_10m,
        nwp_to3,
        nwp_t_prf,
        nwp_z_prf,
        nwp_o3mr_prf,
        nwp_rh_prf,
        nwp_clwmr_prf,
        debug=False
):
    # ds = xr.open_dataset(
    #     'gfs.t12z.pgrb2.0p50.f012', engine='cfgrib', backend_kwargs={
    #         'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'clwmr', 'edition': 2}
    #     }
    # )
    # ds = xr.open_dataset(
    #     'gfs.t12z.pgrb2.0p50.f012', engine='cfgrib', backend_kwargs={
    #         'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh', 'edition': 2}
    #     }
    # )
    # ds = xr.open_dataset(
    #     'gfs.t12z.pgrb2.0p50.f012', engine='cfgrib', backend_kwargs={
    #         'indexpath': '', 'filter_by_keys': {'typeOfLevel': 'isobaricInhPa', 'shortName': 'o3mr', 'edition': 2}
    #     }
    # )
    # dcomp_modes = [3]

    # time_utc = datetime(2021, 10, 7, 4)
    # time_utc = datetime(2021, 10, 7, 12)
    # time_utc = datetime(2021, 7, 15, 12)
    # time_utc = datetime(2021, 10, 25, 12)
    # time_utc = datetime(2021, 10, 25, 20)

    tic = time.time()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--debug', action='store_true', help='debug mode.')
    # args = parser.parse_args()
    # debug = args.debug

    print(f'{debug = }')

    # surface = SurfaceData('auxiliary')
    # surface_ds = surface.get_dataset()
    #
    # print(surface_ds, '\n')
    #
    # sfc_ds_lon = surface_ds.coords['longitude'].values
    # sfc_ds_lat = surface_ds.coords['latitude'].values
    #
    # surface_elev = surface_ds['surface_elevation'].values
    # coast_mask = surface_ds['coast_mask'].values
    # sfc_type = surface_ds['surface_type'].values
    # land_mask = surface_ds['land_sea_mask'].values

    # 用于计算mountain_flag
    # 有缺失值-999.0 最近邻
    with xr.open_dataset('static/GLOBE_1km_digelev.nc', engine='netcdf4', decode_cf=False) as surface_elev_ds:
        surface_elev = surface_elev_ds['surface_elevation'].values
        surface_elev = np.where(surface_elev != -999.0, surface_elev, 0)
        print(surface_elev_ds, '\n')

    # 离散值 最近邻
    # sym_no_coast = 0
    # sym_coast_1km = 1
    # sym_coast_2km = 2
    # sym_coast_3km = 3
    # sym_coast_4km = 4
    # sym_coast_5km = 5
    # sym_coast_6km = 6
    # sym_coast_7km = 7
    # sym_coast_8km = 8
    # sym_coast_9km = 9
    # sym_coast_10km = 10
    with xr.open_dataset('static/coast_mask_1km.nc', engine='netcdf4', decode_cf=False) as coast_mask_ds:
        coast_mask = coast_mask_ds['coast_mask'].values
        print(coast_mask_ds, '\n')

    # 离散值 最近邻
    # sym_water_sfc = 0
    # sym_evergreen_needle_sfc = 1
    # sym_evergreen_broad_sfc = 2
    # sym_deciduous_needle_sfc = 3
    # sym_deciduous_broad_sfc = 4
    # sym_mixed_forests_sfc = 5
    # sym_woodlands_sfc = 6
    # sym_wooded_grass_sfc = 7
    # sym_closed_shrubs_sfc = 8
    # sym_open_shrubs_sfc = 9
    # sym_grasses_sfc = 10
    # sym_croplands_sfc = 11
    # sym_bare_sfc = 12
    # sym_urban_sfc = 13
    with xr.open_dataset('static/gl-latlong-1km-landcover.nc', engine='netcdf4',
                         decode_cf=False) as sfc_type_ds:
        sfc_type = sfc_type_ds['surface_type'].values
        print(sfc_type_ds, '\n')

    # 离散值 最近邻
    # sym_shallow_ocean = 0
    # sym_land = 1
    # sym_coastline = 2
    # sym_shallow_inland_water = 3
    # sym_ephemeral_water = 4
    # sym_deep_inland_water = 5
    # sym_moderate_ocean = 6
    # sym_deep_ocean = 7
    with xr.open_dataset('static/lw_geo_2001001_v03m.nc', engine='netcdf4', decode_cf=False) as land_mask_ds:
        land_mask = land_mask_ds['land_sea_mask'].values
        print(land_mask_ds, '\n')

    sfc_ds_lon = (np.arange(43200, dtype='f4') - 21599.5) / 120
    sfc_ds_lat = (-np.arange(21600, dtype='f4') + 10799.5) / 120

    julian_day_str_map = {
        1: '001', 2: '032', 3: '060', 4: '091', 5: '121', 6: '152',
        7: '182', 8: '213', 9: '244', 10: '274', 11: '305', 12: '335'
    }
    with xr.open_dataset(
            f'static/global_emiss_intABI_2005{julian_day_str_map[time_utc.month]}.nc', engine='netcdf4'
    ) as ems_ds:  # ems有缺失值 最近插值
        ems_ds_lon = ems_ds['lon'].values[0, :]
        ems_ds_lat = ems_ds['lat'].values[:, 0]

    print(ems_ds, '\n')

    day_string = f'{(time_utc.timetuple().tm_yday - 1) // 16 * 16 + 1:0>3}'

    # 有缺失值 最近邻
    with xr.open_dataset(
            f'static/AlbMap.WS.c004.v2.0.00-04.{day_string}.0.659_x4.hdf',
            engine='netcdf4'
    ) as modis_alb_0_66_ds:
        modis_alb_0_66 = modis_alb_0_66_ds['Albedo_Map_0.659'].values
    with xr.open_dataset(
            f'static/AlbMap.WS.c004.v2.0.00-04.{day_string}.0.858_x4.hdf',
            engine='netcdf4'
    ) as modis_alb_0_86_ds:
        modis_alb_0_86 = modis_alb_0_86_ds['Albedo_Map_0.858'].values
    with xr.open_dataset(
            f'static/AlbMap.WS.c004.v2.0.00-04.{day_string}.1.64_x4.hdf',
            engine='netcdf4'
    ) as modis_alb_1_64_ds:
        modis_alb_1_64 = modis_alb_1_64_ds['Albedo_Map_1.64'].values

    modis_alb_lon = (np.arange(5400, dtype='f4') - 2699.5) / 15
    modis_alb_lat = (-np.arange(2700, dtype='f4') + 1349.5) / 15

    nwp_wnd_spd_10m = wind_speed(nwp_u_wnd_10m, nwp_v_wnd_10m)

    bad_nwp_mask = qc_nwp(nwp_p_tropo, nwp_t_tropo, nwp_z_sfc, nwp_p_sfc, nwp_t_sfc)

    if debug:
        show0('bad_nwp_mask', bad_nwp_mask)

    nwp_sfc_level, nwp_tropo_level, nwp_inversion_level, nwp_z_tropo = compute_nwp_levels_segment(
        nwp_p_std, nwp_p_sfc, nwp_p_tropo, nwp_z_prf,
        nwp_t_prf,
        (nwp_n_lat, nwp_n_lon, nwp_n_levels),
    )

    (
        nwp_lcl_hgt,
        nwp_ccl_hgt,
        nwp_lfc_hgt,
        nwp_el_hgt,
    ) = compute_nwp_level_height(
        nwp_t_air,
        nwp_rh_sfc,
        nwp_sfc_level,
        nwp_tropo_level,
        nwp_p_std,
        nwp_t_prf,
        nwp_z_prf,
        (nwp_n_lat, nwp_n_lon, nwp_n_levels),
    )

    if debug:
        show0('nwp_sfc_level', nwp_sfc_level)
        show0('nwp_tropo_level', nwp_tropo_level)

    nwp_cwp = compute_segment_nwp_cloud_parameters(
        nwp_n_lat, nwp_n_lon,
        nwp_tropo_level,
        nwp_sfc_level,
        nwp_clwmr_prf,
        nwp_t_prf,
        nwp_p_std,
    )

    geo_altitude = 35786.0
    sensor_geo_sub_satellite_longitude = 140.7
    sensor_geo_sub_satellite_latitude = 0.0

    # todo 存疑
    # nwp_lon = (np.arange(720.0) + 1.0) * 0.5
    nwp_lon = (np.arange(720.0, dtype='f4')) * 0.5
    nwp_lat = np.arange(361.0, dtype='f4') * 0.5 - 90.0

    sun_earth_distance = 1.0 - 0.016729 * cos(radians(0.9856 * (time_utc.timetuple().tm_yday - 4.0)))

    geo_sol_zen = calculate_solar_zenith(
        time_utc.timetuple().tm_yday, time_utc.hour + time_utc.minute / 60.0, nav_lon, nav_lat
    )
    geo_sol_azi = calculate_solar_azimuth(
        time_utc.timetuple().tm_yday, time_utc.hour + time_utc.minute / 60.0, nav_lon, nav_lat
    )
    # 可以对比验证一下
    # get_sensor_angles(nav_lon, nav_lat, alt, sub_lon, sat_dis, polr_radius, proj_param1, proj_param2)
    geo_sat_zen = calculate_sensor_zenith(
        geo_altitude, sensor_geo_sub_satellite_longitude, sensor_geo_sub_satellite_latitude, nav_lon, nav_lat
    )
    geo_sat_azi = calculate_sensor_azimuth(
        sensor_geo_sub_satellite_longitude, sensor_geo_sub_satellite_latitude, nav_lon, nav_lat
    )
    geo_rel_azi = calculate_relative_azimuth(geo_sol_azi, geo_sat_azi)
    geo_glint_zen = calculate_glint_angle(geo_sol_zen, geo_sat_zen, geo_rel_azi)
    geo_scatter_zen = calculate_scattering_angle(geo_sol_zen, geo_sat_zen, geo_rel_azi)

    if debug:
        show0('geo_sol_zen', geo_sol_zen)
        show0('geo_sol_azi', geo_sol_azi)
        show0('geo_sat_zen', geo_sat_zen)
        show0('geo_sat_azi', geo_sat_azi)
        show0('geo_rel_azi', geo_rel_azi)
        show0('geo_glint_zen', geo_glint_zen)
        show0('geo_scatter_zen', geo_scatter_zen)

    for c in solar_channels:
        ch_ref_toa[c][geo_sol_zen >= 90.0] = nan

    # space_mask = np.zeros(shape, 'b1')
    # space_mask[np.isnan(nav_lon)] = True
    # space_mask[np.isnan(nav_lat)] = True
    space_mask = np.logical_or(np.isnan(nav_lat), np.isnan(nav_lon))
    space_mask[nav_lat < nav_lat_min_limit] = True
    space_mask[nav_lat > nav_lat_max_limit] = True
    space_mask[nav_lon < nav_lon_min_limit] = True
    space_mask[nav_lon > nav_lon_max_limit] = True
    space_mask[geo_sat_zen < geo_sat_zen_min_limit] = True
    space_mask[geo_sat_zen > geo_sat_zen_max_limit] = True
    space_mask[geo_sol_zen < geo_sol_zen_min_limit] = True
    space_mask[geo_sol_zen > geo_sol_zen_max_limit] = True

    if debug:
        show0('space_mask', space_mask)

    bad_pixel_mask = np.zeros(image_shape, 'b1')
    bad_pixel_mask[space_mask] = True
    bad_pixel_mask[np.isnan(geo_sat_zen)] = True
    bad_pixel_mask[np.isnan(geo_sol_zen)] = True
    bad_pixel_mask[np.isnan(geo_rel_azi)] = True
    bad_pixel_mask[(ch_bt_toa[14] - ch_bt_toa[15]) > 20.0] = True
    bad_pixel_mask[ch_bt_toa[14] < 150.0] = True
    bad_pixel_mask[ch_bt_toa[14] > 350.0] = True
    bad_pixel_mask[np.isnan(ch_bt_toa[14])] = True

    bad_pixel_mask[geo_sol_zen < geo_sol_zen_min_limit] = True
    bad_pixel_mask[geo_sol_zen > geo_sol_zen_max_limit] = True

    # nwp_interpolator = GlobalNearestInterpolator(nwp_lon, nwp_lat, nav_lon, nav_lat)
    nwp_interpolator = GlobalLinearInterpolator(nwp_lon, nwp_lat, nav_lon, nav_lat)

    # bad_pixel_mask[nwp_interpolator.interp(bad_nwp_mask)] = True
    bad_pixel_mask[nwp_interpolator.nearest_interp(bad_nwp_mask)] = True

    # number_bad_pixels_thresh = 4950  # 0.9 * image_number_of_elements
    # bad_pixel_mask[np.sum(bad_pixel_mask, axis=1) > number_bad_pixels_thresh] = True
    if debug:
        show0('bad_pixel_mask', bad_pixel_mask)

    # if np.any(bad_pixel_mask ^ space_mask):
    #     print('not equal')
    #     breakpoint()

    # todo compute_pixel_nwp_parameters(smooth_nwp_flag)

    nwp_pix_t_sfc = nwp_interpolator.interp2(nwp_t_sfc)
    nwp_pix_t_tropo = nwp_interpolator.interp2(nwp_t_tropo)
    nwp_pix_z_tropo = nwp_interpolator.interp2(nwp_z_tropo)
    nwp_pix_p_tropo = nwp_interpolator.interp2(nwp_p_tropo)
    nwp_pix_t_air = nwp_interpolator.interp2(nwp_t_air)
    nwp_pix_rh_sfc = nwp_interpolator.interp2(nwp_rh_sfc)
    nwp_pix_p_sfc = nwp_interpolator.interp2(nwp_p_sfc)
    nwp_pix_weasd = nwp_interpolator.interp2(nwp_weasd)
    nwp_pix_sea_ice_frac = nwp_interpolator.interp2(nwp_sea_ice_frac)
    nwp_pix_tpw = nwp_interpolator.interp2(nwp_tpw)
    nwp_pix_to3 = nwp_interpolator.interp2(nwp_to3)
    nwp_pix_cwp = nwp_interpolator.interp2(nwp_cwp)
    nwp_pix_wnd_spd_10m = nwp_interpolator.interp2(nwp_wnd_spd_10m)
    nwp_pix_lcl_hgt = nwp_interpolator.interp2(nwp_lcl_hgt)
    nwp_pix_ccl_hgt = nwp_interpolator.interp2(nwp_ccl_hgt)

    nwp_pix_sfc_level = nwp_interpolator.nearest_interp(nwp_sfc_level)
    nwp_pix_tropo_level = nwp_interpolator.nearest_interp(nwp_tropo_level)
    nwp_pix_inversion_level = nwp_interpolator.nearest_interp(nwp_inversion_level)

    # todo interp or interp2
    nwp_pix_z_sfc = nwp_interpolator.interp2(nwp_z_sfc)

    nwp_pix_t_prf = nwp_interpolator.interp(nwp_t_prf)
    nwp_pix_rh_prf = nwp_interpolator.interp(nwp_rh_prf)
    nwp_pix_z_prf = nwp_interpolator.interp(nwp_z_prf)
    nwp_pix_o3mr_prf = nwp_interpolator.interp(nwp_o3mr_prf)

    # x_index, y_index = np.meshgrid(
    #     np.arange(image_number_of_elements),
    #     np.arange(image_number_of_lines)
    # )
    #
    # todo
    # nwp_pix_y_index = nwp_interpolator.nearest_interp(y_index)
    # nwp_pix_x_index = nwp_interpolator.nearest_interp(x_index)

    if debug:
        show0('nwp_pix_t_sfc', nwp_pix_t_sfc)
        show0('nwp_pix_t_tropo', nwp_pix_t_tropo)
        show0('nwp_pix_z_tropo', nwp_pix_z_tropo)
        show0('nwp_pix_p_tropo', nwp_pix_p_tropo)
        show0('nwp_pix_t_air', nwp_pix_t_air)
        # show0('nwp_pix_rh', nwp_pix_rh)
        # show0('nwp_pix_p_sfc', nwp_pix_p_sfc)
        # show0('nwp_pix_weasd', nwp_pix_weasd)
        # show0('nwp_pix_sea_ice_frac', nwp_pix_sea_ice_frac)
        show0('nwp_pix_tpw', nwp_pix_tpw)
        show0('nwp_pix_wnd_spd_10m', nwp_pix_wnd_spd_10m)
        show0('nwp_pix_ccl_height', nwp_pix_ccl_hgt)

    ems_interpolator = GlobalNearestInterpolator(ems_ds_lon, ems_ds_lat, nav_lon, nav_lat)

    ch_sfc_ems = Dict.empty(i1, f4[:, :])
    for c in thermal_channels:
        ch_sfc_ems[c] = ems_interpolator.interp(ems_ds[f'emiss{c}'].values)
        ch_sfc_ems[c][ch_sfc_ems[c] < 0.0] = 0.99
        if debug:
            show0(f'ch_sfc_ems[{c}]', ch_sfc_ems[c])

    # todo 读的数据比较大，可以加速一下
    # 坐标系不一样,颠倒一下
    surface_interpolator = GlobalNearestInterpolator(sfc_ds_lon, sfc_ds_lat, nav_lon, nav_lat)

    # surface_type
    sfc_sfc_type = surface_interpolator.interp(sfc_type, missing_value_int1)

    # surface_elevation
    sfc_z_sfc_hires = surface_interpolator.interp(surface_elev.astype('f4'))
    sfc_z_sfc = sfc_z_sfc_hires.copy()
    sfc_coast = surface_interpolator.interp(coast_mask, missing_value_int1)

    # sfc_z_sfc = merge_nwp_hires_z_sfc(space_mask, sfc_z_sfc_hires, nwp_z_sfc, nwp_interpolator)

    # land_class
    sfc_land = surface_interpolator.interp(land_mask, missing_value_int1)

    modify_land_class_with_ndvi(bad_pixel_mask, geo_sol_zen, ch_ref_toa, sfc_land)

    if debug:
        show0('sfc_sfc_type', sfc_sfc_type)
        show0('sfc_z_sfc_hires', sfc_z_sfc_hires)
        show0('sfc_coast', sfc_coast)
        show0('sfc_land', sfc_land)

    sfc_land_mask, sfc_coast_mask = compute_binary_land_coast_masks(
        bad_pixel_mask, sfc_land, sfc_sfc_type, sfc_coast
    )
    if debug:
        show0('sfc_land_mask', sfc_land_mask)
        show0('sfc_coast_mask', sfc_coast_mask)

    sst_anal = nwp_pix_t_sfc
    sst_anal_uni = np.full(image_shape, nan, 'f4')

    sfc_nwp_snow = compute_snow_class_nwp(nwp_pix_weasd, nwp_pix_sea_ice_frac)

    # todo 没有用？
    sfc_z_sfc_hires[space_mask] = nan
    sfc_coast[space_mask] = missing_value_int1
    sfc_land[space_mask] = missing_value_int1
    # sfc_snow_ims[space_mask] = missing_value_int1
    # sfc_snow_oisst[space_mask] = missing_value_int1
    sfc_sfc_type[space_mask] = missing_value_int1

    modis_alb_interpolator = GlobalNearestInterpolator(modis_alb_lon, modis_alb_lat, nav_lon, nav_lat)

    ch_sfc_ref_white_sky_map = {
        3: modis_alb_0_66.astype('f4'),
        4: modis_alb_0_86.astype('f4'),
        5: modis_alb_1_64.astype('f4'),
        # 6: modis_alb_2_13.astype('f4')
    }
    ref_sfc_white_sky_water = 5.0

    ch_sfc_ref_white_sky = Dict.empty(i1, f4[:, :])
    for c, v in ch_sfc_ref_white_sky_map.items():
        ch_sfc_ref_white_sky[c] = modis_alb_interpolator.interp(v)
        ch_sfc_ref_white_sky[c][ch_sfc_ref_white_sky[c] == 32767] = nan
        ch_sfc_ref_white_sky[c] *= 0.1
        ch_sfc_ref_white_sky[c] *= 1.1
        ch_sfc_ref_white_sky[c][sfc_land_mask == sym_no] = ref_sfc_white_sky_water
        if debug:
            show0(f'ch_sfc_ref_white_sky[{c}]', ch_sfc_ref_white_sky[c])

    # todo cx_sfc_ems_populate_ch

    bad_pixel_mask[sfc_sfc_type < 0] = True
    # todo 应该是13吧 没什么用
    bad_pixel_mask[sfc_sfc_type > 15] = True
    # bad_pixel_mask[sfc_sfc_type > 13] = True

    geo_cos_zen = np.cos(np.radians(geo_sat_zen))
    # geo_sec_zen = 1.0 / geo_cos_zen
    geo_cos_sol_zen = np.cos(np.radians(geo_sol_zen))

    factor = 1.0 / geo_cos_sol_zen
    # todo ???
    factor *= (sun_earth_distance ** 2)

    for i in solar_channels:
        ch_ref_toa[i] *= factor
        # ch_ref_toa[i][geo_cos_sol_zen > 0.0] *= factor[geo_cos_sol_zen > 0.0]
        # ch_ref_toa[i] = np.where(geo_sol_zen > 0.0, ch_ref_toa[i] * factor, nan)

    terminator_reflectance_sol_zen_thresh = 60.0

    ch_ref_toa[3][geo_sol_zen > terminator_reflectance_sol_zen_thresh] = term_refl_norm(
        geo_cos_sol_zen[geo_sol_zen > terminator_reflectance_sol_zen_thresh],
        ch_ref_toa[3][geo_sol_zen > terminator_reflectance_sol_zen_thresh]
    )

    if debug:
        show0('ch_ref_toa[3]', ch_ref_toa[3])

    ch_ref_toa[7], ems_ch07 = ch07_pseudo_reflectance(
        solar_ch07_nu, geo_cos_sol_zen, ch_rad_toa[7], ch_bt_toa[14], sun_earth_distance, bad_pixel_mask
    )
    if debug:
        show0('ch_ref_toa[7]', ch_ref_toa[7])
        show0('ems_ch07', ems_ch07)

    (
        ref_ch03_min_3x3, ref_ch03_std_3x3,
        bt_ch07_std_3x3, bt_ch14_std_3x3, bt_ch09_max_3x3, bt_ch14_max_3x3, bt_ch14_min_3x3
    ) = compute_min_max_mean_std_metrics(
        bad_pixel_mask,
        ch_ref_toa, ch_sfc_ref_white_sky, ch_bt_toa,
    )

    if debug:
        show0('ref_ch03_min_3x3', ref_ch03_min_3x3)
        show0('ref_ch03_std_3x3', ref_ch03_std_3x3)
        show0('bt_ch07_std_3x3', bt_ch07_std_3x3)
        show0('bt_ch14_std_3x3', bt_ch14_std_3x3)
        show0('bt_ch09_max_3x3', bt_ch09_max_3x3)
        show0('bt_ch14_max_3x3', bt_ch14_max_3x3)
        show0('bt_ch14_min_3x3', bt_ch14_min_3x3)

    covar_ch09_ch14_5x5 = compute_spatial_correlation_arrays(bad_pixel_mask, ch_bt_toa)

    if debug:
        show0('covar_ch09_ch14_5x5', covar_ch09_ch14_5x5)

    ems_ch07_median_3x3, ems_ch07_std_median_3x3 = compute_median_segment(
        ems_ch07, bad_pixel_mask, 1
    )

    if debug:
        show0('ems_ch07_median_3x3', ems_ch07_median_3x3)
        show0('ems_ch07_std_median_3x3', ems_ch07_std_median_3x3)

    # todo ch_ems_tropo尚无数据
    sfc_snow = compute_snow_class(
        sfc_nwp_snow,
        sfc_land,
        ch_ref_toa, ch_bt_toa,  # ch_ems_tropo,
        bt_ch14_std_3x3,  # 新版名字叫ch14_bt_toa_std_3x3,
        nwp_pix_t_sfc, geo_sol_zen,
    )

    if debug:
        show0('sfc_snow', sfc_snow)

    ch_sfc_ems = cx_sfc_ems_correct_for_sfc_type(
        ch_sfc_ems,
        bad_pixel_mask, geo_sat_zen,
        sfc_sfc_type, sfc_snow, sfc_land,
        nwp_pix_wnd_spd_10m, nwp_pix_t_sfc,
    )

    if debug:
        for c in thermal_channels:
            show0(f'ch_sfc_ems[{c}]', ch_sfc_ems[c])

    nwp_pix_t_sfc = modify_nwp_pix_t_sfc(
        bad_pixel_mask,
        nwp_pix_z_sfc,
        nwp_pix_t_sfc,
        nwp_pix_sfc_level,
        sfc_land,
        sfc_z_sfc,
        sfc_land_mask,
        sfc_snow,
        sst_anal,
        nwp_pix_t_prf,
        nwp_pix_z_prf,
    )

    if debug:
        show0('nwp_pix_t_sfc', nwp_pix_t_sfc)

    (
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
    ) = get_pixel_nwp_rtm(
        bad_pixel_mask,
        space_mask,

        nwp_n_levels,
        nwp_p_std,
        nwp_pix_t_prf,
        nwp_pix_rh_prf,
        nwp_pix_sfc_level,
        nwp_pix_z_sfc,
        nwp_pix_t_air,
        nwp_pix_rh_sfc,
        nwp_pix_p_sfc,
        nwp_pix_z_prf,
        nwp_pix_o3mr_prf,
        nwp_pix_p_tropo,

        p_std,

        coef_dry, coef_ozon, coef_wvp_cont, coef_wvp_solid, coef_wvp_liquid,

        geo_cos_zen, geo_cos_sol_zen,
        sfc_land, sfc_z_sfc, nwp_pix_t_sfc,
        ch_sfc_ems, ch_rad_toa,

    )

    if debug:
        for c in thermal_channels:
            show0(f'ch_rad_toa_clear[{c}]', ch_rad_toa_clear[c])
            show0(f'ch_bt_toa_clear[{c}]', ch_bt_toa_clear[c])
        for c in np.array([14, 15, 16], 'i1'):
            show0(f'ch_ems_tropo[{c}]', ch_ems_tropo[c])

        show0('beta_110um_120um_tropo_rtm', beta_110um_120um_tropo_rtm)
        show0('beta_110um_133um_tropo_rtm', beta_110um_133um_tropo_rtm)

    pc_opaque_cloud, zc_opaque_cloud, tc_opaque_cloud = opaque_cloud_height(
        bad_pixel_mask,
        rtm_tropo_level, rtm_sfc_level, rtm_z_prof, rtm_t_prof,
        rtm_ch_rad_bb_cloud_profile, ch_rad_toa, ch_bt_toa
    )

    if debug:
        show0('pc_opaque_cloud', pc_opaque_cloud)
        show0('zc_opaque_cloud', zc_opaque_cloud)
        show0('tc_opaque_cloud', tc_opaque_cloud)

    ch_ref_toa_clear = atmos_corr(
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
    )

    if debug:
        for c in solar_channels:
            show0(f'ch_ref_toa_clear[{c}]', ch_ref_toa_clear[c])

    ch_ref_toa_clear[7], ems_ch07_clear_rtm = ch07_pseudo_reflectance(
        solar_ch07_nu, geo_cos_sol_zen, ch_rad_toa_clear[7], ch_bt_toa_clear[14],
        sun_earth_distance, bad_pixel_mask
    )
    if debug:
        show0('ch_ref_toa_clear[7]', ch_ref_toa_clear[7])
        show0('ems_ch07_clear_rtm', ems_ch07_clear_rtm)

    i_lrc, j_lrc = local_linear_radiative_center(
        ch_bt_toa[14],
        max_lrc_distance,
        min_lrc_jump,
        max_lrc_jump,
        bad_pixel_mask,
        min_bt_110um_lrc,
        max_bt_110um_lrc
    )

    if debug:
        show0('i_lrc', i_lrc)
        show0('j_lrc', j_lrc)

    # glint_mask
    sfc_glint_mask = compute_glint(
        geo_glint_zen,
        ch_ref_toa[3],
        ref_ch03_std_3x3,
        bad_pixel_mask,
        sfc_land_mask,
        sfc_snow,
        ch_bt_toa,
        ch_bt_toa_clear,
        geo_sat_zen,
        bt_ch14_std_3x3
    )

    if debug:
        show0('sfc_glint_mask', sfc_glint_mask)

    cld_mask_bayes_mask_sfc_type = compute_bayes_mask_sfc_type(
        bad_pixel_mask,
        sfc_land,
        sfc_coast_mask,
        sfc_snow,
        sfc_sfc_type,
        nav_lat,
        nav_lon,
        sst_anal_uni,
        ch_sfc_ems,
    )
    smoke_mask = compute_smoke_mask(
        bad_pixel_mask,
        sfc_snow, sfc_land,
        geo_sol_zen,
        ch_ref_toa, ch_ref_toa_clear, ch_bt_toa, ch_bt_toa_clear, ch_ems_tropo,
        ref_ch03_std_3x3, bt_ch14_std_3x3,
    )
    dust_mask = compute_dust_mask(
        bad_pixel_mask,
        sfc_snow, sfc_land,
        ch_bt_toa, ch_bt_toa_clear, ch_ems_tropo,
        bt_ch14_std_3x3,
    )
    fire_mask = compute_fire_mask(
        bad_pixel_mask,
        sfc_land, sfc_sfc_type,
        geo_sol_zen,
        ch_bt_toa,
        bt_ch07_std_3x3, bt_ch14_std_3x3,
    )

    cld_mask_posterior_cld_probability = compute_posterior_cld_probability(
        bad_pixel_mask,
        sfc_glint_mask, sfc_coast_mask, sfc_z_sfc,
        geo_sol_zen, geo_scatter_zen, geo_sat_zen, geo_cos_zen,
        nav_lat, nav_lon,
        # ch_sfc_ems,
        ch_ref_toa, ch_ref_toa_clear, ch_bt_toa, ch_bt_toa_clear, ch_ems_tropo,
        ems_ch07_clear_rtm,
        ref_ch03_min_3x3, bt_ch14_std_3x3,
        ems_ch07_median_3x3, covar_ch09_ch14_5x5,
        cld_mask_bayes_mask_sfc_type,
        nwp_pix_t_sfc, nwp_pix_tpw,
        time_utc.month,
        use_prior_table=True,
        use_core_tables=False,
    )

    cld_mask_cld_mask = compute_cld_mask(
        bad_pixel_mask,
        cld_mask_bayes_mask_sfc_type,
        cld_mask_posterior_cld_probability,
    )

    if debug:
        show0('cld_mask_cld_mask', cld_mask_cld_mask)
        show0('cld_mask_posterior_cld_probability', cld_mask_posterior_cld_probability)
        show0('dust_mask', dust_mask)
        show0('smoke_mask', smoke_mask)
        show0('fire_mask', fire_mask)
        show0('cld_mask_bayes_mask_sfc_type', cld_mask_bayes_mask_sfc_type)

    cld_type = compute_cloud_type(
        geo_sol_zen,
        rtm_sfc_level, rtm_tropo_level, rtm_t_prof, rtm_z_prof,
        rtm_ch_rad_bb_cloud_profile,
        ch_rad_toa, ch_ref_toa, ch_bt_toa, ch_rad_toa_clear, ch_ref_toa_clear, ch_bt_toa_clear,
        ch_ems_tropo, ch_sfc_ems,
        bt_ch14_std_3x3, covar_ch09_ch14_5x5, bt_ch09_max_3x3,
        beta_110um_120um_tropo_rtm, beta_110um_133um_tropo_rtm,
        bad_pixel_mask, cld_mask_cld_mask, dust_mask, smoke_mask, fire_mask,
        i_lrc, j_lrc,
    )

    if debug:
        show0('cld_type', cld_type)

    acha_tc, acha_pc, acha_zc, acha_tau, acha_reff = awg_cloud_height_bridge(
        sensor_spatial_resolution_meters,
        bad_pixel_mask,
        nav_lat, nav_lon,
        geo_cos_zen, geo_sat_zen,
        nwp_pix_t_sfc,
        nwp_pix_t_tropo, nwp_pix_z_tropo, nwp_pix_p_tropo,
        sfc_z_sfc, sfc_snow, sfc_sfc_type,
        ch_rad_toa, ch_bt_toa, ch_rad_toa_clear, ch_sfc_ems,
        cld_mask_cld_mask,
        cld_type,
        tc_opaque_cloud,
        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        rtm_ch_rad_atm_profile, rtm_ch_trans_atm_profile, rtm_ch_rad_bb_cloud_profile,
    )

    if debug:
        show0('acha_tc', acha_tc)
        show0('acha_pc', acha_pc)
        show0('acha_zc', acha_zc)

        show0('acha_tau', acha_tau)
        show0('acha_reff', acha_reff)

    acha_alt = compute_altitude_from_pressure(bad_pixel_mask, acha_pc)
    if debug:
        show0('acha_alt', acha_alt)

    # cx_dncomp
    tau_dcomp, reff_dcomp = awg_cloud_dncomp_algorithm(
        bad_pixel_mask,
        # acha_zc,
        acha_tc, acha_pc,
        nwp_pix_to3, nwp_pix_p_sfc,
        nwp_pix_t_prf, nwp_pix_z_prf, nwp_p_std, nwp_pix_tpw_prof,
        nwp_pix_sfc_level, nwp_pix_tropo_level, nwp_pix_inversion_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof, rtm_sfc_level, rtm_tropo_level, rtm_inversion_level,
        rtm_ch_trans_atm_profile, rtm_ch_rad_atm_profile,
        ch_rad_toa_clear,
        geo_sol_zen, geo_sat_zen,
        ch_ref_toa, ch_sfc_ref_white_sky, ch_rad_toa, ch_sfc_ems,
        solar_rtm_tau_h2o_coef,
        sfc_snow, sfc_land_mask,
        cld_type,
        cld_mask_cld_mask, geo_rel_azi,
        sun_earth_distance, solar_ch07_nu,
        rtm_n_levels, nwp_n_levels,
        ch_phase_cld_alb, ch_phase_cld_trn, ch_phase_cld_sph_alb, ch_phase_cld_refl,
        phase_cld_ems, phase_cld_trn_ems,
    )

    cwp_dcomp = compute_cloud_water_path(
        geo_sol_zen, cld_type, tau_dcomp, reff_dcomp
    )

    base_zc_base = cloud_base_bridge(
        bad_pixel_mask,
        sfc_z_sfc,
        cld_type,
        acha_tc, acha_zc, acha_tau,
        nwp_pix_lcl_hgt, nwp_pix_ccl_hgt,
        cwp_dcomp, nwp_pix_cwp,
        rtm_sfc_level,
        rtm_z_prof,
    )

    ccl_cloud_fraction = ccl_bridge(
        sensor_spatial_resolution_meters,
        bad_pixel_mask,
        cld_mask_posterior_cld_probability,
        acha_alt,
    )

    print('ok')
    toc = time.time()
    print(toc - tic)

    if debug:
        clavrx_ds = xr.open_dataset(
            time_utc.strftime(
                '/data/developer_14/clavrx_data/%Y/%m/%d/%H/%M/clavrx_H08_%Y%m%d_%H%M_B01_FLDK_R.level2.nc'),
            engine='netcdf4'
        )

        nav_lon, nav_lat = pix_lin2lon_lat(
            (5500, 5500),
            *np.meshgrid(np.arange(1, 5501, dtype='f4'), np.arange(1, 5501, dtype='f4')),
            140.7,
            20466275,
            20466275,
            2750.5,
            2750.5,
            42164.0,
            1.006739501,
            1737122264.0,
        )

        mask = ~(np.isnan(nav_lon) | np.isnan(nav_lat))
        x = np.column_stack([nav_lon[mask], nav_lat[mask]])

        latitude = np.linspace(30.5, 24.5, 151)
        longitude = np.linspace(108.5, 114.5, 151)

        lon, lat = np.meshgrid(longitude, latitude)
        p = np.column_stack([lon.flat, lat.flat])

        y = clavrx_ds['cld_temp_acha'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        plt.scatter(acha_tc, z)
        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(acha_tc)
        im = ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()

        plt.plot(np.sort((z - acha_tc).flat))
        plt.show()

        y = clavrx_ds['cld_height_acha'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        plt.scatter(acha_zc, z)
        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(acha_zc)
        im = ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()

        plt.plot(np.sort((z - acha_zc).flat))
        plt.show()

        y = clavrx_ds['cld_opd_acha'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        plt.scatter(acha_tau, z)
        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        im = ax0.imshow(acha_tau)
        ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()

        plt.plot(np.sort((z - acha_tau).flat))
        plt.show()

        y = clavrx_ds['cld_opd_dcomp'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        plt.scatter(tau_dcomp, z)
        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(tau_dcomp)
        im = ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()

        plt.plot(np.sort((z - tau_dcomp).flat))
        plt.show()

        y = clavrx_ds['cld_reff_dcomp'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        plt.scatter(reff_dcomp, z)
        plt.show()

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(reff_dcomp)
        im = ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()

        plt.plot(np.sort((z - reff_dcomp).flat))
        plt.show()

        # breakpoint()

        plt.imshow(base_zc_base)
        plt.colorbar()
        plt.show()

        y = clavrx_ds['cloud_fraction'].values[mask]
        interp = NearestNDInterpolator(x, y)
        z = interp(p).reshape((151, 151))

        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax0.imshow(ccl_cloud_fraction)
        im = ax1.imshow(z)
        plt.colorbar(im, ax=(ax0, ax1))
        plt.show()
    else:
        plt.imshow(base_zc_base)
        plt.colorbar()
        plt.show()

        plt.imshow(ccl_cloud_fraction)
        plt.colorbar()
        plt.show()

    return base_zc_base, ccl_cloud_fraction


def get_gfs_data_from_nc(time_utc):
    # gfs_ds = Dataset(time_utc.strftime('/data/developer_14/clavrx_data/%Y/%m/%d/%H/gfs.%y%m%d%H_F012.hdf'))
    gfs = GFSData('/data/developer_14/clavrx_data')
    gfs_ds = gfs.get_dataset(time_utc, 12)
    print(gfs_ds, '\n')

    # nwp_n_lat = np.int32(361)  # gfs_ds.attrs['NUMBER OF LATITUDES']
    # nwp_n_lon = np.int32(720)  # gfs_ds.attrs['NUMBER OF LONGITUDES']
    # nwp_n_levels = np.int32(26)  # gfs_ds.attrs['NUMBER OF PRESSURE LEVELS']
    nwp_n_lat = 361
    nwp_n_lon = 720
    nwp_n_levels = 26
    print(nwp_n_lat, nwp_n_lon, nwp_n_levels, '\n')

    nwp_p_std = gfs_ds['level'].values  # 1d

    nwp_sea_ice_frac = gfs_ds['ice fraction'].values
    nwp_p_sfc = gfs_ds['surface pressure'].values
    nwp_t_sfc = gfs_ds['surface temperature'].values
    nwp_z_sfc = gfs_ds['surface height'].values
    nwp_t_air = gfs_ds['temperature at sigma=0.995'].values
    nwp_rh_sfc = gfs_ds['rh at sigma=0.995'].values
    nwp_weasd = gfs_ds['water equivalent snow depth'].values
    nwp_weasd = np.where(nwp_weasd != 9.999e20, nwp_weasd, nan)
    nwp_tpw = gfs_ds['total precipitable water'].values
    nwp_t_tropo = gfs_ds['tropopause temperature'].values
    nwp_p_tropo = gfs_ds['tropopause pressure'].values
    nwp_u_wnd_10m = gfs_ds['u-wind at sigma=0.995'].values
    nwp_v_wnd_10m = gfs_ds['v-wind at sigma=0.995'].values
    nwp_to3 = gfs_ds['total ozone'].values

    nwp_t_prf = gfs_ds['temperature'].values
    nwp_z_prf = gfs_ds['height'].values
    nwp_o3mr_prf = gfs_ds['o3mr'].values
    nwp_rh_prf = gfs_ds['rh'].values
    nwp_clwmr_prf = gfs_ds['clwmr'].values

    nwp_rh_prf.flags.writeable = True
    nwp_rh_prf = fix_gfs_rh(nwp_rh_prf, nwp_t_prf, (nwp_n_lat, nwp_n_lon, nwp_n_levels))

    # convert ozone from mass mixing ratio(g/g) to volume missing ratio (ppmv)
    nwp_o3mr_prf.flags.writeable = True
    nwp_o3mr_prf[nwp_o3mr_prf > 0] *= 1.0e06 * 0.602

    # convert nwp_z_prof to meters
    nwp_z_sfc.flags.writeable = True
    nwp_z_prf.flags.writeable = True
    nwp_z_sfc *= 1000.0
    nwp_z_prf *= 1000.0

    nwp_t_tropo.flags.writeable = True
    nwp_t_tropo[nwp_t_tropo < 180.0] = 200.0
    nwp_t_tropo[nwp_t_tropo > 240.0] = 200.0

    return (
        nwp_n_lat, nwp_n_lon, nwp_n_levels,
        nwp_p_std,
        nwp_sea_ice_frac,
        nwp_p_sfc,
        nwp_t_sfc,
        nwp_z_sfc,
        nwp_t_air,
        nwp_rh_sfc,
        nwp_weasd,
        nwp_tpw,
        nwp_t_tropo,
        nwp_p_tropo,
        nwp_u_wnd_10m,
        nwp_v_wnd_10m,
        nwp_to3,
        nwp_t_prf,
        nwp_z_prf,
        nwp_o3mr_prf,
        nwp_rh_prf,
        nwp_clwmr_prf,
    )


def latest_initial_time(valid_time):
    if valid_time.hour >= 18:
        initial_time = valid_time.replace(hour=18, minute=0, second=0, microsecond=0)
    elif valid_time.hour >= 12:
        initial_time = valid_time.replace(hour=12, minute=0, second=0, microsecond=0)
    elif valid_time.hour >= 6:
        initial_time = valid_time.replace(hour=6, minute=0, second=0, microsecond=0)
    else:
        initial_time = valid_time.replace(hour=0, minute=0, second=0, microsecond=0)
    return initial_time


def find_ncep_path(valid_time):
    latest = latest_initial_time(valid_time)
    is_initial_time = valid_time == latest

    for i in range(64 + is_initial_time):
        initial_time = latest - timedelta(hours=i * 6)
        step = int((valid_time - initial_time).total_seconds() / 3600.0)
        step = int(step // 3.0 * 3.0)
        path = f'/mnt/ftp5/NAFP/NCEP/GFS/0p50/{initial_time:%Y%m%d/%H/W_NAFP_C_KWBC_%Y%m%d%H%M%S_P_gfs.t%Hz}.pgrb2.0p50.f{step:0>3}.bin'
        if os.path.exists(path):
            return path
    else:
        raise FileNotFoundError(f'NCEP GFS for {valid_time} not exists')


def find_fy4a_path(time):
    path = glob.glob(
        f'/mnt/ftp5/NAFP/FY4A/L1/{time:%Y/%Y%m%d}'
        f'Z_SATE_C_BAWX_*_P_FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_{time:%Y%m%d%H%M%S}_*_4000M_V0001.HDF '
    )[0]
    return path


def get_gfs_data_from_grib(time_utc):
    # for initial_time, step in NCEPGFSDownloader.time_step_pair(time_utc):
    #     path = (
    #         # f'/mnt/ftp5/NAFP/NCEP/GFS/0p50/{initial_time:%Y%m%d/%H/W_NAFP_C_KWBC_%Y%m%d%H%M%S_P_gfs.t%Hz}.pgrb2.0p50.f{step:0>3}.bin'
    #         f'/data/developer_14/ncep_gfs/{initial_time:%Y%m%d/%H/gfs.t%Hz}.pgrb2.0p50.f{step:0>3}'
    #     )
    #
    #     if os.path.exists(path):
    #         break
    # else:
    #     raise FileNotFoundError(f'NCEP GFS for {time} not exists')

    # downloader = NCEPGFSDownloader()
    # for initial_time, step in NCEPGFSDownloader.initial_time_step_pair(time_utc):
    #     path = downloader.download_0p50_most_by_initial_time(
    #         initial_time, step,
    #         f'/data/developer_14/ncep_gfs/{initial_time:%Y%m%d/%H}'
    #     )
    #     if path is not None:
    #         break
    # else:
    #     raise FileNotFoundError(f'NCEP GFS for {time_utc} not exists')

    # for i in range(0, 3, 384):
    #     if time_utc.hour % 3 == 0 and time_utc.minute == 0:
    #
    #     pass
    # # todo
    # path = 'gfs.t12z.pgrb2.0p50.f012'
    # path = 'W_NAFP_C_KWBC_20220712000000_P_gfs.t00z.pgrb2.0p50.f021.bin'

    path = find_ncep_path(time_utc)

    nwp_n_lat = 361
    nwp_n_lon = 720
    nwp_n_levels = 26

    level = np.array([
        10.0, 20.0, 30.0, 50.0, 70.0, 100.0, 150.0, 200.0, 250.0, 300.0,
        350.0, 400.0, 450.0, 500.0, 550.0, 600.0, 650.0, 700.0, 750.0, 800.0,
        850.0, 900.0, 925.0, 950.0, 975.0, 1000.0
    ], dtype='f4')
    latitude = np.arange(361.0, dtype='f4') * 0.5 - 90.0
    longitude = (np.arange(720.0, dtype='f4')) * 0.5

    nwp_p_std = level

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'surface', 'edition': 2}
        }
    ).sel(latitude=latitude, longitude=longitude)
    nwp_sea_ice_frac = ds['siconc'].values
    nwp_p_sfc = ds['sp'].values / 100.0
    nwp_t_sfc = ds['t'].values
    nwp_z_sfc = ds['orog'].values
    nwp_weasd = ds['sdwe'].values

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'sigma', 'edition': 2}
        }
    ).sel(latitude=latitude, longitude=longitude)
    nwp_t_air = ds['t'].values
    nwp_rh_sfc = ds['r'].values
    nwp_u_wnd_10m = ds['u'].values
    nwp_v_wnd_10m = ds['v'].values

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'tropopause', 'edition': 2}
        }
    ).sel(latitude=latitude, longitude=longitude)
    nwp_t_tropo = ds['t'].values
    nwp_p_tropo = ds['trpp'].values / 100.0

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '',
            'filter_by_keys': {'stepType': 'instant', 'typeOfLevel': 'atmosphereSingleLayer', 'edition': 2}
        }
    ).sel(latitude=latitude, longitude=longitude)
    nwp_tpw = ds['pwat'].values / 10.0
    nwp_to3 = ds['tozne'].values

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {
                'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa', 'shortName': 't', 'edition': 2
            }
        }
    ).sel(latitude=latitude, longitude=longitude, isobaricInhPa=level).transpose(
        'latitude', 'longitude', 'isobaricInhPa'
    )
    nwp_t_prf = ds['t'].values

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {
                'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa', 'shortName': 'gh', 'edition': 2
            }
        }
    ).sel(latitude=latitude, longitude=longitude, isobaricInhPa=level).transpose(
        'latitude', 'longitude', 'isobaricInhPa'
    )
    nwp_z_prf = ds['gh'].values

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {
                'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa', 'shortName': 'o3mr', 'edition': 2
            }
        }
    ).sel(latitude=latitude, longitude=longitude, isobaricInhPa=level[:6]).transpose(
        'latitude', 'longitude', 'isobaricInhPa'
    )
    nwp_o3mr_prf = np.concatenate([ds['o3mr'].values, np.zeros((nwp_n_lat, nwp_n_lon, 20), 'f4')], axis=2)

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {
                'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa', 'shortName': 'r', 'edition': 2
            }
        }
    ).sel(latitude=latitude, longitude=longitude, isobaricInhPa=level[5:]).transpose(
        'latitude', 'longitude', 'isobaricInhPa'
    )
    nwp_rh_prf = np.concatenate([np.zeros((nwp_n_lat, nwp_n_lon, 5), 'f4'), ds['r'].values], axis=2)

    ds = xr.open_dataset(
        path, engine='cfgrib', backend_kwargs={
            'indexpath': '', 'filter_by_keys': {
                'stepType': 'instant', 'typeOfLevel': 'isobaricInhPa', 'shortName': 'clwmr', 'edition': 2
            }
        }
    ).sel(latitude=latitude, longitude=longitude, isobaricInhPa=level[5:]).transpose(
        'latitude', 'longitude', 'isobaricInhPa'
    )
    nwp_clwmr_prf = np.concatenate([np.zeros((nwp_n_lat, nwp_n_lon, 5), 'f4'), ds['clwmr'].values], axis=2)
    nwp_rh_prf = fix_gfs_rh(nwp_rh_prf, nwp_t_prf, (nwp_n_lat, nwp_n_lon, nwp_n_levels))

    # convert ozone from mass mixing ratio(g/g) to volume missing ratio (ppmv)
    nwp_o3mr_prf[nwp_o3mr_prf > 0] *= 1.0e06 * 0.602

    nwp_t_tropo[nwp_t_tropo < 180.0] = 200.0
    nwp_t_tropo[nwp_t_tropo > 240.0] = 200.0

    return (
        nwp_n_lat, nwp_n_lon, nwp_n_levels,
        nwp_p_std,
        nwp_sea_ice_frac,
        nwp_p_sfc,
        nwp_t_sfc,
        nwp_z_sfc,
        nwp_t_air,
        nwp_rh_sfc,
        nwp_weasd,
        nwp_tpw,
        nwp_t_tropo,
        nwp_p_tropo,
        nwp_u_wnd_10m,
        nwp_v_wnd_10m,
        nwp_to3,
        nwp_t_prf,
        nwp_z_prf,
        nwp_o3mr_prf,
        nwp_rh_prf,
        nwp_clwmr_prf,
    )


def get_fy4a_data(time_utc, nav_lon, nav_lat):
    # downloader = FY4FTPDownloader('2017111008064NRT', '_XEJWuZ5')
    # downloader.download_agri_l1(time_utc, 'REGC', 4000, f'/data/developer_14/fy4a/REGC/4000M/{time_utc:%Y/%Y%m%d}')

    ch_rad_toa = Dict.empty(i1, f4[:, :])
    ch_ref_toa = Dict.empty(i1, f4[:, :])
    ch_bt_toa = Dict.empty(i1, f4[:, :])

    # fy4a_path = glob.glob(
    #     f'/data/developer_14/fy4a/REGC/4000M/{time_utc:%Y/%Y%m%d}/'
    #     f'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_{time_utc:%Y%m%d%H%M%S}_*_4000M_V0001.HDF'
    # )[0]
    fy4a_path = find_fy4a_path(time_utc)
    agri_l1 = AgriL1.from_file(fy4a_path)

    ch_rad_toa[3] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[2].astype('f4')
    ch_ref_toa[3] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[2].astype('f4')
    ch_rad_toa[4] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[3].astype('f4')
    ch_ref_toa[4] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[3].astype('f4')
    ch_rad_toa[5] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[5].astype('f4')
    ch_ref_toa[5] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[5].astype('f4')

    ch_rad_toa[7] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[7].astype('f4')
    ch_bt_toa[7] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[7].astype('f4')
    ch_rad_toa[9] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[9].astype('f4')
    ch_bt_toa[9] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[9].astype('f4')
    ch_rad_toa[11] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[11].astype('f4')
    ch_bt_toa[11] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[11].astype('f4')
    ch_rad_toa[14] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[12].astype('f4')
    ch_bt_toa[14] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[12].astype('f4')
    ch_rad_toa[15] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[13].astype('f4')
    ch_bt_toa[15] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[13].astype('f4')
    ch_rad_toa[16] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='radiance(NOAA standard)')[14].astype('f4')
    ch_bt_toa[16] = agri_l1.get_data_by_lon_lat(nav_lon, nav_lat, level='phys')[14].astype('f4')

    return ch_rad_toa, ch_ref_toa, ch_bt_toa


def main_real(time_cst, force=False):
    save_path_format = '/data/disk/clouds/realtime/%Y/%m/%d'
    save_path = time_cst.strftime(save_path_format)
    save_name = f'{save_path}/PROJECT_DSJ_{time_cst:%Y%m%d%H%M%S}_TH_CLOUD_MUTLI_BECS_{time_cst:%Y%m%d%H%M%S}_000.nc'
    if not force and os.path.exists(save_name):
        return

    time_utc = time_cst - timedelta(hours=8)

    nav_lon, nav_lat = np.meshgrid(
        np.linspace(nav_lon_min_limit, nav_lon_max_limit, image_number_of_elements),
        np.linspace(nav_lat_max_limit, nav_lat_min_limit, image_number_of_lines),
    )

    ch_rad_toa, ch_ref_toa, ch_bt_toa = get_fy4a_data(time_utc, nav_lon, nav_lat)

    # (
    #     nwp_n_lat, nwp_n_lon, nwp_n_levels,
    #     nwp_p_std,
    #     nwp_sea_ice_frac,
    #     nwp_p_sfc,
    #     nwp_t_sfc,
    #     nwp_z_sfc,
    #     nwp_t_air,
    #     nwp_rh_sfc,
    #     nwp_weasd,
    #     nwp_tpw,
    #     nwp_t_tropo,
    #     nwp_p_tropo,
    #     nwp_u_wnd_10m,
    #     nwp_v_wnd_10m,
    #     nwp_to3,
    #     nwp_t_prf,
    #     nwp_z_prf,
    #     nwp_o3mr_prf,
    #     nwp_rh_prf,
    #     nwp_clwmr_prf,
    # ) = get_gfs_data_from_nc(time_utc)

    (
        nwp_n_lat, nwp_n_lon, nwp_n_levels,
        nwp_p_std,
        nwp_sea_ice_frac,
        nwp_p_sfc,
        nwp_t_sfc,
        nwp_z_sfc,
        nwp_t_air,
        nwp_rh_sfc,
        nwp_weasd,
        nwp_tpw,
        nwp_t_tropo,
        nwp_p_tropo,
        nwp_u_wnd_10m,
        nwp_v_wnd_10m,
        nwp_to3,
        nwp_t_prf,
        nwp_z_prf,
        nwp_o3mr_prf,
        nwp_rh_prf,
        nwp_clwmr_prf,
    ) = get_gfs_data_from_grib(time_utc)

    base_zc_base, ccl_cloud_fraction = compute(
        time_utc,
        nav_lon, nav_lat,
        ch_rad_toa, ch_ref_toa, ch_bt_toa,
        nwp_n_lat, nwp_n_lon, nwp_n_levels,
        nwp_p_std,
        nwp_sea_ice_frac,
        nwp_p_sfc,
        nwp_t_sfc,
        nwp_z_sfc,
        nwp_t_air,
        nwp_rh_sfc,
        nwp_weasd,
        nwp_tpw,
        nwp_t_tropo,
        nwp_p_tropo,
        nwp_u_wnd_10m,
        nwp_v_wnd_10m,
        nwp_to3,
        nwp_t_prf,
        nwp_z_prf,
        nwp_o3mr_prf,
        nwp_rh_prf,
        nwp_clwmr_prf,
        debug=False
    )
    ccl_cloud_fraction *= 100.0

    latitude = np.linspace(30.5, 24.5, 151)
    longitude = np.linspace(108.5, 114.5, 151)
    dimensions = ('time', 'latitude', 'longitude')
    coordinates = {'time': time_cst, 'latitude': latitude, 'longitude': longitude}
    da_cbh = xr.DataArray(
        base_zc_base[np.newaxis, ...],
        coords=coordinates, dims=dimensions,
        attrs={'units': 'm'}
    )
    da_tcc = xr.DataArray(
        ccl_cloud_fraction[np.newaxis, ...],
        coords=coordinates, dims=dimensions,
        attrs={'units': '%'}
    )

    ds_cloud = xr.Dataset({'CBH': da_cbh, 'TCC': da_tcc})

    os.makedirs(save_path, exist_ok=True)
    ds_cloud.to_netcdf(
        save_name,
        encoding={
            'CBH': {
                'dtype': 'int16', 'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': -32768, 'zlib': True
            },
            'TCC': {
                'dtype': 'int16', 'scale_factor': 1.0e-2, 'add_offset': 0.0, '_FillValue': -32768, 'zlib': True
            }
        }
    )


def main_fore(initial_time_cst, force=False):
    save_path_format = '/data/disk/clouds/forecast/%Y/%m/%d'
    save_path = initial_time_cst.strftime(save_path_format)
    save_name = f'{save_path}/PROJECT_DSJ_{initial_time_cst:%Y%m%d%H%M%S}_TH_CLOUD_MUTLI_BECS_{initial_time_cst:%Y%m%d%H%M%S}_00201.nc'
    if not force and os.path.exists(save_name):
        return

    initial_time_utc = initial_time_cst - timedelta(hours=8)

    nav_lon, nav_lat = np.meshgrid(
        np.linspace(nav_lon_min_limit, nav_lon_max_limit, image_number_of_elements),
        np.linspace(nav_lat_max_limit, nav_lat_min_limit, image_number_of_lines),
    )

    time_cst = []
    base_zc_base_list = []
    ccl_cloud_fraction_list = []
    for i in range(1, 4):
        time_utc = initial_time_utc + timedelta(hours=i)

        (
            nwp_n_lat, nwp_n_lon, nwp_n_levels,
            nwp_p_std,
            nwp_sea_ice_frac,
            nwp_p_sfc,
            nwp_t_sfc,
            nwp_z_sfc,
            nwp_t_air,
            nwp_rh_sfc,
            nwp_weasd,
            nwp_tpw,
            nwp_t_tropo,
            nwp_p_tropo,
            nwp_u_wnd_10m,
            nwp_v_wnd_10m,
            nwp_to3,
            nwp_t_prf,
            nwp_z_prf,
            nwp_o3mr_prf,
            nwp_rh_prf,
            nwp_clwmr_prf,
        ) = get_gfs_data_from_grib(time_utc)

        base_zc_base, ccl_cloud_fraction = compute(
            time_utc,
            nav_lon, nav_lat,
            ch_rad_toa, ch_ref_toa, ch_bt_toa,
            nwp_n_lat, nwp_n_lon, nwp_n_levels,
            nwp_p_std,
            nwp_sea_ice_frac,
            nwp_p_sfc,
            nwp_t_sfc,
            nwp_z_sfc,
            nwp_t_air,
            nwp_rh_sfc,
            nwp_weasd,
            nwp_tpw,
            nwp_t_tropo,
            nwp_p_tropo,
            nwp_u_wnd_10m,
            nwp_v_wnd_10m,
            nwp_to3,
            nwp_t_prf,
            nwp_z_prf,
            nwp_o3mr_prf,
            nwp_rh_prf,
            nwp_clwmr_prf,
            debug=False
        )
        ccl_cloud_fraction *= 100.0

        time_cst.append(time_utc + timedelta(hours=8))
        base_zc_base_list.append(base_zc_base)
        ccl_cloud_fraction_list.append(ccl_cloud_fraction)

    latitude = np.linspace(30.5, 24.5, 151)
    longitude = np.linspace(108.5, 114.5, 151)
    dimensions = ('time', 'latitude', 'longitude')
    coordinates = {
        'time': time_cst, 'latitude': latitude, 'longitude': longitude
    }

    base_zc_base = np.stack(base_zc_base_list, axis=0)
    da_cbh = xr.DataArray(
        base_zc_base,
        coords=coordinates, dims=dimensions,
        attrs={'units': 'm'}
    )
    ccl_cloud_fraction = np.stack(ccl_cloud_fraction_list, axis=0)
    da_tcc = xr.DataArray(
        ccl_cloud_fraction,
        coords=coordinates, dims=dimensions,
        attrs={'units': '%'}
    )
    ds_cloud = xr.Dataset({'CBH': da_cbh, 'TCC': da_tcc})

    os.makedirs(save_path, exist_ok=True)
    ds_cloud.to_netcdf(
        save_name,
        encoding={
            'CBH': {
                'dtype': 'int16', 'scale_factor': 1.0, 'add_offset': 0.0, '_FillValue': -32768, 'zlib': True
            },
            'TCC': {
                'dtype': 'int16', 'scale_factor': 1.0e-2, 'add_offset': 0.0, '_FillValue': -32768, 'zlib': True
            }
        }
    )


if __name__ == '__main__':
    time_utc = datetime(2021, 10, 25, 5)

    time_utc = datetime(2022, 7, 14, 5, 15)
    main_real(time_utc)
