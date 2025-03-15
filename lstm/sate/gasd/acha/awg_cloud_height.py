from math import exp, sqrt, isnan, nan

import numpy as np
from numba import njit, prange

from constants import (
    sym_clear,
    sym_prob_clear,
    sym_clear_type,
    sym_prob_clear_type,
    sym_fog_type,
    sym_water_type,
    sym_supercooled_type,
    sym_mixed_type,
    sym_opaque_ice_type,
    sym_cirrus_type,
    sym_overlap_type,
    sym_overshooting_type,
    sym_unknown_type,
    sym_no,
    sym_yes,

    sym_water_sfc,
    sym_no_snow,
    sym_sea_ice,
    sym_snow,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from .acha_full_retrieval_mod import full_acha_retrieval
from .acha_ice_cloud_microphysical_model_ahi_110um import beta_110um_133um_coef_ice
from .acha_microphysical_module import beta_110um_133um_coef_water
from .acha_num_mod import (
    mean_smooth2, knowing_t_compute_p_z_bottom_up, knowing_z_compute_t_p,
    kd_tree_interp_2pred,
)
from .acha_parameters import (
    sensor_zenith_threshold,

    emissivity_min_cirrus,
    cirrus_box_width_km,

    num_param_simple,

    max_delta_t_inversion,
    tc_ap_uncer_opaque,
    tc_ap_uncer_cirrus_default,

    ec_ap_uncer_opaque,
    ec_ap_uncer_cirrus,

    beta_ap_water,
    beta_ap_uncer_water,
    beta_ap_ice,
    beta_ap_uncer_ice,

    tau_ap_fog_type,
    tau_ap_water_type,
    tau_ap_supercooled_type,
    tau_ap_mixed_type,
    tau_ap_opaque_ice_type,
    tau_ap_cirrus_type,
    tau_ap_overlap_type,

    zc_floor,
    num_lat_cirrus_ap,

    tc_cirrus_mean_lat_vector,
)

# todo
missing_value_integer1 = -128
pass_idx_min = 1
pass_idx_max = 5

n_ts = 7
n_tcs = 9
ts_min = 270.0
d_ts = 5.0
tcs_min = -20.0
d_tcs = 2.0

ocean_lapse_rate_table = np.array([
    [-9.7, -9.4, -9.4, -9.4, -9.2, -8.1, -7.2],
    [-9.9, -9.5, -9.5, -9.6, -9.4, -8.4, -7.4],
    [-10.0, -9.7, -9.6, -9.9, -9.6, -8.5, -7.6],
    [-10.0, -9.7, -9.6, -9.9, -9.6, -8.7, -8.0],
    [-9.6, -9.5, -9.7, -9.9, -9.6, -9.1, -8.7],
    [-9.1, -9.3, -9.5, -9.7, -9.6, -9.6, -9.7],
    [-8.9, -9.1, -9.2, -9.3, -9.5, -10.0, -10.3],
    [-8.1, -8.2, -8.2, -8.4, -8.8, -9.6, -10.3],
    [-7.5, -7.4, -7.3, -7.5, -8.2, -9.1, -9.9],
], 'f4')

land_lapse_rate_table = np.array([
    [-5.9, -6.2, -6.6, -6.9, -7.4, -8.2, -8.9],
    [-5.8, -6.1, -6.5, -6.8, -7.4, -8.3, -9.1],
    [-5.8, -6.0, -6.4, -6.7, -7.2, -8.2, -9.1],
    [-5.7, -5.9, -6.3, -6.5, -7.0, -8.1, -9.1],
    [-5.7, -5.9, -6.2, -6.3, -6.7, -7.8, -8.9],
    [-5.7, -5.9, -6.1, -6.3, -6.5, -7.6, -8.5],
    [-5.8, -5.9, -5.9, -5.9, -6.1, -7.4, -8.7],
    [-5.2, -5.4, -5.4, -5.5, -5.6, -6.8, -7.8],
    [-4.6, -4.9, -4.9, -5.0, -5.1, -6.2, -7.1],
], 'f4')


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def awg_cloud_height_algorithm(
        input_sensor_resolution_km,
        input_invalid_data_mask, input_surface_temperature,
        input_latitude, input_longitude,

        input_sensor_zenith_angle, input_cloud_mask,
        input_bt_110um, input_bt_120um, input_bt_133um,
        input_tropopause_temperature,

        input_tropopause_height, input_tropopause_pressure,
        input_tc_opaque,  # 可能为nan
        input_surface_type, input_snow_class, input_surface_emissivity_038um,

        input_rad_110um, input_rad_clear_110um,
        input_cosine_zenith_angle,
        input_surface_emissivity_110um, input_surface_emissivity_120um, input_surface_emissivity_133um,

        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        rtm_ch_rad_atm_profile, rtm_ch_trans_atm_profile, rtm_ch_rad_bb_cloud_profile,

        cloud_type_tmp,
):
    output_tc = np.full(image_shape, nan, 'f4')
    output_ec = np.full(image_shape, nan, 'f4')
    output_beta = np.full(image_shape, nan, 'f4')

    convergence_criteria_simple = (num_param_simple - 1.0) / 5.0

    # --- determine cirrus spatial interpolation box width
    box_half_width_cirrus = compute_box_width(input_sensor_resolution_km, cirrus_box_width_km)  # 100

    # --- initialize output

    # --- construct a mask to select pixel for lrc computation

    temperature_cirrus = np.full(image_shape, nan, 'f4')

    # --------------------------------------------------------------------------
    # multi-layer logic implemented via cloud type
    # -------------------------------------------------------------------------

    # --------------------------------------------------------------------------
    # determine processing order of pixels
    # --------------------------------------------------------------------------
    output_processing_order = compute_processing_order(input_invalid_data_mask, cloud_type_tmp)

    # --------------------------------------------------------------------------
    # loop through pixels using the processing order
    # --------------------------------------------------------------------------
    for pass_idx in range(pass_idx_min, pass_idx_max + 1):
        # --------------------------------------------------------------------------
        # loop over pixels in scanlines
        # --------------------------------------------------------------------------
        for line_idx in prange(image_number_of_lines):
            for elem_idx in prange(image_number_of_elements):
                # --- check if pixel should be processed in this path
                if pass_idx != pass_idx_max:
                    if pass_idx != output_processing_order[line_idx, elem_idx]:
                        continue

                # ---------------------------------------------------------------
                # check to see if this pixel should be skipped
                # ---------------------------------------------------------------
                bad_input_flag = test_input(
                    input_invalid_data_mask, input_sensor_zenith_angle,
                    input_cloud_mask,
                    input_bt_110um, input_bt_120um, input_bt_133um,
                    input_surface_temperature, input_tropopause_temperature,
                    line_idx, elem_idx
                )

                # --- if a bad pixel encountered, take action
                if bad_input_flag:
                    continue

                # --- for convenience, save nwp indices to local variables

                cloud_type = cloud_type_tmp[line_idx, elem_idx]
                # ---  filter pixels for last pass for cirrus correction
                if pass_idx == pass_idx_max:
                    if cloud_type != sym_cirrus_type and cloud_type != sym_overlap_type:
                        continue

                # initialize smooth nwp flag
                acha_rtm_nwp_sfc_level = rtm_sfc_level[line_idx, elem_idx]
                acha_rtm_nwp_tropo_level = rtm_tropo_level[line_idx, elem_idx]
                acha_rtm_nwp_black_body_rad_prof_110um = rtm_ch_rad_bb_cloud_profile[14][line_idx, elem_idx]

                ems_110um_tropo = compute_reference_level_emissivity(
                    acha_rtm_nwp_tropo_level,
                    input_rad_110um[line_idx, elem_idx],
                    input_rad_clear_110um[line_idx, elem_idx],
                    acha_rtm_nwp_black_body_rad_prof_110um
                )

                # ---- treat thick clouds as single layer
                if cloud_type == sym_overlap_type and ems_110um_tropo > 0.5:
                    cloud_type = sym_cirrus_type

                # --- for bad goes-17 data, all inputs have been switched from 110um to 104um,
                # --- if needed. variables remain the same name.

                # ---------------------------------------------------------------------------
                # select cloud type options
                # ---------------------------------------------------------------------------

                # clear情况下返回值均为nan 此时根据xap_success_flag会继续执行
                # dust，smoke，fire情况下返回值均为nan 此时根据xap_success_flag会跳过循环
                (
                    tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, xap_success_flag
                ) = compute_apriori_based_on_type(
                    cloud_type,
                    input_latitude[line_idx, elem_idx],
                    input_tropopause_temperature[line_idx, elem_idx],
                    input_bt_110um[line_idx, elem_idx],
                    input_tc_opaque[line_idx, elem_idx],
                    input_cosine_zenith_angle[line_idx, elem_idx],
                )

                if not xap_success_flag:
                    continue

                t_tropo = input_tropopause_temperature[line_idx, elem_idx]
                z_tropo = input_tropopause_height[line_idx, elem_idx]
                p_tropo = input_tropopause_pressure[line_idx, elem_idx]

                # --- do various 101 level nwp profiles
                acha_rtm_nwp_p_prof = rtm_p_std
                acha_rtm_nwp_t_prof = rtm_t_prof[line_idx, elem_idx]
                acha_rtm_nwp_z_prof = rtm_z_prof[line_idx, elem_idx]

                # ---- rtm profiles
                acha_rtm_nwp_atm_rad_prof_110um = rtm_ch_rad_atm_profile[14][line_idx, elem_idx]
                acha_rtm_nwp_atm_trans_prof_110um = rtm_ch_trans_atm_profile[14][line_idx, elem_idx]

                acha_rtm_nwp_atm_rad_prof_120um = rtm_ch_rad_atm_profile[15][line_idx, elem_idx]
                acha_rtm_nwp_atm_trans_prof_120um = rtm_ch_trans_atm_profile[15][line_idx, elem_idx]

                acha_rtm_nwp_atm_rad_prof_133um = rtm_ch_rad_atm_profile[16][line_idx, elem_idx]
                acha_rtm_nwp_atm_trans_prof_133um = rtm_ch_trans_atm_profile[16][line_idx, elem_idx]

                # -----------------------------------------------------------------------
                # assign values to y and y_variance
                # ----------------------------------------------------------------------
                # --- for goes-17 mitigation, compute_y may have 11 um data switched with 10.4
                # --- um data.
                y, y_variance = compute_y(
                    input_invalid_data_mask,
                    input_bt_110um, input_bt_120um, input_bt_133um,
                    line_idx, elem_idx,
                )

                # -------------------------------------------------------------------
                # determine surface type for use in forward model
                # 0 = water
                # 1 = land
                # 2 = snow
                # 3 = desert
                # 4 = arctic
                # 5 = antarctic
                # -------------------------------------------------------------------
                sfc_type_forward_model = determine_sfc_type_forward_model(
                    input_surface_type[line_idx, elem_idx],
                    input_snow_class[line_idx, elem_idx],
                    input_latitude[line_idx, elem_idx],
                    input_surface_emissivity_038um[line_idx, elem_idx],
                )

                # ------------------------------------------------------------------------
                # modify tc_ap and tc_uncer for lrc, cirrus and sounder options
                # ------------------------------------------------------------------------
                tc_ap = modify_tc_ap(
                    pass_idx, pass_idx_max,
                    line_idx, elem_idx,
                    temperature_cirrus, tc_ap,
                )

                # ------------------------------------------------------------------------
                #  lower cloud (surface) a prior values
                # ------------------------------------------------------------------------
                t_sfc_est = input_surface_temperature[line_idx, elem_idx]

                # ------------------------------------------------------------------------
                # fill x_ap vector with a priori values  
                # ------------------------------------------------------------------------
                x_ap_simple = np.array([tc_ap, ec_ap, beta_ap], 'f4')
                sa_simple = np.zeros((3, 3), 'f4')
                sa_simple[0, 0] = tc_ap_uncer
                sa_simple[1, 1] = ec_ap_uncer
                sa_simple[2, 2] = beta_ap_uncer
                sa_simple **= 2
                # --- compute inverse of sa matrix
                try:
                    sa_inv_simple = np.linalg.inv(sa_simple)
                except Exception:  # np.linalg.LinAlgErrorException:
                    print('cloud height warning ==> singular sa in acha')
                    continue

                # --------------------------------------------------
                # assign surface emissivity for non-overlap type
                # --------------------------------------------------
                ems_sfc_110um, ems_sfc_120um, ems_sfc_133um = set_surface_emissivity(
                    cloud_type,
                    input_surface_emissivity_110um, input_surface_emissivity_120um, input_surface_emissivity_133um,
                    line_idx, elem_idx,
                )

                x_simple, fail_flag = full_acha_retrieval(
                    y, y_variance, x_ap_simple, sa_inv_simple,
                    convergence_criteria_simple,  # acha_rtm_nwp_z_prof,
                    t_sfc_est, t_tropo, z_tropo, p_tropo, cloud_type, input_cosine_zenith_angle[line_idx, elem_idx],
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
                    num_param_simple
                )

                # --- successful retrieval post processing
                if fail_flag == sym_no:  # successful retrieval if statement
                    # --- save retrievals into the output variables
                    output_tc[line_idx, elem_idx] = x_simple[0]
                    output_ec[line_idx, elem_idx] = x_simple[1]  # note, this is slant
                    output_beta[line_idx, elem_idx] = x_simple[2]
                else:
                    # --- failed
                    output_tc[line_idx, elem_idx] = x_ap_simple[0]
                    output_ec[line_idx, elem_idx] = x_ap_simple[1]
                    output_beta[line_idx, elem_idx] = x_ap_simple[2]

        # ---------------------------------------------------------------------------
        # if selected, compute a background cirrus temperature and use for last pass
        # ---------------------------------------------------------------------------
        if pass_idx == pass_idx_max - 1:
            compute_temperature_cirrus(
                cloud_type_tmp,
                output_tc,
                output_ec,
                emissivity_min_cirrus,
                input_latitude, input_longitude,
                box_half_width_cirrus,
                temperature_cirrus
            )

    return output_tc, output_ec, output_beta


# ----------------------------------------------------------------------
# --- compute the apriori from the cloud phase and e_tropo
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_apriori_based_on_type(
        cloud_type,
        latitude,
        t_tropo,
        t110um,
        tc_opaque,
        mu
):
    # --- calipso values (not multiplier on uncer values)
    tc_ap_cirrus, tc_ap_uncer_cirrus = compute_cirrus_apriori(t_tropo, latitude)

    if isnan(tc_opaque):
        tc_ap_opaque = t110um
    else:
        tc_ap_opaque = tc_opaque

    if cloud_type in (sym_clear_type, sym_prob_clear_type):
        tc_ap = nan
        tc_ap_uncer = nan
        ec_ap = nan
        ec_ap_uncer = nan
        beta_ap = nan
        beta_ap_uncer = nan
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_fog_type:
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_fog_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_water
        beta_ap_uncer = beta_ap_uncer_water
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_water_type:
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_water_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_water
        beta_ap_uncer = beta_ap_uncer_water
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_supercooled_type:
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_supercooled_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_water
        beta_ap_uncer = beta_ap_uncer_water
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_mixed_type:
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_mixed_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_water
        beta_ap_uncer = beta_ap_uncer_water
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_opaque_ice_type:
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_opaque_ice_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_ice
        beta_ap_uncer = beta_ap_uncer_ice
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_cirrus_type:
        tc_ap = tc_ap_cirrus
        tc_ap_uncer = tc_ap_uncer_cirrus
        ec_ap = 1.0 - exp(-1.0 * tau_ap_cirrus_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_cirrus
        beta_ap = beta_ap_ice
        beta_ap_uncer = beta_ap_uncer_ice
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_overlap_type:
        tc_ap = tc_ap_cirrus
        tc_ap_uncer = tc_ap_uncer_cirrus
        ec_ap = 1.0 - exp(-1.0 * tau_ap_overlap_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_cirrus
        beta_ap = beta_ap_ice
        beta_ap_uncer = beta_ap_uncer_ice
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_overshooting_type:  # used opaque ice
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_opaque_ice_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_ice
        beta_ap_uncer = beta_ap_uncer_ice
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    elif cloud_type == sym_unknown_type:  # used mixed
        tc_ap = tc_ap_opaque
        tc_ap_uncer = tc_ap_uncer_opaque
        ec_ap = 1.0 - exp(-1.0 * tau_ap_mixed_type / mu)  # slow
        ec_ap_uncer = ec_ap_uncer_opaque
        beta_ap = beta_ap_water
        beta_ap_uncer = beta_ap_uncer_water
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, True
    else:
        tc_ap = nan
        tc_ap_uncer = nan
        ec_ap = nan
        ec_ap_uncer = nan
        beta_ap = nan
        beta_ap_uncer = nan
        return tc_ap, tc_ap_uncer, ec_ap, ec_ap_uncer, beta_ap, beta_ap_uncer, False


# -------------------------------------------------------------------
# determine surface type for use in forward model
# 0 = water
# 1 = land
# 2 = snow
# 3 = desert
# 4 = arctic
# 5 = antarctic
# -------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def determine_sfc_type_forward_model(
        surface_type,
        snow_class,
        latitude,
        ch07_surface_emissivity
):
    if surface_type == sym_water_sfc:
        sfc_type_forward_model = 0
    else:
        sfc_type_forward_model = 1  # land

    if snow_class == sym_snow and latitude > -60.0:
        sfc_type_forward_model = 2  # snow

    if (surface_type != sym_water_sfc and snow_class == sym_no_snow and ch07_surface_emissivity > 0.90 and
            abs(latitude) < 60.0):
        sfc_type_forward_model = 3  # desert

    if snow_class == sym_sea_ice and latitude > 60.0:
        sfc_type_forward_model = 4  # arctic

    if snow_class != sym_no_snow and latitude < -60.0:
        sfc_type_forward_model = 5  # antarctic

    return sfc_type_forward_model


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_reference_level_emissivity(
        ref_level, toa_radiance,
        toa_radiance_clear,
        black_body_rad_prof_110um
):
    emissivity_ref_level = (toa_radiance - toa_radiance_clear) / (
            black_body_rad_prof_110um[ref_level] - toa_radiance_clear)
    return emissivity_ref_level


# ----------------------------------------------------------------------
# local routine for a standard deviation
#
# data_array - input array of real numbers
# invalid_mask = 0 for good pixels, 1 for invalid pixels
# stddev = standard deviation for valid pixels in data array
#
# num_good = number of valid data point in array
#
# if num_good < 2, we do nothing
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_standard_deviation(data_array, invalid_mask):
    num_good = np.sum(1 - invalid_mask)

    if num_good == 0.0:
        stddev_of_array_r8 = nan
    elif num_good == 1.0:
        stddev_of_array_r8 = 0.0
    else:
        data_sum = np.sum(data_array * (1.0 - invalid_mask))
        data_sum_squared = np.sum((data_array * (1.0 - invalid_mask)) ** 2)
        tmp = data_sum_squared / num_good - (data_sum / num_good) ** 2
        if tmp > 0.0:
            stddev_of_array_r8 = sqrt(tmp)
        else:
            stddev_of_array_r8 = 0.0

    stddev_of_array_r4 = stddev_of_array_r8

    return stddev_of_array_r4


# ====================================================================
#
# make a background field of cirrus temperature from appropriate
# retrievals and use as an apriori constraint
#
# input
#   cld_type = standard cloud type values
#   temperature_cloud = cloud-top temperature
#   emissivity_cloud = cloud emissivity
#   emissivity_thresh = threshold for determining source pixels
#   count_thresh = number of source pixels needed to make a target value
#   box_width = pixel dimension of averaging box
#   missing = missing value to be used
#
# output
#   temperature_cirrus = cloud temperature of target pixels
#
# local
#   mask1 = mask of source pixels
#   mask2 = mask of target pixels
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_temperature_cirrus(
        cld_type,
        temperature_cloud,
        emissivity_cloud,
        emissivity_thresh,
        lat, lon,
        box_width,
        temperature_cirrus,
):
    n_idx_found = 6  # the number of surrounding indices to average
    kdtree_train_count_thresh = 10  # the values need to be larger than number of predictors

    mask1 = np.empty(image_shape, 'b1')
    mask2 = np.empty(image_shape, 'b1')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            mask1[line_idx, elem_idx] = (
                    cld_type[line_idx, elem_idx] in
                    (sym_cirrus_type, sym_opaque_ice_type, sym_overshooting_type, sym_overlap_type) and
                    temperature_cloud[line_idx, elem_idx] < 250.0 and
                    emissivity_cloud[line_idx, elem_idx] >= emissivity_thresh
            )
            mask2[line_idx, elem_idx] = (
                    cld_type[line_idx, elem_idx] in (sym_cirrus_type, sym_overlap_type)
            )

    # kdtree_train_count_thresh needs to be equal or greater than the number of
    # predictors used
    if np.sum(mask1) < kdtree_train_count_thresh:
        mean_smooth2(mask1, mask2, 1, 1, box_width, temperature_cloud, temperature_cirrus)
    else:
        kd_tree_interp_2pred(mask1, mask2, lat, lon, n_idx_found, temperature_cloud, temperature_cirrus)
    return temperature_cirrus


# --------------------------------------------------------------------------
# determine processing order of pixels
#
# processing order description
#
# pass 0 = not processed
# pass 1 = single layer lrc pixels (all phases)
# pass 2 = single layer water cloud pixels
# pass 3 = lrc multi-layer clouds
# pass 4 = all remaining clouds
# pass 5 = if use_cirrus_flag is set on, redo all thin cirrus using a priori
#          temperature from thicker cirrus.
# --------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_processing_order(invalid_data_mask, cloud_type):
    processing_order = np.full(image_shape, missing_value_integer1, 'i1')
    # --- loop through pixels, determine processing order
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            # --- skip data marked as bad
            if invalid_data_mask[line_idx, elem_idx] == sym_yes:
                continue
            # --- skip data marked as bad
            if cloud_type[line_idx, elem_idx] in (sym_clear_type, sym_prob_clear_type):
                processing_order[line_idx, elem_idx] = 0
                continue
            # -- on pass 2, do non-lrc water clouds
            if cloud_type[line_idx, elem_idx] in (sym_fog_type, sym_water_type, sym_mixed_type, sym_supercooled_type):
                processing_order[line_idx, elem_idx] = 2
                continue
            # --  on pass-4 do remaining
            processing_order[line_idx, elem_idx] = 4
    return processing_order


# ----------------------------------------------------------------------
# --- determine cirrus box width
# ---
# --- sensor_resolution_km = the nominal resolution in kilometers
# --- box_width_km = the width of the desired box in kilometers
# --- box_half_width = the half width of the box in pixel-space
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_box_width(sensor_resolution_km, box_width_km):
    if sensor_resolution_km <= 0.0:
        box_half_width = 20
    else:
        box_half_width = int((box_width_km / sensor_resolution_km) / 2)
    return box_half_width


# ----------------------------------------------------------------------
# empirical lapse rate
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def empirical_lapse_rate(t_sfc, tc, land_flag):
    tcs = tc - t_sfc
    i_ts = int((t_sfc - ts_min) / d_ts)
    i_ts = max(0, min(n_ts - 1, i_ts))
    i_tcs = int((tcs - tcs_min) / d_tcs)
    i_tcs = max(0, min(n_tcs - 1, i_tcs))
    if land_flag == 0:
        lapse_rate = ocean_lapse_rate_table[i_tcs, i_ts]
    else:
        lapse_rate = land_lapse_rate_table[i_tcs, i_ts]
    return lapse_rate


# ----------------------------------------------------------------------------
# estimate cirrus apriori temperature and uncertainty from a precomputed
# latitude table (stored in acha_parameters.inc)
# ----------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_cirrus_apriori(t_tropo, latitude):
    lat_min = -90.0
    delta_lat = -10.0

    lat_idx = int((latitude - lat_min) / delta_lat)
    lat_idx = max(0, min(lat_idx - 1, num_lat_cirrus_ap))

    tc_apriori = t_tropo + tc_cirrus_mean_lat_vector[lat_idx]
    # --- values of the std dev are too small so use a fixed value for uncertainty
    tc_apriori_uncer = tc_ap_uncer_cirrus_default
    return tc_apriori, tc_apriori_uncer


# -------------------------------------------------------------------------------------------
# compute the y and y_variance vectors which depend on the chosen mode
# -------------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_y(
        input_invalid_data_mask,
        input_bt_110um, input_bt_120um, input_bt_133um,
        line_idx, elem_idx,
):
    y = np.empty(3, 'f4')
    y_variance = np.empty(3, 'f4')
    # -----------------------------------------------------------------------
    # compute needed channel 3x3 standard deviations
    # -----------------------------------------------------------------------

    i1 = max(0, line_idx - 1)
    i2 = min(image_number_of_lines, line_idx + 2)

    j1 = max(0, elem_idx - 1)
    j2 = min(image_number_of_elements, elem_idx + 2)

    # --- at this point, for goes-17 bad data, bt_110 um should be bt_104 um.
    bt_110um_std = compute_standard_deviation(
        input_bt_110um[i1:i2, j1:j2], input_invalid_data_mask[i1:i2, j1:j2]
    )
    btd_110um_120um_std = compute_standard_deviation(
        input_bt_110um[i1:i2, j1:j2] - input_bt_120um[i1:i2, j1:j2],
        input_invalid_data_mask[i1:i2, j1:j2]
    )
    btd_110um_133um_std = compute_standard_deviation(
        input_bt_110um[i1:i2, j1:j2] - input_bt_133um[i1:i2, j1:j2],
        input_invalid_data_mask[i1:i2, j1:j2]
    )

    y[0] = input_bt_110um[line_idx, elem_idx]
    y[1] = input_bt_110um[line_idx, elem_idx] - input_bt_120um[line_idx, elem_idx]
    y[2] = input_bt_110um[line_idx, elem_idx] - input_bt_133um[line_idx, elem_idx]
    y_variance[0] = bt_110um_std ** 2
    y_variance[1] = btd_110um_120um_std ** 2
    y_variance[2] = btd_110um_133um_std ** 2

    return y, y_variance


# -------------------------------------------------------
# ---  for low clouds over water, force fixed lapse rate estimate of height
# -------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_height_from_lapse_rate(
        surface_type, cloud_type,
        surface_temperature, surface_elevation,
        max_delta_t_inversion,
        tc,
        zc,
        pc,
        acha_rtm_nwp_z_prof, acha_rtm_nwp_p_prof, acha_rtm_nwp_t_prof,
):
    delta_cld_temp_sfc_temp = surface_temperature - tc

    if isnan(tc):
        return zc, pc

    # --- new preferred method is to take out the snow_class check.
    if cloud_type in (sym_water_type, sym_fog_type, sym_supercooled_type):
        if delta_cld_temp_sfc_temp < max_delta_t_inversion:
            # -- select lapse rate  (k/km)
            if surface_type == sym_water_sfc:
                lapse_rate = empirical_lapse_rate(surface_temperature, tc, 0)
            else:
                lapse_rate = empirical_lapse_rate(surface_temperature, tc, 1)

            # --- constrain lapse rate to be with -2 and -10 k/km
            lapse_rate = min(-2.0, max(-10.0, lapse_rate))

            # --- convert lapse rate to k/m
            lapse_rate /= 1000.0  # (k/m)

            # -- compute height
            zc = -1.0 * delta_cld_temp_sfc_temp / lapse_rate + surface_elevation

            # --- some negative cloud heights are observed because of bad height
            # --- nwp profiles.
            if zc < 0:
                zc = zc_floor

            # --- compute pressure
            pc, r4_dummy = knowing_z_compute_t_p(
                acha_rtm_nwp_z_prof, acha_rtm_nwp_p_prof, acha_rtm_nwp_t_prof, zc
            )

    return zc, pc


# ---------------------------------------------------------------
# test the input to check to see if this pixel should be skipped
# if bad_input_flag =  True , acha won't do a retrieval
# ---------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def test_input(
        input_invalid_data_mask, input_sensor_zenith_angle,
        input_cloud_mask,
        input_bt_110um, input_bt_120um, input_bt_133um,
        input_surface_temperature, input_tropopause_temperature,
        line_idx, elem_idx
):
    if input_invalid_data_mask[line_idx, elem_idx] == sym_yes:
        return True

    if input_sensor_zenith_angle[line_idx, elem_idx] > sensor_zenith_threshold:
        return True

    if ((input_cloud_mask[line_idx, elem_idx] == sym_clear) or
            (input_cloud_mask[line_idx, elem_idx] == sym_prob_clear)):
        return True

    if ((input_bt_110um[line_idx, elem_idx] < 170.0) or  # begin data check
            (input_bt_110um[line_idx, elem_idx] > 340.0) or
            (input_surface_temperature[line_idx, elem_idx] < 180.0) or
            (input_surface_temperature[line_idx, elem_idx] > 340.0) or
            (input_tropopause_temperature[line_idx, elem_idx] < 160.0) or
            (input_tropopause_temperature[line_idx, elem_idx] > 270.0)):
        return True

    # --- check for missing values for relevant channels
    if isnan(input_bt_110um[line_idx, elem_idx]):
        return True

    if isnan(input_bt_120um[line_idx, elem_idx]):
        return True

    if isnan(input_bt_133um[line_idx, elem_idx]):
        return True

    return False


# --------------------------------------------------
# assign surface emissivity for non-overlap type
# --------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def set_surface_emissivity(
        cloud_type,
        input_surface_emissivity_110um, input_surface_emissivity_120um, input_surface_emissivity_133um,
        line_idx, elem_idx,
):
    if cloud_type != sym_overlap_type:
        ems_sfc_110um = input_surface_emissivity_110um[line_idx, elem_idx]
        ems_sfc_120um = input_surface_emissivity_120um[line_idx, elem_idx]
        ems_sfc_133um = input_surface_emissivity_133um[line_idx, elem_idx]
    else:
        ems_sfc_110um = 1.0
        ems_sfc_120um = 1.0
        ems_sfc_133um = 1.0

    return ems_sfc_110um, ems_sfc_120um, ems_sfc_133um


# ------------------------------------------------------------------------------------------
# modify the tc_ap and tc_ap_uncer using logic for use_lrc, use_cirrus and
# use_sounder.  this depends on the pass_idx
# ------------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def modify_tc_ap(
        pass_idx, pass_idx_max,
        line_idx, elem_idx,
        temperature_cirrus, tc_ap,
):
    # ------------------------------------------------------------------------
    # set tc apriori to lrc for pass 2 and 4 if using lrc
    # ------------------------------------------------------------------------

    # ------------------------------------------------------------------------
    # set apriori to predetermined cirrus value if use_cirrus_flag = true
    # ------------------------------------------------------------------------
    if pass_idx == pass_idx_max and (not isnan(temperature_cirrus[line_idx, elem_idx])):
        tc_ap = temperature_cirrus[line_idx, elem_idx]
    return tc_ap


# ----------------------------------------------------------------------
# convert the cloud temperature to height and pressure
#
# purpose:  perform the computation of pressure and height from temperature
#
# input:
#
# output:
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def convert_tc_to_pc_and_zc(
        input_invalid_data_mask,
        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        cloud_type,
        input_surface_type, input_surface_temperature, input_surface_elevation,
        input_tropopause_temperature, input_tropopause_height, input_tropopause_pressure,
        output_tc,
):
    output_pc = np.full(image_shape, nan, 'f4')
    output_zc = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- filter
            if input_invalid_data_mask[line_idx, elem_idx] == 1:
                continue
            if isnan(output_tc[line_idx, elem_idx]):
                continue

            # --------------------------------------------------------------------
            # get profiles for this pixel
            # --------------------------------------------------------------------

            # initialize smooth nwp flag
            acha_rtm_nwp_sfc_level = rtm_sfc_level[line_idx, elem_idx]
            acha_rtm_nwp_tropo_level = rtm_tropo_level[line_idx, elem_idx]

            # --- do various 101 level nwp profiles
            acha_rtm_nwp_p_prof = rtm_p_std
            acha_rtm_nwp_t_prof = rtm_t_prof[line_idx, elem_idx]
            acha_rtm_nwp_z_prof = rtm_z_prof[line_idx, elem_idx]

            # --- extract tropopause temp, height and pressure
            t_tropo = input_tropopause_temperature[line_idx, elem_idx]
            z_tropo = input_tropopause_height[line_idx, elem_idx]
            p_tropo = input_tropopause_pressure[line_idx, elem_idx]

            # --- default
            output_pc[line_idx, elem_idx], output_zc[line_idx, elem_idx] = knowing_t_compute_p_z_bottom_up(
                cloud_type[line_idx, elem_idx],
                output_tc[line_idx, elem_idx],
                t_tropo, z_tropo, p_tropo,
                acha_rtm_nwp_tropo_level, acha_rtm_nwp_sfc_level,
                acha_rtm_nwp_p_prof, acha_rtm_nwp_z_prof, acha_rtm_nwp_t_prof
            )

            # ---  for low clouds over water, force fixed lapse rate estimate of height
            (
                output_zc[line_idx, elem_idx], output_pc[line_idx, elem_idx],
            ) = compute_height_from_lapse_rate(
                input_surface_type[line_idx, elem_idx], cloud_type[line_idx, elem_idx],
                input_surface_temperature[line_idx, elem_idx], input_surface_elevation[line_idx, elem_idx],
                max_delta_t_inversion,
                output_tc[line_idx, elem_idx], output_zc[line_idx, elem_idx], output_pc[line_idx, elem_idx],
                acha_rtm_nwp_z_prof, acha_rtm_nwp_p_prof, acha_rtm_nwp_t_prof,
            )

    return output_pc, output_zc
