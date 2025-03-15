from math import cos, radians, nan, isnan

import numpy as np
import xarray as xr
from numba import njit, prange

from constants import (
    missing_value_int1,
    sym_clear,
    sym_prob_clear,
    sym_prob_cloudy,
    sym_cloudy,

    sym_no,
    sym_yes,

    sym_closed_shrubs_sfc,
    sym_open_shrubs_sfc,
    sym_grasses_sfc,
    sym_bare_sfc,
    sym_shallow_ocean,
    sym_land,
    sym_coastline,
    sym_shallow_inland_water,
    sym_ephemeral_water,
    sym_deep_inland_water,
    sym_moderate_ocean,
    sym_sea_ice,
    sym_snow,
)
from .nb_cloud_mask import (
    number_of_non_cloud_flags,

    reflectance_gross_sol_zen_thresh,
    reflectance_spatial_sol_zen_thresh,
    reflectance_gross_airmass_thresh,
    ems_375um_day_sol_zen_thresh,
    ems_375um_night_sol_zen_thresh,
    t_sfc_cold_scene_thresh,
    path_tpw_dry_scene_thresh,
    bt_375um_cold_scene_thresh,
    forward_scatter_scatter_zen_max_thresh,
    forward_scatter_sol_zen_max_thresh,
)
from .nb_cloud_mask_lut_module import compute_core

conf_clear_prob_clear_thresh = np.array([0.01, 0.01, 0.01, 0.10, 0.10, 0.10, 0.10], 'f4')
prob_clear_prob_cloud_thresh = np.array([0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 0.50], 'f4')
prob_cloudy_conf_cloud_thresh = np.array([0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90], 'f4')

ds = xr.open_dataset('static/ahi_default_nb_cloud_mask_lut.nc')

n_class = int(ds.attrs['n_class'])  # 12
n_bounds = int(ds.attrs['n_bounds_reg'])  # 101
n_sfc_type = int(ds.attrs['n_sfc_type'])  # 7

prior_yes = ds['prior_yes'].values
prior_no = ds['prior_no'].values

classifier_bounds_min = ds['bin_start'].values
classifier_bounds_max = ds['bin_end'].values
delta_classifier_bounds = ds['delta_bin'].values
classifier_value_name = ds['classifier_names'].values
class_cond_ratio = ds['class_cond_ratio_reg'].values

class_to_test_idx = np.empty(n_class, 'i4')
# ---set up classifier to test mapping
for class_idx in range(n_class):
    if classifier_value_name[class_idx] == b'T_11                          ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 0
    elif classifier_value_name[class_idx] == b'T_Max-T                       ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 1
    elif classifier_value_name[class_idx] == b'T_Std                         ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 2
    elif classifier_value_name[class_idx] == b'Emiss_Tropo                   ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 3
        class_idx_e_tropo = class_idx
    elif classifier_value_name[class_idx] == b'FMFT                          ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 4
    elif classifier_value_name[class_idx] == b'Btd_11_67                     ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 5
    elif classifier_value_name[class_idx] == b'Bt_11_67_Covar                ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 6
    elif classifier_value_name[class_idx] == b'Btd_11_85                     ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 7
    elif classifier_value_name[class_idx] == b'Emiss_375                     ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 8
    elif classifier_value_name[class_idx] == b'Btd_375_11_All                ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 8
    elif classifier_value_name[class_idx] == b'Btd_375_11_Day                ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 9
    elif classifier_value_name[class_idx] == b'Emiss_375_Day                 ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 9
    elif classifier_value_name[class_idx] == b'Btd_375_11_Night              ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 10
    elif classifier_value_name[class_idx] == b'Emiss_375_Night               ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 10
    elif classifier_value_name[class_idx] == b'Spare                         ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 11
    elif classifier_value_name[class_idx] == b'Ref_063_Day                   ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 12
    elif classifier_value_name[class_idx] == b'Ref_Std                       ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 13
    elif classifier_value_name[class_idx] == b'Ref_063_Min_3x3_Day           ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 14
    elif classifier_value_name[class_idx] == b'Ref_Ratio_Day                 ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 15
    elif classifier_value_name[class_idx] == b'Ref_138_Day                   ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 16
    elif classifier_value_name[class_idx] == b'Ndsi_Day                      ':
        class_to_test_idx[class_idx] = number_of_non_cloud_flags + 17
    else:
        print('unknown classifier naive bayesian cloud mask, returning')
        print('name = ', (classifier_value_name[class_idx]))


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_bayes_sfc_type(
        land_temp, coast_temp, snow_temp, sfc_type_temp,
        lat_temp, lon_temp, sst_back_uni_temp,
        ems_sfc_375um_temp
):
    if land_temp == sym_land:
        # 3 - land
        bayes_mask_sfc_type_temp = 3
    else:
        # 1 - deep ocean
        bayes_mask_sfc_type_temp = 1

    # 2 - shallow ocean
    if land_temp in (sym_moderate_ocean, sym_deep_inland_water, sym_shallow_inland_water, sym_shallow_ocean):
        bayes_mask_sfc_type_temp = 2
    if land_temp != sym_land and sst_back_uni_temp > 0.5:
        bayes_mask_sfc_type_temp = 2

    # 3 - unfrozen land
    if land_temp in (sym_land, sym_coastline, sym_ephemeral_water) or coast_temp:
        bayes_mask_sfc_type_temp = 3

    # 4 - snow covered land
    if lat_temp > -60.0 and snow_temp == sym_snow:
        bayes_mask_sfc_type_temp = 4

    # 5 - arctic
    if lat_temp >= 0.0 and snow_temp == sym_sea_ice:
        bayes_mask_sfc_type_temp = 5

    # 6 - antarctic & greenland
    if lat_temp <= -60.0 and snow_temp == sym_snow:
        bayes_mask_sfc_type_temp = 6
    if lat_temp <= 0.0 and snow_temp == sym_sea_ice:
        bayes_mask_sfc_type_temp = 6
    if (lat_temp >= 60.0 and -75.0 < lon_temp < -10.0 and
            land_temp in (sym_land, sym_coastline) and snow_temp == sym_snow):
        bayes_mask_sfc_type_temp = 6

    # 7 - desert
    if ems_sfc_375um_temp < 0.90 and abs(lat_temp) < 60.0 and sfc_type_temp in (sym_open_shrubs_sfc, sym_bare_sfc):
        bayes_mask_sfc_type_temp = 7
    if (bayes_mask_sfc_type_temp == 3 and ems_sfc_375um_temp < 0.93 and abs(lat_temp) < 60.0 and
            sfc_type_temp in (sym_open_shrubs_sfc, sym_closed_shrubs_sfc, sym_grasses_sfc, sym_bare_sfc)):
        bayes_mask_sfc_type_temp = 7

    return bayes_mask_sfc_type_temp


@njit(nogil=True, error_model='numpy', boundscheck=True)
def split_window_test(t11_clear, t12_clear, t11, t12):
    if t11_clear <= 265.0:
        return t11 - t12
    return (t11 - t12) - (t11_clear - t12_clear) * (t11 - 260.0) / (t11_clear - 260.0)


@njit(nogil=True, error_model='numpy', boundscheck=True)
def reflectance_gross_contrast_test(ref_clear, ref):
    if isnan(ref_clear):
        return nan
    return ref - ref_clear


@njit(nogil=True, error_model='numpy', boundscheck=True)
def relative_visible_contrast_test(ref_min, ref):
    if isnan(ref_min):
        return nan
    return ref - ref_min


@njit(nogil=True, error_model='numpy', boundscheck=True)
def reflectance_ratio_test(ref_vis, ref_nir):
    if isnan(ref_vis) or isnan(ref_nir):
        return nan
    return ref_nir / ref_vis


@njit(nogil=True, error_model='numpy', boundscheck=True)
def ems_375um_day_test(ems, ems_clear):
    if isnan(ems) or isnan(ems_clear):
        return nan
    return (ems - ems_clear) / ems_clear


@njit(nogil=True, error_model='numpy', boundscheck=True)
def ems_375um_night_test(ems, ems_clear):
    if isnan(ems) or isnan(ems_clear):
        return nan
    return ems


# ====================================================================
# subroutine name: cloud_mask_naive_bayes
#
# function:
#   calculates the bayesian cloud mask. the bayesian cloud mask is
#   determined by utilizing the following surface types:
#
# bayesian surface types
# 1 - deep_water
# 2 - shallow_water
# 3 - unfrozen_land
# 4 - frozen_land
# 5 - arctic
# 6 - antarctic
# 7 - desert
#
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def nb_cloud_mask_algorithm(
        input_invalid_data_mask,
        input_bayes_sfc_type,
        input_coastal_mask, input_z_sfc, input_sfc_temp,

        input_oceanic_glint_mask,
        input_sol_zen, input_sen_zen,
        input_scatter_zen,

        input_path_tpw,
        input_bt_110um_std,
        input_ref_063um, input_ref_086um, input_ref_160um, input_ref_375um,
        input_ref_063um_clear,
        input_ref_063um_min,
        input_bt_375um, input_bt_67um, input_bt_85um, input_bt_110um, input_bt_120um,
        input_bt_110um_clear, input_bt_120um_clear,
        input_bt_110um_bt_67um_covar,

        input_ems_375um,
        input_ems_375um_clear,
        input_ems_110um_tropo,

        input_prior,
        use_prior_table,
        use_core_tables
):
    # spare_value = 0

    # cld_flags = np.zeros(number_of_flags, 'i1')
    # cld_flag_bit_depth = np.full(number_of_flags, 2, 'i1')

    cond_ratio = np.empty(n_class, 'f4')
    posterior_cld_probability_by_class = np.empty(n_class, 'f4')
    classifier_value = np.empty(n_class, 'f4')

    output_posterior_cld_probability = nan
    # output_cld_flags_packed = np.zeros(number_of_flag_bytes, 'i1')
    # output_cld_flags_packed = np.full(number_of_flag_bytes, missing_value_int1, 'i1')

    # cld_flags[0] = sym_yes
    # cld_flag_bit_depth[0] = 1
    # --- check for a bad pixel
    if input_invalid_data_mask == sym_yes:
        pass
        # cld_flags[0] = sym_no
    else:
        # ---  compute surface type
        sfc_idx = input_bayes_sfc_type - 1

        # ----- compute prior
        if use_prior_table and not isnan(input_prior):
            prior_yes_temp = input_prior
        else:
            prior_yes_temp = prior_yes[sfc_idx]

        # todo
        # if (Skip_Sfc_Type_Flag(Sfc_Idx) == symbol % YES) then
        # Cld_Flags(1) = symbol % NO

        # --- set some flags to control processing
        oceanic_glint_flag = input_oceanic_glint_mask
        coastal_flag = input_coastal_mask
        # solar_contam_flag = sym_no

        # --- compute airmass
        airmass = 1.0 / cos(radians(input_sol_zen)) + 1.0 / cos(radians(input_sen_zen))

        # --- set day flag for 0.63 micron reflectance gross test
        if input_sol_zen > reflectance_gross_sol_zen_thresh or airmass > reflectance_gross_airmass_thresh:
            day_063_flag = sym_no
        else:
            day_063_flag = sym_yes

        # --- set day flag for 0.63 micron reflectance spatial test
        if input_sol_zen > reflectance_spatial_sol_zen_thresh:
            day_063_spatial_flag = sym_no
        else:
            day_063_spatial_flag = sym_yes

        if input_sol_zen > ems_375um_day_sol_zen_thresh or airmass > reflectance_gross_airmass_thresh:
            day_375_flag = sym_no
        else:
            day_375_flag = sym_yes

        if input_sol_zen < ems_375um_night_sol_zen_thresh:
            night_375_flag = sym_no
        else:
            night_375_flag = sym_yes

        if sfc_idx != 5 and input_z_sfc > 2000.0:
            mountain_flag = sym_yes
        else:
            mountain_flag = sym_no

        if input_scatter_zen < forward_scatter_scatter_zen_max_thresh and input_sol_zen < forward_scatter_sol_zen_max_thresh:
            forward_scattering_flag = sym_yes
        else:
            forward_scattering_flag = sym_no

        if input_bt_375um < bt_375um_cold_scene_thresh:
            cold_scene_375um_flag = sym_yes
        else:
            cold_scene_375um_flag = sym_no

        if input_sfc_temp < t_sfc_cold_scene_thresh:
            cold_scene_flag = sym_yes
        else:
            cold_scene_flag = sym_no

        if input_path_tpw < path_tpw_dry_scene_thresh:
            dry_scene_flag = sym_yes
        else:
            dry_scene_flag = sym_no

        # ----------------------------------------------------------------------------------
        # --- populate elements of Cld_Flags with processing flags
        # ----------------------------------------------------------------------------------
        # cld_flags[1] = day_063_flag
        # cld_flag_bit_depth[1] = 1
        # cld_flags[2] = day_063_spatial_flag
        # cld_flag_bit_depth[2] = 1
        # cld_flags[3] = day_375_flag
        # cld_flag_bit_depth[3] = 1
        # cld_flags[4] = night_375_flag
        # cld_flag_bit_depth[4] = 1
        # cld_flags[5] = 0  # solar_contam_flag
        # cld_flag_bit_depth[5] = 1
        # cld_flags[6] = coastal_flag
        # cld_flag_bit_depth[6] = 1
        # cld_flags[7] = mountain_flag
        # cld_flag_bit_depth[7] = 1
        # cld_flags[8] = forward_scattering_flag
        # cld_flag_bit_depth[8] = 1
        # cld_flags[9] = cold_scene_375um_flag
        # cld_flag_bit_depth[9] = 1
        # cld_flags[10] = cold_scene_flag
        # cld_flag_bit_depth[10] = 1
        # cld_flags[11] = oceanic_glint_flag
        # cld_flag_bit_depth[11] = 1
        # cld_flags[12] = spare_value
        # cld_flag_bit_depth[12] = 1
        # cld_flags[13] = spare_value
        # cld_flag_bit_depth[13] = 1
        # cld_flags[14] = spare_value
        # cld_flag_bit_depth[14] = 1
        # cld_flags[15] = spare_value
        # cld_flag_bit_depth[15] = 1
        # cld_flags[16] = input_bayes_sfc_type  # todo
        # cld_flag_bit_depth[16] = 3
        # # cld_flags[17] = spare_value
        # cld_flags[17] = 0
        # cld_flag_bit_depth[17] = 1

        # ---------------------------------------------------------------------------------
        # initialize all three cond_ratios to 1 (which is neutral impact)
        # ---------------------------------------------------------------------------------
        cond_ratio_core = 1.0
        cond_ratio_1d_all = 1.0
        cond_ratio_1d_cirrus = 1.0

        # ---------------------------------------------------------------------------------
        # compute cloud probability for core classifiers
        # ---------------------------------------------------------------------------------
        prob_clear_core = nan
        prob_water_core = nan
        prob_ice_core = nan

        if use_core_tables:
            prob_clear_core, prob_water_core, prob_ice_core = compute_core(
                input_ems_110um_tropo, input_ref_375um, input_ref_160um, input_bt_110um_std, input_sol_zen,
                sfc_idx,
                oceanic_glint_flag, forward_scattering_flag  # , solar_contam_flag,
            )
            if not isnan(prob_clear_core):
                if prob_clear_core < 1.0:
                    cond_ratio_core = prob_clear_core / (1.0 - prob_clear_core)
                else:
                    cond_ratio_core = 100.0

        # --- run if no core value or if core value is not definite
        if isnan(prob_clear_core) or 0.01 < prob_clear_core < 0.99:  # prob_clear_if_loop

            for class_idx in prange(n_class):

                classifier_value[class_idx] = nan
                cond_ratio[class_idx] = 1.0

                if classifier_value_name[class_idx] == b'T_Std                         ':
                    if use_core_tables:
                        continue
                    if mountain_flag == sym_yes:
                        continue
                    if coastal_flag == sym_yes:
                        continue
                    if isnan(input_bt_110um_std):
                        continue
                    classifier_value[class_idx] = input_bt_110um_std

                elif classifier_value_name[class_idx] == b'Btd_11_67                     ':
                    if isnan(input_bt_110um):
                        continue
                    if isnan(input_bt_67um):
                        continue
                    if sfc_idx == 0:
                        continue
                    if sfc_idx == 1:
                        continue
                    if sfc_idx == 2:
                        continue
                    if sfc_idx == 6:
                        continue
                    classifier_value[class_idx] = input_bt_110um - input_bt_67um

                elif classifier_value_name[class_idx] == b'Btd_11_85                     ':
                    # if cold_scene_flag == sym_yes: # todo
                    #     continue
                    if isnan(input_bt_110um):
                        continue
                    if isnan(input_bt_85um):
                        continue
                    classifier_value[class_idx] = input_bt_110um - input_bt_85um

                elif classifier_value_name[class_idx] == b'Emiss_375_Day                 ':
                    if use_core_tables:
                        continue
                    # if solar_contam_flag == sym_yes:
                    #     continue
                    if oceanic_glint_flag == sym_yes:
                        continue
                    if day_375_flag == sym_no:
                        continue
                    if cold_scene_375um_flag == sym_yes:
                        continue
                    if isnan(input_bt_375um):
                        continue
                    if isnan(input_ems_375um):
                        continue
                    if isnan(input_ems_375um_clear):
                        continue
                    classifier_value[class_idx] = ems_375um_day_test(input_ems_375um, input_ems_375um_clear)

                elif classifier_value_name[class_idx] == b'Emiss_375_Night               ':
                    if use_core_tables:
                        continue
                    # if solar_contam_flag == sym_yes:
                    #     continue
                    if night_375_flag == sym_no:
                        continue
                    if cold_scene_375um_flag == sym_yes:
                        continue
                    if isnan(input_bt_375um):
                        continue
                    if isnan(input_ems_375um):
                        continue
                    if isnan(input_ems_375um_clear):
                        continue
                    classifier_value[class_idx] = ems_375um_night_test(input_ems_375um, input_ems_375um_clear)

                elif classifier_value_name[class_idx] == b'Ref_063_Day                   ':
                    if input_sol_zen >= 90.0:
                        continue
                    if oceanic_glint_flag == sym_yes:
                        continue
                    if forward_scattering_flag == sym_yes:
                        continue
                    if mountain_flag == sym_yes:
                        continue
                    if day_063_flag == sym_no:
                        continue
                    if sfc_idx == 3:
                        continue
                    if isnan(input_ref_063um_clear):
                        continue
                    if isnan(input_ref_063um):
                        continue
                    classifier_value[class_idx] = reflectance_gross_contrast_test(
                        input_ref_063um_clear, input_ref_063um
                    )

                elif classifier_value_name[class_idx] == b'Ref_063_Min_3x3_Day           ':
                    if input_sol_zen >= 90.0:
                        continue
                    if day_063_spatial_flag == sym_no:
                        continue
                    if mountain_flag == sym_yes:
                        continue
                    if coastal_flag == sym_yes:
                        continue
                    if isnan(input_ref_063um_min):
                        continue
                    if isnan(input_ref_063um):
                        continue
                    classifier_value[class_idx] = relative_visible_contrast_test(
                        input_ref_063um_min, input_ref_063um
                    )

                elif classifier_value_name[class_idx] == b'Ref_Ratio_Day                 ':
                    if day_063_flag == sym_no:
                        continue
                    if mountain_flag == sym_yes:
                        continue
                    if oceanic_glint_flag == sym_yes:
                        continue
                    if forward_scattering_flag == sym_yes:
                        continue
                    if isnan(input_ref_063um):
                        continue
                    if isnan(input_ref_086um):
                        continue
                    classifier_value[class_idx] = reflectance_ratio_test(input_ref_063um, input_ref_086um)

                elif classifier_value_name[class_idx] == b'Ndsi_Day                      ':
                    if use_core_tables:
                        continue
                    if forward_scattering_flag == sym_yes:
                        continue
                    if day_063_flag == sym_no:
                        continue
                    if oceanic_glint_flag == sym_yes:
                        continue
                    if isnan(input_ref_160um):
                        continue
                    if isnan(input_ref_063um):
                        continue
                    classifier_value[class_idx] = (
                            (input_ref_063um - input_ref_160um) / (input_ref_063um + input_ref_160um)
                    )
                # todo 应该没用
                # else:
                #     classifier_value[class_idx] = nan

                if isnan(classifier_value[class_idx]):
                    continue

                bin_idx = int(
                    (classifier_value[class_idx] - classifier_bounds_min[sfc_idx, class_idx]) /
                    delta_classifier_bounds[sfc_idx, class_idx]
                )
                bin_idx = max(0, min(n_bounds - 2, bin_idx))
                cond_ratio[class_idx] = class_cond_ratio[sfc_idx, class_idx, bin_idx]

            cond_ratio_1d_all = np.prod(cond_ratio)

            # ---------------------------------------------------------------------------------
            # compute cloud probability for likely ice clouds 1d nb classifiers
            # ---------------------------------------------------------------------------------

            # --- run if no core value or core value shows likely water cloud
            if isnan(prob_water_core) or prob_water_core < 0.5:  # prob_water_if_loop

                for class_idx in prange(n_class):

                    classifier_value[class_idx] = nan
                    cond_ratio[class_idx] = 1.0

                    if classifier_value_name[class_idx] == b'Emiss_Tropo                   ':
                        if isnan(input_ems_110um_tropo):
                            continue
                        if cold_scene_flag == sym_yes:
                            continue
                        classifier_value[class_idx] = input_ems_110um_tropo

                    elif classifier_value_name[class_idx] == b'FMFT                          ':
                        if cold_scene_flag == sym_yes:
                            continue
                        if isnan(input_bt_110um):
                            continue
                        if isnan(input_bt_110um_clear):
                            continue
                        if isnan(input_bt_120um):
                            continue
                        if isnan(input_bt_120um_clear):
                            continue
                        classifier_value[class_idx] = split_window_test(
                            input_bt_110um_clear, input_bt_120um_clear, input_bt_110um, input_bt_120um
                        )

                    elif classifier_value_name[class_idx] == b'Bt_11_67_Covar                ':
                        if dry_scene_flag == sym_yes:
                            continue
                        if isnan(input_bt_110um_bt_67um_covar):
                            continue
                        classifier_value[class_idx] = input_bt_110um_bt_67um_covar

                    # todo 应该没用
                    # else:
                    #     classifier_value[class_idx] = nan

                    # --- turn off classifiers if chosen metric is missing
                    if isnan(classifier_value[class_idx]):
                        continue

                    # --- interpolate class conditional values
                    bin_idx = int(
                        (classifier_value[class_idx] - classifier_bounds_min[sfc_idx, class_idx]) /
                        delta_classifier_bounds[sfc_idx, class_idx]
                    )
                    bin_idx = max(0, min(n_bounds - 2, bin_idx))
                    cond_ratio[class_idx] = class_cond_ratio[sfc_idx, class_idx, bin_idx]

                cond_ratio_1d_cirrus = np.prod(cond_ratio)

            # ------------------------------------------------------------------------------------------------------------
            # --- compute posterior probabilities for each pixel
            # -----------------------------------------------------------------------------------------------------------

        r = cond_ratio_core * cond_ratio_1d_all * cond_ratio_1d_cirrus
        output_posterior_cld_probability = 1.0 / (1.0 + r / prior_yes_temp - r)

        # --- constrain
        if r < 0.001:
            output_posterior_cld_probability = 1.0
        if r > 99.0:
            output_posterior_cld_probability = 0.0

        # --- check for a missing prior
        if isnan(prior_yes_temp):
            output_posterior_cld_probability = nan
            posterior_cld_probability_by_class[:] = nan

        # ------------------------------------------------------------------------------------------------------------
        # --- compute probabilities for each class alone - used for flags - not
        # --- needed for mask or final probability - it should remain optional
        # ------------------------------------------------------------------------------------------------------------

        for class_idx in range(n_class):
            r = cond_ratio[class_idx]
            posterior_cld_probability_by_class[class_idx] = 1.0 / (1.0 + r / prior_yes[sfc_idx] - r)

            # -- set cloud flags
            # cld_flag_bit_depth[class_to_test_idx[class_idx]] = 2
            #
            # if posterior_cld_probability_by_class[class_idx] <= conf_clear_prob_clear_thresh[sfc_idx]:
            #     cld_flags[class_to_test_idx[class_idx]] = sym_clear
            #
            # elif (conf_clear_prob_clear_thresh[sfc_idx] < posterior_cld_probability_by_class[class_idx] <=
            #       prob_clear_prob_cloud_thresh[sfc_idx]):
            #     cld_flags[class_to_test_idx[class_idx]] = sym_prob_clear
            #
            # elif (prob_clear_prob_cloud_thresh[sfc_idx] < posterior_cld_probability_by_class[class_idx] <=
            #       prob_cloudy_conf_cloud_thresh[sfc_idx]):
            #     cld_flags[class_to_test_idx[class_idx]] = sym_prob_cloudy
            #
            # else:
            #     cld_flags[class_to_test_idx[class_idx]] = sym_cloudy

        # output_cld_flags_packed = pack_bits_into_bytes(cld_flags, cld_flag_bit_depth)

    return output_posterior_cld_probability  # , output_cld_flags_packed


@njit(nogil=True, error_model='numpy', boundscheck=True)
def nb_cloud_mask_algorithm2(
        input_invalid_data_mask,
        input_bayes_sfc_type,
        output_posterior_cld_probability
):
    output_cld_mask_bayes = missing_value_int1
    if input_invalid_data_mask == sym_yes:
        pass
    else:
        sfc_idx = input_bayes_sfc_type - 1
        # ------------------------------------------------------------------------------------------------------------
        # --- make a cloud mask
        # ------------------------------------------------------------------------------------------------------------
        # - based on type of sfc could be different thresholds
        if output_posterior_cld_probability >= prob_cloudy_conf_cloud_thresh[sfc_idx]:
            output_cld_mask_bayes = sym_cloudy
        elif output_posterior_cld_probability >= prob_clear_prob_cloud_thresh[sfc_idx]:
            output_cld_mask_bayes = sym_prob_cloudy
        elif output_posterior_cld_probability > conf_clear_prob_clear_thresh[sfc_idx]:
            output_cld_mask_bayes = sym_prob_clear
        else:
            output_cld_mask_bayes = sym_clear

    return output_cld_mask_bayes
