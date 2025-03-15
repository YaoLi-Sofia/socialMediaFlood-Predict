from math import nan

import numpy as np
from numba import njit, prange

from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from utils import show_time
from .nb_cloud_mask_lut_module import compute_prior
from .nb_cloud_mask_module import nb_cloud_mask_algorithm, nb_cloud_mask_algorithm2


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_posterior_cld_probability(
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
        month, use_prior_table, use_core_tables
):
    # ------------------------------------------------------------------------------------------
    # --- on first segment, read table
    # ------------------------------------------------------------------------------------------
    if use_prior_table:
        cld_mask_prior_cld_probability = compute_prior(nav_lon, nav_lat, month)
    else:
        cld_mask_prior_cld_probability = np.full(image_shape, nan, 'f4')
    # -----------    loop over pixels -----

    # cld_test_vector_packed = np.empty((*image_shape, 7), 'i1')
    posterior_cld_probability = np.empty(image_shape, 'f4')

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            input_invalid_data_mask = bad_pixel_mask[i, j]

            input_oceanic_glint_mask = sfc_glint_mask[i, j]
            input_coastal_mask = sfc_coast_mask[i, j]
            input_sol_zen = geo_sol_zen[i, j]
            input_scatter_zen = geo_scatter_zen[i, j]
            input_sen_zen = geo_sat_zen[i, j]
            # input_ems_sfc_375um = ch_sfc_ems[7][i, j]
            input_z_sfc = sfc_z_sfc[i, j]

            input_sfc_temp = nwp_pix_t_sfc[i, j]
            input_path_tpw = nwp_pix_tpw[i, j] / geo_cos_zen[i, j]

            input_ref_063um = ch_ref_toa[3][i, j]
            input_ref_086um = ch_ref_toa[4][i, j]
            input_ref_160um = ch_ref_toa[5][i, j]
            input_ref_375um = ch_ref_toa[7][i, j]

            input_ref_063um_clear = ch_ref_toa_clear[3][i, j]

            input_ref_063um_min = ref_ch03_min_3x3[i, j]

            input_ems_375um = ems_ch07_median_3x3[i, j]
            input_ems_375um_clear = ems_ch07_clear_rtm[i, j]

            input_bt_375um = ch_bt_toa[7][i, j]
            input_bt_67um = ch_bt_toa[9][i, j]
            input_bt_85um = ch_bt_toa[11][i, j]
            input_bt_110um = ch_bt_toa[14][i, j]
            input_bt_120um = ch_bt_toa[15][i, j]

            input_bt_110um_clear = ch_bt_toa_clear[14][i, j]
            input_bt_120um_clear = ch_bt_toa_clear[15][i, j]

            input_bt_110um_bt_67um_covar = covar_ch09_ch14_5x5[i, j]

            input_bt_110um_std = bt_ch14_std_3x3[i, j]

            input_ems_110um_tropo = ch_ems_tropo[14][i, j]

            input_prior = cld_mask_prior_cld_probability[i, j]
            input_bayes_sfc_type = cld_mask_bayes_mask_sfc_type[i, j]

            # ---call cloud mask routine
            # output_cld_mask_bayes, output_posterior_cld_probability, output_cld_flags_packed
            output_posterior_cld_probability = nb_cloud_mask_algorithm(
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
                # input_ems_sfc_375um,
                input_ems_110um_tropo,

                input_prior,
                use_prior_table,
                use_core_tables
            )

            # cld_test_vector_packed[i, j] = output_cld_flags_packed
            posterior_cld_probability[i, j] = output_posterior_cld_probability

    return posterior_cld_probability


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_cld_mask(
        bad_pixel_mask,
        cld_mask_bayes_mask_sfc_type,
        posterior_cld_probability,
):
    cld_mask = np.empty(image_shape, 'i1')

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            input_invalid_data_mask = bad_pixel_mask[i, j]
            input_bayes_sfc_type = cld_mask_bayes_mask_sfc_type[i, j]
            # cld_test_vector_packed[i, j] = output_cld_flags_packed
            output_posterior_cld_probability = posterior_cld_probability[i, j]

            output_cld_mask_bayes = nb_cloud_mask_algorithm2(
                input_invalid_data_mask,
                input_bayes_sfc_type,
                output_posterior_cld_probability
            )
            cld_mask[i, j] = output_cld_mask_bayes

    return cld_mask
