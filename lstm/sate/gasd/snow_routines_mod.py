from math import nan

import numpy as np
from numba import vectorize, njit, prange

from constants import (
    sym_land,
    sym_no_snow,
    sym_sea_ice,
    sym_snow,
)
from public import image_shape, image_number_of_lines, image_number_of_elements


# ==============================================================================
#
# add snow using ndsi from 1.6 and 0.65 um
#
# ==============================================================================
@vectorize(nopython=True, forceobj=False)
def add_ndsi_160_snow(
        snow_class,
        toa_160_reflectance,
        toa_063_reflectance,
        t_sfc,
        ems_tropo,
        bt11_stddev,
        land_class,
        solar_zenith
):
    solar_zenith_max_threshold = 85.0  # orig 75.0
    toa_063_reflectance_threshold = 10.0
    # toa_ndsi_threshold = 0.35  # orig 0.50
    toa_ndsi_threshold = 0.60
    ems_tropo_threshold = 0.5
    bt11_stddev_threshold = 1.0
    t_sfc_threshold = 277

    if t_sfc > t_sfc_threshold:
        return snow_class

    if ems_tropo > ems_tropo_threshold:
        return snow_class

    if bt11_stddev > bt11_stddev_threshold:
        return snow_class

    if solar_zenith > solar_zenith_max_threshold:
        return snow_class

    if toa_063_reflectance > toa_063_reflectance_threshold:
        return snow_class

    if land_class != sym_land:
        return snow_class

    toa_ndsi = (toa_063_reflectance - toa_160_reflectance) / (toa_063_reflectance + toa_160_reflectance)

    if toa_ndsi > toa_ndsi_threshold:
        return sym_snow

    return snow_class


@vectorize(nopython=True, forceobj=False)
def add_ndsi_375_snow(
        snow_class,
        toa_375_reflectance,
        toa_063_reflectance,
        t_sfc,
        ems_tropo,
        bt11_stddev,
        land_class,
        solar_zenith
):
    solar_zenith_max_threshold = 85.0  # orig 75.0
    toa_063_reflectance_threshold = 10.0
    toa_ndsi_threshold = 0.5  # orig 0.75
    ems_tropo_threshold = 0.5
    bt11_stddev_threshold = 1.0
    t_sfc_threshold = 277

    if t_sfc > t_sfc_threshold:
        return snow_class

    if ems_tropo > ems_tropo_threshold:
        return snow_class

    if bt11_stddev > bt11_stddev_threshold:
        return snow_class

    if solar_zenith > solar_zenith_max_threshold:
        return snow_class

    if toa_063_reflectance > toa_063_reflectance_threshold:
        return snow_class

    if land_class != sym_land:
        return snow_class

    toa_ndsi = (toa_063_reflectance - toa_375_reflectance) / (toa_063_reflectance + toa_375_reflectance)

    if toa_ndsi > toa_ndsi_threshold:
        return sym_snow

    return snow_class


# ==============================================================================
# function to 'melt' snow or ice if it is too warm
# ==============================================================================
@vectorize(nopython=True, forceobj=False)
def remove_warm_snow(
        snow_class,
        toa_11_bt,
        surface_temperature,
        land_class
):
    toa_11_bt_max_threshold = 290.0  # orig 277.0
    t_sfc_max_threshold = 320.0  # orig 277.0

    if (toa_11_bt > toa_11_bt_max_threshold or surface_temperature > t_sfc_max_threshold) and land_class == sym_land:
        return sym_no_snow
    return snow_class


# ==============================================================================
# function to 'melt' snow or ice if it is too dark
# ==============================================================================
@vectorize(nopython=True, forceobj=False)
def remove_dark_snow(
        snow_class,
        toa_063_reflectance,
        solar_zenith,
        land_class
):
    solar_zenith_max_threshold = 85.0  # 75.0
    toa_063_reflectance_threshold = 10.0

    if (solar_zenith < solar_zenith_max_threshold and
            toa_063_reflectance < toa_063_reflectance_threshold and
            land_class == sym_land):
        return sym_no_snow

    return snow_class


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
# ---  1 = sym_no_snow
# ---  2 = sym_sea_ice
# ---  3 = sym_snow
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_snow_class(
        snow_class_nwp,
        # snow_class_oisst, snow_class_ims,
        land_class,
        ch_ref_toa, ch_bt_toa,  # ch_ems_tropo,
        ch14_bt_toa_std_3x3,
        nwp_pix_t_sfc, geo_sol_zen,
):
    # snow_class_final = missing_value_int1
    # # --- initialize to nwp
    # snow_class_final = snow_class_nwp
    # --- high res
    snow_class_final = snow_class_nwp.copy()

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            # --- overwrite with oisst
            # if snow_class_oisst[line_idx, elem_idx] == sym_sea_ice:
            #     snow_class_final[line_idx, elem_idx] = sym_sea_ice

            # -- check for consistency of land and snow masks
            if snow_class_final[line_idx, elem_idx] == sym_snow and land_class[line_idx, elem_idx] != sym_land:
                snow_class_final[line_idx, elem_idx] = sym_sea_ice

            if snow_class_final[line_idx, elem_idx] == sym_sea_ice and land_class[line_idx, elem_idx] == sym_land:
                snow_class_final[line_idx, elem_idx] = sym_snow

    snow_class_final = add_ndsi_160_snow(
        snow_class_final, ch_ref_toa[5], ch_ref_toa[3],
        np.full(image_shape, nan, 'f4'), ch14_bt_toa_std_3x3, nwp_pix_t_sfc, land_class, geo_sol_zen
    )

    snow_class_final = add_ndsi_375_snow(
        snow_class_final, ch_ref_toa[7], ch_ref_toa[3],
        np.full(image_shape, nan, 'f4'), ch14_bt_toa_std_3x3, nwp_pix_t_sfc, land_class, geo_sol_zen
    )

    # ---- remove snow under certain conditions
    snow_class_final = remove_warm_snow(snow_class_final, ch_bt_toa[14], nwp_pix_t_sfc, land_class)
    snow_class_final = remove_dark_snow(snow_class_final, ch_ref_toa[3], geo_sol_zen, land_class)

    # for line_idx in prange(image_number_of_lines):
    #     for elem_idx in prange(image_number_of_elements):
    #         # --- make sure everything out of range is missing
    #         if sym_snow < snow_class_final[line_idx, elem_idx] < sym_no_snow:
    #             snow_class_final[line_idx, elem_idx] = missing_value_int1

    return snow_class_final
