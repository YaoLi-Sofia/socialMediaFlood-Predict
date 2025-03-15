import numpy as np
from numba import i1, f4
from numba import njit, prange
from numba.typed import Dict

from constants import (
    missing_value_int1,
    sym_no,
    sym_yes,

    sym_water_sfc,
    sym_evergreen_needle_sfc,
    sym_evergreen_broad_sfc,
    sym_land,
    sym_no_coast,
    sym_coast_5km,
    sym_no_snow,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)

ch03_sfc_alb_umd = Dict.empty(i1, f4)
ch03_sfc_alb_umd.update({
    0: 5.00, 1: 4.06, 2: 4.44, 3: 4.06, 4: 5.42, 5: 4.23, 6: 4.23,
    7: 5.76, 8: 12.42, 9: 17.81, 10: 7.88, 11: 6.77, 12: 31.24, 13: 9.18
})

ch04_sfc_alb_umd = Dict.empty(i1, f4)
ch04_sfc_alb_umd.update({
    0: 1.63, 1: 19.91, 2: 21.79, 3: 19.91, 4: 29.43, 5: 25.00, 6: 25.00,
    7: 27.00, 8: 30.72, 9: 29.42, 10: 29.19, 11: 25.20, 12: 40.19, 13: 24.66
})

ch05_sfc_alb_umd = Dict.empty(i1, f4)
ch05_sfc_alb_umd.update({
    0: 1.63, 1: 20.20, 2: 22.10, 3: 20.20, 4: 29.80, 5: 25.00, 6: 25.00,
    7: 27.00, 8: 34.00, 9: 25.80, 10: 29.80, 11: 25.20, 12: 39.00, 13: 26.50
})

ch07_sfc_alb_umd = Dict.empty(i1, f4)
ch07_sfc_alb_umd.update({
    0: 1.5, 1: 5.0, 2: 4.0, 3: 6.0, 4: 6.0, 5: 6.0, 6: 6.0,
    7: 7.0, 8: 11.0, 9: 11.0, 10: 8.0, 11: 6.0, 12: 14.0, 13: 6.0
})

ch03_snow_sfc_alb_umd = Dict.empty(i1, f4)
ch03_snow_sfc_alb_umd.update({
    0: 66.0, 1: 23.0, 2: 50.0, 3: 42.0, 4: 25.0, 5: 21.0, 6: 49.0,
    7: 59.0, 8: 72.0, 9: 78.0, 10: 70.0, 11: 72.0, 12: 76.0, 13: 65.0
})

ch04_snow_sfc_alb_umd = Dict.empty(i1, f4)
ch04_snow_sfc_alb_umd.update({
    0: 61.0, 1: 29.0, 2: 60.0, 3: 43.0, 4: 34.0, 5: 31.0, 6: 52.0,
    7: 56.0, 8: 60.0, 9: 76.0, 10: 69.0, 11: 71.0, 12: 72.0, 13: 65.0
})

ch05_snow_sfc_alb_umd = Dict.empty(i1, f4)
ch05_snow_sfc_alb_umd.update({
    0: 2.0, 1: 8.0, 2: 64.0, 3: 15.0, 4: 13.0, 5: 9.0, 6: 9.0,
    7: 9.0, 8: 9.0, 9: 10.0, 10: 9.0, 11: 12.0, 12: 9.0, 13: 14.0
})

ch06_snow_sfc_alb_umd = Dict.empty(i1, f4)
ch06_snow_sfc_alb_umd.update({
    0: 13.0, 1: 3.0, 2: 13.0, 3: 8.0, 4: 8.0, 5: 8.0, 6: 8.0,
    7: 8.0, 8: 8.0, 9: 8.0, 10: 8.0, 11: 8.0, 12: 3.0, 13: 8.0
})


# ----------------------------------------------------------------------
# based on the coast and land flags, derive binary land and coast
# masks (yes/no)
#
# note, coast mask is dependent on sensor resolution
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_binary_land_coast_masks(bad_pixel_mask, sfc_land, sfc_sfc_type, sfc_coast):
    sfc_land_mask = np.full(image_shape, missing_value_int1, 'i1')
    sfc_coast_mask = np.full(image_shape, missing_value_int1, 'i1')
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            # --- check for a bad pixel
            if bad_pixel_mask[i, j] == sym_yes:
                continue

            # --- binary land mask
            sfc_land_mask[i, j] = sym_no

            # --- if land mask read in, use it

            if sfc_land[i, j] == sym_land:
                sfc_land_mask[i, j] = sym_yes

            # --- if land mask not read in, base off of surface type
            else:
                if sfc_sfc_type[i, j] != sym_water_sfc:
                    sfc_land_mask[i, j] = sym_yes

            # --- binary coast mask
            sfc_coast_mask[i, j] = sym_no

            # -- for lac, hrpt or frac data
            if sfc_coast[i, j] != sym_no_coast:
                if sfc_coast[i, j] <= sym_coast_5km:
                    sfc_coast_mask[i, j] = sym_yes

    return sfc_land_mask, sfc_coast_mask


# -----------------------------------------------------------------------
# if no sfc emiss data base read in, base it on the surface type.
# also derive surface reflectance from surface emissivity
# also reset sfc emiss if snow covered
# -----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def get_pixel_sfc_ems_from_sfc_type(ch_sfc_ems, bad_pixel_mask, sfc_sfc_type, sfc_snow):
    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            # --- check for a bad pixel
            if bad_pixel_mask[i, j] == sym_yes:
                continue
            # --- based on surface type, assign surface emissivities if seebor not used
            # --- if the sfc_type is missing, treat pixel as bad

            # --- for ocean, use this parameterization from nick nalli - overwrite seebor
            if sfc_sfc_type[i, j] == 0:
                #    sfc_ems_ch31[i,j] =  0.844780 + 0.328921 * cos_zen[i,j]  -0.182375*(cos_zen[i,j]**2)
                #    sfc_ems_ch32[i,j] =  0.775019 + 0.474005 * cos_zen[i,j]  -0.261739*(cos_zen[i,j]**2)
                # --- set to constant until we can verify the above
                ch_sfc_ems[7][i, j] = 0.985
                ch_sfc_ems[9][i, j] = 0.985
                # ch_sfc_ems[10][i, j] = 0.985
                # ch_sfc_ems[11][i, j] = 0.985
                # ch_sfc_ems[12][i, j] = 0.985
                # ch_sfc_ems[13][i, j] = 0.985
                ch_sfc_ems[14][i, j] = 0.985
                ch_sfc_ems[15][i, j] = 0.985
                ch_sfc_ems[16][i, j] = 0.985

            # --- if (snow_mask
            if sfc_snow[i, j] != sym_no_snow:
                if sfc_sfc_type[i, j] not in (sym_evergreen_needle_sfc, sym_evergreen_broad_sfc):
                    ch_sfc_ems[7][i, j] = 0.984
                    ch_sfc_ems[9][i, j] = 0.979
                    # ch_sfc_ems[10][i, j] = 0.979
                    # ch_sfc_ems[11][i, j] = 0.979
                    # ch_sfc_ems[13][i, j] = 0.979
                    ch_sfc_ems[14][i, j] = 0.979
                    ch_sfc_ems[15][i, j] = 0.977
                    ch_sfc_ems[16][i, j] = 0.977

    return ch_sfc_ems
