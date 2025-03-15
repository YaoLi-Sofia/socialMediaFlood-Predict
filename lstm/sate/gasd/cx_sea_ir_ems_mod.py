import numpy as np
import xarray as xr
from numba import njit, prange

from constants import (
    sym_yes,
    sym_land,
    sym_no_snow,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    thermal_channels,
)

angle_table_delta = 5.0

# https://datashare.ed.ac.uk/handle/10283/17?show=full
ds = xr.open_dataset('static/ARCWideangleEmissivitySeaWater.nc')

wnd_spd_table = ds['wind_speed'].values
num_wnd_spd = wnd_spd_table.size
angle_table = ds['view_angle'].values
num_angle = angle_table.size
wvn_table = ds['wavenumber'].values
num_wvn = wvn_table.size
temp_table = ds['temperature'].values
num_temp = temp_table.size
sea_ir_ems_table = ds['emissivity'].values


# ----------------------------------------------------------------------------------------
# interpolate emissivity for conditions of a pixel
#
# ----------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def get_pixel_sea_ir_ems(wvn, wnd_spd, angle, sst):
    sst_idx = np.argmin(np.abs(sst - temp_table))
    sst_idx = max(0, min(num_temp - 1, sst_idx))

    wnd_spd_idx = np.argmin(np.abs(wnd_spd - wnd_spd_table))
    wnd_spd_idx = max(0, min(num_wnd_spd - 1, wnd_spd_idx))

    wvn_idx = np.argmin(np.abs(wvn - wvn_table))
    wvn_idx = max(0, min(num_wvn - 1, wvn_idx))

    angle_idx = min(int(angle / angle_table_delta), num_angle - 2)

    # --- select value from emissivity table with angle interp
    sea_ir_ems_1 = sea_ir_ems_table[sst_idx, wnd_spd_idx, wvn_idx, angle_idx]
    sea_ir_ems_2 = sea_ir_ems_table[sst_idx, wnd_spd_idx, wvn_idx, angle_idx + 1]

    sea_ir_ems = (sea_ir_ems_1 + (sea_ir_ems_2 - sea_ir_ems_1) *
                  (angle - angle_table[angle_idx]) / angle_table_delta)

    # print *, 'test ', angle, angle_table(angle_idx), sea_ir_ems_1, sea_ir_ems_2, sea_ir_ems

    return sea_ir_ems


# ----------------------------------------------------------------------
#  main public routine to compute for a segment
# ----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def get_segment_sea_ir_ems(
        ch_sfc_ems,
        bad_pixel_mask, geo_sat_zen,
        sfc_land, sfc_snow,
        nwp_pix_wnd_spd_10m, nwp_pix_t_sfc,
):
    planck_nu = {
        7: 2575.74,
        8: 1609.03,
        9: 1442.11,
        10: 1361.42,
        11: 1164.48,
        12: 1038.14,
        13: 961.385,
        14: 890.827,
        15: 809.396,
        16: 753.420
    }
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if bad_pixel_mask[line_idx, elem_idx] == sym_yes:
                continue
            if sfc_land[line_idx, elem_idx] != sym_land and sfc_snow[line_idx, elem_idx] == sym_no_snow:
                for c in thermal_channels:
                    ch_sfc_ems[c][line_idx, elem_idx] = get_pixel_sea_ir_ems(
                        planck_nu[c],
                        nwp_pix_wnd_spd_10m[line_idx, elem_idx],
                        geo_sat_zen[line_idx, elem_idx],
                        nwp_pix_t_sfc[line_idx, elem_idx]
                    )

    return ch_sfc_ems
