from math import nan, isnan

import numpy as np
from numba import njit, prange

from constants import (
    sym_clear_type,
    sym_prob_clear_type,
    sym_fog_type,
    sym_water_type,
    sym_supercooled_type,
    sym_mixed_type,
    sym_opaque_ice_type,
    sym_tice_type,
    sym_cirrus_type,
    sym_overlap_type,
    sym_overshooting_type,
    sym_unknown_type,
    sym_dust_type,
    sym_smoke_type,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from utils import show_time


# todo 本模块的函数都是旧版的，用到的需要更新
# -----------------------------------------------------------
# compute cloud water path from the optical depth
# and particle size from the dcomp algorithm
#
# the layer values are computed assuming a linear variation
# in cloud water path from the top to the base of the cloud.
# note cwp = cwp_ice_layer + cwp_water_layer and 
#      cwp_scwater is a component of the water_layer
# 
# -----------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_cloud_water_path(
        geo_sol_zen, cld_type, tau_dcomp, reff_dcomp
):
    rho_water = 1.0  # g/m^3
    rho_ice = 0.917  # g/m^3

    cwp_dcomp = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- assign optical depth and particle size
            if geo_sol_zen[line_idx, elem_idx] < 90.0:
                tau = tau_dcomp[line_idx, elem_idx]
                reff = reff_dcomp[line_idx, elem_idx]
            else:
                tau = nan
                reff = nan

            if isnan(tau):
                continue
            if isnan(reff):
                continue

            # ------------------------------------------------
            # determine phase from cloud type
            # -1 = undetermined, 0 = water, 1 = ice
            # ------------------------------------------------
            i_phase = -1
            if cld_type[line_idx, elem_idx] == sym_clear_type:
                i_phase = -1
            elif cld_type[line_idx, elem_idx] == sym_prob_clear_type:
                i_phase = -1
            elif cld_type[line_idx, elem_idx] == sym_fog_type:
                i_phase = 0
            elif cld_type[line_idx, elem_idx] == sym_water_type:
                i_phase = 0
            elif cld_type[line_idx, elem_idx] == sym_supercooled_type:
                i_phase = 0
            elif cld_type[line_idx, elem_idx] == sym_mixed_type:
                i_phase = 0
            elif cld_type[line_idx, elem_idx] == sym_opaque_ice_type:
                i_phase = 1
            elif cld_type[line_idx, elem_idx] == sym_tice_type:
                i_phase = 1
            elif cld_type[line_idx, elem_idx] == sym_cirrus_type:
                i_phase = 1
            elif cld_type[line_idx, elem_idx] == sym_overlap_type:
                i_phase = 1
            elif cld_type[line_idx, elem_idx] == sym_overshooting_type:
                i_phase = 1
            elif cld_type[line_idx, elem_idx] == sym_unknown_type:
                i_phase = -1
            elif cld_type[line_idx, elem_idx] == sym_dust_type:
                i_phase = -1
            elif cld_type[line_idx, elem_idx] == sym_smoke_type:
                i_phase = -1

            # --- check conditions where this calc should be skipped
            if i_phase == -1:
                continue

            # --- compute cloud water path
            if i_phase == 0:
                cwp_dcomp[line_idx, elem_idx] = 0.55 * tau * reff * rho_water
            else:
                cwp_dcomp[line_idx, elem_idx] = 0.667 * tau * reff * rho_ice

    return cwp_dcomp
