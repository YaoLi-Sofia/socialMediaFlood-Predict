from math import nan, isnan

import numpy as np
from numba import prange, njit, vectorize

from constants import (
    sym_yes,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from .ccl_parameters import (
    ccl_spacing_km, ccl_box_width_km, count_min_ccl,
)


@vectorize(nopython=True, forceobj=False)
def not_nan(x):
    return not isnan(x)


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


# ------------------------------------------------------------------------------
# compute cloud fraction over a nxn array using the bayesian probability
# ------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_cloud_cover_layers(
        input_sensor_resolution_km,
        input_invalid_data_mask,
        input_cloud_probability,
        input_alt,
):
    # -------------------------------------------------------------------------------
    # total cloud fraction and its uncertainty
    # -------------------------------------------------------------------------------

    # --- initialize
    output_total_cloud_fraction = np.full(image_shape, nan, 'f4')

    # --- determine box width
    n = compute_box_width(input_sensor_resolution_km, ccl_box_width_km)
    m = compute_box_width(input_sensor_resolution_km, ccl_spacing_km)

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- select the vertical coordinates based on ccl_type
            z = input_alt[line_idx, elem_idx]

            if not_nan(z):
                z = max(0.0, z)

            # ----- check for valid cloud vertical coordinate
            if isnan(z):
                continue

    # --------------------------------------------------------------------
    # compute pixel-level cloud cover for each layer over the box
    # n = ccl box size
    # m = ccl result spacing
    # --------------------------------------------------------------------
    # todo m=0
    # for i in prange(0, image_number_of_lines, 2 * m + 1):
    for i in prange(image_number_of_lines):
        i1 = max(0, i - n)
        i2 = min(image_number_of_lines, i + n + 1)
        i11 = max(0, i - m)
        i22 = min(image_number_of_lines, i + m + 1)
        # todo m=0
        # for j in prange(0, image_number_of_elements, 2 * m + 1):
        for j in prange(image_number_of_elements):
            j1 = max(0, j - n)
            j2 = min(image_number_of_elements, j + n + 1)
            j11 = max(0, j - m)
            j22 = min(image_number_of_elements, j + m + 1)

            # --- check for a bad pixel pixel
            if input_invalid_data_mask[i, j] == sym_yes:
                continue

            # ---- new layers are in between levels
            num_cloud = np.count_nonzero(input_cloud_probability[i1:i2, j1:j2] >= 0.5)
            num_good = np.count_nonzero(not_nan(input_cloud_probability[i1:i2, j1:j2]))

            # --- see if there are any valid mask points, if not skip this pixel
            if num_good < count_min_ccl:
                continue

            # --- total cloud fraction
            output_total_cloud_fraction[i11:i22, j11:j22] = num_cloud / num_good

    return output_total_cloud_fraction
