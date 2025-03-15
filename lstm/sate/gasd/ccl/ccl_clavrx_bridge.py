# $id: acha_clavrx_bridge_module.f90 1789 2016-09-28 22:20:51z heidinger $
# ------------------------------------------------------------------------------
#  noaa awg cloud height algorithm (acha) bridge code
#
#  this module houses the routines that serve as a bridge between
#  the clavr-x processing system and the acha code.
#
# ------------------------------------------------------------------------------
from numba import njit

from utils import show_time
from .ccl_module import compute_cloud_cover_layers


# ----------------------------------------------------------------------
# ccl bridge subroutine
# ----------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True)
def ccl_bridge(
        sensor_spatial_resolution_meters,
        bad_pixel_mask,
        cld_mask_posterior_cld_probability,
        acha_alt,
):
    input_sensor_resolution_km = sensor_spatial_resolution_meters / 1000.0
    input_invalid_data_mask = bad_pixel_mask
    input_cloud_probability = cld_mask_posterior_cld_probability
    input_alt = acha_alt

    output_total_cloud_fraction = compute_cloud_cover_layers(
        input_sensor_resolution_km,
        input_invalid_data_mask,
        input_cloud_probability,
        input_alt,
    )

    ccl_cloud_fraction = output_total_cloud_fraction

    return ccl_cloud_fraction
