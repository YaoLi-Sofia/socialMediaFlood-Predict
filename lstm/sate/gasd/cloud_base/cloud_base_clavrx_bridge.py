from numba import njit

from utils import show_time
from .cloud_base import cloud_base_algorithm


# $id: acha_clavrx_bridge_mod.f90 580 2014-10-08 03:38:52z heidinger $
# ------------------------------------------------------------------------------
#  noaa awg cloud base algorithm (acba) bridge code
#
#  this module houses the routines that serve as a bridge between
#  the clavr-x processing system and the acba code.
#
# cld_height_acha    zc_acha
# cld_opd_dcomp      tau_dcomp
# cloud_water_path   cwp
# surface_elevation
# ccl_nwp
# cloud_type
# land_class
# solar_zenith_angle
# latitude
# fill_value = -999.0
# qf_fill = 1 
#
# ------------------------------------------------------------------------------


@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True)
def cloud_base_bridge(
        bad_pixel_mask,
        sfc_z_sfc,
        cld_type,
        acha_tc, acha_zc, acha_tau,
        nwp_pix_lcl_height, nwp_pix_ccl_height,
        cwp_dcomp, nwp_pix_cwp,
        rtm_sfc_level,
        rtm_z_prof,
):
    # -----------------------------------------------------------------------
    # ---  clavr-x bridge section
    # -----------------------------------------------------------------------
    # --- initialize input structure pointers

    # --- store integer values

    input_invalid_data_mask = bad_pixel_mask

    input_surface_elevation = sfc_z_sfc

    input_cloud_type = cld_type

    input_tc = acha_tc

    input_zc = acha_zc
    input_tau = acha_tau

    input_lcl = nwp_pix_lcl_height
    input_ccl = nwp_pix_ccl_height
    input_cwp = cwp_dcomp
    input_nwp_cwp = nwp_pix_cwp

    # ---- initialize output structure

    # -----------------------------------------------------------------------
    # --- call algorithm to make cloud geometrical boundaries
    # -----------------------------------------------------------------------

    output_zc_base = cloud_base_algorithm(
        input_invalid_data_mask,
        input_cloud_type,
        rtm_sfc_level,  # rtm_tropo_level,
        # rtm_p_std, rtm_t_prof,
        rtm_z_prof,
        input_surface_elevation,
        input_tau, input_tc, input_zc,
        input_cwp, input_nwp_cwp,
        input_lcl, input_ccl,
    )

    base_zc_base = output_zc_base

    return base_zc_base
