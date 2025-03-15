from numba import njit

from cx_sea_ir_ems_mod import get_segment_sea_ir_ems
from surface_properties import get_pixel_sfc_ems_from_sfc_type


@njit(nogil=True, error_model='numpy', boundscheck=True)
def cx_sfc_ems_correct_for_sfc_type(
        ch_sfc_ems,
        bad_pixel_mask, geo_sat_zen,
        sfc_sfc_type, sfc_snow, sfc_land,
        nwp_pix_wnd_spd_10m, nwp_pix_t_sfc,
):
    ch_sfc_ems = get_pixel_sfc_ems_from_sfc_type(ch_sfc_ems, bad_pixel_mask, sfc_sfc_type, sfc_snow)
    ch_sfc_ems = get_segment_sea_ir_ems(
        ch_sfc_ems,
        bad_pixel_mask, geo_sat_zen,
        sfc_land, sfc_snow,
        nwp_pix_wnd_spd_10m, nwp_pix_t_sfc,
    )
    return ch_sfc_ems
