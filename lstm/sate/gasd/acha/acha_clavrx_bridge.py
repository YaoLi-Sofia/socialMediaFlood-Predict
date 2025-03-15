import numpy as np
from numba import njit

from constants import (
    sym_cirrus_type,
    sym_overlap_type,
)
from utils import show_time
from .acha_comp import acha_comp_algorithm
from .awg_cloud_height import (
    awg_cloud_height_algorithm,
    convert_tc_to_pc_and_zc,
)


# ----------------------------------------------------------------------
# acha bridge subroutine
# ----------------------------------------------------------------------
@show_time
@njit(nogil=True, error_model='numpy', boundscheck=True)
def awg_cloud_height_bridge(
        sensor_spatial_resolution_meters,
        bad_pixel_mask,
        nav_lat, nav_lon,
        geo_cos_zen, geo_sat_zen,
        nwp_pix_t_sfc,
        nwp_pix_t_tropo, nwp_pix_z_tropo, nwp_pix_p_tropo,
        sfc_z_sfc, sfc_snow, sfc_sfc_type,
        ch_rad_toa, ch_bt_toa, ch_rad_toa_clear, ch_sfc_ems,
        cld_mask_cld_mask,
        cld_type,
        tc_opaque_cloud,
        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        rtm_ch_rad_atm_profile, rtm_ch_trans_atm_profile, rtm_ch_rad_bb_cloud_profile,
):
    input_sensor_resolution_km = sensor_spatial_resolution_meters / 1000.0

    input_invalid_data_mask = bad_pixel_mask

    input_bt_110um = ch_bt_toa[14]
    input_bt_120um = ch_bt_toa[15]
    input_bt_133um = ch_bt_toa[16]

    input_rad_110um = ch_rad_toa[14]

    input_cosine_zenith_angle = geo_cos_zen
    input_sensor_zenith_angle = geo_sat_zen

    input_surface_temperature = nwp_pix_t_sfc
    input_tropopause_temperature = nwp_pix_t_tropo
    input_tropopause_height = nwp_pix_z_tropo
    input_tropopause_pressure = nwp_pix_p_tropo

    input_surface_elevation = sfc_z_sfc
    input_latitude = nav_lat
    input_longitude = nav_lon

    input_rad_clear_110um = ch_rad_toa_clear[14]

    input_surface_emissivity_038um = ch_sfc_ems[7]

    input_surface_emissivity_110um = ch_sfc_ems[14]
    input_surface_emissivity_120um = ch_sfc_ems[15]
    input_surface_emissivity_133um = ch_sfc_ems[16]

    input_snow_class = sfc_snow
    input_surface_type = sfc_sfc_type
    input_cloud_mask = cld_mask_cld_mask
    input_cloud_type = cld_type

    input_tc_opaque = tc_opaque_cloud

    # -----------------------------------------------------------------------
    # --- call to awg cloud height algorithm (acha)
    # -----------------------------------------------------------------------
    cloud_type_tmp = np.where(input_cloud_type != sym_cirrus_type, input_cloud_type, sym_overlap_type)

    output_tc, output_ec, output_beta = awg_cloud_height_algorithm(
        input_sensor_resolution_km,
        input_invalid_data_mask, input_surface_temperature,
        input_latitude, input_longitude,
        # input_cloud_type,
        input_sensor_zenith_angle, input_cloud_mask,
        input_bt_110um, input_bt_120um, input_bt_133um,
        input_tropopause_temperature,

        input_tropopause_height, input_tropopause_pressure,
        # input_ice_cloud_probability, input_cloud_probability, input_cloud_phase_uncertainty,
        input_tc_opaque, input_surface_type, input_snow_class, input_surface_emissivity_038um,

        input_rad_110um, input_rad_clear_110um,
        # input_rad_120um, input_rad_clear_120um,
        # input_rad_133um, input_rad_clear_133um,
        input_cosine_zenith_angle,
        input_surface_emissivity_110um, input_surface_emissivity_120um, input_surface_emissivity_133um,
        # input_surface_air_temperature, input_surface_elevation,

        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        rtm_ch_rad_atm_profile, rtm_ch_trans_atm_profile, rtm_ch_rad_bb_cloud_profile,
        # input_sensor_azimuth_angle
        cloud_type_tmp,
    )

    # ---------------------------------------------------------------------------
    # post retrieval processing
    # ---------------------------------------------------------------------------
    # -- perform conversion of tc to zc and pc
    output_pc, output_zc = convert_tc_to_pc_and_zc(
        input_invalid_data_mask,
        rtm_sfc_level, rtm_tropo_level,
        rtm_p_std, rtm_t_prof, rtm_z_prof,
        cloud_type_tmp,
        input_surface_type, input_surface_temperature, input_surface_elevation,
        input_tropopause_temperature, input_tropopause_height, input_tropopause_pressure,
        output_tc,
    )

    # -----------------------------------------------------------------------
    # --- call algorithm to make acha optical and microphysical properties
    # -----------------------------------------------------------------------
    output_ec, output_beta, output_tau, output_reff = acha_comp_algorithm(
        input_cloud_type, input_sensor_zenith_angle, input_cosine_zenith_angle,
        output_zc,
        output_ec, output_beta,
    )

    acha_tc = output_tc
    acha_pc = output_pc
    acha_zc = output_zc
    acha_tau = output_tau
    acha_reff = output_reff

    return acha_tc, acha_pc, acha_zc, acha_tau, acha_reff
