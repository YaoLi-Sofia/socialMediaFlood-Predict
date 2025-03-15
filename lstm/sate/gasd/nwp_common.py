from math import log, nan, isnan

import numpy as np
from numba import njit, prange, vectorize

from constants import missing_value_int1, g, sym_yes, sym_no, sym_land, sym_no_snow
from numerical_routines import vapor, vapor_ice
from public import (
    image_number_of_lines,
    image_number_of_elements,
    p_inversion_min,
    delta_t_inversion,
)


@vectorize(nopython=True, forceobj=False)
def qc_nwp(nwp_p_tropo, nwp_t_tropo, nwp_z_sfc, nwp_p_sfc, nwp_t_sfc):
    return (
            nwp_p_tropo <= 0.0 or nwp_t_tropo <= 0.0 or
            nwp_z_sfc > 10000.0 or nwp_p_sfc > 1500.0 or
            nwp_t_sfc > 400.0 or nwp_t_sfc <= 0.0
    )


# compute nwp levels for each nwp grid cell
#
# must be called after map_pixel_nwp
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_nwp_levels_segment(
        nwp_p_std, nwp_p_sfc, nwp_p_tropo, nwp_z_prof, nwp_t_prof, nwp_shape
):
    nwp_n_lat, nwp_n_lon, nwp_n_levels = nwp_shape
    # nwp_n_lat, nwp_n_lon = nwp_p_sfc.shape
    # nwp_n_levels = nwp_p_std.size
    nwp_sfc_level = np.full((nwp_n_lat, nwp_n_lon), missing_value_int1, 'i1')
    nwp_tropo_level = np.full((nwp_n_lat, nwp_n_lon), missing_value_int1, 'i1')
    nwp_inversion_level_profile = np.zeros((nwp_n_lat, nwp_n_lon, nwp_n_levels), 'i1')
    nwp_inversion_level = np.zeros((nwp_n_lat, nwp_n_lon), 'i1')
    nwp_z_tropo = np.zeros((nwp_n_lat, nwp_n_lon), 'f4')

    for lat_idx in prange(nwp_n_lat):
        for lon_idx in prange(nwp_n_lon):
            # check for valid nwp mapping and data, if not skip
            # if bad_nwp_mask[lat_idx, lon_idx] == sym_yes:
            #     continue
            # find needed nwp levels for this nwp cell, store in global variables
            # subroutine to find key levels in the profiles
            # find surface level (standard closest but less than sfc pressure)
            for k in range(nwp_n_levels - 1, -1, -1):
                if nwp_p_std[k] < nwp_p_sfc[lat_idx, lon_idx]:
                    nwp_sfc_level[lat_idx, lon_idx] = k
                    break
            else:
                nwp_sfc_level[lat_idx, lon_idx] = nwp_n_levels - 1
            # find tropopause level based on tropopause pressure
            # tropopause is between tropopause_level and tropopause_level + 1
            # constrain tropopause pressure to be greater than 75 mb
            p_tropo_tmp = max(nwp_p_tropo[lat_idx, lon_idx], 75.0)
            for k in range(nwp_sfc_level[lat_idx, lon_idx]):
                if nwp_p_std[k] <= p_tropo_tmp < nwp_p_std[k + 1]:
                    nwp_tropo_level[lat_idx, lon_idx] = k
                    break
            else:
                nwp_tropo_level[lat_idx, lon_idx] = 0

            nwp_z_tropo[lat_idx, lon_idx] = nwp_z_prof[lat_idx, lon_idx, nwp_tropo_level[lat_idx, lon_idx]]

            nwp_inversion_level_profile[lat_idx, lon_idx] = sym_no
            # todo nwp_tropo_level[lat_idx, lon_idx] 是不是应该+1,虽然结果一样
            for k in range(nwp_sfc_level[lat_idx, lon_idx], nwp_tropo_level[lat_idx, lon_idx] - 1, -1):
                if nwp_p_std[k] >= p_inversion_min:
                    if nwp_t_prof[lat_idx, lon_idx, k - 1] - nwp_t_prof[lat_idx, lon_idx, k] > delta_t_inversion:
                        nwp_inversion_level_profile[lat_idx, lon_idx, k - 1:k + 1] = sym_yes

            for k in range(nwp_tropo_level[lat_idx, lon_idx], nwp_sfc_level[lat_idx, lon_idx] + 1):
                if nwp_inversion_level_profile[lat_idx, lon_idx, k] == sym_yes:
                    # top_lev_idx = k
                    # nwp_inversion_level[lat_idx, lon_idx] = top_lev_idx
                    nwp_inversion_level[lat_idx, lon_idx] = k
                    break
            else:
                # top_lev_idx = -1
                nwp_inversion_level[lat_idx, lon_idx] = -1

    return nwp_sfc_level, nwp_tropo_level, nwp_inversion_level, nwp_z_tropo


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_nwp_level_height(
        nwp_t_air, nwp_rh_sfc, nwp_sfc_level, nwp_tropo_level, nwp_p_std, nwp_t_prof, nwp_z_prof,
        nwp_shape
):
    nwp_n_lat, nwp_n_lon, nwp_n_levels = nwp_shape
    nwp_lifting_condensation_level_height = np.empty((nwp_n_lat, nwp_n_lon), 'f4')
    nwp_convective_condensation_level_height = np.empty((nwp_n_lat, nwp_n_lon), 'f4')
    nwp_level_free_convection_height = np.empty((nwp_n_lat, nwp_n_lon), 'f4')
    nwp_equilibrium_level_height = np.empty((nwp_n_lat, nwp_n_lon), 'f4')

    for lat_idx in prange(nwp_n_lat):
        for lon_idx in prange(nwp_n_lon):
            # Find the Height of the Lifting(LCL) and Convective Condensation Level(CCL)
            # Heights from the NWP Profiles

            t = nwp_t_air[lat_idx, lon_idx]  # K
            if t > 180.0:
                if t > 253.0:
                    es = vapor(t)  # saturation vapor pressure wrt water hpa
                else:
                    es = vapor_ice(t)  # saturation vapor pressure wrt ice hpa
                e = es * nwp_rh_sfc[lat_idx, lon_idx] / 100.0  # vapor pressure in hPa
                td_sfc = 273.15 + 243.5 * log(e / 6.112) / (17.67 - log(e / 6.112))  # Dewpoint T in K
                nwp_lifting_condensation_level_height[lat_idx, lon_idx] = 1000. * 0.125 * (t - td_sfc)  # meters
                nwp_convective_condensation_level_height[lat_idx, lon_idx] = 1000. * (t - td_sfc) / 4.4  # meters
            else:
                td_sfc = nan
                nwp_lifting_condensation_level_height[lat_idx, lon_idx] = nan
                nwp_convective_condensation_level_height[lat_idx, lon_idx] = nan

            (
                nwp_level_free_convection_height[lat_idx, lon_idx],
                nwp_equilibrium_level_height[lat_idx, lon_idx]
            ) = compute_lfc_el_height(
                nwp_sfc_level[lat_idx, lon_idx],
                nwp_tropo_level[lat_idx, lon_idx],
                nwp_n_levels,
                nwp_p_std,
                nwp_t_prof[lat_idx, lon_idx],
                td_sfc,
                nwp_z_prof[lat_idx, lon_idx],
            )

    return (
        nwp_lifting_condensation_level_height, nwp_convective_condensation_level_height,
        nwp_level_free_convection_height, nwp_equilibrium_level_height,
    )


# ----------------------------------------------------------------
# compute level of free convection and equilibrium level
#
# input:
# p = pressure profile (hpa)
# t = temperature profile (k)
# td_sfc = dew point temperature at surface (k)
# z = height profile (m)
#
# output:
# lfc = level of free convection (m)
# el = equilibrium level (m)
# ----------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_lfc_el_height(sfc_level, tropo_level, n_levels, p, t, td_sfc, z):
    t_moist = np.full(n_levels, nan, 'f4')
    el = nan
    lfc = nan

    t_moist[sfc_level:n_levels] = td_sfc

    for lev_idx in range(sfc_level, tropo_level - 1, -1):
        malr = moist_adiabatic_lapse_rate(t[lev_idx], p[lev_idx])  # k/km
        t_moist[lev_idx - 1] = t_moist[lev_idx] - malr * (z[lev_idx - 1] - z[lev_idx]) / 1000.0  # z is m

        # --- level of free convection
        if isnan(lfc) and (t_moist[lev_idx - 1] > t[lev_idx - 1]) and (t_moist[lev_idx] < t[lev_idx]):
            lfc = z[lev_idx - 1]

        # --- find equilibrium level
        if isnan(el) and (t_moist[lev_idx - 1] < t[lev_idx - 1]) and (t_moist[lev_idx] > t[lev_idx]):
            el = z[lev_idx - 1]

    return lfc, el


# ----------------------------------------------------------------
# moist adiabatic lapse rate
#
# t = temperature (k)
# p = pressure (hpa)
#
# output
# malr = moist adiabatic lapse rate (k/km)
#
# method: http://www.theweatherprediction.com/habyhints/161/
#
# alternatel http://hogback.atmos.colostate.edu/group/dave/pdf/moist_adiabatic_lapse_rate.pdf
# ----------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def moist_adiabatic_lapse_rate(t, p):
    eps = 0.622
    dalr = 9.8
    l = 2.453e06  # j/kg latent heat of vaporization
    cp = 1004.0  # j/kg-k

    if t > 253.0:
        es = vapor(t)
    else:
        es = vapor_ice(t)

    ws = 0.622 * es / (p - es)

    tp = t + 1

    if tp > 253.0:
        esp = vapor(tp)
    else:
        esp = vapor_ice(tp)

    wsp = 0.622 * esp / (p - esp)

    dws_dt = wsp - ws

    malr = dalr / (1.0 + l / cp * dws_dt)

    return malr


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def modify_nwp_pix_t_sfc(
        bad_pixel_mask,
        nwp_pix_z_sfc,
        nwp_pix_t_sfc,
        nwp_pix_sfc_level,
        sfc_land,
        sfc_z_sfc,
        sfc_land_mask,
        sfc_snow,
        sst_anal,
        nwp_pix_t_prof,
        nwp_pix_z_prof,
):
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            if bad_pixel_mask[line_idx, elem_idx]:
                nwp_pix_t_sfc[line_idx, elem_idx] = nan
                continue

            sfc_level_idx = nwp_pix_sfc_level[line_idx, elem_idx]

            if sfc_land[line_idx, elem_idx] == sym_land:
                if (not isnan(nwp_pix_z_sfc[line_idx, elem_idx]) and
                        (not isnan(sfc_z_sfc[line_idx, elem_idx])) and
                        (sfc_level_idx > 0)):

                    # compute the near surface lapse rate (k/m)
                    delta_lapse_rate = (
                            (nwp_pix_t_prof[line_idx, elem_idx, sfc_level_idx] -
                             nwp_pix_t_prof[line_idx, elem_idx, sfc_level_idx - 1]) /
                            (nwp_pix_z_prof[line_idx, elem_idx, sfc_level_idx] -
                             nwp_pix_z_prof[line_idx, elem_idx, sfc_level_idx - 1])
                    )
                else:
                    delta_lapse_rate = 0

                # compute the pertubation to nwp surface temp to account for sub-grid elevation
                delta_z_sfc = sfc_z_sfc[line_idx, elem_idx] - nwp_pix_z_sfc[line_idx, elem_idx]  # meters
                delta_t_sfc = delta_lapse_rate * delta_z_sfc  # k
                nwp_pix_t_sfc[line_idx, elem_idx] = nwp_pix_t_sfc[line_idx, elem_idx] + delta_t_sfc  # k

            if (sfc_land_mask[line_idx, elem_idx] == sym_no and
                    sfc_snow[line_idx, elem_idx] == sym_no_snow and sst_anal[line_idx, elem_idx] > 270):
                nwp_pix_t_sfc[line_idx, elem_idx] = sst_anal[line_idx, elem_idx]

    return nwp_pix_t_sfc


# ======================================================================
# P_Cld_Top = Pressure of Cloud Top (tau = 1)
# ======================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_nwp_cloud_parameters(
        nwp_tropopause_level,
        nwp_surface_level,
        clwmr_profile,
        temperature_profile,
        pressure_profile
):
    max_temperature_ice = 273.15
    min_temperature_water = 253.15

    if nwp_tropopause_level < 0 or nwp_surface_level < 0:
        return nan

    lwp = 0.0
    iwp = 0.0
    cwp = 0.0

    for i_lay in range(nwp_tropopause_level - 1, nwp_surface_level):
        ice_frac_top = min(1.0, max(0.0,
                                    (max_temperature_ice - temperature_profile[i_lay]) /
                                    (max_temperature_ice - min_temperature_water)))
        ice_frac_bot = min(1.0, max(0.0,
                                    (max_temperature_ice - temperature_profile[i_lay + 1]) /
                                    (max_temperature_ice - min_temperature_water)))
        clwmr_ice_layer = 0.5 * (ice_frac_top * clwmr_profile[i_lay] +
                                 ice_frac_bot * clwmr_profile[i_lay + 1])
        clwmr_water_layer = 0.5 * ((1.0 - ice_frac_top) * clwmr_profile[i_lay] +
                                   (1.0 - ice_frac_bot) * clwmr_profile[i_lay + 1])
        # supercooled water ?

        factor = 1000.0 * 100.0 * (pressure_profile[i_lay + 1] - pressure_profile[i_lay]) / g

        iwp += clwmr_ice_layer * factor
        lwp += clwmr_water_layer * factor

        cwp = lwp + iwp

    return cwp


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_segment_nwp_cloud_parameters(
        nwp_n_lat, nwp_n_lon,
        nwp_tropo_level,
        nwp_sfc_level,
        nwp_clwmr_prof,
        nwp_t_prof,
        nwp_p_std,
):
    nwp_cwp = np.empty((nwp_n_lat, nwp_n_lon), 'f4')

    for lat_idx in prange(nwp_n_lat):
        for lon_idx in prange(nwp_n_lon):
            # todo check for bad pixels
            # todo check for space views
            nwp_cwp[lat_idx, lon_idx] = compute_nwp_cloud_parameters(
                nwp_tropo_level[lat_idx, lon_idx],
                nwp_sfc_level[lat_idx, lon_idx],
                nwp_clwmr_prof[lat_idx, lon_idx],
                nwp_t_prof[lat_idx, lon_idx],
                nwp_p_std
            )

    return nwp_cwp
