# $id: awg_cloud_height.f90 583 2014-10-08 03:43:36z heidinger $

from math import floor, nan, isnan

# ---------------------------------------------------------------------
# this module houses the routines associated with...
#
# author:
#
# reference:
#
# ----------------------------------------------------------------------
import numpy as np
from numba import njit, prange

from constants import (
    sym_opaque_ice_type,
    sym_cirrus_type,
    sym_overlap_type,
    sym_overshooting_type,
    sym_yes,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)

water_extinction = 25.00  # 1/km

ice_extinction1 = 1.71
ice_extinction2 = 1.87
ice_extinction3 = 2.24
ice_extinction4 = 2.88
ice_extinction5 = 4.74

cirrus_extinction1 = 0.13
cirrus_extinction2 = 0.25
cirrus_extinction3 = 0.39
cirrus_extinction4 = 0.55
cirrus_extinction5 = 0.67


# ------------------------------------------------------------------------------
# cloud base height algorithm
#
# author: andrew heidinger, noaa
#
# assumptions
#
# limitations
#
# note.  this algorithm use the same input and output structures as 
#        the awg_cloud_height_algorithm.
#        do not overwrite elements of the output structure expect those
#        generated here.
#
#      output_tau
#      output_ec
#      output_reff
#      output_zc_top
#      output_zc_base
#      output_pc_top
#      output_pc_base
#
# ----------------------------------------------------------------------
# modification history
#
# ------------------------------------------------------------------------------


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def cloud_base_algorithm(
        input_invalid_data_mask,
        input_cloud_type,
        rtm_surface_level,  # rtm_tropo_level,
        # rtm_p_std, rtm_t_prof,
        rtm_z_prof,
        input_surface_elevation,
        input_tau, input_tc, input_zc,
        input_cwp, input_nwp_cwp,
        input_lcl, input_ccl,
):
    # output_zc_base_qf = np.ones(image_shape, 'i1')
    output_zc_base = np.full(image_shape, nan, 'f4')
    # output_pc_base = np.full(image_shape, nan, 'f4')
    # output_tc_base = np.full(image_shape, nan, 'f4')
    # output_pc_lower_base = np.full(image_shape, nan, 'f4')
    # output_geo_thickness = np.full(image_shape, nan, 'f4')

    # --------------------------------------------------------------------------
    # loop over pixels in scanlines
    # --------------------------------------------------------------------------
    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):
            if input_invalid_data_mask[line_idx, elem_idx] == sym_yes:
                continue

            # --- for convenience, save nwp indices to local variables
            cloud_type = input_cloud_type[line_idx, elem_idx]

            # -----------------------------------------------------------------------
            # include code to setup local profiles correctly
            # -----------------------------------------------------------------------

            # call services module
            rtm_sfc_level = rtm_surface_level[line_idx, elem_idx]
            # tropo_level_rtm = rtm_tropo_level[line_idx, elem_idx]
            # press_prof_rtm = rtm_p_std

            hgt_prof_rtm = rtm_z_prof[line_idx, elem_idx]
            # temp_prof_rtm = rtm_t_prof[line_idx, elem_idx]

            # -----------------------------------------------------------------------------
            # --- cloud base and top
            # ---
            # --- note 1. extinction values are in km^(-1)
            # --- note 2. all heights and thickness are converted to meters
            # -----------------------------------------------------------------------------
            if (not isnan(input_zc[line_idx, elem_idx])) and (not isnan(input_tau[line_idx, elem_idx])):

                cloud_extinction = water_extinction

                if cloud_type in (sym_opaque_ice_type, sym_overshooting_type):
                    if input_tc[line_idx, elem_idx] < 200:
                        cloud_extinction = ice_extinction1
                    elif input_tc[line_idx, elem_idx] < 220:
                        cloud_extinction = ice_extinction2
                    elif input_tc[line_idx, elem_idx] < 240:
                        cloud_extinction = ice_extinction3
                    elif input_tc[line_idx, elem_idx] < 260:
                        cloud_extinction = ice_extinction4
                    else:
                        cloud_extinction = ice_extinction5

                if cloud_type in (sym_cirrus_type, sym_overlap_type):
                    if input_tc[line_idx, elem_idx] < 200:
                        cloud_extinction = cirrus_extinction1
                    elif input_tc[line_idx, elem_idx] < 220:
                        cloud_extinction = cirrus_extinction2
                    elif input_tc[line_idx, elem_idx] < 240:
                        cloud_extinction = cirrus_extinction3
                    elif input_tc[line_idx, elem_idx] < 260:
                        cloud_extinction = cirrus_extinction4
                    else:
                        cloud_extinction = cirrus_extinction5

                cloud_geometrical_thickness = input_tau[line_idx, elem_idx] / cloud_extinction  # (km)
                cloud_geometrical_thickness *= 1000.0  # (m)

                # output_geo_thickness[line_idx, elem_idx] = cloud_geometrical_thickness

                # if input_tau[line_idx, elem_idx] < 2.0:
                #     cloud_geometrical_thickness_top_offset = cloud_geometrical_thickness / 2.0  # (m)
                # else:
                #     cloud_geometrical_thickness_top_offset = 1000.0 / cloud_extinction  # (m)

                # zc_top_max = hgt_prof_rtm[tropo_level_rtm]
                zc_base_min = hgt_prof_rtm[rtm_sfc_level]

                # output_zc_base_qf[line_idx, elem_idx] = 1

                output_zc_base[line_idx, elem_idx] = min(
                    input_zc[line_idx, elem_idx], max(
                        zc_base_min,
                        input_zc[line_idx, elem_idx] - cloud_geometrical_thickness)
                )
                # if input_zc[line_idx, elem_idx] - cloud_geometrical_thickness < zc_base_min:
                #     output_zc_base_qf[line_idx, elem_idx] = 2
                if input_zc[line_idx, elem_idx] - cloud_geometrical_thickness >= input_zc[line_idx, elem_idx]:
                    # output_zc_base_qf[line_idx, elem_idx] = 4
                    output_zc_base[line_idx, elem_idx] = nan

                if (input_tau[line_idx, elem_idx] > 1.0 and (
                        input_cwp[line_idx, elem_idx] > 0 or input_nwp_cwp[line_idx, elem_idx] > 0)):
                    output_zc_base[line_idx, elem_idx] = cira_base_hgt(
                        input_zc[line_idx, elem_idx],
                        input_cwp[line_idx, elem_idx],
                        input_nwp_cwp[line_idx, elem_idx],
                        input_lcl[line_idx, elem_idx],
                        input_ccl[line_idx, elem_idx],
                        input_surface_elevation[line_idx, elem_idx],
                    )

                # output_pc_base[line_idx, elem_idx], output_tc_base[line_idx, elem_idx], i_lev = knowing_z_compute_t_p(
                #     output_zc_base[line_idx, elem_idx], press_prof_rtm, temp_prof_rtm, hgt_prof_rtm
                # )
                #
                # output_pc_lower_base[line_idx, elem_idx], r4_dummy, i_lev = knowing_z_compute_t_p(
                #     max(input_surface_elevation[line_idx, elem_idx],
                #         min(output_zc_base[line_idx, elem_idx],
                #             (input_lcl[line_idx, elem_idx] + input_ccl[line_idx, elem_idx]) * 0.5)),
                #     press_prof_rtm, temp_prof_rtm, hgt_prof_rtm
                # )

    return output_zc_base


# -----------------------------------------------------------------
# cira's base code, interpret from idl codes
# -----------------------------------------------------------------
# ynoh (cira/csu)
@njit(nogil=True, error_model='numpy', boundscheck=True)
def cira_base_hgt(zc, cwp, nwp_cwp, lcl, ccl, surf_elev):
    #                      min cth       ;slope         y-int          r2      n    median cwp
    n_bin = 9
    n_para = 6
    n_cwp = 2
    min_cbh = 0.0
    max_cbh = 20.0 * 1000

    regr_coeff = np.empty((n_bin, n_para, n_cwp), 'f4')
    regr_coeff[0, :, 0] = [0.00000, 2.25812, 0.405590, 0.0236532, 5921., 0.0710000]
    regr_coeff[0, :, 1] = [0.00000, 0.997031, 0.516989, 0.0900793, 5881., 0.0710000]
    regr_coeff[1, :, 0] = [2.00000, 6.10980, 0.664818, 0.0664282, 3624., 0.114000]
    regr_coeff[1, :, 1] = [2.00000, 0.913021, 1.35698, 0.0735795, 3621., 0.114000]
    regr_coeff[2, :, 0] = [4.00000, 11.5574, 1.22527, 0.0519277, 2340., 0.110000]
    regr_coeff[2, :, 1] = [4.00000, 1.37922, 2.58661, 0.0695758, 2329., 0.110000]
    regr_coeff[3, :, 0] = [6.00000, 14.5382, 1.70570, 0.0568334, 2535., 0.123000]
    regr_coeff[3, :, 1] = [6.00000, 1.68711, 3.62280, 0.0501604, 2511., 0.123000]
    regr_coeff[4, :, 0] = [8.00000, 9.09855, 2.14247, 0.0218789, 3588., 0.131000]
    regr_coeff[4, :, 1] = [8.00000, 2.45953, 3.86957, 0.0727178, 3579., 0.131000]
    regr_coeff[5, :, 0] = [10.0000, 13.5772, 1.86554, 0.0497041, 4249., 0.127000]
    regr_coeff[5, :, 1] = [10.0000, 4.83087, 3.53141, 0.160008, 4218., 0.127000]
    regr_coeff[6, :, 0] = [12.0000, 16.0793, 1.64965, 0.0695903, 3154., 0.115000]
    regr_coeff[6, :, 1] = [12.0000, 5.05173, 3.98610, 0.180965, 3121., 0.115000]
    regr_coeff[7, :, 0] = [14.0000, 14.6030, 2.00010, 0.0429476, 2744., 0.116000]
    regr_coeff[7, :, 1] = [14.0000, 6.06439, 4.03301, 0.239837, 2717., 0.116000]
    regr_coeff[8, :, 0] = [16.0000, 9.26580, 2.29640, 0.0113376, 1455., 0.0990000]
    regr_coeff[8, :, 1] = [16.0000, 6.60431, 3.26442, 0.227116, 1449., 0.0990000]

    # qf = np.array([0, 1, 2, 3, 4], 'i1')
    # cbh_qf = qf[0]

    # start retrieval
    zc_local = zc / 1000.
    cwp_local = cwp / 1000.
    nwp_cwp_local = nwp_cwp / 1000.

    # force large cwp to cap at 1.2 kg/m2
    if zc_local > 20.0:
        zc_local = 20.0
    if cwp_local > 1.2:
        cwp_local = 1.2
    if nwp_cwp_local > 1.2:
        nwp_cwp_local = 1.2
    if cwp_local < 0 and nwp_cwp_local > 0:
        cwp_local = nwp_cwp_local

    z_delta = 2.0
    i_bin = int(floor(zc_local) / floor(z_delta))
    i_bin_max = 8

    if zc_local > 18.0 or i_bin > i_bin_max:
        i_bin = i_bin_max

    i_cwp = 0
    if cwp_local > regr_coeff[i_bin, 5, 0]:
        i_cwp = 1

    slope = regr_coeff[i_bin, 1, i_cwp]
    intercept = regr_coeff[i_bin, 2, i_cwp]
    cloud_geometrical_thickness = slope * cwp_local + intercept
    cloud_geometrical_thickness *= 1000.

    zc_base = zc_local * 1000 - cloud_geometrical_thickness

    # --------
    # ynoh (cira/csu)
    # an adjustment for large cwp greater than 1.0 kg/m2 (no cloud type involved)
    # updated for a smooth transition (20170109)
    # updated for (lcl+ccl)*0.5 (20171227)
    if surf_elev < (lcl + ccl) * 0.5 < zc_base:
        if cwp_local >= 1.2:
            zc_base = (lcl + ccl) * 0.5
        if 1.0 <= cwp_local < 1.2:
            zc_base = zc_base + ((lcl + ccl) * 0.5 - zc_base) * ((cwp_local - 1.0) / (1.2 - 1.0))

        # --------

    # apply quality flag
    if zc_base < surf_elev:
        zc_base = surf_elev

    if zc_base < min_cbh or zc_base > max_cbh:
        zc_base = nan

    if zc_base >= zc_local * 1000:
        zc_base = nan

    return zc_base
