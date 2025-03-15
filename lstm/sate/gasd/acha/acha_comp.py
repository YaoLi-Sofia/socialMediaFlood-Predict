from math import exp, log, log10, nan, isnan

import numpy as np
from numba import prange, njit

from constants import (
    sym_clear_type,
    sym_prob_clear_type,
    sym_fog_type,
    sym_water_type,
    sym_supercooled_type,
    sym_tice_type,
    sym_cirrus_type,
    sym_overlap_type,
    sym_overshooting_type,
    sym_water_phase,
    sym_ice_phase,
    sym_unknown_phase
)
from public import (
    image_shape,
    image_number_of_lines,
    image_number_of_elements,
)
from .acha_ice_cloud_microphysical_model_ahi_110um import (
    re_beta_110um_coef_ice, qe_006um_coef_ice, qe_110um_coef_ice,
    wo_110um_coef_ice, g_110um_coef_ice,
)
from .acha_microphysical_module import (
    re_beta_110um_coef_water, qe_006um_coef_water, qe_110um_coef_water,
    wo_110um_coef_water, g_110um_coef_water,
)
from .acha_parameters import beta_ap_water, beta_ap_ice, sensor_zenith_threshold


# ------------------------------------------------------------------------------
# awg cloud optical and microphysical algorithm (acha-comp)
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
#      output_ec  (modified from acha)
#      output_reff
#      output_beta (modified from acha)
#
# ----------------------------------------------------------------------
# modification history
#
# ------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def acha_comp_algorithm(
        input_cloud_type, input_sensor_zenith_angle, input_cosine_zenith_angle,
        output_zc,
        output_ec, output_beta,
):
    output_tau = np.full(image_shape, nan, 'f4')
    output_reff = np.full(image_shape, nan, 'f4')

    for line_idx in prange(image_number_of_lines):
        for elem_idx in prange(image_number_of_elements):

            # --- for convenience, save nwp indices to local variables
            cloud_type = input_cloud_type[line_idx, elem_idx]

            # ----------------------------------------------------------------------
            # for clear pixels, set opd to zero and reff to missing
            # ----------------------------------------------------------------------
            if ((cloud_type in (sym_clear_type, sym_prob_clear_type) and
                 input_sensor_zenith_angle[line_idx, elem_idx] <= sensor_zenith_threshold)):
                # output_ec[line_idx, elem_idx] = 0.0
                output_tau[line_idx, elem_idx] = 0.0
                output_reff[line_idx, elem_idx] = nan
                # output_beta[line_idx, elem_idx] = nan
                continue

            # ----------------------------------------------------------------------
            # determine cloud phase from cloud type for convenience
            # ----------------------------------------------------------------------
            cloud_phase = sym_unknown_phase

            if cloud_type in (sym_fog_type, sym_water_type, sym_supercooled_type):
                cloud_phase = sym_water_phase

            if cloud_type in (sym_cirrus_type, sym_overlap_type, sym_tice_type, sym_overshooting_type):
                cloud_phase = sym_ice_phase

            # -----------------------------------------------------------------------------
            # estimate cloud optical and microphysical properties
            # -----------------------------------------------------------------------------
            ec_slant = output_ec[line_idx, elem_idx]

            if ((not isnan(output_zc[line_idx, elem_idx])) and
                    (not isnan(output_ec[line_idx, elem_idx])) and
                    (not isnan(output_beta[line_idx, elem_idx]))):

                # --- save nadir adjusted emissivity and optical depth
                if output_ec[line_idx, elem_idx] < 1.00:
                    output_ec[line_idx, elem_idx], output_tau[line_idx, elem_idx], output_reff[
                        line_idx, elem_idx] = compute_tau_reff_acha(
                        output_beta[line_idx, elem_idx],
                        input_cosine_zenith_angle[line_idx, elem_idx],
                        cloud_phase,
                        ec_slant
                    )
                else:
                    output_tau[line_idx, elem_idx] = 20.0
                    output_ec[line_idx, elem_idx] = 1.0

                    if cloud_phase == sym_ice_phase:
                        output_beta[line_idx, elem_idx] = beta_ap_ice
                        output_reff[line_idx, elem_idx] = 20.0
                    else:
                        output_beta[line_idx, elem_idx] = beta_ap_water
                        output_reff[line_idx, elem_idx] = 10.0

    return output_ec, output_beta, output_tau, output_reff


# ---------------------------------------------------------------------------
#
# ---------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_tau_reff_acha(beta, cosine_zenith_angle, cloud_phase, ec_slant):
    reff_min = 1.0
    reff_max = 60.0
    tau_max = 8.0

    tau = nan
    ec = nan

    if cloud_phase == sym_ice_phase:
        temp_r4 = (
                re_beta_110um_coef_ice[0] +
                re_beta_110um_coef_ice[1] * (beta - 1.0) +
                re_beta_110um_coef_ice[2] * (beta - 1.0) ** 2 +
                re_beta_110um_coef_ice[3] * (beta - 1.0) ** 3
        )
    else:
        temp_r4 = (
                re_beta_110um_coef_water[0] +
                re_beta_110um_coef_water[1] * (beta - 1.0) +
                re_beta_110um_coef_water[2] * (beta - 1.0) ** 2 +
                re_beta_110um_coef_water[3] * (beta - 1.0) ** 3
        )

    reff = max(reff_min, min(reff_max, 10.0 ** (1.0 / temp_r4)))  # note inverse here

    if reff > 0.0:
        log10_reff = log10(reff)
    else:
        return ec, tau, reff

    if cloud_phase == sym_ice_phase:
        qe_vis = (
                qe_006um_coef_ice[0] +
                qe_006um_coef_ice[1] * log10_reff +
                qe_006um_coef_ice[2] * log10_reff ** 2
        )

        qe_110um = (
                qe_110um_coef_ice[0] +
                qe_110um_coef_ice[1] * log10_reff +
                qe_110um_coef_ice[2] * log10_reff ** 2
        )

        wo_110um = (
                wo_110um_coef_ice[0] +
                wo_110um_coef_ice[1] * log10_reff +
                wo_110um_coef_ice[2] * log10_reff ** 2
        )

        g_110um = (
                g_110um_coef_ice[0] +
                g_110um_coef_ice[1] * log10_reff +
                g_110um_coef_ice[2] * log10_reff ** 2
        )

    else:
        qe_vis = (
                qe_006um_coef_water[0] +
                qe_006um_coef_water[1] * log10_reff +
                qe_006um_coef_water[2] * log10_reff ** 2
        )

        qe_110um = (
                qe_110um_coef_water[0] +
                qe_110um_coef_water[1] * log10_reff +
                qe_110um_coef_water[2] * log10_reff ** 2
        )

        wo_110um = (
                wo_110um_coef_water[0] +
                wo_110um_coef_water[1] * log10_reff +
                wo_110um_coef_water[2] * log10_reff ** 2
        )

        g_110um = (
                g_110um_coef_water[0] +
                g_110um_coef_water[1] * log10_reff +
                g_110um_coef_water[2] * log10_reff ** 2
        )

    tau_abs_110um = -cosine_zenith_angle * log(1.0 - ec_slant)

    ec = 1.0 - exp(-tau_abs_110um)

    tau = min((qe_vis / qe_110um) * tau_abs_110um / (1.0 - wo_110um * g_110um), tau_max)

    # --- set negative values to be missing 
    if tau < 0:
        tau = nan
        reff = nan

    return ec, tau, reff
