from math import exp

from numba import njit


@njit(nogil=True, error_model='numpy', boundscheck=True)
def dncomp_trans_atm_above_cloud(
        tpw_ac,
        ozone_dobson,
        press_sfc,
        press_cld,
        air_mass,
        gas_coeff,
        ozone_coeff,
        rayleigh_coeff,
):
    assumed_wvp_error = 1.2

    trans_ozone = exp(-(
            ozone_coeff[0] +
            ozone_coeff[1] * ozone_dobson +
            ozone_coeff[2] * ozone_dobson ** 2
    ))
    trans_ozone = min(trans_ozone, 1.0)
    trans_ozone = max(trans_ozone, 0.0)

    trans_ozone_unc = trans_ozone - exp(-1.0 * (
            ozone_coeff[0] +
            ozone_coeff[1] * (1.1 * ozone_dobson) +
            ozone_coeff[2] * (1.1 * ozone_dobson) ** 2
    ))

    trans_ozone_unc = max(min(trans_ozone_unc, 0.02), 0.0)

    trans_rayleigh = exp(-air_mass * (0.044 * (press_cld / press_sfc)) * 0.84)

    trans_wvp = exp(-(
            gas_coeff[0] +
            gas_coeff[1] * tpw_ac +
            gas_coeff[2] * (tpw_ac ** 2)
    ))

    trans_wvp = min(trans_wvp, 1.)

    trans_wvp_unc = abs(trans_wvp - exp(-(
            gas_coeff[0] +
            gas_coeff[1] * (assumed_wvp_error * tpw_ac) +
            gas_coeff[2] * ((assumed_wvp_error * tpw_ac) ** 2)
    )))

    trans_wvp_unc = max(min(trans_wvp_unc, 0.1), 0.0)

    trans_rayleigh = 1.0
    transmission = trans_ozone * trans_rayleigh * trans_wvp
    transmission = min(transmission, 1.0)
    transmission = max(transmission, 0.0)
    trans_unc = trans_ozone_unc + trans_wvp_unc

    return transmission, trans_unc


@njit(nogil=True, error_model='numpy', boundscheck=True)
def trans_atm_above_cloud(
        tpw_ac,
        ozone_dobson,
        press_sfc,
        press_cld,
        air_mass,
        gas_coeff_inp,
        ozone_coeff_inp,
        rayleigh_coeff_inp,

):
    trans, trans_uncert = dncomp_trans_atm_above_cloud(
        tpw_ac,
        ozone_dobson,
        press_sfc,
        press_cld,
        air_mass,
        gas_coeff_inp,
        ozone_coeff_inp,
        rayleigh_coeff_inp,
    )

    trans_uncert = 0.02

    return trans, trans_uncert
