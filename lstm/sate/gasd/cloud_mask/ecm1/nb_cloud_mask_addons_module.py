from math import isnan

from numba import njit

from constants import (
    missing_value_int1,
)
from .nb_cloud_mask import (
    eumetcast_fire_day_sol_zen_thresh,
    eumetcast_fire_night_sol_zen_thresh,

    bt_375um_eumet_fire_day_thresh,
    bt_diff_eumet_fire_day_thresh,
    stddev_110um_eumet_fire_day_thresh,
    stddev_375um_eumet_fire_day_thresh,

    bt_375um_eumet_fire_night_thresh,
    bt_diff_eumet_fire_night_thresh,
    stddev_110um_eumet_fire_night_thresh,
    stddev_375um_eumet_fire_night_thresh,

    refl_065_min_smoke_water_thresh,
    refl_065_max_smoke_water_thresh,
    refl_160_max_smoke_water_thresh,
    refl_375_max_smoke_water_thresh,
    ems_11_tropo_max_smoke_water_thresh,
    t11_std_max_smoke_water_thresh,
    refl_065_std_max_smoke_water_thresh,
    btd_4_11_max_smoke_water_thresh,
    sol_zen_max_smoke_water_thresh,

    refl_065_min_smoke_land_thresh,
    refl_065_max_smoke_land_thresh,
    nir_smoke_ratio_max_land_thresh,
    refl_138_max_smoke_land_thresh,

    refl_375_max_smoke_land_thresh,
    ems_11_tropo_max_smoke_land_thresh,
    t11_std_max_smoke_land_thresh,
    refl_065_std_max_smoke_land_thresh,
    btd_4_11_max_smoke_land_thresh,
    sol_zen_max_smoke_land_thresh,

    btd_11_12_metric_max_dust_thresh,
    btd_11_12_max_dust_thresh,
    bt_11_std_max_dust_thresh,
    bt_11_12_clear_diff_max_dust_thresh,
    ems_11_tropo_max_dust_thresh,
    ems_11_tropo_min_dust_thresh,
    bt_11_clear_diff_min_dust_thresh,
    btd_85_11_max_dust_thresh,
    btd_85_11_min_dust_thresh,
)


@njit(nogil=True, error_model='numpy', boundscheck=True)
def split_window_test(t11_clear, t12_clear, t11, t12):
    if t11_clear <= 265.0:
        return t11 - t12
    return (t11 - t12) - (t11_clear - t12_clear) * (t11 - 260.0) / (t11_clear - 260.0)


# ----------------------------------------------------------------------------
# clavr-x smoke test
#
# daytime and ice-free ocean only.
#
# coded: andy heidinger
# ----------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def clavrx_smoke_over_water_test(
        refl_065, refl_065_clear,
        refl_160,
        refl_375, refl_375_clear,
        bt_375,
        bt_11, bt_11_clear,
        bt_12, bt_12_clear,
        ems_11_tropo, refl_065_std, t11_std, sol_zen
):
    if sol_zen < sol_zen_max_smoke_water_thresh:

        # --- ir test - smoke should be nearly invisible
        if ems_11_tropo > ems_11_tropo_max_smoke_water_thresh:
            return 0
        if t11_std > t11_std_max_smoke_water_thresh:
            return 0

        # --- vis test - smoke should be nearly invisible
        if (refl_065 - refl_065_clear > refl_065_max_smoke_water_thresh or
                refl_065 - refl_065_clear < refl_065_min_smoke_water_thresh):
            return 0
        if refl_065_std > refl_065_std_max_smoke_water_thresh:
            return 0

        # --- nir tests
        if refl_375 - refl_375_clear > refl_375_max_smoke_water_thresh:
            return 0
        if refl_375 > refl_375_max_smoke_water_thresh:
            return 0

        if refl_160 > refl_160_max_smoke_water_thresh:
            return 0

        # --- nir ir_tests
        if (bt_375 - bt_11) > btd_4_11_max_smoke_water_thresh:
            return 0

        # --- split_win_tests
        if abs(split_window_test(bt_11_clear, bt_12_clear, bt_11, bt_12)) > 1:
            return 0

        # --- combine into final answer
        return 1

    return missing_value_int1


# ----------------------------------------------------------------------------
# clavr-x smoke test over land
#
# daytime and snow-free land only.
#
# coded: andy heidinger
# ----------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def clavrx_smoke_over_land_test(
        refl_065, refl_065_clear,
        refl_086,
        refl_138,
        refl_160,
        refl_375, refl_375_clear,
        bt_375,
        bt_11, bt_11_clear,
        bt_12, bt_12_clear,
        ems_11_tropo, refl_065_std, t11_std, sol_zen
):
    if sol_zen < sol_zen_max_smoke_land_thresh:

        # --- ir test - smoke should be nearly invisible
        if ems_11_tropo > ems_11_tropo_max_smoke_land_thresh:
            return 0
        if t11_std > t11_std_max_smoke_land_thresh:
            return 0

        # --- vis test - smoke should be nearly invisible
        if (refl_065 - refl_065_clear > refl_065_max_smoke_land_thresh or
                refl_065 - refl_065_clear < refl_065_min_smoke_land_thresh):
            return 0
        if refl_065_std > refl_065_std_max_smoke_land_thresh:
            return 0

        # --- nir tests
        if refl_375 - refl_375_clear > refl_375_max_smoke_land_thresh:
            return 0
        if refl_375 > refl_375_max_smoke_land_thresh:
            return 0

        nir_smoke_ratio = (refl_160 - refl_065) / (refl_086 - refl_065)
        if nir_smoke_ratio > nir_smoke_ratio_max_land_thresh:
            return 0

        if refl_138 > refl_138_max_smoke_land_thresh:
            return 0

        # --- nir ir_tests
        if (bt_375 - bt_11) > btd_4_11_max_smoke_land_thresh:
            return 0

        # --- split_win_tests
        if abs(split_window_test(bt_11_clear, bt_12_clear, bt_11, bt_12)) > 1:
            return 0

        # --- combine into final answer
        return 1

    return missing_value_int1


# ----------------------------------------------------------------------------
# eumetcast fire detection algorithm
#
# reference: this implements the 'current operational algorithm' described in:
# towards an improved active fire monitoring product for msg satellites
# sauli joro, olivier samain, ahmet yildirim, leo van de berg, hans joachim lutz
# eumetsat, am kavalleriesand 31, darmstadt, germany
#
# coded by william straka iii
#
# -----------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def eumetsat_fire_test(t11, t375, t11_std, t375_std, sol_zen):
    # --- initialize
    # todo 做了修改

    # --- check if all needed data are non-missing
    if not (isnan(t375) or isnan(t375_std) or isnan(t11) or isnan(t11_std)):

        # day
        if sol_zen < eumetcast_fire_day_sol_zen_thresh:
            bt_375um_eumet_fire_thresh = bt_375um_eumet_fire_day_thresh
            bt_diff_eumet_fire_thresh = bt_diff_eumet_fire_day_thresh
            stddev_110um_eumet_fire_thresh = stddev_110um_eumet_fire_day_thresh
            stddev_375um_eumet_fire_thresh = stddev_375um_eumet_fire_day_thresh

        # night
        elif sol_zen > eumetcast_fire_night_sol_zen_thresh:
            bt_375um_eumet_fire_thresh = bt_375um_eumet_fire_night_thresh
            bt_diff_eumet_fire_thresh = bt_diff_eumet_fire_night_thresh
            stddev_110um_eumet_fire_thresh = stddev_110um_eumet_fire_night_thresh
            stddev_375um_eumet_fire_thresh = stddev_375um_eumet_fire_night_thresh

        # twilight
        else:
            # linear fit day -> night
            bt_375um_eumet_fire_thresh = ((-1.0) * sol_zen) + 380.0
            bt_diff_eumet_fire_thresh = ((-0.4) * sol_zen) + 36.0

            # these two don't change, but
            stddev_110um_eumet_fire_thresh = stddev_110um_eumet_fire_night_thresh
            stddev_375um_eumet_fire_thresh = stddev_375um_eumet_fire_night_thresh

        # all of these conditions need to be met
        if ((t375 > bt_375um_eumet_fire_thresh) and ((t375 - t11) > bt_diff_eumet_fire_thresh) and
                (t375_std > stddev_375um_eumet_fire_thresh) and (t11_std < stddev_110um_eumet_fire_thresh)):
            return 1

        return 0

    return missing_value_int1


# ---------------------------------------------------------------------------
# clavr-x ir dust algorithm
#
# coded: andy heidinger
# ---------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def clavrx_dust_test(
        bt_85,
        bt_11,
        bt_12,
        bt_11_clear,
        bt_12_clear,
        bt_11_std,
        ems_11_tropo
):
    # --- split window

    btd_11_12 = (bt_11 - bt_12)
    btd_11_12_metric = split_window_test(bt_11_clear, bt_12_clear, bt_11, bt_12)
    if btd_11_12 > btd_11_12_max_dust_thresh:
        return 0
    if btd_11_12_metric > btd_11_12_metric_max_dust_thresh:
        return 0
    if (bt_11 - bt_12) - (bt_11_clear - bt_12_clear) > bt_11_12_clear_diff_max_dust_thresh:
        return 0

    # --- 8.5-11 should be moderately negative.  ice clouds are positive
    # --- water clouds are very negative

    if (bt_85 - bt_11) > btd_85_11_max_dust_thresh:
        return 0
    if (bt_85 - bt_11) < btd_85_11_min_dust_thresh:
        return 0

    # --- 110um variability

    if bt_11_std > bt_11_std_max_dust_thresh:
        return 0

    # --- ir test - dust should have low to moderate emissivity

    if ems_11_tropo > ems_11_tropo_max_dust_thresh:
        return 0
    if ems_11_tropo < ems_11_tropo_min_dust_thresh:
        return 0
    if bt_11 - bt_11_clear < bt_11_clear_diff_min_dust_thresh:
        return 0

    return 1
