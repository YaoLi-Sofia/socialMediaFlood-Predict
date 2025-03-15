from math import nan, isnan

from numba import njit

et_cloudiness_class_space = 0
# et_cloudiness_class_missing = -999.0
et_cloudiness_class_cloudy = 3
et_cloudiness_class_prob_cloudy = 2
et_cloudiness_class_prob_clear = 1
et_cloudiness_class_clear = 0

# et_cloud_type_first = 0
et_cloud_type_clear = 0
et_cloud_type_prob_clear = 1
et_cloud_type_first_water = 2
et_cloud_type_fog = 2
et_cloud_type_water = 3
et_cloud_type_supercooled = 4
et_cloud_type_last_water = 4
et_cloud_type_mixed = 5
et_cloud_type_first_ice = 6
et_cloud_type_opaque_ice = 6
et_cloud_type_tice = 6
et_cloud_type_cirrus = 7
et_cloud_type_overlap = 8
et_cloud_type_overshooting = 9
et_cloud_type_last_ice = 9
et_cloud_type_unknown = 10
et_cloud_type_dust = 11
et_cloud_type_smoke = 12
et_cloud_type_fire = 13
# et_cloud_type_last = 13
et_cloud_type_missing = -128


# ====================================================================
# function name: height_h2o_channel
#
# function: estimate the cloud temperature/height/pressure
#
# description: use the 110um and 6.7um obs and the rtm cloud bb profiles
# to perform h2o intercept on a pixel level. filters
# restrict this to high clouds only
# 
# dependencies: 
#
# restrictions: 
#
# reference: 
#
# author: andrew heidinger, noaa/nesdis
#
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def height_h2o_channel(
        rad_110um,
        rad_110um_rtm_prof,
        rad_110um_clear,
        rad_h2o,
        rad_h2o_rtm_prof,
        rad_h2o_clear,
        covar_h2o_11,
        tropo_lev,
        sfc_lev,
        t_prof,
        z_prof
):
    rad_110um_thresh = 2.0
    rad_67um_thresh = 0.25
    bt_ch27_ch31_covar_cirrus_thresh = 1.0

    # -------------------------------
    t_cld = nan
    z_cld = nan

    # some tests
    if rad_h2o < 0:
        return t_cld, z_cld

    if rad_h2o_clear - rad_h2o < rad_67um_thresh:
        return t_cld, z_cld

    if rad_110um_clear - rad_110um < rad_110um_thresh:
        return t_cld, z_cld

    if not isnan(covar_h2o_11) and covar_h2o_11 < bt_ch27_ch31_covar_cirrus_thresh:
        return t_cld, z_cld

    # - colder than tropopause
    if rad_110um < rad_110um_rtm_prof[tropo_lev]:
        cld_lev = tropo_lev
    else:
        # --- determine linear regress of h2o (y) as a function of window (x)
        denominator = rad_110um - rad_110um_clear

        slope = (rad_h2o - rad_h2o_clear) / denominator
        intercept = rad_h2o - slope * rad_110um

        cld_lev = -1

        for idx_lev in range(tropo_lev + 1, sfc_lev + 1):
            rad_h2o_bb_prediction = slope * rad_110um_rtm_prof[idx_lev] + intercept
            if rad_h2o_bb_prediction < 0:
                continue

            if ((rad_h2o_bb_prediction > rad_h2o_rtm_prof[idx_lev - 1]) and (
                    rad_h2o_bb_prediction <= rad_h2o_rtm_prof[idx_lev])):
                cld_lev = idx_lev
                break

    # --- adjust back to full rtm profile indices
    if cld_lev >= 0:
        t_cld = t_prof[cld_lev]
        z_cld = z_prof[cld_lev]

    return t_cld, z_cld


# ====================================================================
# function name: height_opaque
#
# function: estimate the cloud temperature/height/pressure
#
# description: use the 110um obs and assume the cloud is back and 
# estimate height from 11 um bb cloud profile
# 
# dependencies: 
#
# restrictions: 
#
# reference: 
#
# author: andrew heidinger, noaa/nesdis
#
# ====================================================================
# -----------------------------------------------
# computes height parameters from nwp profiles
# called in get_ice_probability
# ------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def height_opaque(
        rad31,
        rad31_rtm_prof,
        tropo_lev,
        sfc_lev,
        t_prof,
        z_prof
):
    t_opa = nan
    z_opa = nan
    cld_lev = -1
    if rad31 < rad31_rtm_prof[tropo_lev]:
        cld_lev = tropo_lev
    else:
        # --- restrict levels to consider between tropo level and sfc level
        for idx_lev in range(tropo_lev, sfc_lev + 1):
            if rad31 > rad31_rtm_prof[idx_lev]:
                cld_lev = idx_lev

    # --- select opaque temperature from profile at chosen level
    if cld_lev >= 1:
        t_opa = t_prof[cld_lev]
        z_opa = z_prof[cld_lev]

    return t_opa, z_opa


# et_cloud_type_first = 0


# ====================================================================
# function name: compute_ice_probability_based_on_temperature
#
# function: provide the probability that this pixel is ice
#
# description:
# use the cloud temperature and an assumed relationship to
# determine the probability that the cloud is ice phase
#
# dependencies:
#
# restrictions:
#
# reference:
#
# author: andrew heidinger, noaa/nesdis
#
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_ice_probability_based_on_temperature(t_opa):
    ice_temperature_min = 243.0
    ice_temperature_max = 263.0

    ice_prob = 1.0 - (t_opa - ice_temperature_min) / (ice_temperature_max - ice_temperature_min)
    ice_prob = min(1.0, ice_prob)
    ice_prob = max(0.0, ice_prob)

    return ice_prob


# -----------------------------------------------------------------------------------
# computes ice probability and cirrus water flag and opaque temperature and height
# ------------------------------------------------------------------------------------

@njit(nogil=True, error_model='numpy', boundscheck=True)
def get_ice_probability(
        inp_sat_rad_ch14,
        inp_rtm_rad_ch14_bb_prof,
        inp_rtm_rad_ch14_atm_sfc,
        inp_sat_rad_ch09,
        inp_rtm_rad_ch09_bb_prof,
        inp_rtm_rad_ch09_atm_sfc,
        inp_rtm_covar_ch09_ch14_5x5,
        inp_rtm_tropo_lev,
        inp_rtm_sfc_lev,
        inp_rtm_t_prof,
        inp_rtm_z_prof,
        inp_sat_bt_ch14,
        inp_geo_sol_zen,
        inp_rtm_ref_ch05_clear,
        inp_sat_ref_ch05,
        inp_sfc_ems_ch07,
        inp_sat_ref_ch07,
        inp_sat_bt_ch11,
        inp_rtm_bt_ch14_3x3_std,
        inp_rtm_bt_ch14_atm_sfc,
        inp_rtm_bt_ch15_atm_sfc,
        inp_sat_bt_ch15,
        inp_rtm_bt_ch09_3x3_max
):
    fmft_cold_offset = 0.5  # k
    fmft_cirrus_thresh = 1.0  # k
    bt_110um_std_cirrus_thresh = 4.0  # k
    bt_85_minus_bt_11_test = -1.  # k

    is_water = False
    is_cirrus = False

    # ------------------------------------------------------------------------
    # determine the cloud height and cloud temperature for typing
    # ------------------------------------------------------------------------

    # ---- if possible, use water vapor channel
    t_cld, z_cld = height_h2o_channel(
        inp_sat_rad_ch14,
        inp_rtm_rad_ch14_bb_prof,
        inp_rtm_rad_ch14_atm_sfc,
        inp_sat_rad_ch09,
        inp_rtm_rad_ch09_bb_prof,
        inp_rtm_rad_ch09_atm_sfc,
        inp_rtm_covar_ch09_ch14_5x5,
        inp_rtm_tropo_lev,
        inp_rtm_sfc_lev,
        inp_rtm_t_prof,
        inp_rtm_z_prof,
    )

    # ---- if no water vapor, use the atmos-corrected 11 micron
    if isnan(t_cld):
        t_cld, z_cld = height_opaque(
            inp_sat_rad_ch14,
            inp_rtm_rad_ch14_bb_prof,
            inp_rtm_tropo_lev,
            inp_rtm_sfc_lev,
            inp_rtm_t_prof,
            inp_rtm_z_prof,
        )

    # --- last resort, use raw 11 micron brightness temperature
    if isnan(t_cld):
        t_cld = inp_sat_bt_ch14

        # --- compute the ice probability based on our guess of cloud temperature
    ice_prob = compute_ice_probability_based_on_temperature(t_cld)

    # ------------------------------------------------------------------------
    # modify ice_prob based on spectral tests for ice and water
    # ------------------------------------------------------------------------

    # --- tests for water
    if t_cld > 240.0:

        # 1.6 spectral test
        if inp_geo_sol_zen < 80. and inp_rtm_ref_ch05_clear < 20. and inp_sat_ref_ch05 > 30.:
            is_water = True

        # - 3.75 day
        if inp_geo_sol_zen < 80. and inp_sfc_ems_ch07 > 0.9 and inp_sat_ref_ch07 > 20.0:
            is_water = True

        # - 3.75 night
        if inp_geo_sol_zen > 80. and inp_sat_ref_ch07 > 5.0:
            is_water = True

        # - 8.5-11 test
        if (inp_sat_bt_ch11 - inp_sat_bt_ch14) < bt_85_minus_bt_11_test:
            is_water = True

        # --- modify ice_prob based on water tests
        if is_water:
            ice_prob = 0.0

    # --- tests for ice

    # ---- don't detect cirrus if very high 11 um std deviation
    if inp_rtm_bt_ch14_3x3_std < bt_110um_std_cirrus_thresh and not is_water:

        # - split window test
        if inp_rtm_bt_ch14_atm_sfc <= 265.0:
            h2o_correct = fmft_cold_offset
        else:
            h2o_correct = (inp_rtm_bt_ch14_atm_sfc - inp_rtm_bt_ch15_atm_sfc) * (inp_sat_bt_ch14 - 260.0) / (
                    inp_rtm_bt_ch14_atm_sfc - 260.0)

        fmft = inp_sat_bt_ch14 - inp_sat_bt_ch15 - h2o_correct

        if fmft > fmft_cirrus_thresh:
            is_cirrus = True

        # - 6.7 covariance test
        if inp_rtm_covar_ch09_ch14_5x5 > 1.5 and inp_rtm_bt_ch09_3x3_max < 250.0:
            is_cirrus = True

            # --- modify ice_prob based on cirrus tests
        if is_cirrus:
            ice_prob = 1.0

    return ice_prob, is_cirrus, is_water, t_cld, z_cld


# ---------------------------------------------------------------
# returns type for ice phase pixels ( ice_probability gt 0.5)
# ---------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def determine_type_ice(
        ems_tropo_11,
        bt_11,
        beta_11_12,
        beta_11_13,
        is_water,
        is_cirrus
):
    beta_110um_120um_overlap_thresh = 0.95
    beta_110um_133um_overlap_thresh = 0.70

    c_type = et_cloud_type_opaque_ice

    # ------------------------------------------------------------------------
    # define cirrus vs opaque by ems_tropo thresholds
    # ------------------------------------------------------------------------
    if ems_tropo_11 < 0.8:
        c_type = et_cloud_type_cirrus

        if 0. < beta_11_12 < beta_110um_120um_overlap_thresh:
            c_type = et_cloud_type_overlap

        if 0. < beta_11_13 < beta_110um_133um_overlap_thresh:
            c_type = et_cloud_type_overlap

        if is_cirrus and is_water:
            c_type = et_cloud_type_overlap

    # --- assume clouds colder than homo. freezing point are opaque this should be evaluated
    if bt_11 < 233.0:
        c_type = et_cloud_type_opaque_ice

    # --- define deep convective cloud based on ems_tropo near or greater than 1
    if ems_tropo_11 > 0.95:
        c_type = et_cloud_type_overshooting

    return c_type


# ---------------------------------------------------------------
# returns type for water phase pixels ( ice_probability lt 0.5)
# ---------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def determine_type_water(z_opa, t_opa):
    c_type = et_cloud_type_water
    if t_opa < 273.0 and not isnan(t_opa):
        c_type = et_cloud_type_supercooled
    else:
        if z_opa < 1000.0 and not isnan(z_opa):
            c_type = et_cloud_type_fog
    return c_type
