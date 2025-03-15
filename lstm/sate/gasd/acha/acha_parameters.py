# $id: acha_parameters.inc 4092 2021-03-02 22:05:14z heidinger $
# ----------------------------------------------------------------------
#
# name:
#   acha_parameters.inc
#
# author: andrew heidinger, noaa/nesdis
#
# description:
#   include file that declares a number of physical constants,
#   apriori estimates and uncertainty values
#
# multi_layer_logic_flag
# 0 - (baseline) just use the multilayer id in cloud type
# 1 - treat all multilayer like cirrus (force single layer)
# 2 - assume all cirrus are multilayer and let acha decide
#
# use_type_flag
# 0 - use input type to set a priori
# 1 - use input phase and emissivity tropo to set a priori
# 2 - do not use input phase/type for setting a priori
#
# ----------------------------------------------------------------------

# -----------------------------------------------------------
# include other include files to this file
# -----------------------------------------------------------

# --- clear-sky covariance terms

# -----------------------------------------------------------
# isolate user controlled flags here
# -----------------------------------------------------------

ice_extinction_tuning_factor = 1.0

dt_dz_strato = -0.0065  # k/m
dp_dz_strato = -0.0150  # hpa/m
sensor_zenith_threshold = 88.0  # 70.0

# --- set starting index value
# element_idx_min = 1
# line_idx_min = 1
# lrc_meander_flag = 1


# --- cirrus box parameters
emissivity_min_cirrus = 0.7
cirrus_box_width_km = 400  # 200

# ---------------------------------------------------------------------
# retrieval specific parameters
# ---------------------------------------------------------------------
num_levels_rtm_prof = 101

# --- these parameters control the size of matrices
num_param_simple = 3  # number of retrieved parameters

# --- maximum number of iterations allowed
iter_idx_max = 10  # maximum number of iterations

# --- limits on steps taken in retrieval
delta_x_max = (20.0, 0.1, 0.1, 5.0, 0.2)

# --- parameters that control the bottom-up estimation of zc and pc
max_delta_t_inversion = 20.0  # max temperature difference (surface -cloud) to look for low-level inversion

# --- the parameters that provide the apriori values and their uncertainties
# real(kind=real4), parameter, private:: tc_ap_tropo_offset_cirrus = 15.0      #apriori tc for opaque clouds

tc_ap_uncer_opaque = 20.0  # 10.0            #apriori uncertainty of tc for opaque clouds

tc_ap_uncer_cirrus_default = 30.0  # apriori uncertainty of tc for cirrus
# originally it was 20 in the baseline version of acha


ec_ap_uncer_opaque = 0.2  # 0.1        #apriori uncertainty of ec for opaque clouds
ec_ap_uncer_cirrus = 0.8  # 0.4

beta_ap_water = 1.3
beta_ap_uncer_water = 0.2  # apriori uncertainty of  beta for ice
beta_ap_ice = 1.06
beta_ap_uncer_ice = 0.2  # apriori uncertainty of  beta for water

tau_ap_fog_type = 1.2  # apriori estimate of tau for fog cloud type
tau_ap_water_type = 2.3  # apriori estimate of tau for water cloud type
tau_ap_supercooled_type = 2.3  # apriori estimate of tau for mixed cloud type
tau_ap_mixed_type = 2.3  # apriori estimate of tau for mixed cloud type
tau_ap_opaque_ice_type = 2.3  # apriori estimate of tau for opaque ice cloud type
tau_ap_cirrus_type = 0.9  # apriori estimate of tau for cirrus cloud type
tau_ap_overlap_type = 0.9  # 2.0           #apriori estimate of tau for multilayer cloud type

# --- specify calibration  errors
# --> real(kind=real4), parameter, private:: t110um_120um_cal_uncer = 1.0  #baseline v5
# --> real(kind=real4), parameter, private:: t110um_133um_cal_uncer = 2.0 #baseline v5
t110um_cal_uncer = 1.0
t110um_120um_cal_uncer = 1.0
t110um_133um_cal_uncer = 2.0

# --- these parameters constrain the allowable solutions
min_allowable_tc = 170.0  # k

# parameter that controls observed negative heights and pressures.
zc_floor = 75.0  # set a bottom limit to zc.

# --- cirrus apriori for ice cloud temperature relative to tropopause
# ---- computed in 10 deg lat bands.  first bin is -90 to -80
num_lat_cirrus_ap = 18

tc_cirrus_mean_lat_vector = (
    7.24244, 8.70593, 14.0095, 14.5873, 14.7501, 17.9235,
    22.4626, 15.1277, 14.4136, 13.9811, 15.1359, 21.6522,
    18.0842, 14.8196, 13.4467, 13.8617, 14.5733, 14.9950
)

# --- parameters that allow for cloud geometrical thickness estimation
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
