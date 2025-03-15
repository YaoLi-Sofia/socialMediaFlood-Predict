from math import nan, isnan

import numpy as np
import xarray as xr
from numba import njit

from interp import GlobalNearestInterpolator

ds = xr.open_dataset('static/nb_cloud_mask_calipso_prior.nc')

prior_table = ds['cloud_fraction_table_smoothed'].values

prior_table_lon = (np.arange(36, dtype='f4') - 17.5) * 10
prior_table_lat = (np.arange(18, dtype='f4') - 8.5) * 10

ds = xr.open_dataset('static/nb_cloud_mask_default_2d.nc')

n_bins_metric_110um_std_default_2d = int(ds.attrs['nbins_metric_11umstd'])  # 100
metric_110um_std_min_default_2d = ds.attrs['metric_11umstd_min']
metric_110um_std_bin_default_2d = ds.attrs['metric_11umstd_bin']

prob_clear_default_2d = ds['clear_prob_table'].values
prob_water_default_2d = ds['water_prob_table'].values
prob_ice_default_2d = ds['ice_prob_table'].values
prob_obs_default_2d = ds['obs_prob_table'].values

ds = xr.open_dataset('static/nb_cloud_mask_night_2d.nc')

n_bins_metric_110um_night_2d = int(ds.attrs['nbins_metric_11um'])  # 80
metric_110um_min_night_2d = ds.attrs['metric_11um_min']
metric_110um_bin_night_2d = ds.attrs['metric_11um_bin']
n_bins_metric_375um_night_2d = int(ds.attrs['nbins_metric_375um'])  # 100
metric_375um_min_night_2d = ds.attrs['metric_375um_min']
metric_375um_bin_night_2d = ds.attrs['metric_375um_bin']

prob_clear_night_2d = ds['clear_prob_table'].values
prob_water_night_2d = ds['water_prob_table'].values
prob_ice_night_2d = ds['ice_prob_table'].values
prob_obs_night_2d = ds['obs_prob_table'].values

ds = xr.open_dataset('static/nb_cloud_mask_day_2d_160um.nc')

n_bins_metric_110um_day_160_2d = int(ds.attrs['nbins_metric_11um'])  # 80
metric_110um_min_day_160_2d = ds.attrs['metric_11um_min']
metric_110um_bin_day_160_2d = ds.attrs['metric_11um_bin']
n_bins_metric_160um_day_160_2d = int(ds.attrs['nbins_metric_160um'])  # 100
metric_160um_min_day_160_2d = ds.attrs['metric_160um_min']
metric_160um_bin_day_160_2d = ds.attrs['metric_160um_bin']

prob_clear_day_160_2d = ds['clear_prob_table'].values
prob_water_day_160_2d = ds['water_prob_table'].values
prob_ice_day_160_2d = ds['ice_prob_table'].values
prob_obs_day_160_2d = ds['obs_prob_table'].values

ds = xr.open_dataset('static/nb_cloud_mask_day_2d_375um.nc')

n_bins_metric_110um_day_375_2d = int(ds.attrs['nbins_metric_11um'])  # 80
metric_110um_min_day_375_2d = ds.attrs['metric_11um_min']
metric_110um_bin_day_375_2d = ds.attrs['metric_11um_bin']
n_bins_metric_375um_day_375_2d = int(ds.attrs['nbins_metric_375um'])  # 100
metric_375um_min_day_375_2d = ds.attrs['metric_375um_min']
metric_375um_bin_day_375_2d = ds.attrs['metric_375um_bin']

prob_clear_day_375_2d = ds['clear_prob_table'].values
prob_water_day_375_2d = ds['water_prob_table'].values
prob_ice_day_375_2d = ds['ice_prob_table'].values
prob_obs_day_375_2d = ds['obs_prob_table'].values


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_prior(lon, lat, month):
    diurnal = 0  # 0 = daily averaged
    interpolator = GlobalNearestInterpolator(prior_table_lon, prior_table_lat, lon, lat)
    prior = interpolator.interp(prior_table[diurnal, month - 1])
    return prior


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_default_core_probability(metric_110um, metric_110um_std, i):
    k = int((metric_110um - metric_110um_min_night_2d) / metric_110um_bin_night_2d)
    k = min(n_bins_metric_110um_night_2d - 1, max(0, k))
    j = int((metric_110um_std - metric_110um_std_min_default_2d) / metric_110um_std_bin_default_2d)
    j = min(n_bins_metric_110um_std_default_2d - 1, max(0, j))

    prob_clear = prob_clear_default_2d[i, j, k]
    prob_water = prob_water_default_2d[i, j, k]
    prob_ice = prob_ice_default_2d[i, j, k]

    return prob_clear, prob_water, prob_ice


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_night_core_probability(metric_110um, metric_375um, i):
    k = int((metric_110um - metric_110um_min_night_2d) / metric_110um_bin_night_2d)
    k = min(n_bins_metric_110um_night_2d - 1, max(0, k))
    j = int((metric_375um - metric_375um_min_night_2d) / metric_375um_bin_night_2d)
    j = min(n_bins_metric_375um_night_2d - 1, max(0, j))

    prob_clear = prob_clear_night_2d[i, j, k]
    prob_water = prob_water_night_2d[i, j, k]
    prob_ice = prob_ice_night_2d[i, j, k]

    return prob_clear, prob_water, prob_ice


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_day_375_core_probability(metric_110um, metric_375um, i):
    k = int((metric_110um - metric_110um_min_day_375_2d) / metric_110um_bin_day_375_2d)
    k = min(n_bins_metric_110um_day_375_2d - 1, max(0, k))
    j = int((metric_375um - metric_375um_min_day_375_2d) / metric_375um_bin_day_375_2d)
    j = min(n_bins_metric_375um_day_375_2d - 1, max(0, j))

    prob_clear = prob_clear_day_375_2d[i, j, k]
    prob_water = prob_water_day_375_2d[i, j, k]
    prob_ice = prob_ice_day_375_2d[i, j, k]

    return prob_clear, prob_water, prob_ice


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_day_160_core_probability(metric_110um, metric_160um, i):
    k = int((metric_110um - metric_110um_min_day_160_2d) / metric_110um_bin_day_160_2d)
    k = min(n_bins_metric_110um_day_160_2d - 1, max(0, k))
    j = int((metric_160um - metric_160um_min_day_160_2d) / metric_160um_bin_day_160_2d)
    j = min(n_bins_metric_160um_day_160_2d - 1, max(0, j))

    prob_clear = prob_clear_day_160_2d[i, j, k]
    prob_water = prob_water_day_160_2d[i, j, k]
    prob_ice = prob_ice_day_160_2d[i, j, k]

    return prob_clear, prob_water, prob_ice


@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_core(
        ems_tropo, refl_375um, refl_160um, bt_110um_stddev, sol_zen,
        nb_sfc_type, oceanic_glint_flag, forward_scattering_flag  # , solar_contam_flag
):
    prob_clear = nan
    prob_water = nan
    prob_ice = nan

    # ---- default
    if not (isnan(ems_tropo) or isnan(bt_110um_stddev)):
        prob_clear, prob_water, prob_ice = compute_default_core_probability(ems_tropo, bt_110um_stddev, nb_sfc_type)

    # --- under these conditions, stick the default and do nothing else
    if oceanic_glint_flag == 1 or forward_scattering_flag == 1:  # or solar_contam_flag == 1
        return prob_clear, prob_water, prob_ice

    # ----- night
    if not (isnan(ems_tropo) or isnan(bt_110um_stddev)) and sol_zen > 90.0:
        prob_clear, prob_water, prob_ice = compute_night_core_probability(ems_tropo, refl_375um, nb_sfc_type)

    # ----- day
    if sol_zen < 75.0:
        # --3.75um (preferred)
        if not (isnan(refl_375um) or isnan(ems_tropo)):
            prob_clear, prob_water, prob_ice = compute_day_375_core_probability(ems_tropo, refl_375um, nb_sfc_type)
        # --1.6um
        elif not (isnan(refl_160um) or isnan(ems_tropo)):
            prob_clear, prob_water, prob_ice = compute_day_160_core_probability(ems_tropo, refl_160um, nb_sfc_type)

    return prob_clear, prob_water, prob_ice
