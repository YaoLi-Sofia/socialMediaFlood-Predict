number_of_non_cloud_flags = 18

# --- angular and temperature limit thresholds
reflectance_gross_sol_zen_thresh = 75.0  # was 70.0
reflectance_spatial_sol_zen_thresh = 80.0
reflectance_gross_airmass_thresh = 7.0  # 100.0 # 5.0   # turned off
ems_375um_day_sol_zen_thresh = 80.0  # was 85.0
ems_375um_night_sol_zen_thresh = 90.0  # was 80.0
t_sfc_cold_scene_thresh = 230.0
path_tpw_dry_scene_thresh = 0.5
bt_375um_cold_scene_thresh = 240.0
forward_scatter_scatter_zen_max_thresh = 95.0
forward_scatter_sol_zen_max_thresh = 95.0

# --- eumetcast fire detection parameters
eumetcast_fire_day_sol_zen_thresh = 70.0
eumetcast_fire_night_sol_zen_thresh = 90.0

bt_375um_eumet_fire_day_thresh = 310.0
bt_diff_eumet_fire_day_thresh = 8.0
stddev_110um_eumet_fire_day_thresh = 1.0
stddev_375um_eumet_fire_day_thresh = 4.0

bt_375um_eumet_fire_night_thresh = 290.0
bt_diff_eumet_fire_night_thresh = 0.0
stddev_110um_eumet_fire_night_thresh = 1.0
stddev_375um_eumet_fire_night_thresh = 4.0

# --- clavrx smoke thresholds
refl_065_min_smoke_water_thresh = 2.0
refl_065_max_smoke_water_thresh = 25.0
refl_160_max_smoke_water_thresh = 5.0
refl_375_max_smoke_water_thresh = 3.0
ems_11_tropo_max_smoke_water_thresh = 0.05
t11_std_max_smoke_water_thresh = 0.25
refl_065_std_max_smoke_water_thresh = 3.00
btd_4_11_max_smoke_water_thresh = 3.0
sol_zen_max_smoke_water_thresh = 80.0

refl_065_min_smoke_land_thresh = 10.0
refl_065_max_smoke_land_thresh = 25.0
refl_138_max_smoke_land_thresh = 5.0
nir_smoke_ratio_max_land_thresh = 0.0
refl_375_max_smoke_land_thresh = 3.0
ems_11_tropo_max_smoke_land_thresh = 0.3
t11_std_max_smoke_land_thresh = 2.0
refl_065_std_max_smoke_land_thresh = 3.00
btd_4_11_max_smoke_land_thresh = 5.0
sol_zen_max_smoke_land_thresh = 80.0

# --- clavrx dust thresholds
btd_11_12_metric_max_dust_thresh = -0.5  # -0.75# -0.5 #-1.0
btd_11_12_max_dust_thresh = 0.0
bt_11_std_max_dust_thresh = 1.0
bt_11_12_clear_diff_max_dust_thresh = -0.5
ems_11_tropo_max_dust_thresh = 0.20
ems_11_tropo_min_dust_thresh = 0.01
bt_11_clear_diff_min_dust_thresh = -5.0
btd_85_11_max_dust_thresh = -0.5
btd_85_11_min_dust_thresh = -2.5  # -1.5
