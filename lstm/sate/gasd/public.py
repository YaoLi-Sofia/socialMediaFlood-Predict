import numpy as np

image_number_of_lines = 151
image_number_of_elements = 151

image_shape = (image_number_of_lines, image_number_of_elements)

# user_options
nav_lat_max_limit = 30.5
nav_lat_min_limit = 24.5
nav_lon_max_limit = 114.5
nav_lon_min_limit = 108.5

geo_sat_zen_max_limit = 85.0
geo_sat_zen_min_limit = 0.0
geo_sol_zen_max_limit = 180.0
geo_sol_zen_min_limit = 0.0

p_inversion_min = 700.0
delta_t_inversion = 0.0

channels = np.arange(1, 17, dtype='i1')
# solar_channels = channels[:6]
solar_channels = np.array([3, 4, 5], 'i1')
mixed_channels = np.array([7], 'i1')
# thermal_channels = channels[6:]
thermal_channels = np.array([7, 9, 14, 15, 16], 'i1')
# thermal_channels = np.array([7, 9, 13, 14, 15, 16], 'i1')

sensor_spatial_resolution_meters = 4000
