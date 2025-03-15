c1 = 1.191062e-5
c2 = 1.4387863

# solar_ch07_nu = 14.735645
# 'SOLAR_20_NU': 14.735645,
# 'SUN_EARTH_DISTANCE': 1.0165122,
solar_ch07 = 1.8891729983815806
ew_ch07 = 128.5008840875  # 光谱响应对波数积分
solar_ch07_nu = 1000.0 * solar_ch07 / ew_ch07

# import numpy as np
# import pandas as pd
#
# # spectral_response = np.loadtxt('E:/cloud_acha/ch7_resp_ahi.dat', skiprows=2)
# spectral_response = np.loadtxt('E:/cloud_acha/ch7_resp_fy4.dat', skiprows=2)
#
# xp = 1.0e7 / spectral_response[:, 0]
# fp = spectral_response[:, 1]
# w = ((fp[:-1] + fp[1:]) * (xp[:-1] - xp[1:]) / 2).sum() * 1e-3
#
# spectral_distribution = pd.read_excel(
#     'G:/957-heliosat2_lib/AM0AM1_5.xls', sheet_name='Spectra', header=1, usecols=[5, 6], nrows=1697
# )
#
# min_wave_num = spectral_response[0, 0]
# max_wave_num = spectral_response[-1, 0]
#
# x = spectral_response[:, 0] * 1000.0
# sr = spectral_response[:, 1]
#
# # 光谱响应函数和地外太阳辐射乘积的积分
# xp = spectral_distribution['Wavelength (nm).1'].values
# fp = spectral_distribution['W*m-2*nm-1'].values
# sd = np.interp(x, xp, fp)
# tmp = sr * sd
# io_met = ((tmp[:-1] + tmp[1:]) * (x[1:] - x[:-1]) / 2).sum()
#
# print(io_met)
# print(w)
# print(1000.0 * io_met / w)
#
#
# solar_ch07 = 4.874826974069999
# ew_ch07 = 305.6888765359883  # 光谱响应对波数积分
# solar_ch07_nu = 1000.0 * solar_ch07 / ew_ch07
