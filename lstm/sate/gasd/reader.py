import re
from dataclasses import dataclass, astuple
from math import nan, sin, cos, tan, radians, degrees, sqrt, asin, atan, atan2, exp, log

import numpy as np
import xarray as xr
from numba import njit, vectorize

SSP_LONGITUDE = 104.7
SATELLITE_DISTANCE = 42164.0
EQUATOR_RADIUS = 6378.137
POLAR_RADIUS = 6356.7523
PROJECTION_PARAM1 = 0.0066943844
PROJECTION_PARAM2 = 0.993305616
PROJECTION_PARAM3 = 1.006739501
PROJECTION_PARAM_SD = 1737122264.0

SCALE_UNIT = 1.52587890625e-05  # 2^-16 scaling function


def check_shape(a, /, *args):
    shape = np.shape(a)
    for arr in args:
        if shape != np.shape(arr):
            raise ValueError('Shape conflicts')
    return shape


@njit(nogil=True, error_model='numpy', boundscheck=True)
def _pix_lin2lon_lat(
        pix, lin, ssp_lon, c_fac, l_fac, c_off, l_off,
        satellite_distance, projection_param3, projection_param_sd
):
    x = radians(pix - c_off) / (SCALE_UNIT * c_fac)
    y = radians(lin - l_off) / (SCALE_UNIT * l_fac)

    cos_x = cos(x)
    cos_y = cos(y)
    sin_y = sin(y)
    tmp0 = cos_y * cos_y + projection_param3 * sin_y * sin_y
    tmp1 = satellite_distance * cos_x * cos_y
    sd = (tmp1 * tmp1 - tmp0 * projection_param_sd)
    if sd < 0.0:
        return nan, nan
    sd = sqrt(sd)
    sn = (tmp1 - sd) / tmp0
    s1 = satellite_distance - (sn * cos_x * cos_y)
    s2 = sn * sin(x) * cos_y
    s3 = -sn * sin_y
    sxy = sqrt(s1 * s1 + s2 * s2)

    lon = degrees(atan2(s2, s1)) + ssp_lon
    lat = degrees(atan(projection_param3 * s3 / sxy))

    # check longitude
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon, lat


@njit(nogil=True, error_model='numpy', boundscheck=True)
def pix_lin2lon_lat(
        shape, pix, lin,
        sub_lon, c_fac, l_fac, c_off, l_off,
        satellite_distance, projection_param3, projection_param_sd
):
    lon = np.empty_like(pix)
    lat = np.empty_like(lin)
    for idx in np.ndindex(shape):
        lon[idx], lat[idx] = _pix_lin2lon_lat(
            pix[idx], lin[idx], sub_lon, c_fac, l_fac, c_off, l_off,
            satellite_distance, projection_param3, projection_param_sd
        )
    return lon, lat


@njit(nogil=True, error_model='numpy', boundscheck=True)
def _lon_lat2pix_lin(
        lon, lat, ssp_lon, c_fac, l_fac, c_off, l_off,
        satellite_distance, polar_radius, projection_param1, projection_param2
):
    if abs(lat) > 90.0:
        return nan, nan

    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0

    phi = atan(projection_param2 * tan(radians(lat)))
    cos_phi = cos(phi)

    re_ = polar_radius / sqrt(1.0 - projection_param1 * cos_phi * cos_phi)

    tmp = radians(lon) - radians(ssp_lon)
    r1 = satellite_distance - re_ * cos_phi * cos(tmp)
    r2 = -re_ * cos_phi * sin(tmp)
    r3 = re_ * sin(phi)

    if (r1 * (r1 - satellite_distance) + (r2 * r2) + (r3 * r3)) > 0:
        return nan, nan

    rn = sqrt(r1 * r1 + r2 * r2 + r3 * r3)
    x = degrees(atan2(-r2, r1))
    y = degrees(asin(-r3 / rn))

    pix = c_off + x * SCALE_UNIT * c_fac
    lin = l_off + y * SCALE_UNIT * l_fac

    return pix, lin


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lon_lat2pix_lin(
        shape, lon, lat,
        sub_lon, c_fac, l_fac, c_off, l_off,
        satellite_distance, polar_radius, projection_param1, projection_param2
):
    pix = np.empty_like(lon)
    lin = np.empty_like(lat)
    for idx in np.ndindex(shape):
        pix[idx], lin[idx] = _lon_lat2pix_lin(
            lon[idx], lat[idx], sub_lon, c_fac, l_fac, c_off, l_off,
            satellite_distance, polar_radius, projection_param1, projection_param2
        )
    return pix, lin


@dataclass
class ProjectionInfo(object):
    sub_lon: float
    c_fac: int
    l_fac: int
    c_off: float
    l_off: float
    satellite_distance: float
    equator_radius: float
    polar_radius: float
    projection_param1: float
    projection_param2: float
    projection_param3: float
    projection_param_sd: float

    def lon_lat2pix_lin(self, lon, lat):
        shape = check_shape(lon, lat)
        return lon_lat2pix_lin(
            shape, np.asarray(lon), np.asarray(lat), self.sub_lon,
            self.c_fac, self.l_fac, self.c_off, self.l_off,
            self.satellite_distance, self.polar_radius, self.projection_param1, self.projection_param2
        )

    def pix_lin2lon_lat(self, pix, lin):
        shape = check_shape(pix, lin)
        return pix_lin2lon_lat(
            shape, np.asarray(pix), np.asarray(lin), self.sub_lon,
            self.c_fac, self.l_fac, self.c_off, self.l_off,
            self.satellite_distance, self.projection_param3, self.projection_param_sd
        )


PROJECTION_0500 = ProjectionInfo(
    sub_lon=SSP_LONGITUDE,
    c_fac=81865099,
    l_fac=81865099,
    c_off=10991.5,
    l_off=10991.5,
    satellite_distance=SATELLITE_DISTANCE,
    equator_radius=EQUATOR_RADIUS,
    polar_radius=POLAR_RADIUS,
    projection_param1=PROJECTION_PARAM1,
    projection_param2=PROJECTION_PARAM2,
    projection_param3=PROJECTION_PARAM3,
    projection_param_sd=PROJECTION_PARAM_SD,
)

PROJECTION_1000 = ProjectionInfo(
    sub_lon=SSP_LONGITUDE,
    c_fac=40932549,
    l_fac=40932549,
    c_off=5495.5,
    l_off=5495.5,
    satellite_distance=SATELLITE_DISTANCE,
    equator_radius=EQUATOR_RADIUS,
    polar_radius=POLAR_RADIUS,
    projection_param1=PROJECTION_PARAM1,
    projection_param2=PROJECTION_PARAM2,
    projection_param3=PROJECTION_PARAM3,
    projection_param_sd=PROJECTION_PARAM_SD,
)

PROJECTION_2000 = ProjectionInfo(
    sub_lon=SSP_LONGITUDE,
    c_fac=20466274,
    l_fac=20466274,
    c_off=2747.5,
    l_off=2747.5,
    satellite_distance=SATELLITE_DISTANCE,
    equator_radius=EQUATOR_RADIUS,
    polar_radius=POLAR_RADIUS,
    projection_param1=PROJECTION_PARAM1,
    projection_param2=PROJECTION_PARAM2,
    projection_param3=PROJECTION_PARAM3,
    projection_param_sd=PROJECTION_PARAM_SD,
)

PROJECTION_4000 = ProjectionInfo(
    sub_lon=SSP_LONGITUDE,
    c_fac=10233137,
    l_fac=10233137,
    c_off=1373.5,
    l_off=1373.5,
    satellite_distance=SATELLITE_DISTANCE,
    equator_radius=EQUATOR_RADIUS,
    polar_radius=POLAR_RADIUS,
    projection_param1=PROJECTION_PARAM1,
    projection_param2=PROJECTION_PARAM2,
    projection_param3=PROJECTION_PARAM3,
    projection_param_sd=PROJECTION_PARAM_SD,
)

DEFAULT_PROJECTION = {
    500: PROJECTION_0500, 1000: PROJECTION_1000, 2000: PROJECTION_2000, 4000: PROJECTION_4000,
}
DEFAULT_MAX_NUMBER = {
    500: 21983, 1000: 10991, 2000: 5495, 4000: 2747,
}


# 北边界纬度/º	80.56672132
# 南边界纬度/º	-80.56672132
# 东边界经度/º	-174.71662309
# 西边界经度/º	24.11662309

# a = np.fromfile('E:/data reference/FullMask_Grid_4000.raw', dtype='f8', count=2748 * 2748 * 2).reshape((2748, 2748, 2))
# b = np.where(a != 999999.9999, a, np.nan)
# print(b[1374, 1374])
# # np.nanmax(b[...,0])
# # Out: 80.88329584917955
# # np.nanmin(b[...,0])
# # Out: -80.88329584917955
# # np.nanmin(b[...,1])
# # Out: -179.99970521126536
# # np.nanmax(b[...,1])
# # Out: 179.99930066785407
# print(pix_lin2lon_lat(
#     1374, 1374, SSP_LONGITUDE, 10233137, 10233137, 1373.5, 1373.5,
#     SATELLITE_DISTANCE, PROJECTION_PARAM3, PROJECTION_PARAM_SD
# ))
# print(lon_lat2pix_lin(*pix_lin2lon_lat(
#     1374, 1374, SSP_LONGITUDE, 10233137, 10233137, 1373.5, 1373.5,
#     SATELLITE_DISTANCE, PROJECTION_PARAM3, PROJECTION_PARAM_SD
# ), SSP_LONGITUDE, 10233137, 10233137, 1373.5, 1373.5,
#                       SATELLITE_DISTANCE, POLAR_RADIUS, PROJECTION_PARAM1, PROJECTION_PARAM2)
#       )
# print(pix_lin2lon_lat(
#     1375, 1375, SSP_LONGITUDE, 10233137, 10233137, 1373.5, 1373.5,
#     SATELLITE_DISTANCE, PROJECTION_PARAM3, PROJECTION_PARAM_SD
# ))

@vectorize(nopython=True, forceobj=False)
def count2radiance(count, gain_cnt2rad, const_cnt2rad, error_count, out_count):
    if count in (error_count, out_count):
        return nan
    return count * gain_cnt2rad + const_cnt2rad


@vectorize(nopython=True, forceobj=False)
def radiance_nasa2noaa(radiance, wave_len):
    return radiance * wave_len * wave_len / 10.0


@dataclass
class CalibrationInfo(object):
    band_num: int
    wave_len: float
    bit_pix: int
    error_count: int
    out_count: int
    gain_cnt2rad: float
    const_cnt2rad: float
    valid_min: float
    valid_max: float
    lookup_table: np.ndarray

    def count2radiance(self, count):
        return count2radiance(count, self.gain_cnt2rad, self.const_cnt2rad, self.error_count, self.out_count)

    def radiance_nasa2noaa(self, radiance):
        return radiance_nasa2noaa(radiance, self.wave_len)

    def count2phys(self, count):
        mask = np.logical_and(count >= self.valid_min, count <= self.valid_max)
        phys = np.full(mask.shape, nan, 'f4')
        phys[mask] = self.lookup_table[count[mask]]
        return phys


@dataclass
class SegmentInfo(object):
    start_line_num: int
    end_line_num: int
    start_pixel_num: int
    end_pixel_num: int


@dataclass
class AgriL1(object):
    projection: ProjectionInfo
    calibration: dict[int, CalibrationInfo]
    segment_info: SegmentInfo
    count: dict[int, np.ndarray]
    _radiance_nasa = None
    _radiance_noaa = None
    _phys = None

    @classmethod
    def from_file(cls, file: str):
        dataset = xr.open_dataset(file, engine='netcdf4')
        resolution = int(dataset.attrs['File Name'][74:78])
        projection = DEFAULT_PROJECTION[resolution]

        calibration_coefficient = dataset['CALIBRATION_COEF(SCALE+OFFSET)'].values
        pattern = re.compile(r'(\d*(\.\d*)?)um')
        calibration = {}
        count = {}
        for c in range(1, 15):
            wave_len = float(pattern.match(dataset[f'CALChannel{c:0>2}'].attrs['center_wavelength']).group(1))
            scale, offset = calibration_coefficient[c - 1]
            valid_min, valid_max = dataset[f'NOMChannel{c:0>2}'].attrs['valid_range']
            # todo
            valid_max = min(valid_max, 64047)
            lookup_table = dataset[f'CALChannel{c:0>2}'].values
            calibration[c] = CalibrationInfo(
                c, wave_len, 16, 65534, 65535, scale, offset, valid_min, valid_max, lookup_table
            )
            for i in range(1, 15):
                count[i] = dataset[f'NOMChannel{i:0>2}'].values

        start_line = dataset.attrs['Begin Line Number']
        end_line = dataset.attrs['End Line Number']
        start_pixel = dataset.attrs['Begin Pixel Number']
        end_pixel = dataset.attrs['End Pixel Number']
        segment_info = SegmentInfo(start_line, end_line, start_pixel, end_pixel)

        return cls(projection, calibration, segment_info, count)

    @property
    def radiance(self):
        if self._radiance_nasa is None:
            self._radiance_nasa = {}
            for c in range(1, 15):
                self._radiance_nasa[c] = self.calibration[c].count2radiance(self.count[c])
        return self._radiance_nasa

    @property
    def radiance_nasa(self):
        if self._radiance_nasa is None:
            self._radiance_nasa = {}
            for c in range(1, 15):
                self._radiance_nasa[c] = self.calibration[c].count2radiance(self.count[c])
        return self._radiance_nasa

    @property
    def radiance_noaa(self):
        if self._radiance_noaa is None:
            self._radiance_noaa = {}
            for c in range(1, 15):
                self._radiance_noaa[c] = self.calibration[c].radiance_nasa2noaa(self.radiance[c])
        return self._radiance_noaa

    @property
    def phys(self):
        if self._phys is None:
            self._phys = {}
            for c in range(1, 15):
                self._phys[c] = self.calibration[c].count2phys(self.count[c])
        return self._phys

    def get_level(self, level=None):
        if level == 'count':
            return self.count
        if level == 'radiance' or level == 'radiance(NASA standard)':
            return self.radiance
        if level == 'radiance(NOAA standard)':
            return self.radiance_noaa
        if level is None or level == 'phys':
            return self.phys
        raise ValueError('Invalid level')

    def get_data_by_pix_lin(self, h_pix, h_lin, level=None):
        shape = check_shape(h_pix, h_lin)

        start_line, end_line, start_pixel, end_pixel = astuple(self.segment_info)

        mask = np.logical_and(
            np.logical_and(h_lin >= start_line - 0.5, h_lin < end_line + 0.5),
            np.logical_and(h_pix >= start_pixel - 0.5, h_pix < end_pixel + 0.5)
        )

        pix = np.array(h_pix[mask] + 0.5 - start_pixel, 'i4')
        lin = np.array(h_lin[mask] + 0.5 - start_line, 'i4')

        if level == 'count':
            fill_value = 65535
        else:
            fill_value = nan

        data = self.get_level(level)

        value = {}
        for i in range(1, 15):
            value[i] = np.full(shape, fill_value, data[i].dtype)
            value[i][mask] = data[i][lin, pix]

        return value

    def get_data_by_lon_lat(self, lon, lat, level=None):
        shape = check_shape(lon, lat)

        h_pix, h_lin = self.projection.lon_lat2pix_lin(lon, lat)

        start_line, end_line, start_pixel, end_pixel = astuple(self.segment_info)

        mask = np.logical_and(
            np.logical_and(h_lin >= start_line - 0.5, h_lin < end_line + 0.5),
            np.logical_and(h_pix >= start_pixel - 0.5, h_pix < end_pixel + 0.5)
        )

        pix = np.array(h_pix[mask] + 0.5 - start_pixel, 'i4')
        lin = np.array(h_lin[mask] + 0.5 - start_line, 'i4')

        if level == 'count':
            fill_value = 65535
        else:
            fill_value = nan

        data = self.get_level(level)

        value = {}
        for i in range(1, 15):
            value[i] = np.full(shape, fill_value, data[i].dtype)
            value[i][mask] = data[i][lin, pix]

        return value


if __name__ == '__main__':
    from sklearn.linear_model import LinearRegression

    c1 = 1.191062e-5
    c2 = 1.4387863


    @vectorize(nopython=True, forceobj=False)
    def planck_rad(c, t):
        planck_a1 = 0.0
        planck_a2 = 1.0
        if c == 7:
            planck_nu = 2688.1720430107525
        elif c == 8:
            planck_nu = 2688.1720430107525
        elif c == 9:
            planck_nu = 1600.0
        elif c == 10:
            planck_nu = 1408.4507042253522
        elif c == 11:
            planck_nu = 1176.4705882352941
        elif c == 12:
            planck_nu = 925.9259259259259
        elif c == 13:
            planck_nu = 833.3333333333334
        elif c == 14:
            planck_nu = 740.7407407407408
        else:
            raise ValueError

        tmp0 = t - planck_a1
        tmp1 = planck_a2 * c2 * planck_nu / tmp0
        tmp2 = exp(tmp1)
        tmp3 = tmp2 - 1.0
        b = c1 * planck_nu * planck_nu * planck_nu / tmp3

        return b


    @vectorize(nopython=True, forceobj=False)
    def planck_temp(c, b):
        planck_a1 = 0.0
        planck_a2 = 1.0
        if c == 7:
            planck_nu = 2688.1720430107525
        elif c == 8:
            planck_nu = 2688.1720430107525
        elif c == 9:
            planck_nu = 1600.0
        elif c == 10:
            planck_nu = 1408.4507042253522
        elif c == 11:
            planck_nu = 1176.4705882352941
        elif c == 12:
            planck_nu = 925.9259259259259
        elif c == 13:
            planck_nu = 833.3333333333334
        elif c == 14:
            planck_nu = 740.7407407407408
        else:
            raise ValueError

        tmp0 = c1 * planck_nu * planck_nu * planck_nu / b
        tmp1 = tmp0 + 1.0
        tmp2 = log(tmp1)
        tmp3 = planck_a2 * c2 * planck_nu / tmp2
        t = planck_a1 + tmp3

        return t


    def fy4a_agri_test():
        agri_l1 = AgriL1.from_file(
            'C:/Users/dell/Desktop/'
            'FY4A-_AGRI--_N_REGC_1047E_L1-_FDI-_MULT_NOM_20220706012336_20220706012753_4000M_V0001.HDF'
        )
        print(agri_l1)
        lon, lat = np.meshgrid(np.linspace(80.0, 140.0, 3001), np.linspace(60.0, 0.0, 3001))
        radiance = agri_l1.get_data_by_lon_lat(lon, lat, level='radiance(NOAA standard)')
        phys = agri_l1.get_data_by_lon_lat(lon, lat, level='phys')
        for c in range(7, 15):
            print(c)
            # 波段8看起来不是很准确, 其他结果都明显可以
            x = planck_temp(c, radiance[c])
            y = phys[c]

            reg = LinearRegression(fit_intercept=True)
            reg.fit(x[~np.isnan(x)].reshape((-1, 1)), y[~np.isnan(x)].reshape(-1))
            print(reg.coef_, reg.intercept_)

            # plt.scatter(x.flat, y.flat)
            # plt.show()
            #
            # plt.scatter(radiance[c].flat, planck_rad(c, phys[c]).flat)
            # plt.show()

            # plt.imshow(radiance[c])
            # plt.colorbar()
            # plt.show()
            #
            # plt.imshow(fy4a_agri_l1.radiance[c])
            # plt.colorbar()
            # plt.show()


    fy4a_agri_test()
    # todo 利用attrs信息完善AgriL1类
