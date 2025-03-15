# coding: UTF-8
"""
@Project: codes
@FileName: geo_utils.py
@author: shi yao  <aiwei169@sina.com>
@create date: 2022/2/15 16:09
@Software: PyCharm
python version: python 3.8.12
"""


import numpy as np
from numpy import ndarray
import datetime as dt
import components
import pysolar.solar as Sun


class sun:
    def __init__(self):
        pass

    def get_altitude(self, time: dt.datetime, lat: ndarray or float or int = None,
                     lon: ndarray or float or int = None):
        """

        Args:
            time:
            lat:
            lon:

        Returns:
            地日距离(天文单位)，太阳赤纬角(度)，太阳高度角(弧度)，太阳方位角(弧度)
        >>> time = dt.datetime(1999, 6, 23, 21, 56, 9, tzinfo=dt.timezone.utc)
        >>> sun().get_altitude(time, np.array([23.442, 24, 25, 26]), np.array([110, 111, 112]))
        """
        lat, lon = self.fix_lat_lon(lat, lon)
        return Sun.get_altitude(lat, lon, time)

    @staticmethod
    def fix_lat_lon(lat: ndarray or float or int = None, lon: ndarray or float or int = None):
        # 修正经纬度到指定格式、类型和数值
        lat = np.arange(-90, 91, 0.01) if lat is None else lat
        lon = np.arange(-179, 181, 0.01) if lon is None else lon
        assert (type(lat) == np.ndarray) or (type(lat) == float) or (type(lat) == int), \
            f"lat不是numpy.ndarray、float、int类型，而是{type(lat)}"
        assert (type(lon) == np.ndarray) or (type(lon) == float) or (type(lon) == int), \
            f"lon不是numpy.ndarray、float、int类型，而是{type(lon)}"
        if type(lat) != np.ndarray:
            lat = np.nan if np.abs(lat) > 90 else lat
            lat = np.array(lat).astype(float)
        else:
            lat[np.abs(lat) > 90] = np.nan
        if type(lon) != np.ndarray:
            lon = np.array(lon).astype(float)
        lon = (lon + 180) % 360 - 180
        if (lat.ndim == 1) & (lon.ndim == 1):
            new_lon, nwe_lat = np.meshgrid(lat, lon)
            lat, lon = new_lon.T, nwe_lat.T
        elif (lat.ndim == 2) & (lon.ndim == 2):
            pass
        else:
            components.raiseE('经纬度出错')
        return lat, lon

    @staticmethod
    def compute_earth_sun_distance(lat: ndarray or float, lon: ndarray or float, time: dt.datetime):
        """

        Args:
            lat: 纬度(m, n) or float
            lon: 经度(m, n) or float
            time: UTC时间

        Returns:
            地日距离的平方(天文单位²)，太阳赤纬角(度)，太阳高度角(弧度)，太阳方位角(弧度)
        >>> time = dt.datetime(1999, 6, 23, 12, 42) - dt.timedelta(hours=8)
        >>> sun().compute_earth_sun_distance(23.442, 110, time)
        (1.0329443798023188, 23.440526806457726, 1.5707486498968404, 1.001201210328165)
        """
        if (type(lat) == np.ndarray) & (type(lon) == np.ndarray):
            if (lat.ndim == 1) & (lon.ndim == 1):
                new_lon, nwe_lat = np.meshgrid(lat, lon)
                lat, lon = new_lon.T, nwe_lat.T
            if (lat.ndim > 2) | (lon.ndim > 2):
                components.raiseE('经纬度出错')
        # 当前的地方时时间是一年中的第几天(精确到秒，单位是天)
        N = (time - dt.datetime(time.year, 1, 1, tzinfo=dt.timezone.utc)).days + (time.hour / 24)\
            + (time.minute / 1440) + (time.second / 86400) + (lon / 15) / 24
        N0 = 79.6764 + 0.2422 * (time.year - 1985) - int((time.year - 1985) / 4)
        # 日角(弧度)
        theta = (N - N0) * (2 * np.pi / 365.2422)
        # 日地距离(AU 天文单位)的平方
        ER = 1.000423 + 0.032359 * np.sin(theta) + 0.000086 * np.sin(2 * theta) - 0.008349 * np.cos(theta)\
             + 0.000115 * np.cos(2 * theta)
        return ER