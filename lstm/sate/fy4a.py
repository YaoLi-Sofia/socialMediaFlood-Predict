#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
@Project: hunan_TH
@Author: shi yao <euler_shiyao@foxmail.com>
@Create Time: 2022/9/6 16:01
python version: python 3.8.12
"""


from typing import Tuple
from CFG import CFG
from numpy import ndarray
import copy
import numpy as np
from scipy import interpolate as ip
import xarray as xr
import components
import logging
import utils
import os
import re
import datetime as dt
import sys
import sate.geo_utils as gu


class FY4A(CFG):
    """
    读取风云四号共57种卫星产品，并提供插值到指定区域的功能
    除GIIRS外，向每个数据dataset对象中写入主产品数据块对应的lat矩阵，lon矩阵，flag矩阵，lat,lon,flag的大小与主产品的数据块大小一致
    flag为1代表数据可用，为0代表不可用
    没有对原始数据做任何的更改，没有做定标、定位和插值，但是提供了定标、定位以及插值的方法供用户调用
    """
    def __init__(self, args: list or str, screen_show_info: bool = True):
        super().__init__(args, screen_show_info)
                                                # 计算参数设置
        self.pi = 3.1415926535897932384626      # 圆周率
        self.ea = 6378.137                      # 地球的半长轴
        self.eb = 6356.7523                     # 地球的短半轴
        self.h = 42164                          # 地心到卫星质心的距离
        self.lambda_d = 104.7                   # 卫星星下点所在的经度
        self.l_max = {                          # 标称投影上的行号的最大值(从0开始)
            '500m': 21983,
            '1000m': 10991,
            '2000m': 5495,
            '4000m': 2747,
            '12000m': 916
        }
        self.COFF = {                           # 列偏移
            '500m': 10991.5,
            '1000m': 5495.5,
            '2000m': 2747.5,
            '4000m': 1373.5,
            '12000m': 457.5
        }
        self.CFAC = {                           # 列比例因子
            '500m': 81865099,
            '1000m': 40932549,
            '2000m': 20466274,
            '4000m': 10233137,
            '12000m': 3411045
        }
        self.c_max = self.l_max                 # 标称投影上的列号的最大值
        self.LOFF = self.COFF                   # 行偏移
        self.LFAC = self.CFAC                   # 行比例因子

                                                # 经纬度范围设置
        self.lat_max = 80.56672132              # 标称图上全圆盘经纬度有效范围
        self.lat_min = -80.56672132             # 标称图上全圆盘经纬度有效范围
        self.lon_max = 185.28337691             # 标称图上全圆盘经纬度有效范围: 360 - 174.71662309
        self.lon_min = 24.11662309              # 标称图上全圆盘经纬度有效范围

    def fix_lat_lon(self, lat: ndarray, lon: ndarray, **kwargs) -> Tuple[ndarray, ndarray]:
        lat_temp = copy.deepcopy(lat)
        lon_temp = copy.deepcopy(lon)
        # 根据官方指定的经纬度范围做修正
        lat_temp[lat > self.lat_max] = np.nan
        lat_temp[lat < self.lat_min] = np.nan
        lon_temp[lon > self.lon_max] = np.nan
        lon_temp[lon < self.lon_min] = np.nan
        if kwargs.get('l') is not None and kwargs.get('c') is not None:
            lon_temp[kwargs['l'] == -1] = np.nan
            lon_temp[kwargs['c'] == -1] = np.nan
            lat_temp[kwargs['l'] == -1] = np.nan
            lat_temp[kwargs['c'] == -1] = np.nan
        return lat_temp, lon_temp

    def get_flag(self, data: ndarray, ori_lat: ndarray, ori_lon: ndarray, l_bias: int = 0, c_bias: int = 0)\
            -> Tuple[ndarray, ndarray]:
        # 根据经纬度矩阵，去除无效范围，设置flag标记有效值的区域，无效为0，有效为1
        ori_lat, ori_lon = self.fix_lat_lon(ori_lat, ori_lon)
        hh, ww = data.shape
        hh += l_bias
        ww += c_bias
        lat = ori_lat[l_bias:hh, c_bias:ww]
        lon = ori_lon[l_bias:hh, c_bias:ww]
        return lat, lon

    def gen_new_data(self, data: ndarray, ori_lat: ndarray, ori_lon: ndarray, new_lat: ndarray, new_lon: ndarray,
                     method: str) -> ndarray:
        """
        插值得到所需的数据

        Args:
            data:       原始数据，维度为(m, n)
            ori_lat:    原始的纬度矩阵，维度为(m, n)
            ori_lon:    原始的经度矩阵，维度为(m, n)
            new_lat:    需要插值到的纬度数列，维度为(m1, )
            new_lon:    需要插值到的经度数列，维度为(n1, )
            method:     插值方法
        Returns:
            维度为(m1, n1)的数据块
        """
        new_lon_2d, new_lat_2d = np.meshgrid(new_lon, new_lat)
        ori_lat[np.isnan(data)] = np.nan
        ori_lon[np.isnan(data)] = np.nan
        min_lat, max_lat = np.nanmin(new_lat) - 0.04, np.nanmax(new_lat) + 0.04
        min_lon, max_lon = np.nanmin(new_lon) - 0.04, np.nanmax(new_lon) + 0.04
        ori_lat[(ori_lat < min_lat) | (ori_lat > max_lat)] = np.nan
        ori_lon[(ori_lon < min_lon) | (ori_lon > max_lon)] = np.nan
        select_data = ~np.isnan(data) * ~np.isnan(ori_lat) * ~np.isnan(ori_lon)
        values = data[select_data]
        points = np.c_[ori_lat[select_data], ori_lon[select_data]]
        new_data = ip.griddata(points, values, (new_lat_2d, new_lon_2d), method=method)
        return new_data

    def l_c_to_Lat_lon(self, data: ndarray, l_bias: int, c_bias: int, **kwargs) -> Tuple[ndarray, ndarray]:
        """
        根据data的行列，计算得到data每个数据点的经纬度，并插值到指定经纬度网格，存储为nc数据

        Args:
            data:       读取到的行列数据
            l_bias:     行偏移，即：(data的第l行，相当于标称投影中的 (l + l_bias) 行)
            c_bias:     列偏移，即：(data的第c列，相当于标称投影中的 (c + c_bias) 列)
        Returns:
            原始数据块对应的经纬度矩阵
        """

        # 根据原始数据块得到行列矩阵
        hh, ww = data.shape
        if kwargs.get('l') is not None and kwargs.get('c') is not None and kwargs['l'].ndim == 2 and \
                kwargs['l'].ndim == 2:
            l = kwargs['l']
            c = kwargs['c']
        else:
            l = np.repeat(np.array([np.arange(hh)]).T, ww, axis=1) + l_bias
            c = np.repeat(np.array([np.arange(ww)]), hh, axis=0) + c_bias
        if kwargs.get('res') is None:
            kwargs['res'] = '4000m'
        # Step1.求 x,y
        x = (self.pi * (c - self.COFF[kwargs['res']])) / (180 * (2 ** (-1 * 16)) * self.CFAC[kwargs['res']])
        y = (self.pi * (l - self.LOFF[kwargs['res']])) / (180 * (2 ** (-1 * 16)) * self.LFAC[kwargs['res']])

        # Step2.求 sd,sn,s1,s2,s3,sxy
        s_d = np.sqrt(((self.h * np.cos(x) * np.cos(y)) ** 2) -
                      ((((np.cos(y)) ** 2) + ((self.ea ** 2) * (np.sin(y) ** 2) / (self.eb ** 2))) * ((self.h ** 2) -
                                                                                           (self.ea ** 2))))
        s_n = (self.h * np.cos(x) * np.cos(y) - s_d) / (((np.cos(y)) ** 2) +
                                                        ((self.ea ** 2) * (np.sin(y) ** 2) / (self.eb ** 2)))
        s1 = self.h - s_n * np.cos(x) * np.cos(y)
        s2 = s_n * np.sin(x) * np.cos(y)
        s3 = -1 * s_n * np.sin(y)
        s_xy = np.sqrt((s1 ** 2) + (s2 ** 2))

        # Step3 求原始的lon,lat
        ori_lon = (180 / self.pi) * np.arctan(s2 / s1) + self.lambda_d
        ori_lat = (180 / self.pi) * np.arctan(((self.ea ** 2) / (self.eb ** 2)) * (s3 / s_xy))
        return ori_lat, ori_lon

    def read_FY4A(self, file_path: str):
        """
        读取FY4A数据。添加经纬度信息（包括中国区和全圆盘），再转存为nc文件
        Args:
            file_path:  FY4A文件路径
        """
        return self.read_AGRI_L1_REF(file_path)

    def read_AGRI_L1_REF(self, file_path: str) -> xr.Dataset:
        ds = xr.open_dataset(file_path, engine='netcdf4')
        data = ds['NOMChannel01'].values
        l_bias = ds.attrs['Begin Line Number']
        c_bias = ds.attrs['Begin Pixel Number']
        # 根据行列获得原始数据的经纬点
        ori_lat, ori_lon = self.l_c_to_Lat_lon(data, l_bias, c_bias)
        ori_lat, ori_lon = self.get_flag(data, ori_lat, ori_lon)
        ds['lat'] = (ds['NOMChannel01'].dims, ori_lat)
        ds['lon'] = (ds['NOMChannel01'].dims, ori_lon)
        return ds


class FY4APreprocess:
    def __init__(self, distance_path: str):
        # distance_path = r”D:\FY4A\codes\Earth-Sun_distance.csv“
        """
        波段平均太阳辐照度的计算，参考了文献[1]
        [1] 白杰、王国杰、牛铮、邬明权. FY-4A/AGRI传感器波段平均太阳辐照度计算及不确定性分析[J]. 气象科学, 2020, v.40(04):98-104.
        """
        self.distance_path = distance_path

    @staticmethod
    def get_data_from_4km_ds(ds: xr.Dataset) -> ndarray:
        # 获取数据
        all_band = np.array([ds['NOMChannel01'].values, ds['NOMChannel02'].values, ds['NOMChannel03'].values,
                             ds['NOMChannel04'].values, ds['NOMChannel05'].values, ds['NOMChannel06'].values,
                             ds['NOMChannel07'].values, ds['NOMChannel08'].values, ds['NOMChannel09'].values,
                             ds['NOMChannel10'].values, ds['NOMChannel11'].values, ds['NOMChannel12'].values,
                             ds['NOMChannel13'].values, ds['NOMChannel14'].values]).astype(float)
        return all_band

    def get_sv_name_from_ds(self, ds: xr.Dataset) -> str:
        return utils.time2str(self.get_time_from_file(ds)[0], "%Y%m%d_%H_%M_%S.nc")

    @staticmethod
    def get_cal_from_4km_ds(ds: xr.Dataset) -> list:
        # 获取数据
        all_CAL = [ds['CALChannel01'].values, ds['CALChannel02'].values, ds['CALChannel03'].values,
                   ds['CALChannel04'].values, ds['CALChannel05'].values, ds['CALChannel06'].values,
                   ds['CALChannel07'].values, ds['CALChannel08'].values, ds['CALChannel09'].values,
                   ds['CALChannel10'].values, ds['CALChannel11'].values, ds['CALChannel12'].values,
                   ds['CALChannel13'].values, ds['CALChannel14'].values]
        return all_CAL

    @staticmethod
    def get_time_from_file(ds: xr.Dataset) -> Tuple[dt.datetime, dt.datetime]:
        # 获取fy4a文件中的时间
        time_start = utils.str2time(ds.attrs['Observing Beginning Date'], '%Y-%m-%d')
        time_start_hour = utils.str2time(ds.attrs['Observing Beginning Time'], '%H:%M:%S.%f')
        start_time = dt.datetime(year=time_start.year, month=time_start.month, day=time_start.day,
                           hour=time_start_hour.hour, minute=time_start_hour.minute, second=time_start_hour.second,
                           tzinfo=dt.timezone.utc)
        time_end = utils.str2time(ds.attrs['Observing Ending Date'], '%Y-%m-%d')
        time_end_hour = utils.str2time(ds.attrs['Observing Ending Time'], '%H:%M:%S.%f')
        end_time = dt.datetime(year=time_end.year, month=time_end.month, day=time_end.day,
                           hour=time_end_hour.hour, minute=time_end_hour.minute, second=time_end_hour.second,
                           tzinfo=dt.timezone.utc)
        return start_time, end_time

    def find_ref_by_4km_ds(self, ds: xr.Dataset):
        # 根据定标表定标
        logging.debug("根据定标表定标")
        all_band = self.get_data_from_4km_ds(ds)  # 获取原始DN值
        all_CAL = self.get_cal_from_4km_ds(ds)  # 获取定标表
        for i in range(14):
            # 设置定标表的无效值为nan
            if i == 6:
                all_CAL[i][all_CAL[i] < 100] = np.nan
                all_CAL[i][all_CAL[i] > 500] = np.nan
                all_band[i, :, :][all_band[i, :, :] < 0] = np.nan
                all_band[i, :, :][all_band[i, :, :] > 65534] = np.nan
            elif i < 6:
                all_CAL[i][all_CAL[i] < 0] = np.nan
                all_CAL[i][all_CAL[i] > 1.5] = np.nan
                all_band[i, :, :][all_band[i, :, :] < 0] = np.nan
                all_band[i, :, :][all_band[i, :, :] > 4095] = np.nan
            else:
                all_CAL[i][all_CAL[i] < 100] = np.nan
                all_CAL[i][all_CAL[i] > 500] = np.nan
                all_band[i, :, :][all_band[i, :, :] < 0] = np.nan
                all_band[i, :, :][all_band[i, :, :] > 4095] = np.nan
            # 查表，定标
            temp = np.zeros(all_band[i, :, :].shape).astype(float) + np.nan
            where = np.where(np.isnan(all_band[i, :, :]) == False)  # 有值的行列号位置
            where_data = all_band[i, :, :][where].astype(int)  # 上述位置中对应的值
            temp[where] = all_CAL[i][where_data]
            all_band[i, :, :] = temp
        return all_band

    def compute_toa_bt_by_4km_ds(self, ds: xr.Dataset, ori_flag: bool = False, h_theta: ndarray = -1):
        # 计算表观反射率和亮温
        all_band = self.find_ref_by_4km_ds(ds)  # 风云定标
        logging.debug("计算太阳高度角和地日距离")
        ER = self.sun_angles(ds)
        if not ori_flag:
            t = self.get_time_from_file(ds)[0] + (self.get_time_from_file(ds)[1] - self.get_time_from_file(ds)[0]) / 2
            h_theta = np.deg2rad(gu.sun().get_altitude(t, ds['lat'].values, ds['lon'].values))
        h_theta_flag = copy.deepcopy(h_theta)
        h_theta_flag[h_theta_flag <= 0], h_theta_flag[h_theta_flag > 0] = np.nan, 1
        logging.debug("计算表观反射率和亮温")
        for i in range(14):
            if i > 5:
                continue
            h_theta_flag_temp = copy.deepcopy(h_theta_flag)
            h_theta_flag_temp[np.isnan(all_band[i])] = np.nan
            all_band[i] = all_band[i] * ER / np.sin(h_theta)
            all_band[i][np.isnan(h_theta_flag_temp) & (np.isnan(all_band[i])==False)] = 0
            all_band[i][(all_band[i] < 0) | (all_band[i] > 1)] = 0  # 大于1的值一般在晨昏线附近，赋值为0更合适
        return all_band

    def sun_angles(self, ds: xr.Dataset):
        # 地日距离(天文单位)
        return gu.sun().compute_earth_sun_distance(ds.lat.values, ds.lon.values, self.get_time_from_file(ds)[0])


class Fy4aData(CFG):
    def __init__(self, file_path: str, time: dt.datetime, args: list or str, screen_show_info: bool = True):
        self.args = args
        self.file_path = file_path
        self.time = time
        self.ori_flag = False
        super().__init__(args, screen_show_info)
        self.lat, self.lon, self.h, self.w = \
            self.extrapolate['lat'], self.extrapolate['lon'], self.extrapolate['h'], self.extrapolate['w']
        self.file_list = self.find_data()

    def gen_all_data(self, file_path: str, new_lat: ndarray, new_lon: ndarray):
        band = np.arange(14)
        self.sv_root_path = utils.time2str(self.time, self.extrapolate['dataset'])
        utils.ispath(self.sv_root_path)
        save_path = os.path.join(self.sv_root_path, utils.time2str(self.time, "%Y%m%d%H%M%S.nc"))
        if os.path.exists(save_path) and not self.re_flag:
            logging.debug(f"{save_path}已存在")
            return
        logging.debug(f"初始化{file_path}")
        self.fy4a = FY4A(self.args, False)
        self.fy4a_preprocess = FY4APreprocess(self.extrapolate['distance_path'])
        logging.debug("获取dataset对象")
        fy4a_ds = self.fy4a.read_AGRI_L1_REF(file_path)
        # _, start = self.fy4a_preprocess.get_time_from_file(fy4a_ds)
        ori_lat, ori_lon = fy4a_ds['lat'].values, fy4a_ds['lon'].values
        # 表观反射率
        h_theta = -1
        if self.ori_flag:
            geo_path = file_path.replace('_FDI', '_GEO').replace(os.sep + '4000M', os.sep + 'GEO')
            logging.debug(f"读取GEO数据：{geo_path}")
            h_theta = np.deg2rad(90 - xr.open_dataset(geo_path, engine='netcdf4')['NOMSunZenith'].values)
        all_band = self.fy4a_preprocess.compute_toa_bt_by_4km_ds(fy4a_ds, self.ori_flag, h_theta)
        logging.debug("数据裁剪与整合")
        start_time, end_time = self.fy4a_preprocess.get_time_from_file(fy4a_ds)
        ds = xr.Dataset()
        ds.coords['lat'] = ('lat', new_lat)
        ds.coords['lon'] = ('lon', new_lon)
        ds.coords['band'] = ('band',band)
        ds['time'] = ('time', np.array([0]))
        ds['time'].attrs['long_name'] = "Time(BJT)"
        ds['time'].attrs['units'] = 'minutes since ' + self.time.strftime('%Y-%m-%d %H:%M:%S')
        ds['time'].attrs['tips'] = f"Observing from {utils.utc2bjt(start_time)} to {utils.utc2bjt(end_time)}"
        data = []
        for i in range(14):
            logging.debug(f"开始处理band{i + 1}")
            new_data = self.fy4a.gen_new_data(all_band[i], ori_lat, ori_lon, new_lat, new_lon, "cubic")
            if i < 6:
                new_data[new_data < 0] = 0
                new_data[new_data > 1.5] = 1.5
            else:
                new_data[new_data < 100] = 100
                new_data[new_data > 500] = 500
            ds[fr"band{i + 1}"] = (('time', 'lat', 'lon'), np.array(new_data).reshape((1, self.h, self.w)))
        #     data.append(new_data)
        # ds["Var"] = (('time', 'band', 'lat', 'lon'), np.array(data).reshape((1, len(band), len(new_lat),
        #                                                                      len(new_lon))))
        self.sv_ds(ds, save_path)
        logging.debug(f"保存成nc文件到{save_path}")
        # self.sv_npy(ds, save_path)

    @staticmethod
    def sv_ds(ds: xr.Dataset, save_path: str):
        ds['lat'].attrs['units'] = "degrees_north"
        ds['lat'].attrs['long_name'] = "Latitude"
        ds['lon'].attrs['units'] = "degrees_east"
        ds['lon'].attrs['long_name'] = "Longitude"
        ds.attrs['CreateTime'] = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ds.attrs['Author'] = "shi yao <euler_shiyao@foxmail.com>"
        # 存储
        encoding = {}
        for i in range(14):
            data = ds[f'band{i + 1}'].values
            ds[f'band{i + 1}'].attrs['unit'] = 'NUL' if i < 6 else 'K'
            ds[f'band{i + 1}'].attrs['tips'] = \
                'reflectance on the top of atmosphere' if i < 6 else 'brightness temperature'
            scale_factor, add_offset = utils.calc_scale_and_offset(np.nanmin(data), np.nanmax(data), 16)
            encoding[f'band{i + 1}'] = {
                'dtype': 'int16',
                'scale_factor': scale_factor,
                'complevel': 9,
                'zlib': True,
                '_FillValue': -9999,
                'add_offset': add_offset
            }
        # data = ds['Var'].values
        # scale_factor, add_offset = utils.calc_scale_and_offset(np.nanmin(data), np.nanmax(data), 16)
        # encoding['Var'] = {
        #     'dtype': 'int16',
        #     'scale_factor': scale_factor,
        #     'complevel': 9,
        #     'zlib': True,
        #     '_FillValue': -9999,
        #     'add_offset': add_offset
        # }
        ds.to_netcdf(
            save_path,
            engine='netcdf4',
            encoding=encoding
        )

    def sv_npy(self, ds: xr.Dataset, save_path: str):
        data = np.zeros((14, self.h, self.w))
        save_path = save_path.replace('.nc', '.npy')
        for i in range(14):
            data[i] = ds[f'band{i + 1}'].values
        np.save(save_path, data)
        logging.debug(f"保存成npy文件到{save_path}")

    def find_data(self) -> list:
        return [self.file_path]

    def gen_dataset(self):
        for file_path in self.file_list:
            try:
                self.gen_all_data(file_path, self.lat, self.lon)
            except OSError as e:
                logging.debug(f"文件解析出现问题{file_path}")
                logging.debug(e)