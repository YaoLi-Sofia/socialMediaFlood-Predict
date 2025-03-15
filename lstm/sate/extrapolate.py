#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Project         hunan_TH
Author          shi yao
Create Time     2022/7/12 10:42
python version  python 3.8.12
"""
import copy
import os
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sate import convlstm, ext
import utils
import components
import logging
from CFG import CFG
import datetime as dt
import xarray as xr
import pygrib as pb
from sate.fy4a import Fy4aData
from numpy import ndarray


class EXTRAPOLATE(CFG):
    # TODO: 逐1小时生成未来2小时逐1小时的卫星外推

    def __init__(self, args: list or str, screen_show_info: bool = False):
        super().__init__(args, screen_show_info)
        self.args = args
        self.lat, self.lon, self.h, self.w = \
            self.extrapolate['lat'], self.extrapolate['lon'], self.extrapolate['h'], self.extrapolate['w']
        # 用过去的self.data_pre张图外推未来的self.data_for张图
        self.data_pre, self.data_for = int(self.extrapolate['pre']), int(self.extrapolate['for'])
        self.train_num, self.test_num = int(self.extrapolate['train_num']), int(self.extrapolate['test_num'])
        self.batch = int(self.extrapolate['batch'])


    def gen_product(self):
        if self.fy4a_flag:  # 生成.npy文件
            logging.debug("开始生成npy文件，进行定标处理")
            self.FY4A()
            return
        for time in self.time_list_1HOR:
            # ext_conv = convlstm.convlstm(self.args, False)
            ext_conv = ext.convlstm(self.args, False)
            pre_data, for_path, for_time_list = self.find_dataset(time)
            model_root_path = utils.time2str(time, self.extrapolate['model'])
            utils.ispath(model_root_path)
            model_name = f'net_{time:%Y%m%d%H%M}.pth'
            model_path = os.path.join(model_root_path, model_name)
            model = ext_conv.run(pre_data, self.data_pre, self.data_for, self.train_num, self.test_num, model_path,
                                 self.batch)
            # import torch
            # model = torch.load(model_path)
            ext_conv.ext(pre_data[-self.data_pre:], model, for_path, self.lat, self.lon, for_time_list, time)


    def FY4A(self):
        for time in self.time_list_15MIN:
            try:
                file_path = self.find_fy4a_by_time(time)
            except FileNotFoundError as e:
                logging.debug(e)
                continue
            Fy4aData(file_path, time, self.args, False).gen_dataset()  # 生成.npy的定标后的数据

    def find_dataset(self, time: dt.datetime) -> Tuple[ndarray, list, pd.date_range]:
        num = self.data_pre + self.data_for + self.train_num + self.test_num - 1
        ds = xr.Dataset()
        ds.coords['band'] = ('band', np.arange(14))
        ds.coords['lat'] = ('lat', self.lat)
        ds.coords['lon'] = ('lon', self.lon)
        start_time = time - dt.timedelta(minutes=(num * 15))
        pre_time_list = pd.date_range(start_time, freq='15min', periods=num)
        for_time_list = pd.date_range(time, freq='15min', periods=self.data_for)
        pre_path, for_path = [], []
        tl = []
        for t in pre_time_list:
            temp_path = utils.time2str(t, os.path.join(self.extrapolate['dataset'], r'%Y%m%d%H%M%S.nc'))
            if os.path.exists(temp_path):
                pre_path.append(temp_path)
                tl.append(t)
                logging.debug(fr"选择{t}的{temp_path}作为训练集")
        ds.coords['time'] = ('time', tl)
        ds['Var'] = (('time', 'band', 'lat', 'lon'), np.zeros((len(pre_path), 14, self.h, self.w)))
        for k in range(len(pre_path)):
            data = xr.open_dataset(pre_path[k]).to_array().values[:, 0, :, :]
            ds['Var'].values[k] = data
        if len(tl) < len(pre_time_list):
            ds = ds.interp(time=pre_time_list, kwargs={"fill_value": "extrapolate"})

        temp = copy.deepcopy(ds['Var'].values[:, 6:, :, :])
        temp[temp < 100] = 100
        temp[temp > 500] = 500
        ds['Var'].values[:, 6:, :, :] = temp
        del temp
        temp = copy.deepcopy(ds['Var'].values[:, :6, :, :])
        temp[temp < 0] = 0
        temp[temp > 1.5] = 1.5
        ds['Var'].values[:, :6, :, :] = temp

        for t in for_time_list:
            sv_root_path = utils.time2str(t, os.path.join(self.extrapolate['ext_path']))
            utils.ispath(sv_root_path)
            sv_path = utils.time2str(t, os.path.join(sv_root_path, r'%Y%m%d%H%M%S.nc'))
            for_path.append(sv_path)
            logging.debug(fr"{t}的外推输出路径是{sv_path}")

        return ds['Var'].values, for_path, for_time_list

    def find_fy4a_by_time(self, time: dt.datetime) -> str:
        utc_time = utils.bjt2utc(time)
        root_path = utils.time2str(utc_time, self.download['fy4a'])
        time_list, file_list = [], []
        for file_name in os.listdir(root_path):
            time_list.append(utils.str2time(file_name.split('_')[-3], '%Y%m%d%H%M%S'))
            file_list.append(os.path.join(root_path, file_name))
        time_delta = np.abs(np.array(time_list) - utc_time)
        tag = np.where(time_delta == np.nanmin(time_delta))[0][0]
        file_path = file_list[tag]
        logging.debug(fr"{time}对应的最近的风云卫星数据文件是{file_path}")
        if time_delta[tag] <= dt.timedelta(minutes=7.5):
            return file_path
        else:
            msg = fr"{time}对应的最近的风云卫星数据文件缺失"
            raise FileNotFoundError(msg)

