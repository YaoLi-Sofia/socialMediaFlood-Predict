import copy
from typing import Tuple

import numpy as np
import os
import torch
import torch.nn as nn
import sys
import components
import logging
import matplotlib.pyplot as plt
import utils
from CFG import CFG
import datetime as dt
import xarray as xr
from numpy import ndarray
import scipy.stats as ss
import torch.nn.functional as F



# def get_corr(X, Y):  # 计算两个向量person相关系数与1的差
#     X, Y = X.flatten(), Y.flatten()
#     D = torch.var
#     r = 0.5 * (D(X + Y) - D(X) - D(Y)) / (torch.sqrt(D(X) * D(Y)))
#     return 1 - (r ** 2)

def get_corr(X, Y):  # 计算两个向量person相关系数与1的差
    h, w = X.shape[-2], X.shape[-1]
    n = int(len(X.flatten()) / (h * w))
    X = X.reshape((n, h, w))
    Y = Y.reshape((n, h, w))
    r = 0
    person_ = []
    for i in range(n):
        x, y = X[i], Y[i]
        x, y = x.flatten(), y.flatten()
        D = torch.var
        person = ((0.5 * (D(x + y) - D(x) - D(y)) / (torch.sqrt(D(x) * D(y)))) ** 2)
        person_ .append(person.detach().numpy())
        rmse = torch.sqrt(torch.mean((x - y) ** 2))
        logging.debug(f"{i}的相关系数：{person}")
        logging.debug(f"{i}的RMSE：{rmse}")
        logging.debug(f"{i}的相对RMSE：{rmse / torch.mean(y)}")
        r += ((1 - person) + rmse)
    person_ = np.array(person_)
    logging.debug(f"最小相关系数{np.nanmin(person_)}")
    logging.debug(f"平均相关系数{np.nanmean(person_)}")
    logging.debug(f"最大相关系数{np.nanmax(person_)}")
    return r / n, np.nanmin(person_), np.nanmax(person_), np.nanmean(person_)

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride):
        super(ConvLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        self.conv_x = nn.Sequential(
            nn.Conv2d(in_channel, num_hidden * 4, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.conv_h = nn.Sequential(
            nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size,
                      stride=stride, padding=self.padding),
            nn.LayerNorm([num_hidden * 4, width, width])
        )
        self.Wci = nn.Parameter(torch.zeros(1, num_hidden, width, width))
        self.Wcf = nn.Parameter(torch.zeros(1, num_hidden, width, width))
        self.Wco = nn.Parameter(torch.zeros(1, num_hidden, width, width))

    def forward(self, x_t, h_t, c_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        i_x, f_x, g_x, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h + self.Wci * c_t)
        f_t = torch.sigmoid(f_x + f_h + self.Wcf * c_t + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        o_t = torch.sigmoid(o_x + o_h + self.Wco * c_t)
        h_new = o_t * torch.tanh(c_new)

        return h_new, c_new


class my_model(nn.Module):
    def __init__(self, num_hidden, channel, width, len_x, len_y, filter_size, batch):
        super(my_model, self).__init__()
        self.batch = batch
        self.num_hidden = num_hidden
        self.len_y = len_y
        self.len_x = len_x
        stride = 1

        self.conv_lstm = ConvLSTMCell(channel, num_hidden, width, filter_size, stride)
        self.conv_last = nn.Conv2d(num_hidden, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames):
        # frames.shape = [seq, batch, channel, height, width]
        height = frames.shape[3]
        width = frames.shape[4]
        zeros = torch.zeros((self.batch, self.num_hidden, height, width))
        h_t = zeros
        c_t = zeros
        next_frames = []  # 存预测的结果

        for t in range(self.len_x):  # 输入的图片为训练的时间点
            net = frames[:, t]
            h_t, c_t = self.conv_lstm(net, h_t, c_t)

        x_gen = self.conv_last(h_t)
        next_frames.append(x_gen)  # 生成预测的第一个时间点的

        for t in range(self.len_y - 1):  # 生成预测的第二个 至 第len_y个时间点
            h_t, c_t = self.conv_lstm(x_gen, h_t, c_t)
            x_gen = self.conv_last(h_t)
            next_frames.append(x_gen)

        # import pdb;pdb.set_trace()
        next_frames = torch.stack(next_frames, dim=1)  # 转换成和预测的y一样的shape

        return next_frames


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, inputs, targets):
        # inputs = inputs.detach().numpy()
        # targets = targets.detach().numpy()
        # return torch.tensor(np.nanmin(r_), requires_grad=True)
        return get_corr(inputs, targets)


class convlstm(CFG):
    def __init__(self, args: list or str, screen_show_info: bool = False):
        super().__init__(args, screen_show_info)
        logging.debug("开始外推训练")

    @staticmethod
    def data_preprocess(pre_data: ndarray, len_x: int, len_y: int, train_num: int, test_num: int) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        data = pre_data
        data[:, :6, :, :] = data[:, :6, :, :] / 1.5
        data[:, 6:, :, :] = (data[:, 6:, :, :] - 100) / 400  # 亮温归一化，band7~band14都是亮温波段，数值范围是[100, 500]


        data = np.nan_to_num(data, nan=0)  # 把nan转换成数值，默认是0
        data = np.array(data).astype(np.float32)
        # 构造x 和 y，（用x外推y，x为输入，y是输出，合起来是一对样本集）
        data_x, data_y = [], []
        for i in range(len_x, len(data) - len_y + 1):
            data_x.append(data[i - len_x: i])
            data_y.append(data[i: i + len_y])
        data_x = np.array(data_x)
        data_y = np.array(data_y)
        # 划分训练集和测试集，并转化成tensor
        train_x = torch.from_numpy(data_x[:-test_num])
        train_y = torch.from_numpy(data_y[:-test_num])
        test_x = torch.from_numpy(data_x[-test_num:])
        test_y = torch.from_numpy(data_y[-test_num:])
        return train_x, train_y, test_x, test_y

    def run(self, pre_data: ndarray, len_x, len_y, train_num, test_num, model_path, batch):
        # 用 len_x 个时间点预测未来 len_y 个时间点
        num_epoch = int(self.extrapolate['num_epoch'])  # 迭代次数
        lr = float(self.extrapolate['learning_rate'])  # 学习率
        filter_size = int(self.extrapolate['filter_size'])  # 卷积核大小
        train_x, train_y, test_x, test_y = self.data_preprocess(pre_data, len_x, len_y, train_num, test_num)
        model = my_model(num_hidden=64,
                         channel=train_x.shape[2],
                         width=train_x.shape[-1],
                         len_x=len_x,
                         len_y=len_y,
                         filter_size=filter_size,
                         batch=batch)
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)  # 优化器
        # loss_func = nn.CrossEntropyLoss()          # 损失函数
        # loss_func = nn.MSELoss()          # 损失函数
        loss_func = MyLoss()          # 损失函数
        # loss_func = nn.KLDivLoss(reduction='batchmean')          # 损失函数
        # loss_func = nn.KLDivLoss(reduction='mean')          # 损失函数
        # 训练及测试
        logging.debug("模型初始化完毕，开始训练")
        logging.debug(fr"共有{train_num}个训练集样本，{test_num}个测试集样本，每个样本包含{len_x + len_y}个数据都是用前{len_x}个数据外推后{len_y}个数据，每个数据的维度是{train_x[0, 0, :, :, :].shape}，channel有{train_x.shape[2]}个")
        for epoch in range(num_epoch):
        # for epoch in range(1):
            model.train()
            train_loss = []
            for i, x in enumerate(train_x):
                model.train()
                output = model(x.unsqueeze(0)) # unsqueeze(0)效果是在最低维度增加一个维度，比如(4,4) -> (1,4,4)
                loss_, _, _, _ = loss_func(output, train_y[i].unsqueeze(0))
                train_loss.append(loss_.item())
                optimizer.zero_grad()
                loss_.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                test_loss, min_p, max_p, mean_p = [], [], [], []
                for i, x in enumerate(test_x):
                    pre_y = model(x.unsqueeze(0))
                    loss_, min_per, max_per, mean_per = loss_func(pre_y, test_y[i].unsqueeze(0))
                    test_loss.append(loss_.item())
                    min_p.append(min_per)
                    max_p.append(max_per)
                    mean_p.append(mean_per)
                logging.debug(f'epoch:{epoch}, '
                              f'train loss:{sum(train_loss) / len(train_loss)}, '
                              f'test loss:{sum(test_loss) / len(test_loss)}')
                logging.debug(fr"测试集的最小相关系数{min_per}, 最大相关系数{max_per}, 平均相关系数{mean_per}")
                min_p = np.array(min_p)
                max_p = np.array(max_p)
                mean_p = np.array(mean_p)
                if np.nanmin(min_p) >= 0.8:
                    if np.nanmin(max_p) >= 0.9:
                        break
                if np.nanmin(mean_p) > 0.8:
                    break

        torch.save(model, model_path)
        logging.debug(fr"模型存储在{model_path}")
        return torch.load(model_path)

    def ext(self, data: ndarray, model, for_path: list, lat: ndarray, lon: ndarray, for_time_list, time):
        k, b, h, w = data.shape
        data = np.nan_to_num(data, nan=0)
        ext_data = torch.from_numpy(data.reshape((1, k, b, h, w)).astype(np.float32))
        model.eval()
        pre_y = np.array([])
        with torch.no_grad():
            for i, x in enumerate(ext_data):
                pre_y = model(x.unsqueeze(0))
        pre_y = np.nan_to_num(pre_y, nan=0)
        for i in range(len(for_path)):
            out = pre_y[0][i]
            out[:6, :, :] = out[:6, :, :] * 1.5
            out[6:, :, :] = (out[6:, :, :] * 400) + 100

            sv_path = for_path[i]
            ds = xr.Dataset()
            ds['lat'] = ('lat', lat)
            ds['lon'] = ('lon', lon)
            ds['time'] = ('time', np.array([0]))
            ds['time'].attrs['units'] = 'minutes since ' + for_time_list[i].strftime('%Y-%m-%d %H:%M:%S')

            for j in range(14):
                new_data = out[j]
                if j < 6:
                    new_data[new_data < 0] = 0
                    new_data[new_data > 1.5] = 1.5
                else:
                    new_data[new_data < 100] = 100
                    new_data[new_data > 500] = 500
                ds[fr"band{j + 1}"] = (('time', 'lat', 'lon'), np.array(new_data).reshape((1, h, w)))

            encoding = {}
            for j in range(b):
                data = ds[f'band{j + 1}'].values
                ds[f'band{j + 1}'].attrs['unit'] = 'NUL' if j < 6 else 'K'
                ds[f'band{j + 1}'].attrs['tips'] = \
                    'reflectance on the top of atmosphere' if j < 6 else 'brightness temperature'
                scale_factor, add_offset = utils.calc_scale_and_offset(np.nanmin(data), np.nanmax(data), 16)
                encoding[f'band{j + 1}'] = {
                    'dtype': 'int16',
                    'scale_factor': scale_factor,
                    'complevel': 9,
                    'zlib': True,
                    '_FillValue': -999,
                    'add_offset': add_offset
                }
            ds.to_netcdf(
                sv_path,
                engine='netcdf4',
                encoding=encoding
            )
            logging.debug(fr"{time}起报的{for_time_list[i]}的外推结果存储在{sv_path}")




