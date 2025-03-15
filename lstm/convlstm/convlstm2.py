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
    def __init__(self, num_hidden, channel, width, len_x, len_y, filter_size):
        super(my_model, self).__init__()
        self.num_hidden = num_hidden
        self.len_y = len_y
        self.len_x = len_x
        stride = 1

        self.conv_lstm = ConvLSTMCell(channel, num_hidden, width, filter_size, stride)
        self.conv_last = nn.Conv2d(num_hidden, channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames):
        # frames.shape = [seq, batch, channel, height, width]
        batch = frames.shape[0]  # 数据很少，设置batch size为1
        height = frames.shape[3]
        width = frames.shape[4]

        zeros = torch.zeros([batch, self.num_hidden, height, width])
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


def data_preprocess(dir, len_x, len_y):
    # 读文件夹中所有文件
    filenames = os.listdir(dir)
    data = []
    for filename in filenames:
        data.append(np.load(os.path.join(dir, filename)))
    data = np.array(data).astype(np.float32)
    data[:, 6:, :, :] = (data[:, 6:, :, :] - 100) / 400  # 亮温归一化
    # logging.debug(data.shape)
    # t_ = 0
    # for n in data:
    #     tt = 0
    #     for p in n:
    #         logging.debug((t_, tt, np.nanmin(p), np.nanmax(p)))
    #         tt += 1
    #     t_ += 1
    #     exit(0)
    # logging.debug(np.nanmax(data))
    data = np.nan_to_num(data, nan=0)  # 把nan转换成数值，默认是0
    # data = (data / np.max(data)).astype(np.float32)  # 把数据归一化到0 1之间
    # 构造x 和 y
    data_x, data_y = [], []
    for i in range(len_x, len(data) - len_y + 1):
        data_x.append(data[i - len_x: i])
        data_y.append(data[i: i + len_y])
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    # 划分训练集和测试集，并转化成tensor
    test_num = len(data_x) // 5
    train_x = torch.from_numpy(data_x[:-test_num])
    train_y = torch.from_numpy(data_y[:-test_num])
    test_x = torch.from_numpy(data_x[-test_num:])
    test_y = torch.from_numpy(data_y[-test_num:])

    return train_x, train_y, test_x, test_y



class convlstm(CFG):
    def __init__(self, args: list or str, screen_show_info: bool = False):
        super().__init__(args, screen_show_info)
        logging.debug("开始外推训练")

    def run(self):
        dir = '/mnt/ftp188/tmp/temp/train'
        num_epoch = 50         # 迭代次数
        lr = 0.005              # 学习率
        filter_size = 3        # 卷积核大小
        len_x = 8              # 用 len_x 个时间点预测未来 len_y 个时间点
        len_y = 2
        train_x, train_y, test_x, test_y = data_preprocess(dir, len_x, len_y)
        logging.debug(train_x.shape)
        logging.debug(train_y.shape)
        logging.debug(test_x.shape)
        logging.debug(test_y.shape)

        model = my_model(num_hidden=64,
                         channel=train_x.shape[2],
                         width=train_x.shape[-1],
                         len_x=len_x,
                         len_y=len_y,
                         filter_size=filter_size)
        # print(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 优化器
        loss_func = nn.MSELoss()          # 损失函数
        # loss_func = nn.KLDivLoss(reduction='batchmean')          # 损失函数
        # loss_func = nn.KLDivLoss(reduction='mean')          # 损失函数
        # 训练及测试
        for epoch in range(num_epoch):
            model.train()
            train_loss = []
            for i, x in enumerate(train_x):
                model.train()
                output = model(x.unsqueeze(0)) # unsqueeze(0)效果是在最低维度增加一个维度，比如(4,4) -> (1,4,4)
                loss = loss_func(output, train_y[i].unsqueeze(0))
                # loss = loss_func(output, train_y[i].unsqueeze(0))
                train_loss.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                test_loss = []
                for i, x in enumerate(test_x):
                    pre_y = model(x.unsqueeze(0))
                    test_loss.append(loss_func(pre_y, test_y[i].unsqueeze(0)).item())
                logging.debug(f'epoch:{epoch}, train loss:{sum(train_loss) / len(train_loss)}, test loss:{sum(test_loss) / len(test_loss)}')

