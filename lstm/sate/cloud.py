#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Project         hunan_TH
Author          zhang qing hao
Create Time     2022/7/12 10:42
python version  python 3.8.12
"""


import logging
from CFG import CFG

from gasd import main_real, main_fore


class CLOUD(CFG):
    # TODO: 逐15分钟生成过去15分钟的云底高

    # TODO: 逐1小时生成未来2小时逐1小时的云底高

    def __init__(self, args: list or str, screen_show_info: bool = False):
        super().__init__(args, screen_show_info)
        """
        在160上，按如下格式输入之后要能够正确运行
        docker run --rm -e LANG="en_US.UTF-8"  -v /home:/home  -v /mnt:/mnt -v /data:/data --net=host --privileged=True shiyao:shiyao /shiyao/miniconda3/envs/shiyao/bin/python /home/developer_13/hunan_TH/main.py -d -cloud -t 202205010000 202205020000
        """
        # 纬度序列、经度序列、长度、宽度
        self.lat, self.lon, self.h, self.w = self.base['lat'], self.base['lon'], self.base['h'], self.base['w']

    def gen_product(self):

        if self.real_flag:
            logging.debug("生成实况产品")

        if self.fore_flag:
            logging.debug("生成预报品")

        logging.debug("开始生成CLOUD产品")
        """
        1.注意添加是否覆盖式回算的功能
        2.使用logging.debug统一进行日志输出
        3.把要用的参数写进config.xml
        4.如果项目运行的时候，数据有延迟，可以用-del num 参数，如在后面加argparse参数-del 1.5，则表示计算1.5小时之前的数据
        5.注意气象要素和文件名的命名
        """
        if self.re_flag:
            """
            需要覆盖式回算
            """
            force = True
        else:
            """
            不需要覆盖式回算
            """
            force = False

        for time in self.time_list_15MIN:
            logging.debug(time)  # time的类型是dt.datetime
            """
            如果是self.time_list_10MIN则表示
            argparse输入的时间段或者时刻，对应到整10分钟
            如果输入-t 202201010000 202201020000
            则self.time_list_10MIN = [
            202201010000, 202201010010, 202201010020, 202201010030
            ……
            ……
            202201020000
            ]是一个逐10分钟的列表
            如果输入-t 202201010003
            则self.time_list_10MIN = [202201010000]  # 会规约到整10分钟
            """
            if self.real_flag:
                main_real(time, force)

        for time in self.time_list_1HOR:  # 命令行默认为空的话，是当前时间，写的话代表的是传进来的数据的时间，大家可以进入命令行尝试一下
            logging.debug(time)  # time的类型是dt.datetime

            if self.fore_flag:
                main_fore(time, force)
