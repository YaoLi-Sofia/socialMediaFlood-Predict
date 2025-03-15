import itertools
import math
import os
import ssl
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime, timedelta
from ftplib import FTP

import numpy as np
import xarray as xr


def callback(block_num, block_size, total_size):
    total_num = math.ceil(total_size / block_size)
    fraction = min(block_num * block_size / total_size * 100, 100.0)
    display_len = int(fraction / 10.0)
    progress = '*' * display_len + ' ' * (10 - display_len)
    print(f'Processing {block_num}: {fraction:3.0f}%|{progress}| {block_num}/{total_num}')
    # bar = tqdm.trange(math.ceil(total_size / block_size))
    # bar.set_description(f'Processing {block_num}')
    # bar = tqdm.std.Bar(block_num * block_size / total_size)
    # tqdm.tqdm


def callback(block_num, block_size, total_size):
    """回调函数
       @block_num: 已经下载的数据块
       @block_size: 数据块的大小
       @total_size: 远程文件的大小
    """
    percent = min((block_num * block_size / total_size * 100.0), 100.0)
    sys.stdout.write('\r>> Download %.2f%%' % percent)
    sys.stdout.flush()


import sys


def progressbar(cur, total=100):
    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    # sys.stdout.write('[%-50s] %s' % ('=' * int(math.floor(cur * 50 / total)),percent))
    sys.stdout.write('[%-100s] %s' % ('=' * int(cur), percent))
    sys.stdout.flush()


def schedule(blocknum, blocksize, totalsize):
    """
    blocknum:当前已经下载的块
    blocksize:每次传输的块大小
    totalsize:网页文件总大小
    """
    if totalsize == 0:
        percent = 0
    else:
        percent = blocknum * blocksize / totalsize
    if percent > 1.0:
        percent = 1.0
    percent *= 100
    print('download : %.2f%%' % percent)
    progressbar(percent)


# url = 'http://www.xxxx.com/xxxxx'  # 下载文件的url
# path = r'C:\Users\Administrator\Desktop\download\xxx'  # 文件下载后保存的本地路径
# try:
#     request.urlretrieve(url, path, schedule)
# except Exception:  # np.linalg.LinAlgErrorerror.HTTPError as e:
#     print(e)
#     print('\r\n' + url + ' download failed!' + '\r\n')
# else:
#     print('\r\n' + url + ' download successfully!')


class GFSData(object):
    url_format = 'https://geodb.ssec.wisc.edu/ancillary/%Y_%m_%d_%j'

    def __init__(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

    @staticmethod
    def get_time_before(time):
        if time.hour < 6:
            return time.replace(hour=0)
        elif time.hour < 12:
            return time.replace(hour=6)
        elif time.hour < 18:
            return time.replace(hour=12)
        else:
            return time.replace(hour=18)

    @staticmethod
    def get_time_after(time):
        if time.hour > 18:
            return time.replace(hour=0) + timedelta(days=1)
        elif time.hour > 12:
            return time.replace(hour=18)
        elif time.hour > 6:
            return time.replace(hour=12)
        else:
            return time.replace(hour=6)

    @staticmethod
    def get_nearest_initial_time(time, forecast_time=0):
        if time.hour < 6:
            return time.replace(hour=0) - timedelta(hours=forecast_time)
        elif time.hour < 12:
            return time.replace(hour=6) - timedelta(hours=forecast_time)
        elif time.hour < 18:
            return time.replace(hour=12) - timedelta(hours=forecast_time)
        else:
            return time.replace(hour=18) - timedelta(hours=forecast_time)

    @staticmethod
    def _open_dataset(filename, initial_time, forecast_time):
        ds = xr.open_dataset(filename, engine='netcdf4').rename({
            'pressure levels': 'level',
            'fakeDim0': 'level',
            'fakeDim1': 'latitude', 'fakeDim2': 'longitude',
            'fakeDim3': 'latitude', 'fakeDim4': 'longitude',
            'fakeDim5': 'latitude', 'fakeDim6': 'longitude',
            'fakeDim7': 'latitude', 'fakeDim8': 'longitude',
            'fakeDim9': 'latitude', 'fakeDim10': 'longitude',
            'fakeDim11': 'latitude', 'fakeDim12': 'longitude',
            'fakeDim13': 'latitude', 'fakeDim14': 'longitude',
            'fakeDim15': 'latitude', 'fakeDim16': 'longitude',
            'fakeDim17': 'latitude', 'fakeDim18': 'longitude',
            'fakeDim19': 'latitude', 'fakeDim20': 'longitude',
            'fakeDim21': 'latitude', 'fakeDim22': 'longitude',
            'fakeDim23': 'latitude', 'fakeDim24': 'longitude',
            'fakeDim25': 'latitude', 'fakeDim26': 'longitude',
            'fakeDim27': 'latitude', 'fakeDim28': 'longitude',
            'fakeDim29': 'latitude', 'fakeDim30': 'longitude',
            'fakeDim31': 'latitude', 'fakeDim32': 'longitude',
            'fakeDim33': 'latitude', 'fakeDim34': 'longitude',
            'fakeDim35': 'latitude', 'fakeDim36': 'longitude', 'fakeDim37': 'level',
            'fakeDim38': 'latitude', 'fakeDim39': 'longitude', 'fakeDim40': 'level',
            'fakeDim41': 'latitude', 'fakeDim42': 'longitude', 'fakeDim43': 'level',
            'fakeDim44': 'latitude', 'fakeDim45': 'longitude', 'fakeDim46': 'level',
            'fakeDim47': 'latitude', 'fakeDim48': 'longitude', 'fakeDim49': 'level',
            'fakeDim50': 'latitude', 'fakeDim51': 'longitude', 'fakeDim52': 'level',
            'fakeDim53': 'latitude', 'fakeDim54': 'longitude', 'fakeDim55': 'level'
        }).set_coords({
            'level': 'level',
        }).expand_dims({
            'time': [initial_time + timedelta(hours=forecast_time)]
        })
        ds.coords['latitude'] = np.arange(361.0, dtype='f4') * 0.5 - 90.0
        ds.coords['longitude'] = (np.arange(720.0, dtype='f4')) * 0.5
        return ds
        # todo
        # return ds.where(ds != 9.999e+20)

    def get_dataset(self, time, forecast_time=0):
        init_time_before = self.get_time_before(time) - timedelta(hours=forecast_time)
        init_time_after = self.get_time_after(time) - timedelta(hours=forecast_time)
        if init_time_before != init_time_after:
            directory = init_time_before.strftime(f'{self.root_dir}/%Y/%m/%d/%H')
            filename = init_time_before.strftime(f'gfs.%y%m%d%H_F{forecast_time:0>3}.hdf')
            path = f'{directory}/{filename}'
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(path):
                os.makedirs(directory, exist_ok=True)
                url = init_time_before.strftime(f'{self.url_format}/{filename}')
                print(f'download {url}')
                urllib.request.urlretrieve(url, path, callback)
            dataset_before = self._open_dataset(path, init_time_before, forecast_time)

            directory = init_time_after.strftime(f'{self.root_dir}/%Y/%m/%d/%H')
            filename = init_time_after.strftime(f'gfs.%y%m%d%H_F{forecast_time:0>3}.hdf')
            path = f'{directory}/{filename}'
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(path):
                os.makedirs(directory, exist_ok=True)
                url = init_time_after.strftime(f'{self.url_format}/{filename}')
                print(f'download {url}')
                urllib.request.urlretrieve(url, path, callback)
            dataset_after = self._open_dataset(path, init_time_after, forecast_time)

            dataset = xr.merge([dataset_before, dataset_after]).interp(time=time)
        else:
            directory = init_time_before.strftime(f'{self.root_dir}/%Y/%m/%d/%H')
            filename = init_time_before.strftime(f'gfs.%y%m%d%H_F{forecast_time:0>3}.hdf')
            path = f'{directory}/{filename}'
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(path):
                os.makedirs(directory, exist_ok=True)
                url = init_time_before.strftime(f'{self.url_format}/{filename}')
                print(f'download {url}')
                urllib.request.urlretrieve(url, path, callback)
            dataset = self._open_dataset(path, init_time_before, forecast_time).sel(time=time)

        return dataset


class OISSTData(object):
    url_format = 'https://geodb.ssec.wisc.edu/ancillary/%Y_%m_%d_%j'

    def __init__(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

    @staticmethod
    def get_nearest_time(time):
        return time.replace(hour=0, minute=0, second=0, microsecond=0)

    @staticmethod
    def _open_dataset(filename, time):
        ds = xr.open_dataset(filename, engine='netcdf4').rename({
            'lat': 'latitude', 'lon': 'longitude'
        }).sel({
            'time': time.replace(hour=12), 'zlev': 0.0
        }).drop_vars({
            'zlev': 'raise'
        })
        return ds

    def get_dataset(self, time):
        directory = time.strftime(f'{self.root_dir}/%Y/%m/%d')
        filename = time.strftime(f'avhrr-only-v2.%Y%m%d.nc')
        path = f'{directory}/{filename}'
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(path):
            os.makedirs(directory, exist_ok=True)
            url = time.strftime(f'{self.url_format}/{filename}')
            try:
                print(f'download {url}')
                urllib.request.urlretrieve(url, path, callback)
            except Exception:  # np.linalg.LinAlgErrorHTTPError:
                print(f'{url} not exists')
                filename = time.strftime(f'avhrr-only-v2.%Y%m%d_preliminary.nc')
                path = f'{directory}/{filename}'
                if not os.path.exists(path):
                    url = time.strftime(f'{self.url_format}/{filename}')
                    print(f'download {url}')
                    urllib.request.urlretrieve(url, path, callback)
        dataset = self._open_dataset(path, time)
        return dataset


class SnowMapData(object):
    url_format = 'https://geodb.ssec.wisc.edu/ancillary/%Y_%m_%d_%j'

    def __init__(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

    @staticmethod
    def get_nearest_time(time):
        return time.replace(hour=0, minute=0, second=0, microsecond=0)

    def get_dataset(self, time):
        directory = time.strftime(f'{self.root_dir}/%Y/%m/%d')
        filename = time.strftime('snow_map_4km_%y%m%d.nc')
        path = f'{directory}/{filename}'
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(path):
            os.makedirs(directory, exist_ok=True)
            url = time.strftime(f'{self.url_format}/{filename}')
            print(f'download {url}')
            urllib.request.urlretrieve(url, path, callback)
        dataset = xr.open_dataset(path)
        return dataset


class SurfaceData(object):
    def __init__(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

    @staticmethod
    def _open_dataset(filename):
        ds = xr.open_dataset(filename, engine='netcdf4').rename({
            'n_lat': 'latitude', 'n_lon': 'longitude'
        })
        ds.coords['latitude'] = (-np.arange(21600, dtype='f4') + 10799.5) / 120
        ds.coords['longitude'] = (np.arange(43200, dtype='f4') - 21599.5) / 120
        return ds

    def get_dataset(self):
        surface_elev = self._open_dataset(f'{self.root_dir}/GLOBE_1km_digelev.nc')
        coast_mask = self._open_dataset(f'{self.root_dir}/coast_mask_1km.nc')
        sfc_type = self._open_dataset(f'{self.root_dir}/gl-latlong-1km-landcover.nc')
        land_mask = self._open_dataset(f'{self.root_dir}/lw_geo_2001001_v03m.nc')
        dataset = xr.merge([surface_elev, coast_mask, sfc_type, land_mask])
        return dataset


class EmissData(object):
    pass
    # def __init__(self, root_dir):
    #     os.makedirs(root_dir, exist_ok=True)
    #     self.root_dir = root_dir
    #
    # @staticmethod
    # def _open_dataset(filename):
    #     ds = xr.open_dataset(filename, engine='netcdf4').rename({
    #         'n_lat': 'latitude', 'n_lon': 'longitude'
    #     })
    #     ds.coords['latitude'] = (-np.arange(21600, dtype='f4') + 10799.5) / 120
    #     ds.coords['longitude'] = (np.arange(43200, dtype='f4') - 21599.5) / 120
    #     return ds
    #
    # def get_dataset(self):
    #     surface_elev = self._open_dataset(f'{self.root_dir}/GLOBE_1km_digelev.nc')
    #     coast_mask = self._open_dataset(f'{self.root_dir}/coast_mask_1km.nc')
    #     sfc_type = self._open_dataset(f'{self.root_dir}/gl-latlong-1km-landcover.nc')
    #     land_mask = self._open_dataset(f'{self.root_dir}/lw_geo_2001001_v03m.nc')
    #     dataset = xr.merge([surface_elev, coast_mask, sfc_type, land_mask])
    #     return dataset


# class MyFTP(FTP):
#     """
#     cmd:命令
#     callback:回调函数
#     fsize:服务器中文件总大小
#     rest:已传送文件大小
#     """
#
#     def retrbinary(self, cmd, callback, fsize=0, rest=0):
#         cmpsize = rest
#         self.voidcmd('TYPE I')
#         # 此命令实现从指定位置开始下载,以达到续传的目的
#
#         with self.transfercmd(cmd, rest) as conn:
#             while True:
#                 if fsize:
#                     if (fsize - cmpsize) >= 1024:
#                         blocksize = 1024
#                     else:
#                         blocksize = fsize - cmpsize
#                     ret = float(cmpsize) / fsize
#                     num = ret * 100
#                     # 实现同一行打印下载进度
#                     print('下载进度: %.2f%%' % num)
#                     data = conn.recv(blocksize)
#                     if not data:
#                         break
#                     callback(data)
#                 cmpsize += blocksize
#
#         return self.voidresp()

class MyFTP(FTP):
    def retrbinary(self, cmd, callback, blocksize=8192, rest=None):
        self.voidcmd('TYPE I')

        total_size = self.size(cmd[5:])
        num = 0
        size = 0

        with self.transfercmd(cmd, rest) as conn:
            while True:

                data = conn.recv(blocksize)
                if not data:
                    break

                fraction = min(size / total_size * 100, 100.0)
                display_len = int(fraction / 10.0)
                progress = '█' * display_len + ' ' * (10 - display_len)
                print(f'Processing {num}: {fraction:3.0f}%|{progress}|')

                num += 1
                size += len(data)

                callback(data)
            # shutdown ssl layer
            if isinstance(conn, ssl.SSLSocket):
                conn.unwrap()

        return self.voidresp()


class CLAVRxData(object):
    def __init__(self, root_dir):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

    def get_dataset(self, time):
        directory = time.strftime(f'{self.root_dir}/%Y/%m/%d/%H/%M')
        filename = time.strftime('clavrx_H08_%Y%m%d_%H%M_B01_FLDK_R.level2.nc')
        path = f'{directory}/{filename}'
        os.makedirs(directory, exist_ok=True)
        if not os.path.exists(path):
            CLAVRxDownloader().download(time, path)
        dataset = xr.open_dataset(path)
        return dataset


# username = input('Please Enter Your USERNAME: ')
# password = getpass('Please Enter Your PASSWORD: ')

username = '810110663_qq.com'
password = 'SP+wari8'


class HimawariStandardData(object):
    _res = {
        1: 10, 2: 10, 3: 5, 4: 10, 5: 20, 6: 20, 7: 20, 8: 20,
        9: 20, 10: 20, 11: 20, 12: 20, 13: 20, 14: 20, 15: 20, 16: 20
    }

    def __init__(self, root_dir, time):
        self.root_dir = root_dir
        self.time = time

    def get_local_path(self, channel, segment):
        directory = f'{self.root_dir}/{self.time:%Y/%m/%d/%H/%M}'
        # directory = f'{self.root_dir}/{self.time:%Y/%Y%m/%Y%m%d}'
        filename = f'HS_H08_{self.time:%Y%m%d_%H%M}_B{channel:0>2}_FLDK_R{self._res[channel]:0>2}_S{segment:0>2}10.DAT.bz2'
        return directory, filename, f'{directory}/{filename}'

    def get_remote_path(self, channel, segment):
        directory = f'/jma/hsd/{self.time:%Y%m/%d/%H}'
        filename = f'HS_H08_{self.time:%Y%m%d_%H%M}_B{channel:0>2}_FLDK_R{self._res[channel]:0>2}_S{segment:0>2}10.DAT.bz2'
        return directory, filename, f'{directory}/{filename}'


class HSDData(object):
    def __init__(self, root_dir, time, channels=range(1, 17), segments=range(1, 11)):
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

        self.time = time
        self.channels = channels
        self.segments = segments
        self.hsd = HimawariStandardData(root_dir, time)

        self.host = 'ftp.ptree.jaxa.jp'
        self.port = 21
        self.username = username
        self.password = password

    def download(self, c, s):
        directory, filename, local_path = self.hsd.get_local_path(c, s)
        if not os.path.exists(local_path):
            os.makedirs(directory, exist_ok=True)
            remote_path = self.hsd.get_remote_path(c, s)[2]
            # downloader.download(local_path, self.hsd.get_remote_path(c, s)[2])
            print(f'download {remote_path}')
            ftp = MyFTP()
            ftp.connect(host=self.host, port=self.port)
            ftp.login(self.username, self.password)
            with open(local_path, 'wb') as fp:
                ftp.retrbinary(f'RETR {remote_path}', fp.write)
            ftp.quit()

    def get_dataset(self):
        # downloader = Himawari8Downloader(username, password)
        # with ThreadPoolExecutor() as executor:
        #     executor.map(self.download, itertools.product(self.channels, self.segments))
        # executor.map会早退出？
        futures = []
        with ThreadPoolExecutor() as executor:
            for c, s in itertools.product(self.channels, self.segments):
                futures.append(executor.submit(self.download, c, s))

        for future in futures:
            if future.exception() is not None:
                print(type(future.exception()).__name__, ':', future.exception())


class HimawariL1GriddedData(object):
    """
    Full-disk
     Projection: EQR
     Observation area: 60S-60N, 80E-160W
     Temporal resolution: 10-minutes
     Spatial resolution: 5km (Pixel number: 2401, Line number: 2401)
                         2km (Pixel number: 6001, Line number: 6001)
     Data: albedo(reflectance*cos(SOZ) of band01~band06)
           Brightness temperature of band07~band16
           satellite zenith angle, satellite azimuth angle,
           solar zenith angle, solar azimuth angle, observation hours (UT)
    """
    _number = {2: 6001, 5: 2401}

    def __init__(self, time):
        self.time = time

    def get_path(self):
        return f'/jma/netcdf/{self.time:%Y%m/%d}'
        # return self.time.strftime('/jma/netcdf/%Y%m/%d')

    def get_filename(self, resolution):
        pixel_number = line_number = f'{self._number[resolution]:0>5}'
        return f'NC_H08_{self.time:%Y%m%d_%H%M}_R21_FLDK.{pixel_number}_{line_number}.nc'

    def get_full_path(self, resolution):
        return f'{self.get_path()}/{self.get_filename(resolution)}'


class Himawari8Downloader(object):

    def __init__(self, username, password):
        self.host = 'ftp.ptree.jaxa.jp'
        self.port = 21
        self.username = username
        self.password = password

    def download(self, local_path, remote_path):
        print(f'download {remote_path}')
        ftp = MyFTP()
        ftp.connect(host=self.host, port=self.port)
        ftp.login(self.username, self.password)
        with open(local_path, 'wb') as fp:
            ftp.retrbinary(f'RETR {remote_path}', fp.write)
        ftp.quit()

    # def download_himawari_standard_data(self, periods, local_path, channels=range(1, 17), segments=range(1, 11)):
    #     ftp = MyFTP()
    #     ftp.connect(host=self.host, port=self.port)
    #     ftp.login(self.username, self.password)
    #     for time in periods:
    #         remote_path = HimawariStandardData(time).get_remote_dir()
    #         for c in channels:
    #             for s in segments:
    #                 filename = HimawariStandardData(time).get_remote_filename(c, s)
    #                 with open(local_path, 'wb') as fp:
    #                     ftp.retrbinary(f'RETR {remote_path}', fp.write)
    #     ftp.quit()

    # def download_himawari_l1_gridded_data(self, periods, local_path):
    #     ftp = MyFTP()
    #     ftp.connect(host=self.host, port=self.port)
    #     ftp.login(self.username, self.password)
    #     for time in periods:
    #         remote_path = HimawariL1GriddedData(time).get_full_path(resolution=2)
    #         # filename = HimawariL1GriddedData(time).get_filename(resolution=2)
    #         with open(local_path, 'wb') as fp:
    #             ftp.retrbinary(f'RETR {remote_path}', fp.write)
    #     ftp.quit()

    def download_himawari_l1_gridded_data(self, periods, paths):
        def _download(time, path):
            ftp = MyFTP()
            ftp.connect(host=self.host, port=self.port)
            ftp.login(self.username, self.password)
            remote_path = HimawariL1GriddedData(time).get_full_path(resolution=2)
            # filename = HimawariL1GriddedData(time).get_filename(resolution=2)
            with open(path, 'wb') as fp:
                ftp.retrbinary(f'RETR {remote_path}', fp.write)
            ftp.quit()

        # ProcessPoolExecutor 多进程为什么不会打印,比多线程下载速度快一丢丢，几乎没差别
        with ThreadPoolExecutor() as executor:
            executor.map(_download, periods, paths)


class CLAVRxDownloader(object):
    def __init__(self):
        self.host = 'ftp.ssec.wisc.edu'
        self.port = 21

    @staticmethod
    def get_nearest_time(time):
        if time.minute < 30:
            return time.replace(minute=0, second=0, microsecond=0)
        else:
            return time.replace(minute=30, second=0, microsecond=0)

    def download(self, time, local_path):
        remote_path = time.strftime('/pub/clavrx/real_time/level2/ahi/clavrx_H08_%Y%m%d_%H%M_B01_FLDK_R.level2.nc')
        print(f'download {remote_path}')
        ftp = MyFTP()
        ftp.connect(host=self.host, port=self.port)
        ftp.login()
        with open(local_path, 'wb') as fp:
            ftp.retrbinary(f'RETR {remote_path}', fp.write)
        ftp.quit()


# periods = pd.date_range(datetime(2019, 1, 1), datetime(2020, 1, 1), freq='10T', closed='left')
# print(len(periods))
#
#
# def process1(t):
#     remote_path = t.strftime('/jma/netcdf/%Y%m/%d')
#     local_path = t.strftime('/public/himawari/TOA/%Y/%m/%d')
#     filename = t.strftime('NC_H08_%Y%m%d_%H%M_R21_FLDK.06001_06001.nc')
#     os.makedirs(local_path, exist_ok=True)
#     download(filename, local_path, remote_path)
#     with xr.open_dataset(f'{local_path}/{filename}') as ds:
#         data_vars = ['albedo_01', 'albedo_02', 'albedo_03', 'albedo_04', 'albedo_05', 'albedo_06',
#                      'SAZ', 'SAA', 'SOZ', 'SOA', 'Hour']
#         gen_ds = xr.Dataset({name: ds[name] for name in data_vars})
#         gen_ds = gen_ds.expand_dims(dim={'time': [t]})
#         gen_ds = gen_ds.isel(latitude=np.arange(3001), longitude=np.arange(3001))
#         gen_ds.to_netcdf(f'{local_path}/{t.strftime('NC_H08_%Y%m%d_%H%M_R21_FLDK.03001_03001.nc')}',
#                          encoding={name: dict(zlib=True, complevel=9) for name in data_vars})
#     os.remove(f'{local_path}/{filename}')
#     print(t)
#
#
# def process2(t):
#     remote_path = t.strftime('/pub/himawari/L2/PAR/010/%Y%m/%d/%H')
#     local_path = t.strftime('/public/himawari/SWR/%Y/%m/%d')
#     filename = t.strftime('H08_%Y%m%d_%H%M_RFL010_FLDK.02401_02401.nc')
#     os.makedirs(local_path, exist_ok=True)
#     download(filename, local_path, remote_path)
#     with xr.open_dataset(f'{local_path}/{filename}') as ds:
#         data_vars = ['TAOT_02', 'TAAE', 'PAR', 'SWR', 'UVA', 'UVB', 'QA_flag']
#         gen_ds = xr.Dataset({name: ds[name] for name in data_vars})
#         gen_ds = gen_ds.expand_dims(dim={'time': [t]})
#         gen_ds = gen_ds.isel(latitude=np.arange(1201), longitude=np.arange(1201))
#         gen_ds.to_netcdf(f'{local_path}/{t.strftime('H08_%Y%m%d_%H%M_RFL010_FLDK.01201_01201.nc')}',
#                          encoding={name: dict(zlib=True, complevel=9) for name in data_vars})
#     os.remove(f'{local_path}/{filename}')
#     print(t)
#
#
# with ThreadPoolExecutor() as executor:
#     fs1 = list(executor.map(process1, periods))
#     fs2 = list(executor.map(process2, periods))


# data = []
# for t in periods:
#     if t.hour in (2, 14) and t.minute == 40:
#         data.append([t, float('nan')])
#         continue
#     try:
#         ds = xr.open_dataset(
#             t.strftime('/mnt/ftp148/himawari/SWR/%Y/%m/%d/H08_%Y%m%d_%H%M_RFL010_FLDK.02401_02401.nc'))
#     except Exception:  # np.linalg.LinAlgErrorException:
#         data.append([t, float('nan')])
#     else:
#         swr = ds['SWR'].interp(longitude=100.577, latitude=36.117, method='nearest').item()
#         data.append([t, swr])
#
# df = pd.DataFrame(data, columns=['datetime', 'swr'])
# df['cst'] = df['datetime'] + timedelta(hours=8)
# df.to_pickle('h8swr.pkl')


# def download2(filename, local_path, remote_path, rest, size, block_size=4096):
#     ftp = MyFTP()
#     ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
#     ftp.login(username, password)
#     # ftp.set_debuglevel(2)
#     ftp.getwelcome()
#     # ftp.cwd(remote_path)
#     ftp.voidcmd('TYPE I')
#     finished_size = 0
#     connection = ftp.transfercmd(f'RETR {remote_path}/{filename}', rest)
#     fp = open(f'{local_path}/{filename}', 'wb')
#     while size > 0:
#         data = connection.recv(block_size)
#         fp.write(data)
#         size -= block_size
#     connection.close()
#     fp.close()
#     ftp.quit()
#
#
# ftp = MyFTP()
# ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
# ftp.login(username, password)
# with ThreadPoolExecutor() as executor:
#     temp = ftp.sendcmd(f'SIZE {remote_path}/{filename}')
#     remote_file_size = int(temp.split()[1])
#     for i in range(0, math.ceil(remote_file_size / 1048576)):
#         executor.map(download2, )

# if __name__ == '__main__':
#     def download6(filename, local_path, remote_path):
#         ftp = MyFTP()
#         ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
#         ftp.login(username, password)
#         # ftp.set_debuglevel(2)
#         ftp.getwelcome()
#         ftp.set_pasv(True)
#         # ftp.cwd(remote_path)
#         ftp.voidcmd('TYPE I')
#         size = ftp.size(f'{remote_path}/{filename}')
#         ftp.quit()
#         block = 8192
#         print('???')
#
#         def xxx(rest):
#             ftp = MyFTP()
#             ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
#             ftp.login(username, password)
#             ftp.getwelcome()
#             ftp.set_pasv(True)
#             ftp.voidcmd('TYPE I')
#             print(5)
#             connection = ftp.transfercmd(f'RETR {remote_path}/{filename}', rest)
#             print(6)
#             data = connection.recv(block)
#             print(7)
#             connection.close()
#             print(8)
#             ftp.quit()
#             print(9)
#             return data
#
#         with ThreadPoolExecutor() as executor:
#             ret = executor.map(xxx, range(0, size, block))
#         fp = open(f'{local_path}/{filename}', 'wb')
#         fp.write(b''.join(ret))
#         fp.close()
#         print('ok')
#
#
#     download6(filename='HS_H08_20210418_0920_B01_FLDK_R10_S0210.DAT.bz2',
#               # local_path=r'C:\Users\dell\Desktop',
#               local_path=r'/data/developer_14',
#               remote_path='/jma/hsd/202104/18/09')
#
#     download(filename='HS_H08_20210418_0920_B01_FLDK_R10_S0210.DAT.bz2',
#              # local_path=r'C:\Users\dell\Desktop',
#              local_path=r'/data/developer_14',
#              remote_path='/jma/hsd/202104/18/09')
# import re
#
# import requests
#
# base = 'https://opendap.larc.nasa.gov/opendap/CALIPSO/LID_L1-Standard-V4-10/2020/06/contents.html'
# text = requests.get(base).text
#
# pattern = '<a itemprop='contentUrl'\n                              href='(CAL_LID_L1-Standard-V4-10\.\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z[DN].hdf)'>file</a>'
# reg = re.compile(pattern)
# match = reg.findall(text)
#
# # print(text)
# print(match)
# print(len(match))
#
# 'https://opendap.larc.nasa.gov/opendap/CALIPSO/LID_L2_01kmCLay-Standard-V4-20/2020/06/CAL_LID_L2_01kmCLay-Standard-V4-20.2020-06-01T00-05-38ZD.hdf.nc4?'

if __name__ == '__main__':
    # time_utc = datetime(2021, 10, 25, 5)
    # # gfs = GFSData('/data/developer_14/clavrx_data')
    # # gfs_ds = gfs.get_dataset(time_utc, 12)
    # # print(gfs_ds, '\n')
    #
    # hsd = HSDData('/data/developer_14/clavrx_data', time_utc, channels=range(1, 2), segments=range(1, 2))
    # hsd.get_dataset()

    import pandas as pd


    # downloader = Himawari8Downloader(username, password)
    # # downloader.download_himawari_l1_gridded_data([datetime(2021, 10, 25, 5)], '/data/developer_14/clavrx_data/x.nc')
    #
    # # ProcessPoolExecutor
    # # with ThreadPoolExecutor() as executor:
    # #     executor.map(_download, periods, paths)
    #
    # periods = pd.date_range(datetime(2021, 10, 25, 5), datetime(2021, 10, 25, 18, 20), freq='10T', closed='left')
    # paths = [f'/data/developer_14/clavrx_data/x{i}.nc' for i in range(80)]
    # downloader.download_himawari_l1_gridded_data(periods, paths)

    # def _download(time, path):
    #     ftp = MyFTP()
    #     ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
    #     ftp.login('810110663_qq.com', 'SP+wari8')
    #     remote_path = HimawariL1GriddedData(time).get_full_path(resolution=2)
    #     # filename = HimawariL1GriddedData(time).get_filename(resolution=2)
    #     with open(path, 'wb') as fp:
    #         ftp.retrbinary(f'RETR {remote_path}', fp.write)
    #     ftp.quit()

    def process(time: datetime):
        if time.minute != 40 or time.hour not in (2, 14):
            ftp = FTP()
            ftp.connect(host='ftp.ptree.jaxa.jp', port=21)
            ftp.login(user='810110663_qq.com', passwd='SP+wari8')

            remote_path = f'/jma/netcdf/{time:%Y%m/%d}'
            remote_filename = f'NC_H08_{time:%Y%m%d_%H%M}_R21_FLDK.06001_06001.nc'
            local_path = f'/mnt/ftp188/DSJShare/satellite/HMW8/{time:%Y/%m/%d/%H}'
            local_filename = f'{time:%Y%m%d_%H%M%S}.nc'

            os.makedirs(local_path, exist_ok=True)
            if f'{remote_path}/{remote_filename}' in ftp.nlst(remote_path):
                print(f'{remote_path}/{remote_filename}', 'exists')
                with open(f'{local_path}/{remote_filename}', 'wb') as fp:
                    ftp.retrbinary(f'RETR {remote_path}/{remote_filename}', fp.write)
            ftp.quit()
            ds = xr.open_dataset(f'{local_path}/{remote_filename}')
            ds = ds[[
                'albedo_01', 'albedo_02', 'albedo_03', 'albedo_04', 'albedo_05', 'albedo_06',
                'tbb_07', 'tbb_08', 'tbb_09', 'tbb_10', 'tbb_11',
                'tbb_12', 'tbb_13', 'tbb_14', 'tbb_15', 'tbb_16',
            ]].isel(latitude=np.arange(250, 2251), longitude=np.arange(3001)).expand_dims({'time': [time]}, 0)
            ds.attrs = {}
            ds.to_netcdf(f'{local_path}/{local_filename}')
            print(time, 'finished')


    futures = []
    periods = pd.date_range(datetime(2020, 1, 1), datetime(2021, 10, 1), freq='10T', closed='left')
    # with ThreadPoolExecutor() as executor:
    with ProcessPoolExecutor() as executor:
        for time in periods:
            futures.append(executor.submit(process, time))

    print('----------------------------------------------------------------------------------')
    for future in futures:
        if future.exception() is not None:
            print(type(future.exception()).__name__, ':', future.exception())
