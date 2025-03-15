import functools
import time
from math import nan

import matplotlib.pyplot as plt
import numpy as np

from constants import missing_value_int1, missing_value_int2, missing_value_int4


def show_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        print(f'{func.__name__} running')
        tic = time.time()
        result = func(*args, **kwargs)
        toc = time.time()
        print(toc - tic)
        return result

    return wrapper


def show(x):
    plt.figure()
    # plt.title()
    if x.dtype == np.dtype('i1'):
        plt.matshow(np.where(x != missing_value_int1, x, nan))
    elif x.dtype == np.dtype('i2'):
        plt.matshow(np.where(x != missing_value_int2, x, nan))
    elif x.dtype == np.dtype('i4'):
        plt.matshow(np.where(x != missing_value_int4, x, nan))
    elif x.dtype == np.dtype('f4'):
        plt.matshow(x)
    elif x.dtype == np.dtype('f8'):
        plt.matshow(x)
    else:
        plt.matshow(x)
    plt.colorbar()
    plt.show()


def show0(name, x):
    print(name)
    plt.figure(num=name)
    # plt.title(name)
    if x.dtype == np.dtype('i1'):
        plt.matshow(np.where(x != missing_value_int1, x, nan))
    elif x.dtype == np.dtype('i2'):
        plt.matshow(np.where(x != missing_value_int2, x, nan))
    elif x.dtype == np.dtype('i4'):
        plt.matshow(np.where(x != missing_value_int4, x, nan))
    elif x.dtype == np.dtype('f4'):
        plt.matshow(x)
    elif x.dtype == np.dtype('f8'):
        plt.matshow(x)
    else:
        plt.matshow(x)
    plt.colorbar()
    plt.show()


def show1(name, x):
    print(name)
    plt.figure(num=name)
    # plt.title(name)
    plt.matshow(x)
    plt.colorbar()
    plt.show()


def show2(name, x, y):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, num=name)
    ax0.matshow(x)
    ax1.matshow(y)
    diff = x - y
    im = ax2.matshow(diff)
    print(f'{np.nanmax(diff) = }')
    print(f'{np.nanmin(diff) = }')
    print(f'{np.nanmean(diff) = }')
    print(f'{np.nanmean(np.abs(diff)) = }')
    fig.colorbar(im, ax=(ax0, ax1, ax2))
    plt.show()


def show3(name, x, y):
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, num=name)
    ax0.matshow(x)
    ax1.matshow(y)
    diff = x != y
    im = ax2.matshow(diff)
    print(np.nansum(diff) / np.nansum(~np.isnan(diff)))
    fig.colorbar(im, ax=(ax0, ax1, ax2))
    plt.show()


def show4(name, x, y, scatter=False):
    diff = x - y
    print(f'{np.nanmax(diff) = }')
    print(f'{np.nanmin(diff) = }')
    print(f'{np.nanmean(diff) = }')
    print(f'{np.nanmean(np.abs(diff)) = }')
    fig, (ax0, ax1) = plt.subplots(1, 2, num=name)
    ax0.matshow(x)
    im = ax1.matshow(y)
    fig.colorbar(im, ax=(ax0, ax1))
    plt.show()

    diff = x - y
    fig, ax = plt.subplots(num=f'{name} diff')
    im = ax.matshow(diff)
    plt.colorbar(im, ax=ax)
    plt.show()

    if scatter:
        # xy = np.vstack([x.flat, y.flat])
        fig, ax = plt.subplots(num=f'{name} scatter')
        ax.scatter(x.flat, y.flat)
        # ax.scatter(x.flat, y.flat, c=gaussian_kde(xy)(xy), cmap='Spectral')
        # ax.plot((0.0, 1.2), (0.0, 1.2), transform=ax.transAxes, ls='--')
        plt.show()

    fig, ax = plt.subplots(num=f'{name} diff sort')
    ax.plot(np.sort(diff.flat))
    plt.show()
