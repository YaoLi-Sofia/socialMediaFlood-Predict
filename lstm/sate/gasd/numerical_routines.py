from math import sqrt, exp, nan, isnan

import numpy as np
from numba import njit, vectorize, prange

from constants import (
    sym_no,
)
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)


@vectorize(nopython=True, forceobj=False)
def vapor(t):
    """
    functions to compute some needed water vapor parameters

    t in Kelvin
    es in mbar
    """
    return 6.112 * exp(17.67 * (t - 273.16) / (t - 29.66))


@vectorize(nopython=True, forceobj=False)
def vapor_ice(t):
    """
    saturation vapor pressure for ice
    """
    return 6.1078 * exp(21.8745584 * (t - 273.16) / (t - 7.66))


@vectorize(nopython=True, forceobj=False)
def wind_speed(u, v):
    return sqrt(u * u + v * v)


@njit(nogil=True, error_model='numpy', boundscheck=True)
def locate(xx, n, x):
    assert xx.size > 1
    if xx[0] < xx[-1]:
        return min(np.searchsorted(xx, x, side='right'), n - 1) - 1
    else:
        return min(n - np.searchsorted(xx[::-1], x, side='right'), n - 1) - 1


# ==============================================================
# subroutine compute_median(z,mask,z_median,z_mean,z_std_median)
#
# median filter
# ==============================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def compute_median(z, mask):
    # the purpose of this function is to find
    # median (emed), minimum (emin) and maximum (emax)
    # for the array elem with nelem elements.
    nx, ny = z.shape
    x = []
    for i in range(nx):
        for j in range(ny):
            if mask[i, j] == sym_no and not isnan(z[i, j]):
                x.append(z[i, j])
    if len(x) > 0:
        x = np.array(x)
        z_median = np.median(x)
        z_std_median = sqrt(np.mean(np.square(x - z_median)))
    else:
        z_median = z_std_median = nan
    return z_median, z_std_median


# compute standard deviaion of an array wrt to the median
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def compute_median_segment(z, mask, n):
    z_median = np.full(image_shape, nan, 'f4')
    z_std_median = np.full(image_shape, nan, 'f4')

    for i in prange(image_number_of_lines):
        if i < n:
            i1 = 0
        else:
            i1 = i - n
        if i > image_number_of_lines - n - 1:
            i2 = image_number_of_lines
        else:
            i2 = i + n + 1
        for j in prange(image_number_of_elements):
            if j < n:
                j1 = 0
            else:
                j1 = j - n
            if j > image_number_of_elements - n:
                j2 = image_number_of_elements
            else:
                j2 = j + n

            z_median[i, j], z_std_median[i, j] = compute_median(z[i1:i2, j1:j2], mask[i1:i2, j1:j2])

    return z_median, z_std_median


# ====================================================================
# function name: covariance
#
# function:
#    compute the covariance for two mxn arrays
#
# description: covariance = e(xy) - e(x)*e(y)
#
# calling sequence: bt_wv_bt_window_covar[line_idx,elem_idx] = covariance( &
#                       sat%bt10(arr_right:arr_left,arr_top:arr_bottom), &
#                       sat%bt14(arr_right:arr_left,arr_top:arr_bottom), &
#                      array_width, array_hgt)
#
#
# inputs:
#   array 1 - the first array (x)
#   array 2 - the second array (y)
#   elem_size
#   line_size
#
# outputs:
#   covariance of x and y
#
# dependencies:
#        none
#
# restrictions:  none
#
# reference: standard definition for the covariance computation
#
# ====================================================================
@njit(nogil=True, error_model='numpy', boundscheck=True)
def covariance(array_one, array_two, invalid_data_mask):
    # --- skip computation for pixel arrays with any missing data
    if np.any(invalid_data_mask):
        return nan

    # todo f8其余均为f4
    mean_array_one = np.mean(array_one)
    mean_array_two = np.mean(array_two)
    mean_array_one_x_array_two = np.mean(array_one * array_two)
    return mean_array_one_x_array_two - mean_array_one * mean_array_two

# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def polynomial_value(p, x):
#     y = 0.0
#     for v in p:
#         y *= x
#         y += v
#     return y
#
#
# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def square(x):
#     return x * x
#
#
# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def cubic(x):
#     return x * x * x
