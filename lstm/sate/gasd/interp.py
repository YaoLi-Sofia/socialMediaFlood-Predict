from math import floor, nan, isnan, copysign

import numpy as np
from numba import prange, b1, i4, f4
from numba.experimental import jitclass

from numerical_routines import locate

# nearest_1d_interpolator_spec = {
#     'zp_size': i4,
#     'index_z0': i4[:],
#     'index_z1': i4[:],
# }

linear_1d_interpolator_spec = {
    'zp_size': i4,
    'invalid': b1[:],
    'index_z0': i4[:],
    'index_z1': i4[:],
    'weight0': f4[:],
    'weight1': f4[:]
}

nearest_2d_interpolator_spec = {
    'line_size': i4,
    'elem_size': i4,
    'invalid': b1[:, :],
    'index_x0': i4[:, :],
    'index_y0': i4[:, :]
}

linear_2d_interpolator_spec = {
    'line_size': i4,
    'elem_size': i4,

    'invalid': b1[:, :],

    'index_x0': i4[:, :],
    'index_y0': i4[:, :],
    'index_x1': i4[:, :],
    'index_y1': i4[:, :],
    # todo 去掉
    'weight_x0': f4[:, :],
    'weight_x1': f4[:, :],
    'weight_y0': f4[:, :],
    'weight_y1': f4[:, :],

    'weight00': f4[:, :],
    'weight01': f4[:, :],
    'weight10': f4[:, :],
    'weight11': f4[:, :]
}


# @jitclass(spec=nearest_1d_interpolator_spec)
# class VerticalNearestInterpolator(object):
#     def __init__(self, z, zp):
#         z_size = z.size
#         zp_size = zp.size
#
#         assert np.all(zp >= z[0])
#         assert np.all(zp <= z[-1])
#
#         index_z = np.empty(zp_size, dtype='i4')
#
#         for i in prange(zp_size):
#             index_z[i] = locate(z, z_size, zp[i])
#
#         self.zp_size = zp_size
#         self.index_z = index_z
#
#     def interp(self, values):
#         zp_size = self.zp_size
#         index_z = self.index_z
#
#         ret = np.empty(zp_size, dtype=values.dtype)
#         for i in prange(zp_size):
#             ret[i] = values[index_z[i]]
#         return ret


@jitclass(spec=linear_1d_interpolator_spec)
class VerticalLinearInterpolator(object):
    def __init__(self, z, zp):
        z_size = z.size
        zp_size = zp.size

        assert np.all(zp >= z[0])
        assert np.all(zp <= z[-1])

        index_z0 = np.empty(zp_size, dtype='i4')
        index_z1 = np.empty(zp_size, dtype='i4')
        weight0 = np.empty(zp_size, dtype='f4')
        weight1 = np.empty(zp_size, dtype='f4')

        for i in prange(zp_size):
            index_z0[i] = locate(z, z_size, zp[i])
            index_z1[i] = index_z0[i] + 1
            weight0[i] = (zp[i] - z[index_z0[i]]) / (z[index_z1[i]] - z[index_z0[i]])
            weight1[i] = 1 - weight0[i]

        self.zp_size = zp_size
        self.index_z0 = index_z0
        self.index_z1 = index_z1
        self.weight0 = weight0
        self.weight1 = weight1

    def interp(self, values):
        zp_size = self.zp_size
        index_z0 = self.index_z0
        index_z1 = self.index_z1
        weight0 = self.weight0
        weight1 = self.weight1

        ret = np.empty(zp_size, dtype=values.dtype)
        for i in prange(zp_size):
            ret[i] = values[index_z0[i]] * weight1[i] + values[index_z1[i]] * weight0[i]
        return ret


# interpolator = VerticalLinearInterpolator(np.arange(10), np.array([0.0, 1.5, 2.2, 9.0]))
# print(interpolator.index_z0)
# print(interpolator.index_z1)
# print(interpolator.interp(np.arange(10.0) + 10.0))


@jitclass(spec=nearest_2d_interpolator_spec)
class NearestInterpolator(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = (x[-1] - x[0]) / (x_size - 1)
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')
        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i, j] = True
                else:
                    index_x0[i][j] = floor((xp[i][j] - x[0]) / delta_x + 0.5)
                    index_y0[i][j] = floor((yp[i][j] - y[0]) / delta_y + 0.5)

                    # index_x0[i][j] = round((xp[i][j] - x[0]) / delta_x)
                    # index_y0[i][j] = round((yp[i][j] - y[0]) / delta_y)

                    assert 0 <= index_x0[i][j] < x_size
                    assert 0 <= index_y0[i][j] < y_size

        self.line_size = line_size
        self.elem_size = elem_size
        self.invalid = invalid
        self.index_x0 = index_x0
        self.index_y0 = index_y0

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = values[index_y0[i][j], index_x0[i][j], ...]
        return ret


@jitclass(spec=linear_2d_interpolator_spec)
class LinearInterpolator0(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = (x[-1] - x[0]) / (x_size - 1)
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')
        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')
        index_x1 = np.empty((line_size, elem_size), dtype='i4')
        index_y1 = np.empty((line_size, elem_size), dtype='i4')
        weight00 = np.empty((line_size, elem_size), dtype='f4')
        weight01 = np.empty((line_size, elem_size), dtype='f4')
        weight10 = np.empty((line_size, elem_size), dtype='f4')
        weight11 = np.empty((line_size, elem_size), dtype='f4')

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i][j] = True
                else:
                    index_x = (xp[i][j] - x[0]) / delta_x
                    index_y = (yp[i][j] - y[0]) / delta_y

                    index_x0[i][j] = floor(index_x)
                    index_y0[i][j] = floor(index_y)

                    x_weight = index_x - index_x0[i][j]
                    y_weight = index_y - index_y0[i][j]

                    assert 0 <= index_x0[i][j] < x_size
                    assert 0 <= index_y0[i][j] < y_size

                    index_x1[i][j] = index_x0[i][j] + 1
                    index_y1[i][j] = index_y0[i][j] + 1

                    assert 0 <= index_x1[i][j] < x_size
                    assert 0 <= index_y1[i][j] < y_size

                    weight00[i][j] = y_weight * x_weight
                    weight01[i][j] = y_weight * (1.0 - x_weight)
                    weight10[i][j] = (1.0 - y_weight) * x_weight
                    weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

        self.line_size = line_size
        self.elem_size = elem_size
        self.invalid = invalid
        self.index_x0 = index_x0
        self.index_y0 = index_y0
        self.index_x1 = index_x1
        self.index_y1 = index_y1
        self.weight00 = weight00
        self.weight01 = weight01
        self.weight10 = weight10
        self.weight11 = weight11

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret


@jitclass(spec=linear_2d_interpolator_spec)
class LinearInterpolator(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = (x[-1] - x[0]) / (x_size - 1)
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')
        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')
        index_x1 = np.empty((line_size, elem_size), dtype='i4')
        index_y1 = np.empty((line_size, elem_size), dtype='i4')
        weight00 = np.empty((line_size, elem_size), dtype='f4')
        weight01 = np.empty((line_size, elem_size), dtype='f4')
        weight10 = np.empty((line_size, elem_size), dtype='f4')
        weight11 = np.empty((line_size, elem_size), dtype='f4')

        desc_x = delta_x < 0
        desc_y = delta_y < 0

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i][j] = True
                else:
                    index_x = (xp[i][j] - x[0]) / delta_x
                    index_y = (yp[i][j] - y[0]) / delta_y

                    # index_x0[i][j] = floor(index_x)
                    # index_y0[i][j] = floor(index_y)
                    #
                    # index_x1[i][j] = ceil(index_x)
                    # index_y1[i][j] = ceil(index_y)
                    #
                    # x_weight = index_x - index_x0[i][j]
                    # y_weight = index_y - index_y0[i][j]
                    #
                    # assert 0 <= index_x0[i][j] < x_size
                    # assert 0 <= index_y0[i][j] < y_size
                    # assert 0 <= index_x1[i][j] < x_size
                    # assert 0 <= index_y1[i][j] < y_size
                    #
                    # weight00[i][j] = y_weight * x_weight
                    # weight01[i][j] = y_weight * (1.0 - x_weight)
                    # weight10[i][j] = (1.0 - y_weight) * x_weight
                    # weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

                    index_x0[i][j] = floor(index_x + 0.5)
                    index_y0[i][j] = floor(index_y + 0.5)

                    if (xp[i][j] > x[index_x0[i][j]]) ^ desc_x:
                        x_weight = index_x - index_x0[i][j]
                        index_x0[i][j] %= x_size
                        index_x1[i][j] = index_x0[i][j] + 1
                        if index_x1[i][j] >= x_size:
                            index_x1[i][j] -= x_size
                    else:
                        x_weight = index_x0[i][j] - index_x
                        index_x0[i][j] %= x_size
                        index_x1[i][j] = index_x0[i][j] - 1
                        if index_x1[i][j] < 0:
                            index_x1[i][j] += x_size

                    if (yp[i][j] > y[index_y0[i][j]]) ^ desc_y:
                        y_weight = index_y - index_y0[i][j]
                        assert 0 <= index_y0[i][j] < y_size
                        index_y1[i][j] = index_y0[i][j] + 1
                        assert 0 <= index_y1[i][j] < y_size
                    else:
                        y_weight = index_y0[i][j] - index_y
                        assert 0 <= index_y0[i][j] < y_size
                        index_y1[i][j] = index_y0[i][j] - 1
                        assert 0 <= index_y1[i][j] < y_size

                    assert 0 <= x_weight <= 0.5
                    assert 0 <= y_weight <= 0.5

                    weight00[i][j] = y_weight * x_weight
                    weight01[i][j] = y_weight * (1.0 - x_weight)
                    weight10[i][j] = (1.0 - y_weight) * x_weight
                    weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

        self.line_size = line_size
        self.elem_size = elem_size
        self.invalid = invalid
        self.index_x0 = index_x0
        self.index_y0 = index_y0
        self.index_x1 = index_x1
        self.index_y1 = index_y1
        self.weight00 = weight00
        self.weight01 = weight01
        self.weight10 = weight10
        self.weight11 = weight11

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def interp2(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                elif (isnan(values[index_y0[i][j]][index_x1[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x0[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x1[i][j]])):
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def nearest_interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
        return ret


@jitclass(spec=nearest_2d_interpolator_spec)
class GlobalNearestInterpolator(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = copysign(360.0, x[-1] - x[0]) / x_size
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit
        delta = x[0] - x[-1]
        if delta < 0.0:
            delta += 360.0
        elif delta >= 360.0:
            delta -= 360.0
        assert abs(delta - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for x_ in xp.flat:
            assert isnan(x_) or -180.0 <= x_ <= 180.0
        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')
        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i, j] = True
                else:
                    xpo = xp[i][j]
                    while xpo < x[0] and xpo < x[-1]:
                        xpo += 360.0
                    while xpo > x[0] and xpo > x[-1]:
                        xpo -= 360.0
                    ypo = yp[i][j]

                    # index_x0[i][j] = floor((xpo - x[0]) / delta_x + 0.5) % x_size
                    index_x0[i][j] = floor((xpo - x[0]) / delta_x + 0.5)
                    index_y0[i][j] = floor((ypo - y[0]) / delta_y + 0.5)

                    assert 0 <= index_y0[i][j] < y_size
        self.line_size = line_size
        self.elem_size = elem_size
        self.invalid = invalid
        self.index_x0 = index_x0
        self.index_y0 = index_y0

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
        return ret


@jitclass(spec=linear_2d_interpolator_spec)
class GlobalLinearInterpolator0(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = copysign(360.0, x[-1] - x[0]) / x_size
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit
        delta = x[0] - x[-1]
        if delta < 0.0:
            delta += 360.0
        elif delta >= 360.0:
            delta -= 360.0
        assert abs(delta - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for x_ in xp.flat:
            assert isnan(x_) or -180.0 <= x_ <= 180.0
        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')
        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')
        index_x1 = np.empty((line_size, elem_size), dtype='i4')
        index_y1 = np.empty((line_size, elem_size), dtype='i4')
        weight00 = np.empty((line_size, elem_size), dtype='f4')
        weight01 = np.empty((line_size, elem_size), dtype='f4')
        weight10 = np.empty((line_size, elem_size), dtype='f4')
        weight11 = np.empty((line_size, elem_size), dtype='f4')

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i][j] = True
                else:
                    index_x = (xp[i][j] - x[0]) / delta_x
                    index_y = (yp[i][j] - y[0]) / delta_y

                    index_x0[i][j] = floor(index_x)
                    index_y0[i][j] = floor(index_y)

                    x_weight = index_x - index_x0[i][j]
                    y_weight = index_y - index_y0[i][j]

                    index_x0[i][j] %= x_size
                    assert 0 <= index_y0[i][j] < y_size

                    index_x1[i][j] = index_x0[i][j] + 1
                    index_y1[i][j] = index_y0[i][j] + 1

                    if index_x1[i][j] >= x_size:
                        index_x1[i][j] -= x_size
                    assert 0 <= index_y1[i][j] < y_size

                    weight00[i][j] = y_weight * x_weight
                    weight01[i][j] = y_weight * (1.0 - x_weight)
                    weight10[i][j] = (1.0 - y_weight) * x_weight
                    weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

        self.line_size = line_size
        self.elem_size = elem_size
        self.invalid = invalid
        self.index_x0 = index_x0
        self.index_y0 = index_y0
        self.index_x1 = index_x1
        self.index_y1 = index_y1
        self.weight00 = weight00
        self.weight01 = weight01
        self.weight10 = weight10
        self.weight11 = weight11

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def interp2(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                elif (isnan(values[index_y0[i][j]][index_x1[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x0[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x1[i][j]])):
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def nearest_interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
        return ret


@jitclass(spec=linear_2d_interpolator_spec)
class GlobalLinearInterpolator(object):
    def __init__(self, x, y, xp, yp):
        x_size = x.size
        y_size = y.size

        delta_x = copysign(360.0, x[-1] - x[0]) / x_size
        delta_y = (y[-1] - y[0]) / (y_size - 1)

        limit = abs(delta_x) * 1.0e-2
        for i in prange(x_size - 1):
            assert abs(x[i + 1] - x[i] - delta_x) < limit
        delta = x[0] - x[-1]
        if delta < 0.0:
            delta += 360.0
        elif delta >= 360.0:
            delta -= 360.0
        assert abs(delta - delta_x) < limit

        line_size, elem_size = np.shape(xp)
        assert (line_size, elem_size) == np.shape(yp)

        for x_ in xp.flat:
            assert isnan(x_) or -180.0 <= x_ <= 180.0
        for y_ in y:
            assert isnan(y_) or -90.0 <= y_ <= 90.0
        for y_ in yp.flat:
            assert isnan(y_) or -90.0 <= y_ <= 90.0

        invalid = np.zeros((line_size, elem_size), dtype='b1')

        index_x0 = np.empty((line_size, elem_size), dtype='i4')
        index_y0 = np.empty((line_size, elem_size), dtype='i4')
        index_x1 = np.empty((line_size, elem_size), dtype='i4')
        index_y1 = np.empty((line_size, elem_size), dtype='i4')

        weight_x0 = np.empty((line_size, elem_size), dtype='f4')
        weight_x1 = np.empty((line_size, elem_size), dtype='f4')
        weight_y0 = np.empty((line_size, elem_size), dtype='f4')
        weight_y1 = np.empty((line_size, elem_size), dtype='f4')

        weight00 = np.empty((line_size, elem_size), dtype='f4')
        weight01 = np.empty((line_size, elem_size), dtype='f4')
        weight10 = np.empty((line_size, elem_size), dtype='f4')
        weight11 = np.empty((line_size, elem_size), dtype='f4')

        desc_x = delta_x < 0
        desc_y = delta_y < 0

        for i in prange(line_size):
            for j in prange(elem_size):
                if isnan(xp[i][j]) or isnan(yp[i][j]):
                    invalid[i][j] = True
                else:
                    xpo = xp[i][j]
                    while xpo < x[0] and xpo < x[-1]:
                        xpo += 360.0
                    while xpo > x[0] and xpo > x[-1]:
                        xpo -= 360.0
                    ypo = yp[i][j]

                    index_x = (xpo - x[0]) / delta_x
                    index_y = (ypo - y[0]) / delta_y

                    # index_x0[i][j] = floor((xpo - x[0]) / delta_x + 0.5) % x_size
                    # index_y0[i][j] = floor((ypo - y[0]) / delta_y + 0.5)

                    index_x0[i][j] = floor(index_x + 0.5)
                    index_y0[i][j] = floor(index_y + 0.5)

                    if (xpo > x[index_x0[i][j]]) ^ desc_x:
                        x_weight = index_x - index_x0[i][j]
                        # index_x0[i][j] %= x_size
                        assert 0 <= index_x0[i][j] < x_size
                        index_x1[i][j] = index_x0[i][j] + 1
                        if index_x1[i][j] >= x_size:
                            index_x1[i][j] -= x_size
                    else:
                        x_weight = index_x0[i][j] - index_x
                        # index_x0[i][j] %= x_size
                        assert 0 <= index_x0[i][j] < x_size
                        index_x1[i][j] = index_x0[i][j] - 1
                        if index_x1[i][j] < 0:
                            index_x1[i][j] += x_size

                    if (ypo > y[index_y0[i][j]]) ^ desc_y:
                        y_weight = index_y - index_y0[i][j]
                        assert 0 <= index_y0[i][j] < y_size
                        index_y1[i][j] = index_y0[i][j] + 1
                        assert 0 <= index_y1[i][j] < y_size
                    else:
                        y_weight = index_y0[i][j] - index_y
                        assert 0 <= index_y0[i][j] < y_size
                        index_y1[i][j] = index_y0[i][j] - 1
                        assert 0 <= index_y1[i][j] < y_size

                    assert 0 <= x_weight <= 0.5
                    assert 0 <= y_weight <= 0.5
                    # if not 0 <= x_weight <= 0.5:
                    #     print('x')
                    #     print(x_weight)
                    #     print(index_x, index_x0[i][j], index_x1[i][j])
                    #     print(xpo, x[index_x0[i][j]])
                    # if not 0 <= y_weight <= 0.5:
                    #     print('y')
                    #     print(y_weight)
                    #     print(index_y, index_y0[i][j], index_y1[i][j])
                    #     print(ypo, y[index_y0[i][j]])

                    # print(x_weight)
                    # print(y_weight)

                    weight_x0[i][j] = x_weight
                    weight_x1[i][j] = 1.0 - x_weight
                    weight_y0[i][j] = y_weight
                    weight_y1[i][j] = 1.0 - y_weight

                    weight00[i][j] = y_weight * x_weight
                    weight01[i][j] = y_weight * (1.0 - x_weight)
                    weight10[i][j] = (1.0 - y_weight) * x_weight
                    weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

                    # x_weight = index_x - index_x0[i][j]
                    # y_weight = index_y - index_y0[i][j]
                    #
                    # index_x0[i][j] %= x_size
                    # assert 0 <= index_y0[i][j] < y_size
                    #
                    # index_x1[i][j] = index_x0[i][j] + 1
                    # index_y1[i][j] = index_y0[i][j] + 1
                    #
                    # if index_x1[i][j] >= x_size:
                    #     index_x1[i][j] -= x_size
                    # assert 0 <= index_y1[i][j] < y_size
                    #
                    # weight00[i][j] = y_weight * x_weight
                    # weight01[i][j] = y_weight * (1.0 - x_weight)
                    # weight10[i][j] = (1.0 - y_weight) * x_weight
                    # weight11[i][j] = (1.0 - y_weight) * (1.0 - x_weight)

        self.line_size = line_size
        self.elem_size = elem_size

        self.invalid = invalid

        self.index_x0 = index_x0
        self.index_y0 = index_y0
        self.index_x1 = index_x1
        self.index_y1 = index_y1

        self.weight_x0 = weight_x0
        self.weight_x1 = weight_x1
        self.weight_y0 = weight_y0
        self.weight_y1 = weight_y1

        self.weight00 = weight00
        self.weight01 = weight01
        self.weight10 = weight10
        self.weight11 = weight11

    def interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size

        invalid = self.invalid

        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1

        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def interp2(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0
        index_x1 = self.index_x1
        index_y1 = self.index_y1
        weight00 = self.weight00
        weight01 = self.weight01
        weight10 = self.weight10
        weight11 = self.weight11

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                # todo 只能处理二维的数据
                elif (isnan(values[index_y0[i][j]][index_x1[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x0[i][j]]) or
                      isnan(values[index_y1[i][j]][index_x1[i][j]])):
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
                else:
                    ret[i][j] = (
                            values[index_y0[i][j]][index_x0[i][j]] * weight11[i][j] +
                            values[index_y0[i][j]][index_x1[i][j]] * weight10[i][j] +
                            values[index_y1[i][j]][index_x0[i][j]] * weight01[i][j] +
                            values[index_y1[i][j]][index_x1[i][j]] * weight00[i][j]
                    )
        return ret

    def nearest_interp(self, values, missing=nan):
        line_size = self.line_size
        elem_size = self.elem_size
        invalid = self.invalid
        index_x0 = self.index_x0
        index_y0 = self.index_y0

        ret = np.empty((line_size, elem_size, *values.shape[2:]), dtype=values.dtype)
        for i in prange(line_size):
            for j in prange(elem_size):
                if invalid[i][j]:
                    ret[i][j] = missing
                else:
                    ret[i][j] = values[index_y0[i][j]][index_x0[i][j]]
        return ret

# if __name__ == '__main__':
#     import time
#
#     from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
#     import matplotlib.pyplot as plt
#
#     x = np.linspace(80, 200, 61)
#     y = np.linspace(55, 15, 21)
#     print(x)
#     print(y)
#     z = np.random.random((21, 61)) * 100
#     x_size = 121
#     y_size = 121
#     X = np.linspace(100, 130, x_size)
#     Y = np.linspace(50, 20, y_size)
#     print(X)
#     print(Y)
#
#     tic = time.time()
#     interp = NearestNDInterpolator(np.stack(np.meshgrid(x, y), axis=2).reshape((-1, 2)), z.flat)
#     Z0 = interp(np.stack(np.meshgrid(X, Y), axis=2).reshape((-1, 2))).reshape((y_size, x_size))
#     toc = time.time()
#     print(toc - tic)
#
#     tic = time.time()
#     interpolator = NearestInterpolator(x, y, *np.meshgrid(X, Y))
#     Z1 = interpolator.interp(z)
#     toc = time.time()
#     print(toc - tic)
#
#     fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
#     ax0.matshow(Z0)
#     ax1.matshow(Z1)
#     diff = Z0 - Z1
#     im = ax2.matshow(diff)
#     print(np.amax(diff))
#     print(np.amin(diff))
#     print(np.mean(diff))
#     print(np.mean(np.abs(diff)))
#     fig.colorbar(im, ax=(ax0, ax1, ax2))
#     plt.savefig()
#
#     plt.plot(np.sort(diff.flat))
#     plt.savefig()
#     plt.plot(np.sort(np.abs(diff.flat)))
#     plt.savefig()
#     print()
#
#     tic = time.time()
#     interp = LinearNDInterpolator(np.stack(np.meshgrid(x, y), axis=2).reshape((-1, 2)), z.flat)
#     Z0 = interp(np.stack(np.meshgrid(X, Y), axis=2).reshape((-1, 2))).reshape((y_size, x_size))
#     toc = time.time()
#     print(toc - tic)
#
#     tic = time.time()
#     interpolator = LinearInterpolator0(x, y, *np.meshgrid(X, Y))
#     Z1 = interpolator.interp(z)
#     toc = time.time()
#     print(toc - tic)
#
#     fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
#     ax0.matshow(Z0)
#     ax1.matshow(Z1)
#     diff = Z0 - Z1
#     im = ax2.matshow(diff)
#     print(np.amax(diff))
#     print(np.amin(diff))
#     print(np.mean(diff))
#     print(np.mean(np.abs(diff)))
#     fig.colorbar(im, ax=(ax0, ax1, ax2))
#     plt.savefig()
#
#     plt.plot(np.sort(diff.flat))
#     plt.savefig()
#     plt.plot(np.sort(np.abs(diff.flat)))
#     plt.savefig()
#     print()
#
#     tic = time.time()
#     interpolator = LinearInterpolator(x, y, *np.meshgrid(X, Y))
#     Z1 = interpolator.interp(z)
#     toc = time.time()
#     print(toc - tic)
#
#     fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
#     ax0.matshow(Z0)
#     ax1.matshow(Z1)
#     diff = Z0 - Z1
#     im = ax2.matshow(diff)
#     print(np.amax(diff))
#     print(np.amin(diff))
#     print(np.mean(diff))
#     print(np.mean(np.abs(diff)))
#     fig.colorbar(im, ax=(ax0, ax1, ax2))
#     plt.savefig()
#
#     plt.plot(np.sort(diff.flat))
#     plt.savefig()
#     plt.plot(np.sort(np.abs(diff.flat)))
#     plt.savefig()
#     print()
#
#     # z = np.random.random((401, 1201, 2)) * 100
#     #
#     # tic = time.time()
#     # interpolator = NearestInterpolator(x, y, *np.meshgrid(X, Y))
#     # Z1 = interpolator.interp(z)
#     # toc = time.time()
#     # print(toc - tic)
#     #
#     # tic = time.time()
#     # interpolator = LinearInterpolator0(x, y, *np.meshgrid(X, Y))
#     # Z1 = interpolator.interp(z)
#     # toc = time.time()
#     # print(toc - tic)
