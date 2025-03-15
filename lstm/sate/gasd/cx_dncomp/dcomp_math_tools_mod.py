# subroutine to find the inverse of a square matrix
# author : louisda16th a.k.a ashwith j. rego
# reference : algorithm has been well explained in:
# http://math.uww.edu/~mcfarlat/inverse.htm           
# http://www.tutor.ms.unimelb.edu.au/matrix/matrix_inverse.html
from math import floor

from numba import njit

from numerical_routines import locate


@njit(nogil=True, error_model='numpy', boundscheck=True)
def dcomp_interpolation_weight(count, value, data_in):
    assert data_in.size > 1

    index = locate(data_in, count, value)

    index = max(0, min(count - 2, index))
    index2 = max(0, min(count - 1, index + 1))

    weight = (value - data_in[index]) / (data_in[index2] - data_in[index])

    if weight < 0.:
        index = 0
        weight = 0.

    weight_out = weight
    index_out = index

    return weight_out, index_out


@njit(nogil=True, error_model='numpy', boundscheck=True)
def dcomp_interpolation_weight2(count, value, data_in):
    assert data_in.size > 1

    index = locate(data_in, count, value)

    index = max(0, min(count - 2, index))
    index2 = max(0, min(count - 1, index + 1))

    weight = (value - data_in[index]) / (data_in[index2] - data_in[index])

    if weight < 0.:
        index = 0
        weight = 0.

    near_index = floor(index + weight + 0.5)

    return near_index


# ---------------------------------------------------------------------------------
#
#  interpolate_2d
#
#  linear interpolation for a 2 x 2 
# 
#  returns interpolated value of a 2d array with 2 elements for each dimension
#
#  input: 
#     table:      3d array
#     wgt_dim1, wgt_dim2 : weights for each dimension
#
# ----------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def interpolate_2d(table, wgt_dim1, wgt_dim2, delta_1, delta_2):
    r = wgt_dim1
    s = wgt_dim2

    value = (
            (1.0 - r) * (1.0 - s) * table[0, 0] + r * (1.0 - s) * table[1, 0] +
            (1.0 - r) * s * table[0, 1] + r * s * table[1, 1]
    )
    d_val_d2 = ((1.0 - r) * (table[0, 1] - table[0, 0]) + r * (table[1, 1] - table[1, 0])) / delta_2
    d_val_d1 = ((1.0 - s) * (table[1, 0] - table[0, 0]) + s * (table[1, 1] - table[0, 1])) / delta_1

    return value, d_val_d1, d_val_d2
