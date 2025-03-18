from math import nan, isnan, copysign

import numpy as np
from numba import njit, prange

from constants import (
    sym_overshooting_type,
)
from kdtree import kdtree2_result, kdtree2_create, kdtree2_n_nearest
from numerical_routines import locate
from public import (
    image_number_of_lines,
    image_number_of_elements,
    image_shape,
)
from .acha_parameters import (
    dt_dz_strato,
    dp_dz_strato,
    num_levels_rtm_prof,
)


# -----------------------------------------------------------------
# interpolate within profiles knowing z to determine t and p
# -----------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def knowing_z_compute_t_p(acha_rtm_nwp_z_prof, acha_rtm_nwp_p_prof, acha_rtm_nwp_t_prof, z):
    # --- initialize
    t = nan
    p = nan

    # --- check for missing
    if isnan(z):
        return p, t

    # --- interpolate pressure profile
    lev_idx = locate(acha_rtm_nwp_z_prof, num_levels_rtm_prof, z)
    lev_idx = max(0, min(num_levels_rtm_prof - 2, lev_idx))

    dp = acha_rtm_nwp_p_prof[lev_idx + 1] - acha_rtm_nwp_p_prof[lev_idx]
    dt = acha_rtm_nwp_t_prof[lev_idx + 1] - acha_rtm_nwp_t_prof[lev_idx]
    dz = acha_rtm_nwp_z_prof[lev_idx + 1] - acha_rtm_nwp_z_prof[lev_idx]

    # --- perform interpolation
    if dz != 0.0:
        t = acha_rtm_nwp_t_prof[lev_idx] + dt / dz * (z - acha_rtm_nwp_z_prof[lev_idx])
        p = acha_rtm_nwp_p_prof[lev_idx] + dp / dz * (z - acha_rtm_nwp_z_prof[lev_idx])
    else:
        t = acha_rtm_nwp_t_prof[lev_idx]
        p = acha_rtm_nwp_p_prof[lev_idx]

    return p, t


# -----------------------------------------------------------------
# interpolate within profiles knowing t to determine p and z
# look at the bottom first and move up
# -----------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def knowing_t_compute_p_z_bottom_up(
        cloud_type, t, t_tropo, z_tropo, p_tropo,
        acha_rtm_nwp_tropo_level, acha_rtm_nwp_sfc_level,
        acha_rtm_nwp_p_prof, acha_rtm_nwp_z_prof, acha_rtm_nwp_t_prof
):
    z = nan
    p = nan

    # --- check for missing
    if isnan(t):
        return p, z

    # --- test for existence of a valid solution with troposphere
    k_start = acha_rtm_nwp_tropo_level
    k_end = acha_rtm_nwp_sfc_level

    # --- check to see if warmer than max, than assume at surface
    if t > np.amax(acha_rtm_nwp_t_prof[k_start:k_end + 1]):
        p = acha_rtm_nwp_p_prof[k_end]
        z = acha_rtm_nwp_z_prof[k_end]

        return p, z

    # --- check to see if colder than min, than assume above tropopause
    # --- and either limit height to tropopause or extrapolate in stratosphere
    # todo 会不会有amin,amax,mean 等因nan导致的错误
    if t < np.amin(acha_rtm_nwp_t_prof[k_start:k_end + 1]) or t < t_tropo:
        if cloud_type == sym_overshooting_type:
            z = z_tropo + (t - t_tropo) / dt_dz_strato
            p = p_tropo + (z - z_tropo) * dp_dz_strato
        else:
            p = p_tropo
            z = z_tropo

        return p, z

    for k_lev in range(k_end, k_start - 2, -1):
        if ((acha_rtm_nwp_t_prof[k_lev] <= t < acha_rtm_nwp_t_prof[k_lev - 1]) or
                (acha_rtm_nwp_t_prof[k_lev] >= t > acha_rtm_nwp_t_prof[k_lev - 1])):
            break
    else:
        return p, z
    # --- general inversion
    dp = acha_rtm_nwp_p_prof[k_lev] - acha_rtm_nwp_p_prof[k_lev - 1]
    dt = acha_rtm_nwp_t_prof[k_lev] - acha_rtm_nwp_t_prof[k_lev - 1]
    dz = acha_rtm_nwp_z_prof[k_lev] - acha_rtm_nwp_z_prof[k_lev - 1]

    if dt != 0.0:
        p = acha_rtm_nwp_p_prof[k_lev] + dp / dt * (t - acha_rtm_nwp_t_prof[k_lev])
        z = acha_rtm_nwp_z_prof[k_lev] + dz / dt * (t - acha_rtm_nwp_t_prof[k_lev])
    else:
        p = acha_rtm_nwp_p_prof[k_lev]
        z = acha_rtm_nwp_z_prof[k_lev]

    return p, z


# ------------------------------------------------------------------------
# subroutine to compute the iteration in x due to optimal
# estimation
#
# the notation in this routine follows that of clive rodgers (1976,2000)
#
# input to this routine:
# iter_idx - the number of the current iteration
# iter_idx_max - the maximum number of iterations allowed
# nx - the number of x values
# ny - the number of y values
# convergence_criteria - the convergence criteria
# y - the vector of observations
# f - the vector of observations predicted by the forward model
# x - the vector of retrieved parameters
# x_ap - the vector of the apriori estimate of the retrieved parameters
# k - the kernel matrix
# sy - the covariance matrix of y and f
# sa_inv - the inverse of the covariance matrix of x_ap
# delta_x_max - the maximum step allowed for each delta_x value
#
# output of this routine:
# sx - the covariance matrix of x 
# delta_x - the increment in x for the next iteration
# converged_flag - flag indicating if convergence was met (yes or no)
# fail_flag - flag indicating if this process failed (yes or no)
#
# local variables:
# sx_inv - the inverse of sx
# delta_x_dir - the unit direction vectors for delta-x 
# delta_x_distance - the total length in x-space of delta_x
# delta_x_constrained - the values of delta_x after being constrained
# -----------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def optimal_estimation(
        iter_idx, iter_idx_max,
        convergence_criteria, delta_x_max,
        y, f, x, x_ap, k, sy, sa_inv,
        delta_x_prev,
        num_param,
):
    delta_x_constrained = np.empty(num_param, 'f4')

    converged_flag = 0
    fail_flag = 0
    delta_x = np.full(num_param, nan, 'f4')

    try:
        sy_inv = np.linalg.inv(sy)
    except Exception:  # np.linalg.LinAlgErrorException:
        print('cloud height warning ==> singular sy in acha ')
        fail_flag = 1  # todo !!!个人理解应该是1
        converged_flag = 0
        return delta_x, converged_flag, fail_flag

    # ---- compute next step
    akm = (k.T @ (sy_inv @ k))  # step saving
    sx_inv = sa_inv + akm  # (eq.102 rodgers)

    try:
        sx = np.linalg.inv(sx_inv)
    except Exception:  # np.linalg.LinAlgErrorException:
        print('cloud height warning ==> singular sx in acha ')
        fail_flag = 1
        converged_flag = 0
        return delta_x, converged_flag, fail_flag

    delta_x = sx @ ((k.T @ (sy_inv @ (y - f))) + (sa_inv @ (x_ap - x)))

    # --------------------------------------------------------------
    # check for convergence
    # --------------------------------------------------------------

    # --- compute convergence metric
    conv_test = abs(np.sum(delta_x * (sx_inv @ delta_x)))

    # -------------------------------------------------------------------
    # a direct constraint to avoid too large steps
    # -------------------------------------------------------------------
    for ix in range(num_param):
        delta_x_constrained[ix] = copysign(min(abs(delta_x[ix]), delta_x_max[ix]), delta_x[ix])

    for ix in range(num_param):
        delta_x[ix] = delta_x_constrained[ix]

    # if current and previous iteration delta_x has opposite signs, reduce current
    # magnitude
    if (not isnan(delta_x_prev[0])) and (delta_x_prev[0] * delta_x[0] < 0):
        for ix in range(num_param):
            delta_x[ix] = delta_x_constrained[ix] / 5.0

    # --- check for traditional convergence
    if conv_test < convergence_criteria:
        converged_flag = 1
        fail_flag = 0

    # --- check for exceeding allowed number of interactions
    if iter_idx > iter_idx_max:
        converged_flag = 0
        fail_flag = 1

    # return sx, akm, delta_x, conv_test, cost, goodness, converged_flag, fail_flag
    return delta_x, converged_flag, fail_flag


# -------------------------------------------------------------------------------------------------
# smooth a field using a mean over an area
#
# description
#    values of z_in with mask_in = 1 are used to populate pixels with mask_out
#    = 1
#    z_out is computed as the mean of z_in*mask_in over a box whose size is
#    defined by n.
#
# input
#    mask_in - binary mask of point used as the source of the smoothing
#    mask_out - binary mask of points to have a results upon exit
#    missing = missing value used as fill for z_out
#    count_thresh - number of source points to compute an output
#    n - half-width of smoothing box (x and y)
#    num_elements = size of array in x-direction
#    num_lines = size of array in y-direction
#    z_in - source values
#    z_out - output values
#    di = number of pixels to skip in the i direction (0=none,1=every other
#    ...)
#    dj = number of pixels to skip in the j direction (0=none,1=every other
#    ...)
#
# -------------------------------------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def mean_smooth2(mask_in, mask_out, di, dj, n, z_in, z_out):
    # z_out = np.zeros(image_shape, 'f4')
    z_out[:, :] = 0
    count_out = np.zeros(image_shape, 'i4')

    for i in range(di, image_number_of_lines - di, di + 1):

        i1 = min(image_number_of_lines - 1, max(0, i - n))
        i2 = min(image_number_of_lines, max(1, i + n + 1))

        for j in range(dj, image_number_of_elements - dj, dj + 1):

            if mask_in[i, j] == 0:
                continue
            if z_out[i, j] > 0:
                continue

            j1 = min(image_number_of_elements - 1, max(0, j - n))
            j2 = min(image_number_of_elements, max(1, j + n + 1))

            mask_in_sub = mask_in[i1:i2, j1:j2]
            mask_out_sub = mask_out[i1:i2, j1:j2]
            if np.sum(mask_out_sub) == 0:
                continue
            z_in_sub = z_in[i1:i2, j1:j2]
            count_temporary = np.sum(mask_in_sub)

            z_out[i1:i2, j1:j2] += np.sum(z_in_sub * mask_in_sub) / count_temporary
            count_out[i1:i2, j1:j2] += 1

    for i in prange(image_number_of_lines):
        for j in prange(image_number_of_elements):
            if count_out[i, j] > 0:
                z_out[i, j] /= count_out[i, j]
            elif count_out[i, j] == 0:
                z_out[i, j] = nan
            if mask_out[i, j] == 0:
                z_out[i, j] = nan

    # return z_out


@njit(nogil=True, error_model='numpy', boundscheck=True, parallel=True)
def kd_tree_interp_2pred(mask_in, mask_out, pred_var1, pred_var2, nnbrute, z_in, z_out):
    # this function performs kd tree search. z_in is the original data, and
    # z_out is the output where indices flagged by mask_output are reassinged
    # using average of values flagged by mask_in.
    # currently values where mask_in is set are not changed. nnbrute is the
    # number of found closet indices for each search

    # z_out_1d = np.full(image_number_of_lines * image_number_of_elements, nan, 'f4')
    z_out_1d = z_out.reshape(-1)

    # find indices of training and query variables
    # values at training indices (mask_in) can be computed again but original
    # values are kept here
    n_training = np.sum(mask_in)
    n_query = np.sum(mask_out) - np.sum(np.logical_and(mask_out, mask_in))

    # perform kd-tree only if both training and query indices are found
    if n_training > 0 and n_query > 0:
        # convert 2d to 1d array

        predictor_1 = pred_var1.reshape(image_number_of_lines * image_number_of_elements)
        predictor_2 = pred_var2.reshape(image_number_of_lines * image_number_of_elements)
        out_temp_1d = z_in.reshape(image_number_of_lines * image_number_of_elements)

        ind_training = np.empty(n_training, 'i4')
        ind_query = np.empty(n_query, 'i4')

        # search for training and query indices and stored
        i = 0
        j = 0
        for line_idx in range(image_number_of_lines):
            for elem_idx in range(image_number_of_elements):

                if (mask_in[line_idx, elem_idx] and
                        (not isnan(pred_var1[line_idx, elem_idx])) and
                        (not isnan(pred_var2[line_idx, elem_idx]))):
                    ind_training[i] = line_idx * image_number_of_elements + elem_idx
                    i += 1

                if (mask_out[line_idx, elem_idx] and (not mask_in[line_idx, elem_idx]) and
                        (not isnan(pred_var1[line_idx, elem_idx])) and
                        (not isnan(pred_var2[line_idx, elem_idx]))):
                    ind_query[j] = line_idx * image_number_of_elements + elem_idx
                    j += 1

        tmp = np.empty(n_training, 'f4')
        for i in range(n_training):
            tmp[i] = out_temp_1d[ind_training[i]]
        if np.any(np.isnan(tmp)):
            print('kdtree training is not correct')

        # if np.any(isnan(out_temp_1d[ind_training])):
        #     print('kdtree training is not correct')

        my_array = np.empty((2, n_training), 'f4')
        query_vec = np.empty(2, 'f4')
        # if ( not  allocated(my_array)) allocate(my_array(2,n_training))
        # if ( not  allocated(query_vec)) allocate(query_vec(2))

        # assign training data
        tmp = np.empty(n_training, 'f4')
        for i in range(n_training):
            tmp[i] = predictor_1[ind_training[i]]
        my_array[0, :] = tmp
        tmp = np.empty(n_training, 'f4')
        for i in range(n_training):
            tmp[i] = predictor_2[ind_training[i]]
        my_array[1, :] = tmp

        # my_array[0, :] = predictor_1[ind_training]
        # my_array[1, :] = predictor_2[ind_training]

        # create tree, set sort to true slows it down but the output indices are
        # ordered from closet to farthest; set rearrange as true speeds searches
        # but requires extra memory

        tree = kdtree2_create(my_array, sort=True, rearrange=True)  # this is how you create a tree.
        results1 = np.empty(nnbrute, kdtree2_result)

        # set 1d output variable values at training indices
        for i in range(n_training):
            z_out_1d[ind_training[i]] = out_temp_1d[ind_training[i]]

        # perform tree search for each query index
        for i in range(n_query):
            query_vec[0] = predictor_1[ind_query[i]]
            query_vec[1] = predictor_2[ind_query[i]]

            # results1 has both indices and distances 
            kdtree2_n_nearest(tp=tree, qv=query_vec, nn=nnbrute, results=results1)

            # average values for the all found indices
            tmp = np.empty(nnbrute, 'f4')
            for j in range(nnbrute):
                tmp[j] = out_temp_1d[ind_training[results1[j]['idx']]]

            # todo
            # z_out_1d[ind_query[i]] = np.sum(temp) / results1.size
            z_out_1d[ind_query[i]] = np.sum(tmp) / nnbrute
            # z_out_1d[ind_query[i]] = sum(out_temp_1d[ind_training[results1['idx']]]) / size(results1['idx'])

        # destroy tree and release memory
        # kdtree2_destroy(tree)

    # change 1d array back to 2d; if no kdtree search is performed, array is
    # empty
    # z_out = z_out_1d.reshape(image_shape)
    #
    # return z_out
