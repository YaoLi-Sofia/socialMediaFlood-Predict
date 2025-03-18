import numpy as np
from numba import njit

from calibration_constants import c1, c2
from numerical_routines import locate

n_planck = 161
t_planck_min = 180.0
delta_t_planck = 1.0

t_planck = t_planck_min + np.arange(n_planck, dtype='f4') * delta_t_planck

planck_nu = np.array([
    [2575.74],  # [2688.1720430107525],
    [1609.03],  # [2688.1720430107525],
    [1442.11],  # [1600.0],
    [1361.42],  # [1408.4507042253522],
    [1164.48],  # [1176.4705882352941],
    [1038.14],
    [961.385],
    [890.827],  # [925.9259259259259],
    [809.396],  # [833.3333333333334],
    [753.420]  # [740.7407407407408]
], 'f4')

# planck_a1 = np.array([
#     [-0.48017157],  # [-4.039856001490023],
#     [-1.7375338],  # [-4.051184456676424],
#     [-0.33773876],  # [-3.0556708891877236],
#     [-0.063970289],  # [-0.8320809914960137],
#     [-0.16272536],  # [-3.229269661150738],
#     [-0.11570521],
#     [-0.12402298],
#     [-0.25851106],  # [-1.0889860229248143],
#     [-0.38964267],  # [-1.7943497349614859],
#     [-0.10497392]  # [-0.8575749497480842]
# ], 'f4')
# planck_a2 = np.array([
#     [1.0007819],  # [1.00359256],
#     [1.0045224],  # [1.00362997],
#     [1.0009682],  # [1.0089376],
#     [1.0001925],  # [1.00197997],
#     [1.0005736],  # [1.00983844],
#     [1.0004572],
#     [1.0005265],
#     [1.0011875],  # [1.00441814],
#     [1.0019678],  # [1.00774111],
#     [1.0005657],  # [1.00394511]
# ], 'f4')

planck_a1 = np.zeros((10, 1), 'f4')
planck_a2 = np.ones((10, 1), 'f4')

bb_rad = c1 * planck_nu ** 3 / (np.exp(planck_a2 * c2 * planck_nu / (t_planck - planck_a1)) - 1.0)


# ------------------------------------------------------------------
# function planck_rad_fast(c, t, db_dt) results(b)
#
# subroutine to convert brightness temperature to radiance using a
# look-up table *function that returns a scalar*.
# ------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def planck_rad_fast(c, t):
    # --- compute planck emission for cloud temperature
    loc = int((t - t_planck_min) / delta_t_planck)
    loc = max(0, min(n_planck - 2, loc))

    db_dt = (bb_rad[c - 7, loc + 1] - bb_rad[c - 7, loc]) / (t_planck[loc + 1] - t_planck[loc])
    b = bb_rad[c - 7, loc] + (t - t_planck[loc]) * db_dt

    return b, db_dt


# ------------------------------------------------------------------------
# function planck_temp_fast(c, b, db_dt) results(t)
#
# subroutine to convert radiance (b) to brightness temperature(t) using a
# look-up table *function that returns a scalar*.
# ------------------------------------------------------------------
@njit(nogil=True, error_model='numpy', boundscheck=True)
def planck_temp_fast(c, b):
    # ---- compute brightness temperature
    loc = locate(bb_rad[c - 7, :], n_planck, b)
    loc = max(0, min(n_planck - 2, loc))
    db_dt = (bb_rad[c - 7, loc + 1] - bb_rad[c - 7, loc]) / (t_planck[loc + 1] - t_planck[loc])
    t = t_planck[loc] + (b - bb_rad[c - 7, loc]) / db_dt

    return t, db_dt

#
# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def planck_rad(c, t):
#     planck_nu = {
#         7: 2575.74,
#         8: 1609.03,
#         9: 1442.11,
#         10: 1361.42,
#         11: 1164.48,
#         12: 1038.14,
#         13: 961.385,
#         14: 890.827,
#         15: 809.396,
#         16: 753.420
#     }
#     planck_a1 = {
#         7: -0.48017157,
#         8: -1.7375338,
#         9: -0.33773876,
#         10: -0.063970289,
#         11: -0.16272536,
#         12: -0.11570521,
#         13: -0.12402298,
#         14: -0.25851106,
#         15: -0.38964267,
#         16: -0.10497392
#     }
#     planck_a2 = {
#         7: 1.0007819,
#         8: 1.0045224,
#         9: 1.0009682,
#         10: 1.0001925,
#         11: 1.0005736,
#         12: 1.0004572,
#         13: 1.0005265,
#         14: 1.0011875,
#         15: 1.0019678,
#         16: 1.0005657
#     }
#     return c1 * planck_nu[c] ** 3 / (
#             exp(planck_a2[c] * c2 * planck_nu[c] / (t - planck_a1[c])) - 1.0)
#
#
# @njit(nogil=True, error_model='numpy', boundscheck=True)
# def planck_temp(c, b):
#     if b <= 0:
#         return nan
#     planck_nu = {
#         7: 2575.74,
#         8: 1609.03,
#         9: 1442.11,
#         10: 1361.42,
#         11: 1164.48,
#         12: 1038.14,
#         13: 961.385,
#         14: 890.827,
#         15: 809.396,
#         16: 753.420
#     }
#     planck_a1 = {
#         7: -0.48017157,
#         8: -1.7375338,
#         9: -0.33773876,
#         10: -0.063970289,
#         11: -0.16272536,
#         12: -0.11570521,
#         13: -0.12402298,
#         14: -0.25851106,
#         15: -0.38964267,
#         16: -0.10497392
#     }
#     planck_a2 = {
#         7: 1.0007819,
#         8: 1.0045224,
#         9: 1.0009682,
#         10: 1.0001925,
#         11: 1.0005736,
#         12: 1.0004572,
#         13: 1.0005265,
#         14: 1.0011875,
#         15: 1.0019678,
#         16: 1.0005657
#     }
#     return planck_a1[c] + planck_a2[c] * c2 * planck_nu[c] / log(1.0 + c1 * planck_nu[c] ** 3 / b)
