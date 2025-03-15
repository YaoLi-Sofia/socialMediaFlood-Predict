from math import log, exp, sqrt

import numpy as np
from numba import prange, njit


@njit(nogil=True, error_model='numpy', boundscheck=True)
def calpir(
        t_avg_ref, amt_wet_ref, amt_ozo_ref,
        t_avg, amt_wet, amt_ozo,
        p_avg, sec_theta, n_layers,
        n_dry_pred, n_wet_pred, n_ozo_pred,
        n_con_pred
):
    # version of 18.03.03
    # purpose:
    # routine to calculate the predictors for the dry (temperature),
    # wet and ozone components of a fast transmittance model for a
    # scanning satellite based instrument.
    # references:
    # airs ftc package science notes and software, s. hannon and l. strow,
    # uni. of maryland, baltimore county (umbc)
    # created:
    # 19-sep-1996 hmw
    # arguments:
    # input
    # t_avg_ref - reference layer average temperature array (k)
    # amt_wet_ref - reference water vapour amount array (k.mol)/cm^2
    # amt_ozo_ref - reference ozone amount array (k.mol)/cm^2
    # t_avg - layer average temperature array (k)
    # amt_wet - water vapour amount array (k.mol)/cm^2
    # amt_ozo - ozone amount array (k.mol)/cm^2
    # p_avg - layer average pressure array (mb)
    # sec_theta - secant of the zenith angle array
    # n_layers - number of atmospheric layers
    # n_dry_pred - number of dry (temperature) predictors
    # n_wet_pred - number of water vapour predictors
    # n_ozo_pred - number of ozone predictors
    # n_con_pred - number of water vapour continuum predictors
    # output
    # pred_dry - dry gas (temperature) predictor matrix
    # pred_wet - water vapour predictor matrix
    # pred_ozo - ozone predictor matrix
    # pred_con - water vapour continuum predictor matrix
    # comments:
    # levels or layers?
    # profile data is input at a number of *layers*.
    # layer numbering pt. a
    # layer 1 => atmosphere between levels 1 & 2
    # layer 2 => atmosphere between levels 2 & 3
    # layer l-1 => atmosphere between levels l-1 & l
    # layer numbering pt. b
    # for the his instrument, layer 1 is at the top of the atmosphere
    # and layer l-1 is at the surface.
    # layer numbering pt. c
    # in this routine the number of *layers* is passed in the argument
    # list, _not_ the number of levels. this was done to improve
    # the readability of this code, i.e. loop from 1->layers
    # rather than from 1->levels-1.
    # =======================================================================
    # parameters

    max_layers = 100

    max_dry_pred = 8
    max_wet_pred = 13
    max_ozo_pred = 9
    max_con_pred = 4

    pred_dry = np.empty((max_layers, max_dry_pred), dtype='f4')
    pred_wet = np.empty((max_layers, max_wet_pred), dtype='f4')
    pred_ozo = np.empty((max_layers, max_ozo_pred), dtype='f4')
    pred_con = np.empty((max_layers, max_con_pred), dtype='f4')

    # arrays

    # pressure
    p_dp = np.empty(max_layers)
    p_norm = np.empty(max_layers)

    # temperature
    delta_t = np.empty(max_layers)
    t_ratio = np.empty(max_layers)
    pw_t_ratio = np.empty(max_layers)  # pressure weighted

    # water vapour
    wet_ratio = np.empty(max_layers)
    pw_wet = np.empty(max_layers)  # pressure weighted
    pw_wet_ref = np.empty(max_layers)  # pressure weighted
    pw_wet_ratio = np.empty(max_layers)  # pressure weighted

    # ozone
    ozo_ratio = np.empty(max_layers)
    pw_ozo_ratio = np.empty(max_layers)  # pressure weighted
    pow_t_ratio = np.empty(max_layers)  # pressure/ozone weighted

    # check that n_layers is o.k.
    assert n_layers <= max_layers
    # check that numbers of predictors is consistent
    # of dry (temperature) predictors
    assert n_dry_pred == max_dry_pred
    # of water vapour predictors
    assert n_wet_pred == max_wet_pred
    # of ozone predictors
    assert n_ozo_pred == max_ozo_pred
    # of water vapour continuum predictors
    assert n_con_pred == max_con_pred

    # calculate ratios, offsets, etc, for top layer

    # pressure variables
    p_dp[0] = p_avg[0] * (p_avg[1] - p_avg[0])
    p_norm[0] = 0.0

    # temperature variables
    delta_t[0] = t_avg[0] - t_avg_ref[0]
    t_ratio[0] = t_avg[0] / t_avg_ref[0]
    pw_t_ratio[0] = 0.0

    # amount variables
    # water vapour
    wet_ratio[0] = amt_wet[0] / amt_wet_ref[0]
    pw_wet[0] = p_dp[0] * amt_wet[0]
    pw_wet_ref[0] = p_dp[0] * amt_wet_ref[0]
    pw_wet_ratio[0] = wet_ratio[0]

    # ozone
    ozo_ratio[0] = amt_ozo[0] / amt_ozo_ref[0]
    pw_ozo_ratio[0] = 0.0
    pow_t_ratio[0] = 0.0

    # calculate ratios, offsets, etc, for all layers
    for l in range(1, n_layers):
        # pressure variables
        p_dp[l] = p_avg[l] * (p_avg[l] - p_avg[l - 1])
        p_norm[l] = p_norm[l - 1] + p_dp[l]

        # temperature variables
        delta_t[l] = t_avg[l] - t_avg_ref[l]
        t_ratio[l] = t_avg[l] / t_avg_ref[l]
        pw_t_ratio[l] = pw_t_ratio[l - 1] + (p_dp[l] * t_ratio[l - 1])

        # amount variables

        # water vapour
        wet_ratio[l] = amt_wet[l] / amt_wet_ref[l]
        pw_wet[l] = pw_wet[l - 1] + (p_dp[l] * amt_wet[l])
        pw_wet_ref[l] = pw_wet_ref[l - 1] + (p_dp[l] * amt_wet_ref[l])

        # ozone
        ozo_ratio[l] = amt_ozo[l] / amt_ozo_ref[l]
        pw_ozo_ratio[l] = pw_ozo_ratio[l - 1] + (p_dp[l] * ozo_ratio[l - 1])
        pow_t_ratio[l] = pow_t_ratio[l - 1] + (p_dp[l] * ozo_ratio[l - 1] * delta_t[l - 1])

    # scale the pressure dependent variables

    for l in prange(1, n_layers):
        pw_t_ratio[l] = pw_t_ratio[l] / p_norm[l]
        pw_wet_ratio[l] = pw_wet[l] / pw_wet_ref[l]
        pw_ozo_ratio[l] = pw_ozo_ratio[l] / p_norm[l]
        pow_t_ratio[l] = pow_t_ratio[l] / p_norm[l]

    # load up predictor arrays

    for l in prange(n_layers):
        # temperature predictors
        pred_dry[l, 0] = sec_theta[l]
        pred_dry[l, 1] = sec_theta[l] * sec_theta[l]
        pred_dry[l, 2] = sec_theta[l] * t_ratio[l]
        pred_dry[l, 3] = pred_dry[l, 2] * t_ratio[l]
        pred_dry[l, 4] = t_ratio[l]
        pred_dry[l, 5] = t_ratio[l] * t_ratio[l]
        pred_dry[l, 6] = sec_theta[l] * pw_t_ratio[l]
        pred_dry[l, 7] = pred_dry[l, 6] / t_ratio[l]

        # water vapour predictors
        pred_wet[l, 0] = sec_theta[l] * wet_ratio[l]
        pred_wet[l, 1] = sqrt(pred_wet[l, 0])
        pred_wet[l, 2] = pred_wet[l, 0] * delta_t[l]
        pred_wet[l, 3] = pred_wet[l, 0] * pred_wet[l, 0]
        pred_wet[l, 4] = abs(delta_t[l]) * delta_t[l] * pred_wet[l, 0]
        pred_wet[l, 5] = pred_wet[l, 0] * pred_wet[l, 3]
        pred_wet[l, 6] = sec_theta[l] * pw_wet_ratio[l]
        pred_wet[l, 7] = pred_wet[l, 1] * delta_t[l]
        pred_wet[l, 8] = sqrt(pred_wet[l, 1])
        pred_wet[l, 9] = pred_wet[l, 6] * pred_wet[l, 6]
        pred_wet[l, 10] = sqrt(pred_wet[l, 6])
        pred_wet[l, 11] = pred_wet[l, 0]
        # +++ old
        # pred_wet(13,l) = pred_wet(2,l)
        # +++ new
        pred_wet[l, 12] = pred_wet[l, 0] / pred_wet[l, 9]

        # ozone predictors
        pred_ozo[l, 0] = sec_theta[l] * ozo_ratio[l]
        pred_ozo[l, 1] = sqrt(pred_ozo[l, 0])
        pred_ozo[l, 2] = pred_ozo[l, 0] * delta_t[l]
        pred_ozo[l, 3] = pred_ozo[l, 0] * pred_ozo[l, 0]
        pred_ozo[l, 4] = pred_ozo[l, 1] * delta_t[l]
        pred_ozo[l, 5] = sec_theta[l] * pw_ozo_ratio[l]
        pred_ozo[l, 6] = sqrt(pred_ozo[l, 5]) * pred_ozo[l, 0]
        pred_ozo[l, 7] = pred_ozo[l, 0] * pred_wet[l, 0]
        pred_ozo[l, 8] = sec_theta[l] * pow_t_ratio[l] * pred_ozo[l, 0]

        # water vapour continuum predictors
        pred_con[l, 0] = sec_theta[l] * wet_ratio[l] / (t_ratio[l] * t_ratio[l])
        pred_con[l, 1] = pred_con[l, 0] * pred_con[l, 0] / sec_theta[l]
        pred_con[l, 2] = sec_theta[l] * wet_ratio[l] / t_ratio[l]
        pred_con[l, 3] = pred_con[l, 2] * wet_ratio[l]

    return pred_dry, pred_wet, pred_ozo, pred_con


@njit(nogil=True, error_model='numpy', boundscheck=True)
def conpir(p, t, w, o, n_levels):
    # version of 19.09.96
    # purpose:
    # function to convert atmospheric water vapour (g/kg) and ozone (ppmv)
    # profiles specified at n_levels layer boundaries to n_levels-1
    # integrated layer amounts of units (k.moles)/cm^2. the average
    # layer pressure and temperature are also returned.
    # references:
    # airs layers package science notes, s. hannon and l. strow, uni. of
    # maryland, baltimore county (umbc)
    # created:
    # 19-sep-1996 hmw
    # arguments:
    # input
    #
    # p - pressure array (mb)
    # t - temperature profile array (k)
    # w - water vapour profile array (g/kg)
    # o - ozone profile array (ppmv)
    # n_levels - number of elements used in passed arrays
    # output
    #
    # p_avg - average layer pressure array (mb)
    # t_avg - average layer temperature (k)
    # w_amt - integrated layer water vapour amount array (k.moles)/cm^2
    # o_amt - integrated layer ozone amount array (k.moles)/cm^2
    # routines:
    # subroutines:
    #
    # gph_ite - calculates geopotential height given profile data.
    # comments:
    # levels or layers?
    # profile data is input at a number of *levels*. number densitites
    # are calculated for *layers* that are bounded by these levels.
    # so, for l levels there are l-1 layers.
    # layer numbering
    # layer 1 => atmosphere between levels 1 & 2
    # layer 2 => atmosphere between levels 2 & 3
    # layer l-1 => atmosphere between levels l-1 & l
    # =======================================================================
    # prevent implicit typing of variables
    # arguments
    # local variables
    # parameters

    max_levels = 101  # maximum number of layers

    p_avg = np.empty(max_levels - 1, dtype='f4')
    t_avg = np.empty(max_levels - 1, dtype='f4')
    w_amt = np.empty(max_levels - 1, dtype='f4')
    o_amt = np.empty(max_levels - 1, dtype='f4')

    r_equator = 6.378388e+06  # earth radius at equator
    r_polar = 6.356911e+06  # earth radius at pole
    r_avg = 0.5 * (r_equator + r_polar)

    g_sfc = 9.80665  # gravity at surface

    rho_ref = 1.2027e-12  # reference air 'density'

    mw_dryair = 28.97  # molec. wgt. of dry air (g/mol)
    mw_h2o = 18.0152  # molec. wgt. of water
    # mw_o3 = 47.9982 # molec. wgt. of ozone

    r_gas = 8.3143  # ideal gas constant (j/mole/k)
    r_air = 0.9975 * r_gas  # gas constant for air (worst case)

    # arrays

    g = np.empty(max_levels, dtype='f4')  # acc. due to gravity (m/s/s)
    mw_air = np.empty(max_levels, dtype='f4')  # molec. wgt. of air (g/mol)
    rho_air = np.empty(max_levels, dtype='f4')  # air mass density (kg.mol)/m^3
    c = np.empty(max_levels, dtype='f4')  # (kg.mol.k)/(n.m)
    w_ppmv = np.empty(max_levels, dtype='f4')  # h2o level amount (ppmv)

    # calculate initial values of pressure heights

    z = gph_ite(p, t, w, 0.0, n_levels)

    # set loop bounds for direction sensitive calculations
    # so loop iterates from surface to the top

    # data stored top down
    l_start = n_levels - 1
    l_end = -1

    # air molecular mass and density, and gravity
    # as a function of level

    # loop from bottom to top
    for l in prange(l_start, l_end, -1):
        # convert water vapour g/kg -> ppmv
        w_ppmv[l] = 1.0e+03 * w[l] * mw_dryair / mw_h2o

        # calculate molecular weight of air (g/mol)
        mw_air[l] = ((1.0 - (w_ppmv[l] / 1.0e+6)) * mw_dryair) + ((w_ppmv[l] / 1.0e+06) * mw_h2o)

        # air mass density
        c[l] = 0.001 * mw_air[l] / r_air  # 0.001 factor for g -> kg
        rho_air[l] = c[l] * p[l] / t[l]

        # gravity
        r_hgt = r_avg + z[l]  # m
        g[l] = g_sfc - g_sfc * (1.0 - ((r_avg * r_avg) / (r_hgt * r_hgt)))  # m/s^2

    # layer quantities

    # loop from bottom to top
    for l in range(l_start, l_end + 1, -1):
        # determine output array index. this is done so that the
        # output data is always ordered from 1 -> l-1 regardless
        # of the orientation of the input data. this is true by
        # default only for the bottom-up case. for the top down
        # case no correction would give output layers from 2 -> l

        l_indx = l - 1

        # assign current layer boundary densities

        rho1 = rho_air[l]
        rho2 = rho_air[l - 1]

        # average c
        c_avg = ((rho1 * c[l]) + (rho2 * c[l - 1])) / (rho1 + rho2)

        # average t
        t_avg[l_indx] = ((rho1 * t[l]) + (rho2 * t[l - 1])) / (rho1 + rho2)

        # average p
        p1 = p[l]
        p2 = p[l - 1]

        # z1 = z[l]
        # z2 = z[l - 1]

        dp = p2 - p1

        # a = log(p2 / p1) / (z2 - z1)
        # b = p1 / exp(a * z1)

        p_avg[l_indx] = dp / log(p2 / p1)

        # layer thickness (rather long-winded as it is not
        # assumed the layers are thin) in m. includes
        # correction for altitude/gravity.

        # initial values
        g_avg = g[l]
        dz = -1.0 * dp * t_avg[l_indx] / (g_avg * c_avg * p_avg[l_indx])

        # calculate z_avg
        z_avg = z[l] + (0.5 * dz)

        # calculate new g_avg
        r_hgt = r_avg + z_avg
        g_avg = g_sfc - g_sfc * (1.0 - ((r_avg * r_avg) / (r_hgt * r_hgt)))

        # calculate new dz
        dz = -1.0 * dp * t_avg[l_indx] / (g_avg * c_avg * p_avg[l_indx])

        # calculate layer amounts for water vapour

        w1 = w_ppmv[l]
        w2 = w_ppmv[l - 1]
        w_avg = ((rho1 * w1) + (rho2 * w2)) / (rho1 + rho2)
        w_amt[l_indx] = rho_ref * w_avg * dz * p_avg[l_indx] / t_avg[l_indx]

        # calculate layer amounts for ozone

        o1 = o[l]
        o2 = o[l - 1]
        o_avg = ((rho1 * o1) + (rho2 * o2)) / (rho1 + rho2)
        o_amt[l_indx] = rho_ref * o_avg * dz * p_avg[l_indx] / t_avg[l_indx]

    return p_avg, t_avg, w_amt, o_amt


@njit(nogil=True, error_model='numpy', boundscheck=True)
def gph_ite(p, t, w, z_sfc, n_levels):
    # version of 18.05.00

    # purpose:

    # routine to compute geopotential height given the atmospheric state.
    # includes virtual temperature adjustment.

    # created:

    # 19-sep-1996 received from hal woolf, recoded by paul van delst
    # 18-may-2000 logic error related to z_sfc corrected by hal woolf

    # arguments:

    # input
    #
    # p - pressure array (mb)

    # t - temperature profile array (k)

    # w - water vapour profile array (g/kg)

    # z_sfc - surface height (m). 0.0 if not known.

    # n_levels - number of elements used in passed arrays

    # 1 - direction of increasing layer number

    # 1 = +1, level[0] == p(top)  } satellite/ac
    # level[n_levels-1] == p(sfc) } case

    # 1 = -1, level[0] == p(sfc)  } ground-based
    # level[n_levels-1] == p(top) } case

    # output
    #
    # z - pressure level height array (m)

    # comments:

    # dimension of height array may not not be the same as that of the
    # input profile data.

    # =======================================================================

    z = np.empty(n_levels, dtype='f4')
    # parameters

    rog = 29.2898
    fac = 0.5 * rog

    # calculate virtual temperature adjustment and exponential
    # pressure height for level above surface. also set integration
    # loop bounds

    # data stored top down
    v_lower = t[n_levels - 1] * (1.0 + (0.00061 * w[n_levels - 1]))

    algp_lower = log(p[n_levels - 1])

    i_start = n_levels - 2
    i_end = -1

    hgt = z_sfc

    # following added 18 may 2000 . previously, z[n_levels-1] for downward
    # (usual) case was not defined#

    z[n_levels - 1] = z_sfc

    # end of addition

    # loop over layers always from sfc -> top
    for l in prange(i_start, i_end, -1):
        # apply virtual temperature adjustment for upper level
        v_upper = t[l]
        if p[l] >= 300.0:
            v_upper = v_upper * (1.0 + (0.00061 * w[l]))

        # calculate exponential pressure height for upper layer

        algp_upper = log(p[l])

        # calculate height

        hgt = hgt + (fac * (v_upper + v_lower) * (algp_lower - algp_upper))

        # overwrite values for next layer

        v_lower = v_upper
        algp_lower = algp_upper

        # store heights in same direction as other data
        z[l] = hgt

    return z


@njit(nogil=True, error_model='numpy', boundscheck=True)
def tau_doc(cc, xx):
    # * strow-woolf model . for dry, ozo(ne), and wco (water-vapor continuum)
    # version of 05.09.02

    n_lay, n_c = cc.shape
    # nx = xx.size

    tau = np.zeros(n_lay + 1, dtype='f4')
    tau[0] = 1.
    tau_lyr = 1.

    # loop over all layers from top
    for j in range(n_lay):
        # assume background stored in last column of coeffecients
        yy = cc[j, n_c - 1]

        # yy += cc[j, :nc - 2] @ xx[j, :nc - 2]
        yy += np.sum(cc[j, :n_c - 2] * xx[j, :n_c - 2])
        if yy > 0.:
            tau_lyr = exp(-yy)
        tau[j + 1] = tau[j] * tau_lyr

    return tau


# * strow-woolf model . for 'wet' (water-vapor other than continuum)
# version of 05.09.02
@njit(nogil=True, error_model='numpy', boundscheck=True)
def tau_wtr(ccs, ccl, xx):
    n_lay, nc = ccs.shape
    # nx = xx.size

    tau = np.zeros(n_lay + 1, dtype='f4')
    tau[0] = 1.
    tau_lyr = 1.

    od_sum = 0.

    cc = ccs

    for j in range(n_lay):

        # differentiate between ice and water
        # old
        # if ( od_sum >= 5.0 ) cc => ccl
        # heidinger fix begin
        if od_sum >= 5.0:
            cc = ccl
            nc = ccl.shape[1]

        # heidinger fix end

        # assume background stored in last column of coeffecients
        yy = cc[j, nc - 1]
        # ccm  yy = yy + dot_product( cc(:,j), xx(:nc-1,j) )
        # yy += cc[j, :nc - 2] @ xx[j, :nc - 2]  # not sure
        yy += np.sum(cc[j, :nc - 2] * xx[j, :nc - 2])  # not sure
        # end ccm
        od_sum += max(yy, 0.)
        if yy > 0.:
            tau_lyr = exp(-yy)
        tau[j + 1] = tau[j] * tau_lyr

    return tau
