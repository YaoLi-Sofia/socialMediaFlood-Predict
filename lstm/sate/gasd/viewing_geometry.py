from math import pi, degrees, radians, acos, sin, cos, asin, sqrt, nan, isnan

from numba import vectorize

r_earth = 6378.2064


@vectorize(nopython=True, forceobj=False)
def calculate_sensor_zenith(h, sat_lon, sat_lat, pix_lon, pix_lat):
    if isnan(pix_lon) or isnan(pix_lat):
        return nan
    x_lon = radians(pix_lon - sat_lon)
    x_lat = radians(pix_lat - sat_lat)
    r = r_earth
    tmp = cos(x_lat) * cos(x_lon)
    if tmp <= r / (r + h):
        return nan
    beta = acos(cos(x_lat) * cos(x_lon))
    sensor_zenith = (r + h) * sin(beta) / sqrt(r ** 2 + (r + h) ** 2 - 2.0 * r * (r + h) * cos(beta))
    sensor_zenith = max(-1.0, min(1.0, sensor_zenith))
    sensor_zenith = degrees(asin(sensor_zenith))
    return sensor_zenith


@vectorize(nopython=True, forceobj=False)
def calculate_relative_azimuth(sol_azi, sen_azi):
    if isnan(sen_azi) or isnan(sol_azi):
        return nan
    relative_azimuth = abs(sol_azi - sen_azi)
    # if relative_azimuth > 180.0:
    # relative_azimuth = 360.0 - relative_azimuth
    # relative_azimuth = 180.0 - relative_azimuth
    if relative_azimuth > 180.0:
        relative_azimuth -= 180.0
    else:
        relative_azimuth = 180.0 - relative_azimuth
    return relative_azimuth


@vectorize(nopython=True, forceobj=False)
def calculate_glint_angle(sol_zen, sen_zen, rel_az):
    if isnan(sol_zen) or isnan(sen_zen) or isnan(rel_az):
        return nan
    glint_angle = (
            cos(radians(sol_zen)) * cos(radians(sen_zen)) +
            sin(radians(sol_zen)) * sin(radians(sen_zen)) * cos(radians(rel_az))
    )
    glint_angle = max(-1.0, min(glint_angle, 1.0))
    glint_angle = degrees(acos(glint_angle))
    return glint_angle


@vectorize(nopython=True, forceobj=False)
def calculate_scattering_angle(sol_zen, sen_zen, rel_az):
    if isnan(sol_zen) or isnan(sen_zen) or isnan(rel_az):
        return nan
    scattering_angle = (
            cos(radians(sol_zen)) * cos(radians(sen_zen)) -
            sin(radians(sol_zen)) * sin(radians(sen_zen)) * cos(radians(rel_az))
    )
    scattering_angle *= -1.0
    scattering_angle = max(-1.0, min(scattering_angle, 1.0))
    scattering_angle = degrees(acos(scattering_angle))
    return scattering_angle


@vectorize(nopython=True, forceobj=False)
def calculate_sensor_azimuth(sat_lon, sat_lat, pix_lon, pix_lat):
    if isnan(pix_lon) or isnan(pix_lat):
        return nan
    x_lon = radians(pix_lon - sat_lon)
    x_lat = radians(pix_lat - sat_lat)
    beta = acos(cos(x_lat) * cos(x_lon))
    sine_beta = sin(beta)
    # todo
    # if abs(sine_beta) > np.finfo(sine_beta).eps:
    if abs(sine_beta) > 0.0:
        sensor_azimuth = sin(x_lon) / sine_beta
        sensor_azimuth = min(1.0, max(-1.0, sensor_azimuth))
        sensor_azimuth = degrees(asin(sensor_azimuth))
    else:
        sensor_azimuth = 0.0

    if x_lat < 0.0:
        sensor_azimuth = 180.0 - sensor_azimuth

    if sensor_azimuth < 0.0:
        sensor_azimuth += 360.0
    sensor_azimuth -= 180.0
    return sensor_azimuth


@vectorize(nopython=True, forceobj=False)
def calculate_solar_zenith(julian_day, tu, x_lon, x_lat):
    if isnan(x_lon) or isnan(x_lat):
        return nan
    tsm = tu + x_lon / 15.0
    x_lo = radians(x_lon)
    x_la = radians(x_lat)
    xj = float(julian_day)
    a1 = radians(1.00554 * xj - 6.28306)
    a2 = radians(1.93946 * xj + 23.35089)
    et = -7.67825 * sin(a1) - 10.09176 * sin(a2)
    tsv = tsm + et / 60.0
    tsv = (tsv - 12.0)
    ah = radians(tsv * 15.0)
    a3 = radians(0.9683 * xj - 78.00878)
    delta = radians(23.4856 * sin(a3))
    amuzero = sin(x_la) * sin(delta) + cos(x_la) * cos(delta) * cos(ah)
    elev = asin(amuzero)
    elev = degrees(elev)
    asol = 90.0 - elev
    return asol


@vectorize(nopython=True, forceobj=False)
def calculate_solar_azimuth(julian_day, tu, x_lon, x_lat):
    if isnan(x_lon) or isnan(x_lat):
        return nan
    tsm = tu + x_lon / 15.0
    xlo = radians(x_lon)
    xla = radians(x_lat)
    xj = float(julian_day)
    a1 = radians(1.00554 * xj - 6.28306)
    a2 = radians(1.93946 * xj + 23.35089)
    et = -7.67825 * sin(a1) - 10.09176 * sin(a2)
    tsv = tsm + et / 60.0
    tsv = (tsv - 12.0)
    ah = radians(tsv * 15.0)
    a3 = radians(0.9683 * xj - 78.00878)
    delta = radians(23.4856 * sin(a3))
    amuzero = sin(xla) * sin(delta) + cos(xla) * cos(delta) * cos(ah)
    elev = asin(amuzero)
    az = cos(delta) * sin(ah) / cos(elev)
    caz = (-cos(xla) * sin(delta) + sin(xla) * cos(delta) * cos(ah)) / cos(elev)
    if az >= 1.0:
        azim = asin(1.0)
    elif az <= -1.0:
        azim = asin(-1.0)
    else:
        azim = asin(az)

    if caz <= 0.0:
        azim = pi - azim

    if az <= 0.0 < caz:
        azim += 2 * pi
    azim += pi
    pi2 = 2 * pi
    if azim > pi2:
        azim -= pi2
    phis = degrees(azim)
    if phis > 180.0:
        phis -= 360.0
    return phis
