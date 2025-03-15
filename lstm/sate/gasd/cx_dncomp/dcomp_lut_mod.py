#   PURPOSE : return data from LUT
#   input: channel, phase, cod (log10), cps(log10)
#   output : transmission, reflectance, cloud albedo, spherical albedo
#          several derivates
#   derivates we need:
#       d_refl_d_cps
#       d_refl_d_cod

from math import nan

import numpy as np
import xarray as xr
from numba import i1, f4
from numba import njit
from numba.core.types import UniTuple
from numba.typed import Dict

from .dcomp_math_tools_mod import interpolate_2d

# ch03_ref_ice_ds = xr.open_dataset('static/FY4A_ch2_ref_lut_ice_cld.hdf', engine='netcdf4')
# ch03_ice_cld_alb = ch03_ref_ice_ds['albedo'].values
# ch03_ice_cld_trn = ch03_ref_ice_ds['transmission'].values
# ch03_ice_cld_sph_alb = ch03_ref_ice_ds['spherical_albedo'].values
# ch03_ice_cld_refl = ch03_ref_ice_ds['reflectance'].values
#
# ch03_ref_wat_ds = xr.open_dataset('static/FY4A_ch2_ref_lut_wat_cld.hdf', engine='netcdf4')
# ch03_wat_cld_alb = ch03_ref_wat_ds['albedo'].values
# ch03_wat_cld_trn = ch03_ref_wat_ds['transmission'].values
# ch03_wat_cld_sph_alb = ch03_ref_wat_ds['spherical_albedo'].values
# ch03_wat_cld_refl = ch03_ref_wat_ds['reflectance'].values
#
# ch05_ref_ice_ds = xr.open_dataset('static/FY4A_ch5_ref_lut_ice_cld.hdf', engine='netcdf4')
# ch05_ice_cld_alb = ch05_ref_ice_ds['albedo'].values
# ch05_ice_cld_trn = ch05_ref_ice_ds['transmission'].values
# ch05_ice_cld_sph_alb = ch05_ref_ice_ds['spherical_albedo'].values
# ch05_ice_cld_refl = ch05_ref_ice_ds['reflectance'].values
#
# ch05_ref_wat_ds = xr.open_dataset('static/FY4A_ch5_ref_lut_wat_cld.hdf', engine='netcdf4')
# ch05_wat_cld_alb = ch05_ref_wat_ds['albedo'].values
# ch05_wat_cld_trn = ch05_ref_wat_ds['transmission'].values
# ch05_wat_cld_sph_alb = ch05_ref_wat_ds['spherical_albedo'].values
# ch05_wat_cld_refl = ch05_ref_wat_ds['reflectance'].values
#
# ch06_ref_ice_ds = xr.open_dataset('static/FY4A_ch6_ref_lut_ice_cld.hdf', engine='netcdf4')
# ch06_ice_cld_alb = ch06_ref_ice_ds['albedo'].values
# ch06_ice_cld_trn = ch06_ref_ice_ds['transmission'].values
# ch06_ice_cld_sph_alb = ch06_ref_ice_ds['spherical_albedo'].values
# ch06_ice_cld_refl = ch06_ref_ice_ds['reflectance'].values
#
# ch06_ref_wat_ds = xr.open_dataset('static/FY4A_ch6_ref_lut_wat_cld.hdf', engine='netcdf4')
# ch06_wat_cld_alb = ch06_ref_wat_ds['albedo'].values
# ch06_wat_cld_trn = ch06_ref_wat_ds['transmission'].values
# ch06_wat_cld_sph_alb = ch06_ref_wat_ds['spherical_albedo'].values
# ch06_wat_cld_refl = ch06_ref_wat_ds['reflectance'].values
#
# ch07_ref_ice_ds = xr.open_dataset('static/FY4A_ch7_ref_lut_ice_cld.hdf', engine='netcdf4')
# ch07_ice_cld_alb = ch07_ref_ice_ds['albedo'].values
# ch07_ice_cld_trn = ch07_ref_ice_ds['transmission'].values
# ch07_ice_cld_sph_alb = ch07_ref_ice_ds['spherical_albedo'].values
# ch07_ice_cld_refl = ch07_ref_ice_ds['reflectance'].values
#
# ch07_ref_wat_ds = xr.open_dataset('static/FY4A_ch7_ref_lut_wat_cld.hdf', engine='netcdf4')
# ch07_wat_cld_alb = ch07_ref_wat_ds['albedo'].values
# ch07_wat_cld_trn = ch07_ref_wat_ds['transmission'].values
# ch07_wat_cld_sph_alb = ch07_ref_wat_ds['spherical_albedo'].values
# ch07_wat_cld_refl = ch07_ref_wat_ds['reflectance'].values
#
# dims_sat_zen = ch07_ref_ice_ds['sensor_zenith_angle'].values
# dims_sol_zen = ch07_ref_ice_ds['solar_zenith_angle'].values
# dims_rel_azi = ch07_ref_ice_ds['relative_azimuth_angle'].values
# dims_cod = ch07_ref_ice_ds['log10_optical_depth'].values
# dims_cps = ch07_ref_ice_ds['log10_eff_radius'].values
#
# ch07_ems_ice_ds = xr.open_dataset('static/FY4A_ch7_ems_lut_ice_cld.hdf', engine='netcdf4')
# ch07_ice_cld_ems = ch07_ems_ice_ds['cloud_emissivity'].values
# ch07_ice_cld_trn_ems = ch07_ems_ice_ds['cloud_transmission'].values
#
# ch07_ems_wat_ds = xr.open_dataset('static/FY4A_ch7_ems_lut_wat_cld.hdf', engine='netcdf4')
# ch07_wat_cld_ems = ch07_ems_wat_ds['cloud_emissivity'].values
# ch07_wat_cld_trn_ems = ch07_ems_wat_ds['cloud_transmission'].values

ch03_ref_ice_ds = xr.open_dataset('static/AHI_ch3_ref_lut_ice_cld.nc', engine='netcdf4')
ch03_ice_cld_alb = ch03_ref_ice_ds['albedo'].values
ch03_ice_cld_trn = ch03_ref_ice_ds['transmission'].values
ch03_ice_cld_sph_alb = ch03_ref_ice_ds['spherical_albedo'].values
ch03_ice_cld_refl = ch03_ref_ice_ds['reflectance'].values

ch03_ref_wat_ds = xr.open_dataset('static/AHI_ch3_ref_lut_wat_cld.nc', engine='netcdf4')
ch03_wat_cld_alb = ch03_ref_wat_ds['albedo'].values
ch03_wat_cld_trn = ch03_ref_wat_ds['transmission'].values
ch03_wat_cld_sph_alb = ch03_ref_wat_ds['spherical_albedo'].values
ch03_wat_cld_refl = ch03_ref_wat_ds['reflectance'].values

ch05_ref_ice_ds = xr.open_dataset('static/AHI_ch5_ref_lut_ice_cld.nc', engine='netcdf4')
ch05_ice_cld_alb = ch05_ref_ice_ds['albedo'].values
ch05_ice_cld_trn = ch05_ref_ice_ds['transmission'].values
ch05_ice_cld_sph_alb = ch05_ref_ice_ds['spherical_albedo'].values
ch05_ice_cld_refl = ch05_ref_ice_ds['reflectance'].values

ch05_ref_wat_ds = xr.open_dataset('static/AHI_ch5_ref_lut_wat_cld.nc', engine='netcdf4')
ch05_wat_cld_alb = ch05_ref_wat_ds['albedo'].values
ch05_wat_cld_trn = ch05_ref_wat_ds['transmission'].values
ch05_wat_cld_sph_alb = ch05_ref_wat_ds['spherical_albedo'].values
ch05_wat_cld_refl = ch05_ref_wat_ds['reflectance'].values

ch06_ref_ice_ds = xr.open_dataset('static/AHI_ch6_ref_lut_ice_cld.nc', engine='netcdf4')
ch06_ice_cld_alb = ch06_ref_ice_ds['albedo'].values
ch06_ice_cld_trn = ch06_ref_ice_ds['transmission'].values
ch06_ice_cld_sph_alb = ch06_ref_ice_ds['spherical_albedo'].values
ch06_ice_cld_refl = ch06_ref_ice_ds['reflectance'].values

ch06_ref_wat_ds = xr.open_dataset('static/AHI_ch6_ref_lut_wat_cld.nc', engine='netcdf4')
ch06_wat_cld_alb = ch06_ref_wat_ds['albedo'].values
ch06_wat_cld_trn = ch06_ref_wat_ds['transmission'].values
ch06_wat_cld_sph_alb = ch06_ref_wat_ds['spherical_albedo'].values
ch06_wat_cld_refl = ch06_ref_wat_ds['reflectance'].values

ch07_ref_ice_ds = xr.open_dataset('static/AHI_ch7_ref_lut_ice_cld.nc', engine='netcdf4')
ch07_ice_cld_alb = ch07_ref_ice_ds['albedo'].values
ch07_ice_cld_trn = ch07_ref_ice_ds['transmission'].values
ch07_ice_cld_sph_alb = ch07_ref_ice_ds['spherical_albedo'].values
ch07_ice_cld_refl = ch07_ref_ice_ds['reflectance'].values

ch07_ref_wat_ds = xr.open_dataset('static/AHI_ch7_ref_lut_wat_cld.nc', engine='netcdf4')
ch07_wat_cld_alb = ch07_ref_wat_ds['albedo'].values
ch07_wat_cld_trn = ch07_ref_wat_ds['transmission'].values
ch07_wat_cld_sph_alb = ch07_ref_wat_ds['spherical_albedo'].values
ch07_wat_cld_refl = ch07_ref_wat_ds['reflectance'].values

dims_sat_zen = ch07_ref_ice_ds['sensor_zenith_angle'].values
dims_sol_zen = ch07_ref_ice_ds['solar_zenith_angle'].values
dims_rel_azi = ch07_ref_ice_ds['relative_azimuth_angle'].values
dims_cod = ch07_ref_ice_ds['log10_optical_depth'].values
dims_cps = ch07_ref_ice_ds['log10_eff_radius'].values

ch07_ems_ice_ds = xr.open_dataset('static/AHI_ch7_ems_lut_ice_cld.nc', engine='netcdf4')
ch07_ice_cld_ems = ch07_ems_ice_ds['cloud_emissivity'].values
ch07_ice_cld_trn_ems = ch07_ems_ice_ds['cloud_transmission'].values

ch07_ems_wat_ds = xr.open_dataset('static/AHI_ch7_ems_lut_wat_cld.nc', engine='netcdf4')
ch07_wat_cld_ems = ch07_ems_wat_ds['cloud_emissivity'].values
ch07_wat_cld_trn_ems = ch07_ems_wat_ds['cloud_transmission'].values

dims_n_sat_zen = 45
dims_n_sol_zen = 45
dims_n_rel_azi = 45
dims_n_cod = 29
dims_n_cps = 9

ch_phase_cld_alb = Dict.empty(UniTuple(i1, 2), f4[:, :, :])
ch_phase_cld_alb.update({
    (3, 1): ch03_ice_cld_alb,
    (3, 0): ch03_wat_cld_alb,
    (5, 1): ch05_ice_cld_alb,
    (5, 0): ch05_wat_cld_alb,
    (6, 1): ch06_ice_cld_alb,
    (6, 0): ch06_wat_cld_alb,
    (7, 1): ch07_ice_cld_alb,
    (7, 0): ch07_wat_cld_alb,
})
ch_phase_cld_trn = Dict.empty(UniTuple(i1, 2), f4[:, :, :])
ch_phase_cld_trn.update({
    (3, 1): ch03_ice_cld_trn,
    (3, 0): ch03_wat_cld_trn,
    (5, 1): ch05_ice_cld_trn,
    (5, 0): ch05_wat_cld_trn,
    (6, 1): ch06_ice_cld_trn,
    (6, 0): ch06_wat_cld_trn,
    (7, 1): ch07_ice_cld_trn,
    (7, 0): ch07_wat_cld_trn,
})
ch_phase_cld_sph_alb = Dict.empty(UniTuple(i1, 2), f4[:, :])
ch_phase_cld_sph_alb.update({
    (3, 1): ch03_ice_cld_sph_alb,
    (3, 0): ch03_wat_cld_sph_alb,
    (5, 1): ch05_ice_cld_sph_alb,
    (5, 0): ch05_wat_cld_sph_alb,
    (6, 1): ch06_ice_cld_sph_alb,
    (6, 0): ch06_wat_cld_sph_alb,
    (7, 1): ch07_ice_cld_sph_alb,
    (7, 0): ch07_wat_cld_sph_alb,
})
ch_phase_cld_refl = Dict.empty(UniTuple(i1, 2), f4[:, :, :, :, :])
ch_phase_cld_refl.update({
    (3, 1): ch03_ice_cld_refl,
    (3, 0): ch03_wat_cld_refl,
    (5, 1): ch05_ice_cld_refl,
    (5, 0): ch05_wat_cld_refl,
    (6, 1): ch06_ice_cld_refl,
    (6, 0): ch06_wat_cld_refl,
    (7, 1): ch07_ice_cld_refl,
    (7, 0): ch07_wat_cld_refl,
})

phase_cld_ems = Dict.empty(i1, f4[:, :, :])
phase_cld_ems.update({
    1: ch07_ice_cld_ems,
    0: ch07_wat_cld_ems,
})
phase_cld_trn_ems = Dict.empty(i1, f4[:, :, :])
phase_cld_trn_ems.update({
    1: ch07_ice_cld_trn_ems,
    0: ch07_wat_cld_trn_ems,
})

# - parameter for kernel computation
ref_diff = 0.2
cod_diff = 0.1


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_get_cld_refl(
        idx_chn, idx_phase, pos_sol, pos_sat, pos_azi,
        wgt_cod, pos_cod, wgt_cps, pos_cps,
        ch_phase_cld_refl,
):
    cld_refl = ch_phase_cld_refl[idx_chn, idx_phase]
    rfl_cld_2x2 = cld_refl[pos_azi, pos_sat, pos_sol, pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    out_refl, out_d_refl_d_cps, out_d_refl_d_cod = interpolate_2d(
        rfl_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )

    return out_refl, out_d_refl_d_cps, out_d_refl_d_cod,


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_get_cld_trn(
        idx_chn, idx_phase, pos_sol, pos_sat,
        wgt_cod, pos_cod, wgt_cps, pos_cps,
        ch_phase_cld_trn,
):
    cld_trn = ch_phase_cld_trn[idx_chn, idx_phase]
    trn_sol_cld_2x2 = cld_trn[pos_sol, pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    trn_sat_cld_2x2 = cld_trn[pos_sat, pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    out_trn_sol, out_d_trans_sol_d_cps, out_d_trans_sol_d_cod = interpolate_2d(
        trn_sol_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )
    out_trn_sat, out_d_trans_sat_d_cod, out_d_trans_sat_d_cps = interpolate_2d(
        trn_sat_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )

    return (
        out_trn_sol, out_d_trans_sol_d_cps, out_d_trans_sol_d_cod,
        out_trn_sat, out_d_trans_sat_d_cod, out_d_trans_sat_d_cps,
    )


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_get_cld_sph_alb(
        idx_chn, idx_phase,
        wgt_cod, pos_cod, wgt_cps, pos_cps,
        ch_phase_cld_sph_alb,
):
    cld_sph_alb = ch_phase_cld_sph_alb[idx_chn, idx_phase]
    alb_sph_cld_2x2 = cld_sph_alb[pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    out_alb_sph, out_d_sph_alb_d_cod, out_d_sph_alb_d_cps = interpolate_2d(
        alb_sph_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )

    return out_alb_sph, out_d_sph_alb_d_cod, out_d_sph_alb_d_cps,


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_get_cld_ems(
        idx_phase, pos_sat,
        wgt_cod, pos_cod, wgt_cps, pos_cps,
        phase_cld_ems,
):
    cld_ems = phase_cld_ems[idx_phase]
    ems_cld_2x2 = cld_ems[pos_sat, pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    out_ems, out_d_ems_d_cps, out_d_ems_d_cod = interpolate_2d(
        ems_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )

    return out_ems, out_d_ems_d_cps, out_d_ems_d_cod,


@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_get_cld_trn_ems(
        idx_phase, pos_sat,
        wgt_cod, pos_cod, wgt_cps, pos_cps,
        phase_cld_trn_ems,
):
    cld_trn_ems = phase_cld_trn_ems[idx_phase]
    trn_ems_cld_2x2 = cld_trn_ems[pos_sat, pos_cod:pos_cod + 2, pos_cps:pos_cps + 2]
    out_trn_ems, out_d_trn_ems_d_cps, out_d_trn_ems_d_cod = interpolate_2d(
        trn_ems_cld_2x2, wgt_cps, wgt_cod, ref_diff, cod_diff
    )

    return out_trn_ems, out_d_trn_ems_d_cps, out_d_trn_ems_d_cod


# 波段 7
@njit(nogil=True, error_model='numpy', boundscheck=True)
def lut_thick_cloud_rfl(
        pos_sol, pos_sat, pos_azi, idx_chn, idx_phase,
        ch_phase_cld_refl, phase_cld_ems, has_ems,
):
    cld_refl = ch_phase_cld_refl[idx_chn, idx_phase]
    rfl = cld_refl[pos_azi, pos_sat, pos_sol, 28, :]

    if has_ems:
        cld_ems = phase_cld_ems[idx_phase]
        ems = cld_ems[pos_sat, 28, :]
    else:
        ems = np.full(9, nan, 'f4')

    return rfl, ems
