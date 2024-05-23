# -*- coding: utf-8 -*-
# vim:fileencoding=utf-8:et
#
# Copyright (c) 2023 Stefan Bender
#
# This module is part of pymipas_noxy.
# pymipas_noxy is free software: you can redistribute it or modify
# it under the terms of the GNU General Public License as published
# by the Free Software Foundation, version 2.
# See accompanying LICENSE file or http://www.gnu.org/licenses/gpl-2.0.html.
"""MIPAS EPP-NOy calculator command line interface
"""

# %%
import logging

# %%
import numpy as np
import pandas as pd
import xarray as xr

from scipy.interpolate import interp1d

# %%
import click
import toml

# %%
from .noy_ubc import (
    Green_filter_F16,
    Nm_func_F16,
    days_since_Jul01,
    noy_ese,
    time_matrix,
)

# %%
logging.basicConfig(
    level=logging.INFO,
    format=
        "[%(levelname)-8s] (%(asctime)s) "
        "%(filename)s:%(lineno)d:%(funcName)s() %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S %z",
)

# %%
logger = logging.getLogger()

# Ap window width (days)
DW = 250

# constants
R_EARTH = 6356e5  # Earth radius (cm)
N_AVO = 6.02214129e+23  # Avogadro constant (mol-1)

# references pressure levels [hPa]
PREF = [1.00, 0.70, 0.50, 0.30, 0.20, 0.15, 0.10, 0.07, 0.05, 0.03, 0.02, 0.01]
PREF = np.array(PREF)

# references latitude centers
LREF = [-85.,-75.,-65.,-55.,-45.,-35.,-25.,-15., -5., 5., 15., 25., 35., 45., 55., 65., 75., 85.]
LREF = np.array(LREF)

# references latitude centers per hemisphere
LREFS = [-85.,-75.,-65.,-55.,-45.,-35.,-25.,-15., -5.]
LREFN = [  5., 15., 25., 35., 45., 55., 65., 75., 85.]
LREFS = np.array(LREFS)
LREFN = np.array(LREFN)

# start days of ESE (julian day)
DESE = [
    "1985-01-23 12:00:00", "1987-02-08 12:00:00", "2004-01-12 12:00:00",
    "2006-01-25 12:00:00", "2009-02-06 12:00:00", "2013-01-22 12:00:00",
    "2019-01-15 12:00:00",
]

# mean transport times (NH and SH)
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8N1i2b.log
TXNR = [55.9,  48.8,  43.1,  32.0,  32.1,  32.8,  32.9,  33.6,  33. ,  29.3,  27.6,  26.1]
# V8 update 1
# TXSR = [41.6,  32.9,  26.6,  20.3,  18.7,  18. ,  17.3,  15.8,  13. , 10.7,  11.1,  11.6]
# V8 update 2
# TXSR = [32.69, 27.68, 23.65, 21.12, 20.6 , 20.4 , 17.85, 14.63, 12.41, 10.59, 10.05, 10.07]
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8S1i2b.log
TXSR = [42.6,  35.5,  29.8, 23.7,  22. ,  21.2,  20.6,  19.1,  16.3,  13.6,  14.0,  14.6]
TXNR = np.array(TXNR)
TXSR = np.array(TXSR)

# maximum density occurence times (NH and SH)
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8N1i2b.log
TMNR = [180.4, 179.7, 185.8, 189.1, 189.1, 189.6, 189.7, 190.2, 189.8, 187.1, 185.8, 184.7 ]
# V8 update 1
# TMSR = [198.46, 191.88, 187.05, 182.23, 181., 180.46, 179.94, 178.81, 176.71, 174.9, 175.19, 175.63]
# V8 update 2
# TMSR = [197.8 , 191.54, 186.84, 182.07, 181.02, 180.67, 180.76, 179.97, 177.79, 175.47, 175.8 , 176.3]
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8S1i2b.log
TMSR = [ 197.0, 191.7, 187.4, 182.8, 181.5, 180.9, 180.5, 179.4, 177.2, 175.2, 175.6, 176.0 ]
TMNR = np.array(TMNR)
TMSR = np.array(TMSR)

#  maximum flux occurence times (NH and SH)
TFNR = [158.7, 155.4, 152.8, 150.0, 148.5, 147.8, 147.1, 146.8, 146.6, 146.4, 146.0, 145.8]
TFSR = [185.6, 180.6, 176.6, 171.8, 168.9, 167.2, 165.4, 164.0, 163.0, 161.5, 160.1, 159.8]
TFNR = np.array(TFNR)
TFSR = np.array(TFSR)

# maximum densities (NH and SH)
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8N1i2b.log
NMNR = [0.00096, 0.00080, 0.00072, 0.00074, 0.00081, 0.00085, 0.00092, 0.00098, 0.00106, 0.00115, 0.00118, 0.00130]
# V8 update 1
# NMSR = [0.0042871, 0.0034717, 0.0027649, 0.0019782, 0.0016212, 0.0014701, 0.0013533, 0.0013022, 0.0012917, 0.0013359, 0.0013005, 0.0014607]
# V8 update 2
# NMSR = [0.0043643, 0.0034964, 0.0027627, 0.0019783, 0.0016206, 0.0014618, 0.0013401, 0.0012944, 0.0012847, 0.0013343, 0.0012995, 0.00146]
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8S1i2b.log
NMSR = [0.00484, 0.00379, 0.00290, 0.00203, 0.00166, 0.00150, 0.00136, 0.00130, 0.00129, 0.00134, 0.00130, 0.00146]
NMNR = np.array(NMNR)
NMSR = np.array(NMSR)

# maximum fluxes (NH and SH)
NFNR = [0.00048, 0.00054, 0.00057, 0.00058, 0.00059, 0.00060, 0.00061, 0.00062, 0.00062, 0.00063, 0.00064, 0.00070]
NFSR = [0.00156, 0.00163, 0.00169, 0.00178, 0.00186, 0.00191, 0.00200, 0.00207, 0.00214, 0.00223, 0.00227, 0.00240]
NFNR = np.array(NFNR)
NFSR = np.array(NFSR)

#  width parameter densities (NH and SH)
# days^-1
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8N1i2b.log
WMNR = [0.0604, 0.0499, 0.0369, 0.0297, 0.0309, 0.0334, 0.037 , 0.0399, 0.0416, 0.0426, 0.0431, 0.042 ]
# V8 update 1
# WMSR = [0.0656, 0.0631, 0.0584, 0.0488, 0.0443, 0.0431, 0.0424, 0.042 , 0.041 , 0.0394, 0.0384, 0.037]
# V8 update 2
# WMSR = [0.0665, 0.0634, 0.0583, 0.0488, 0.0443, 0.0429, 0.0422, 0.0422, 0.0414, 0.0398, 0.0388, 0.0375]
# V8 update 3 (ubc1) ubc_fit_jax_numpyro_v8S1i2b.log
WMSR = [0.0731, 0.0688, 0.0617, 0.0507, 0.0459, 0.0443, 0.0431, 0.0424, 0.0413, 0.0397, 0.0387, 0.0373]
WMNR = np.array(WMNR)
WMSR = np.array(WMSR)

#  width parameter fluxes (NH and SH)
WFNR = [0.1032, 0.0994, 0.0939, 0.0838, 0.0759, 0.0711, 0.0660, 0.0633, 0.0622, 0.0622, 0.0625, 0.0627]
WFSR = [0.0754, 0.0751, 0.0751, 0.0752, 0.0753, 0.0754, 0.0754, 0.0753, 0.0752, 0.0751, 0.0752, 0.0754]
WFNR = np.array(WFNR)
WFSR = np.array(WFSR)

#  latitude distribution of densities
#lat distribution SH (85S-5S)
LDS = np.zeros((12, 9), dtype=float)
# # V8 update 1
# LDS[:, 0] = [ 0.288, 0.29 , 0.293, 0.299, 0.309, 0.322, 0.338, 0.357, 0.379, 0.399, 0.408, 0.407]
# LDS[:, 1] = [ 0.258, 0.251, 0.248, 0.25 , 0.256, 0.266, 0.28 , 0.293, 0.305, 0.312, 0.313, 0.309]
# LDS[:, 2] = [ 0.215, 0.207, 0.2  , 0.196, 0.193, 0.192, 0.19 , 0.187, 0.182, 0.173, 0.169, 0.169]
# LDS[:, 3] = [ 0.16 , 0.161, 0.158, 0.153, 0.144, 0.134, 0.12 , 0.104, 0.086, 0.074, 0.069, 0.07 ]
# LDS[:, 4] = [ 0.064, 0.075, 0.082, 0.083, 0.079, 0.069, 0.058, 0.046, 0.036, 0.03 , 0.028, 0.029]
# LDS[:, 5] = [ 0.013, 0.015, 0.017, 0.018, 0.018, 0.016, 0.013, 0.011, 0.01 , 0.009, 0.01 , 0.011]
# LDS[:, 6] = [ 0.001, 0.001, 0.002, 0.002, 0.002, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005]
# LDS[:, 7] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
# LDS[:, 8] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
# V8 update 2
LDS[:, 0] = [ 0.288, 0.289, 0.292, 0.299, 0.309, 0.321, 0.335, 0.354, 0.375, 0.392, 0.402, 0.405]
LDS[:, 1] = [ 0.258, 0.251, 0.248, 0.249, 0.255, 0.265, 0.279, 0.293, 0.306, 0.313, 0.315, 0.311]
LDS[:, 2] = [ 0.216, 0.207, 0.200, 0.196, 0.193, 0.193, 0.192, 0.189, 0.184, 0.178, 0.174, 0.171]
LDS[:, 3] = [ 0.159, 0.160, 0.158, 0.153, 0.145, 0.135, 0.121, 0.104, 0.087, 0.074, 0.068, 0.069]
LDS[:, 4] = [ 0.065, 0.075, 0.083, 0.084, 0.080, 0.070, 0.058, 0.046, 0.036, 0.030, 0.027, 0.028]
LDS[:, 5] = [ 0.013, 0.015, 0.017, 0.018, 0.017, 0.015, 0.013, 0.011, 0.010, 0.009, 0.010, 0.011]
LDS[:, 6] = [ 0.001, 0.001, 0.002, 0.002, 0.001, 0.001, 0.001, 0.002, 0.002, 0.003, 0.004, 0.005]
LDS[:, 7] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDS[:, 8] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDS[np.where(LDS < 1.e-20)]=1e-20

# lat distribution NH (5N-85N)
LDN=np.zeros((12, 9), dtype=float)
# #  V8 update 1
# LDN[:, 0] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
# LDN[:, 1] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
# LDN[:, 2] = [ 0.002, 0.003, 0.003, 0.004, 0.005, 0.007, 0.008, 0.009, 0.009, 0.009, 0.01 , 0.01 ]
# LDN[:, 3] = [ 0.009, 0.01 , 0.012, 0.015, 0.019, 0.022, 0.024, 0.024, 0.022, 0.021, 0.021, 0.022]
# LDN[:, 4] = [ 0.035, 0.037, 0.045, 0.055, 0.066, 0.069, 0.066, 0.062, 0.055, 0.051, 0.05 , 0.052]
# LDN[:, 5] = [ 0.102, 0.096, 0.105, 0.122, 0.144, 0.147, 0.143, 0.134, 0.121, 0.111, 0.106, 0.11 ]
# LDN[:, 6] = [ 0.19 , 0.181, 0.186, 0.202, 0.228, 0.238, 0.237, 0.234, 0.225, 0.217, 0.211, 0.212]
# LDN[:, 7] = [ 0.295, 0.298, 0.287, 0.272, 0.257, 0.258, 0.266, 0.278, 0.291, 0.296, 0.295, 0.287]
# LDN[:, 8] = [ 0.368, 0.375, 0.361, 0.33 , 0.282, 0.259, 0.256, 0.26 , 0.277, 0.295, 0.308, 0.307]
#  V8 update 2
LDN[:, 0] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDN[:, 1] = [ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDN[:, 2] = [ 0.002, 0.003, 0.004, 0.004, 0.004, 0.006, 0.008, 0.009, 0.009, 0.009, 0.009, 0.010]
LDN[:, 3] = [ 0.010, 0.013, 0.014, 0.016, 0.018, 0.022, 0.024, 0.024, 0.022, 0.020, 0.020, 0.022]
LDN[:, 4] = [ 0.038, 0.047, 0.057, 0.066, 0.070, 0.070, 0.068, 0.062, 0.056, 0.051, 0.049, 0.052]
LDN[:, 5] = [ 0.112, 0.118, 0.131, 0.144, 0.151, 0.149, 0.143, 0.134, 0.121, 0.110, 0.105, 0.108]
LDN[:, 6] = [ 0.206, 0.206, 0.213, 0.225, 0.236, 0.239, 0.237, 0.234, 0.225, 0.217, 0.212, 0.212]
LDN[:, 7] = [ 0.294, 0.285, 0.273, 0.262, 0.256, 0.258, 0.266, 0.278, 0.292, 0.300, 0.300, 0.290]
LDN[:, 8] = [ 0.338, 0.329, 0.308, 0.283, 0.264, 0.256, 0.254, 0.259, 0.275, 0.293, 0.305, 0.307]
LDN[np.where(LDN < 1.e-20)]=1e-20

# lat distribution NH ESE (5N-85N)
LDE=np.zeros((12, 9), dtype=float)
LDE[:, 0]=[ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDE[:, 1]=[ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDE[:, 2]=[ 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
LDE[:, 3]=[ 0.001, 0.001, 0.002, 0.002, 0.002, 0.002, 0.002, 0.002, 0.001, 0.001, 0.001, 0.001]
LDE[:, 4]=[ 0.009, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.008, 0.007, 0.007, 0.006, 0.006]
LDE[:, 5]=[ 0.044, 0.042, 0.039, 0.036, 0.034, 0.035, 0.037, 0.040, 0.042, 0.042, 0.042, 0.041]
LDE[:, 6]=[ 0.146, 0.147, 0.148, 0.139, 0.135, 0.132, 0.137, 0.145, 0.150, 0.153, 0.153, 0.150]
LDE[:, 7]=[ 0.333, 0.331, 0.330, 0.334, 0.339, 0.341, 0.344, 0.343, 0.341, 0.336, 0.332, 0.330]
LDE[:, 8]=[ 0.467, 0.471, 0.472, 0.481, 0.481, 0.482, 0.471, 0.462, 0.459, 0.462, 0.467, 0.473]
LDE[np.where(LDE < 1.e-20)]=1e-20

#  background densities
#  background annual means
AMR=np.zeros((12, 18), dtype=float)
# V8 update
AMR[:, 0] = [1.54e+08, 8.91e+07, 5.49e+07, 2.63e+07, 1.53e+07, 1.04e+07, 5.88e+06, 3.49e+06, 2.09e+06, 1.09e+06, 6.98e+05, 2.45e+05]
AMR[:, 1] = [1.62e+08, 9.42e+07, 5.74e+07, 2.78e+07, 1.59e+07, 1.05e+07, 5.83e+06, 3.42e+06, 2.04e+06, 1.02e+06, 6.26e+05, 1.86e+05]
AMR[:, 2] = [1.71e+08, 9.89e+07, 6.09e+07, 2.95e+07, 1.65e+07, 1.08e+07, 5.84e+06, 3.35e+06, 2.02e+06, 9.87e+05, 5.67e+05, 1.33e+05]
AMR[:, 3] = [1.97e+08, 1.15e+08, 7.01e+07, 3.30e+07, 1.82e+07, 1.17e+07, 6.12e+06, 3.50e+06, 2.14e+06, 1.00e+06, 5.05e+05, 1.08e+05]
AMR[:, 4] = [2.35e+08, 1.38e+08, 8.34e+07, 3.84e+07, 2.09e+07, 1.32e+07, 6.75e+06, 3.89e+06, 2.44e+06, 1.06e+06, 4.81e+05, 1.25e+05]
AMR[:, 5] = [2.61e+08, 1.51e+08, 8.94e+07, 4.02e+07, 2.16e+07, 1.34e+07, 6.89e+06, 4.09e+06, 2.59e+06, 1.10e+06, 4.79e+05, 1.45e+05]
AMR[:, 6] = [2.75e+08, 1.55e+08, 8.84e+07, 3.80e+07, 1.98e+07, 1.22e+07, 6.36e+06, 3.93e+06, 2.55e+06, 1.07e+06, 4.57e+05, 1.46e+05]
AMR[:, 7] = [2.74e+08, 1.49e+08, 8.29e+07, 3.51e+07, 1.81e+07, 1.10e+07, 5.69e+06, 3.52e+06, 2.33e+06, 1.03e+06, 4.20e+05, 1.28e+05]
AMR[:, 8] = [2.68e+08, 1.40e+08, 7.65e+07, 3.30e+07, 1.72e+07, 1.05e+07, 5.26e+06, 3.15e+06, 2.09e+06, 9.52e+05, 3.79e+05, 1.07e+05]
AMR[:, 9] = [2.74e+08, 1.41e+08, 7.66e+07, 3.29e+07, 1.72e+07, 1.04e+07, 5.08e+06, 2.89e+06, 1.87e+06, 9.33e+05, 4.24e+05, 1.31e+05]
AMR[:,10] = [2.90e+08, 1.55e+08, 8.44e+07, 3.49e+07, 1.79e+07, 1.08e+07, 5.26e+06, 3.02e+06, 1.93e+06, 9.67e+05, 4.59e+05, 1.62e+05]
AMR[:,11] = [2.97e+08, 1.66e+08, 9.26e+07, 3.85e+07, 1.99e+07, 1.20e+07, 5.88e+06, 3.35e+06, 2.09e+06, 1.00e+06, 4.86e+05, 1.85e+05]
AMR[:,12] = [2.74e+08, 1.57e+08, 9.04e+07, 3.92e+07, 2.05e+07, 1.25e+07, 6.17e+06, 3.46e+06, 2.12e+06, 1.00e+06, 4.95e+05, 1.82e+05]
AMR[:,13] = [2.31e+08, 1.35e+08, 8.02e+07, 3.59e+07, 1.87e+07, 1.15e+07, 5.77e+06, 3.22e+06, 1.96e+06, 8.95e+05, 4.34e+05, 1.47e+05]
AMR[:,14] = [1.93e+08, 1.13e+08, 6.77e+07, 3.05e+07, 1.61e+07, 1.01e+07, 5.16e+06, 2.90e+06, 1.72e+06, 6.93e+05, 3.06e+05, 9.02e+04]
AMR[:,15] = [1.74e+08, 9.92e+07, 5.93e+07, 2.73e+07, 1.46e+07, 9.26e+06, 4.80e+06, 2.69e+06, 1.52e+06, 4.72e+05, 1.94e+05, 5.17e+04]
AMR[:,16] = [1.65e+08, 9.19e+07, 5.40e+07, 2.49e+07, 1.35e+07, 8.70e+06, 4.60e+06, 2.57e+06, 1.33e+06, 3.50e+05, 1.39e+05, 3.54e+04]
AMR[:,17] = [1.64e+08, 8.99e+07, 5.27e+07, 2.43e+07, 1.34e+07, 8.68e+06, 4.59e+06, 2.52e+06, 1.24e+06, 2.96e+05, 1.14e+05, 2.75e+04]

#  background annual amplitude (relative to mean)
A1R=np.zeros((12, 18), dtype=float)
# V8 update
A1R[:, 0] = [0.611, 0.658, 0.718, 0.803, 0.877, 0.927, 0.979, 1.015, 1.050, 1.191, 1.347, 1.569]
A1R[:, 1] = [0.540, 0.586, 0.645, 0.729, 0.798, 0.839, 0.873, 0.893, 0.914, 1.052, 1.234, 1.419]
A1R[:, 2] = [0.392, 0.432, 0.488, 0.587, 0.649, 0.678, 0.694, 0.697, 0.728, 0.867, 1.059, 1.009]
A1R[:, 3] = [0.213, 0.262, 0.314, 0.421, 0.484, 0.513, 0.536, 0.555, 0.611, 0.730, 0.806, 0.370]
A1R[:, 4] = [0.191, 0.211, 0.245, 0.312, 0.350, 0.369, 0.395, 0.452, 0.554, 0.611, 0.536, 0.202]
A1R[:, 5] = [0.242, 0.244, 0.250, 0.250, 0.232, 0.218, 0.223, 0.314, 0.446, 0.502, 0.356, 0.169]
A1R[:, 6] = [0.266, 0.266, 0.268, 0.254, 0.235, 0.218, 0.226, 0.305, 0.388, 0.409, 0.277, 0.119]
A1R[:, 7] = [0.225, 0.202, 0.184, 0.156, 0.138, 0.136, 0.184, 0.292, 0.366, 0.336, 0.217, 0.109]
A1R[:, 8] = [0.127, 0.105, 0.106, 0.118, 0.124, 0.113, 0.082, 0.200, 0.280, 0.280, 0.234, 0.207]
A1R[:, 9] = [0.089, 0.159, 0.218, 0.255, 0.245, 0.202, 0.093, 0.037, 0.070, 0.374, 0.620, 0.703]
A1R[:,10] = [0.176, 0.250, 0.336, 0.402, 0.357, 0.284, 0.141, 0.057, 0.125, 0.443, 0.627, 0.636]
A1R[:,11] = [0.203, 0.254, 0.335, 0.415, 0.366, 0.298, 0.184, 0.146, 0.231, 0.499, 0.583, 0.573]
A1R[:,12] = [0.149, 0.201, 0.273, 0.341, 0.315, 0.289, 0.240, 0.240, 0.324, 0.511, 0.500, 0.501]
A1R[:,13] = [0.142, 0.185, 0.264, 0.342, 0.338, 0.332, 0.307, 0.304, 0.373, 0.486, 0.473, 0.495]
A1R[:,14] = [0.120, 0.157, 0.260, 0.358, 0.382, 0.387, 0.373, 0.378, 0.438, 0.518, 0.571, 0.658]
A1R[:,15] = [0.085, 0.155, 0.260, 0.399, 0.447, 0.466, 0.460, 0.470, 0.502, 0.636, 0.775, 0.846]
A1R[:,16] = [0.226, 0.276, 0.355, 0.476, 0.527, 0.558, 0.573, 0.584, 0.519, 0.699, 0.914, 0.966]
A1R[:,17] = [0.322, 0.389, 0.459, 0.560, 0.601, 0.634, 0.650, 0.645, 0.534, 0.691, 0.950, 1.010]


#  background semi-annual amplitude (relative to mean)
A2R=np.zeros((12, 18), dtype=float)
# V8 update
A2R[:, 0] = [0.040, 0.111, 0.172, 0.211, 0.294, 0.360, 0.420, 0.434, 0.471, 0.600, 0.743, 1.123]
A2R[:, 1] = [0.037, 0.069, 0.124, 0.164, 0.248, 0.320, 0.387, 0.404, 0.448, 0.615, 0.793, 1.146]
A2R[:, 2] = [0.112, 0.099, 0.117, 0.135, 0.192, 0.253, 0.326, 0.357, 0.421, 0.627, 0.827, 0.972]
A2R[:, 3] = [0.169, 0.168, 0.162, 0.138, 0.170, 0.220, 0.300, 0.345, 0.415, 0.609, 0.722, 0.561]
A2R[:, 4] = [0.159, 0.185, 0.197, 0.183, 0.202, 0.246, 0.318, 0.378, 0.468, 0.603, 0.616, 0.452]
A2R[:, 5] = [0.172, 0.210, 0.227, 0.242, 0.261, 0.283, 0.331, 0.407, 0.501, 0.607, 0.587, 0.458]
A2R[:, 6] = [0.164, 0.189, 0.207, 0.251, 0.249, 0.249, 0.288, 0.364, 0.440, 0.513, 0.496, 0.396]
A2R[:, 7] = [0.133, 0.141, 0.155, 0.205, 0.218, 0.233, 0.298, 0.370, 0.397, 0.360, 0.305, 0.260]
A2R[:, 8] = [0.116, 0.112, 0.122, 0.162, 0.188, 0.225, 0.325, 0.409, 0.417, 0.304, 0.203, 0.166]
A2R[:, 9] = [0.125, 0.103, 0.099, 0.153, 0.198, 0.233, 0.304, 0.333, 0.317, 0.328, 0.399, 0.433]
A2R[:,10] = [0.114, 0.105, 0.114, 0.210, 0.258, 0.280, 0.316, 0.325, 0.324, 0.382, 0.427, 0.418]
A2R[:,11] = [0.110, 0.111, 0.131, 0.232, 0.275, 0.290, 0.317, 0.343, 0.384, 0.517, 0.553, 0.500]
A2R[:,12] = [0.128, 0.145, 0.163, 0.239, 0.309, 0.355, 0.409, 0.436, 0.494, 0.653, 0.685, 0.634]
A2R[:,13] = [0.135, 0.160, 0.179, 0.257, 0.355, 0.409, 0.458, 0.485, 0.545, 0.675, 0.690, 0.659]
A2R[:,14] = [0.088, 0.119, 0.169, 0.290, 0.398, 0.451, 0.500, 0.527, 0.578, 0.640, 0.637, 0.648]
A2R[:,15] = [0.107, 0.141, 0.202, 0.344, 0.464, 0.529, 0.578, 0.596, 0.628, 0.556, 0.531, 0.605]
A2R[:,16] = [0.173, 0.226, 0.284, 0.416, 0.533, 0.614, 0.663, 0.676, 0.648, 0.460, 0.405, 0.420]
A2R[:,17] = [0.225, 0.291, 0.337, 0.468, 0.574, 0.657, 0.704, 0.702, 0.659, 0.473, 0.447, 0.427]

#  background ter-annual amplitude (relative to mean)
A3R=np.zeros((12, 18), dtype=float)
# V8 update
A3R[:, 0] = [0.105, 0.084, 0.099, 0.153, 0.133, 0.113, 0.114, 0.136, 0.147, 0.101, 0.121, 0.576]
A3R[:, 1] = [0.076, 0.035, 0.052, 0.128, 0.124, 0.101, 0.090, 0.101, 0.095, 0.053, 0.172, 0.686]
A3R[:, 2] = [0.060, 0.027, 0.018, 0.083, 0.108, 0.104, 0.102, 0.111, 0.074, 0.083, 0.275, 0.633]
A3R[:, 3] = [0.083, 0.082, 0.092, 0.093, 0.123, 0.128, 0.124, 0.127, 0.087, 0.088, 0.214, 0.202]
A3R[:, 4] = [0.079, 0.069, 0.070, 0.076, 0.110, 0.126, 0.135, 0.118, 0.063, 0.070, 0.121, 0.093]
A3R[:, 5] = [0.045, 0.047, 0.036, 0.017, 0.050, 0.088, 0.118, 0.087, 0.026, 0.061, 0.058, 0.017]
A3R[:, 6] = [0.022, 0.036, 0.016, 0.030, 0.012, 0.019, 0.045, 0.038, 0.030, 0.061, 0.038, 0.044]
A3R[:, 7] = [0.006, 0.015, 0.024, 0.003, 0.013, 0.030, 0.053, 0.081, 0.089, 0.043, 0.063, 0.096]
A3R[:, 8] = [0.048, 0.031, 0.044, 0.024, 0.028, 0.038, 0.065, 0.099, 0.105, 0.061, 0.099, 0.099]
A3R[:, 9] = [0.061, 0.027, 0.028, 0.026, 0.047, 0.051, 0.036, 0.038, 0.071, 0.157, 0.209, 0.231]
A3R[:,10] = [0.063, 0.046, 0.036, 0.022, 0.045, 0.049, 0.044, 0.053, 0.080, 0.174, 0.209, 0.177]
A3R[:,11] = [0.050, 0.039, 0.033, 0.045, 0.082, 0.084, 0.074, 0.066, 0.076, 0.161, 0.196, 0.160]
A3R[:,12] = [0.043, 0.045, 0.042, 0.046, 0.075, 0.082, 0.080, 0.062, 0.069, 0.174, 0.199, 0.184]
A3R[:,13] = [0.052, 0.059, 0.062, 0.058, 0.073, 0.085, 0.092, 0.079, 0.084, 0.166, 0.177, 0.175]
A3R[:,14] = [0.084, 0.096, 0.109, 0.113, 0.109, 0.109, 0.122, 0.125, 0.140, 0.250, 0.286, 0.303]
A3R[:,15] = [0.085, 0.110, 0.140, 0.132, 0.096, 0.075, 0.089, 0.110, 0.163, 0.452, 0.498, 0.530]
A3R[:,16] = [0.069, 0.101, 0.121, 0.121, 0.078, 0.036, 0.056, 0.078, 0.185, 0.551, 0.614, 0.653]
A3R[:,17] = [0.077, 0.102, 0.130, 0.143, 0.095, 0.049, 0.052, 0.089, 0.233, 0.653, 0.729, 0.789]

#  background annual phase
P1R=np.zeros((12, 18), dtype=float)
# V8 update
P1R[:, 0] = [ 1.366,  1.400,  1.407,  1.511,  1.529,  1.538,  1.536,  1.535,  1.500,  1.426,  1.355,  1.343]
P1R[:, 1] = [ 1.379,  1.374,  1.420,  1.483,  1.502,  1.523,  1.543,  1.559,  1.532,  1.451,  1.378,  1.374]
P1R[:, 2] = [ 1.441,  1.436,  1.417,  1.462,  1.472,  1.484,  1.504,  1.525,  1.515,  1.443,  1.383,  1.432]
P1R[:, 3] = [ 1.076,  1.204,  1.226,  1.348,  1.371,  1.392,  1.417,  1.434,  1.432,  1.400,  1.365,  1.279]
P1R[:, 4] = [ 0.363,  0.650,  0.850,  1.175,  1.294,  1.365,  1.410,  1.425,  1.417,  1.404,  1.367,  0.869]
P1R[:, 5] = [ 0.449,  0.715,  0.915,  1.182,  1.417,  1.574,  1.671,  1.627,  1.565,  1.524,  1.523,  1.013]
P1R[:, 6] = [ 0.824,  1.029,  1.201,  1.393,  1.660,  1.839,  1.942,  1.849,  1.773,  1.739,  1.806,  1.549]
P1R[:, 7] = [ 1.223,  1.395,  1.563,  1.705,  1.957,  2.124,  1.989,  1.792,  1.780,  1.957,  2.273,  2.912]
P1R[:, 8] = [ 1.845,  2.422,  3.039, -2.773, -2.655, -2.697,  2.579,  1.810,  1.794,  2.100,  2.407,  2.680]
P1R[:, 9] = [-2.657, -2.408, -2.304, -2.256, -2.170, -2.106, -2.131,  2.770, -2.701, -1.921, -1.852, -1.907]
P1R[:,10] = [-1.930, -1.957, -1.981, -2.035, -1.966, -1.905, -1.803, -1.932, -2.088, -1.934, -1.898, -1.980]
P1R[:,11] = [-1.762, -1.793, -1.849, -1.923, -1.871, -1.850, -1.916, -2.140, -2.154, -1.984, -1.947, -1.982]
P1R[:,12] = [-2.139, -2.084, -2.074, -2.063, -2.091, -2.155, -2.268, -2.339, -2.258, -2.110, -2.127, -2.120]
P1R[:,13] = [-3.099, -2.739, -2.533, -2.366, -2.398, -2.434, -2.449, -2.494, -2.461, -2.433, -2.547, -2.544]
P1R[:,14] = [ 2.768, -2.905, -2.660, -2.517, -2.523, -2.536, -2.515, -2.550, -2.594, -2.897, -3.115, -3.060]
P1R[:,15] = [-1.961, -2.109, -2.269, -2.239, -2.257, -2.256, -2.250, -2.310, -2.495,  2.789,  2.615,  2.836]
P1R[:,16] = [-1.537, -1.723, -1.917, -1.943, -1.978, -1.978, -1.984, -2.042, -2.346,  2.498,  2.341,  2.526]
P1R[:,17] = [-1.568, -1.697, -1.782, -1.775, -1.844, -1.861, -1.885, -1.939, -2.243,  2.396,  2.231,  2.348]

#  background semi-annual phase
P2R=np.zeros((12, 18), dtype=float)
# V8 update
P2R[:, 0] = [ 1.368,  0.696,  0.522,  0.845,  1.039,  1.101,  1.112,  1.116,  1.039,  1.032,  0.978,  1.036]
P2R[:, 1] = [-2.411, -0.356,  0.125,  0.546,  0.880,  0.990,  1.058,  1.111,  1.066,  1.070,  1.032,  1.077]
P2R[:, 2] = [-2.374, -1.550, -0.887, -0.020,  0.508,  0.682,  0.807,  0.886,  0.950,  1.032,  1.036,  1.097]
P2R[:, 3] = [-2.194, -1.831, -1.393, -0.607,  0.030,  0.332,  0.572,  0.701,  0.837,  0.987,  1.003,  0.844]
P2R[:, 4] = [-1.760, -1.362, -1.034, -0.409,  0.090,  0.397,  0.654,  0.803,  0.935,  1.050,  1.052,  0.849]
P2R[:, 5] = [-1.202, -0.822, -0.464,  0.081,  0.411,  0.616,  0.830,  0.967,  1.069,  1.149,  1.166,  1.059]
P2R[:, 6] = [-0.724, -0.410, -0.032,  0.435,  0.642,  0.771,  0.923,  1.035,  1.147,  1.245,  1.245,  1.185]
P2R[:, 7] = [-0.139,  0.052,  0.296,  0.760,  0.978,  1.024,  1.002,  1.019,  1.099,  1.269,  1.319,  1.292]
P2R[:, 8] = [ 0.675,  0.542,  0.479,  0.848,  1.059,  1.062,  0.979,  0.951,  1.000,  1.174,  1.271,  1.053]
P2R[:, 9] = [ 0.921,  0.808,  0.747,  1.057,  1.145,  1.112,  1.012,  0.952,  0.961,  1.018,  0.920,  0.816]
P2R[:,10] = [ 0.453,  0.692,  0.989,  1.131,  1.111,  1.093,  1.053,  1.049,  1.104,  1.158,  1.068,  0.941]
P2R[:,11] = [-0.132,  0.155,  0.613,  0.958,  1.021,  1.096,  1.225,  1.324,  1.364,  1.234,  1.152,  1.008]
P2R[:,12] = [-0.537, -0.314,  0.050,  0.560,  0.799,  0.958,  1.156,  1.261,  1.262,  1.144,  1.080,  0.956]
P2R[:,13] = [-0.686, -0.474, -0.140,  0.368,  0.626,  0.774,  0.945,  1.042,  1.049,  0.977,  0.921,  0.813]
P2R[:,14] = [-0.546, -0.248,  0.070,  0.395,  0.574,  0.677,  0.814,  0.887,  0.879,  0.720,  0.595,  0.381]
P2R[:,15] = [ 0.287,  0.314,  0.392,  0.614,  0.730,  0.810,  0.905,  0.936,  0.861,  0.426,  0.280, -0.032]
P2R[:,16] = [ 0.363,  0.500,  0.620,  0.821,  0.918,  0.986,  1.052,  1.069,  0.936,  0.523,  0.449,  0.035]
P2R[:,17] = [ 0.287,  0.575,  0.751,  0.951,  1.023,  1.080,  1.129,  1.141,  0.993,  0.696,  0.740,  0.518]

#  background ter-annual phase
P3R=np.zeros((12, 18), dtype=float)
# V8 update
P3R[:, 0] = [-0.945, -0.778, -1.201, -1.811, -1.842, -1.643, -1.315, -1.249, -1.027, -0.826, -0.194,  0.813]
P3R[:, 1] = [-0.891, -1.267, -1.366, -1.998, -2.037, -1.854, -1.578, -1.586, -1.273,  0.034,  0.628,  0.954]
P3R[:, 2] = [ 0.075,  1.079,  2.945, -2.446, -2.432, -2.351, -2.164, -2.081, -2.022,  1.080,  1.010,  1.246]
P3R[:, 3] = [ 0.925,  1.621,  2.014,  2.891, -3.024, -2.897, -2.643, -2.436, -2.438,  1.374,  1.264,  1.736]
P3R[:, 4] = [ 1.148,  1.638,  1.929,  2.665,  3.074, -3.101, -2.859, -2.634, -2.718,  1.737,  1.796,  2.502]
P3R[:, 5] = [ 1.866,  2.835,  2.824,  2.841, -2.930, -2.864, -2.628, -2.467, -3.004,  1.485,  1.844,  1.855]
P3R[:, 6] = [-3.026, -2.627, -2.562,  0.595,  0.225, -1.951, -1.794, -1.089, -0.023,  1.059,  0.726, -0.321]
P3R[:, 7] = [ 0.765, -2.470, -2.658,  1.828, -1.094, -1.087, -0.509,  0.123,  0.319,  0.029, -0.862, -1.003]
P3R[:, 8] = [ 1.394,  2.808, -2.828, -2.323, -1.639, -1.104, -0.250,  0.291,  0.379, -0.358, -0.800, -0.688]
P3R[:, 9] = [ 1.012,  2.021,  3.004, -2.318, -2.147, -2.080, -2.159, -2.937, -2.809, -2.543, -2.609, -2.717]
P3R[:,10] = [ 0.665,  1.372,  1.997, -2.876, -2.489, -2.500, -2.809,  3.133, -2.853, -2.572, -2.610, -2.871]
P3R[:,11] = [ 0.279,  1.049,  1.667, -3.137, -3.016, -3.084,  3.134, -3.138, -2.896, -2.600, -2.605, -2.839]
P3R[:,12] = [ 0.048,  0.734,  1.113,  2.593,  2.816,  2.762,  2.731,  2.897, -2.873, -2.660, -2.709, -2.761]
P3R[:,13] = [-0.263,  0.362,  0.645,  1.399,  1.775,  1.947,  2.105,  2.221,  2.650,  3.028,  3.031,  3.091]
P3R[:,14] = [-0.640, -0.086,  0.218,  0.733,  1.110,  1.342,  1.746,  1.973,  2.275,  2.315,  2.204,  2.249]
P3R[:,15] = [-0.657, -0.056,  0.207,  0.499,  0.784,  1.104,  1.801,  2.010,  2.215,  1.764,  1.636,  1.643]
P3R[:,16] = [-0.125,  0.252,  0.438,  0.735,  0.833,  1.345,  2.299,  2.310,  1.944,  1.411,  1.288,  1.240]
P3R[:,17] = [ 0.242,  0.513,  0.767,  1.048,  1.030,  1.287,  1.938,  1.810,  1.687,  1.280,  1.175,  1.131]


def _interp(x, xp, fp, **kwargs):
    _fill_value = kwargs.pop("fill_value", "extrapolate")
    # only scipy seems to do linear extrapolation
    _intp = interp1d(xp, fp, fill_value=_fill_value, **kwargs)
    ret = _intp(x)
    return ret


def noy_bg(pres, times, xtype="dens"):
    ds = times.dt.dayofyear.rename("ds").values
    f1 = 2 * np.pi * ds / 365.0
    f2 = 4 * np.pi * ds / 365.0
    f3 = 6 * np.pi * ds / 365.0
    if xtype == "dens":
        am = np.atleast_3d(_interp(pres, PREF, AMR, axis=0))
        a1 = np.atleast_3d(_interp(pres, PREF, A1R, axis=0))
        a2 = np.atleast_3d(_interp(pres, PREF, A2R, axis=0))
        a3 = np.atleast_3d(_interp(pres, PREF, A3R, axis=0))
        p1 = np.atleast_3d(_interp(pres, PREF, P1R, axis=0))
        p2 = np.atleast_3d(_interp(pres, PREF, P2R, axis=0))
        p3 = np.atleast_3d(_interp(pres, PREF, P3R, axis=0))
        bg = am * (1.0 + a1 * np.sin(f1 + p1) + a2 * np.sin(f2 + p2) + a3 * np.sin(f3 + p3))
        return bg
    else:
        return np.zeros((len(pres), len(times)), dtype=float)


def noy_epp(pres, lats, times, ap_da, dw=DW, xtype="dens"):
    dn = days_since_Jul01(times).rename("dn")
    ds = times.dt.dayofyear.rename("ds")
    dl = np.arange(dw, dtype=float)
    dl[0] = 0.5

    if xtype == "dens":
        tn = _interp(pres, PREF, TMNR)
        ts = _interp(pres, PREF, TMSR)
        nn = _interp(pres, PREF, NMNR)
        ns = _interp(pres, PREF, NMSR)
        wn = _interp(pres, PREF, WMNR)
        ws = _interp(pres, PREF, WMSR)
    elif xtype == "flux":
        tn = _interp(pres, PREF, TFNR)
        ts = _interp(pres, PREF, TFSR)
        nn = _interp(pres, PREF, NFNR)
        ns = _interp(pres, PREF, NFSR)
        wn = _interp(pres, PREF, WFNR)
        ws = _interp(pres, PREF, WFSR)
    else:
        logger.error("Unknown `xtype` '%s'.", xtype)
        return -3

    seasn = Nm_func_F16(dn.values[:, None], nn, wn, tn)
    seass = Nm_func_F16(ds.values[:, None], ns, ws, ts)
    logger.info("seasn: %s", seasn.shape)
    logger.info("seass: %s", seass.shape)

    txn = _interp(pres, PREF, TXNR)[:, None]
    txs = _interp(pres, PREF, TXSR)[:, None]
    dxn = (np.sqrt(0.7 * txn) + 6) / np.sqrt(2.0)
    dxs = (np.sqrt(0.7 * txs) + 6) / np.sqrt(2.0)
    filtern = Green_filter_F16(dl, txn, dxn)
    filters = Green_filter_F16(dl, txs, dxs)
    logger.info("filtern: %s", filtern.shape)
    logger.info("filters: %s", filters.shape)

    apts = time_matrix(times.values, dw=dw)
    aph = ap_da.sel(time=apts).values.T
    xn = seasn.T * filtern.dot(aph)
    xs = seass.T * filters.dot(aph)
    return xn, xs


def noy_ese1(pres, times, ese_dates, ap_da, avtype="daily", dw=DW, xtype="dens"):
    def _to_dataarray(d, noy_es):
        tt1 = pd.to_timedelta(range(len(noy_es)), unit="D")
        noy_ese_da = xr.DataArray(
            noy_es, dims=["time", "pressure"],
            coords={"time": d.values + tt1.values, "pressure": pres},
        )
        return noy_ese_da.T

    if xtype == "dens":
        tn = _interp(pres, PREF, TMNR)
        nn = _interp(pres, PREF, NMNR)
        wn = _interp(pres, PREF, WMNR)
    elif xtype == "flux":
        tn = _interp(pres, PREF, TFNR)
        nn = _interp(pres, PREF, NFNR)
        wn = _interp(pres, PREF, WFNR)
    else:
        logger.error("Unknown `xtype` '%s'.", xtype)
        return -3
    noy_es = noy_ese(ese_dates[0], pres, nn, wn, tn, ap_da, avtype=avtype, dw=dw, xtype=xtype)
    logger.debug("ese: %s", noy_es.shape)
    noy_es_da = _to_dataarray(ese_dates[0], noy_es)
    logger.debug("ese: %s", noy_es_da)
    noy_es_da = noy_es_da.reindex(time=times).fillna(0.)
    logger.debug("ese: %s", noy_es_da)
    for d in ese_dates[1:]:
        noy_esd = noy_ese(d, pres, nn, wn, tn, ap_da, avtype=avtype, dw=dw, xtype=xtype)
        noy_es_dda = _to_dataarray(d, noy_esd)
        noy_es_da += noy_es_dda.reindex(time=times).fillna(0.)
    logger.debug("ese: %s", noy_es_da)
    return noy_es_da


def lat_distr(pres, lats, xx, ld, lref, xtype="dens"):
    hh = _interp(pres, PREF, ld, axis=0)
    xld = _interp(lats, lref, hh, axis=1)
    logger.info("xld: %s", xld.shape)
    dlat = np.gradient(lats)
    lat_hi = lats + 0.5 * dlat
    lat_lo = lats - 0.5 * dlat
    dsinl = np.sin(np.radians(lat_hi)) - np.sin(np.radians(lat_lo))
    # dsinl = np.abs(np.gradient(np.sin(np.radians(lats))))
    area = 2 * np.pi * dsinl * R_EARTH**2  # cm2
    logger.info("area: %s", area)
    # Gmol / km to cm-3
    fac = xld / np.sum(xld * area, axis=1, keepdims=True) * N_AVO * 1e4
    if xtype == "flux":
        # Gmol / day to cm-2 s-1
        fac = fac / 86400 * 1e9
    fac = np.atleast_3d(fac)
    logger.info("fac: %s", fac.shape)
    ret = xx[:, None, :] * fac
    logger.info("epp: %s", ret[0, 0])
    return ret


def finalize(da):
    da["latitude"].attrs.update(dict(
        axis = "Y",
        long_name = "central latitude",
        standard_name = "latitude",
        _CoordinateAxisType = "Lat",
        units = "degrees_north",
        # bounds = "lat_bnds",
    ))
    da["pressure"].attrs.update(dict(
        axis = "Z",
        long_name = "Pressure level",
        standard_name = "air_pressure",
        _CoordinateAxisType = "Pressure",
        units = "hPa",
    ))
    da["time"].attrs.update(dict(
        axis = "T",
        long_name = "time",
        standard_name = "time",
        _CoordinateAxisType = "Time",
    ))
    return da


# %%
CONTEXT_SETTINGS = dict(help_option_names=["-h", "--help"], show_default=True)


# %%
@click.command(context_settings=CONTEXT_SETTINGS)
@click.option(
    "-c",
    "--config-file",
    default=None,
    help="Configuration file to use, takes precedence over the argument.",
)
@click.argument("configfile", required=False)
@click.option(
    "-a", "--apfile", default=None, type=str, help="Ap forcing file."
)
@click.option(
    "-t", "--xtype", default="dens", type=str, help="Density (dens) or flux (flux) UBC."
)
@click.option(
    "-E", "--include-ese/--exclude-ese", default=True, is_flag=True, help="Include ESE NOy."
)
@click.option(
    "-o", "--output", default="ubc_test.nc", type=str, help="Name of the output file."
)
# Do not write output file
@click.option(
    "-n", "--dry-run", default=False, is_flag=True, help="Dry run, no output written."
)
# logging options
@click.option(
    "-v", "--verbose", "loglevel", flag_value="INFO", help="Sets 'loglevel' to 'INFO'."
)
@click.option(
    "-q", "--quiet", "loglevel", flag_value="ERROR", help="Sets 'loglevel' to 'ERROR'."
)
@click.option(
    "-l",
    "--loglevel",
    default="WARN",
    help="Sets 'loglevel' directly to the given level.",
    type=click.Choice(["DEBUG", "INFO", "WARN", "ERROR"], case_sensitive=False),
)
def main(
    config_file,
    configfile,
    apfile,
    include_ese,
    output,
    xtype,
    dry_run,
    loglevel,
):
    try:
        xr.set_options(keep_attrs=True)
        xr.set_options(display_max_rows=1000)
    except ValueError:
        pass
    np.set_printoptions(precision=6)

    # %%
    xr.set_options(display_width=100)
    xr.set_options(cmap_divergent="PuOr_r", cmap_sequential="cividis")

    # %%
    logger = logging.getLogger()
    debug = logger.debug
    info = logger.info
    warn = logger.warning
    error = logger.error
    info("setting loglevel: %s", loglevel)
    logger.setLevel(getattr(logging, loglevel.upper(), None))
    warn("new loglevel: %s", logging.getLogger().getEffectiveLevel())

    if not config_file:
        if not configfile:
            error(
                "No config file specified, "
                "use `--config-file` or pass as argument."
            )
            return -1
        else:
            config_file = configfile

    with open(config_file, mode="rb") as _fp:
        config = toml.loads(_fp.read().decode())
    debug("config: %s", config)

    # Time setup, command line takes precedence,
    # but works only if both year and month are given.
    times = pd.DatetimeIndex([])
    for _ts in config.get("times", []):
        times = times.union(pd.date_range(**_ts))
    debug("times: %s", times)
    times = xr.DataArray(times, dims=["time"])
    dn = days_since_Jul01(times).rename("dn")
    ds = times.dt.dayofyear.rename("ds")
    debug("dn: %s", dn)
    debug("ds: %s", ds)

    lats = np.array(config.get("latitudes", LREF))
    pres = np.array(config.get("pressures", PREF))
    eses = xr.DataArray(pd.to_datetime(config.get("ese_dates", DESE)), dims=["time"])

    apfile = apfile or config.get("apfile", None)
    if apfile is None:
        return -2
    sw_ds = xr.open_dataset(apfile)
    ap_da = sw_ds.ap

    xn, xs = noy_epp(pres, lats, times, ap_da, dw=DW, xtype=xtype)
    info("xn: %s", xn.shape)
    info("xs: %s", xs.shape)

    eppn = lat_distr(pres, lats[lats >= 0], xn, LDN, LREFN, xtype=xtype)
    epps = lat_distr(pres, lats[lats < 0], xs, LDS, LREFS, xtype=xtype)
    epp = np.hstack([epps, eppn])
    info("epp: %s", epp.shape)

    bg = noy_bg(pres, times, xtype=xtype)
    info("bg: %s", bg.shape)

    ubc = epp + bg

    if include_ese:
        ese = noy_ese1(pres, times, eses, ap_da, avtype="daily", dw=DW, xtype=xtype)
        esen = lat_distr(pres, lats[lats >= 0], ese.values, LDE, LREFN, xtype=xtype)
        eseall = np.hstack([np.zeros_like(epps), esen])
        info("ese: %s", eseall.shape)
        ubc += eseall

    info("ubc: %s", ubc.shape)

    ubc_da = xr.DataArray(
        ubc,
        dims=["pressure", "latitude", "time"],
        coords={"pressure": pres, "latitude": lats, "time": times},
        name="noy",
        attrs={
            "long_name": "NOy density",
            "standard_name": "mole_concentration_of_noy_expressed_as_nitrogen_in_air",
            "units": "cm-3",
        }
    )
    ubc_ds = finalize(ubc_da).to_dataset().transpose("time", "pressure", "latitude")
    info("ubc: %s", ubc_ds)

    if not dry_run:
        ubc_ds["time"].encoding = {"units": "days since 1850-01-01", "dtype": float}
        ubc_ds.to_netcdf(output, unlimited_dims=["time"])

    return 0


if __name__ == "__main__":
    main()
