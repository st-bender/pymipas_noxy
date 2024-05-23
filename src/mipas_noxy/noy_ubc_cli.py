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
# V5 / standard UBC (F16)
TXNR = [40.3,  33.3,  29.8,  38.0,  41.4,  41.1,  36.8,  29.9,  22.7,  13.7,  12.2,  11.7]
TXSR = [40.3,  33.3,  28.1,  22.7,  20.2,  19.2,  18.0,  17.0,  15.8,  13.3,  10.6,  10.4]
TXNR = np.array(TXNR)
TXSR = np.array(TXSR)

# maximum density occurence times (NH and SH)
# V5 / standard UBC (F16)
TMNR = [179.8, 183.1, 187.4, 193.6, 196.1, 195.9, 192.7, 187.5, 182.1, 175.3, 174.2, 173.8]
TMSR = [195.3, 190.0, 186.1, 182.1, 180.2, 179.4, 178.5, 177.8, 176.9, 175.0, 173.0, 172.8]
TMNR = np.array(TMNR)
TMSR = np.array(TMSR)

#  maximum flux occurence times (NH and SH)
TFNR = [158.7, 155.4, 152.8, 150.0, 148.5, 147.8, 147.1, 146.8, 146.6, 146.4, 146.0, 145.8]
TFSR = [185.6, 180.6, 176.6, 171.8, 168.9, 167.2, 165.4, 164.0, 163.0, 161.5, 160.1, 159.8]
TFNR = np.array(TFNR)
TFSR = np.array(TFSR)

# maximum densities (NH and SH)
# V5 / standard UBC (F16)
NMNR = [0.00128, 0.00110, 0.00091, 0.00067, 0.00058, 0.00057, 0.00066, 0.00079, 0.00092, 0.00104, 0.00100, 0.00115]
NMSR = [0.00413, 0.00346, 0.00288, 0.00216, 0.00174, 0.00150, 0.00128, 0.00115, 0.00111, 0.00114, 0.00122, 0.00135]
NMNR = np.array(NMNR)
NMSR = np.array(NMSR)

# maximum fluxes (NH and SH)
NFNR = [0.00048, 0.00054, 0.00057, 0.00058, 0.00059, 0.00060, 0.00061, 0.00062, 0.00062, 0.00063, 0.00064, 0.00070]
NFSR = [0.00156, 0.00163, 0.00169, 0.00178, 0.00186, 0.00191, 0.00200, 0.00207, 0.00214, 0.00223, 0.00227, 0.00240]
NFNR = np.array(NFNR)
NFSR = np.array(NFSR)

#  width parameter densities (NH and SH)
# V5 / standard UBC (F16)
WMNR = [0.0558, 0.0469, 0.0383, 0.0280, 0.0242, 0.0244, 0.0284, 0.0345, 0.0409, 0.0479, 0.0475, 0.0474]
WMSR = [0.0631, 0.0598, 0.0562, 0.0508, 0.0480, 0.0470, 0.0469, 0.0477, 0.0483, 0.0479, 0.0473, 0.0472]
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
# V5 / standard UBC (F16)
AMR[:, 0] = [1.53e+08, 8.81e+07, 5.44e+07, 2.73e+07, 1.61e+07, 1.09e+07, 6.21e+06, 3.89e+06, 1.57e+06, 1.27e+06, 1.06e+06, 9.00e+05]
AMR[:, 1] = [1.59e+08, 9.16e+07, 5.70e+07, 2.84e+07, 1.64e+07, 1.09e+07, 5.99e+06, 3.65e+06, 1.87e+06, 1.45e+06, 9.96e+05, 8.53e+05]
AMR[:, 2] = [1.70e+08, 9.86e+07, 6.15e+07, 2.97e+07, 1.69e+07, 1.11e+07, 5.90e+06, 3.59e+06, 2.44e+06, 1.65e+06, 9.38e+05, 8.13e+05]
AMR[:, 3] = [2.01e+08, 1.17e+08, 7.23e+07, 3.40e+07, 1.91e+07, 1.25e+07, 6.50e+06, 3.97e+06, 2.70e+06, 1.63e+06, 9.35e+05, 8.10e+05]
AMR[:, 4] = [2.41e+08, 1.44e+08, 8.92e+07, 4.11e+07, 2.23e+07, 1.41e+07, 6.95e+06, 4.11e+06, 2.72e+06, 1.67e+06, 9.58e+05, 8.30e+05]
AMR[:, 5] = [2.68e+08, 1.60e+08, 9.80e+07, 4.46e+07, 2.39e+07, 1.49e+07, 6.99e+06, 3.95e+06, 2.77e+06, 1.59e+06, 9.19e+05, 7.97e+05]
AMR[:, 6] = [2.81e+08, 1.63e+08, 9.65e+07, 4.28e+07, 2.22e+07, 1.37e+07, 6.35e+06, 3.47e+06, 2.37e+06, 1.40e+06, 8.08e+05, 7.00e+05]
AMR[:, 7] = [2.81e+08, 1.57e+08, 8.94e+07, 3.83e+07, 1.93e+07, 1.17e+07, 5.33e+06, 2.87e+06, 1.95e+06, 1.17e+06, 6.73e+05, 5.83e+05]
AMR[:, 8] = [2.74e+08, 1.46e+08, 8.09e+07, 3.45e+07, 1.74e+07, 1.05e+07, 4.66e+06, 2.46e+06, 1.64e+06, 9.87e+05, 5.69e+05, 4.93e+05]
AMR[:, 9] = [2.79e+08, 1.48e+08, 8.15e+07, 3.44e+07, 1.70e+07, 1.01e+07, 4.32e+06, 2.22e+06, 1.48e+06, 8.93e+05, 5.15e+05, 4.47e+05]
AMR[:,10] = [2.96e+08, 1.62e+08, 9.04e+07, 3.70e+07, 1.75e+07, 1.02e+07, 4.32e+06, 2.24e+06, 1.46e+06, 8.53e+05, 4.92e+05, 4.27e+05]
AMR[:,11] = [3.04e+08, 1.73e+08, 9.97e+07, 4.16e+07, 2.02e+07, 1.19e+07, 5.23e+06, 2.71e+06, 1.73e+06, 9.93e+05, 5.73e+05, 4.97e+05]
AMR[:,12] = [2.81e+08, 1.65e+08, 9.82e+07, 4.28e+07, 2.16e+07, 1.28e+07, 5.76e+06, 3.04e+06, 1.95e+06, 1.09e+06, 6.27e+05, 5.43e+05]
AMR[:,13] = [2.36e+08, 1.41e+08, 8.52e+07, 3.82e+07, 1.97e+07, 1.20e+07, 5.63e+06, 3.08e+06, 1.97e+06, 1.07e+06, 6.15e+05, 5.33e+05]
AMR[:,14] = [1.99e+08, 1.18e+08, 7.16e+07, 3.28e+07, 1.76e+07, 1.10e+07, 5.29e+06, 2.89e+06, 1.92e+06, 1.05e+06, 6.04e+05, 5.23e+05]
AMR[:,15] = [1.82e+08, 1.04e+08, 6.26e+07, 2.97e+07, 1.64e+07, 1.05e+07, 5.18e+06, 2.78e+06, 1.92e+06, 1.01e+06, 5.73e+05, 4.97e+05]
AMR[:,16] = [1.75e+08, 9.65e+07, 5.76e+07, 2.81e+07, 1.64e+07, 1.10e+07, 5.97e+06, 3.42e+06, 2.66e+06, 1.47e+06, 8.38e+05, 7.20e+05]
AMR[:,17] = [1.71e+08, 9.43e+07, 5.66e+07, 2.77e+07, 1.76e+07, 1.30e+07, 8.19e+06, 5.14e+06, 3.04e+06, 1.65e+06, 1.06e+06, 8.87e+05]

#  background annual amplitude (relative to mean)
A1R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
A1R[:, 0] = [0.548, 0.592, 0.635, 0.691, 0.714, 0.697, 0.678, 0.648, 0.815, 1.035, 1.098, 1.158]
A1R[:, 1] = [0.483, 0.532, 0.570, 0.621, 0.636, 0.615, 0.596, 0.576, 0.742, 0.962, 1.004, 1.036]
A1R[:, 2] = [0.333, 0.381, 0.412, 0.463, 0.481, 0.472, 0.458, 0.440, 0.570, 0.813, 0.838, 0.843]
A1R[:, 3] = [0.167, 0.235, 0.271, 0.315, 0.333, 0.332, 0.331, 0.356, 0.505, 0.805, 0.823, 0.823]
A1R[:, 4] = [0.164, 0.187, 0.216, 0.227, 0.209, 0.185, 0.177, 0.228, 0.345, 0.583, 0.592, 0.592]
A1R[:, 5] = [0.217, 0.211, 0.202, 0.147, 0.107, 0.103, 0.149, 0.206, 0.254, 0.375, 0.376, 0.376]
A1R[:, 6] = [0.245, 0.221, 0.201, 0.159, 0.158, 0.172, 0.215, 0.258, 0.256, 0.285, 0.285, 0.285]
A1R[:, 7] = [0.213, 0.172, 0.151, 0.139, 0.151, 0.177, 0.238, 0.280, 0.270, 0.268, 0.268, 0.268]
A1R[:, 8] = [0.117, 0.099, 0.099, 0.114, 0.142, 0.174, 0.232, 0.273, 0.272, 0.264, 0.264, 0.264]
A1R[:, 9] = [0.097, 0.172, 0.220, 0.220, 0.186, 0.193, 0.213, 0.230, 0.242, 0.243, 0.243, 0.243]
A1R[:,10] = [0.170, 0.244, 0.303, 0.299, 0.229, 0.201, 0.169, 0.165, 0.152, 0.148, 0.148, 0.148]
A1R[:,11] = [0.192, 0.246, 0.297, 0.278, 0.216, 0.187, 0.137, 0.095, 0.097, 0.106, 0.106, 0.106]
A1R[:,12] = [0.132, 0.190, 0.239, 0.227, 0.190, 0.205, 0.207, 0.128, 0.147, 0.112, 0.112, 0.112]
A1R[:,13] = [0.112, 0.166, 0.231, 0.246, 0.237, 0.248, 0.238, 0.173, 0.193, 0.139, 0.143, 0.143]
A1R[:,14] = [0.084, 0.136, 0.229, 0.283, 0.290, 0.301, 0.278, 0.200, 0.161, 0.220, 0.231, 0.231]
A1R[:,15] = [0.053, 0.126, 0.227, 0.291, 0.327, 0.351, 0.329, 0.259, 0.294, 0.623, 0.648, 0.649]
A1R[:,16] = [0.191, 0.234, 0.324, 0.425, 0.433, 0.407, 0.348, 0.185, 0.265, 0.552, 0.580, 0.591]
A1R[:,17] = [0.278, 0.342, 0.446, 0.602, 0.443, 0.251, 0.178, 0.324, 0.296, 0.538, 0.646, 0.724]


#  background semi-annual amplitude (relative to mean)
A2R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
A2R[:, 0] = [0.039, 0.115, 0.175, 0.228, 0.254, 0.270, 0.337, 0.359, 0.477, 0.540, 0.514, 0.484]
A2R[:, 1] = [0.020, 0.090, 0.136, 0.180, 0.215, 0.243, 0.318, 0.333, 0.427, 0.496, 0.491, 0.484]
A2R[:, 2] = [0.097, 0.139, 0.164, 0.166, 0.185, 0.206, 0.251, 0.278, 0.369, 0.416, 0.413, 0.411]
A2R[:, 3] = [0.155, 0.194, 0.222, 0.200, 0.182, 0.184, 0.229, 0.266, 0.321, 0.349, 0.346, 0.346]
A2R[:, 4] = [0.150, 0.192, 0.210, 0.192, 0.155, 0.157, 0.225, 0.268, 0.268, 0.262, 0.259, 0.259]
A2R[:, 5] = [0.162, 0.192, 0.201, 0.187, 0.169, 0.172, 0.211, 0.250, 0.239, 0.220, 0.219, 0.219]
A2R[:, 6] = [0.162, 0.168, 0.168, 0.172, 0.170, 0.175, 0.203, 0.213, 0.213, 0.205, 0.205, 0.205]
A2R[:, 7] = [0.128, 0.129, 0.133, 0.149, 0.156, 0.167, 0.196, 0.203, 0.207, 0.202, 0.202, 0.202]
A2R[:, 8] = [0.106, 0.106, 0.114, 0.142, 0.150, 0.161, 0.188, 0.200, 0.225, 0.229, 0.229, 0.229]
A2R[:, 9] = [0.113, 0.096, 0.099, 0.144, 0.160, 0.176, 0.221, 0.235, 0.253, 0.250, 0.250, 0.250]
A2R[:,10] = [0.095, 0.097, 0.112, 0.159, 0.176, 0.189, 0.225, 0.225, 0.239, 0.237, 0.237, 0.237]
A2R[:,11] = [0.095, 0.096, 0.115, 0.159, 0.192, 0.211, 0.271, 0.293, 0.295, 0.296, 0.296, 0.296]
A2R[:,12] = [0.116, 0.125, 0.142, 0.173, 0.215, 0.237, 0.312, 0.344, 0.363, 0.368, 0.368, 0.368]
A2R[:,13] = [0.126, 0.144, 0.166, 0.196, 0.231, 0.243, 0.303, 0.332, 0.341, 0.374, 0.370, 0.370]
A2R[:,14] = [0.077, 0.116, 0.169, 0.240, 0.270, 0.270, 0.288, 0.284, 0.323, 0.357, 0.353, 0.353]
A2R[:,15] = [0.107, 0.144, 0.197, 0.295, 0.327, 0.329, 0.312, 0.215, 0.260, 0.380, 0.383, 0.383]
A2R[:,16] = [0.154, 0.214, 0.269, 0.371, 0.421, 0.449, 0.476, 0.430, 0.557, 0.674, 0.680, 0.680]
A2R[:,17] = [0.185, 0.249, 0.301, 0.399, 0.502, 0.603, 0.728, 0.785, 0.820, 0.720, 0.689, 0.665]

#  background ter-annual amplitude (relative to mean)
A3R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
A3R[:, 0] = [0.096, 0.097, 0.094, 0.105, 0.084, 0.082, 0.053, 0.027, 0.117, 0.174, 0.210, 0.250]
A3R[:, 1] = [0.082, 0.056, 0.038, 0.046, 0.040, 0.041, 0.021, 0.026, 0.106, 0.131, 0.137, 0.143]
A3R[:, 2] = [0.082, 0.048, 0.022, 0.026, 0.034, 0.037, 0.040, 0.062, 0.119, 0.092, 0.094, 0.097]
A3R[:, 3] = [0.102, 0.098, 0.077, 0.058, 0.070, 0.080, 0.072, 0.072, 0.090, 0.076, 0.077, 0.077]
A3R[:, 4] = [0.084, 0.087, 0.081, 0.074, 0.074, 0.076, 0.058, 0.035, 0.043, 0.027, 0.025, 0.025]
A3R[:, 5] = [0.044, 0.045, 0.044, 0.034, 0.022, 0.027, 0.048, 0.048, 0.067, 0.073, 0.073, 0.073]
A3R[:, 6] = [0.028, 0.032, 0.027, 0.011, 0.018, 0.021, 0.023, 0.019, 0.029, 0.048, 0.048, 0.048]
A3R[:, 7] = [0.006, 0.011, 0.017, 0.003, 0.015, 0.025, 0.040, 0.051, 0.048, 0.052, 0.052, 0.052]
A3R[:, 8] = [0.052, 0.025, 0.019, 0.005, 0.014, 0.019, 0.023, 0.037, 0.047, 0.053, 0.053, 0.053]
A3R[:, 9] = [0.067, 0.035, 0.009, 0.003, 0.016, 0.026, 0.041, 0.050, 0.033, 0.026, 0.026, 0.026]
A3R[:,10] = [0.054, 0.043, 0.027, 0.021, 0.020, 0.023, 0.037, 0.033, 0.014, 0.020, 0.020, 0.020]
A3R[:,11] = [0.036, 0.027, 0.023, 0.031, 0.034, 0.035, 0.055, 0.082, 0.078, 0.068, 0.068, 0.068]
A3R[:,12] = [0.035, 0.028, 0.026, 0.029, 0.033, 0.044, 0.083, 0.103, 0.070, 0.060, 0.060, 0.060]
A3R[:,13] = [0.051, 0.051, 0.063, 0.060, 0.052, 0.055, 0.076, 0.096, 0.076, 0.090, 0.091, 0.091]
A3R[:,14] = [0.076, 0.084, 0.103, 0.091, 0.069, 0.056, 0.056, 0.056, 0.028, 0.050, 0.051, 0.051]
A3R[:,15] = [0.081, 0.094, 0.125, 0.112, 0.079, 0.043, 0.050, 0.104, 0.185, 0.171, 0.168, 0.167]
A3R[:,16] = [0.084, 0.089, 0.106, 0.093, 0.087, 0.095, 0.186, 0.230, 0.326, 0.272, 0.260, 0.252]
A3R[:,17] = [0.071, 0.098, 0.144, 0.156, 0.125, 0.257, 0.461, 0.570, 0.666, 0.572, 0.545, 0.526]

#  background annual phase
P1R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
P1R[:, 0] = [ 1.602,  1.679,  1.741,  1.801,  1.818,  1.809,  1.755,  1.747,  1.790,  1.785,  1.773,  1.762]
P1R[:, 1] = [ 1.651,  1.723,  1.748,  1.806,  1.828,  1.814,  1.764,  1.777,  1.845,  1.846,  1.846,  1.846]
P1R[:, 2] = [ 1.694,  1.749,  1.748,  1.836,  1.850,  1.842,  1.824,  1.902,  1.939,  1.879,  1.877,  1.877]
P1R[:, 3] = [ 1.219,  1.443,  1.499,  1.702,  1.804,  1.845,  1.919,  2.041,  1.996,  1.895,  1.893,  1.893]
P1R[:, 4] = [ 0.432,  0.846,  1.110,  1.452,  1.695,  1.857,  2.231,  2.427,  2.249,  2.031,  2.028,  2.028]
P1R[:, 5] = [ 0.630,  0.843,  1.082,  1.499,  2.313,  2.828, -2.911, -2.987,  2.863,  2.421,  2.419,  2.419]
P1R[:, 6] = [ 1.038,  1.199,  1.444,  1.983,  2.627,  2.939, -2.945, -2.925,  3.090,  2.774,  2.774,  2.774]
P1R[:, 7] = [ 1.477,  1.656,  1.879,  2.346,  2.854,  3.101, -2.988, -2.942, -3.014,  3.133,  3.133,  3.133]
P1R[:, 8] = [ 2.157,  2.820, -2.956, -2.741, -2.698, -2.716, -2.815, -2.912, -2.936, -2.981, -2.981, -2.981]
P1R[:, 9] = [-2.317, -2.115, -1.991, -2.006, -2.281, -2.494, -2.779, -2.903, -2.961, -3.033, -3.033, -3.033]
P1R[:,10] = [-1.631, -1.698, -1.693, -1.765, -1.971, -2.168, -2.546, -2.668, -2.933,  3.120,  3.120,  3.120]
P1R[:,11] = [-1.416, -1.487, -1.500, -1.557, -1.596, -1.693, -1.977, -2.292, -2.981, -2.993, -2.993, -2.993]
P1R[:,12] = [-1.726, -1.725, -1.704, -1.801, -1.948, -2.124, -2.327, -2.928,  2.714, -2.950, -2.949, -2.949]
P1R[:,13] = [-2.752, -2.334, -2.184, -2.241, -2.358, -2.446, -2.586, -3.115,  2.718, -2.271, -2.233, -2.233]
P1R[:,14] = [ 3.018, -2.473, -2.344, -2.417, -2.477, -2.476, -2.567, -3.038,  3.025, -1.889, -1.855, -1.855]
P1R[:,15] = [-1.407, -1.861, -2.034, -2.075, -2.125, -2.135, -2.178, -2.254, -1.874, -1.433, -1.422, -1.423]
P1R[:,16] = [-1.115, -1.459, -1.661, -1.662, -1.660, -1.654, -1.594, -1.550, -1.308, -1.283, -1.294, -1.307]
P1R[:,17] = [-1.160, -1.413, -1.562, -1.595, -1.492, -1.167,  0.099,  0.825,  0.323, -1.031, -1.156, -1.227]

#  background semi-annual phase
P2R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
P2R[:, 0] = [ 1.221,  0.838,  0.889,  1.131,  1.276,  1.233,  1.254,  1.334,  1.657,  1.800,  1.824,  1.845]
P2R[:, 1] = [-0.509,  0.295,  0.541,  0.951,  1.152,  1.157,  1.229,  1.271,  1.629,  1.752,  1.743,  1.729]
P2R[:, 2] = [-1.334, -0.756, -0.422,  0.216,  0.520,  0.643,  0.873,  0.991,  1.469,  1.646,  1.644,  1.637]
P2R[:, 3] = [-1.420, -1.015, -0.756, -0.228,  0.208,  0.539,  1.001,  1.257,  1.574,  1.730,  1.731,  1.731]
P2R[:, 4] = [-1.209, -0.827, -0.579, -0.177,  0.253,  0.667,  1.172,  1.400,  1.564,  1.675,  1.671,  1.671]
P2R[:, 5] = [-0.682, -0.389, -0.094,  0.371,  0.794,  1.033,  1.401,  1.556,  1.661,  1.757,  1.756,  1.756]
P2R[:, 6] = [-0.226,  0.024,  0.325,  0.776,  1.054,  1.170,  1.406,  1.531,  1.638,  1.672,  1.672,  1.672]
P2R[:, 7] = [ 0.332,  0.554,  0.771,  1.138,  1.290,  1.347,  1.433,  1.496,  1.526,  1.487,  1.487,  1.487]
P2R[:, 8] = [ 1.278,  1.177,  1.096,  1.287,  1.349,  1.368,  1.412,  1.458,  1.519,  1.504,  1.504,  1.504]
P2R[:, 9] = [ 1.611,  1.572,  1.471,  1.584,  1.614,  1.622,  1.721,  1.816,  1.862,  1.854,  1.854,  1.854]
P2R[:,10] = [ 1.050,  1.384,  1.604,  1.703,  1.701,  1.731,  1.844,  1.993,  2.003,  1.965,  1.965,  1.965]
P2R[:,11] = [ 0.388,  0.798,  1.191,  1.530,  1.687,  1.767,  1.886,  1.987,  2.122,  2.186,  2.186,  2.186]
P2R[:,12] = [-0.033,  0.302,  0.607,  1.055,  1.386,  1.542,  1.814,  1.966,  2.128,  2.175,  2.175,  2.175]
P2R[:,13] = [-0.132,  0.136,  0.390,  0.850,  1.219,  1.406,  1.719,  1.938,  2.087,  2.089,  2.086,  2.086]
P2R[:,14] = [ 0.322,  0.513,  0.635,  0.973,  1.237,  1.353,  1.604,  1.868,  2.040,  1.987,  1.978,  1.978]
P2R[:,15] = [ 1.113,  1.019,  1.040,  1.258,  1.398,  1.474,  1.604,  1.686,  1.882,  1.766,  1.756,  1.759]
P2R[:,16] = [ 1.123,  1.206,  1.337,  1.524,  1.573,  1.558,  1.550,  1.407,  1.567,  1.660,  1.673,  1.687]
P2R[:,17] = [ 0.943,  1.215,  1.490,  1.677,  1.459,  1.291,  1.161,  1.023,  1.110,  1.306,  1.370,  1.425]

#  background ter-annual phase
P3R=np.zeros((12, 18), dtype=float)
# V5 / standard UBC (F16)
P3R[:, 0] = [ 0.448,  0.187, -0.035, -0.485, -0.565, -0.414,  0.045,  0.360,  2.115,  1.859,  1.852,  1.861]
P3R[:, 1] = [ 0.737,  0.413,  0.061, -0.658, -0.809, -0.647,  0.268,  0.527,  1.914,  1.613,  1.633,  1.676]
P3R[:, 2] = [ 1.398,  1.610,  2.634, -2.644, -2.495, -2.476,  3.027,  3.027,  2.738,  2.250,  2.216,  2.238]
P3R[:, 3] = [ 1.887,  2.285,  2.792, -2.928, -2.646, -2.635, -3.067,  2.843,  2.545,  2.115,  2.089,  2.089]
P3R[:, 4] = [ 2.027,  2.352,  2.725, -3.046, -2.612, -2.314, -2.126, -2.121, -2.048, -1.658, -1.688, -1.688]
P3R[:, 5] = [ 2.861, -2.965, -2.795, -2.678, -2.540, -1.907, -1.385, -1.339, -1.487, -1.397, -1.403, -1.403]
P3R[:, 6] = [-2.067, -1.755, -1.456, -0.213,  0.761,  0.865,  0.864,  0.363, -1.469, -1.543, -1.543, -1.543]
P3R[:, 7] = [ 2.714, -1.745, -1.472, -0.564,  1.315,  1.389,  1.420,  1.417,  1.636,  1.689,  1.689,  1.689]
P3R[:, 8] = [ 2.188, -3.141, -1.970, -2.523,  2.400,  2.127,  2.204,  2.330,  2.112,  1.834,  1.834,  1.834]
P3R[:, 9] = [ 1.937,  2.421, -3.100, -2.837,  2.314,  2.349,  2.700,  2.701,  2.348,  1.823,  1.823,  1.823]
P3R[:,10] = [ 1.582,  2.197,  2.809, -2.635, -2.687, -2.845, -2.902, -2.912, -1.696, -0.804, -0.804, -0.804]
P3R[:,11] = [ 0.977,  1.943,  2.775, -2.766, -2.609, -2.459, -1.773, -1.823, -1.656, -1.346, -1.346, -1.346]
P3R[:,12] = [ 0.678,  1.592,  2.017,  2.682, -3.079, -2.848, -2.254, -2.171, -1.930, -1.349, -1.349, -1.349]
P3R[:,13] = [ 0.591,  1.260,  1.521,  1.800,  2.296,  2.699, -2.759, -2.599, -2.212, -1.564, -1.534, -1.534]
P3R[:,14] = [ 0.356,  0.909,  1.232,  1.488,  1.817,  2.057,  3.141, -2.852, -1.689, -1.104, -1.054, -1.054]
P3R[:,15] = [ 0.306,  0.897,  1.089,  1.094,  1.102,  0.988, -0.897, -0.704, -0.600, -0.877, -0.888, -0.895]
P3R[:,16] = [ 0.568,  1.026,  1.309,  1.313,  0.804,  0.214, -0.353, -0.250, -0.277, -0.392, -0.400, -0.412]
P3R[:,17] = [ 0.989,  1.509,  1.872,  2.185,  0.844,  0.276,  0.081,  0.081, -0.090, -0.413, -0.502, -0.572]


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
