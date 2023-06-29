__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import numpy as np
import sys
import cv2 as cv
import datetime
from scipy import signal
import os
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os
from scipy.interpolate import griddata

from khen_hardware_libs.Scripts.Scripts_General import *
from khen_hardware_libs.Scripts.Scripts_3D import *
from khen_hardware_libs.Scripts.Scripts_Images import *
from khen_hardware_libs.Scripts.Scripts_Videos import *
from khen_hardware_libs.Scripts.Scripts_Stereo import *



def solve_Ainv_Btranspose_C_eq_x(A,B,C):
    return ( np.linalg.inv(A) @ np.transpose(B) @ C  )


def get_sigma_for_mean_estimation(confidence, max_error, number_of_samples):
    # Hoffding - Chernhoff probability bound:
    return ( np.sqrt( np.log( 1/(1-confidence) ) / (2*number_of_samples) ) * max_error )
