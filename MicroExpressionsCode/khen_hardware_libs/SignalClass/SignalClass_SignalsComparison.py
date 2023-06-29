__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from SignalClass.SignalClass import *
import numpy as np
from matplotlib import pyplot as plt

def l2_distance(self, other, shift=0):
    # Returns the L2 distance between the signals
    # Assuming two signals with the same size
    return np.sqrt(np.square(np.abs(self.signal[shift:] - other.signal[:-shift])).sum()) / (
    self.signal[shift:].shape[0])


def angular_distance(self, other, shift=0):
    # Returns the L2 distance between the signals
    # Assuming two signals with the same size
    f1_norm = np.sqrt(np.square(np.abs(self.signal[shift:])).sum()).astype(np.float64) + 1e-30
    f2_norm = np.sqrt(np.square(np.abs(other.signal[:-shift])).sum()).astype(np.float64) + 1e-30
    product = np.multiply(np.abs(self.signal[shift:]), np.abs(other.signal[:-shift])).sum().astype(np.float64)
    # print( product, f1_norm, f2_norm )
    cos_theta = product / (f1_norm * f2_norm)
    return np.arccos(cos_theta)


def pearson_correlation(self, other):
    from scipy import stats
    return stats.pearsonr(self.signal, other.signal)