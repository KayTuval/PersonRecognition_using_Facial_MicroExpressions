__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import numpy as np
from matplotlib import pyplot as plt
from scipy import fftpack

def rotate_spectrum(self, mid_freq, freq_distance):
    from khen_hardware_libs.SignalClass.SignalClass import SignalClass, get_spectral_filter

    # Pos
    bpf = get_spectral_filter(self.frequencies, mid_freq - freq_distance, mid_freq + freq_distance)
    bpf_filter = SignalClass(self.time_vec, self.time_step)
    bpf_filter.set_signal_by_spectrum(bpf)
    filtered_signal_pos = self * bpf_filter
    # filtered_signal_pos.show_spectrum()
    # Flip the frequencies around the center axis
    mid_index = np.abs(self.frequencies - mid_freq).argmin()
    index_vec = range(len(self.frequencies))
    index_vec = index_vec - 2 * (index_vec - mid_index)
    filtered_signal_pos.set_signal_by_spectrum(filtered_signal_pos.signal_fft[index_vec])
    # + Neg
    filtered_signal_pos.set_signal(2 * np.real(filtered_signal_pos.signal))
    return filtered_signal_pos


def move_spectrum(self, start_freq, end_freq, move_freq):
    from khen_hardware_libs.SignalClass.SignalClass import SignalClass, get_spectral_filter

    move_filter = SignalClass(self.time_vec, self.time_step)
    lpf = get_spectral_filter(self.frequencies, start_freq, end_freq)
    # lpf[ self.frequencies == start_freq ] = 1
    # lpf[ self.frequencies == end_freq ] = 1
    lpf[np.abs(self.frequencies - start_freq).argmin()] = 1
    lpf[np.abs(self.frequencies - end_freq).argmin()] = 1
    # lpf = get_BPF(self.frequencies, start_freq, end_freq)

    move_filter.set_signal_by_spectrum(lpf)
    return (self * move_filter) & SignalClass(self.get_exp(move_freq), self.time_step, self.name)


def cross_signal(self, start_freq, end_freq):
    pos_side = self.move_spectrum(start_freq, end_freq, -(end_freq + start_freq))
    neg_side = self.move_spectrum(-end_freq, -start_freq, end_freq + start_freq)

    # 0 freq counts twice! reducing the DC:
    # pos_side.signal_fft[ np.abs( self.frequencies - start_freq ).argmin() ] = 1
    # neg_side.signal_fft[ np.abs( self.frequencies - end_freq ).argmin() ] = 1

    # neg_side.signal_fft[ self.frequencies == start_freq ] *= 0.5
    # pos_side.signal_fft[ self.frequencies == start_freq ] *= 0.5
    # pos_side.set_signal_by_spectrum( pos_side.signal_fft )
    # neg_side.set_signal_by_spectrum( neg_side.signal_fft )

    pos_side.signal -= pos_side.signal.mean() / 2
    neg_side.signal -= neg_side.signal.mean() / 2

    self.set_signal((pos_side.signal + neg_side.signal))
    return

def filter_signal(self, spectral_filter):
    from khen_hardware_libs.SignalClass.SignalClass import SignalClass

    self.signal_fft = np.multiply(self.signal_fft, spectral_filter)
    self.signal = fftpack.ifft(self.signal_fft)
    self.signal *= len(self.signal) # TODO
    return

def filter_white_noise(self, threshold = 0.0):
    self.signal_fft[ np.abs(self.signal_fft) <= threshold ] = 0.0
    self.set_signal_by_spectrum(self.signal_fft)
    return
