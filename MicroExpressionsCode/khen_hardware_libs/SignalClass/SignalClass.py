__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

# import SignalClass.SignalClass_Scripts as Scripts
import khen_hardware_libs.SignalClass.SignalClass_SignalsComparison as SignalsComparison
import khen_hardware_libs.SignalClass.SignalClass_SignalManipulations as SignalManipulations
from khen_hardware_libs.SignalClass.SignalClass_Scripts import *

def get_spectral_filter(frequencies, min_cutoff, max_cutoff):
    spectral_filter = np.ones_like(frequencies).astype(np.float64)
    spectral_filter[frequencies < min_cutoff] = 0
    spectral_filter[frequencies > max_cutoff] = 0
    return spectral_filter

def get_LPF(frequencies, cutoff_freq):
    return get_spectral_filter(frequencies, -cutoff_freq, cutoff_freq)

def get_BPF(frequencies, min_cutoff, max_cutoff):
    return get_spectral_filter(frequencies, min_cutoff, max_cutoff) + get_spectral_filter(frequencies, -max_cutoff, -min_cutoff)

class SignalClass():
    """
    A class used to represent a signal
    ...

    Attributes
    ----------
    name : str
        An optional name for the signal. Mainly used for graphs and display reasons.

    time_step : float
        A temporal interval between two samples.

    time_vec : np.array of floats
        A vector of time values (at seconds).

    frequencies : np.array of floats
        A vector of frequencies (at Hertz).

    signal : np.array of floats
        The signal vector.

    signal_fft : np.array of floats
        The fft of the signal.

    Methods
    -------
    __init__(signal, time_step, name = "")
        Get a signal and a time step to initialize the class (name is optonal)

    rotate_spectrum(mid_freq, freq_distance)
        Rotate the spectrum values around a specific frequency (mid_freq) and up to a specific "raius" (freq_distance).
        Used for Anti-Aliasing methods.

    move_spectrum(start_freq, end_freq, move_freq)
        Crop part of the frequency domain (from start_freq to end_freq) and move it at move_freq frequency to the right.

    cross_signal(start_freq, end_freq)
        Flip the signal spectral values from start_freq to end_freq around the middle frequency.

    get_exp(freq)
        Returns an exponent function with a single frequency.

    get_sin(freq)
        Returns a sinus function with a single frequency.

    get_cos(freq)
        Returns an cosine function with a single frequency.

    set_signal(signal)
        Gets a signal vector and substitute it for the signal class signal. Calculate the signal fft in addition.

    clean_phase()
        Nullify the phase for the entire spectrum range.

    set_signal_by_spectrum(self, spectrum_signal)
        Define a signal class using the spectrum domain. Then, generate the signal using inverse-fft

    get_spectrum()
        Return the spectrum of the signal.

    filter_signal(spectral_filter)
        Filters the spectrum with spectral_filter profile.

    get_resampled_signal(density_factor)
        return SignalClass object with temporal upsampled signal version (with factor density_factor). But with the
        same frequency content.

    show_signal()
        Display the signal versus the time.

    show_spectrum(range = '')
        Display the spectrum graph (amplitude and phase). If range is given it should be given at template [fmin, fmax]
        while it reduces the spectrum graph representation for that range only.

    __add__(other)
        In case the other is another SignalClass, returns another SignalClass which is the element-wise addition
        between the two signals.

    __and__(other)
        In case the other is another SignalClass, returns another SignalClass which is the temporal multiplication of
        the two signals.

    __mul__(other)
        If other is a float or an int, the signal is multiplied by this factor. In case the other is another
        SignalClass, returns another SignalClass which is the temporal convolution (spectral multiplication) between
        these two signals

    __imul__(other)
        See mul.

    __iadd__(other)
        See add.

    __iand__(other)
        See and.

    __sub__(other)
        In case the other is another SignalClass, returns another SignalClass which is the element-wise subtraction
        between the two signals.

    __truediv__(other)
        In case the other is another SignalClass, returns another SignalClass which is the element-wise division
        between the two signals.

    __mod__(other)
        In case the other is another SignalClass, returns another SignalClass which is the element-wise division
        between the two signals' spectrum.

    __invert__()
        Calculate a signal which is 1 over the original signal.

    __le__(other)
        Compare the signals with infinity norm manner (compare max distances)

    __ge__(other)
        Compare the signals with infinity norm manner (compare max distances)

    filter_white_noise(threshold)
        Get read from all spectral components with amplitude less than threshold.

    l2_distance(other, shift = 0)
        Uses L2 distance metric to evaluate the signals similarity.

    angular_distance(other, shift = 0)
        Uses angular distance metric to evaluate the signals similarity.

    pearson_correlation(other)
        Uses pearson correlation metric to evaluate the signals similarity.

    """
    def __init__(self, signal, time_step, name = ""):
        self.name = name
        self.time_step = time_step
        self.set_signal( signal )
        # self.original_signal = signal.copy()
        self.time_vec = np.arange( len(signal) ) * time_step
        return

    # Scripts
    get_exp = get_exp
    get_sin = get_sin
    get_cos = get_cos

    # Signal Manipulations
    rotate_spectrum = SignalManipulations.rotate_spectrum
    move_spectrum = SignalManipulations.move_spectrum
    cross_signal = SignalManipulations.cross_signal
    filter_signal = SignalManipulations.filter_signal
    filter_white_noise = SignalManipulations.filter_white_noise

    # Signals Comparison
    l2_distance = SignalsComparison.l2_distance
    angular_distance = SignalsComparison.angular_distance
    pearson_correlation = SignalsComparison.pearson_correlation


    def set_signal(self, signal):
        self.signal = signal
        self.get_spectrum()
        # self.original_signal = self.signal.copy()
        return

    def clean_phase(self):
        self.set_signal_by_spectrum( np.abs(self.signal_fft) )
        return

    def set_signal_by_spectrum(self, spectrum_signal):
        self.signal_fft = spectrum_signal
        self.signal = fftpack.ifft(self.signal_fft)
        self.signal *= len(self.signal) # TODO
        self.frequencies = fftpack.fftfreq(self.signal.size, d=self.time_step)
        # self.original_signal = self.signal.copy()
        return

    def get_spectrum(self):
        self.signal_fft = fftpack.fft(self.signal) #, n = 4*len(self.signal))  # The FFT of the signal
        self.signal_fft /= len(self.signal_fft) # TODO
        self.frequencies = fftpack.fftfreq(self.signal.size, d=self.time_step)  # The corresponding frequencies
        return

    def get_resampled_signal(self, dense_factor = 2):
        from scipy import signal as sg
        new_signal = sg.resample(self.signal, len(self.signal) * dense_factor)
        return SignalClass(new_signal, time_step = self.time_step/dense_factor, name = self.name)

    def show_signal(self):
        plt.figure(figsize=(15, 5))
        plt.title('Temporal Signal ' + self.name)
        plt.plot(self.time_vec , np.real(self.signal))
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.show()
        return

    def show_spectrum(self, range = ''):
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.title(self.name)
        plt.plot(self.frequencies , np.abs(self.signal_fft))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Amplitude')
        if range != '':
            plt.xlim(range)
        # plt.ylim([0.0, 0.4])

        plt.subplot(1, 2, 2)
        plt.title(self.name)
        plt.plot(self.frequencies, np.angle(self.signal_fft))
        plt.xlabel('frequency [Hz]')
        plt.ylabel('Phase')
        plt.show()

        # plt.subplot(1, 2, 1)
        # plt.plot(self.frequencies , np.abs(self.signal))
        # plt.xlabel('Time [s]')
        # plt.ylabel('Amplitude')
        # plt.show()
        return

    def __add__(self, other):
        return SignalClass( self.signal + other.signal , self.time_step, self.name)

    def __and__(self, other):   # And = temporal multiplying
        return SignalClass( np.multiply(self.signal, other.signal), self.time_step, self.name)

    def __mul__(self, other):   # Conv = temporal convolution
        if type(other) == int or type(other) == float or ( type(other) is np.float64 ):
            return SignalClass(other*self.signal, self.time_step, self.name)
        return SignalClass(fftpack.ifft( np.multiply(self.signal_fft, other.signal_fft) ) * len(self.signal_fft) , self.time_step, self.name)

    def __imul__(self, other):
        self.signal_fft = np.multiply(self.signal_fft, other.signal_fft)
        self.get_spectrum()
        return

    def __iadd__(self, other):
        self.signal = self.signal + other.signal
        self.get_spectrum()
        return

    def __iand__(self, other):
        self.signal = np.multiply(self.signal, other.signal)
        self.get_spectrum()
        return

    def __sub__(self, other):
        return SignalClass( self.signal - other.signal , self.time_step, self.name)

    def __truediv__(self, other):   # Signals Ratio
        signals_ratio = np.nan_to_num( np.true_divide(self.signal,other.signal) , 0.0 )
        signals_ratio [ signals_ratio == np.inf ] = 0.0
        return SignalClass( signals_ratio , self.time_step, self.name)

    def __mod__(self, other):       # Spectrums Ratio
        signals_fft_ratio = np.nan_to_num( np.true_divide(self.signal_fft,other.signal_fft) , 0.0 )
        signals_fft_ratio [ np.abs(signals_fft_ratio) == np.inf ] = 0.0
        new_signal = SignalClass( np.ones_like(signals_fft_ratio), self.time_step, self.name )
        new_signal.set_signal_by_spectrum( signals_fft_ratio )
        return new_signal

    def __invert__(self):           # Invert the Spectrum!
        ones_signal = SignalClass(np.ones_like(self.signal), self.time_step, self.name)
        ones_signal.set_signal_by_spectrum(np.ones_like(self.signal))       # spectrums of 1
        return ones_signal % self

    def __le__(self, other):        # In infinity norm meaning
        return np.max( self.signal - other.signal ) < 1

    def __ge__(self, other):        # In infinity norm meaning
        return np.max( other.signal - self.signal ) < 1


# if __name__ == '__main__':
#     print('test')