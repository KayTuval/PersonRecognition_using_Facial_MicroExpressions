__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from SignalClass.SignalClass import *
import numpy as np
from matplotlib import pyplot as plt


def get_exp(self, freq):
    return np.exp(1j * self.time_vec * 2 * np.pi * freq)

def get_sin(self, freq):
    return np.sin(self.time_vec * 2 * np.pi * freq)

def get_cos(self, freq):
    return np.cos(self.time_vec * 2 * np.pi * freq)


def show_spectrum(frequencies, signal, title = ""):
    fig = plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.title(title)

    plt.plot(frequencies, np.abs(signal))
    plt.xlabel('frequency [1/pixels]')
    plt.ylabel('Amplitude')

    plt.subplot(1,2,2)
    plt.plot(frequencies, np.angle(signal))
    plt.xlabel('frequency [1/pixels]')
    plt.ylabel('Phase')

    plt.show()
    return

def get_spectral_filter(frequencies, min_cutoff, max_cutoff):
    spectral_filter = np.ones_like(frequencies).astype(np.float64)
    spectral_filter[frequencies < min_cutoff] = 0
    spectral_filter[frequencies > max_cutoff] = 0
    return spectral_filter

def get_LPF(frequencies, cutoff_freq):
    return get_spectral_filter(frequencies, -cutoff_freq, cutoff_freq)

def get_BPF(frequencies, min_cutoff, max_cutoff):
    return get_spectral_filter(frequencies, min_cutoff, max_cutoff) + get_spectral_filter(frequencies, -max_cutoff, -min_cutoff)





def compare_signals(signalClass_list, xrange = '', trange = '', name = ""):
    fig = plt.figure(figsize=(18, 5))
    colors = lambda i: ['b', 'r', 'g', 'm', 'orange', 'y'][i % 5]
    legends = []

    for signal_id in range(len(signalClass_list)):
        signal = signalClass_list[signal_id]
        plt.subplot(1, 2, 1)
        plt.plot(signal.frequencies, np.abs(signal.signal_fft), linewidth = 0.75+0.3*(len(signalClass_list)-1-signal_id)/len(signalClass_list), color = colors(signal_id))
        plt.subplot(1, 2, 2)
        plt.plot(signal.time_vec, np.real(signal.signal), color = colors(signal_id))

        legends += [signal.name]

    plt.subplot(1, 2, 1)
    plt.title('Spectrum (abs)')
    if xrange != '':
        plt.xlim(xrange)
    plt.legend(legends)
    plt.subplot(1, 2, 2)
    plt.title('Temporal Signal (Real)')
    if trange != '':
        plt.xlim(trange)
    plt.legend(legends)

    if name != '':
        plt.savefig('Plots/'+name+'.png')
    plt.show()

    return



def get_modulator(time_vec, time_step, camera_fps, modulation_parameters):
    print(modulation_parameters)
    fcut_off = camera_fps/2
    blue_modulator = signalClass(time_vec, time_step, 'Blue Modulator')
    blue_modulator.set_signal( modulation_parameters['a_blue']+2*modulation_parameters['b_blue']*blue_modulator.get_cos(fcut_off) )
    red_modulator = signalClass(time_vec, time_step, 'Red Modulator')
    red_modulator.set_signal( modulation_parameters['a_red'] + 2*modulation_parameters['b_red']*red_modulator.get_cos(2*fcut_off) )
    blue_modulator.clean_phase()   # Real Signal
    red_modulator.clean_phase()   # Real Signal
    return blue_modulator, red_modulator
