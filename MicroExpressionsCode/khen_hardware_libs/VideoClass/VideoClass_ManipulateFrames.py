__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

# from Scripts.Scripts import *
import datetime
from scipy import signal
import cv2 as cv
import numpy as np
import os
import sys
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
# from VideoClass.VideoClass import VideoClass
# import VideoClass.VideoClass as VideoClass
# VideoClass = VideoClass.VideoClass
from khen_hardware_libs.ImageClass.ImageClass import *

def _manipulate_img_by_method(self, frame, frame_id, filter_method = 'gausian_filter', parameters_dict = None):
    # Frame is an ImageClass object

    # 1. Gaussian Filter
    if filter_method == 'gausian_filter':
        if 'sigma' in parameters_dict.keys():
            if type(parameters_dict['sigma']) is int or type(parameters_dict['sigma']) is float:
                sigma = parameters_dict['sigma']
            else:
                sigma = parameters_dict['sigma'][frame_id]

            new_frame = frame.gausian_filter(sigma = sigma)
        else:
            new_frame = frame.gausian_filter()

    # 2. Median Filter:
    if filter_method == 'median_filter':
        if 'ksize' in parameters_dict.keys():
            if type(parameters_dict['ksize']) is int:
                ksize = parameters_dict['ksize']
            else:
                ksize = parameters_dict['ksize'][frame_id]
            new_frame = frame.median_filter(ksize = ksize)
        else:
            new_frame = frame.median_filter()

    # 3. Canny Edge Filter:
    if filter_method == 'canny_edge_detector':
        if 'threshold1' in parameters_dict.keys() and 'threshold2' in parameters_dict.keys():
            if len(parameters_dict['threshold1']) > 1:
                threshold1 = parameters_dict['threshold1'][frame_id]
                threshold2 = parameters_dict['threshold2'][frame_id]
            else:
                threshold1 = parameters_dict['threshold1']
                threshold2 = parameters_dict['threshold2']
            new_frame = frame.canny_edge_detector(threshold1 = threshold1, threshold2 = threshold2)
        else:
            new_frame = frame.canny_edge_detector()

    # 4. Resize frames
    if filter_method == 'resize_image':
        if 'width' in parameters_dict.keys() and 'height' in parameters_dict.keys():
            if type(parameters_dict['width']) is int:
                width = parameters_dict['width']
                height = parameters_dict['height']
            else:
                width = parameters_dict['width'][frame_id]
                height = parameters_dict['height'][frame_id]
            new_frame = frame.resize_image(width = width, height = height)
        else:
            new_frame = frame.resize_image()

    # 5. Crop frames
    if filter_method == 'crop_image':
        if 'rect' in parameters_dict.keys():
            if type(parameters_dict['rect'][0]) is int:
                rect = parameters_dict['rect']
            else:
                rect = parameters_dict['rect'][frame_id]
            new_frame = frame.crop_image(rect = rect)
        else:
            new_frame = frame.crop_image()

    # 6. Conv with kernel
    if filter_method == 'conv_with_kernel':
        if 'ksize' in parameters_dict.keys():
            if len(parameters_dict['kernel']) > 1:
                kernel = parameters_dict['kernel'][frame_id]
            else:
                kernel = parameters_dict['kernel']
            if 'boundary' in parameters_dict.keys() and 'mode' in parameters_dict.keys():
                new_frame = frame.conv_with_kernel(kernel = kernel, boundary = parameters_dict['boundary'], mode = parameters_dict['mode'])
            else:
                new_frame = frame.conv_with_kernel(kernel = kernel)
        else:
            new_frame = frame.conv_with_kernel()

    # 7. X derivative
    if filter_method == 'x_diff':
        new_frame = frame.x_diff()

    # 8. Y derivative
    if filter_method == 'y_diff':
        new_frame = frame.y_diff()

    # 9. FFT of the entire video
    if filter_method == 'img_fft':
        new_frame = frame.img_fft()

    # 10. Filtering all frames
    if filter_method == 'filter_with_spectral_kernel':
        if 'spectral_kernel' in parameters_dict.keys():
            if len(parameters_dict['spectral_kernel']) > 1:
                spectral_kernel = parameters_dict['spectral_kernel'][frame_id]
            else:
                spectral_kernel = parameters_dict['spectral_kernel']
            new_frame = frame.filter_with_spectral_kernel(spectral_kernel = spectral_kernel)
        else:
            new_frame = frame.filter_with_spectral_kernel()

    # 11. Rotation of all frames
    if filter_method == 'rotate_image':
        if 'angle' in parameters_dict.keys():
            if type(parameters_dict['angle']) is int or type(parameters_dict['angle']) is float:
                angle = parameters_dict['angle']
            else:
                angle = parameters_dict['angle'][frame_id]
            new_frame = frame.rotate_image(angle = angle)
        else:
            new_frame = frame.rotate_image()

    return new_frame


def manipulate_video(self, filter_method = '', parameters_dict = None):
    new_frames = []
    new_grayscale_frames = []
    for frame_id in range(len(self.frames)):
        frame = self.frames[frame_id]
        frame = ImageClass(frame)        # Now frame is an ImageClass object

        new_frame = self._manipulate_img_by_method(frame, frame_id, filter_method = filter_method, parameters_dict = parameters_dict)
        grayscale_frame = new_frame.grayscale_image()

        new_frames += [ new_frame.img ]
        new_grayscale_frames += [ grayscale_frame.img ]

    from khen_hardware_libs.VideoClass.VideoClass import VideoClass
    return VideoClass(video_filename = new_frames, fps = self.fps, grayscale_frames = new_grayscale_frames)


def resize_video(self, width, height):
    return self.manipulate_video(filter_method='resize_image', parameters_dict={'width': width, 'height': height})

def digital_zoom(self, rect_list):
    return self.manipulate_video(filter_method='crop_image', parameters_dict={'rect': rect_list})

def rotate_video(self, angle):
    return self.manipulate_video(filter_method='rotate_image', parameters_dict={'angle': angle})