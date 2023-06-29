__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import cv2 as cv
import numpy as np
import argparse
import os.path
import numpy as np
import matplotlib.pyplot as plt
from khen_hardware_libs.Scripts.Scripts import *
from khen_hardware_libs.VideoClass.VideoClass import *


# ------------- Grayscale Video ---------------#
def generate_grayscale_video_from_event_stream(self, fps=1000, dynamic_range=15, threshold=5):
    print('Generate grayscale video frames')

    if len(self.grayscale_frames_list) == 0:
        self.generate_video_from_event_stream(fps = fps)

    self.dynamic_range = dynamic_range  # DB
    max_intensity_value = 10 ** (dynamic_range / 10)
    self.threshold = threshold

    grayscale_frame = np.ones((self.rows, self.cols))
    grayscale_frames_list = list()

    # for progress bar:
    number_of_frames = len(self.frames_list)
    frame_counter = 1
    # ----------------

    # frames are between 0 and 1
    for frame in self.frames_list:
        grayscale_frame = np.multiply(grayscale_frame, np.power(threshold, frame))
        grayscale_frames_list += [ np.clip(grayscale_frame, 0, max_intensity_value) / max_intensity_value ]
        # frame *= 0.999

        # progress bar
        progressBar = "\r\t" + ProgressBar(Total=20, Progress=int(20 * (frame_counter) / number_of_frames),
                                           BarLength=20,
                                           ProgressIcon="|", BarIcon="-")
        ShowBar(progressBar)
        frame_counter += 1
        # -----------

    self.grayscale_video = VideoClass(grayscale_frames_list, fps = fps)
    return self.grayscale_video

