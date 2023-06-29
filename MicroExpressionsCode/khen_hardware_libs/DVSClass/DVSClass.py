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
from khen_hardware_libs.VideoClass.VideoClass import SynVideoClass

import khen_hardware_libs.DVSClass.DVSClass_Loader as Loader
import khen_hardware_libs.DVSClass.DVSClass_Grayscale as Grayscale
import khen_hardware_libs.DVSClass.DVSClass_DataGenerator as DataGenerator


def save_events_list_to_txt_file(txt_file_path, event_stream_list, cols, rows):
    with open(txt_file_path, 'w') as f:
        f.write(np.array2string(np.array([cols, rows]))[1:-1])
        for event in event_stream_list:
            f.write("\n" + str(event).replace(",", "")[1:-1])
        f.close()
    print('Created file: {}'.format(txt_file_path))
    return




class DVSClass:
    """
    A class used to represent a DVS events stream
    ...

    Attributes
    ----------
    filename : str
        The filename of the events stream (a txt file)

    resize_method : str
        Represents the resize frame method if used (mainly not used): 'pad' for padding and 'scale' for linear scaling

    timestamp_events_offset : float
        Enable sub-frame time shift for the events stream

    width : int
        The width of the frames

    height : int
        The height of the frames

    list_of_events : list
        List of tuples for each event (t,x,y,p,r) while t is the time step, x and y are coordinates, p is the
        polarity = -1 or +1 and r is a counter (mainly 1) which is different than 1 only for synthetic event streams.

    events_video : SynVideoClass
        A video for events-frame

    Methods
    -------
    __init__(filename = None, width = 0, height = 0, resize_method = "pad", timestamp_events_offset = 0, pixel_dist = 3)
        Gets a txt file for example: 'dvs.txt'. If only bin file is exist, the algorithm creates a txt file from the bin.

    staticmethod: generate_event_file_from_grayscale_video(dvs_video, fps=30, threshold=1.15, filename="data/event_file.txt",
                                             override=False)
        Takes a standard video (as .avi or .mp4) and extract from it a synthetic events stream (old implementaion)

    staticmethod: from_bin_to_event_file(bin_file_path, txt_file_path, cols = 640, rows = 480, override = False, get_short_version = 0))
        Reads a bin file (originally implemented in C) and convert it to a txt file

    load_list_of_events()
        Run over the txt file of events and create a list of events (of tuples (t,x,y,p,r) )

    generate_video_from_event_stream(fps)
        Generate synthetic events-frames from the events list in fps frame-per-second rate

    get_high_fps_frame_sequence_from_event_list(event_list, high_fps))
        Using a sub-set of the total events, the event_list, and create from it events-frames at high_fps rate.
        It is mainly used when one wants to extract additional frames between two standard frames.

    generate_grayscale_video_from_event_stream(fps=1000, dynamic_range=15, threshold=5))
        Uses the events list and try to generate a grayscale frames with the naive integral way.

    """

    events_video = SynVideoClass    # For the interpreter

    def __init__(self, filename = None, width = 0, height = 0, resize_method = "pad", timestamp_events_offset = 0, pixel_dist = 3):
        if filename is None:
            return

        self.width = width
        self.height = height
        self.filename = filename
        self.resize_method = resize_method
        self.timestamp_events_offset = timestamp_events_offset
        # self.generate_neighborhood_list(pixel_dist = pixel_dist)

        self.load_list_of_events()
        return

    def __del__(self):

        return


    #----------------- Static Methods -----------------#
    generate_event_file_from_grayscale_video = DataGenerator.generate_event_file_from_grayscale_video
    from_bin_to_event_file = DataGenerator.from_bin_to_event_file

    # ----------------- Load data & Frames Generator -----------------#
    load_list_of_events = Loader.load_list_of_events
    generate_video_from_event_stream = Loader.generate_video_from_event_stream
    get_high_fps_frame_sequence_from_event_list = Loader.get_high_fps_frame_sequence_from_event_list

    # ----------------- Grayscale Video -----------------#
    generate_grayscale_video_from_event_stream = Grayscale.generate_grayscale_video_from_event_stream




# if __name__ == "__main__":

    # print('DVS test')

    # if 0:
    #     testVideo = VideoClass('data/self3.MOV', 0)
    #     generated_event_file = "data/generated/self3.txt"
    #     DVSVideoClass.generate_event_file_from_grayscale_video(testVideo.grayscale_frames, fps=testVideo.fps, threshold=1.2,
    #                                                            filename=generated_event_file, override=False)
    #     testDVS = DVSVideoClass(generated_event_file)
    #     testDVS.generate_video_from_event_stream(fps=testVideo.fps)
    #     testDVS.load_crop_rect_list('data/self3_face_crop.txt')
    #     testDVS.run_video(only_rect_crop=True)
    # elif 0:
    #     testVideo = VideoClass('data/alona1.mp4', 0)
    #     generated_event_file = "data/generated/alona1.txt"
    #     DVSVideoClass.generate_event_file_from_grayscale_video(testVideo.grayscale_frames, fps=testVideo.fps, threshold=1.2,
    #                                                            filename=generated_event_file, override=False)
    #     testDVS = DVSVideoClass(generated_event_file)
    #     testDVS.generate_video_from_event_stream(fps=testVideo.fps)
    #     testDVS.load_crop_rect_list('data/alona1_face_crop.txt')
    #     testDVS.generate_neighborhood_list(pixel_dist=3)
    #     # testDVS.generate_neighborhood_list(pixel_dist=3, rect_crop=True)
    #     # testDVS.run_video(only_rect_crop=False)
    #     testDVS.generate_active_area(patch_size=7, threshold=30)
    #     testDVS.run_video(only_rect_crop=True, display_only_active=True)
    #     # testDVS.get_optical_flow_per_event(event_id=1000005, show=1)
    #     # testDVS.get_optical_flow_per_event(event_id=1000005, show=1, method='linear')
    # else:
    #     testDVS = DVSVideoClass("data/popping_air_balloon.txt")
    #     testDVS.generate_video_from_event_stream(fps=30)
    #     testDVS.generate_short_time_frames(short_fps=30)
    #     testDVS.generate_segmantation_per_short_frame()
    #     testDVS.run_segmanted_video()
        # Semantic segmantation:

        #testDVS.get_optical_flow_per_event(event_id=1000002, show=1, epsilon = 0.1, grad_threshold = 0)
        #testDVS.get_optical_flow_per_event(event_id=1000002, show=1, method='linear', epsilon = 0.1, grad_threshold = 0)


    # testDVS.run_short_time_frames_video()
    # testDVS.get_optical_flow_per_event_by_short_time_rep(5)
    # testDVS.generate_neighborhood_list(pixel_dist=3)
    # testDVS.generate_active_area(patch_size=20, threshold=100)
    # testDVS.run_video()

    # print( testDVS.get_optical_flow_per_event(event_id=1000000, show=1) )
    # testDVS.run_optical_flow_video()
