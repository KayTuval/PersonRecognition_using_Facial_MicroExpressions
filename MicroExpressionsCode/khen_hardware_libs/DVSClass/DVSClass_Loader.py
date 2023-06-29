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


# ------------- From txt Files ---------------#
def load_list_of_events(self):
    print('Load list of events')

    # Assume the file is ordered in time
    # First row: cols, rows
    # Format: t, x, y, p, r

    if not os.path.isfile(self.filename):
        print("\t{} Does not exist!".format(self.filename))
        if os.path.isfile(self.filename[:-3]+'bin'):
            print('\tFound bin file, generating txt')
            self.from_bin_to_event_file(self.filename[:-3]+'bin',self.filename)
        else:
            print('exit')
            return

    file = open(self.filename, 'r')

    header = file.readline().rstrip().split(" ")
    self.cols = int(header[0])
    self.rows = int(header[1])

    first_run_flag = True
    time_offset = 0.0
    T = 0.0
    self.list_of_events = list()

    for event in file.readlines():
        if event.rstrip() == "":
            continue
        event_vec = event.rstrip().split(" ")
        r = 1
        if len(event_vec) == 5:
            r = event_vec[4]
        t, x, y, p = event_vec[:4]
        x = int(x)
        y = int(y)
        p = int(p)
        r = int(r)

        if first_run_flag:
            time_offset = np.float64(t)
            first_run_flag = False
        t = np.float64(t) - time_offset
        t = t + self.timestamp_events_offset    # To allign videos time
        self.list_of_events.append([t, x, y, p, r])

    file.close()
    print('\tNumber of events: {:,}'.format(len(self.list_of_events)))
    return self.list_of_events



# ------------- First Building Blocks ---------------#

def generate_video_from_event_stream(self, fps):
    from ImageClass.ImageClass import ImageClass

    print('Generate video frames')

    self.fps = fps
    # first_run_flag = True
    # time_offset = 0.0
    time_step = 1 / fps
    T = 0.0

    frame = np.zeros((self.rows, self.cols))
    frames_list = list()
    self.event_per_frame_list = list()
    event_id_list = list()
    event_id = 0
    last_time = self.list_of_events[-1][0]
    for event in self.list_of_events:
        t, x, y, p, r = event
        # # Time offset
        # if first_run_flag:
        #     time_offset = np.float64(t)
        #     # for progress bar
        #     last_time = float(self.list_of_events[-1][0] - time_offset)
        #     first_run_flag = False
        # # -----------

        # t = np.float64(t) - time_offset
        t = np.float64(t)
        # old:
        i = self.rows - y - 1       # When dealing with DVS true file
        # i = y   # Test!!# TODO!   # When dealing with DVS synthetic - fix!
        j = x

        # Coordinate system start in the down left corner, but opencv needs it left up:
        # i = y
        # j = x

        frame[i, j] += p * r
        event_id_list.append(event_id)

        # Resize scale of events - No!!
        # if (self.width != 0 and self.width != self.cols) or (self.height != 0 and self.height != self.rows):
        #     if self.resize_method == "scale":
        #         self.list_of_events[event_id][1] = round( max(min(x*self.width/self.cols,self.width-1),0) )
        #         self.list_of_events[event_id][2] = round( max(min(y*self.height/self.rows,self.height-1),0) )
        #     elif self.resize_method == "pad":
        #         self.list_of_events[event_id][1] += (self.width - self.cols) // 2
        #         self.list_of_events[event_id][2] += (self.height - self.rows) // 2
        # -----


        event_id += 1
        if round(T + time_step,10) <= round(t,10):   #TODO addition round, and =
            # if len(event_id_list) > 100:  # To avoid noisy effects
            # Resize scale of events:
            if (self.width != 0 and self.width != self.cols) or (self.height != 0 and self.height != self.rows):
                if self.resize_method == "scale":
                    frame = cv.resize(frame, (self.width, self.height), interpolation=cv.INTER_LINEAR)
                elif self.resize_method == "pad":
                    frame = np.pad(frame, ( ((self.height - self.rows) // 2, (self.height - self.rows + 1) // 2), \
                                            ((self.width - self.cols) // 2, (self.width - self.cols + 1) // 2)))
            # -----

            frames_list += [ frame.astype(np.float32) ]
            self.event_per_frame_list.append(event_id_list)
            event_id_list = list()

            # print(T)
            # frame = frame*0
            frame = np.zeros((self.rows, self.cols))
            T += time_step

            # progress bar
            progressBar = "\r\t" + ProgressBar(Total=20, Progress=int(20 * (T + time_step) / last_time), BarLength=20,
                                               ProgressIcon="|", BarIcon="-")
            ShowBar(progressBar)
            # -----------

    self.events_video = SynVideoClass(frames_list,fps = fps)
    return self.events_video

def get_high_fps_frame_sequence_from_event_list(self, event_list, high_fps):
    # print('Generate high fps video frames')

    self.high_fps = high_fps
    first_run_flag = True
    # time_offset = 0.0
    time_step = 1 / high_fps
    T = 0.0

    frame = np.zeros((self.rows, self.cols))
    high_fps_frames_list = list()
    high_fps_event_per_frame_list = list()
    event_id_list = list()
    for event_i in event_list:
        t, x, y, p, r = self.list_of_events[event_i]
        # Time offset
        if first_run_flag:
            time_offset = np.float64(t)
            first_run_flag = False
        # -----------

        t = np.float64(t) - time_offset + self.timestamp_events_offset

        # t = np.float64(t)
        # old:
        i = self.rows - y - 1
        j = int(x)
        # Coordinate system start in the down left corner, but opencv needs it left up:
        # i = y
        # j = x
        # print('t')
        # print(event_i)
        # print(self.cols)
        # print(t)
        frame[i, j] += p * r
        event_id_list += [event_i]

        # # Resize scale of events:
        # if (self.width != 0 and self.width != self.cols) or (self.height != 0 and self.height != self.rows):
        #     if self.resize_method == "scale":
        #         self.list_of_events[event_id][1] = round( max(min(x*self.width/self.cols,self.width-1),0) )
        #         self.list_of_events[event_id][2] = round( max(min(y*self.height/self.rows,self.height-1),0) )
        #     elif self.resize_method == "pad":
        #         self.list_of_events[event_id][1] += (self.width - self.cols) // 2
        #         self.list_of_events[event_id][2] += (self.height - self.rows) // 2
        # # -----

        if T + time_step < t:
            # Resize scale of events:
            if (self.width != 0 and self.width != self.cols) or (self.height != 0 and self.height != self.rows):
                if self.resize_method == "scale":
                    frame = cv.resize(frame, (self.width, self.height), interpolation=cv.INTER_LINEAR)
                elif self.resize_method == "pad":
                    frame = np.pad(frame, ( ((self.height - self.rows) // 2, (self.height - self.rows + 1) // 2), \
                                            ((self.width - self.cols) // 2, (self.width - self.cols + 1) // 2)))
            # -----

            high_fps_frames_list += [frame]
            high_fps_event_per_frame_list += [event_id_list]
            event_id_list = list()

            # print(T)
            # frame = frame*0
            frame = np.zeros((self.rows, self.cols))
            T += time_step

            # progress bar
            # progressBar = "\r\t" + ProgressBar(Total=20, Progress=int(20 * (T + time_step) / last_time), BarLength=20,
            #                                    ProgressIcon="|", BarIcon="-")
            # ShowBar(progressBar)
            # -----------


    return SynVideoClass(high_fps_frames_list,fps = high_fps), high_fps_event_per_frame_list




# def generate_neighborhood_list(self, pixel_dist=3, time_dist=1000, rect_crop=False):
#     print('Load list of neighborhoods')
#     if rect_crop and len(self.crop_rect_list) == 0:
#         print("\tNo crop rect. Load neighborhoods for all events.")
#         rect_crop = False
#
#     # Get list of events in the format: t, x, y, p
#     # And distance to define a neighborhood r => (2pixel_dist+1)x(2pixel_dist+1)
#     # Assuming sorted time events
#     # An event is a neighbor of its own
#
#     self.pixel_dist = pixel_dist
#     # Init the neighborhoods to be 0 everywhere before filling
#     self.neighborhood_id = [0] * len(self.list_of_events)
#     self.neighborhood_time = [0] * len(self.list_of_events)
#
#     pos_timing_matrix_t = np.zeros((self.rows, self.cols))
#     pos_timing_matrix_id = -np.ones((self.rows, self.cols))
#     neg_timing_matrix_t = np.zeros((self.rows, self.cols))
#     neg_timing_matrix_id = -np.ones((self.rows, self.cols))
#
#     if rect_crop:
#         show_id_list = self.cropped_list_of_events_id
#     else:
#         show_id_list = range(len(self.list_of_events))
#     event_counter = 0
#     number_of_events = len(show_id_list)
#
#     for event_id in show_id_list:
#         event_t, event_x, event_y, event_p, __ = self.list_of_events[event_id]
#         i_vec = self.rows - event_y - 1 + np.arange(-pixel_dist, pixel_dist + 1)
#         j_vec = event_x + np.arange(-pixel_dist, pixel_dist + 1)
#         i_vec = i_vec[np.where(0 <= i_vec) and np.where(i_vec < self.rows)]
#         j_vec = j_vec[np.where(0 <= j_vec) and np.where(j_vec < self.cols)]
#
#         # Update the last time step & id in this pixel
#         if event_p == 1:
#             pos_timing_matrix_t[self.rows - event_y - 1, event_x] = event_t
#             pos_timing_matrix_id[self.rows - event_y - 1, event_x] = event_id
#             id_neighbors = pos_timing_matrix_id[i_vec, :][:, j_vec]
#             timing_neighbors = pos_timing_matrix_t[i_vec, :][:, j_vec]
#         elif event_p == -1:
#             neg_timing_matrix_t[self.rows - event_y - 1, event_x] = event_t
#             neg_timing_matrix_id[self.rows - event_y - 1, event_x] = event_id
#             id_neighbors = neg_timing_matrix_id[i_vec, :][:, j_vec]  # neg_timing_matrix_id[i_vec, j_vec]
#             timing_neighbors = neg_timing_matrix_t[i_vec, :][:, j_vec]
#
#         self.neighborhood_id[event_id] = id_neighbors
#         self.neighborhood_time[event_id] = timing_neighbors
#         # self.time_surface_list
#         # time_surface = self.get_time_surface(event_id, ts_decay_factor=ts_decay_factor, kernel=kernel)
#         # self.neighborhood_id.append(id_neighbors)
#         # self.neighborhood_time.append(timing_neighbors)
#
#         # progress bar
#         progressBar = "\r\t" + ProgressBar(Total=20, Progress=int(20 * (event_counter + 1) / number_of_events),
#                                            BarLength=20,
#                                            ProgressIcon="|", BarIcon="-")
#         ShowBar(progressBar)
#         event_counter += 1
#         # -----------
#
#     return self.neighborhood_id


# def run_video(self, display_fps=None, only_rect_crop=False, display_only_active=False, width = 640, height = 480):
#     if len(self.frames_list) == 0:
#         print('Run generate_video_from_event_stream with fps before!')
#         return
#         # self.generate_video_from_event_stream(fps=fps)
#     if only_rect_crop and len(self.crop_rect_list) == 0:
#         print("\tNo crop rect. Show full frame video.")
#         only_rect_crop = False
#
#     cv.namedWindow('Frame', cv.WINDOW_NORMAL)
#     cv.resizeWindow('Frame', width, height)
#
#     if display_fps == None:
#         frame_time_delay = 1000/self.fps
#     elif display_fps == 0:
#         frame_time_delay = 0
#     else:
#         frame_time_delay = 1000/display_fps
#
#     for frame in self.frames_list:
#         cv.imshow('Frame', 0.5 * np.clip(frame, -1, 1) + 0.5)
#         # Press Q on keyboard to  exit
#         if cv.waitKey(int(frame_time_delay)) & 0xFF == ord('q'):
#             cv.destroyAllWindows()
#             break
#
#     # For crop & active areas:
#     # for frame_id in range(len(self.frames_list)):
#     #     frame = self.frames_list[frame_id]
#     #     frame = np.dstack((frame, frame, frame))
#     #     if len(self.active_areas_list) != 0:
#     #         active_area = self.active_areas_list[frame_id]
#     #         frame[:, :, 1] = frame[:, :, 1] + active_area
#     #         if display_only_active:
#     #             frame[:, :, 0] = 0
#     #             # frame[:,:,1] = active_area
#     #             frame[:, :, 2] = 0
#     #
#     #     if len(self.crop_rect_list) != 0:
#     #         crop_rect = self.crop_rect_list[frame_id]
#     #         if only_rect_crop:
#     #             # Show only face
#     #             # frame = self.frames_list[frame_id]
#     #             # frame = frame[self.rows - 1 - crop_rect[3]:self.rows - 1 - crop_rect[1], crop_rect[0]:crop_rect[2]]
#     #             frame = frame[crop_rect[1]:crop_rect[3], crop_rect[0]:crop_rect[2]]
#     #         else:
#     #             # Show entire frame with rect on face
#     #             cv.rectangle(frame, (crop_rect[0], crop_rect[1]), (crop_rect[2], crop_rect[3]), (0, 0, 255), 2)
#     #
#     #     cv.imshow('Frame', 0.5 * np.clip(frame, -1, 1) + 0.5)
#     #
#     #     # Press Q on keyboard to  exit
#     #     if cv.waitKey(int(1000/self.fps)) & 0xFF == ord('q'):
#     #         break
#
#     cv.destroyAllWindows()
#     return



# def load_cropped_list_of_events_id(self):
#     if len(self.crop_rect_list) == 0:
#         print('Crop before!')
#         return
#         # self.load_crop_rect_list()
#
#     print('Get cropped list of event')
#
#     self.cropped_list_of_events_id = list()
#     number_of_frames = len(self.frames_list)
#     # Run over all frames and then for each frame over all its events
#     for frame_id in range(number_of_frames):
#         min_x, min_y, max_x, max_y = self.crop_rect_list[frame_id]
#         events_per_frame = self.event_per_frame_list[frame_id]
#         for event_id in events_per_frame:
#             event = self.list_of_events[event_id]
#             __, event_x, event_y, __, __ = event
#             if (min_x < event_x) and (event_x < max_x) and (min_y < event_y) and (event_y < max_y):
#                 self.cropped_list_of_events_id.append(event_id)
#
#     print('\tNumber of cropped events:', len(self.cropped_list_of_events_id))
#     return self.cropped_list_of_events_id
#
#
# def load_crop_rect_list(self, rect_file=None):
#     if len(self.frames_list) == 0:
#         self.generate_video_from_event_stream()
#
#     print('Load crop rect list')
#     # This function based on Face Detection generated txt file.
#     # crop_rect_list is a list of rect of the interest area (face) for each frame
#     self.crop_rect_list = list()
#
#     extra_border = 30  # pixels
#     file = open(rect_file, 'r')
#
#     for line in file.readlines():
#         if len(line) <= 4:
#             continue
#         x_min, y_min, x_max, y_max = line.strip().replace(" ", "").split(",")
#         self.crop_rect_list.append([int(x_min) - extra_border, int(y_min) - extra_border, int(x_max) + extra_border,
#                                     int(y_max) + extra_border])
#
#     file.close()
#
#     return self.crop_rect_list
