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



# ------------- data generation ---------------#
@staticmethod
def generate_event_file_from_grayscale_video(dvs_video, fps=30, threshold=1.15, filename="data/event_file.txt",
                                             override=False):
    print('Generate event txt file')
    if os.path.isfile(filename):
        print("\tThe event file is already exists.")
        if override:
            print("\tOverride.")
        else:
            print("\tDoes not override.")
            return

    # This method takes a grayscale set of frames & write a txt file of events
    # get grayscale video frames & generate a txt file of event stream
    # fps is the video fps

    old_frame = dvs_video[0]
    # threshold = threshold
    time_step = 1 / fps
    event_matrix = np.zeros_like(old_frame)
    t = 0.0

    # for progress bar:
    number_of_frames = len(dvs_video)
    frame_counter = 2
    # ----------------

    rows = np.shape(old_frame)[0]
    cols = np.shape(old_frame)[1]
    event_stream_list = list()
    for frame in dvs_video[1:]:
        log_tau = np.divide(np.log(np.divide(frame, old_frame + 1e-25) + 1e-25), np.log(threshold) + 1e-25)

        ON_mask = frame > old_frame
        OFF_mask = frame < old_frame
        event_matrix[ON_mask] = np.floor(log_tau[ON_mask])
        event_matrix[OFF_mask] = - np.ceil(log_tau[OFF_mask])

        # print(np.count_nonzero(event_matrix[ON_mask]))
        # print(np.count_nonzero(event_matrix[OFF_mask]))

        for i in np.arange(rows):
            for j in np.arange(cols):
                p = 0
                if 0 < event_matrix[i, j]:
                    if ON_mask[i, j]:
                        p = 1
                    elif OFF_mask[i, j]:
                        p = -1

                # p =  np.sign( event_matrix[i,j] )

                if p == 0:
                    continue

                event_stream_list.append([t, int(j), int(rows - i - 1), int(p), int(event_matrix[i, j])])

        t += time_step
        old_frame = frame

        # progress bar
        progressBar = "\r\t" + ProgressBar(Total=20, Progress=int(20 * (frame_counter) / number_of_frames),
                                           BarLength=20,
                                           ProgressIcon="|", BarIcon="-")
        ShowBar(progressBar)
        frame_counter += 1
        # -----------

    with open(filename, 'w') as f:
        f.write(np.array2string(np.array([cols, rows]))[1:-1])
        for event in event_stream_list:
            f.write("\n" + str(event).replace(",", "")[1:-1])
        f.close()

    print("\tGenerated", filename)
    return filename








@staticmethod
def from_bin_to_event_file(bin_file_path, txt_file_path, cols = 640, rows = 480, override = False, get_short_version = 0):
    # Coordinate system origin of the sensor is located in left upper corner
    # Output x, y should be in left down corner, then transoformation of Y coordinate is nessesary
    # get_short_version = 7000000 #TODO
    # cols, rows = (640,480)
    # filename = "recordings_left_down_corner"
    # filename = "recordings_time"
    # filename = "2"

    # filename = "data/"+folder_name+"/dvs.bin"
    # filename_txt = filename[:-3]+'txt'

    print('From bin file to event stream file')
    # print(filename[:-3]+'txt')
    if os.path.isfile(txt_file_path):
        print("\tThe event file is already exists.")
        if override:
            print("\tOverride.")
        else:
            print("\tDoes not override.")
            return


    if not(os.path.isfile(bin_file_path)):
        print('\tThe file does not exist: {}'.format(bin_file_path))
        return

    with open(bin_file_path, "rb") as f:
        buff = f.read()

    event_stream_list = list()

    timeStamp = 0.0
    posX = 0
    posY = 0
    polarity = 0

    # shortTs = 1/50  #50MHz

    shortTs = 1     # 1 micro-sec period due to Samsung DVS team
    longTs = 0.0    # 1 milli-sec period due to Samsung DVS team
    first_run_flag = True
    dataLen = len(buff)
    i = 0
    while (i<dataLen):
        header = buff[i] & 0x7C
        if (buff[i] & 0x80):    # Group Events Packet
            grpAddr = (buff[i+1] & 0xFC) >> 2
            if (buff[i + 3]):
                posY0 = grpAddr << 3
                polarity = buff[i+1] & 0x01
                for n in range(8):
                    if ((buff[i+3] >> n) & 0x01):
                        posY = posY0 + n
                        # ------------- Here event with +1 polarity -------------
                        # Now you have (PosX, posY, polarity, timeStamp) data for an event
                        # print(polarity)
                        if first_run_flag:
                            continue
                        posY = rows - 1 - posY
                        event_stream_list.append([timeStamp*1e-6, posX, posY, 1])
                        # exit()
                        # -------------------------------------------------------
            if (buff[i + 2]):
                grpAddr += (header >> 2)    #Offset
                posY0 = grpAddr << 3
                polarity = buff[i+1] & 0x02
                for n in range(8):
                    if ((buff[i+2] >> n) & 0x01):
                        posY = posY0 + n
                        # ------------- Here event with -1 polarity -------------
                        # Now you have (PosX, posY, polarity, timeStamp) data for an event
                        if first_run_flag:
                            continue
                        posY = rows - 1 - posY
                        event_stream_list.append([timeStamp*1e-6, posX, posY, -1])
                        # exit()
                        # -------------------------------------------------------
        else:
            if header == 0x04:
                if (buff[i+1] & 0x20):
                    timeStamp = longTs + shortTs
                    if longTs > 0:
                        first_run_flag = False      # Just to avoid starting garbage

                posX = (((buff[i + 2] & 0x03) << 8) | (buff[i + 3] & 0xFF))
            elif header == 0x08:
                longTs = (((buff[i + 1] & 0x3F) << 16) | ((buff[i + 2] & 0xFF) << 8) | (buff[i + 3] & 0xFF)) * 1000
                # print(longTs*1e-6)
            elif header == 0x40:
                continue
            elif header == 0x00:
                continue

        i += 4
        # Just for short checks:
        if get_short_version > 0:
            if i > get_short_version:
               break

    with open(txt_file_path, 'w') as f:
        f.write(np.array2string(np.array([cols, rows]))[1:-1])
        for event in event_stream_list:
            f.write("\n" + str(event).replace(",", "")[1:-1])
        f.close()
    print('\tCreated file: {}'.format(txt_file_path))

