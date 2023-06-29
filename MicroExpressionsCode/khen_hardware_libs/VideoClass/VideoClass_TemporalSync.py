__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import cv2 as cv
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt



# Get the time shift between two videos using corellation-peak method
def get_time_shift_video(self, other, show=0):
    # Find videos related shift
    video1 = np.array([f for f in self.frames])
    video2 = np.array([f for f in other.frames])

    # need to add idx_shift to video2 or sub from video1
    conv_videos = signal.fftconvolve(video1, video2[-1::-1, :, :], axes=0, mode='full')
    time_correlation = np.sum(np.abs(conv_videos), axis=(1, 2))
    idx_shift = (len(video2) - 1 - np.argmax(time_correlation))
    time_shift = idx_shift / self.fps
    if show:
        plt.plot(time_correlation)
        plt.show()
        plt.pause(0)

        cv.namedWindow('video1', cv.WINDOW_NORMAL)
        cv.resizeWindow('video1', 640, 480)
        cv.namedWindow('video2', cv.WINDOW_NORMAL)
        cv.resizeWindow('video2', 640, 480)

        # max_frames_len = min(len(video1), len(video2), len(video1) - abs(idx_shift), len(video2) - abs(idx_shift))
        max_frames_len = min(len(video1) - abs(idx_shift), len(video2) - abs(idx_shift))
        new_video1 = video1[-idx_shift:max_frames_len - idx_shift] if idx_shift < 0 else video1[:max_frames_len]
        new_video2 = video2[:max_frames_len] if idx_shift < 0 else video2[idx_shift:max_frames_len + idx_shift]
        # print(len(new_video1))
        # print(len(new_video2))
        # print(idx_shift)
        # print(max_frames_len)
        for i in range(max_frames_len):
            # if idx_shift < 0:
            #     frame1 = video1[i - idx_shift]
            #     frame2 = video2[i]
            # else:
            #     frame1 = video1[i]
            #     frame2 = video2[i + idx_shift]

            cv.imshow('video1', new_video1[i])
            cv.waitKey(0)
            cv.imshow('video2', new_video2[i])
            cv.waitKey(0)
            if i > 30:
                break
        cv.destroyAllWindows()
        print( (idx_shift, time_shift) )

    return (idx_shift, time_shift)

def find_idx_shift_manually(self, other):
    video1 = np.array([f for f in self.frames])
    video2 = np.array([f for f in other.frames])

    # print(left_video[0].shape, right_video[0].shape)
    # print(len(left_video),len(right_video))
    frame1 = 0;
    frame2 = 0
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.resizeWindow('img', 1600, 600)
    while (True):
        if frame1 < 0:
            frame1 = 0
        if frame2 < 0:
            frame2 = 0
        if frame1 >= len(video1):
            frame1 = len(video1) - 1
        if frame2 >= len(video2):
            frame2 = len(video2) - 1

        print('frame1 =', frame1, 'frame2 =', frame2)
        print('idx_shift =', frame2 - frame1)

        cv.imshow('img', np.hstack((video1[frame1].astype(np.uint8), video2[frame2].astype(np.uint8))))
        q = cv.waitKey(0)
        if q == 27:
            break
        elif q == ord('w'):  # up
            frame1 += 1
        elif q == ord('s'):  # down
            frame1 -= 1
        elif q == ord('a'):  # left
            frame2 -= 1
        elif q == ord('d'):  # right
            frame2 += 1

    idx_shift = frame2 - frame1

    cv.namedWindow('video1', cv.WINDOW_NORMAL)
    cv.resizeWindow('video1', 800, 600)
    cv.namedWindow('video2', cv.WINDOW_NORMAL)
    cv.resizeWindow('video2', 800, 600)
    max_frames_len = min(len(video1) - abs(idx_shift), len(video2) - abs(idx_shift))
    new_video1 = video1[-idx_shift:max_frames_len - idx_shift] if idx_shift < 0 else video1[:max_frames_len]
    new_video2 = video2[:max_frames_len] if idx_shift < 0 else video2[idx_shift:max_frames_len + idx_shift]
    for i in range(max_frames_len):
        cv.imshow('video1', new_video1[i])
        cv.waitKey(0)
        cv.imshow('video2', new_video2[i])
        cv.waitKey(0)

    return idx_shift