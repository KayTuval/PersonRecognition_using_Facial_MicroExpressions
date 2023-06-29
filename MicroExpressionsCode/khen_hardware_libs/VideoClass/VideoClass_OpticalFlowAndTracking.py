__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import cv2 as cv
import numpy as np
from khen_hardware_libs.Scripts.Scripts import show_optical_flow_on_img
# from VideoClass.videoClass import *
from khen_hardware_libs.ImageClass.ImageClass import *


def get_optical_flow_for_video(self, start_frame=0, end_frame=np.inf, method='LK', p0=None, show=1):
    start_frame = max(start_frame, 0)
    end_frame = min(end_frame, len(self.frames) - 1)

    if p0 is None:
        p0 = ImageClass( self.grayscale_frames[start_frame] ).get_good_features()

    for frame_id in range(start_frame, end_frame):
        frame = ImageClass( self.grayscale_frames[frame_id] )
        next_frame = ImageClass( self.grayscale_frames[frame_id + 1] )
        p0, delta_p0, p1, st = frame.get_optical_flow_from_me_to_other(next_frame, method=method, p0=p0)
        if show:
            show_optical_flow_on_img(p0, delta_p0, frame.img, show=1)

    return



def follow_points_in_video(self, init_tracking_points = None, show = 1):
    # FIXME: Very clumsy implementation here

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=(15, 15),
                     maxLevel=2,
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

    # init_tracking_points in format: (points X coordinates) = (p,2)

    # p0 shape: (10, 1, 2)
    if init_tracking_points == None:
        init_tracking_points = self.grayscale_frames[0].get_good_features()

    if (show == 1):
        for interestPoint in self.intertPoints:
            # print((interestPoint[0][0],interestPoint[0][1]))
            cv.circle(self.frames[0], (int(interestPoint[0][0]), int(interestPoint[0][1])), 1, (0, 0, 200), 10)

    self.point_and_step_per_frame = list()
    old_gray = self.grayscale_frames[0]

    # Create a mask image for drawing purposes
    mask = np.zeros_like(self.frames[0])

    # Create some random colors
    color = np.random.randint(0, 255, (100, 3))
    frame_counter = 0
    p0 = init_tracking_points
    for frame, frame_gray in zip(self.frames[1:], self.grayscale_frames[1:]):

        # print(p0)
        # calculate optical flow
        # p0
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        dx_dy = good_new - good_old

        self.point_and_step_per_frame.append( np.concatenate((good_old, dx_dy), axis=1) )

        if show == 1:
            # draw the tracks
            for i, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel()
                c, d = old.ravel()
                mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
                frame = cv.circle(frame, (a, b), 5, color[i].tolist(), -1)
            img = cv.add(frame, mask)

            cv.namedWindow('frame', cv.WINDOW_NORMAL)
            cv.resizeWindow('frame', 600, 800)
            cv.imshow('frame', img)
            k = cv.waitKey(30) & 0xff
            if k == 27:
                break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)
    return