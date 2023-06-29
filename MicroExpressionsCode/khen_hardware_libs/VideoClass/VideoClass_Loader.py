__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import cv2 as cv
import argparse
import os.path
import numpy as np
from khen_hardware_libs.ImageClass.ImageClass import *


def load_video_manually(self, frames, grayscale_frames, fps):
    self.frames = frames
    self.grayscale_frames = grayscale_frames
    self.fps = fps
    self.height = self.frames[0].shape[0]
    self.width = self.frames[0].shape[1]
    return


def load_video_from_folder_imgs(self, foldername, fps):
    # This function gets a folder name & generate a video file in this folder
    self.fps = fps
    self.frames = []
    self.grayscale_frames = []

    for filename in os.listdir(foldername):
        frame = ImageClass(foldername + '/' + filename)
        self.frames += [frame.img]
        self.grayscale_frames += [frame.grayscale_image().img]

    self.height = self.frames[0].shape[0]
    self.width = self.frames[0].shape[1]
    return


def load_video_from_file(self, video_filename):
    cap = cv.VideoCapture(video_filename)

    if (cap.isOpened() == False):
        print("Error opening video file:", video_filename)
        return

    self.fps = cap.get(5)  # fps
    self.width = int(cap.get(3))  # width
    self.height = int(cap.get(4))  # height
    self.frames = []
    self.grayscale_frames = []

    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        frame = ImageClass(frame)
        if ret == True:
            self.frames += [frame.img]
            self.grayscale_frames += [frame.grayscale_image().img]
        else:
            break
    cap.release()

    return


def load_video(self, video_filename, fps = None, grayscale_frames = None):
    # There are three ways to load a video: (1) Manually (2) Via .avi / .MOV / .mp4 /... file (3) Via folder of images

    if type(video_filename) != str:
        self.load_video_manually(frames = video_filename, grayscale_frames = grayscale_frames, fps = fps)
        return

    if os.path.isdir(video_filename):
        self.load_video_from_folder_imgs(foldername = video_filename, fps = fps)
        return

    self.load_video_from_file(video_filename = video_filename)
    return