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

# from Scripts.Scripts import *

# Save video to .avi file
def save_video(save_filename, frames, fps):
    display_height = np.shape(frames[0])[0]
    display_width = np.shape(frames[0])[1]
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    # fourcc = cv.VideoWriter_fourcc(*'MPEG')
    out = cv.VideoWriter(save_filename+".avi", fourcc, fps, (display_width, display_height))#, 0)
    for frame in frames:
        # Problem - fix! only grayscale work :(
        if len(frame.shape) == 2:
            frame = np.concatenate( (frame[:,:,np.newaxis],frame[:,:,np.newaxis],frame[:,:,np.newaxis]), axis = 2)
        # if len(frame.shape) == 3:
        #     frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        out.write(frame)
    out.release()
    print('Created file:',save_filename+".avi")
    return

# # Save video to .avi file
# def save_video(save_filename, frames, fps, format = 'rgb'):
#     if len(frames) == 0:
#         return
#     display_height = np.shape(frames[0])[0]
#     display_width = np.shape(frames[0])[1]
#     # fourcc = cv.VideoWriter_fourcc(*'XVID')
#     # fourcc = cv.VideoWriter_fourcc(*'MJPG')
#     fourcc = cv.VideoWriter_fourcc(*'MPEG')
#     # fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
#     # if len(frames[0].shape) == 3 and format == 'rgb': # RGB:
#     #     out = cv.VideoWriter(save_filename+".avi", fourcc, fps, (display_width, display_height,3))
#     # else:
#     out = cv.VideoWriter(save_filename+".avi", fourcc, fps, (display_width, display_height))
#     for frame in frames:
#         # Problem - fix! only grayscale work :(
#         if format == 'gray' and len(frame.shape) == 3:
#             frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
#         out.write(frame)
#     out.release()
#     print('Created file:',save_filename+".avi")
#     return


def save_imgs_in_folder(images_list, folder_path, preface = ""):
    img_counter = 1
    for img in images_list:
        img_name = folder_path + preface +str(img_counter)+'.png'
        cv.imwrite(img_name, img)
        img_counter += 1
    print('Saved {} images in folder: {}'.format(img_counter, folder_path))
    return

def load_imgs_from_folder(folder_path, format = 'rgb', preface = ""):
    images_list = {}
    for filename in os.listdir(folder_path):
        if not filename.startswith(preface):
            continue
        img_name = folder_path + filename
        if format == 'gray':
            img_obj = cv.imread(img_name,cv.IMREAD_GRAYSCALE)
        else:
            img_obj = cv.imread(img_name)
        # print(filename[:-4])
        # images_list[int(filename[:-4])] = img_obj
        images_list[filename[:-4]] = img_obj
    return images_list


# Display video
def run_video(frame_list, fps=None, width = 640, height = 480):
    cv.namedWindow('Frame', cv.WINDOW_NORMAL)
    cv.resizeWindow('Frame', width, height)
    if fps == None or fps == 0:
        delay_time = 0
    else:
        delay_time = 1000/fps
    for frame in frame_list:
        cv.imshow('Frame', frame)
        # Press Q on keyboard to  exit
        if cv.waitKey(int(delay_time)) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            break
    cv.destroyAllWindows()
    return


# Create new resized png images in entire folder with '_resized.png' addition
def resize_imges_in_folder(folder_path, width, height):
    # path = "C:/Users/khen/Documents/MATLAB/camera/R/"
    # path = "C:/Users/khen/Documents/MATLAB/camera/stereo/R/"
    # path = "C:/Users/khen/PycharmProjects/DVS_proj/stereo/stereoSyntheticData/lowscale/"
    for filename in os.listdir(folder_path):
        if filename.endswith('_resized.png'):
            continue
        img = cv.imread(folder_path+filename)
        # cv.imshow('test',img)
        # cv.waitKey(0)
        # img = cv.resize(img, (640//2,480//2), interpolation=cv.INTER_AREA)
        img = cv.resize(img, (width,height), interpolation=cv.INTER_AREA)
        img_name = folder_path + str(filename)[:-4] + '_resized.png'
        cv.imwrite(img_name, img)
        print('Img:',img_name)



def from_imgs_in_folder_to_avi_video(folder_name, video_name, fps):
    video = []
    for filename in os.listdir(folder_name):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            frame = cv.imread(folder_name+'/'+filename)
            video += [frame]

    save_video(video,save_filename = video_name, fps = fps)
    return video




def get_optical_flow_for_points_dense(old_frame, new_frame, p0=""):
    if p0 == "":
        # params for ShiTomasi corner detection
        feature_params = dict(maxCorners=1000,
                              qualityLevel=0.3,
                              minDistance=7,
                              blockSize=7)
        p0 = cv.goodFeaturesToTrack(old_frame, mask=None, **feature_params)
    p0 = np.int16( np.round( p0 ) )

    p0[:,0,0] = np.clip( p0[:,0,0], 0, new_frame.shape[1]-1 )
    p0[:,0,1] = np.clip( p0[:,0,1], 0, new_frame.shape[0]-1 )

    # calculate optical flow
    flow = cv.calcOpticalFlowFarneback(old_frame, new_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    delta_p0 = flow[ [p0[:,0,1], p0[:,0,0]] ]

    delta_p0 = delta_p0.reshape(-1, 1, 2)
    return delta_p0, None, p0, None


def get_good_points_for_optical_flow(frame, maxCorners=1000, qualityLevel=0.3, minDistance=7, blockSize=7):
    # params for ShiTomasi corner detection
    feature_params = dict(maxCorners=maxCorners,
                          qualityLevel=qualityLevel,
                          minDistance=minDistance,
                          blockSize=blockSize)
    p0 = cv.goodFeaturesToTrack(frame, mask=None, **feature_params)
    return p0


def get_optical_flow_for_points(old_frame, new_frame, p0="", winSize = (15,15), minErrorThreshold = 1):
    # Take first frame and find corners in it
    if p0 == "":
        p0 = get_good_points_for_optical_flow(old_frame, maxCorners=5000, qualityLevel=0.3, minDistance=7, blockSize=7)
        # p0 = get_good_points_for_optical_flow(old_frame, maxCorners=10000, qualityLevel=0.1, minDistance=3, blockSize=7)

    # Parameters for lucas kanade optical flow
    lk_params = dict(winSize=winSize,
                     maxLevel=4,#4,#4
                     criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 1000, 0.001))

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_frame, new_frame, p0, None, **lk_params)

    # Select good points
    err[st == 0] = np.inf
    st[np.abs(err) > minErrorThreshold] = 0

    # good_new = p1[st == 1][err[st==1] < minErrorThreshold]  # Filtering error under 1, TODO arbitrarily
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    # print(minErrorThreshold, err[st==1].max())

    delta_p0 = good_new - good_old

    p0_old = good_old.reshape(-1, 1, 2)
    p0 = good_new.reshape(-1, 1, 2)
    delta_p0 = delta_p0.reshape(-1, 1, 2)
    return delta_p0, p0, p0_old, st



# Get the time shift between two videos using corellation-peak method
def get_time_shift_video(video1, video2, fps, show=0):
    # Find videos related shift
    # need to add idx_shift to video2 or sub from video1
    conv_videos = signal.fftconvolve(video1, video2[-1::-1,:,:],axes=0,mode='full')
    time_correlation = np.sum(np.abs(conv_videos),axis=(1,2))
    idx_shift = ( len(video2) - 1 - np.argmax( time_correlation ))
    time_shift = idx_shift/fps
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
        new_video1 = video1[-idx_shift:max_frames_len-idx_shift] if idx_shift < 0 else video1[:max_frames_len]
        new_video2 = video2[:max_frames_len] if idx_shift < 0 else video2[idx_shift:max_frames_len+idx_shift]
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

    return (idx_shift, time_shift)



def find_idx_shift_manually(video1, video2):
    # print(left_video[0].shape, right_video[0].shape)
    # print(len(left_video),len(right_video))
    frame1 = 0;    frame2 = 0
    cv.namedWindow('img', cv.WINDOW_NORMAL)
    cv.resizeWindow('img', 1600, 600)
    while (True):
        if frame1 < 0:
            frame1 = 0
        if frame2 < 0:
            frame2 = 0
        if frame1 >= len(video1):
            frame1 = len(video1)-1
        if frame2 >= len(video2):
            frame2 = len(video2)-1

        print('frame1 =',frame1,'frame2 =',frame2)
        print('idx_shift =',frame2-frame1)

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

    idx_shift = frame2-frame1

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

