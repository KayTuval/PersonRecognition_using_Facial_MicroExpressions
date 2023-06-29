__author__ = 'Khen Cohen'
__credits__ = ['Khen Cohen']
__email__ = 'khencohen@mail.tau.ac.il'
__date__ = '5.10.2021'

import cv2 as cv
import numpy as np
import argparse
import os.path
import numpy as np
import cv2

from khen_hardware_libs.Scripts.Scripts import *
from khen_hardware_libs.SignalClass.SignalClass import SignalClass
from khen_hardware_libs.ImageClass.ImageClass import *


import khen_hardware_libs.VideoClass.VideoClass_Loader as Loader
import khen_hardware_libs.VideoClass.VideoClass_ManipulateFrames as ManipulateFrames
import khen_hardware_libs.VideoClass.VideoClass_OpticalFlowAndTracking as OpticalFlowAndTracking
import khen_hardware_libs.VideoClass.VideoClass_TemporalSync as TemporalSync

class VideoClass:
    """
    A class used to represent a video
    ...

    Attributes
    ----------
    frames : list of frames
        list of frames, each one is a matrix (RGB)

    grayscale_frames : list of grayscale frames
        list of frames, each one is a matrix (Grayscale)

    width : int
        the video frames width

    height : int
        the video frames height

    fps : float
        the frame-per-second of the video

    Methods
    -------
    __init__(video_filename = None, fps = None, grayscale_frames = None)
        Initilize the video class.

    load_video_manually(frames, grayscale_frames, fps)
        Uses for the loading process. Gets the video values manually (frame list, grayscale frame list and the frame
        per second rate) and assign it for the video class

    load_video_from_folder_imgs(foldername, fps)
        Uses for the loading process. Used if one want to load a video as a sequence of images in a folder.

    load_video_from_file(video_filename)
        Uses for the loading process. Used for loading a video file

    load_video(video_filename, fps = None, grayscale_frames = None)
        Uses for the loading process. It is possible with three ways (See function documentation in code). You can pass
        a filename of a video (.avi, .mp4, .MOV, ...) or manually send the frames, the fps and grayscale frames or a
        folder name of image sequence.

    manipulate_video(filter_method = '', parameters_dict = None)
        This function runs over the entire video frames and apply for each a ImageClass function according to the
        filter_method name and the parameters_dict parameters.
        For example: filter_method = 'median_filter', parameters_dict = {'ksize': 5}
        It is also possible to pass a list of parameters (at the length of the video frame sequences) to change the
        parameters for each frame separately.

    manipulate_img_by_method(frame, frame_id, filter_method = 'gausian_filter', parameters_dict = None)
        Uses for the manipulation function process. Apply a single function of the ImageClass according to the
        arguments (see manipulate_video documentation)

    resize_video(width, height)
        Change the size of the entire video frames.

    digital_zoom(rect_list)
        Apply digital zoom (crop) for the entire video frames.

    rotate_video(angle)
        Rotate the entire videos at angle degree (counter-clockwise).

    follow_points_in_video(init_tracking_points = None, show = 1)
        Used to follow points in the video (due to optical flow drift) from an init location (old implementation).

    get_optical_flow_for_video(start_frame=0, end_frame=np.inf, method='LK', p0=None, show=1)
        Runs over the video frames (from start_frame to end_frame) and calculates for each Optical flow with a method
        ('LK' for Lukas Canade or 'Gunnar' for gunnar). p0 are the point locations to evaluate the optical flows, if
        empty, a Shi-Tomati edge detector is used.

    show(display_fps = None)
        Run the video. If display_fps = None the video is displayed at natural rate, if display_fps = 0 one has to
        click to move between frames, different value for display_fps can be used to set the frame-per-second rate.

    generate_event_based_frames(threshold = 1.15)
        Synthetic generation of events-frames using log and clipping definition.

    get_temporal_profile_for_pixel(pixel, neighborhood = 0, normalize = 1, white_noise_filter = 0, name = '')
        Returns a SignalClass object of a pixel intensity value versus time. neighborhood set an averaging area option,
        normalize = 1 normalize the data to be between 0 to 1 (normalize = 0 does nothing), white_noise_filter is used
        to clip the lower amplitude frequencies under the given value. name is an optional argument for the signal.

    save_video(video_filename)
        Generate a .avi file for the video

    get_time_shift_video(other, show=0)
        Using a correlation method to get a time shift between two videos

    find_idx_shift_manually(other)
        Using manual toolbox to check the time shift between two videos. keys: w or s (for left) and a or d (for right)

    def __mul__(other)
        Uses only for type(other) == int or type(other) == float. Multiply each of the frames with other value.

    def __add__(other)
        Uses only for type(other) == int or type(other) == float. Add for each of the frames anther value.

    def __sub__(other)
        Uses only for type(other) == int or type(other) == float. Subtract from each of the frames anther value.

    def __truediv__(other)
        Uses only for type(other) == int or type(other) == float. Divide from each of the frames anther value.
    """

    frames = [ ]     # For the interpreter only

    def __init__(self, video_filename = None, fps = None, grayscale_frames = None):
        if video_filename is None:
            return
        self.load_video(video_filename, fps = fps, grayscale_frames = grayscale_frames)
        return

    def __and__(self, other):   # And = temporal concatinating
        if self.fps != other.fps or self.width != other.width or self.height != other.height:
            print('Warning! different FPS or shape!')
        return VideoClass(video_filename=self.frames + other.frames, fps=self.fps, grayscale_frames=self.grayscale_frames + other.grayscale_frames)


    def __repr__(self):
        print('fps =', self.fps, 'Hz')
        print('width =', self.width, 'Pixels')
        print('height =', self.height, 'Pixels')
        print('number of frames =', len(self.frames))
        return

    # Loading video
    load_video_manually = Loader.load_video_manually
    load_video_from_folder_imgs = Loader.load_video_from_folder_imgs
    load_video_from_file = Loader.load_video_from_file
    load_video = Loader.load_video

    # Manipulation frames
    manipulate_video = ManipulateFrames.manipulate_video
    _manipulate_img_by_method = ManipulateFrames._manipulate_img_by_method
    resize_video = ManipulateFrames.resize_video
    digital_zoom = ManipulateFrames.digital_zoom
    rotate_video = ManipulateFrames.rotate_video

    # Optical Flow and Tracking tasks
    follow_points_in_video = OpticalFlowAndTracking.follow_points_in_video
    get_optical_flow_for_video = OpticalFlowAndTracking.get_optical_flow_for_video

    # Temporal Synchronization
    get_time_shift_video = TemporalSync.get_time_shift_video
    find_idx_shift_manually = TemporalSync.find_idx_shift_manually


    def show(self, display_fps = None, width = None, height = None):
        if width is None:
            width = self.width
        if height is None:
            height = self.height

        cv.namedWindow('Video', cv.WINDOW_NORMAL)
        cv.resizeWindow('Video', width, height)
        if display_fps is None:
            delay_time = int(1000/self.fps)
        elif display_fps == 0:
            delay_time = 0
        else:
            delay_time = 1000/display_fps

        for frame in self.frames:
            cv.imshow('Video', frame)
            # Press Q on keyboard to  exit
            if cv.waitKey(int(delay_time)) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
        cv.destroyAllWindows()
        return

    def generate_event_based_frames(self, threshold=1.15):
        new_frames = []
        for frame_id in range(len(self.frames)-1):
            # Simulate the DVS operation method in time:
            event_based_frame = np.divide(
                np.log(np.divide(self.grayscale_frames[frame_id + 1], self.grayscale_frames[frame_id] + 1e-25) + 1e-25),
                np.log(threshold) + 1e-25)
            event_based_frame[event_based_frame >= 1] = 1           # ON event
            event_based_frame[event_based_frame <= -1] = -1         # OFF event
            event_based_frame[np.abs(event_based_frame) < 1] = 0

            new_frames += [ event_based_frame.astype(np.float32) ]

        return SynVideoClass(video_filename=new_frames, fps=self.fps, grayscale_frames=new_frames)
        # return VideoClass(video_filename=new_frames, fps=self.fps, grayscale_frames=new_frames)

    def get_temporal_profile_for_pixel(self, pixel, neighborhood = 0, normalize = 1, white_noise_filter = 0, name = ''):
        # pixel = (x,y) , neighborhood is the distance to average pixels in a window
        x, y = pixel
        signal_vec = []
        for frame in self.grayscale_frames:
            signal_vec += [frame[y - neighborhood:y + neighborhood + 1, x - neighborhood:x + neighborhood + 1].mean()]

        signal_vec = np.array(signal_vec)
        if normalize:
            signal_vec /= 255.

        # signal_vec = temporal_dense_vec(signal_vec, int(1 / (video_class.fps * time_step)))

        recovered_signal = SignalClass(signal_vec, time_step= 1/self.fps, name=name)
        if white_noise_filter > 0:
            recovered_signal.filter_white_noise(white_noise_filter)
        return recovered_signal

    def save_video(self, video_filename):
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        # fourcc = cv.VideoWriter_fourcc(*'MPEG')
        out = cv.VideoWriter(video_filename+".avi", fourcc, self.fps, (self.width, self.height))#, 0)
        for frame in self.frames:
            # Problem - fix! only grayscale work :(
            if len(frame.shape) == 2:
                frame = np.concatenate( (frame[:,:,np.newaxis],frame[:,:,np.newaxis],frame[:,:,np.newaxis]), axis = 2)
            # if len(frame.shape) == 3:
            #     frame = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
            out.write(frame)
        out.release()
        print('Created file:',video_filename+".avi")
        return

    def __mul__(self, other):
        if type(other) != int and type(other) != float:
            return
        for frame_id in range(len(self.frames)):
            self.frames[frame_id] = self.frames[frame_id]* other
        return self

    def __add__(self, other):
        if type(other) != int and type(other) != float:
            return
        for frame_id in range(len(self.frames)):
            self.frames[frame_id] = self.frames[frame_id] + other
        return self

    def __sub__(self, other):
        if type(other) != int and type(other) != float:
            return
        for frame_id in range(len(self.frames)):
            self.frames[frame_id] = self.frames[frame_id] - other
        return self

    def __truediv__(self, other):
        if type(other) != int and type(other) != float:
            return
        for frame_id in range(len(self.frames)):
            self.frames[frame_id] = self.frames[frame_id] / other
        return self



class SynVideoClass(VideoClass):

    """
    A class used to represent a video of events (events-frames)
    ...

    Attributes
    ----------

    Methods
    -------
    get_optical_flow_for_video(self,start_frame=0, end_frame=np.inf, method='LK', p0=None, show=1)
        Same as get_optical_flow_for_video for VideoClass. A modified version for events-frames

    get_active_area_mask(frame_id, patch_size=20, threshold=20, show=0)
        Using active area definition (area with accumulated responsivity is high enough) to define active areas.
        frame_id is the frame index, patch_size is the size of the patch to consider as examined area and the threshold
        is the number of events that have to occur in the area such it be considered as an active area. This function
        returns a binary mask of active areas (1)

    """

    def manipulate_video(self, filter_method='', parameters_dict=None):
        new_frames = []
        for frame_id in range(len(self.frames)):
            frame = self.frames[frame_id]
            frame = ImageClass(frame)        # Now frame is an ImageClass object

            new_frame = self._manipulate_img_by_method(frame, frame_id, filter_method=filter_method,
                                                      parameters_dict=parameters_dict)

            new_frames += [new_frame.img]

        from VideoClass.VideoClass import VideoClass
        video_class = SynVideoClass(video_filename=new_frames, fps=self.fps)
        video_class.grayscale_frames = video_class.frames
        return video_class


    # ----------------- Active Areas -----------------#
    def get_active_area_mask(self, frame_id, patch_size=20, threshold=20, show=0):
        if len(self.events_video) == 0:
            print('Generate frames before!')
            return

        img = np.abs( self.events_video.frames[frame_id] )

        img = img.conv_with_kernel(np.ones((patch_size, patch_size)))
        img[img < threshold] = 0
        img[img >= threshold] = 1
        if show:
            img.show()
        return img

    # ----------------- Optical Flow -----------------#
    def get_optical_flow_for_video(self,start_frame=0, end_frame=np.inf, method='LK', p0=None, show=1):
        event_f2g = lambda f: (((np.clip(f,-1.0,1.0)+1.0)/2)*255).astype(np.uint8)

        start_frame = max(start_frame, 0)
        end_frame = min(end_frame, len(self.frames) - 1)
        frame_list = self.frames

        frame_list[start_frame] = event_f2g( frame_list[start_frame] )

        if p0 is None:
            p0 = frame_list[start_frame].get_good_features()

        for frame_id in range(start_frame, end_frame):
            frame_list[frame_id+1] = event_f2g( frame_list[frame_id+1] )
            frame = frame_list[frame_id]
            next_frame = frame_list[frame_id + 1]
            p0, delta_p0, p1, st = frame.get_optical_flow_from_me_to_other(next_frame, method=method, p0=p0)
            if show:
                show_optical_flow_on_img(p0, delta_p0, frame, show=1)

        return


# if __name__ == '__main__':
#     print('test')